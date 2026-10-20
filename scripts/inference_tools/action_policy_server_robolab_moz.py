# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Serve post-trained MOZ1 Edge DCPs over the OpenPI WebSocket transport."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path

import numpy as np

from cosmos_framework.scripts.inference_tools.moz1_policy_contract import (
    CAMERAS,
    FIELD_ORDER,
    PROTOCOL,
    STORED_SPACE,
    compose_image,
    load_stats,
    resolve_checkpoint,
    validate_request,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-path",
        "--checkpoint_path",
        type=Path,
        default=os.environ.get("MOZ1_SERVER_CKPT"),
        help="Run root, checkpoints directory, or explicit DCP iteration; latest complete save is selected.",
    )
    parser.add_argument("--stats-path", type=Path, default=os.environ.get("MOZ1_STATS_PATH"))
    parser.add_argument("--experiment", default="action_policy_moz1_edge")
    parser.add_argument("--experiment-override", action="append", default=[])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic-seed", action="store_true")
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=1.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--use-torch-compile", action="store_true")
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument(
        "--check-only", action="store_true", help="Resolve DCP and validate stats without CUDA/model loading."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/cosmos3_action_server/moz1"))
    args = parser.parse_args(argv)
    if args.checkpoint_path is None:
        parser.error("--checkpoint-path or MOZ1_SERVER_CKPT is required")
    if args.stats_path is None:
        root = os.environ.get("MOZ1_ROOT")
        if root:
            args.stats_path = Path(root) / "norm_stats.json"
        else:
            parser.error("--stats-path, MOZ1_STATS_PATH or MOZ1_ROOT is required")
    if not 1 <= args.port <= 65535 or args.num_steps < 1 or args.seed < 0:
        parser.error("port must be in 1..65535; num-steps must be positive; seed must be nonnegative")
    if not np.isfinite([args.guidance, args.shift]).all() or args.shift <= 0:
        parser.error("guidance and shift must be finite; shift must be positive")
    return args


def prepare(args):
    args.checkpoint_path = resolve_checkpoint(args.checkpoint_path)
    stats, fingerprint = load_stats(args.stats_path.expanduser())
    metadata = {
        "protocol": PROTOCOL,
        "domain_name": "moz1",
        "action_dim": 20,
        "action_chunk_size": 16,
        "conditioning_fps": 30,
        "resolution": "256",
        "state_space": STORED_SPACE,
        "action_space": STORED_SPACE,
        "field_order": list(FIELD_ORDER),
        "cameras": list(CAMERAS),
        "stats_sha256": fingerprint,
        "checkpoint": str(args.checkpoint_path),
        "use_ema_weights": False,
        "generation": {
            "seed": args.seed,
            "deterministic_seed": args.deterministic_seed,
            "num_steps": args.num_steps,
            "guidance": args.guidance,
            "shift": args.shift,
        },
    }
    return stats, metadata


def build_normalizer(stats):
    import torch

    from cosmos_framework.data.generator.action.action_processing import (
        StateActionNormalizer,
        resolve_action_normalization,
    )

    return StateActionNormalizer(
        state_normalizer=resolve_action_normalization(
            "quantile", {k: torch.tensor(stats["state"][k], dtype=torch.float32) for k in ("q01", "q99")}
        ),
        action_normalizer=resolve_action_normalization(
            "quantile", {k: torch.tensor(stats["actions"][k], dtype=torch.float32) for k in ("q01", "q99")}
        ),
        expected_dim=20,
    )


class Moz1PolicyService:
    def __init__(self, args, stats, metadata):
        import torch

        from cosmos_framework.inference.args import OmniSetupOverrides
        from cosmos_framework.inference.common.init import init_output_dir
        from cosmos_framework.inference.inference import OmniInference
        from cosmos_framework.scripts.action_policy_server_utils import maybe_init_distributed

        if not torch.cuda.is_available():
            raise RuntimeError("MOZ1 policy inference requires CUDA")
        maybe_init_distributed()
        # This is the same module-config, non-EMA path as the MOZ1 open-loop launcher.
        setup = OmniSetupOverrides(
            checkpoint_path=str(args.checkpoint_path),
            experiment=args.experiment,
            experiment_overrides=args.experiment_override,
            output_dir=args.output_dir,
            parallelism_preset="throughput",
            use_ema_weights=False,
            use_torch_compile=args.use_torch_compile,
        ).build_setup()
        setup.guardrails = False
        setup.offload_guardrail_models = False
        init_output_dir(setup.output_dir)
        self.pipe = OmniInference.create(setup)
        self.model = self.pipe.model
        self.model.eval()
        self.normalizer = build_normalizer(stats)
        self.args = args
        self.metadata = metadata
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(args.seed)

    def _build_batch(self, obs):
        import torch

        from cosmos_framework.inference.action import build_action_batch
        from cosmos_framework.inference.args import ModelMode

        state, prompt = validate_request(obs)
        initial_state = self.normalizer.state_normalizer.normalize_action(torch.from_numpy(state.copy()))
        # Preserve the existing open-loop padding, prompt formatter, state row,
        # and inverse-normalization record. Do not normalize the state twice.
        return build_action_batch(
            video=compose_image(obs),
            action=torch.zeros(16, self.model.config.max_action_dim),
            raw_action_dim=20,
            prompt=prompt,
            view_point="concat_view",
            domain_name="moz1",
            model_mode=ModelMode.POLICY,
            action_chunk_size=16,
            fps=30,
            resolution="256",
            input_video_key=self.model.input_video_key,
            initial_state=initial_state,
            action_normalizer=self.normalizer,
            batch_size=1,
            device="cuda",
        )

    def infer(self, obs):
        import torch

        with self._lock, torch.inference_mode():
            batch = self._build_batch(obs)
            seed = self.args.seed if self.args.deterministic_seed else int(self._rng.integers(0, 2**31))
            samples = self.model.generate_samples_from_batch(
                batch,
                guidance=self.args.guidance,
                seed=[seed],
                num_steps=self.args.num_steps,
                shift=self.args.shift,
            )
            # generate_samples_from_batch already externalizes with the record.
            # Trim only the absolute-state row; never flip/binarize the last
            # channel (it is torso yaw, not the DROID gripper).
            action = samples["action"][0].detach().float().cpu().numpy()
            if action.shape != (17, 20) or not np.isfinite(action).all():
                raise ValueError(f"Expected finite generated [17,20] state+actions; got {action.shape}")
            result = {
                "action": action[1:].copy(),
                "protocol": PROTOCOL,
                "action_space": STORED_SPACE,
                "seed": seed,
                "stats_sha256": self.metadata["stats_sha256"],
            }
            if self.args.decode_video:
                video = self.model.decode(samples["vision"][0])
                result["video"] = (
                    ((video[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
                )
            return result


def main(argv=None):
    args = parse_args(argv)
    stats, metadata = prepare(args)
    print(json.dumps(metadata, indent=2), flush=True)
    if args.check_only:
        return
    from cosmos_framework.inference.common.init import init_script

    init_script()
    from cosmos_framework.scripts.action_policy_server_robolab import (
        _load_openpi_websocket_policy_server,
    )

    service = Moz1PolicyService(args, stats, metadata)
    server = _load_openpi_websocket_policy_server()
    print(f"MOZ1 ready: ws://{args.host}:{args.port} (20-D, 16 steps, 30 FPS)", flush=True)
    server(policy=service, host=args.host, port=args.port, metadata=metadata).serve_forever()


if __name__ == "__main__":
    main()
