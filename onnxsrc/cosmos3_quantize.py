#!/usr/bin/env python3
"""Generate a DOPT config or calibrate the fixed-layout Cosmos3 Policy net."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

from cosmos_framework.data.generator.sequence_packing import SequencePlan
from cosmos_framework.model.generator.utils.data_and_condition import GenerationDataClean
from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
from cosmos_framework.scripts.export_action_policy_onnx import (
    ExportArgs,
    PolicyDenoiserOnnxWrapper,
    _example_inputs,
    _install_onnx_attention,
    _separate_timestep_embedders_for_onnx,
)


def _import_dopt(dopt_sim_path: Path) -> tuple[Any, ...]:
    if not dopt_sim_path.is_dir():
        raise FileNotFoundError(f"DOPT simulator directory does not exist: {dopt_sim_path}")
    sys.path.insert(0, str(dopt_sim_path))
    from dopt_lm.do_opt import (  # type: ignore[import-not-found]
        generate_config_file,
        generate_quant_params,
        optimize_model,
        set_calibrate_state,
        set_quant_state,
    )

    return (
        generate_config_file,
        generate_quant_params,
        optimize_model,
        set_calibrate_state,
        set_quant_state,
    )


def _manifest_input_shapes(manifest_path: Path) -> dict[str, tuple[int, ...]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"Layout manifest has no object-valued inputs: {manifest_path}")
    required = {
        "prompt_token_ids",
        "video_latent",
        "action_latent",
        "vision_timestep",
        "action_timestep",
        "action_domain_id",
    }
    missing = required - inputs.keys()
    if missing:
        raise ValueError(f"Layout manifest is missing inputs: {sorted(missing)}")
    return {
        name: tuple(int(dimension) for dimension in inputs[name]["shape"])
        for name in required
    }


def _parse_condition_indexes(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def _load_fixed_policy(args: argparse.Namespace) -> tuple[RobolabPolicyService, ExportArgs, Any, tuple[torch.Tensor, ...]]:
    export_args = ExportArgs(
        checkpoint_path=args.checkpoint_path,
        prompt=args.prompt,
        image_height=args.image_height,
        image_width=args.image_width,
        action_chunk_size=args.action_chunk_size,
        action_dim=args.action_dim,
        conditioning_fps=args.conditioning_fps,
        resolution=args.resolution,
        domain_name=args.domain_name,
        export_dtype="float32",
        simplify_onnx=False,
    )
    server_args = RobolabServerArgs(
        checkpoint_path=export_args.checkpoint_path,
        image_height=export_args.image_height,
        image_width=export_args.image_width,
        action_chunk_size=export_args.action_chunk_size,
        action_dim=export_args.action_dim,
        conditioning_fps=export_args.conditioning_fps,
        resolution=export_args.resolution,
        domain_name=export_args.domain_name,
        guidance=1.0,
        num_steps=1,
        use_torch_compile=False,
    )
    service = RobolabPolicyService(server_args)
    _separate_timestep_embedders_for_onnx(service.model.net)
    shapes = _manifest_input_shapes(args.layout_manifest)
    device = service.model.tensor_kwargs["device"]
    video_latent = torch.zeros(shapes["video_latent"], dtype=torch.float32, device=device)
    action_latent = torch.zeros(shapes["action_latent"], dtype=torch.float32, device=device)
    domain_id = torch.full(shapes["action_domain_id"], 8, dtype=torch.long, device=device)

    special_token_overhead = 2 + int("bos_token_id" in service.model.llm_special_tokens)
    raw_text_length = shapes["prompt_token_ids"][0] - special_token_overhead
    if raw_text_length < 1:
        raise ValueError(
            f"Manifest prompt length {shapes['prompt_token_ids']} is too short for "
            f"{special_token_overhead} required special tokens"
        )
    synthetic_text_ids = [1] * raw_text_length
    sequence_plans = [
        SequencePlan(
            has_text=True,
            has_vision=True,
            condition_frame_indexes_vision=_parse_condition_indexes(args.condition_vision_frames),
            has_action=True,
            condition_frame_indexes_action=_parse_condition_indexes(args.condition_action_frames),
        )
    ]
    clean_data = GenerationDataClean(
        batch_size=1,
        is_image_batch=False,
        x0_tokens_vision=[video_latent],
        fps_vision=torch.tensor([args.conditioning_fps]),
        x0_tokens_action=[action_latent],
        fps_action=torch.tensor([args.conditioning_fps]),
        action_domain_id=[domain_id],
        raw_action_dim=[torch.tensor(args.action_dim, dtype=torch.long, device=device)],
    )
    packed = service.model._pack_input_sequence(
        sequence_plans,
        [synthetic_text_ids],
        clean_data,
        torch.tensor([1.0], dtype=torch.float32),
        include_end_of_generation_token=service.model._derive_include_end_of_generation_token(),
    )
    packed.to_cuda()
    _install_onnx_attention(service.model.net)
    inputs = _example_inputs(packed)
    generated_shapes = {
        name: tuple(value.shape)
        for name, value in zip(
            (
                "prompt_token_ids",
                "video_latent",
                "action_latent",
                "vision_timestep",
                "action_timestep",
                "action_domain_id",
            ),
            inputs,
            strict=True,
        )
    }
    mismatches = {
        name: {"expected": shapes[name], "generated": generated_shapes[name]}
        for name in shapes
        if shapes[name] != generated_shapes[name]
    }
    if mismatches:
        raise ValueError(
            "Synthetic packed layout does not match the exported ONNX manifest. "
            "Adjust condition frame indexes or use the matching manifest: "
            f"{mismatches}"
        )
    return service, export_args, packed, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("gen-config", "quant"), required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dopt-sim-path", type=Path, required=True)
    parser.add_argument("--quant-config", type=Path, required=True)
    parser.add_argument("--fakequant-weight", type=Path, required=True)
    parser.add_argument("--quant-output-dir", type=Path, required=True)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--condition-vision-frames", default="0")
    parser.add_argument("--condition-action-frames", default="0")
    parser.add_argument("--embedding-separate", type=int, choices=(0, 1), default=1)
    parser.add_argument("--prompt", default="Pick up the object and place it in the target location.")
    parser.add_argument("--image-height", type=int, default=540)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--action-chunk-size", type=int, default=16)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--conditioning-fps", type=float, default=5.0)
    parser.add_argument("--resolution", default="480")
    parser.add_argument("--domain-name", default="droid_lerobot")
    args = parser.parse_args()
    if not args.layout_manifest.is_file():
        parser.error(f"layout manifest does not exist: {args.layout_manifest}")

    (
        generate_config_file,
        generate_quant_params,
        optimize_model,
        set_calibrate_state,
        set_quant_state,
    ) = _import_dopt(args.dopt_sim_path)
    service, _export_args, packed, inputs = _load_fixed_policy(args)
    net = service.model.net.float().eval()

    if args.stage == "gen-config":
        args.quant_config.parent.mkdir(parents=True, exist_ok=True)
        generate_config_file(net, str(args.quant_config))
        print(f"Generated DOPT config: {args.quant_config}")
        return

    if not args.quant_config.is_file():
        raise FileNotFoundError(f"Run gen-config and modify the config first: {args.quant_config}")
    quant_net = optimize_model(net, str(args.quant_config)).eval()
    set_quant_state(quant_net, weight_state=True, input_state=True)
    set_calibrate_state(quant_net, True)
    wrapper = PolicyDenoiserOnnxWrapper(quant_net, packed).float().eval()
    float_inputs = tuple(value.float() if value.is_floating_point() else value for value in inputs)
    with torch.inference_mode():
        outputs = wrapper(*float_inputs)
    print("Calibration output shapes:", [tuple(value.shape) for value in outputs])
    set_calibrate_state(quant_net, False)

    args.fakequant_weight.parent.mkdir(parents=True, exist_ok=True)
    torch.save(quant_net.state_dict(), args.fakequant_weight)
    generate_quant_params(
        quant_net,
        str(args.quant_output_dir),
        quant_param_2=False,
        embedding_separate=bool(args.embedding_separate),
    )
    print(f"Saved fake-quant state dict: {args.fakequant_weight}")
    print(f"Generated quant params with base path: {args.quant_output_dir}")


if __name__ == "__main__":
    main()
