# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Load one MOZ1 sample through the policy preprocessing path without a GPU."""

from __future__ import annotations

import argparse

import torch

from cosmos_framework.data.generator.action.datasets.action_sft_dataset import get_action_moz1_sft_dataset


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="MOZ1 LeRobot-v3 dataset root")
    parser.add_argument("--stats-path", default=None, help="Defaults to <root>/norm_stats.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--chunk-length", type=int, default=16)
    parser.add_argument("--resolution", default="256")
    parser.add_argument("--no-normalization", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = get_action_moz1_sft_dataset(
        root=args.root,
        stats_path=args.stats_path,
        fps=args.fps,
        chunk_length=args.chunk_length,
        action_normalization=None if args.no_normalization else "quantile",
        resolution=args.resolution,
        tokenizer_config=None,
        cfg_dropout_rate=0.0,
        append_viewpoint_info=False,
        append_duration_fps_timestamps=False,
        append_resolution_info=False,
        append_idle_frames=False,
        format_prompt_as_json=False,
        iterable_shuffle=False,
    )
    raw_dataset = dataset._dataset  # noqa: SLF001 - diagnostic script intentionally reports resolved keys
    sample = dataset[args.index]

    print(f"dataset_size={len(dataset)} index={args.index}")
    print(f"image_features={raw_dataset._image_features}")  # noqa: SLF001
    print(f"state_features={raw_dataset._state_features}")  # noqa: SLF001
    print(f"action_features={raw_dataset._action_features}")  # noqa: SLF001
    for key in ("video", "action_raw", "action"):
        value = sample[key]
        finite = bool(torch.isfinite(value).all()) if torch.is_floating_point(value) else True
        print(f"{key}: shape={tuple(value.shape)} dtype={value.dtype} finite={finite}")
    print(f"raw_action_dim={int(sample['raw_action_dim'])}")
    print(f"domain_id={int(sample['domain_id'])}")
    print(f"sequence_plan={sample['sequence_plan'].as_dict()}")


if __name__ == "__main__":
    main()
