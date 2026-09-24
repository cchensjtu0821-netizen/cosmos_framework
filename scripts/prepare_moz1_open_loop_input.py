# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Prepare one offline MOZ1 policy sample from three camera images and a state JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _read_rgb(path: Path) -> torch.Tensor:
    with path.open("rb") as stream:
        image = Image.open(stream).convert("RGB")
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0


def _compose_views(high: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Match the MOZ1 dataset's high/top and two half-size wrists/bottom layout."""
    height, width = high.shape[-2:]
    if height < 2 or width < 2 or height % 2 or width % 2:
        raise ValueError(f"High camera must have even dimensions at least 2x2, got {height}x{width}")
    half = (height // 2, width // 2)
    left = F.interpolate(left[None], size=half, mode="bilinear", align_corners=False)[0]
    right = F.interpolate(right[None], size=half, mode="bilinear", align_corners=False)[0]
    return torch.cat([high, torch.cat([left, right], dim=-1)], dim=-2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--high", type=Path, required=True, help="High third-person keyframe")
    parser.add_argument("--left-wrist", type=Path, required=True)
    parser.add_argument("--right-wrist", type=Path, required=True)
    parser.add_argument(
        "--state", type=Path, required=True, help="JSON list of 20 current absolute-state numbers"
    )
    parser.add_argument(
        "--stats", type=Path, required=True, help="norm_stats.json used by the checkpoint's training job"
    )
    parser.add_argument("--prompt", required=True, help="Task description")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="New directory for the composite PNG and inference JSON"
    )
    parser.add_argument("--name", default="moz1_open_loop")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    state = np.asarray(json.loads(args.state.read_text()), dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        parser.error("--state must be a JSON list of 20 finite values in the training state-field order")
    stats = json.loads(args.stats.read_text()).get("norm_stats", {})
    if not all(key in stats for key in ("state", "actions")):
        parser.error("--stats must contain norm_stats.state and norm_stats.actions")
    if not args.prompt.strip():
        parser.error("--prompt must be nonempty")

    composite = _compose_views(_read_rgb(args.high), _read_rgb(args.left_wrist), _read_rgb(args.right_wrist))
    pixels = (composite * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    image_path = args.out_dir / "conditioning.png"
    Image.fromarray(pixels, mode="RGB").save(image_path)
    sample = {
        "name": args.name,
        "model_mode": "policy",
        "domain_name": "moz1",
        "vision_path": str(image_path.resolve()),
        "state_path": str(args.state.resolve()),
        "stats_path": str(args.stats.resolve()),
        "view_point": "concat_view",
        "prompt": args.prompt,
        "fps": 30,
        "action_chunk_size": 16,
        "image_size": 256,
        "num_steps": 30,
        "guidance": 1.0,
        "shift": 3.0,
        "seed": args.seed,
    }
    sample_path = args.out_dir / "sample.json"
    sample_path.write_text(json.dumps(sample, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {image_path} and {sample_path}")


if __name__ == "__main__":
    main()
