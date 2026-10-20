# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""CPU-only validation for the MOZ1 online policy contract."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

PROTOCOL = "moz1_policy_v1"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
FIELD_ORDER = ("left_pose_6", "left_gripper_1", "right_pose_6", "right_gripper_1", "torso_pose_6")
STORED_SPACE = "dataset_stored_20d"


def complete_dcp(path: Path) -> bool:
    model = path / "model" if (path / "model").is_dir() else path
    shards = list(model.glob("*.distcp"))
    metadata = model / ".metadata"
    return (
        metadata.is_file()
        and metadata.stat().st_size > 0
        and bool(shards)
        and all(shard.stat().st_size > 0 for shard in shards)
    )


def resolve_checkpoint(path: Path) -> Path:
    """Accept a run/checkpoints/iteration/model path; skip incomplete saves.

    Prefer the trainer's published latest_checkpoint.txt. If it is missing or
    points to an incomplete save, scan numbered complete iterations. Never use
    mtime, which can change when a run is copied.
    """
    path = path.expanduser().resolve()
    if complete_dcp(path):
        return path.parent if path.name == "model" else path
    if re.fullmatch(r"iter_\d{9}", path.name) or path.name == "model":
        raise ValueError(f"Incomplete DCP checkpoint: {path}")
    checkpoints = path / "checkpoints" if (path / "checkpoints").is_dir() else path
    pointer = checkpoints / "latest_checkpoint.txt"
    if pointer.is_file():
        name = pointer.read_text().strip()
        if not re.fullmatch(r"iter_\d{9}", name):
            raise ValueError(f"Invalid checkpoint name in {pointer}")
        if complete_dcp(checkpoints / name):
            return checkpoints / name
    candidates = sorted(
        (p for p in checkpoints.glob("iter_*") if re.fullmatch(r"iter_\d{9}", p.name) and complete_dcp(p)),
        reverse=True,
    )
    if not candidates:
        raise ValueError(f"No complete DCP found under {path}; set --checkpoint-path or MOZ1_SERVER_CKPT")
    return candidates[0]


def load_stats(path: Path) -> tuple[dict, str]:
    data = path.read_bytes()
    stats = json.loads(data)
    stats = stats.get("norm_stats", stats)
    for block in ("state", "actions"):
        for key in ("q01", "q99"):
            values = np.asarray(stats[block][key], dtype=np.float32)
            if values.shape != (20,) or not np.isfinite(values).all():
                raise ValueError(f"norm_stats.{block}.{key} must contain 20 finite numbers")
        if np.any(np.asarray(stats[block]["q99"]) < np.asarray(stats[block]["q01"])):
            raise ValueError(f"norm_stats.{block}: q99 must be >= q01")
    return stats, hashlib.sha256(data).hexdigest()


def rgb_image(value, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) < 2 or image.dtype != np.uint8:
        raise ValueError(f"{name} must be RGB uint8 [H,W,3], at least 2x2; got {image.shape}/{image.dtype}")
    return np.ascontiguousarray(image)


def validate_request(obs: dict) -> tuple[np.ndarray, str]:
    if obs.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected protocol={PROTOCOL}; the DROID client is not compatible")
    if obs.get("state_space") != STORED_SPACE:
        raise ValueError(f"Expected state_space={STORED_SPACE}; supply the training dataset state representation")
    state = np.asarray(obs.get("observation/state"), dtype=np.float32)
    if state.shape != (20,) or not np.isfinite(state).all():
        raise ValueError("observation/state must be one finite 20-D current state (not joint angles)")
    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be nonempty text")
    if "observation/image" in obs and "observation/images" in obs:
        raise ValueError("Supply either a MOZ1 composite or three cameras, not both")
    if "observation/image" in obs:
        rgb_image(obs["observation/image"], "observation/image")
    else:
        images = obs.get("observation/images", {})
        for camera in CAMERAS:
            if camera not in images:
                raise ValueError(f"Missing observation/images/{camera}")
            rgb_image(images[camera], camera)
        h, w = np.asarray(images["cam_high"]).shape[:2]
        if h % 2 or w % 2:
            raise ValueError("cam_high dimensions must be even for the training three-view layout")
    return state, prompt.strip()


def compose_image(obs: dict):
    """Return uint8 [C,1,H,W], matching prepare_moz1_open_loop_input."""
    import torch
    import torch.nn.functional as F

    if "observation/image" in obs:
        image = torch.from_numpy(rgb_image(obs["observation/image"], "observation/image").copy())
        return image.permute(2, 0, 1).unsqueeze(1)
    images = obs["observation/images"]
    high, left, right = [
        torch.from_numpy(rgb_image(images[name], name).copy()).permute(2, 0, 1).float() / 255.0 for name in CAMERAS
    ]
    h, w = high.shape[-2:]
    left = F.interpolate(left[None], size=(h // 2, w // 2), mode="bilinear", align_corners=False)[0]
    right = F.interpolate(right[None], size=(h // 2, w // 2), mode="bilinear", align_corners=False)[0]
    composite = torch.cat([high, torch.cat([left, right], dim=-1)], dim=-2)
    return (composite * 255.0).clamp(0, 255).to(torch.uint8).unsqueeze(1)
