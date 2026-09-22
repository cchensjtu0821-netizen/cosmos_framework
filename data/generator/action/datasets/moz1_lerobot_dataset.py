# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""MOZ1 LeRobot-v3 adapter for provisional relative-action policy training.

The adapter intentionally does *not* derive actions from state.  It treats the
three configured 6-D Cartesian action fields as already-relative
``[dx, dy, dz, droll, dpitch, dyaw]`` values, while the first action row is the
20-D absolute robot state.  This lets the training stack be exercised before
the producer-side relative-pose convention is recovered.

Layout::

    [left pose(6), left gripper(1),
     right pose(6), right gripper(1),
     torso pose(6)]

The assumption is isolated in :meth:`_build_action_chunk`: replace that method
when the authoritative conversion becomes available.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from cosmos_framework.data.generator.action.action_processing import (
    ActionNormalizer,
    resolve_action_normalization,
)
from cosmos_framework.data.generator.action.datasets.cosmos3_action_lerobot import (
    BaseActionLeRobotDataset,
    Gripper,
    Pos,
    Rot,
    build_action_spec,
)

_ACTION_DIM = 20
_POSE_DIM = 6


@dataclass(frozen=True)
class StateActionNormalizer:
    """Apply separate normalizers to the absolute state row and action rows."""

    state_normalizer: ActionNormalizer
    action_normalizer: ActionNormalizer
    state_rows: int = 1

    def _apply(self, values: torch.Tensor, method: str) -> torch.Tensor:
        if values.ndim < 2 or values.shape[-1] != _ACTION_DIM:
            raise ValueError(f"Expected [..., T, {_ACTION_DIM}] state/action tensor, got {tuple(values.shape)}")
        if values.shape[-2] < self.state_rows:
            raise ValueError(
                f"Expected at least {self.state_rows} state rows, got shape {tuple(values.shape)}"
            )
        state_fn = getattr(self.state_normalizer, method)
        action_fn = getattr(self.action_normalizer, method)
        state = state_fn(values[..., : self.state_rows, :])
        action = action_fn(values[..., self.state_rows :, :])
        return torch.cat([state, action], dim=-2)

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._apply(action, "normalize_action")

    def denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        return self._apply(action, "denormalize_action")


def _unwrap_norm_stats(raw: dict[str, Any], block: str, path: Path) -> dict[str, torch.Tensor]:
    """Read ``norm_stats.{state,actions}`` and validate the fixed 20-D contract."""

    container = raw.get("norm_stats", raw)
    if not isinstance(container, dict) or block not in container:
        raise KeyError(f"Missing normalization block norm_stats.{block} in {path}")
    values = container[block]
    if not isinstance(values, dict):
        raise TypeError(f"Normalization block norm_stats.{block} in {path} must be an object")

    result: dict[str, torch.Tensor] = {}
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        if key not in values:
            continue
        tensor = torch.as_tensor(values[key], dtype=torch.float32)
        if tensor.shape != (_ACTION_DIM,):
            raise ValueError(
                f"norm_stats.{block}.{key} in {path} must have {_ACTION_DIM} values, "
                f"got shape {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"norm_stats.{block}.{key} in {path} contains non-finite values")
        result[key] = tensor
    return result


def _resolve_feature(
    features: dict[str, Any],
    requested: str,
    *,
    role: str,
    expected_dtype: str | None = None,
    aliases: tuple[str, ...] = (),
) -> str:
    """Resolve either an exact LeRobot key or an unambiguous leaf-name key."""

    def matches_for(candidate: str) -> list[str]:
        matches = [candidate] if candidate in features else [
            key for key in features if key.rsplit(".", 1)[-1] == candidate
        ]
        if expected_dtype is not None:
            matches = [key for key in matches if features[key].get("dtype") == expected_dtype]
        return matches

    matches = matches_for(requested)
    if not matches:
        matches = sorted({key for alias in aliases for key in matches_for(alias)})
    if len(matches) != 1:
        available = sorted(
            key
            for key, spec in features.items()
            if expected_dtype is None or spec.get("dtype") == expected_dtype
        )
        raise KeyError(
            f"Could not resolve {role} feature {requested!r}; matches={matches}. "
            f"Available candidates: {available}"
        )
    return matches[0]


def _as_matrix(value: Any, *, length: int, width: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if width == 1 and tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.shape != (length, width):
        raise ValueError(f"Feature {name!r} must have shape {(length, width)}, got {tuple(tensor.shape)}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"Feature {name!r} contains non-finite values")
    return tensor


class MOZ1LeRobotDataset(BaseActionLeRobotDataset):
    """MOZ1 three-camera, 20-D state/action policy dataset.

    This is a provisional adapter.  The configured action pose fields are used
    verbatim as relative targets; no subtraction or SE(3) conversion occurs.
    Rotation triples are temporarily named as Euler XYZ channels, but idle-frame
    detection is disabled because the relative-pose convention is unknown.
    """

    def __init__(
        self,
        root: str,
        fps: float = 30.0,
        chunk_length: int = 16,
        mode: str = "policy",
        split: str = "train",
        split_seed: int = 42,
        split_val_ratio: float = 0.01,
        tolerance_s: float = 1e-4,
        action_normalization: str | None = "quantile",
        stats_path: str | None = None,
        sample_stride: int = 1,
        high_camera_feature: str = "cam_high",
        left_wrist_camera_feature: str = "cam_left_wrist",
        right_wrist_camera_feature: str = "cam_right_wrist",
        left_state_feature: str = "leftarm_state_cart_pos",
        left_gripper_state_feature: str = "leftarm_gripper_state_pos",
        right_state_feature: str = "rightarm_state_cart_pos",
        right_gripper_state_feature: str = "rightarm_gripper_state_pos",
        torso_state_feature: str = "torso_state_cart_pos",
        left_action_feature: str = "leftarm_cmd_cart_pos",
        left_gripper_action_feature: str = "leftarm_gripper_cmd_pos",
        right_action_feature: str = "rightarm_cmd_cart_pos",
        right_gripper_action_feature: str = "rightarm_gripper_cmd_pos",
        torso_action_feature: str = "torso_cmd_cart_pos",
    ) -> None:
        root_path = Path(root)
        info_path = root_path / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        features = info.get("features")
        if not isinstance(features, dict):
            raise TypeError(f"Expected a feature map in {info_path}")
        native_fps = float(info.get("fps", fps))
        if abs(native_fps - float(fps)) > 1e-6:
            raise ValueError(
                f"The provisional MOZ1 adapter requires native fps={native_fps}; got fps={fps}. "
                "Temporal resampling needs a horizon-aware episode index and is intentionally disabled for this smoke."
            )

        self._image_features = {
            "high": _resolve_feature(features, high_camera_feature, role="high camera", expected_dtype="video"),
            "left_wrist": _resolve_feature(
                features, left_wrist_camera_feature, role="left wrist camera", expected_dtype="video"
            ),
            "right_wrist": _resolve_feature(
                features, right_wrist_camera_feature, role="right wrist camera", expected_dtype="video"
            ),
        }
        self._state_features = (
            _resolve_feature(features, left_state_feature, role="left state"),
            _resolve_feature(
                features,
                left_gripper_state_feature,
                role="left gripper state",
                aliases=("left_gripper_state", "left_gripper_state_pos"),
            ),
            _resolve_feature(features, right_state_feature, role="right state"),
            _resolve_feature(
                features,
                right_gripper_state_feature,
                role="right gripper state",
                aliases=("right_gripper_state", "right_gripper_state_pos"),
            ),
            _resolve_feature(features, torso_state_feature, role="torso state"),
        )
        self._action_features = (
            _resolve_feature(
                features,
                left_action_feature,
                role="left action",
                aliases=("leftarm_action_cart_pos", "leftarm_command_cart_pos", "left_arm_action_cart_pos"),
            ),
            _resolve_feature(
                features,
                left_gripper_action_feature,
                role="left gripper action",
                aliases=("left_gripper_action", "left_gripper_cmd", "left_gripper_cmd_pos", "left_gripper_action_pos"),
            ),
            _resolve_feature(
                features,
                right_action_feature,
                role="right action",
                aliases=("rightarm_action_cart_pos", "rightarm_command_cart_pos", "right_arm_action_cart_pos"),
            ),
            _resolve_feature(
                features,
                right_gripper_action_feature,
                role="right gripper action",
                aliases=(
                    "right_gripper_action",
                    "right_gripper_cmd",
                    "right_gripper_cmd_pos",
                    "right_gripper_action_pos",
                ),
            ),
            _resolve_feature(
                features,
                torso_action_feature,
                role="torso action",
                aliases=("torso_action_cart_pos", "torso_command_cart_pos"),
            ),
        )
        if stats_path:
            self._stats_file = Path(stats_path)
            if not self._stats_file.is_absolute():
                self._stats_file = root_path / self._stats_file
        else:
            self._stats_file = root_path / "norm_stats.json"
        self._stats_raw = json.loads(self._stats_file.read_text()) if action_normalization is not None else None

        super().__init__(
            fps=fps,
            chunk_length=chunk_length,
            split_seed=split_seed,
            split_val_ratio=split_val_ratio,
            split=split,
            mode=mode,
            embodiment_type="moz1",
            viewpoint="concat_view",
            pose_convention=None,
            rotation_format="euler_xyz",
            action_normalization=action_normalization,
            tolerance_s=tolerance_s,
            sample_stride=sample_stride,
        )

        if action_normalization is not None:
            assert self._stats_raw is not None
            state_stats = _unwrap_norm_stats(self._stats_raw, "state", self._stats_file)
            state_normalizer = resolve_action_normalization(action_normalization, state_stats)
            assert self._action_normalizer is not None
            self._action_normalizer = StateActionNormalizer(
                state_normalizer=state_normalizer,
                action_normalizer=self._action_normalizer,
            )

        observation_ts = [i * self._dt for i in range(self._chunk_length + 1)]
        action_ts = [i * self._dt for i in range(self._chunk_length)]
        self._delta_timestamps = {
            **{key: observation_ts for key in self._image_features.values()},
            **{key: observation_ts for key in self._state_features},
            **{key: action_ts for key in self._action_features},
        }
        self._all_shard_roots = [str(root_path)]
        self._register_sources()

    def _load_norm_stats(self, action_normalization: str) -> dict[str, torch.Tensor]:
        if self._stats_raw is None:
            raise RuntimeError("MOZ1 normalization stats were not loaded")
        return _unwrap_norm_stats(self._stats_raw, "actions", self._stats_file)

    def _build_action_spec(self):
        return build_action_spec(
            Pos(prefix="left"),
            Rot("euler_xyz", prefix="left"),
            Gripper(prefix="left"),
            Pos(prefix="right"),
            Rot("euler_xyz", prefix="right"),
            Gripper(prefix="right"),
            Pos(prefix="torso"),
            Rot("euler_xyz", prefix="torso"),
        )

    def _compute_idle_frames(self, raw_action: torch.Tensor) -> None:
        """Skip semantic idle detection until the relative-pose convention is known."""

        return None

    def _build_action_chunk(self, sample: dict[str, Any]) -> torch.Tensor:
        """Concatenate stored action fields verbatim under the relative-action assumption."""

        left, left_gripper, right, right_gripper, torso = self._action_features
        return torch.cat(
            [
                _as_matrix(sample[left], length=self._chunk_length, width=_POSE_DIM, name=left),
                _as_matrix(sample[left_gripper], length=self._chunk_length, width=1, name=left_gripper),
                _as_matrix(sample[right], length=self._chunk_length, width=_POSE_DIM, name=right),
                _as_matrix(sample[right_gripper], length=self._chunk_length, width=1, name=right_gripper),
                _as_matrix(sample[torso], length=self._chunk_length, width=_POSE_DIM, name=torso),
            ],
            dim=-1,
        )

    def _build_initial_state(self, sample: dict[str, Any]) -> torch.Tensor:
        left, left_gripper, right, right_gripper, torso = self._state_features
        full = torch.cat(
            [
                _as_matrix(sample[left], length=self._chunk_length + 1, width=_POSE_DIM, name=left),
                _as_matrix(sample[left_gripper], length=self._chunk_length + 1, width=1, name=left_gripper),
                _as_matrix(sample[right], length=self._chunk_length + 1, width=_POSE_DIM, name=right),
                _as_matrix(sample[right_gripper], length=self._chunk_length + 1, width=1, name=right_gripper),
                _as_matrix(sample[torso], length=self._chunk_length + 1, width=_POSE_DIM, name=torso),
            ],
            dim=-1,
        )
        return full[0]

    def _compose_video(self, sample: dict[str, Any]) -> torch.Tensor:
        high = sample[self._image_features["high"]]
        left = sample[self._image_features["left_wrist"]]
        right = sample[self._image_features["right_wrist"]]
        if not all(isinstance(video, torch.Tensor) and video.ndim == 4 for video in (high, left, right)):
            raise ValueError("MOZ1 camera features must be [T,C,H,W] tensors")
        if not (high.shape[0] == left.shape[0] == right.shape[0] == self._chunk_length + 1):
            raise ValueError(
                "MOZ1 camera windows must contain chunk_length + 1 frames; "
                f"got high={tuple(high.shape)}, left={tuple(left.shape)}, right={tuple(right.shape)}"
            )
        half_h, half_w = high.shape[-2] // 2, high.shape[-1] // 2
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        return torch.cat([high, torch.cat([left, right], dim=-1)], dim=-2)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        mode, _, _, sample = self._fetch_sample(int(idx))
        action_chunk = self._build_action_chunk(sample)
        initial_state = self._build_initial_state(sample)
        state_action = torch.cat([initial_state.unsqueeze(0), action_chunk], dim=0)
        if state_action.shape != (self._chunk_length + 1, _ACTION_DIM):
            raise AssertionError(f"Unexpected MOZ1 state/action shape: {tuple(state_action.shape)}")

        task = str(sample["task"])
        caption = random.choice([part.strip() for part in task.split(" | ") if part.strip()] or [task])
        idle_frames = self._compute_idle_frames(action_chunk)
        extras: dict[str, Any] = {
            "additional_view_description": (
                "The top row is the high third-person camera. The bottom-left and "
                "bottom-right views are the left and right wrist cameras."
            )
        }
        if idle_frames is not None:
            extras["idle_frames"] = idle_frames
        return self._build_result(
            mode=mode,
            video=self._compose_video(sample),
            action=state_action,
            ai_caption=caption,
            **extras,
        )


__all__ = ["MOZ1LeRobotDataset", "StateActionNormalizer"]
