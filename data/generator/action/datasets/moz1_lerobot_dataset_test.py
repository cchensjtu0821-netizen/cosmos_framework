# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import json

import pytest
import torch

from cosmos_framework.data.generator.action.action_processing import ActionAffineNormalization
from cosmos_framework.data.generator.action.datasets.moz1_lerobot_dataset import (
    MOZ1LeRobotDataset,
    StateActionNormalizer,
    _resolve_feature,
    _unwrap_norm_stats,
)


def test_state_action_normalizer_uses_separate_first_row_stats():
    state = ActionAffineNormalization(offset=torch.full((20,), 10.0), scale=torch.full((20,), 2.0))
    action = ActionAffineNormalization(offset=torch.full((20,), -3.0), scale=torch.full((20,), 4.0))
    normalizer = StateActionNormalizer(state_normalizer=state, action_normalizer=action)
    values = torch.stack([torch.full((20,), 12.0), torch.full((20,), 1.0), torch.full((20,), 5.0)])

    normalized = normalizer.normalize_action(values)

    torch.testing.assert_close(normalized[0], torch.ones(20))
    torch.testing.assert_close(normalized[1], torch.ones(20))
    torch.testing.assert_close(normalized[2], torch.full((20,), 2.0))
    torch.testing.assert_close(normalizer.denormalize_action(normalized), values)


def test_nested_norm_stats_are_20d(tmp_path):
    path = tmp_path / "norm_stats.json"
    raw = {"norm_stats": {"actions": {"q01": [0.0] * 20, "q99": [1.0] * 20}}}
    path.write_text(json.dumps(raw))

    parsed = _unwrap_norm_stats(raw, "actions", path)

    assert parsed["q01"].shape == (20,)
    assert parsed["q99"].shape == (20,)


def test_nested_norm_stats_reject_wrong_width(tmp_path):
    path = tmp_path / "norm_stats.json"
    raw = {"norm_stats": {"state": {"q01": [0.0] * 19}}}

    with pytest.raises(ValueError, match="must have 20 values"):
        _unwrap_norm_stats(raw, "state", path)


def test_feature_resolution_accepts_leaf_name_and_rejects_ambiguity():
    features = {
        "observation.images.cam_high": {"dtype": "video"},
        "observation.state.leftarm_state_cart_pos": {"dtype": "float32"},
    }
    assert (
        _resolve_feature(features, "cam_high", role="camera", expected_dtype="video")
        == "observation.images.cam_high"
    )

    ambiguous = {**features, "action.leftarm_state_cart_pos": {"dtype": "float32"}}
    with pytest.raises(KeyError, match="matches="):
        _resolve_feature(ambiguous, "leftarm_state_cart_pos", role="state")


def test_feature_resolution_accepts_moz1_gripper_alias():
    features = {"leftarm_gripper_state_pos": {"dtype": "float32"}}

    resolved = _resolve_feature(
        features,
        "left_gripper_state",
        role="left gripper state",
        aliases=("leftarm_gripper_state_pos",),
    )

    assert resolved == "leftarm_gripper_state_pos"


def test_action_chunk_is_direct_concatenation_without_differencing():
    dataset = object.__new__(MOZ1LeRobotDataset)
    dataset._chunk_length = 2
    dataset._action_features = ("la", "lg", "ra", "rg", "torso")
    sample = {
        "la": torch.arange(12, dtype=torch.float32).reshape(2, 6),
        "lg": torch.tensor([0.25, 0.75]),
        "ra": torch.arange(12, 24, dtype=torch.float32).reshape(2, 6),
        "rg": torch.tensor([[1.0], [0.0]]),
        "torso": torch.arange(24, 36, dtype=torch.float32).reshape(2, 6),
    }

    action = dataset._build_action_chunk(sample)

    assert action.shape == (2, 20)
    torch.testing.assert_close(action[:, :6], sample["la"])
    torch.testing.assert_close(action[:, 6], sample["lg"])
    torch.testing.assert_close(action[:, 7:13], sample["ra"])
    torch.testing.assert_close(action[:, 13:14], sample["rg"])
    torch.testing.assert_close(action[:, 14:], sample["torso"])
