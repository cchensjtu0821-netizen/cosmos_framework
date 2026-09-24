# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for conditioned MOZ1 open-loop policy input construction."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_moz1_state_row_matches_training_contract(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from cosmos_framework.data.generator.action.action_processing import ActionProcessor
    from cosmos_framework.inference import action as action_inference
    from cosmos_framework.inference.args import ModelMode

    state_path = tmp_path / "state.json"
    stats_path = tmp_path / "norm_stats.json"
    state_path.write_text(json.dumps([2.0] * 20))
    stats_path.write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"q01": [0.0] * 20, "q99": [4.0] * 20},
                    "actions": {"q01": [-2.0] * 20, "q99": [2.0] * 20},
                }
            }
        )
    )
    monkeypatch.setattr(
        action_inference,
        "read_media_frames",
        lambda path, max_frames: (torch.zeros(3, 1, 360, 320, dtype=torch.uint8), 30.0),
    )
    batch = action_inference.get_action_sample_data(
        model_config=SimpleNamespace(input_video_key="video"),
        batch_size=1,
        prompt="Move the object.",
        vision_path=tmp_path / "unused.png",
        model_mode=ModelMode.POLICY,
        action_path=None,
        state_path=state_path,
        stats_path=stats_path,
        domain_name="moz1",
        view_point="concat_view",
        resolution="256",
        action_chunk_size=16,
        max_action_dim=64,
        fps=30,
        device="cpu",
    )

    action = batch["action"][0][0]
    plan = batch["sequence_plan"][0]
    record = batch["action_processing_record"][0]
    assert action.shape == (17, 64)
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == [0]
    assert "The top row is the high third-person camera" in batch["ai_caption"][0]
    torch.testing.assert_close(action[0], torch.zeros(64))
    torch.testing.assert_close(action[1:], torch.zeros(16, 64))
    predicted = torch.ones(17, 64)
    external = ActionProcessor.postprocess_action(predicted, record)
    assert external.shape == (17, 20)
    torch.testing.assert_close(external[0], torch.full((20,), 4.0))
    torch.testing.assert_close(external[1:], torch.full((16, 20), 2.0))
