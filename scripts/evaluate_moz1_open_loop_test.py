# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Tests for strict pairing and action-error reporting in MOZ1 open-loop eval."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.scripts.evaluate_moz1_open_loop import _action_matrix, score


def _matrix(value: float) -> list[list[float]]:
    return [[value] * 20 for _ in range(16)]


class EvaluateMoz1OpenLoopTest(unittest.TestCase):
    def test_all_predictions_are_required_and_scored_in_stored_space(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            samples = []
            for index, prediction in enumerate((_matrix(2.0), _matrix(0.0))):
                name = f"sample_{index}"
                target_path = root / f"target_{index}.json"
                target_path.write_text(json.dumps(_matrix(1.0)))
                prediction_dir = root / "predictions" / name
                prediction_dir.mkdir(parents=True)
                (prediction_dir / "action.json").write_text(json.dumps(prediction))
                (prediction_dir / "sample_outputs.json").write_text(json.dumps({"status": "success"}))
                samples.append(
                    {"name": name, "dataset_index": index, "episode_id": index, "target_action": str(target_path)}
                )
            (root / "manifest.json").write_text(
                json.dumps({"split": "val", "split_seed": 42, "sample_seed": 42, "samples": samples})
            )

            score(out_dir=root)
            metrics = json.loads((root / "metrics.json").read_text())
            self.assertEqual(metrics["num_samples"], 2)
            self.assertEqual(metrics["num_episodes"], 2)
            self.assertEqual(metrics["mae"], 1.0)
            self.assertEqual(metrics["rmse"], 1.0)
            self.assertEqual(metrics["zero_action_mae"], 1.0)
            self.assertEqual(metrics["per_step_mae"], [1.0] * 16)
            self.assertEqual(metrics["groups"]["left_position"]["mae"], 1.0)

            (root / "metrics.json").unlink()
            (root / "predictions" / "sample_1" / "action.json").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "Missing predictions for 1 of 2"):
                score(out_dir=root)
            self.assertFalse((root / "metrics.json").exists())

    def test_rejects_nonfinite_or_malformed_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _action_matrix(_matrix(float("nan")), name="bad")
        with self.assertRaisesRegex(ValueError, "shape"):
            _action_matrix([[0.0] * 20] * 15, name="short")


if __name__ == "__main__":
    unittest.main()
