#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cosmos_framework.onnxsrc.modify_dopt_config import update_config


class UpdateConfigTest(unittest.TestCase):
    def test_selects_static_int8_linear_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "dopt_config.json"
            report_path = root / "report.json"
            config_path.write_text(
                json.dumps(
                    {
                        "layer_strategy": {
                            "language_model.layers.0.linear": {
                                "type": "torch.nn.modules.linear.Linear",
                                "quant_strategy": "legacy",
                                "output": {"bit": 4},
                            },
                            "vision_time_embedder.mlp.0": {
                                "type": "torch.nn.modules.linear.Linear",
                            },
                            "vision_time_embedder.mlp.2": {
                                "type": "torch.nn.modules.linear.Linear",
                            },
                            "action_time_embedder.mlp.0": {
                                "type": "torch.nn.modules.linear.Linear",
                            },
                            "action_time_embedder.mlp.2": {
                                "type": "torch.nn.modules.linear.Linear",
                            },
                            "language_model.lm_head": {
                                "type": "torch.nn.modules.linear.Linear",
                            },
                            "activation": {"type": "torch.nn.modules.activation.ReLU"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            update_config(
                config_path,
                bit=8,
                exclude_regexes=[],
                report_path=report_path,
            )

            updated = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(updated["layer_strategy"]),
                [
                    "language_model.layers.0.linear",
                    "vision_time_embedder.mlp.0",
                    "vision_time_embedder.mlp.2",
                    "action_time_embedder.mlp.0",
                    "action_time_embedder.mlp.2",
                ],
            )
            selected = updated["layer_strategy"]["language_model.layers.0.linear"]
            self.assertEqual(
                selected["weight"],
                {
                    "bit": 8,
                    "per_channel": True,
                    "weight_algo": "min_max",
                    "unsigned_quant": False,
                },
            )
            self.assertEqual(
                selected["input"],
                {"bit": 8, "per_channel": False, "input_algo": "min_max"},
            )
            self.assertNotIn("output", selected)

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertIn("static per-tensor min_max input", report["policy"])
            self.assertEqual(report["selected_linear_count"], 5)
            self.assertEqual(report["excluded_linear_count"], 1)
            self.assertEqual(report["non_linear_count"], 1)

    def test_rejects_legacy_shared_timestep_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "dopt_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "layer_strategy": {
                            "time_embedder.mlp.0": {"type": "torch.nn.modules.linear.Linear"},
                            "time_embedder.mlp.2": {"type": "torch.nn.modules.linear.Linear"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Rerun pipeline Step 1"):
                update_config(
                    config_path,
                    bit=8,
                    exclude_regexes=[],
                    report_path=root / "report.json",
                )


if __name__ == "__main__":
    unittest.main()
