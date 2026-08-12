#!/usr/bin/env python3

from __future__ import annotations

import unittest
from pathlib import Path

from cosmos_framework.scripts.inspect_action_policy_onnx import DEFAULT_TARGET_OPS, REWRITE_HINTS


class TargetOperatorTest(unittest.TestCase):
    def test_pow_is_rejected_for_policy_deployment(self) -> None:
        self.assertIn("Pow", DEFAULT_TARGET_OPS)
        self.assertIn("Mul", REWRITE_HINTS["Pow"])

    def test_nemotron_relu_squared_avoids_tensor_square(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py"
        ).read_text()
        self.assertNotIn(".square()", source)
        self.assertIn("class ReLUSquaredMulActivation(nn.Module)", source)
        self.assertIn("return activated * activated", source)
        self.assertIn('ACT2FN["relu2"] = ReLUSquaredMulActivation', source)


if __name__ == "__main__":
    unittest.main()
