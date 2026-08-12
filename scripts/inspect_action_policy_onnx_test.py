#!/usr/bin/env python3

from __future__ import annotations

import unittest

from cosmos_framework.scripts.inspect_action_policy_onnx import DEFAULT_TARGET_OPS, REWRITE_HINTS


class TargetOperatorTest(unittest.TestCase):
    def test_pow_is_rejected_for_policy_deployment(self) -> None:
        self.assertIn("Pow", DEFAULT_TARGET_OPS)
        self.assertIn("Mul", REWRITE_HINTS["Pow"])


if __name__ == "__main__":
    unittest.main()
