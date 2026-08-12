#!/usr/bin/env python3

from __future__ import annotations

import unittest
import sys
from types import SimpleNamespace

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    # This unit only exercises a graph-structure rewrite that does not use NumPy.
    sys.modules["numpy"] = SimpleNamespace()

from cosmos_framework.scripts.rewrite_action_policy_onnx import _decompose_gemm_nodes


class _Helper:
    @staticmethod
    def get_attribute_value(attribute):
        return attribute.value

    @staticmethod
    def make_node(op_type, inputs, outputs, name="", **attributes):
        return SimpleNamespace(
            op_type=op_type,
            input=list(inputs),
            output=list(outputs),
            name=name,
            attribute=[
                SimpleNamespace(name=key, value=value) for key, value in attributes.items()
            ],
        )


class DecomposeGemmTest(unittest.TestCase):
    def _rewrite(self, inputs, **attributes):
        onnx = SimpleNamespace(helper=_Helper())
        graph = SimpleNamespace(
            node=[_Helper.make_node("Gemm", inputs, ["output"], name="linear", **attributes)],
            input=[],
            output=[],
            value_info=[],
            initializer=[],
        )
        changes = []
        count = _decompose_gemm_nodes(graph, onnx, changes)
        return graph, changes, count

    def test_biasless_gemm_becomes_only_matmul(self):
        graph, changes, count = self._rewrite(["input", "linear.weight"])

        self.assertEqual(count, 1)
        self.assertEqual([node.op_type for node in graph.node], ["MatMul"])
        self.assertEqual(graph.node[0].output, ["output"])
        self.assertFalse(changes[0]["has_bias"])

    def test_bias_add_is_emitted_only_for_present_bias(self):
        graph, changes, _ = self._rewrite(["input", "linear.weight", "linear.bias"])

        self.assertEqual([node.op_type for node in graph.node], ["MatMul", "Add"])
        self.assertEqual(graph.node[1].input[1], "linear.bias")
        self.assertTrue(changes[0]["has_bias"])

    def test_empty_optional_bias_does_not_emit_add(self):
        graph, _, _ = self._rewrite(["input", "linear.weight", ""])

        self.assertEqual([node.op_type for node in graph.node], ["MatMul"])

    def test_transposed_weight_gets_explicit_transpose(self):
        graph, _, _ = self._rewrite(["input", "linear.weight"], transB=1)

        self.assertEqual([node.op_type for node in graph.node], ["Transpose", "MatMul"])
        self.assertEqual(graph.node[0].attribute[0].value, [1, 0])


if __name__ == "__main__":
    unittest.main()
