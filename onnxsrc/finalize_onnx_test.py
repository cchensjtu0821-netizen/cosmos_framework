#!/usr/bin/env python3

from __future__ import annotations

import unittest
from types import SimpleNamespace

from cosmos_framework.onnxsrc.finalize_onnx import (
    _consecutive_row_runs,
    _linear_weight_name,
    _prune_unreachable_nodes,
)


class ConsecutiveRowRunsTest(unittest.TestCase):
    def test_empty_rows(self) -> None:
        self.assertEqual(_consecutive_row_runs([]), [])

    def test_one_contiguous_run(self) -> None:
        self.assertEqual(
            _consecutive_row_runs([7, 8, 9, 10]),
            [(0, 4, 7, 11)],
        )

    def test_multiple_runs_preserve_update_offsets(self) -> None:
        self.assertEqual(
            _consecutive_row_runs([2, 3, 4, 10, 11, 19]),
            [
                (0, 3, 2, 5),
                (3, 5, 10, 12),
                (5, 6, 19, 20),
            ],
        )

    def test_single_row(self) -> None:
        self.assertEqual(_consecutive_row_runs([5]), [(0, 1, 5, 6)])


class LinearWeightNameTest(unittest.TestCase):
    def test_matmul_direct_weight_maps_to_module_name(self) -> None:
        node = type("Node", (), {"input": ["hidden", "net.layers.0.linear.weight"]})()
        self.assertEqual(
            _linear_weight_name(node, {"net.layers.0.linear.weight"}),
            ("layers.0.linear", ["net.layers.0.linear.weight"]),
        )

    def test_dynamic_matmul_is_not_treated_as_linear(self) -> None:
        node = type("Node", (), {"input": ["query", "key"]})()
        self.assertEqual(_linear_weight_name(node, set()), (None, []))

    def test_matmul_traces_weight_through_transpose(self) -> None:
        node = type("Node", (), {"input": ["hidden", "transposed_weight"]})()
        transpose = type(
            "Node",
            (),
            {"op_type": "Transpose", "input": ["net.layers.0.linear.weight"]},
        )()
        self.assertEqual(
            _linear_weight_name(
                node,
                {"net.layers.0.linear.weight"},
                {"transposed_weight": transpose},
            ),
            ("layers.0.linear", ["net.layers.0.linear.weight"]),
        )


class PruneUnreachableNodesTest(unittest.TestCase):
    @staticmethod
    def _node(name: str, op_type: str, inputs: list[str], outputs: list[str]):
        return SimpleNamespace(
            name=name,
            op_type=op_type,
            input=inputs,
            output=outputs,
            attribute=[],
        )

    def test_removes_dead_cast_but_preserves_live_cast(self) -> None:
        graph = SimpleNamespace(
            node=[
                self._node("live_cast", "Cast", ["input"], ["live"]),
                self._node("output_add", "Add", ["live", "bias"], ["output"]),
                self._node("dead_cast", "Cast", ["dead_indices"], ["dead"]),
            ],
            input=[SimpleNamespace(name="input")],
            output=[SimpleNamespace(name="output")],
            initializer=[SimpleNamespace(name="bias"), SimpleNamespace(name="dead_indices")],
            value_info=[SimpleNamespace(name="live"), SimpleNamespace(name="dead")],
        )

        report = _prune_unreachable_nodes(graph)

        self.assertFalse(report["skipped"])
        self.assertEqual(report["removed_node_count"], 1)
        self.assertEqual(report["removed_node_op_counts"], {"Cast": 1})
        self.assertEqual([node.name for node in graph.node], ["live_cast", "output_add"])
        self.assertEqual([tensor.name for tensor in graph.initializer], ["bias"])
        self.assertEqual([value.name for value in graph.value_info], ["live"])

    def test_preserves_shared_dependency_reachable_from_output(self) -> None:
        graph = SimpleNamespace(
            node=[
                self._node("source", "Mul", ["input", "scale"], ["shared"]),
                self._node("left", "Add", ["shared", "bias"], ["left_out"]),
                self._node("right", "Add", ["shared", "bias"], ["right_out"]),
                self._node("output", "Add", ["left_out", "right_out"], ["result"]),
            ],
            input=[SimpleNamespace(name="input")],
            output=[SimpleNamespace(name="result")],
            initializer=[SimpleNamespace(name="scale"), SimpleNamespace(name="bias")],
            value_info=[],
        )

        report = _prune_unreachable_nodes(graph)

        self.assertEqual(report["removed_node_count"], 0)
        self.assertEqual(len(graph.node), 4)


if __name__ == "__main__":
    unittest.main()
