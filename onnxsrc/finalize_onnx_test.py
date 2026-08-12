#!/usr/bin/env python3

from __future__ import annotations

import unittest

from cosmos_framework.onnxsrc.finalize_onnx import _consecutive_row_runs, _linear_weight_name


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


if __name__ == "__main__":
    unittest.main()
