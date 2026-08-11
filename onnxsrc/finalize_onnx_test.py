#!/usr/bin/env python3

from __future__ import annotations

import unittest

from cosmos_framework.onnxsrc.finalize_onnx import _consecutive_row_runs


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


if __name__ == "__main__":
    unittest.main()
