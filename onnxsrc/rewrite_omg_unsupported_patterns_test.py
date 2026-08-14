#!/usr/bin/env python3

from __future__ import annotations

import unittest

from cosmos_framework.onnxsrc.rewrite_omg_unsupported_patterns import (
    _build_argument_parser,
    _depth_to_space_target,
    _space_to_depth_output_channel,
)


class SpatialChannelMappingTest(unittest.TestCase):
    def test_space_to_depth_matches_dcr_depth_order(self) -> None:
        channels = 3
        block_size = 2
        actual = [
            _space_to_depth_output_channel(
                channel,
                block_row,
                block_col,
                channels,
                block_size,
            )
            for block_row in range(block_size)
            for block_col in range(block_size)
            for channel in range(channels)
        ]
        self.assertEqual(actual, list(range(channels * block_size * block_size)))

    def test_dcr_depth_to_space_is_space_to_depth_inverse(self) -> None:
        channels = 3
        block_size = 2
        for block_row in range(block_size):
            for block_col in range(block_size):
                for channel in range(channels):
                    input_channel = _space_to_depth_output_channel(
                        channel,
                        block_row,
                        block_col,
                        channels,
                        block_size,
                    )
                    self.assertEqual(
                        _depth_to_space_target(
                            input_channel,
                            channels,
                            block_size,
                            "DCR",
                        ),
                        (channel, block_row, block_col),
                    )

    def test_crd_depth_order(self) -> None:
        channels = 3
        block_size = 2
        for channel in range(channels):
            for block_row in range(block_size):
                for block_col in range(block_size):
                    input_channel = (
                        channel * block_size * block_size
                        + block_row * block_size
                        + block_col
                    )
                    self.assertEqual(
                        _depth_to_space_target(
                            input_channel,
                            channels,
                            block_size,
                            "CRD",
                        ),
                        (channel, block_row, block_col),
                    )

    def test_rejects_unknown_depth_to_space_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported DepthToSpace mode"):
            _depth_to_space_target(0, 3, 2, "UNKNOWN")


class ProfilingOnlySwitchTest(unittest.TestCase):
    def test_dynamic_trig_bypass_is_disabled_by_default(self) -> None:
        args = _build_argument_parser().parse_args(
            [
                "--input",
                "input.onnx",
                "--output",
                "output.onnx",
                "--report",
                "report.json",
            ]
        )
        self.assertFalse(args.bypass_dynamic_trig_for_profiling)

    def test_dynamic_trig_bypass_requires_explicit_switch(self) -> None:
        args = _build_argument_parser().parse_args(
            [
                "--input",
                "input.onnx",
                "--output",
                "output.onnx",
                "--report",
                "report.json",
                "--bypass-dynamic-trig-for-profiling",
            ]
        )
        self.assertTrue(args.bypass_dynamic_trig_for_profiling)


if __name__ == "__main__":
    unittest.main()
