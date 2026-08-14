#!/usr/bin/env python3

from __future__ import annotations

import unittest

from cosmos_framework.onnxsrc.analyze_omg_unsupported_patterns import (
    _binary_bypass_candidate,
    _broadcast_shape,
    _depth_to_space_conv_transpose_candidate,
    _single_axis_concat_candidate,
    _space_to_depth_conv_candidate,
)


class BroadcastShapeTest(unittest.TestCase):
    def test_broadcasts_singleton_dimension(self) -> None:
        self.assertEqual(
            _broadcast_shape((3093, 1, 128), (3093, 16, 128)),
            (3093, 16, 128),
        )

    def test_rejects_incompatible_dimensions(self) -> None:
        self.assertIsNone(_broadcast_shape((3093, 124), (3093, 128)))


class ExpandCandidateTest(unittest.TestCase):
    def test_single_axis_expand_can_be_concat(self) -> None:
        self.assertEqual(
            _single_axis_concat_candidate((4, 1, 8), (4, 2, 8)),
            {"axis": 1, "copies": 2},
        )

    def test_multi_axis_expand_is_not_one_concat(self) -> None:
        self.assertIsNone(_single_axis_concat_candidate((1, 1, 8), (4, 2, 8)))

    def test_binary_consumer_can_absorb_expand(self) -> None:
        proof = _binary_bypass_candidate(
            (3093, 1, 128),
            (3093, 16, 128),
            (3093, 16, 128),
            (3093, 16, 128),
        )
        self.assertIsNotNone(proof)

    def test_binary_consumer_cannot_absorb_real_width_mismatch(self) -> None:
        self.assertIsNone(
            _binary_bypass_candidate(
                (3093, 124),
                (3093, 128),
                (3093, 128),
                (3093, 128),
            )
        )


class SpatialRearrangementCandidateTest(unittest.TestCase):
    def test_space_to_depth_can_be_fixed_conv(self) -> None:
        report = _space_to_depth_conv_candidate(
            (9, 48, 66, 80),
            (9, 192, 33, 40),
            2,
        )
        self.assertTrue(report["eligible"])
        self.assertEqual(report["weight_shape"], [192, 48, 2, 2])
        self.assertEqual(report["weight_nonzero_count"], 192)

    def test_depth_to_space_can_be_fixed_conv_transpose(self) -> None:
        report = _depth_to_space_conv_transpose_candidate(
            (9, 192, 33, 40),
            (9, 48, 66, 80),
            2,
            "DCR",
        )
        self.assertTrue(report["eligible"])
        self.assertEqual(report["weight_shape"], [192, 48, 2, 2])
        self.assertEqual(report["weight_nonzero_count"], 192)

    def test_depth_to_space_rejects_bad_channel_count(self) -> None:
        report = _depth_to_space_conv_transpose_candidate(
            (9, 190, 33, 40),
            (9, 48, 66, 80),
            2,
            "DCR",
        )
        self.assertFalse(report["eligible"])


if __name__ == "__main__":
    unittest.main()
