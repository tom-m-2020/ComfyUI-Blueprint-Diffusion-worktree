from __future__ import annotations

import unittest

import torch

from experiments.blueprint_staged_refinement_geometry import (
    Geometry,
    bounded_transfer,
    evaluate,
    footprints,
    full_transfer,
    starts,
)


class TestStagedRefinementGeometry(unittest.TestCase):
    def test_end_aligned_overlap_plan_is_deterministic(self):
        geometry = Geometry((37, 61), (131, 227), (41, 53), (29, 37), (64, 64))
        planned = footprints(geometry)
        self.assertEqual(planned, footprints(geometry))
        self.assertEqual(planned[0], (0, 0, 41, 53))
        self.assertEqual(planned[-1], (90, 174, 41, 53))

    def test_invalid_stride_or_footprint_fails_closed(self):
        for arguments in ((10, 11, 5), (10, 5, 0), (10, 5, 6)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                starts(*arguments)

    def test_bounded_transfer_matches_complete_map(self):
        geometry = Geometry((37, 61), (131, 227), (41, 53), (29, 37), (64, 64))
        value = torch.randn((1, 4, *geometry.g_hw), generator=torch.Generator().manual_seed(9), dtype=torch.float64)
        for rect in footprints(geometry):
            expected = full_transfer(value, geometry, rect)
            actual, source_hw = bounded_transfer(value, geometry, rect)
            self.assertLessEqual(float((expected - actual).abs().max()), 1e-12)
            self.assertLess(source_hw[0] * source_hw[1], geometry.h_hw[0] * geometry.h_hw[1])

    def test_coverage_and_bounded_residency(self):
        result = evaluate(
            Geometry((48, 80), (257, 513), (48, 64), (32, 40), (64, 64)),
            7005,
        )
        self.assertGreater(result["coverage_min"], 0.0)
        self.assertTrue(result["finite_assembly"])
        self.assertLess(result["max_bounded_source_elements"], result["full_mapped_anchor_elements"])


if __name__ == "__main__":
    unittest.main()

