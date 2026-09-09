from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_photometric_reconciliation as experiment
import local_edit_postdecode_composite as policy


class PhotometricReconciliationTests(unittest.TestCase):
    def test_affine_recovers_known_luminance_map(self) -> None:
        generated = np.arange(24, 224, dtype=np.float64).reshape(10, 20, 1).repeat(3, axis=2)
        source = generated * 1.25 + 7
        scale, offset = experiment.robust_luminance_affine(source, generated)
        self.assertAlmostEqual(scale, 1.25, places=10)
        self.assertAlmostEqual(offset, 7.0, places=10)

    def test_reconciled_ownership_is_exact(self) -> None:
        rng = np.random.default_rng(8)
        source = rng.integers(0, 256, (8, 1024, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (8, 1024, 3), dtype=np.uint8)
        result, _ = experiment.reconciled_composite(source, generated)
        self.assertTrue(np.array_equal(result[:, 280:744], source[:, 280:744]))
        self.assertTrue(np.array_equal(result[:, :256], generated[:, :256]))
        self.assertTrue(np.array_equal(result[:, 768:], generated[:, 768:]))

    def test_outer_transition_pixels_are_generated(self) -> None:
        source = np.full((4, 1024, 3), 220, dtype=np.uint8)
        generated = np.full_like(source, 30)
        result, _ = experiment.reconciled_composite(source, generated)
        self.assertTrue(np.array_equal(result[:, 256], generated[:, 256]))
        self.assertTrue(np.array_equal(result[:, 767], generated[:, 767]))

    def test_policy_geometry_is_unchanged(self) -> None:
        self.assertEqual(policy.TRANSITION_WIDTH, 24)
        self.assertEqual((policy.LEFT, policy.RIGHT), (256, 768))


if __name__ == "__main__":
    unittest.main()
