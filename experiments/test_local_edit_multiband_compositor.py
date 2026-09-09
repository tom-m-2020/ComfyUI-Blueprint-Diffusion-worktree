from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_multiband_compositor as experiment


class MultibandCompositorTests(unittest.TestCase):
    def test_fixed_supports(self) -> None:
        self.assertEqual((experiment.HIGH_SUPPORT, experiment.MID_SUPPORT, experiment.LOW_SUPPORT), (24, 64, 128))
        self.assertEqual((experiment.HIGH_BLUR_RADIUS, experiment.LOW_BLUR_RADIUS), (2.0, 8.0))

    def test_weights_are_inward_and_monotonic(self) -> None:
        for support in (24, 64, 128):
            weight = experiment.inward_weight(support)
            self.assertTrue(np.all(weight[:256] == 0))
            self.assertTrue(np.all(weight[768:] == 0))
            self.assertTrue(np.all(np.diff(weight[256:256 + support]) >= 0))
            self.assertTrue(np.all(np.diff(weight[768 - support:768]) <= 0))

    def test_exact_interior_and_generated_exterior(self) -> None:
        rng = np.random.default_rng(17)
        source = rng.integers(0, 256, (32, 1024, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (32, 1024, 3), dtype=np.uint8)
        result = experiment.multiband_composite(source, generated)
        self.assertTrue(np.array_equal(result[:, 384:640], source[:, 384:640]))
        self.assertTrue(np.array_equal(result[:, :256], generated[:, :256]))
        self.assertTrue(np.array_equal(result[:, 768:], generated[:, 768:]))

    def test_identical_inputs_are_identity(self) -> None:
        rng = np.random.default_rng(21)
        image = rng.integers(0, 256, (16, 1024, 3), dtype=np.uint8)
        self.assertTrue(np.array_equal(experiment.multiband_composite(image, image), image))


if __name__ == "__main__":
    unittest.main()
