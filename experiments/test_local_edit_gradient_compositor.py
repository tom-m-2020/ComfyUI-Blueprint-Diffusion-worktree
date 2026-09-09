from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_gradient_compositor as experiment


class GradientCompositorTests(unittest.TestCase):
    def test_fixed_support(self) -> None:
        self.assertEqual(experiment.SUPPORT, 64)

    def test_mixed_gradient_uses_larger_magnitude_per_channel(self) -> None:
        result = experiment._mixed_gradient(
            np.array([10.0, 10.0, 10.0]), np.array([8.0, 4.0, 15.0]),
            np.array([10.0, 10.0, 10.0]), np.array([3.0, 8.0, 11.0]),
        )
        np.testing.assert_array_equal(result, np.array([7.0, 6.0, -5.0]))

    def test_uint8_inputs_do_not_wrap_gradient_sign(self) -> None:
        result = experiment._mixed_gradient(
            np.array([0.0]), np.array([0.0]), np.array([3], dtype=np.uint8), np.array([250], dtype=np.uint8)
        )
        np.testing.assert_array_equal(result, np.array([-247.0]))

    def test_ownership_is_exact(self) -> None:
        rng = np.random.default_rng(41)
        source = rng.integers(0, 256, (12, 1024, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (12, 1024, 3), dtype=np.uint8)
        result = experiment.gradient_composite(source, generated)
        self.assertTrue(np.array_equal(result[:, 320:704], source[:, 320:704]))
        self.assertTrue(np.array_equal(result[:, :256], generated[:, :256]))
        self.assertTrue(np.array_equal(result[:, 768:], generated[:, 768:]))

    def test_identical_constant_inputs_are_identity(self) -> None:
        image = np.full((12, 1024, 3), 123, dtype=np.uint8)
        self.assertTrue(np.array_equal(experiment.gradient_composite(image, image), image))


if __name__ == "__main__":
    unittest.main()
