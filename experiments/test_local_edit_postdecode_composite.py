from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_postdecode_composite as experiment


class PostDecodeCompositeTests(unittest.TestCase):
    def test_transition_is_inside_source_footprint(self) -> None:
        weight = experiment.source_weight()
        self.assertTrue(np.all(weight[:experiment.LEFT] == 0))
        self.assertTrue(np.all(weight[experiment.RIGHT:] == 0))
        self.assertEqual(weight[experiment.LEFT], 0)
        self.assertEqual(weight[experiment.RIGHT - 1], 0)
        self.assertEqual(weight[experiment.LEFT + experiment.TRANSITION_WIDTH - 1], 1)
        self.assertEqual(weight[experiment.RIGHT - experiment.TRANSITION_WIDTH], 1)

    def test_exact_source_interior(self) -> None:
        source = np.full((2, 1024, 3), 211, dtype=np.uint8)
        generated = np.full_like(source, 17)
        result = experiment.composite(source, generated, experiment.source_weight())
        left = experiment.LEFT + experiment.TRANSITION_WIDTH
        right = experiment.RIGHT - experiment.TRANSITION_WIDTH
        self.assertTrue(np.array_equal(result[:, left:right], source[:, left:right]))

    def test_generated_exterior_is_bit_exact(self) -> None:
        rng = np.random.default_rng(5)
        source = rng.integers(0, 256, (2, 1024, 3), dtype=np.uint8)
        generated = rng.integers(0, 256, (2, 1024, 3), dtype=np.uint8)
        result = experiment.composite(source, generated, experiment.source_weight())
        self.assertTrue(np.array_equal(result[:, :experiment.LEFT], generated[:, :experiment.LEFT]))
        self.assertTrue(np.array_equal(result[:, experiment.RIGHT:], generated[:, experiment.RIGHT:]))

    def test_weight_is_monotonic_on_both_strips(self) -> None:
        weight = experiment.source_weight()
        left = weight[experiment.LEFT:experiment.LEFT + experiment.TRANSITION_WIDTH]
        right = weight[experiment.RIGHT - experiment.TRANSITION_WIDTH:experiment.RIGHT]
        self.assertTrue(np.all(np.diff(left) >= 0))
        self.assertTrue(np.all(np.diff(right) <= 0))


if __name__ == "__main__":
    unittest.main()
