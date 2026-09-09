from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_source_coevolution as experiment


class SourceCoevolutionTests(unittest.TestCase):
    def test_support_partition(self):
        value = torch.zeros((1, 128, 32, 64))
        masks = experiment.support_masks(value)
        self.assertFalse(bool((masks["boundary_source"] * masks["distant_source"]).any()))
        self.assertTrue(torch.equal(masks["boundary_source"] + masks["distant_source"], masks["source"]))
        self.assertEqual(int(masks["boundary_source"].sum()), 4 * 32)

    def test_frequency_split_reconstructs_delta(self):
        value = torch.randn((1, 4, 8, 16))
        source = torch.zeros((1, 1, 8, 16))
        source[..., 4:12] = 1
        delta = value * source
        low, high = experiment.frequency_split(delta, source)
        self.assertTrue(torch.allclose(low + high, delta))
        self.assertEqual(float(low.masked_select((1 - source).bool().expand_as(low)).abs().max()), 0.0)

    def test_directional_identity(self):
        ordinary = torch.zeros((1, 1, 2, 2))
        epsilon = torch.ones_like(ordinary)
        result = experiment.directional(epsilon, ordinary, epsilon, torch.ones_like(ordinary))
        self.assertAlmostEqual(result["recovery_ratio"], 1.0)
        self.assertAlmostEqual(result["cosine"], 1.0)
        self.assertAlmostEqual(result["projection"], 1.0)


if __name__ == "__main__":
    unittest.main()
