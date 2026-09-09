from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_source_spatial_structure as experiment


class SpatialStructureTests(unittest.TestCase):
    def setUp(self):
        self.value = torch.zeros((1, 4, 32, 64))
        self.value[..., 16:48] = torch.arange(4 * 32 * 32.0).reshape(1, 4, 32, 32)

    def test_transforms_remain_source_only_and_preserve_values(self):
        variants = experiment.transform_variants(self.value)
        reference = torch.sort(experiment.source_crop(self.value).flatten()).values
        for pulse in variants.values():
            self.assertEqual(float(pulse[..., :16].abs().max()), 0.0)
            self.assertEqual(float(pulse[..., 48:].abs().max()), 0.0)
            self.assertTrue(torch.equal(torch.sort(experiment.source_crop(pulse).flatten()).values, reference))

    def test_predeclared_block_permutation_is_bijective(self):
        self.assertEqual(sorted(experiment.BLOCK_PERMUTATION), list(range(8)))

    def test_translation_is_fixed_quarter_width(self):
        self.assertEqual(experiment.TRANSLATION, (8, 16))


if __name__ == "__main__":
    unittest.main()
