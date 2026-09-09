from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_source_block_localization as experiment


class BlockLocalizationTests(unittest.TestCase):
    def test_native_block_sets(self):
        self.assertEqual(experiment.BOUNDARIES, ("before_double_0", "after_double_4", "after_single_4",
                                                "after_single_9", "after_single_14", "after_single_19"))
        self.assertEqual(experiment.ATTENTION_BLOCKS, (("double", 0), ("double", 4), ("single", 4),
                                                     ("single", 9), ("single", 14), ("single", 19)))

    def test_crossover_replaces_only_source(self):
        true = torch.zeros((1, 2048, 3)); translated = torch.ones_like(true)
        editable, source = experiment.generated_masks(torch.device("cpu"))
        modified = true.clone(); modified[:, source] = translated[:, source]
        result = experiment.crossover_invariants(modified, true, translated, editable, source)
        self.assertTrue(result["editable_unchanged"])
        self.assertTrue(result["source_equals_translated"])


if __name__ == "__main__":
    unittest.main()
