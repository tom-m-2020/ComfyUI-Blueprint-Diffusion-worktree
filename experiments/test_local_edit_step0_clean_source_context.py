from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_step0_clean_source_context as experiment


class CleanSourceContextTests(unittest.TestCase):
    def test_clean_and_noisy_source_are_distinct(self) -> None:
        clean = torch.zeros((1, 128, 2, 3))
        noise = torch.ones_like(clean)
        self.assertFalse(torch.equal(clean, noise))

    def test_clean_probe_has_independent_storage(self) -> None:
        sampler = experiment.CleanSourceStepZeroSampler(
            torch.zeros((1, 128, 32, 64)), torch.ones((1, 128, 32, 64)), 0
        )
        self.assertIsNot(sampler.noisy_probe, sampler.clean_probe)

    def test_query_partition_is_unchanged(self) -> None:
        editable, locked = experiment.noisy.generated_masks(torch.device("cpu"))
        self.assertEqual(int(editable.sum()), 1024)
        self.assertEqual(int(locked.sum()), 1024)


if __name__ == "__main__":
    unittest.main()
