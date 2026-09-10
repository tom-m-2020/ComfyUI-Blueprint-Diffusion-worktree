from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_source_stitch_trajectory as experiment


class SourceStitchTests(unittest.TestCase):
    def test_const_source_trajectory_endpoints(self):
        source = torch.tensor([[[[2.0]]]])
        noise = torch.tensor([[[[-3.0]]]])
        self.assertTrue(torch.equal(experiment.source_state(source, noise, torch.tensor(1.0)), noise))
        self.assertTrue(torch.equal(experiment.source_state(source, noise, torch.tensor(0.0)), source))

    def test_stitch_changes_only_source(self):
        proposal = torch.arange(8.0).reshape(1, 1, 1, 8)
        target = torch.full_like(proposal, -2)
        editable = torch.tensor([[[[1, 1, 0, 0, 0, 0, 1, 1]]]], dtype=torch.float32)
        result = experiment.stitch(proposal, target, editable, True)
        self.assertTrue(torch.equal(result * editable, proposal * editable))
        self.assertTrue(torch.equal(result * (1 - editable), target * (1 - editable)))

    def test_disabled_stitch_is_bit_exact(self):
        proposal = torch.randn((1, 2, 3, 4))
        result = experiment.stitch(proposal, torch.zeros_like(proposal), torch.ones((1, 1, 3, 4)), False)
        self.assertTrue(torch.equal(result, proposal))

    def test_fixed_policies(self):
        self.assertEqual(experiment.ARMS["A_ORDINARY"], frozenset())
        self.assertEqual(experiment.ARMS["B_SINGLE_STITCH"], frozenset({0}))
        self.assertIsNone(experiment.ARMS["C_REPEATED_STITCH"])


if __name__ == "__main__":
    unittest.main()
