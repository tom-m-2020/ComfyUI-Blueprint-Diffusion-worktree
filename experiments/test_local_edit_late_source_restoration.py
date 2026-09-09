from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_late_source_restoration as experiment


class LateSourceRestorationTests(unittest.TestCase):
    def test_mask_is_binary_source_center(self) -> None:
        direct = torch.ones((1, 8, 16))
        direct[..., 4:12] = 0
        latent = torch.zeros((1, 3, 2, 4))
        edit, source = experiment.latent_masks(direct, latent)
        self.assertTrue(torch.equal(edit + source, torch.ones_like(edit)))
        self.assertTrue(torch.all(source[..., 1:3] == 1))
        self.assertTrue(torch.all(source[..., :1] == 0))
        self.assertTrue(torch.all(source[..., 3:] == 0))

    def test_const_source_trajectory_and_terminal(self) -> None:
        y = torch.tensor([[[[2.0, 3.0]]]])
        z = torch.tensor([[[[7.0, 11.0]]]])
        sigma = torch.tensor(0.25)
        state = (1 - sigma) * y + sigma * z
        self.assertTrue(torch.equal(state, torch.tensor([[[[3.25, 5.0]]]])))
        terminal = (1 - torch.tensor(0.0)) * y + torch.tensor(0.0) * z
        self.assertTrue(torch.equal(terminal, y))

    def test_post_proposal_restore_changes_only_source(self) -> None:
        proposal = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        target = torch.tensor([[[[9.0, 8.0, 7.0, 6.0]]]])
        source = torch.tensor([[[[0.0, 1.0, 1.0, 0.0]]]])
        result = (1 - source) * proposal + source * target
        self.assertTrue(torch.equal(result[..., [0, 3]], proposal[..., [0, 3]]))
        self.assertTrue(torch.equal(result[..., 1:3], target[..., 1:3]))

    def test_declared_divergence_steps(self) -> None:
        self.assertEqual({experiment.reference.STEPS - 1}, {7})
        self.assertEqual({experiment.reference.STEPS - 2, experiment.reference.STEPS - 1}, {6, 7})


if __name__ == "__main__":
    unittest.main()
