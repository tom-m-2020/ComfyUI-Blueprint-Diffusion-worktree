from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_prediction_state_ownership as experiment


class PredictionStateOwnershipTests(unittest.TestCase):
    def test_mask_convention(self) -> None:
        source = torch.zeros((1, 4, 32, 64))
        edit, locked = experiment.latent_masks(source)
        self.assertTrue(torch.all(edit[..., :16] == 1))
        self.assertTrue(torch.all(edit[..., 16:48] == 0))
        self.assertTrue(torch.all(edit[..., 48:] == 1))
        self.assertTrue(torch.equal(locked, 1 - edit))

    def test_source_trajectory_terminal_and_fixed_noise(self) -> None:
        source = torch.randn((1, 4, 2, 3))
        noise = torch.randn_like(source)
        self.assertTrue(torch.equal(experiment.source_state(source, noise, torch.tensor(0.0)), source))
        self.assertTrue(torch.equal(experiment.source_state(source, noise, torch.tensor(1.0)), noise))

    def test_authoritative_derivative_predicts_clean_source(self) -> None:
        source = torch.randn((1, 4, 2, 3))
        state = torch.randn_like(source)
        sigma = torch.tensor(0.37)
        derivative = (state - source) / sigma
        predicted = state - sigma * derivative
        torch.testing.assert_close(predicted, source)

    def test_euler_authoritative_derivative_reaches_next_source_state(self) -> None:
        source = torch.randn((1, 4, 2, 3))
        noise = torch.randn_like(source)
        sigma = torch.tensor(0.7)
        sigma_next = torch.tensor(0.4)
        state = experiment.source_state(source, noise, sigma)
        derivative = (state - source) / sigma
        proposal = state + (sigma_next - sigma) * derivative
        torch.testing.assert_close(proposal, experiment.source_state(source, noise, sigma_next))

    def test_boundary_mask_is_two_latent_columns_each_side(self) -> None:
        value = torch.zeros((1, 4, 32, 64))
        mask = experiment.latent_boundary_mask(value)
        self.assertEqual(int(mask.sum()), 8 * 32)
        self.assertTrue(torch.all(mask[..., 14:18] == 1))
        self.assertTrue(torch.all(mask[..., 46:50] == 1))


if __name__ == "__main__":
    unittest.main()
