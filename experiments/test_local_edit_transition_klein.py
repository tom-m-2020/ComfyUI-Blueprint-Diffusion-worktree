from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_hard_source_klein as base
import local_edit_transition_klein as transition


class TransitionAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(91)
        self.source = torch.randn((1, 4, 3, 7), generator=generator)
        self.noise = torch.randn((1, 4, 3, 7), generator=generator)
        self.denoised = torch.randn((1, 4, 3, 7), generator=generator)
        self.editable = torch.zeros((1, 1, 3, 7))
        self.editable[..., 4:] = 1
        self.locked = 1 - self.editable
        self.band = base.derive_transition_band(self.editable, 2)
        self.weights = transition.linear_transition_weights(self.editable, 2)
        self.sigma = torch.tensor(0.8)
        self.sigma_next = torch.tensor(0.3)
        self.x = base.source_trajectory(self.source, self.noise, self.sigma)

    def test_transition_disjoint_from_lock(self) -> None:
        self.assertFalse(bool(torch.any(self.band * self.locked)))

    def test_transition_subset_of_editable(self) -> None:
        self.assertTrue(bool(torch.all(self.band <= self.editable)))

    def test_locked_derivative_and_state_match_hard_source_bit_exactly(self) -> None:
        hard, _, _ = base.hard_source_derivative(
            self.x, self.denoised, self.source, self.editable, self.sigma
        )
        guided, _, _ = transition.transition_derivative(
            self.x, self.denoised, self.source, self.editable, self.weights, self.sigma
        )
        self.assertTrue(torch.equal(hard * self.locked, guided * self.locked))
        hard_next = base.restore_after_euler(
            self.x + (self.sigma_next - self.sigma) * hard,
            self.source, self.noise, self.editable, self.sigma_next,
        )
        transition_next = base.restore_after_euler(
            self.x + (self.sigma_next - self.sigma) * guided,
            self.source, self.noise, self.editable, self.sigma_next,
        )
        self.assertTrue(torch.equal(hard_next * self.locked, transition_next * self.locked))

    def test_editable_interior_is_ordinary_euler_bit_exact(self) -> None:
        guided, model, _ = transition.transition_derivative(
            self.x, self.denoised, self.source, self.editable, self.weights, self.sigma
        )
        interior = self.editable - self.band
        self.assertTrue(torch.equal(guided * interior, model * interior))

    def test_weights_are_deterministic_and_monotonic(self) -> None:
        again = transition.linear_transition_weights(self.editable, 2)
        self.assertTrue(torch.equal(self.weights, again))
        self.assertEqual(self.weights[0, 0, 0, 4].item(), 1.0)
        self.assertEqual(self.weights[0, 0, 0, 5].item(), 0.0)
        self.assertGreaterEqual(self.weights[0, 0, 0, 4], self.weights[0, 0, 0, 5])

    def test_transition_is_not_source_restored_after_accept(self) -> None:
        proposal = torch.full_like(self.source, 17.0)
        restored = base.restore_after_euler(
            proposal, self.source, self.noise, self.editable, self.sigma_next
        )
        self.assertTrue(torch.equal(restored[self.band.bool().expand_as(restored)], proposal[self.band.bool().expand_as(proposal)]))

    def test_terminal_locked_latent_is_source(self) -> None:
        terminal = base.restore_after_euler(
            self.denoised, self.source, self.noise, self.editable, torch.tensor(0.0)
        )
        self.assertTrue(torch.equal(terminal * self.locked, self.source * self.locked))


if __name__ == "__main__":
    unittest.main()
