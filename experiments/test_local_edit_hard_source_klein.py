from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENTS))

import local_edit_hard_source_klein as local_edit


class HardSourceAlgebraTests(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        self.source = torch.randn((1, 4, 3, 5), generator=generator)
        self.noise = torch.randn((1, 4, 3, 5), generator=generator)
        self.denoised = torch.randn((1, 4, 3, 5), generator=generator)
        self.editable = torch.zeros((1, 1, 3, 5))
        self.editable[..., 3:] = 1.0
        self.locked = 1.0 - self.editable
        self.sigma = torch.tensor(0.75)
        self.sigma_next = torch.tensor(0.25)
        self.x = local_edit.source_trajectory(self.source, self.noise, self.sigma)

    def test_locked_derivative_is_exact_source_trajectory_derivative(self) -> None:
        guided, _, _ = local_edit.hard_source_derivative(
            self.x, self.denoised, self.source, self.editable, self.sigma
        )
        expected = self.noise - self.source
        self.assertLessEqual(float((self.locked * (guided - expected)).abs().max()), 1e-6)

    def test_euler_lands_on_next_source_trajectory(self) -> None:
        guided, _, _ = local_edit.hard_source_derivative(
            self.x, self.denoised, self.source, self.editable, self.sigma
        )
        proposal = self.x + (self.sigma_next - self.sigma) * guided
        expected = local_edit.source_trajectory(self.source, self.noise, self.sigma_next)
        self.assertLessEqual(float((self.locked * (proposal - expected)).abs().max()), 1e-6)

    def test_restoration_is_idempotent(self) -> None:
        first = local_edit.restore_after_euler(
            self.denoised, self.source, self.noise, self.editable, self.sigma_next
        )
        second = local_edit.restore_after_euler(
            first, self.source, self.noise, self.editable, self.sigma_next
        )
        self.assertTrue(torch.equal(first, second))

    def test_editable_coordinates_match_ordinary_euler_bit_exactly(self) -> None:
        guided, model, _ = local_edit.hard_source_derivative(
            self.x, self.denoised, self.source, self.editable, self.sigma
        )
        ordinary = self.x + (self.sigma_next - self.sigma) * model
        proposal = self.x + (self.sigma_next - self.sigma) * guided
        restored = local_edit.restore_after_euler(
            proposal, self.source, self.noise, self.editable, self.sigma_next
        )
        self.assertTrue(torch.equal(ordinary[..., 3:], restored[..., 3:]))

    def test_locked_coordinates_ignore_model_derivative(self) -> None:
        guided_a, _, _ = local_edit.hard_source_derivative(
            self.x, self.denoised, self.source, self.editable, self.sigma
        )
        guided_b, _, _ = local_edit.hard_source_derivative(
            self.x, self.denoised * 1000, self.source, self.editable, self.sigma
        )
        self.assertTrue(torch.equal(self.locked * guided_a, self.locked * guided_b))

    def test_mask_convention_one_means_editable(self) -> None:
        all_editable = torch.ones_like(self.editable)
        guided, model, _ = local_edit.hard_source_derivative(
            self.x, self.denoised, self.source, all_editable, self.sigma
        )
        self.assertTrue(torch.equal(guided, model))

    def test_noise_mask_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rejects incoming noise_mask"):
            local_edit.reject_noise_mask(torch.ones((1, 3, 5)))

    def test_fixed_noise_is_reused_across_trajectory(self) -> None:
        recovered = []
        for sigma in (torch.tensor(1.0), torch.tensor(0.8), torch.tensor(0.3)):
            state = local_edit.source_trajectory(self.source, self.noise, sigma)
            recovered.append(self.source + (state - self.source) / sigma)
        self.assertTrue(all(torch.allclose(item, self.noise) for item in recovered))

    def test_terminal_sigma_zero_is_clean_source(self) -> None:
        terminal = local_edit.source_trajectory(self.source, self.noise, torch.tensor(0.0))
        self.assertTrue(torch.equal(terminal, self.source))

    def test_transition_band_is_inside_editable_mask(self) -> None:
        transition = local_edit.derive_transition_band(self.editable, 1)
        self.assertTrue(torch.all(transition <= self.editable))
        self.assertFalse(bool(torch.any(transition * self.locked)))


if __name__ == "__main__":
    unittest.main()
