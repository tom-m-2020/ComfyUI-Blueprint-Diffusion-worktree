from __future__ import annotations

import unittest

import torch

from experiments.flux2_persistent_coarse_guidance import guided_prediction


class TestPersistentCoarseGuidance(unittest.TestCase):
    def test_prediction_mix_matches_velocity_mix(self):
        state = torch.randn((1, 3, 4, 5), generator=torch.Generator().manual_seed(1))
        model_x0 = torch.randn(state.shape, generator=torch.Generator().manual_seed(2))
        guide = torch.randn(state.shape, generator=torch.Generator().manual_seed(3))
        sigma = 0.25
        weight = 1.0 / 3.0
        used_x0 = guided_prediction(model_x0, guide, weight)
        expected_velocity = (1.0 - weight) * (state - model_x0) / sigma + weight * (state - guide) / sigma
        self.assertTrue(torch.allclose((state - used_x0) / sigma, expected_velocity))

    def test_zero_and_one_are_explicit_endpoints(self):
        model_x0 = torch.tensor([1.0])
        guide = torch.tensor([3.0])
        self.assertTrue(torch.equal(guided_prediction(model_x0, guide, 0.0), model_x0))
        self.assertTrue(torch.equal(guided_prediction(model_x0, guide, 1.0), guide))

    def test_invalid_contract_fails_closed(self):
        with self.assertRaises(ValueError):
            guided_prediction(torch.zeros(1), torch.zeros(1), -0.1)
        with self.assertRaises(ValueError):
            guided_prediction(torch.zeros(1), torch.zeros(2), 0.5)


if __name__ == "__main__":
    unittest.main()

