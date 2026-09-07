from __future__ import annotations

import unittest

import torch

from experiments.flux2_phase46_epsilon_projection_cw import (
    WEIGHT,
    epsilon_projection_cw,
    rowwise_collinear,
)


class TestPhase46ProjectionCW(unittest.TestCase):
    def test_rowwise_projection_decomposition(self):
        x = torch.randn((4, 5), generator=torch.Generator().manual_seed(1))
        y = torch.randn((4, 5), generator=torch.Generator().manual_seed(2))
        col = rowwise_collinear(x, y)
        residual = x - col
        self.assertTrue(torch.allclose((residual * y).sum(dim=1), torch.zeros(4), atol=1e-6))

    def test_operator_is_finite_and_records_channel_ratios(self):
        shape = (1, 4, 5, 6)
        raw = torch.randn(shape, generator=torch.Generator().manual_seed(3))
        guide_d = torch.randn(shape, generator=torch.Generator().manual_seed(4))
        model = torch.randn(shape, generator=torch.Generator().manual_seed(5))
        guide = torch.randn(shape, generator=torch.Generator().manual_seed(6))
        projected, record = epsilon_projection_cw(raw, guide_d, model, guide)
        self.assertTrue(torch.isfinite(projected).all())
        self.assertEqual(len(record["effective_weight_ratios"]), 4)
        self.assertAlmostEqual(sum(record["effective_weight_ratios"]), 4.0, places=5)
        self.assertNotEqual(record["raw_derivative_hash"], record["projected_derivative_hash"])

    def test_fixed_contract_fails_closed(self):
        value = torch.zeros((1, 2, 3, 4))
        with self.assertRaises(ValueError):
            epsilon_projection_cw(value, value, value, value, weight=WEIGHT + 0.1)
        with self.assertRaises(ValueError):
            epsilon_projection_cw(value, value[:, :, :, :3], value, value)


if __name__ == "__main__":
    unittest.main()
