import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from flux2_phase42_accepted_delta_exchange import (
    GEOMETRY,
    inverse_shift,
    map_global_delta,
    restrict_h_delta,
    scheduler_midpoint,
    shift,
    transfer_pair,
)


class TestPhase42AcceptedDeltaExchange(unittest.TestCase):
    def test_scheduler_midpoint_uses_inverse_shift_coordinate(self):
        midpoint = scheduler_midpoint(1.0, 0.75)
        expected = shift((inverse_shift(1.0) + inverse_shift(0.75)) * 0.5)
        self.assertEqual(midpoint, expected)
        self.assertGreater(1.0, midpoint)
        self.assertGreater(midpoint, 0.75)

    def test_A_maps_constant_delta_with_complete_coverage(self):
        delta = torch.full((1, 128, *GEOMETRY.blueprint_hw), 0.125)
        mapped, coverage, records = map_global_delta(delta)
        self.assertEqual(tuple(mapped.shape), (1, 128, *GEOMETRY.destination_hw))
        self.assertEqual(len(records), 49)
        self.assertGreater(float(coverage.min()), 0.0)
        error = mapped - 0.125
        self.assertLessEqual(float(error.square().mean().sqrt()), 1e-7)
        self.assertLessEqual(float(error.abs().max()), 1e-6)

    def test_A_and_R_accepted_delta_identities(self):
        generator = torch.Generator(device="cpu").manual_seed(42)
        h = torch.randn((1, 128, *GEOMETRY.destination_hw), generator=generator)
        g = torch.randn((1, 128, *GEOMETRY.blueprint_hw), generator=generator)
        g_mid = g + 0.01 * torch.randn(g.shape, generator=generator)
        first = transfer_pair(h, g, g_mid)
        h_next = first["h_mid"] + 0.01 * torch.randn(h.shape, generator=generator)
        completed = transfer_pair(h, g, g_mid, h_next)
        forward_error = completed["h_mid"] - h - completed["mapped_delta_g"]
        self.assertLessEqual(float(forward_error.square().mean().sqrt()), 1e-7)
        self.assertLessEqual(float(forward_error.abs().max()), 1e-6)
        restricted = restrict_h_delta(h_next - completed["h_mid"])
        actual = (completed["g_next"] - g_mid).float()
        error = actual - restricted
        self.assertLessEqual(float(error.square().mean().sqrt()), 1e-7)
        self.assertLessEqual(float(error.abs().max()), 1e-6)

    def test_invalid_midpoint_interval_fails_closed(self):
        for pair in ((0.25, 0.25), (0.25, 0.5), (1.1, 0.5), (0.5, -0.1)):
            with self.assertRaises(ValueError):
                scheduler_midpoint(*pair)


if __name__ == "__main__":
    unittest.main()
