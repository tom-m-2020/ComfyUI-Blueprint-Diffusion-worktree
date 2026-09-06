import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from flux2_persistent_h_shared_provenance import (
    feedback_state,
    max_shared_overlap_error,
    shared_initial_states,
)
from blueprint_diffusion.regions import Region
from blueprint_diffusion.terminal_resampling import TerminalResamplingGeometry


class TestPersistentHSharedProvenance(unittest.TestCase):
    def test_initial_g_is_deterministically_derived_from_h(self):
        geometry = TerminalResamplingGeometry.for_destination((128, 128))
        h1, g1 = shared_initial_states(7, geometry, "cpu")
        h2, g2 = shared_initial_states(7, geometry, "cpu")
        self.assertTrue(torch.equal(h1, h2))
        self.assertTrue(torch.equal(g1, g2))
        self.assertEqual(tuple(g1.shape[-2:]), geometry.blueprint_hw)

    def test_overlap_provenance_detects_disagreement(self):
        regions = (Region(0, 0, 0, 2, 2), Region(1, 0, 1, 2, 2))
        canvas = torch.arange(6.0).reshape(1, 1, 2, 3)
        crops = [canvas[..., r.y:r.y2, r.x:r.x2] for r in regions]
        self.assertEqual(max_shared_overlap_error(crops, regions), 0.0)
        crops[1] = crops[1].clone(); crops[1][..., 0] += 1
        self.assertEqual(max_shared_overlap_error(crops, regions), 1.0)

    def test_feedback_fails_closed(self):
        with self.assertRaises(ValueError):
            feedback_state(torch.zeros((1, 2, 2)), (2, 2))


if __name__ == "__main__":
    unittest.main()
