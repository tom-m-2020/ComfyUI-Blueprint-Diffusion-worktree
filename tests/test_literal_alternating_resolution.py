import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "experiments" / "flux2_literal_alternating_resolution.py"
SPEC = importlib.util.spec_from_file_location("literal_alternating", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TestLiteralAlternatingResolution(unittest.TestCase):
    def test_interval_ownership_is_strictly_alternating(self):
        self.assertEqual([MODULE.interval_owner(i) for i in range(4)], ["G", "W", "G", "W"])

    def test_euler_accepts_the_post_update_state(self):
        state = torch.tensor([[[[2.0]]]])
        prediction = torch.tensor([[[[1.0]]]])
        actual = MODULE.euler_step(state, prediction, 1.0, 0.5)
        self.assertTrue(torch.equal(actual, torch.tensor([[[[1.5]]]])))

    def test_h_to_g_is_plain_bilinear_replacement(self):
        h = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        expected = torch.nn.functional.interpolate(h, size=(2, 2), mode="bilinear", align_corners=False)
        self.assertTrue(torch.equal(MODULE.downscale_h_to_g(h, (2, 2)), expected))


if __name__ == "__main__":
    unittest.main()
