import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from flux2_recurrent_interleaved_blueprint import euler_step, feedback_state


class TestRecurrentInterleavedBlueprint(unittest.TestCase):
    def test_euler_endpoints(self):
        state = torch.tensor([[[[2.0]]]])
        prediction = torch.tensor([[[[0.5]]]])
        self.assertTrue(torch.equal(euler_step(state, prediction, 0.25, 0.0), prediction))

    def test_feedback_is_plain_bounded_interpolation(self):
        h = torch.arange(64.0).reshape(1, 1, 8, 8)
        result = feedback_state(h, (4, 4))
        self.assertEqual(tuple(result.shape), (1, 1, 4, 4))
        self.assertTrue(torch.isfinite(result).all())

    def test_invalid_interval_fails_closed(self):
        value = torch.zeros((1, 1, 2, 2))
        with self.assertRaises(ValueError):
            euler_step(value, value, 0.25, 0.25)
        with self.assertRaises(ValueError):
            euler_step(value, value[..., :1], 0.25, 0.0)
        with self.assertRaises(ValueError):
            feedback_state(value[0], (2, 2))


if __name__ == "__main__":
    unittest.main()
