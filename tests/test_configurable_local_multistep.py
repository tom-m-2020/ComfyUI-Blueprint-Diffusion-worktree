import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "flux2_configurable_local_multistep.py"
SPEC = importlib.util.spec_from_file_location("configurable_local_multistep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TestConfigurableLocalMultistep(unittest.TestCase):
    def test_fixed_schedules_are_monotonic_and_share_endpoints(self):
        for intervals in (1, 2, 3):
            values = MODULE.local_schedule(intervals)
            self.assertEqual(len(values), intervals + 1)
            self.assertEqual(values[0], 0.25)
            self.assertEqual(values[-1], 0.0)
            self.assertTrue(all(left > right for left, right in zip(values[:-1], values[1:])))

    def test_schedule_count_fails_closed(self):
        for value in (0, -1, 1.0, True):
            with self.assertRaises(ValueError):
                MODULE.local_schedule(value)

    def test_terminal_update_is_prediction_and_intermediate_is_euler(self):
        working = torch.tensor([2.0])
        prediction = torch.tensor([1.0])
        sigma = torch.tensor(0.25)
        self.assertTrue(torch.equal(
            MODULE.accepted_local_update(working, prediction, sigma, 0.0), prediction
        ))
        expected = working + (0.125 - 0.25) * (working - prediction) / sigma
        self.assertTrue(torch.equal(
            MODULE.accepted_local_update(working, prediction, sigma, 0.125), expected
        ))


if __name__ == "__main__":
    unittest.main()
