from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_composite_robustness as experiment
import local_edit_postdecode_composite as policy


class CompositeRobustnessTests(unittest.TestCase):
    def test_three_required_case_classes(self) -> None:
        self.assertEqual(set(experiment.CASES), {"rigid_bridge", "organic_tree", "photometric_desert"})

    def test_policy_is_the_qualified_fixed_policy(self) -> None:
        self.assertEqual(policy.TRANSITION_WIDTH, 24)
        self.assertEqual((policy.LEFT, policy.RIGHT), (256, 768))

    def test_every_case_satisfies_pixel_invariants(self) -> None:
        weight = policy.source_weight()
        for config in experiment.CASES.values():
            source = policy.load_rgb(config["source"])
            generated = policy.load_rgb(config["generated"])
            result = policy.composite(source, generated, weight)
            self.assertTrue(np.array_equal(result[:, 280:744], source[:, 280:744]))
            self.assertTrue(np.array_equal(result[:, :256], generated[:, :256]))
            self.assertTrue(np.array_equal(result[:, 768:], generated[:, 768:]))

    def test_transition_never_expands_outside_source(self) -> None:
        weight = policy.source_weight()
        self.assertTrue(np.all(weight[:256] == 0))
        self.assertTrue(np.all(weight[768:] == 0))


if __name__ == "__main__":
    unittest.main()
