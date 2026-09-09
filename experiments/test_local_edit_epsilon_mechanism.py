import unittest
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from local_edit_epsilon_mechanism import editable_only_correction, masks


class EpsilonMechanismTests(unittest.TestCase):
    def test_editable_only_retains_exact_correction_only_in_edit(self):
        raw = torch.arange(64.0).reshape(1, 1, 1, 64)
        guided = raw + 7
        edit = masks(raw)["editable"]
        result = editable_only_correction(raw, guided, edit)
        self.assertTrue(torch.equal(result.masked_select(edit), guided.masked_select(edit)))
        self.assertTrue(torch.equal(result.masked_select(~edit), raw.masked_select(~edit)))

    def test_regions_use_center_as_source(self):
        value = torch.zeros((1, 1, 32, 64))
        region = masks(value)
        self.assertEqual(int(region["source"].sum()), 32 * 32)
        self.assertEqual(int(region["editable"].sum()), 32 * 32)
        self.assertFalse(bool((region["source"] & region["editable"]).any()))


if __name__ == "__main__":
    unittest.main()
