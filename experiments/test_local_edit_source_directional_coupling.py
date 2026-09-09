from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_source_directional_coupling as experiment


class DirectionalCouplingTests(unittest.TestCase):
    def setUp(self):
        self.editable = torch.tensor([True, False, False, True])
        self.source = ~self.editable

    def test_no_editable_to_source_changes_only_source_query_editable_key_edges(self):
        mask, record = experiment.directional_mask(
            6, 2, 6, self.editable, self.source,
            "NO_EDITABLE_TO_SOURCE", torch.float32, torch.device("cpu")
        )
        blocked = mask[0, 0] != 0
        expected = torch.zeros((6, 6), dtype=torch.bool)
        expected[torch.tensor([3, 4])[:, None], torch.tensor([2, 5])[None, :]] = True
        self.assertTrue(torch.equal(blocked, expected))
        self.assertEqual(record["blocked_edges"], 4)

    def test_no_source_to_editable_changes_only_editable_query_source_key_edges(self):
        mask, record = experiment.directional_mask(
            6, 2, 6, self.editable, self.source,
            "NO_SOURCE_TO_EDITABLE", torch.float32, torch.device("cpu")
        )
        blocked = mask[0, 0] != 0
        expected = torch.zeros((6, 6), dtype=torch.bool)
        expected[torch.tensor([2, 5])[:, None], torch.tensor([3, 4])[None, :]] = True
        self.assertTrue(torch.equal(blocked, expected))
        self.assertTrue(record["text_query_edges_unchanged"])
        self.assertTrue(record["text_key_edges_unchanged"])

    def test_partition_rejects_overlap(self):
        with self.assertRaises(AssertionError):
            experiment.directional_mask(
                6, 2, 6, self.editable, self.editable,
                "NO_SOURCE_TO_EDITABLE", torch.float32, torch.device("cpu")
            )


if __name__ == "__main__":
    unittest.main()
