from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_step0_source_context as experiment


class StepZeroSourceContextTests(unittest.TestCase):
    def test_token_masks(self) -> None:
        editable, locked = experiment.generated_masks(torch.device("cpu"))
        self.assertEqual(int(editable.sum()), 1024)
        self.assertEqual(int(locked.sum()), 1024)
        self.assertFalse(bool((editable & locked).any()))

    def test_query_restoration_keeps_text_and_locked(self) -> None:
        probe = experiment.SourceContextProbe()
        key = ("double", 0)
        normal = torch.randn((1, experiment.TEXT_TOKENS + experiment.GENERATED_TOKENS, 8))
        changed = normal + 1
        probe.normal_attention[key] = normal
        result = probe.restore_noneditable_queries(
            changed, {"block_type": "double", "block_index": 0,
                      "img_slice": (experiment.TEXT_TOKENS,
                                    experiment.TEXT_TOKENS + experiment.GENERATED_TOKENS)}
        )
        editable, locked = experiment.generated_masks(torch.device("cpu"))
        self.assertTrue(torch.equal(result[:, :experiment.TEXT_TOKENS], normal[:, :experiment.TEXT_TOKENS]))
        self.assertTrue(torch.equal(result[:, experiment.TEXT_TOKENS:][:, locked],
                                    normal[:, experiment.TEXT_TOKENS:][:, locked]))
        self.assertTrue(torch.equal(result[:, experiment.TEXT_TOKENS:][:, editable],
                                    changed[:, experiment.TEXT_TOKENS:][:, editable]))

    def test_const_sigma_one_removes_source(self) -> None:
        source = torch.randn((1, 128, 2, 3))
        noise = torch.randn_like(source)
        sigma = torch.tensor(1.0)
        state = (1 - sigma) * source + sigma * noise
        self.assertTrue(torch.equal(state, noise))


if __name__ == "__main__":
    unittest.main()
