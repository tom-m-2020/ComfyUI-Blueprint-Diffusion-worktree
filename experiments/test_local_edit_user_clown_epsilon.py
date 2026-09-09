from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))
import local_edit_user_clown_epsilon_reproduction as experiment


class UploadedWorkflowSemanticsTests(unittest.TestCase):
    def test_workflow_geometry_and_mask_polarity(self) -> None:
        prepared = experiment.prepare_inputs()
        self.assertEqual(tuple(prepared["padded"].shape), (1, 1024, 2048, 3))
        self.assertEqual(tuple(prepared["resized"].shape), (1, 512, 1024, 3))
        mask = prepared["direct_mask"]
        self.assertTrue(torch.all(mask[..., :256] == 1))
        self.assertTrue(torch.all(mask[..., 256:768] == 0))
        self.assertTrue(torch.all(mask[..., 768:] == 1))
        internal = 1 - mask
        self.assertTrue(torch.all(internal[..., 256:768] == 1))

    def test_blurred_sampling_mask_is_distinct_and_soft(self) -> None:
        prepared = experiment.prepare_inputs()
        blurred = prepared["blurred_mask"]
        self.assertGreater(int(torch.unique(blurred).numel()), 2)
        self.assertFalse(torch.equal(blurred, prepared["direct_mask"]))
        self.assertEqual(float(blurred[..., 0].min()), 1.0)
        self.assertEqual(float(blurred[..., 512].max()), 0.0)

    def test_clown_node_constructs_nonchannelwise_projection_and_inverts_mask(self) -> None:
        guide = {"samples": torch.zeros((1, 128, 32, 64))}
        mask = torch.ones((1, 512, 1024))
        config = experiment.make_guides(guide, mask, True)
        self.assertEqual(config["guide_mode"], "epsilon_projection")
        self.assertTrue(torch.equal(config["mask"], torch.zeros_like(mask)))
        self.assertEqual(float(config["weights_masked"][0]), 1.0)

    def test_projection_operator_is_batch_global_and_not_plain_source_pull(self) -> None:
        generator = torch.Generator().manual_seed(4)
        model = torch.randn((1, 3, 4, 5), generator=generator)
        guide = torch.randn((1, 3, 4, 5), generator=generator)
        mask = torch.zeros((1, 1, 4, 5))
        mask[..., 1:4] = 1
        q = mask * guide
        projected = experiment.get_collinear(model, q) + experiment.get_orthogonal(q, model)
        plain = model + mask * (guide - model)
        self.assertFalse(torch.allclose(projected, plain))
        changed_q = q.clone()
        changed_q[..., 0, 0] += 10
        projected_changed = (
            experiment.get_collinear(model, changed_q)
            + experiment.get_orthogonal(changed_q, model)
        )
        self.assertFalse(torch.equal(projected[..., 3, 4], projected_changed[..., 3, 4]))


if __name__ == "__main__":
    unittest.main()
