from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
SPEC = importlib.util.spec_from_file_location(
    "blueprint_diffusion",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
if "blueprint_diffusion" not in sys.modules:
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules["blueprint_diffusion"] = MODULE
    SPEC.loader.exec_module(MODULE)

from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.policies import HardNonterminalTerminalRelease
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler, validate_schedule
from blueprint_diffusion.sampling.euler import BlueprintCoordinator
from blueprint_diffusion.state import BlueprintState


class TestBlockDCTGeometry(unittest.TestCase):
    def setUp(self):
        self.geometry = BlockDCTGeometry()

    def test_right_inverse(self):
        value = torch.randn(1, 3, 24, 48, generator=torch.Generator().manual_seed(7))
        error = self.geometry.max_right_inverse_error(value)
        self.assertLessEqual(error, self.geometry.TOLERANCE)

    def test_constant_field_preservation(self):
        value = torch.full((1, 2, 32, 64), 3.25)
        reconstructed = self.geometry.prolong(self.geometry.restrict(value))
        self.assertLessEqual(float((reconstructed - value).abs().max()), 2e-6)


class TestRegions(unittest.TestCase):
    def test_coverage_and_normalization(self):
        regions = FixedCropPlanner().plan((32, 64))
        assembler = OverlapAssembler()
        predictions = [torch.ones(1, 2, 32, 32) for _ in regions]
        assembled, coverage = assembler.assemble(predictions, regions, (32, 64))
        self.assertGreater(float(coverage.min()), 0.0)
        self.assertTrue(torch.equal(assembled, torch.ones_like(assembled)))
        self.assertTrue(torch.allclose(coverage[:, :, :, :24], torch.ones_like(coverage[:, :, :, :24])))


class TestStateAndPolicy(unittest.TestCase):
    def test_state_is_frozen(self):
        state = BlueprintState(torch.zeros(1), torch.zeros(1), 1.0, 0, "initial")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.ordinal = 1

    def test_nonterminal_coupling_invariant(self):
        geometry = BlockDCTGeometry()
        policy = HardNonterminalTerminalRelease()
        h_star = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(1))
        g_star = torch.randn(1, 2, 24, 48, generator=torch.Generator().manual_seed(2))
        accepted = policy.accept(
            g_star=g_star, h_star=h_star, sigma_next=0.25, geometry=geometry
        )
        error = float((geometry.restrict(accepted.h) - accepted.g).abs().max())
        self.assertFalse(accepted.terminal_release)
        self.assertLessEqual(error, geometry.TOLERANCE)

    def test_terminal_release(self):
        geometry = BlockDCTGeometry()
        policy = HardNonterminalTerminalRelease()
        h_star = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(3))
        g_star = torch.randn(1, 2, 24, 48, generator=torch.Generator().manual_seed(4))
        accepted = policy.accept(
            g_star=g_star, h_star=h_star, sigma_next=0.0, geometry=geometry
        )
        self.assertTrue(accepted.terminal_release)
        self.assertIs(accepted.h, h_star)
        self.assertIs(accepted.g, g_star)

    def test_coordinator_detects_model_mutation(self):
        class MutatingAdapter:
            def validate_run(self, **kwargs):
                pass

            def predict_global(self, *, g, **kwargs):
                g.add_(1.0)
                return torch.zeros_like(g)

            def predict_region(self, *, h_view, **kwargs):
                return torch.zeros_like(h_view)

        coordinator = BlueprintCoordinator()
        coordinator.adapter = MutatingAdapter()
        h = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(9))
        state = coordinator.initialize(h, torch.tensor(1.0))
        with self.assertRaisesRegex(RuntimeError, "mutated accepted state"):
            coordinator.evaluate(
                guider=object(),
                state=state,
                sigma=torch.tensor(1.0),
                sigma_next=torch.tensor(0.5),
                model_options={},
                seed=1,
            )


class TestValidation(unittest.TestCase):
    def test_schedule_validation(self):
        validate_schedule(torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]))
        invalid = (
            torch.tensor([1.0, 0.5, 0.0]),
            torch.tensor([1.0, 0.75, 0.75, 0.25, 0.0]),
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.1]),
            torch.tensor([1.0, 0.75, float("nan"), 0.25, 0.0]),
        )
        for sigmas in invalid:
            with self.subTest(sigmas=sigmas):
                with self.assertRaises(ValueError):
                    validate_schedule(sigmas)

    def test_fail_closed_geometry_batch_and_model(self):
        geometry = BlockDCTGeometry()
        with self.assertRaises(ValueError):
            geometry.validate_high(torch.zeros(2, 128, 32, 64))
        with self.assertRaises(ValueError):
            geometry.validate_high(torch.zeros(1, 128, 16, 32))

        class Guider:
            cfg = 1.0
            inner_model = object()

        with self.assertRaises(ValueError):
            Flux2Adapter().validate_run(
                guider=Guider(),
                high_shape=(1, 128, 32, 64),
                global_shape=(1, 128, 24, 48),
                crops=FixedCropPlanner().plan((32, 64)),
                sigmas=torch.tensor([1.0, 0.0]),
                latent=torch.zeros(1, 128, 32, 64),
            )

    def test_fail_closed_mask_and_edit_latent(self):
        sampler = BlueprintEulerSampler()
        with self.assertRaises(ValueError):
            sampler.sample(
                None,
                torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
                {},
                None,
                torch.zeros(1, 128, 32, 64),
                torch.zeros(1, 128, 32, 64),
                torch.ones(1, 1, 32, 64),
            )
        with self.assertRaises(ValueError):
            sampler.sample(
                None,
                torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
                {},
                None,
                torch.zeros(1, 128, 32, 64),
                torch.ones(1, 128, 32, 64),
                None,
            )


if __name__ == "__main__":
    unittest.main()
