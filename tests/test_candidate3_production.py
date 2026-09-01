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
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler, Region
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler, validate_schedule
from blueprint_diffusion.sampling.euler import BlueprintCoordinator
from blueprint_diffusion.state import BlueprintState


class TestBlockDCTGeometry(unittest.TestCase):
    def setUp(self):
        self.geometry = BlockDCTGeometry((32, 64))

    def test_right_inverse(self):
        value = torch.randn(1, 3, 24, 48, generator=torch.Generator().manual_seed(7))
        error = self.geometry.max_right_inverse_error(value)
        self.assertLessEqual(error, self.geometry.TOLERANCE)

    def test_constant_field_preservation(self):
        value = torch.full((1, 2, 32, 64), 3.25)
        reconstructed = self.geometry.prolong(self.geometry.restrict(value))
        self.assertLessEqual(float((reconstructed - value).abs().max()), 2e-6)

    def test_arbitrary_compatible_geometries(self):
        generator = torch.Generator().manual_seed(8)
        for high_hw in ((32, 32), (64, 32), (32, 80), (36, 68)):
            with self.subTest(high_hw=high_hw):
                geometry = BlockDCTGeometry(high_hw)
                expected_global = tuple(value // 4 * 3 for value in high_hw)
                self.assertEqual(geometry.GLOBAL_HW, expected_global)
                global_value = torch.randn(1, 2, *expected_global, generator=generator)
                self.assertLessEqual(
                    geometry.max_right_inverse_error(global_value),
                    geometry.TOLERANCE,
                )
                high_value = torch.full((1, 2, *high_hw), 2.5)
                reconstructed = geometry.prolong(geometry.restrict(high_value))
                self.assertLessEqual(
                    float((reconstructed - high_value).abs().max()), 2e-6
                )

    def test_incompatible_geometry_fails(self):
        for high_hw in ((31, 64), (32, 66), (0, 32)):
            with self.subTest(high_hw=high_hw), self.assertRaises(ValueError):
                BlockDCTGeometry(high_hw)


class TestRegions(unittest.TestCase):
    def test_coverage_and_normalization(self):
        regions = FixedCropPlanner().plan((32, 64))
        assembler = OverlapAssembler()
        predictions = [torch.ones(1, 2, 32, 32) for _ in regions]
        assembled, coverage = assembler.assemble(predictions, regions, (32, 64))
        self.assertGreater(float(coverage.min()), 0.0)
        self.assertTrue(torch.equal(assembled, torch.ones_like(assembled)))
        self.assertTrue(torch.allclose(coverage[:, :, :, :24], torch.ones_like(coverage[:, :, :, :24])))

    def test_portrait_square_and_wide_coverage(self):
        planner = FixedCropPlanner()
        assembler = OverlapAssembler()
        expected = {
            (64, 32): ((0, 0), (24, 0), (32, 0)),
            (32, 32): ((0, 0),),
            (32, 80): ((0, 0), (0, 24), (0, 48)),
            (64, 80): tuple(
                (y, x) for y in (0, 24, 32) for x in (0, 24, 48)
            ),
        }
        for target_hw, positions in expected.items():
            with self.subTest(target_hw=target_hw):
                regions = planner.plan(target_hw)
                self.assertEqual(tuple((r.y, r.x) for r in regions), positions)
                predictions = [torch.ones(1, 2, 32, 32) for _ in regions]
                assembled, coverage = assembler.assemble(
                    predictions, regions, target_hw
                )
                self.assertGreater(float(coverage.min()), 0.0)
                self.assertTrue(torch.equal(assembled, torch.ones_like(assembled)))

    def test_exact_qualified_geometry_window_selection(self):
        expected = {
            (64, 128): (64, 48, ((0, 0), (0, 48), (0, 64))),
            (48, 96): (48, 36, ((0, 0), (0, 36), (0, 48))),
        }
        planner = FixedCropPlanner()
        assembler = OverlapAssembler()
        for target_hw, (crop_size, stride, positions) in expected.items():
            with self.subTest(target_hw=target_hw):
                regions = planner.plan(target_hw)
                self.assertEqual(tuple((r.y, r.x) for r in regions), positions)
                self.assertTrue(
                    all((r.height, r.width) == (crop_size, crop_size) for r in regions)
                )
                self.assertEqual(positions[1][1] - positions[0][1], stride)
                predictions = [
                    torch.ones(1, 2, crop_size, crop_size) for _ in regions
                ]
                assembled, coverage = assembler.assemble(
                    predictions, regions, target_hw
                )
                self.assertGreater(float(coverage.min()), 0.0)
                self.assertTrue(torch.equal(assembled, torch.ones_like(assembled)))

    def test_unlisted_geometry_retains_32_by_32_policy(self):
        regions = FixedCropPlanner().plan((64, 80))
        self.assertEqual(len(regions), 9)
        self.assertTrue(all((r.height, r.width) == (32, 32) for r in regions))
        self.assertEqual(
            tuple(sorted({r.x for r in regions})), (0, 24, 48)
        )

    def test_small_or_indivisible_target_fails(self):
        for target_hw in ((28, 32), (32, 28), (33, 64)):
            with self.subTest(target_hw=target_hw), self.assertRaises(ValueError):
                FixedCropPlanner().plan(target_hw)


class TestFluxCoordinates(unittest.TestCase):
    def test_global_endpoint_scale_and_crop_offsets(self):
        calls = []

        def guider(value, sigma, *, model_options, seed):
            calls.append(model_options["transformer_options"]["rope_options"])
            return torch.zeros_like(value)

        adapter = Flux2Adapter()
        base_options = {"transformer_options": {"sentinel": True}}
        g = torch.zeros(1, 2, 24, 60)
        adapter.predict_global(
            guider=guider,
            g=g,
            sigma=torch.tensor(1.0),
            canvas=(32, 80),
            model_options=base_options,
            seed=1,
        )
        region = Region(0, 0, 48)
        adapter.predict_region(
            guider=guider,
            h_view=torch.zeros(1, 2, 32, 32),
            sigma=torch.tensor(1.0),
            canvas=(32, 80),
            region=region,
            model_options=base_options,
            seed=1,
        )
        self.assertEqual(
            calls[0],
            {"scale_y": 31.0 / 23.0, "scale_x": 79.0 / 59.0},
        )
        self.assertEqual(calls[1], {"shift_y": 0.0, "shift_x": 48.0})
        self.assertEqual(base_options, {"transformer_options": {"sentinel": True}})


class TestStateAndPolicy(unittest.TestCase):
    def test_state_is_frozen(self):
        state = BlueprintState(torch.zeros(1), torch.zeros(1), 1.0, 0, "initial")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.ordinal = 1

    def test_nonterminal_coupling_invariant(self):
        geometry = BlockDCTGeometry((32, 64))
        policy = HardNonterminalTerminalRelease()
        h_star = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(1))
        g_star = torch.randn(1, 2, 24, 48, generator=torch.Generator().manual_seed(2))
        accepted = policy.accept(
            g_star=g_star, h_star=h_star, sigma_next=0.25, geometry=geometry
        )
        error = float((geometry.restrict(accepted.h) - accepted.g).abs().max())
        self.assertFalse(accepted.terminal_release)
        self.assertTrue(accepted.global_synchronized)
        self.assertLessEqual(error, geometry.TOLERANCE)

    def test_terminal_release(self):
        policy = HardNonterminalTerminalRelease()
        h_star = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(3))
        retained_g = torch.randn(1, 2, 24, 48, generator=torch.Generator().manual_seed(4))
        accepted = policy.accept_terminal(
            retained_g=retained_g, h_star=h_star, sigma_next=0.0
        )
        self.assertTrue(accepted.terminal_release)
        self.assertIs(accepted.h, h_star)
        self.assertIs(accepted.g, retained_g)
        self.assertIsNone(accepted.projection_rms)
        self.assertFalse(accepted.global_synchronized)

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

    def test_terminal_coordinator_skips_global_and_retains_diagnostic_g(self):
        class TerminalAdapter:
            def __init__(self):
                self.global_calls = 0
                self.local_calls = 0

            def validate_run(self, **kwargs):
                pass

            def predict_global(self, **kwargs):
                self.global_calls += 1
                raise AssertionError("Terminal interval must not predict global state.")

            def predict_region(self, *, h_view, **kwargs):
                self.local_calls += 1
                return torch.zeros_like(h_view)

            def describe_work(self, *, global_shape, crops):
                from blueprint_diffusion.adapters.base import WorkEstimate
                return WorkEstimate(
                    global_tokens=global_shape[-2] * global_shape[-1],
                    local_tokens=sum(region.height * region.width for region in crops),
                    model_predictions=1 + len(crops),
                )

        coordinator = BlueprintCoordinator()
        adapter = TerminalAdapter()
        coordinator.adapter = adapter
        h = torch.randn(1, 2, 32, 64, generator=torch.Generator().manual_seed(10))
        initialized = coordinator.initialize(h, torch.tensor(1.0))
        state = BlueprintState(initialized.g, initialized.h, 0.25, 3, "accepted:2")
        next_state, _ = coordinator.evaluate(
            guider=object(),
            state=state,
            sigma=torch.tensor(0.25),
            sigma_next=torch.tensor(0.0),
            model_options={},
            seed=1,
        )
        accepted = coordinator.telemetry[-1]
        self.assertEqual(adapter.global_calls, 0)
        self.assertEqual(adapter.local_calls, len(coordinator.planner.plan((32, 64))))
        self.assertIs(next_state.g, state.g)
        self.assertFalse(accepted["global_forward_performed"])
        self.assertTrue(accepted["terminal_global_unused"])
        self.assertFalse(accepted["global_synchronized"])
        self.assertEqual(accepted["global_state_status"], "retained_preterminal_unsynchronized")
        self.assertIsNone(accepted["projection_rms"])


class TestValidation(unittest.TestCase):
    def test_schedule_validation(self):
        for steps in (1, 2, 4, 8, 20):
            with self.subTest(steps=steps):
                validate_schedule(torch.linspace(1.0, 0.0, steps + 1))
        invalid = (
            torch.tensor([]),
            torch.tensor([1.0]),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([1.0, 0.75, 0.75, 0.25, 0.0]),
            torch.tensor([1.0, 0.75, 0.5, 0.25, 0.1]),
            torch.tensor([1.0, 0.75, float("nan"), 0.25, 0.0]),
            torch.tensor([0.9, 0.7, 0.5, 0.25, 0.0]),
            torch.tensor([1.0, 0.5, 0.0, -0.1]),
        )
        for sigmas in invalid:
            with self.subTest(sigmas=sigmas):
                with self.assertRaises(ValueError):
                    validate_schedule(sigmas)

    def test_fail_closed_geometry_batch_and_model(self):
        geometry = BlockDCTGeometry((32, 64))
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
        sampler.last_telemetry = ({"stale": True},)
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
        self.assertEqual(sampler.last_telemetry, ())
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
