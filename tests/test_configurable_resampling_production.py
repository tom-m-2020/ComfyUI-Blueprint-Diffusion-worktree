from __future__ import annotations

import importlib.util
import sys
import unittest
from unittest import mock
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
if "blueprint_diffusion" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "blueprint_diffusion", PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blueprint_diffusion"] = module
    spec.loader.exec_module(module)

from blueprint_diffusion.configurable_resampling import (
    AUTO_GEOMETRY_PROFILES,
    ConfigurableResamplingGeometry,
    ConfigurableResamplingProcedure,
    bounded_blueprint_transfer,
    configurable_region_noise,
    geometry_from_pixels,
    restrict_configurable_working,
    validate_refinement_sigma,
)
from blueprint_diffusion.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    TerminalResamplingProcedure,
)


class ConstSampling:
    noise_scale = 1.0

    @staticmethod
    def noise_scaling(sigma, noise, latent_image, max_denoise=False):
        del max_denoise
        while sigma.ndim < noise.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * latent_image


class FakeInner:
    model_sampling = ConstSampling()


class FakeGuider:
    inner_model = FakeInner()


class FakeAdapter:
    def __init__(self, fail_call=None):
        self.calls = []
        self.fail_call = fail_call

    def validate_prepared(self, **kwargs):
        pass

    def predict_native(self, *, value, expected_hw, **kwargs):
        self.calls.append(expected_hw)
        if len(self.calls) == self.fail_call:
            raise RuntimeError("injected configurable failure")
        return value * 0.5


class TestConfigurableGeometry(unittest.TestCase):
    def test_registration_coexists(self):
        self.assertIn("BlueprintCandidate3EulerSampler", NODE_CLASS_MAPPINGS)
        self.assertIn("BlueprintTerminalResampling", NODE_CLASS_MAPPINGS)
        self.assertIn("BlueprintConfigurablePrototype", NODE_CLASS_MAPPINGS)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["BlueprintConfigurablePrototype"],
                         "Blueprint Configurable Prototype")
        self.assertIn("BlueprintDiffusion", NODE_CLASS_MAPPINGS)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["BlueprintDiffusion"],
                         "Blueprint Diffusion (Terminal Refine)")
        self.assertEqual(tuple(NODE_CLASS_MAPPINGS["BlueprintConfigurablePrototype"].INPUT_TYPES()["required"]), (
            "guider", "sigmas", "noise_seed", "destination", "blueprint_width",
            "blueprint_height", "footprint_width", "footprint_height", "stride_x",
            "stride_y", "working_width", "working_height", "refinement_sigma",
        ))
        public = NODE_CLASS_MAPPINGS["BlueprintDiffusion"].INPUT_TYPES()["required"]
        self.assertEqual(public["geometry_mode"][0], ["auto", "manual"])
        self.assertIn("tooltip", public["tile_width"][1])
        self.assertIn("tooltip", public["working_width"][1])

    def test_auto_profiles_are_exact_and_completely_covered(self):
        expected = {
            (128, 256): ((32, 64), 55),
            (128, 128): ((45, 45), 49),
            (256, 256): ((48, 48), 121),
            (192, 128): ((48, 32), 40),
            (128, 192): ((32, 48), 40),
        }
        self.assertEqual(set(AUTO_GEOMETRY_PROFILES), set(expected))
        for destination_hw, (blueprint_hw, region_count) in expected.items():
            geometry = geometry_from_pixels(
                destination_hw, "auto", blueprint_width=720, blueprint_height=720,
                tile_width=512, tile_height=512, tile_overlap_x=256,
                tile_overlap_y=256, working_width=1024, working_height=1024,
            )
            with self.subTest(destination_hw=destination_hw):
                self.assertEqual(geometry.blueprint_hw, blueprint_hw)
                self.assertEqual(len(geometry.regions()), region_count)

    def test_manual_pixels_convert_without_rounding(self):
        geometry = geometry_from_pixels(
            (128, 128), "manual", blueprint_width=720, blueprint_height=720,
            tile_width=512, tile_height=512, tile_overlap_x=256,
            tile_overlap_y=256, working_width=1024, working_height=1024,
        )
        self.assertEqual(geometry, ConfigurableResamplingGeometry(
            (45, 45), (128, 128), (32, 32), (16, 16), (64, 64)
        ))

    def test_public_wrapper_resolves_auto_pixels_and_preserves_legacy_path(self):
        legacy = NODE_CLASS_MAPPINGS["BlueprintConfigurablePrototype"]
        public = NODE_CLASS_MAPPINGS["BlueprintDiffusion"]()
        output = {"blueprint_configurable_telemetry": {}}
        with mock.patch.object(legacy, "sample", return_value=(output, output.copy())) as delegated:
            result, _ = public.sample(
                None, None, 7, {"samples": torch.zeros((1, 128, 128, 128))}, "auto",
                720, 720, 512, 512, 256, 256, 1024, 1024, 0.25,
            )
        self.assertEqual(delegated.call_args.args[4:13],
                         (45, 45, 32, 32, 16, 16, 64, 64, 0.25))
        telemetry = result["blueprint_configurable_telemetry"]
        self.assertEqual(telemetry["user_geometry_mode"], "auto")
        self.assertEqual(telemetry["pixel_scale"], 16)
        self.assertEqual(telemetry["resolved_pixels"]["destination"],
                         {"width": 2048, "height": 2048})
        self.assertEqual(telemetry["resolved_pixels"]["overlap"], {"x": 256, "y": 256})

    def test_pixel_contract_errors_explain_failed_relation(self):
        common = dict(
            blueprint_width=720, blueprint_height=720, tile_width=512,
            tile_height=512, tile_overlap_x=256, tile_overlap_y=256,
            working_width=1024, working_height=1024,
        )
        with self.assertRaisesRegex(ValueError, "divisible by 16"):
            geometry_from_pixels((128, 128), "manual", **{**common, "tile_width": 513})
        with self.assertRaisesRegex(ValueError, "overlap.*smaller"):
            geometry_from_pixels((128, 128), "manual", **{**common, "tile_overlap_x": 512})
        with self.assertRaisesRegex(ValueError, "must fit inside destination"):
            geometry_from_pixels((16, 16), "manual", **common)
        with self.assertRaisesRegex(ValueError, "integer enlargement"):
            geometry_from_pixels((128, 128), "manual", **{**common, "tile_width": 768})
        with self.assertRaisesRegex(ValueError, "outside the qualified 16..64"):
            geometry_from_pixels((128, 128), "manual", **{**common, "blueprint_width": 1280})
        with self.assertRaisesRegex(ValueError, "no qualified profile"):
            geometry_from_pixels((96, 96), "auto", **common)
        with self.assertRaisesRegex(ValueError, "Geometry mode"):
            geometry_from_pixels((128, 128), "automatic", **common)

    def test_planner_coverage_invariant_fails_closed(self):
        geometry = ConfigurableResamplingGeometry((32, 32), (128, 128),
                                                   (32, 32), (24, 24), (64, 64))
        with mock.patch.object(ConfigurableResamplingGeometry, "_starts",
                               side_effect=((0, 40, 96), (0, 40, 96))):
            with self.assertRaisesRegex(ValueError, "cannot be completely covered"):
                geometry.regions()

    def test_multiple_geometries_and_overlap(self):
        cases = (
            (ConfigurableResamplingGeometry((45, 45), (128, 128), (32, 32), (16, 16), (64, 64)), 49),
            (ConfigurableResamplingGeometry((36, 54), (128, 192), (32, 32), (24, 20), (64, 64)), 45),
            (ConfigurableResamplingGeometry((48, 48), (256, 256), (32, 32), (24, 24), (64, 64)), 121),
        )
        for geometry, count in cases:
            with self.subTest(geometry=geometry):
                regions = geometry.regions()
                self.assertEqual(len(regions), count)
                coverage = torch.zeros(geometry.destination_hw)
                for region in regions:
                    coverage[region.y:region.y2, region.x:region.x2] += 1
                self.assertGreater(float(coverage.min()), 0.0)
                self.assertEqual(regions[0].index, 0)
                self.assertEqual(regions[-1].index, len(regions) - 1)

    def test_bounded_transfer_matches_full_map(self):
        geometry = ConfigurableResamplingGeometry((37, 61), (131, 227), (32, 32), (23, 19), (64, 64))
        value = torch.randn((1, 128, 37, 61), generator=torch.Generator().manual_seed(4), dtype=torch.float64)
        mapped = F.interpolate(value, geometry.destination_hw, mode="bilinear", align_corners=False)
        for region in geometry.regions():
            actual, source_hw = bounded_blueprint_transfer(value, geometry, region)
            expected = F.interpolate(mapped[..., region.y:region.y2, region.x:region.x2],
                                     geometry.working_hw, mode="nearest")
            self.assertLessEqual(float((actual - expected).abs().max()), 1e-12)
            self.assertLess(source_hw[0] * source_hw[1], geometry.blueprint_hw[0] * geometry.blueprint_hw[1])

    def test_lift_restrict_and_noise_are_deterministic(self):
        geometry = ConfigurableResamplingGeometry((32, 32), (96, 128), (16, 32), (12, 24), (32, 64))
        region = geometry.regions()[3]
        footprint = torch.randn((1, 128, *geometry.footprint_hw), generator=torch.Generator().manual_seed(2))
        working = F.interpolate(footprint, geometry.working_hw, mode="nearest")
        self.assertTrue(torch.equal(restrict_configurable_working(working, geometry), footprint))
        first = configurable_region_noise(9, region, geometry.working_hw, device="cpu", dtype=torch.float32)
        second = configurable_region_noise(9, region, geometry.working_hw, device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(first, second))

    def test_invalid_contracts_fail_closed(self):
        invalid = (
            ConfigurableResamplingGeometry((8, 32), (128, 128), (32, 32), (24, 24), (64, 64)),
            ConfigurableResamplingGeometry((32, 32), (128, 128), (32, 32), (40, 24), (64, 64)),
            ConfigurableResamplingGeometry((32, 32), (128, 128), (30, 32), (24, 24), (64, 64)),
            ConfigurableResamplingGeometry((64, 64), (1024, 128), (32, 32), (24, 24), (64, 64)),
        )
        for geometry in invalid:
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                geometry.validate()
        for sigma in (0.0, 0.09, 0.51, float("nan")):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                validate_refinement_sigma(sigma)


class TestConfigurableProcedure(unittest.TestCase):
    def test_exact_oracle_dispatch_is_bit_exact(self):
        geometry = ConfigurableResamplingGeometry((32, 64), (128, 256), (32, 32), (24, 24), (64, 64))
        args = (FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
                torch.zeros((1, 128, 128, 256)), torch.zeros((1, 128, 128, 256)), None)
        legacy = TerminalResamplingProcedure(seed=11, adapter=FakeAdapter())
        configurable = ConfigurableResamplingProcedure(seed=11, geometry=geometry,
                                                       refinement_sigma=0.25, adapter=FakeAdapter())
        expected = legacy.sample(*args)
        actual = configurable.sample(*args)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(configurable.telemetry["exact_terminal_oracle"])
        self.assertEqual(configurable.telemetry["execution"], "configurable_exact_oracle_dispatch")

    def test_new_geometry_executes_bounded_streaming_path(self):
        geometry = ConfigurableResamplingGeometry((32, 32), (96, 128), (32, 32), (24, 24), (64, 64))
        adapter = FakeAdapter()
        procedure = ConfigurableResamplingProcedure(seed=12, geometry=geometry,
                                                    refinement_sigma=0.20, adapter=adapter)
        destination = torch.zeros((1, 128, 96, 128))
        result = procedure.sample(
            FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
            torch.zeros_like(destination), destination, None,
        )
        self.assertEqual(tuple(result.shape), tuple(destination.shape))
        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(procedure.telemetry["blueprint_predictions"], 4)
        self.assertEqual(procedure.telemetry["local_predictions"], 20)
        self.assertEqual(procedure.telemetry["destination_model_predictions"], 0)
        self.assertGreater(procedure.telemetry["coverage_min"], 0.0)
        self.assertFalse(procedure.telemetry["exact_terminal_oracle"])
        self.assertEqual(adapter.calls.count((32, 32)), 4)
        self.assertEqual(adapter.calls.count((64, 64)), 20)

    def test_cancellation_then_retry_is_clean_and_deterministic(self):
        geometry = ConfigurableResamplingGeometry((32, 32), (96, 128),
                                                   (32, 32), (24, 24), (64, 64))
        destination = torch.zeros((1, 128, 96, 128))
        args = (FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
                torch.zeros_like(destination), destination, None)
        cancelled = ConfigurableResamplingProcedure(
            seed=15, geometry=geometry, refinement_sigma=0.25, adapter=FakeAdapter()
        )
        interrupts = 0

        def cancel():
            nonlocal interrupts
            interrupts += 1
            if interrupts == 6:
                raise RuntimeError("injected cancellation")

        with mock.patch("comfy.model_management.throw_exception_if_processing_interrupted",
                        side_effect=cancel), self.assertRaisesRegex(RuntimeError, "injected cancellation"):
            cancelled.sample(*args)
        self.assertEqual(cancelled.telemetry, {})

        first = ConfigurableResamplingProcedure(
            seed=15, geometry=geometry, refinement_sigma=0.25, adapter=FakeAdapter()
        )
        second = ConfigurableResamplingProcedure(
            seed=15, geometry=geometry, refinement_sigma=0.25, adapter=FakeAdapter()
        )
        self.assertTrue(torch.equal(first.sample(*args), second.sample(*args)))
        self.assertEqual(first.telemetry["final_H_hash"], second.telemetry["final_H_hash"])


if __name__ == "__main__":
    unittest.main()
