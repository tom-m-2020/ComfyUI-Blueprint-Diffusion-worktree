from __future__ import annotations

import importlib.util
import sys
import unittest
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
    ConfigurableResamplingGeometry,
    ConfigurableResamplingProcedure,
    bounded_blueprint_transfer,
    configurable_region_noise,
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
    def __init__(self):
        self.calls = []

    def validate_prepared(self, **kwargs):
        pass

    def predict_native(self, *, value, expected_hw, **kwargs):
        self.calls.append(expected_hw)
        return value * 0.5


class TestConfigurableGeometry(unittest.TestCase):
    def test_registration_coexists(self):
        self.assertIn("BlueprintCandidate3EulerSampler", NODE_CLASS_MAPPINGS)
        self.assertIn("BlueprintTerminalResampling", NODE_CLASS_MAPPINGS)
        self.assertIn("BlueprintConfigurablePrototype", NODE_CLASS_MAPPINGS)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["BlueprintConfigurablePrototype"],
                         "Blueprint Configurable Prototype")

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


if __name__ == "__main__":
    unittest.main()
