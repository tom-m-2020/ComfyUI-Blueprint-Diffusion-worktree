from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


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

from blueprint_diffusion.adapters.base import RegionPredictionSet
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.regions import FixedCropPlanner
from blueprint_diffusion.sampling.euler import BlueprintCoordinator


class TestBulkRegionBoundary(unittest.TestCase):
    def test_nonterminal_is_ordered_ordinary_loop(self):
        adapter = Flux2Adapter()
        regions = FixedCropPlanner().plan((32, 64))
        h = torch.randn(1, 2, 32, 64)
        seen = []

        def predict(**kwargs):
            seen.append(kwargs["region"].index)
            return torch.full_like(kwargs["h_view"], kwargs["region"].index)

        adapter.predict_region = predict
        result = adapter.predict_regions(
            guider=object(), g=torch.empty(1, 2, 24, 48), h=h,
            sigma=torch.tensor(1.0), sigma_next=torch.tensor(0.5),
            canvas=(32, 64), regions=regions, model_options={}, seed=0,
        )
        self.assertEqual(seen, list(range(len(regions))))
        self.assertEqual(len(result.predictions), len(regions))
        self.assertFalse(result.telemetry["terminal_context_source_performed"])

    def test_specialized_failure_publishes_nothing(self):
        coordinator = BlueprintCoordinator()
        h = torch.randn(1, 2, 32, 64)
        state = coordinator.initialize(h, torch.tensor(1.0))
        before_h = state.h.clone()
        before_g = state.g.clone()
        coordinator.adapter.validate_run = lambda **kwargs: None
        coordinator.adapter.predict_global = lambda **kwargs: torch.zeros_like(kwargs["g"])
        coordinator.adapter.predict_regions = lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("injected specialized failure")
        )
        with self.assertRaisesRegex(RuntimeError, "injected specialized failure"):
            coordinator.evaluate(
                guider=object(), state=state, sigma=torch.tensor(1.0),
                sigma_next=torch.tensor(0.5), model_options={}, seed=0,
            )
        self.assertTrue(torch.equal(state.h, before_h))
        self.assertTrue(torch.equal(state.g, before_g))
        self.assertEqual(len(coordinator.telemetry), 1)

    def test_exact_terminal_geometry_dispatches_specialized(self):
        adapter = Flux2Adapter()
        regions = FixedCropPlanner().plan((128, 256))
        expected = RegionPredictionSet(tuple(), {"sentinel": True})
        with patch(
            "blueprint_diffusion.adapters.flux2_executor.Flux2BlockExecutor.predict_regions",
            return_value=expected,
        ) as specialized:
            result = adapter.predict_regions(
                guider=object(),
                g=torch.empty((1, 128, 96, 192), device="meta"),
                h=torch.empty((1, 128, 128, 256), device="meta"),
                sigma=torch.tensor(0.9), sigma_next=torch.tensor(0.0),
                canvas=(128, 256), regions=regions, model_options={}, seed=0,
            )
        self.assertIs(result, expected)
        specialized.assert_called_once()

    def test_other_geometry_never_dispatches_specialized(self):
        adapter = Flux2Adapter()
        adapter.predict_region = lambda **kwargs: torch.zeros_like(kwargs["h_view"])
        regions = FixedCropPlanner().plan((64, 128))
        with patch(
            "blueprint_diffusion.adapters.flux2_executor.Flux2BlockExecutor.predict_regions"
        ) as specialized:
            result = adapter.predict_regions(
                guider=object(), g=torch.empty(1, 2, 48, 96),
                h=torch.empty(1, 2, 64, 128), sigma=torch.tensor(0.9),
                sigma_next=torch.tensor(0.0), canvas=(64, 128),
                regions=regions, model_options={}, seed=0,
            )
        specialized.assert_not_called()
        self.assertEqual(len(result.predictions), len(regions))


if __name__ == "__main__":
    unittest.main()
