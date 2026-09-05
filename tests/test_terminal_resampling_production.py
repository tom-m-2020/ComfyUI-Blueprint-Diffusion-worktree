from __future__ import annotations

import dataclasses
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
SPEC = importlib.util.spec_from_file_location(
    "blueprint_diffusion",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
if "blueprint_diffusion" not in sys.modules:
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules["blueprint_diffusion"] = MODULE
    SPEC.loader.exec_module(MODULE)

from blueprint_diffusion.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from blueprint_diffusion.regions import OverlapAssembler
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    BlueprintRunState,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    TerminalResamplingProcedure,
    build_working_canvas,
    initialize_blueprint,
    lift_region,
    map_blueprint_to_destination,
    region_noise,
    region_seed,
    restrict_working_prediction,
    tensor_hash,
    validate_terminal_schedule,
)


class ConstSampling:
    noise_scale = 1.0

    @staticmethod
    def noise_scaling(sigma, noise, latent_image, max_denoise=False):
        del max_denoise
        while sigma.ndim < noise.ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma * noise + (1.0 - sigma) * latent_image


class TestTerminalResamplingPureFunctions(unittest.TestCase):
    def test_node_registration_coexists_with_legacy(self):
        self.assertIn("BlueprintCandidate3EulerSampler", NODE_CLASS_MAPPINGS)
        self.assertIn("BlueprintTerminalResampling", NODE_CLASS_MAPPINGS)
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["BlueprintTerminalResampling"],
            "Blueprint Terminal Resampling",
        )

    def test_exact_geometry_and_region_order(self):
        geometry = TerminalResamplingGeometry()
        regions = geometry.regions()
        self.assertEqual(len(regions), 55)
        self.assertEqual(tuple(sorted({r.y for r in regions})), (0, 24, 48, 72, 96))
        self.assertEqual(
            tuple(sorted({r.x for r in regions})),
            (0, 24, 48, 72, 96, 120, 144, 168, 192, 216, 224),
        )
        self.assertEqual((regions[0].index, regions[0].y, regions[0].x), (0, 0, 0))
        self.assertEqual((regions[-1].index, regions[-1].y, regions[-1].x), (54, 96, 224))

    def test_unqualified_geometry_rejected(self):
        with self.assertRaises(ValueError):
            TerminalResamplingGeometry(destination_hw=(64, 128)).validate()

    def test_region_seed_and_noise_are_order_independent(self):
        regions = TerminalResamplingGeometry().regions()
        expected = [region_seed(17, region) for region in regions]
        self.assertEqual(expected[0], 22_000_020)
        self.assertEqual(expected[-1], 22_054_506)
        forward = {r.index: tensor_hash(region_noise(17, r, device="cpu", dtype=torch.float32)) for r in regions}
        reverse = {r.index: tensor_hash(region_noise(17, r, device="cpu", dtype=torch.float32)) for r in reversed(regions)}
        self.assertEqual(forward, reverse)

    def test_blueprint_initialization_is_deterministic(self):
        first = initialize_blueprint(20260911)
        second = initialize_blueprint(20260911)
        other = initialize_blueprint(20260912)
        self.assertEqual(tuple(first.shape), (1, 128, 32, 64))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, other))

    def test_const_quarter_sigma_identity(self):
        anchor = torch.randn(1, 128, 64, 64, generator=torch.Generator().manual_seed(3))
        noise = torch.randn(1, 128, 64, 64, generator=torch.Generator().manual_seed(4))
        working, error = build_working_canvas(
            model_sampling=ConstSampling(), anchor=anchor, noise=noise
        )
        self.assertEqual(error, 0.0)
        self.assertTrue(torch.equal(working, 0.75 * anchor + 0.25 * noise))

    def test_mapping_lift_and_restriction(self):
        blueprint = torch.arange(32 * 64, dtype=torch.float32).reshape(1, 1, 32, 64).expand(1, 128, 32, 64)
        mapped = map_blueprint_to_destination(blueprint, TerminalResamplingGeometry())
        expected = F.interpolate(blueprint, (128, 256), mode="bilinear", align_corners=False)
        self.assertTrue(torch.equal(mapped, expected))
        crop = mapped[:, :, :32, :32]
        lifted = lift_region(crop)
        self.assertTrue(torch.equal(lifted, F.interpolate(crop, scale_factor=2, mode="nearest")))
        self.assertTrue(torch.equal(restrict_working_prediction(lifted), crop))

    def test_streaming_assembler_matches_list_assembler(self):
        geometry = TerminalResamplingGeometry()
        regions = geometry.regions()
        generator = torch.Generator().manual_seed(8)
        predictions = [torch.randn(1, 128, 32, 32, generator=generator) for _ in regions]
        expected, expected_coverage = OverlapAssembler().assemble(
            predictions, regions, geometry.destination_hw
        )
        stream = StreamingOverlapAssembler(
            regions=regions, target_hw=geometry.destination_hw, template=predictions[0]
        )
        for prediction, region in zip(predictions, regions):
            stream.add(prediction, region)
        actual, actual_coverage = stream.finish()
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual_coverage, expected_coverage))

    def test_run_state_is_frozen_and_has_no_h(self):
        state = BlueprintRunState(torch.zeros(1), 1.0, 0, "initial")
        self.assertFalse(hasattr(state, "h"))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.ordinal = 1

    def test_exact_schedule_only(self):
        validate_terminal_schedule(torch.tensor(QUALIFIED_SIGMAS))
        invalid = (
            torch.tensor([1.0, 0.0]),
            torch.tensor([1.0, 0.9, 0.8, 0.7, 0.0]),
            torch.tensor([*QUALIFIED_SIGMAS[:-1], float("nan")]),
        )
        for sigmas in invalid:
            with self.subTest(sigmas=sigmas), self.assertRaises(ValueError):
                validate_terminal_schedule(sigmas)

    def test_batch_rejected_by_procedure(self):
        procedure = TerminalResamplingProcedure(seed=1, adapter=object())
        with self.assertRaisesRegex(ValueError, "destination"):
            procedure.sample(
                object(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
                torch.zeros(2, 128, 128, 256), torch.zeros(2, 128, 128, 256), None,
            )


class FakeAdapter:
    def __init__(self, fail_call=None, nonfinite_call=None):
        self.calls = 0
        self.fail_call = fail_call
        self.nonfinite_call = nonfinite_call

    def validate_prepared(self, **kwargs):
        pass

    def predict_native(self, *, value, **kwargs):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("injected local failure")
        result = value * 0.5
        if self.calls == self.nonfinite_call:
            result = result.clone(); result.flatten()[0] = float("nan")
        return result


class FakeInner:
    model_sampling = ConstSampling()


class FakeGuider:
    inner_model = FakeInner()


class TestTerminalProcedureFailure(unittest.TestCase):
    def test_cancellation_clears_run_and_fresh_retry_is_deterministic(self):
        calls = 0

        def interrupt_after_first_blueprint():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected cancellation")

        cancelled = TerminalResamplingProcedure(seed=9, adapter=FakeAdapter())
        with mock.patch(
            "comfy.model_management.throw_exception_if_processing_interrupted",
            side_effect=interrupt_after_first_blueprint,
        ), self.assertRaisesRegex(RuntimeError, "injected cancellation"):
            cancelled.sample(
                FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
                torch.zeros(1, 128, 128, 256), torch.zeros(1, 128, 128, 256), None,
            )
        self.assertEqual(cancelled.telemetry, {})

        first = TerminalResamplingProcedure(seed=9, adapter=FakeAdapter())
        second = TerminalResamplingProcedure(seed=9, adapter=FakeAdapter())
        args = (
            FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, None,
            torch.zeros(1, 128, 128, 256), torch.zeros(1, 128, 128, 256), None,
        )
        self.assertTrue(torch.equal(first.sample(*args), second.sample(*args)))
        self.assertEqual(first.telemetry["final_H_hash"], second.telemetry["final_H_hash"])

    def test_injected_local_failure_publishes_no_telemetry_or_callback(self):
        adapter = FakeAdapter(fail_call=7)
        procedure = TerminalResamplingProcedure(seed=9, adapter=adapter)
        callbacks = []
        with self.assertRaisesRegex(RuntimeError, "injected local failure"):
            procedure.sample(
                FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, lambda *args: callbacks.append(args),
                torch.zeros(1, 128, 128, 256), torch.zeros(1, 128, 128, 256), None,
            )
        self.assertEqual(procedure.telemetry, {})
        self.assertEqual(len(callbacks), 4)

    def test_nonfinite_local_prediction_does_not_publish_final(self):
        adapter = FakeAdapter(nonfinite_call=5)
        procedure = TerminalResamplingProcedure(seed=9, adapter=adapter)
        callbacks = []
        with self.assertRaisesRegex(RuntimeError, "Invalid local prediction"):
            procedure.sample(
                FakeGuider(), torch.tensor(QUALIFIED_SIGMAS), {"model_options": {}}, lambda *args: callbacks.append(args),
                torch.zeros(1, 128, 128, 256), torch.zeros(1, 128, 128, 256), None,
            )
        self.assertEqual(procedure.telemetry, {})
        self.assertEqual(len(callbacks), 4)


if __name__ == "__main__":
    unittest.main()
