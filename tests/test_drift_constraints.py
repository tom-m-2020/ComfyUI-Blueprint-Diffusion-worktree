from __future__ import annotations

import dataclasses
import importlib.util
import sys
import unittest
from itertools import pairwise
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]
SPEC = importlib.util.spec_from_file_location(
    "blueprint_diffusion",
    PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
if "blueprint_diffusion" not in sys.modules:
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules["blueprint_diffusion"] = MODULE
    SPEC.loader.exec_module(MODULE)

import comfy.model_sampling
import comfy.sample
import drift_fbsdiff_like_klein as phase4
import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1
from blueprint_diffusion.drift_constraints import (
    MODES,
    DriftConstrainedEuler,
    DriftConstraintNoise,
    FBSDiffConfig,
    FSSConfig,
    ILVRConfig,
    dct_2d,
    dct_matrix,
    fbsdiff_active,
    fbsdiff_substitute,
    idct_2d,
    ilvr_correct,
    lowpass,
    make_policy,
    structured_noise,
)
from blueprint_diffusion.nodes import NODE_CLASS_MAPPINGS, DriftConstrainedSampling


def policy(mode="none", source=None, seed=17, reference=None):
    if source is None:
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    latent = {"samples": source}
    return make_policy(
        latent, mode, seed, 2.0, 2.0, 4, 1.0, 5.0 / 63.0, 0.0, 0.5,
        reference,
    )


class FakeConst(comfy.model_sampling.CONST):
    def __init__(self):
        self.sigma_max = torch.tensor(1.0)
        self.noise_scale = 1.0


class FakeInner:
    def __init__(self):
        self.model_sampling = FakeConst()


class FakeGuider:
    def __init__(self, denoised_value=0.0):
        self.inner_model = FakeInner()
        self.denoised_value = denoised_value
        self.inputs = []

    def __call__(self, state, sigma, model_options, seed):
        del model_options, seed
        self.inputs.append((state.detach().clone(), sigma.detach().clone()))
        return torch.full_like(state, self.denoised_value)


class TestPureMechanisms(unittest.TestCase):
    def test_native_gaussian_endpoint_and_batch_index(self):
        source = torch.zeros(2, 4, 8, 8)
        latent = {"samples": source, "batch_index": [3, 3]}
        configured = make_policy(
            latent, "none", 31, 2, 2, 4, 1, 5 / 63, 0, 0.5, None
        )
        actual = DriftConstraintNoise(configured).generate_noise(latent)
        expected = comfy.sample.prepare_noise(source, 31, [3, 3])
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[0], actual[1]))

    def test_fss_matches_research_and_is_row_independent(self):
        source = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(2))
        gaussian = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(3))
        config = FSSConfig(2.0, 2.0)
        actual = structured_noise(source, gaussian, config)
        expected, _ = phase1.structured_noise(source, gaussian, 2.0)
        self.assertTrue(torch.equal(actual, expected))
        changed = source.clone(); changed[1].add_(20)
        other = structured_noise(changed, gaussian, config)
        self.assertTrue(torch.equal(actual[0], other[0]))

    def test_full_phase_preserves_gaussian_magnitude_and_rms(self):
        source = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(4))
        gaussian = torch.randn(1, 3, 16, 16, generator=torch.Generator().manual_seed(5))
        actual = structured_noise(source, gaussian, FSSConfig(100.0, 2.0))
        magnitude_error = (
            torch.fft.fft2(actual).abs() - torch.fft.fft2(gaussian).abs()
        ).square().mean().sqrt()
        self.assertLess(float(magnitude_error), 2e-5)
        self.assertAlmostEqual(float(actual.square().mean()), float(gaussian.square().mean()), places=5)

    def test_ilvr_matches_research_projection(self):
        proposal = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(6))
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(7))
        actual = ilvr_correct(proposal, source, ILVRConfig(4, 1.0))
        expected = proposal + (phase2.lowpass(source, 4) - phase2.lowpass(proposal, 4))
        self.assertTrue(torch.equal(actual, expected))
        self.assertLess(float((lowpass(actual, 4) - lowpass(source, 4)).abs().max()), 1e-6)

    def test_dct_roundtrip_mask_and_phase5_equivalence(self):
        target = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(8))
        reference = torch.randn(1, 3, 32, 32, generator=torch.Generator().manual_seed(9))
        matrix = dct_matrix(32, target.device)
        self.assertLess(float((idct_2d(dct_2d(target, matrix), matrix) - target).square().mean().sqrt()), 3e-6)
        config = FBSDiffConfig(5 / 63, 0, 0.5, [[None, {}]])
        actual = fbsdiff_substitute(target, reference, config)
        target_dct = phase4.dct_2d(target, matrix)
        reference_dct = phase4.dct_2d(reference, matrix)
        rows = torch.arange(32).unsqueeze(1); columns = torch.arange(32).unsqueeze(0)
        mask = ((rows + columns) / 31 > 5 / 63).to(target.dtype)
        expected = phase4.idct_2d(reference_dct * mask + target_dct * (1 - mask), matrix)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(int(mask.sum()), 1018)
        self.assertEqual([fbsdiff_active(config, i, 6) for i in range(6)], [True] * 3 + [False] * 3)


class TestPublicContract(unittest.TestCase):
    def test_nodes_and_exact_socket_types(self):
        self.assertIn("DriftConstrainedSampling", NODE_CLASS_MAPPINGS)
        self.assertIn("DriftConstrainedEulerSampler", NODE_CLASS_MAPPINGS)
        schema = DriftConstrainedSampling.INPUT_TYPES()
        self.assertEqual(schema["required"]["source"], ("LATENT",))
        self.assertEqual(schema["required"]["mode"], (MODES,))
        self.assertEqual(schema["optional"]["reference_conditioning"], ("CONDITIONING",))
        self.assertEqual(DriftConstrainedSampling.RETURN_TYPES, ("DRIFT_CONSTRAINT", "NOISE"))

    def test_policy_is_frozen_and_owns_source(self):
        source = torch.zeros(1, 4, 8, 8)
        configured = policy(source=source)
        source.add_(1)
        self.assertTrue(torch.equal(configured.source, torch.zeros_like(source)))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            configured.mode = "fss"

    def test_reference_conditioning_required_and_stateful_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            policy("fbsdiff")
        with self.assertRaisesRegex(ValueError, "stateful"):
            policy("fbsdiff", reference=[[None, {"control": object()}]])

    def test_paired_provenance_and_source_fingerprint_fail_closed(self):
        source = torch.zeros(1, 4, 8, 8)
        first = policy(source=source, seed=1)
        second = policy(source=source, seed=2)
        first_noise = DriftConstraintNoise(first).generate_noise({"samples": source})
        sampler = DriftConstrainedEuler(second)
        with self.assertRaisesRegex(ValueError, "Paired drift NOISE"):
            sampler.sample(FakeGuider(), torch.tensor([0.8, 0.0]), {"model_options": {}, "seed": 2}, None, first_noise, source)
        changed = source.clone(); changed[..., 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "differs"):
            DriftConstraintNoise(first).generate_noise({"samples": changed})

    def test_batch_mismatch_and_mask_rejected(self):
        source = torch.zeros(2, 4, 8, 8)
        configured = policy(source=source)
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        sampler = DriftConstrainedEuler(configured)
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            sampler.sample(FakeGuider(), torch.tensor([0.8, 0.0]), {"model_options": {}, "seed": 17}, None, endpoint[:1], source[:1])
        with self.assertRaisesRegex(ValueError, "noise masks"):
            sampler.sample(FakeGuider(), torch.tensor([0.8, 0.0]), {"model_options": {}, "seed": 17}, None, endpoint, source, torch.ones(2, 1, 8, 8))

    def test_batch_two_lifecycle_preserves_row_ownership(self):
        source = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(30))
        configured = policy("fss_ilvr", source, seed=31)
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        callbacks = []
        output = DriftConstrainedEuler(configured).sample(
            FakeGuider(), torch.tensor([0.8, 0.4, 0.0]),
            {"model_options": {}, "seed": 31},
            lambda *args: callbacks.append(args), endpoint, source,
        )
        self.assertEqual(tuple(output.shape), (2, 4, 8, 8))
        self.assertEqual(len(callbacks), 2)
        modified = source.clone()
        modified[1].add_(10)
        other = policy("fss_ilvr", modified, seed=31)
        other_endpoint = DriftConstraintNoise(other).generate_noise({"samples": modified})
        other_output = DriftConstrainedEuler(other).sample(
            FakeGuider(), torch.tensor([0.8, 0.4, 0.0]),
            {"model_options": {}, "seed": 31}, None, other_endpoint, modified,
        )
        self.assertTrue(torch.equal(output[0], other_output[0]))

    def test_swapped_same_shape_sampler_source_rejected(self):
        source = torch.zeros(1, 4, 8, 8)
        configured = policy(source=source)
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        swapped = source.clone()
        swapped[..., 0, 0] = 1
        with self.assertRaisesRegex(ValueError, "Sampler source differs"):
            DriftConstrainedEuler(configured).sample(
                FakeGuider(), torch.tensor([0.8, 0.0]),
                {"model_options": {}, "seed": 17}, None, endpoint, swapped,
            )


class TestConstEulerLifecycle(unittest.TestCase):
    def run_sampler(self, mode, source, sigmas, reference=None):
        configured = policy(mode, source, seed=23, reference=reference)
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        guider = FakeGuider(0.0)
        callbacks = []
        output = DriftConstrainedEuler(configured).sample(
            guider, sigmas, {"model_options": {}, "seed": 23},
            lambda *args: callbacks.append(args), endpoint, source,
        )
        return configured, endpoint, guider, callbacks, output

    def test_none_initialization_euler_and_research_parity(self):
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(10))
        sigmas = torch.tensor([0.8, 0.4, 0.0])
        _, endpoint, guider, callbacks, actual = self.run_sampler("none", source, sigmas)
        initial = 0.8 * endpoint + 0.2 * source
        self.assertTrue(torch.allclose(guider.inputs[0][0], initial))
        state = initial
        for sigma, sigma_next in pairwise(sigmas):
            state = state + state / sigma * (sigma_next - sigma)
        self.assertTrue(torch.allclose(actual, state))
        self.assertTrue(torch.equal(callbacks[-1][2], actual))

        research = phase1.EulerCaptureSampler(source)
        expected = research.sample(FakeGuider(), sigmas, {"model_options": {}, "seed": 23}, None, endpoint, source)
        self.assertTrue(torch.equal(actual, expected))

    def test_ilvr_after_proposal_no_terminal_correction_and_research_parity(self):
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(11))
        sigmas = torch.tensor([0.8, 0.4, 0.0])
        configured, endpoint, _, callbacks, actual = self.run_sampler("ilvr", source, sigmas)
        sampling = FakeConst()
        initial = sampling.noise_scaling(sigmas[0], endpoint, source, False)
        proposal = initial + initial / sigmas[0] * (sigmas[1] - sigmas[0])
        source_next = sampling.noise_scaling(sigmas[1], endpoint, source, False)
        corrected = ilvr_correct(proposal, source_next, configured.ilvr)
        self.assertTrue(torch.allclose(callbacks[0][2], corrected))
        terminal = corrected + corrected / sigmas[1] * (sigmas[2] - sigmas[1])
        self.assertTrue(torch.allclose(actual, terminal))

        research = phase2.EulerILVRSampler(source, endpoint, 4, 1.0, True)
        expected = research.sample(FakeGuider(), sigmas, {"model_options": {}, "seed": 23}, None, endpoint, source)
        self.assertTrue(torch.equal(actual, expected))

    def test_hybrid_uses_structured_endpoint_for_matching_source(self):
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(12))
        sigmas = torch.tensor([0.8, 0.4, 0.0])
        configured, endpoint, _, callbacks, _ = self.run_sampler("fss_ilvr", source, sigmas)
        initial = 0.8 * endpoint + 0.2 * source
        proposal = initial + initial / 0.8 * (0.4 - 0.8)
        matched = 0.4 * endpoint + 0.6 * source
        expected = ilvr_correct(proposal, matched, configured.ilvr)
        self.assertTrue(torch.allclose(callbacks[0][2], expected))

    def test_fbsdiff_precedes_target_and_reference_is_synchronized(self):
        source = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(13))
        sigmas = torch.tensor([0.8, 0.6, 0.4, 0.2, 0.0])
        configured = policy("fbsdiff", source, seed=29, reference=[[None, {}]])
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        guider = FakeGuider(0.0)
        sampler = DriftConstrainedEuler(configured)
        sampler._prepare_reference = mock.Mock(return_value="processed-reference")
        reference_calls = []

        def reference_predict(inner, conds, state, sigma, options):
            del inner, conds, options
            reference_calls.append((state.detach().clone(), sigma.detach().clone()))
            return [torch.ones_like(state)]

        with mock.patch("comfy.samplers.calc_cond_batch", side_effect=reference_predict):
            sampler.sample(guider, sigmas, {"model_options": {}, "seed": 29}, None, endpoint, source)
        initial = 0.8 * endpoint + 0.2 * source
        target_proposal = initial + initial / 0.8 * (0.6 - 0.8)
        reference_proposal = initial + (initial - 1) / 0.8 * (0.6 - 0.8)
        expected_second = fbsdiff_substitute(target_proposal, reference_proposal, configured.fbsdiff)
        self.assertTrue(torch.allclose(guider.inputs[1][0], expected_second, atol=2e-6))
        self.assertTrue(torch.allclose(
            torch.tensor([float(call[1][0]) for call in reference_calls]),
            torch.tensor([0.8, 0.6, 0.4, 0.2]),
        ))

    def test_invalid_schedule_and_non_const_rejected(self):
        source = torch.zeros(1, 4, 8, 8)
        configured = policy(source=source)
        endpoint = DriftConstraintNoise(configured).generate_noise({"samples": source})
        with self.assertRaisesRegex(ValueError, "strictly decreasing"):
            DriftConstrainedEuler(configured).sample(FakeGuider(), torch.tensor([0.8, 0.8, 0.0]), {"model_options": {}}, None, endpoint, source)
        guider = FakeGuider(); guider.inner_model.model_sampling = object()
        with self.assertRaisesRegex(TypeError, "CONST"):
            DriftConstrainedEuler(configured).sample(guider, torch.tensor([0.8, 0.0]), {"model_options": {}}, None, endpoint, source)


if __name__ == "__main__":
    unittest.main()
