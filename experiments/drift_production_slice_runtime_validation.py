"""Real stock-Klein migration parity for the five production drift modes."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
SOURCE = ROOT / "experiments" / "drift_phase_2_ilvr_results" / "portrait_bronze_SOURCE.png"
OUTPUT = ROOT / "experiments" / "drift_production_slice_validation"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

SPEC = importlib.util.spec_from_file_location(
    "blueprint_diffusion", PACKAGE_ROOT / "__init__.py",
    submodule_search_locations=[str(PACKAGE_ROOT)],
)
if "blueprint_diffusion" not in sys.modules:
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules["blueprint_diffusion"] = MODULE
    SPEC.loader.exec_module(MODULE)

import comfy.model_management
import comfy.model_sampling
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1
import drift_sampler_latent_fbsdiff_klein as phase5
import latent_preview
from blueprint_diffusion.nodes import (
    DriftConstrainedEulerSampler,
    DriftConstrainedSampling,
)
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_flux import get_schedule

PROMPT = (
    "Transform the same woman into an aged bronze museum statue with oxidized verdigris, "
    "dramatic gallery lighting, metallic surface texture, preserve pose, face geometry, "
    "framing, clothing silhouette, and background composition"
)


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def load_pixels(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return data.reshape(image.height, image.width, 3).float().div(255).unsqueeze(0)


def guider(model, positive, negative):
    value = comfy.samplers.CFGGuider(model)
    value.set_conds(positive, negative)
    value.set_cfg(1.0)
    return value


def research_output(mode, model, source, gaussian, fss, sigmas, positive, empty, control):
    if mode == "none":
        sampler, endpoint = phase1.EulerCaptureSampler(source), gaussian
    elif mode == "fss":
        sampler, endpoint = phase1.EulerCaptureSampler(source), fss
    elif mode == "ilvr":
        sampler, endpoint = phase2.EulerILVRSampler(source, gaussian, 4, 1.0, True), gaussian
    elif mode == "fss_ilvr":
        sampler, endpoint = phase2.EulerILVRSampler(source, fss, 4, 1.0, True), fss
    elif mode == "fbsdiff":
        reference_sampler = phase1.EulerCaptureSampler(source)
        with torch.inference_mode():
            comfy.sample.sample_custom(
                model, gaussian.clone(), 1.0, reference_sampler, sigmas.clone(), empty,
                empty, source.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=phase1.SEED,
            )
        references = [state for state, _ in reference_sampler.tensors]
        sampler = phase5.SamplerLatentHighFBS(
            source, references, 5 / 63, {0, 1, 2}, control
        )
        endpoint = gaussian
    else:
        raise AssertionError(mode)
    with torch.inference_mode():
        output = comfy.sample.sample_custom(
            model, endpoint.clone(), 1.0, sampler, sigmas.clone(), positive, empty,
            source.clone(), callback=lambda *unused: None,
            disable_pbar=True, seed=phase1.SEED,
        ).detach().float().cpu()
    return output, sampler.tensors


def production_output(mode, model, source, sigmas, positive, empty):
    policy, noise = DriftConstrainedSampling().build(
        {"samples": source.clone()}, mode, phase1.SEED, 2.0, 2.0, 4, 1.0,
        5 / 63, 0.0, 0.5, empty,
    )
    sampler = DriftConstrainedEulerSampler().build(policy)[0]
    captures = []

    def prepare_callback(*unused):
        return lambda ordinal, x0, state, total: captures.append(
            (ordinal, x0.detach().float().cpu(), state.detach().float().cpu(), total)
        )

    started = time.perf_counter()
    with (
        mock.patch.object(latent_preview, "prepare_callback", side_effect=prepare_callback),
        torch.inference_mode(),
    ):
        result = SamplerCustomAdvanced.execute(
            noise, guider(model, positive, empty), sampler, sigmas.clone(),
            {"samples": source.clone()},
        )[0]["samples"].detach().float().cpu()
    return result, captures, time.perf_counter() - started


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase1.VAE), safe_load=True))
    with torch.inference_mode():
        source = vae.encode(load_pixels(SOURCE)).detach().float().cpu()
    del vae
    model = comfy.sd.load_diffusion_model(str(phase1.MODEL), model_options={})
    if not isinstance(model.model.model_sampling, comfy.model_sampling.CONST):
        raise TypeError("Runtime validation requires CONST Klein")
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(phase1.STEPS, 32 * 32).float()[phase1.START_INDEX:].clone()
    gaussian = comfy.sample.prepare_noise(source, phase1.SEED)
    fss, _ = phase1.structured_noise(source, gaussian, 2.0)

    control_sampler = phase1.EulerCaptureSampler(source)
    with torch.inference_mode():
        comfy.sample.sample_custom(
            model, gaussian.clone(), 1.0, control_sampler, sigmas.clone(), positive,
            empty, source.clone(), callback=lambda *unused: None,
            disable_pbar=True, seed=phase1.SEED,
        )
    control = control_sampler.tensors
    report = {
        "status": "RUNNING", "source": str(SOURCE), "modes": {},
        "configuration": {"seed": phase1.SEED, "sigmas": sigmas.tolist(),
                          "sampler": "Euler", "cfg": 1.0, "batch": 1,
                          "native_entrypoint": "SamplerCustomAdvanced.execute"},
    }
    for mode in ("none", "fss", "ilvr", "fss_ilvr", "fbsdiff"):
        expected, expected_trajectory = research_output(
            mode, model, source, gaussian, fss, sigmas, positive, empty, control
        )
        actual, actual_trajectory, seconds = production_output(
            mode, model, source, sigmas, positive, empty
        )
        endpoint = fss if mode in {"fss", "fss_ilvr"} else gaussian
        expected_accepted = []
        for ordinal, (state, denoised) in enumerate(expected_trajectory):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            proposal = state + (state - denoised) / sigma * (sigma_next - sigma)
            if mode in {"ilvr", "fss_ilvr"} and float(sigma_next) > 0:
                source_next = model.model.model_sampling.noise_scaling(
                    sigma_next, endpoint, source, False
                )
                proposal = phase2.lowpass(source_next, 4) - phase2.lowpass(proposal, 4) + proposal
            expected_accepted.append(proposal)
        trajectory_errors = [
            {"ordinal": ordinal, "denoised_rms": rms(capture[1] - expected_trajectory[ordinal][1]),
             "accepted_state_rms": rms(capture[2] - expected_accepted[ordinal])}
            for ordinal, capture in enumerate(actual_trajectory)
        ]
        report["modes"][mode] = {
            "wall_seconds": seconds,
            "final_rms_error": rms(actual - expected),
            "max_trajectory_rms_error": max(
                max(item["denoised_rms"], item["accepted_state_rms"])
                for item in trajectory_errors
            ),
            "evaluations": len(actual_trajectory),
            "trajectory_errors": trajectory_errors,
            "finite": bool(torch.isfinite(actual).all()),
        }
        phase2.atomic_json(OUTPUT / "results.json", report)
    tolerance = 2e-5
    report["tolerance"] = tolerance
    report["status"] = "PASS" if all(
        item["final_rms_error"] <= tolerance and item["max_trajectory_rms_error"] <= tolerance
        and item["finite"] for item in report["modes"].values()
    ) else "FAIL"
    phase2.atomic_json(OUTPUT / "results.json", report)
    del model
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
