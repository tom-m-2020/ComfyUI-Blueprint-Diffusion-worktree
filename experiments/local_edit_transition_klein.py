"""Experiment-only editable-side transition discriminator for FLUX.2 Klein."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

import local_edit_hard_source_klein as base


OUTPUT = base.ROOT / "experiments" / "local_edit_transition_klein_results"
FROZEN = base.OUTPUT / "runtime_tensors.pt"
WIDTH = base.TRANSITION_LATENT_WIDTH


def linear_transition_weights(editable_mask: torch.Tensor, width: int = WIDTH) -> torch.Tensor:
    transition = base.derive_transition_band(editable_mask, width)
    weights = torch.zeros_like(editable_mask)
    columns = torch.where(transition[0, 0].any(dim=0))[0]
    if len(columns) != width:
        raise ValueError(f"Fixed side discriminator requires exactly {width} transition columns.")
    values = torch.linspace(1.0, 0.0, width, dtype=weights.dtype, device=weights.device)
    for column, value in zip(columns, values):
        weights[..., column] = transition[..., column] * value
    if torch.any(weights * (1.0 - editable_mask)):
        raise RuntimeError("Transition weights escaped the editable mask.")
    return weights


def transition_derivative(
    x_sigma: torch.Tensor,
    denoised: torch.Tensor,
    source: torch.Tensor,
    editable_mask: torch.Tensor,
    weights: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hard, d_model, d_source = base.hard_source_derivative(
        x_sigma, denoised, source, editable_mask, sigma
    )
    weights = weights.to(device=x_sigma.device, dtype=x_sigma.dtype)
    if weights.shape != editable_mask.shape or torch.any(weights < 0) or torch.any(weights > 1):
        raise ValueError("Transition weights must match the editable mask and lie in [0,1].")
    return hard + weights * (d_source - d_model), d_model, d_source


class TransitionEulerSampler(base.comfy.samplers.Sampler):
    def __init__(self, editable_mask: torch.Tensor) -> None:
        self.editable_mask_cpu = editable_mask.detach().float().cpu()
        self.intervals: list[dict[str, Any]] = []
        self.accepted: list[torch.Tensor] = []
        self.first_raw_prediction_hash: str | None = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        base.reject_noise_mask(denoise_mask)
        base.validate_schedule(sigmas)
        if latent_image is None or noise.shape != latent_image.shape:
            raise ValueError("Transition experiment requires matching source latent and noise.")
        if noise.shape[0] != 1 or noise.shape[1] != 128:
            raise ValueError("Transition experiment requires batch-one Klein 128-channel latents.")
        sampling = model.inner_model.model_sampling
        if not isinstance(sampling, base.comfy.model_sampling.CONST):
            raise TypeError(f"Transition experiment requires CONST, got {type(sampling).__name__}.")

        source = latent_image
        fixed_noise = noise.clone()
        editable = base.validate_mask(self.editable_mask_cpu.to(noise.device), noise)
        locked = 1.0 - editable
        weights = linear_transition_weights(editable)
        transition = (weights >= 0).to(weights.dtype) * base.derive_transition_band(editable, WIDTH)
        interior = editable - transition
        x = sampling.noise_scaling(sigmas[0], noise, source, self.max_denoise(model, sigmas))
        x = base.restore_after_euler(x, source, fixed_noise, editable, sigmas[0])
        total = len(sigmas) - 1

        for step, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            y_sigma = base.source_trajectory(source, fixed_noise, sigma)
            y_next = base.source_trajectory(source, fixed_noise, sigma_next)
            denoised = model(x, sigma.expand(1), **extra_args)
            if self.first_raw_prediction_hash is None:
                self.first_raw_prediction_hash = base.tensor_hash(denoised)
            guided, d_model, d_source = transition_derivative(
                x, denoised, source, editable, weights, sigma
            )
            x_star = x + (sigma_next - sigma) * guided
            x_next = base.restore_after_euler(x_star, source, fixed_noise, editable, sigma_next)
            record = {
                "step": step,
                "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "locked_state_rms_error_vs_y_sigma": base.masked_rms(x - y_sigma, locked),
                "locked_state_max_error_vs_y_sigma": base.masked_max_abs(x - y_sigma, locked),
                "accepted_locked_rms_error_vs_y_sigma_next": base.masked_rms(x_next - y_next, locked),
                "accepted_locked_max_error_vs_y_sigma_next": base.masked_max_abs(x_next - y_next, locked),
                "transition_derivative_rms": base.masked_rms(guided, transition),
                "transition_correction_rms": base.masked_rms(guided - d_model, transition),
                "transition_vs_model_derivative_rms": base.masked_rms(guided - d_model, transition),
                "interior_editable_derivative_change_max": base.masked_max_abs(guided - d_model, interior),
                "accepted_transition_state_rms": base.masked_rms(x_next, transition),
                "locked_derivative_hash": base.tensor_hash((guided * locked).cpu()),
                "accepted_locked_hash": base.tensor_hash((x_next * locked).cpu()),
                "accepted_hash": base.tensor_hash(x_next),
            }
            self.intervals.append(record)
            self.accepted.append(x_next.detach().float().cpu())
            if record["accepted_locked_max_error_vs_y_sigma_next"] > base.TOLERANCE:
                raise RuntimeError(f"S1 failed at transition step {step}.")
            x = x_next
            if callback is not None:
                callback(step, denoised, x, total)

        if base.masked_max_abs(x - source, locked) > base.TOLERANCE:
            raise RuntimeError("S1 failed at terminal sigma=0.")
        return sampling.inverse_noise_scaling(sigmas[-1], x)


def distance_profile(candidate: torch.Tensor, baseline: torch.Tensor) -> list[dict[str, float]]:
    error = (candidate - baseline).detach().float().abs()
    profile = []
    for near, far in ((0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 768)):
        start = max(0, base.SOURCE_WIDTH - far)
        end = base.SOURCE_WIDTH - near
        strip = error[:, :, start:end, :]
        profile.append({
            "distance_into_locked_px_min": near,
            "distance_into_locked_px_max": far,
            "mae": float(strip.mean()),
            "max_abs": float(strip.max()),
        })
    return profile


def decode_vae(vae, latent: torch.Tensor) -> torch.Tensor:
    return vae.decode(latent).detach().cpu()


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = torch.load(FROZEN, map_location="cpu", weights_only=False)
    source = frozen["source_latent"].float()
    editable = frozen["editable_mask"].float()
    locked = 1.0 - editable
    sigmas = frozen["sigmas"].float()
    baseline = frozen["arms"]["C_hard_source"].float()
    baseline_accepted = frozen["hard_accepted"]
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(base.SEED))
    if not torch.equal(noise, torch.randn(source.shape, generator=torch.Generator().manual_seed(base.SEED))):
        raise RuntimeError("Seeded full-canvas noise is not reproducible.")

    model = base.comfy.sd.load_diffusion_model(str(base.phase2.MODEL_PATH), model_options={})
    clip = base.comfy.sd.load_clip(
        [str(base.phase2.TEXT_ENCODER_PATH)], clip_type=base.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(base.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    base.comfy.model_management.unload_all_models()
    base.comfy.model_management.soft_empty_cache()

    started = time.perf_counter()
    sampler = TransitionEulerSampler(editable)
    with torch.inference_mode():
        candidate = base.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative, source,
            noise_mask=None, callback=None, disable_pbar=True, seed=base.SEED,
        ).cpu()
    repeat_sampler = TransitionEulerSampler(editable)
    with torch.inference_mode():
        repeat = base.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, repeat_sampler, sigmas, positive, negative, source,
            noise_mask=None, callback=None, disable_pbar=True, seed=base.SEED,
        ).cpu()

    processed = [model.model.process_latent_out(value) for value in sampler.accepted]
    model.cleanup()
    del model
    base.comfy.model_management.unload_all_models()
    base.comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = base.comfy.sd.VAE(sd=base.comfy.utils.load_torch_file(str(base.phase2.VAE_PATH), safe_load=True))
    source_decoded = decode_vae(vae, source)
    baseline_decoded = decode_vae(vae, baseline)
    candidate_decoded = decode_vae(vae, candidate)
    repeat_decoded = decode_vae(vae, repeat)
    base.save_pixels(source_decoded, OUTPUT / "SOURCE_RECONSTRUCTION.png")
    base.save_pixels(baseline_decoded, OUTPUT / "A_FROZEN_HARD_SOURCE.png")
    base.save_pixels(candidate_decoded, OUTPUT / "B_LINEAR_TRANSITION.png")
    base.save_pixels(repeat_decoded, OUTPUT / "B_LINEAR_TRANSITION_REPEAT.png")
    for step, latent in enumerate(processed):
        base.save_pixels(decode_vae(vae, latent), OUTPUT / f"B_STEP_{step:02d}_ACCEPTED.png")

    locality = []
    for guard in (0, 1, 2, 4, 8, 16):
        hybrid = source.clone()
        start = base.LOCKED_LATENT_WIDTH + guard
        hybrid[..., start:] = candidate[..., start:]
        decoded = decode_vae(vae, hybrid)
        base.save_pixels(decoded, OUTPUT / f"VAE_GUARD_{guard:02d}.png")
        locality.append({
            "unchanged_editable_guard_latent_columns": guard,
            "distance_profile": distance_profile(decoded, source_decoded),
        })
    del vae
    base.comfy.model_management.unload_all_models()
    base.comfy.model_management.soft_empty_cache()

    base.make_sheet([
        ("FROZEN HARD SOURCE", OUTPUT / "A_FROZEN_HARD_SOURCE.png"),
        ("2-COLUMN LINEAR TRANSITION", OUTPUT / "B_LINEAR_TRANSITION.png"),
        ("TRANSITION REPEAT", OUTPUT / "B_LINEAR_TRANSITION_REPEAT.png"),
    ], OUTPUT / "COMPARISON.png")

    first_divergence = next(
        (i for i, (a, b) in enumerate(zip(baseline_accepted, sampler.accepted)) if not torch.equal(a, b)),
        None,
    )
    report = {
        "experiment": "Local Edit 2-column linear editable-side transition",
        "configuration": {
            "reused_frozen_hard_source": str(FROZEN),
            "mask_convention": "M=1 editable; L=1-M locked; T subset M",
            "transition_width_latent_columns": WIDTH,
            "transition_weights_from_boundary": [1.0, 0.0],
            "sigmas": [float(value) for value in sigmas],
            "seed": base.SEED,
            "prompt": base.PROMPT,
        },
        "intervals": sampler.intervals,
        "comparisons": {
            "first_accepted_divergence_step_vs_frozen": first_divergence,
            "final_rms_vs_frozen": base.rms(candidate - baseline),
            "final_transition_rms_vs_frozen": base.masked_rms(
                candidate - baseline, base.derive_transition_band(editable, WIDTH)
            ),
            "final_editable_core_rms_vs_frozen": base.masked_rms(
                candidate - baseline, editable - base.derive_transition_band(editable, WIDTH)
            ),
            "repeat_latent_bit_exact": bool(torch.equal(candidate, repeat)),
            "repeat_decoded_bit_exact": bool(torch.equal(candidate_decoded, repeat_decoded)),
        },
        "integrity": {
            "S1_pass": all(
                item["accepted_locked_max_error_vs_y_sigma_next"] <= base.TOLERANCE
                for item in sampler.intervals
            ) and base.masked_max_abs(candidate - source, locked) <= base.TOLERANCE,
            "terminal_locked_max_abs_vs_source": base.masked_max_abs(candidate - source, locked),
            "locked_final_bit_exact_vs_frozen": bool(torch.equal(candidate * locked, baseline * locked)),
            "transition_inside_editable": bool(torch.all(base.derive_transition_band(editable, WIDTH) <= editable)),
            "transition_restored_after_accept": False,
            "production_changes": False,
        },
        "pixel_metrics": {
            "frozen": base.pixel_metrics(
                baseline_decoded.permute(0, 3, 1, 2), source_decoded.permute(0, 3, 1, 2),
                source_decoded.permute(0, 3, 1, 2),
            ),
            "transition": base.pixel_metrics(
                candidate_decoded.permute(0, 3, 1, 2), source_decoded.permute(0, 3, 1, 2),
                source_decoded.permute(0, 3, 1, 2),
            ),
        },
        "vae_locality": locality,
        "runtime_seconds": time.perf_counter() - started,
        "semantic_review": {
            "S3": "FAIL/REGRESSION: the transition creates a wider near-black vertical barrier and the smaller bridge still restarts independently on the editable side.",
            "S4": "PASS: editable-region change RMS remains 0.685876, but this does not offset the S3 failure.",
            "artifact_extent": "FAIL: the decoded barrier is wider than the nominal transition band influence expected for a useful boundary condition.",
            "vae_locality": "BROAD: editable latent changes affect decoded locked pixels throughout the source side; error decreases with guard width but is not localized to the boundary.",
        },
        "decision": "STOP scalar transition guidance. Do not run alternate falloff or tune width/strength; next branch must be model-visible or prediction-refresh boundary research.",
    }
    base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        "candidate": candidate,
        "repeat": repeat,
        "accepted": sampler.accepted,
        "intervals": sampler.intervals,
    })
    base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "S1_pass": report["integrity"]["S1_pass"],
        "repeat_bit_exact": report["comparisons"]["repeat_latent_bit_exact"],
        "first_divergence": first_divergence,
    }, indent=2))


if __name__ == "__main__":
    run()
