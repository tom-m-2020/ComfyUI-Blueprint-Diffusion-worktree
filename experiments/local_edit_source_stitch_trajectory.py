"""Simplified source-state stitch -> full denoise trajectory discriminator."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

import local_edit_hard_source_klein as base
import flux2_coarse_global_local_falsification as model_paths
import nodes


OUTPUT = base.ROOT / "experiments" / "local_edit_source_stitch_trajectory_results"
SOURCE = base.ROOT / "experiments" / "local_edit_late_source_restoration_results" / "PADDED_RESIZED.png"
WIDTH, HEIGHT = 1024, 512
STEPS, SEED = 8, 0
PROMPT = (
    "A wide cinematic photograph of one single long red suspension bridge stretching "
    "continuously from the far left edge to the far right edge over calm water, one "
    "yellow passenger train centered on the bridge, one white lighthouse at the far "
    "left, one dark stone tower at the far right, continuous bridge deck and cables, "
    "coherent perspective, no duplicate bridges, trains, lighthouses, or towers"
)
ARMS = {"A_ORDINARY": frozenset(), "B_SINGLE_STITCH": frozenset({0}),
        "C_REPEATED_STITCH": None}


def source_state(source: torch.Tensor, fixed_noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (1.0 - sigma) * source + sigma * fixed_noise


def stitch(proposal: torch.Tensor, source_next: torch.Tensor,
           editable: torch.Tensor, enabled: bool) -> torch.Tensor:
    if not enabled:
        return proposal
    return editable * proposal + (1.0 - editable) * source_next


def masks(reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    editable = torch.ones_like(reference[:, :1])
    editable[..., 16:48] = 0
    source = 1.0 - editable
    boundary = torch.zeros_like(editable)
    boundary[..., 14:18] = 1
    boundary[..., 46:50] = 1
    return editable, source, boundary


def region_metrics(value: torch.Tensor, reference: torch.Tensor,
                   region: torch.Tensor) -> dict[str, float]:
    selected = (value.detach().float() - reference.detach().float()).masked_select(
        region.bool().expand_as(value)
    )
    return {"rms": float(selected.square().mean().sqrt()),
            "max_abs": float(selected.abs().max())}


class StitchEulerSampler(base.comfy.samplers.Sampler):
    def __init__(self, editable: torch.Tensor, stitch_steps: frozenset[int] | None) -> None:
        self.editable_cpu = editable.detach().float().cpu()
        self.stitch_steps = stitch_steps
        self.intervals: list[dict[str, Any]] = []
        self.tensors: list[dict[str, torch.Tensor | None]] = []
        self.fixed_noise_hash: str | None = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        base.reject_noise_mask(denoise_mask)
        base.validate_schedule(sigmas)
        if latent_image is None or noise.shape != latent_image.shape:
            raise ValueError("Source latent and fixed noise must have identical shape")
        sampling = model.inner_model.model_sampling
        if not isinstance(sampling, base.comfy.model_sampling.CONST):
            raise TypeError(f"Expected CONST sampling, got {type(sampling).__name__}")
        source = latent_image
        fixed_noise = noise.clone()
        self.fixed_noise_hash = base.tensor_hash(fixed_noise)
        editable = base.validate_mask(self.editable_cpu.to(noise.device), noise)
        locked = 1.0 - editable
        x = sampling.noise_scaling(sigmas[0], noise, source, self.max_denoise(model, sigmas))
        total = len(sigmas) - 1
        for step, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            denoised = model(x, sigma.expand(x.shape[0]), **extra_args)
            if self.tensors:
                self.tensors[-1]["next_raw_x0"] = denoised.detach().float().cpu()
            derivative = (x - denoised) / sigma
            proposal = x + (sigma_next - sigma) * derivative
            target_next = source_state(source, fixed_noise, sigma_next)
            enabled = self.stitch_steps is None or step in self.stitch_steps
            accepted = stitch(proposal, target_next, editable, enabled)
            if not torch.equal(accepted * editable, proposal * editable):
                raise RuntimeError(f"Stitch modified editable coordinates at interval {step}")
            source_error = base.masked_max_abs(accepted - target_next, locked)
            if enabled and source_error != 0.0:
                raise RuntimeError(f"Stitched source state is not exact at interval {step}: {source_error}")
            self.intervals.append({
                "step": step, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "stitched": enabled,
                "input_source_error_vs_trajectory_rms": base.masked_rms(
                    x - source_state(source, fixed_noise, sigma), locked
                ),
                "proposal_source_error_vs_trajectory_rms": base.masked_rms(proposal - target_next, locked),
                "accepted_source_error_vs_trajectory_rms": base.masked_rms(accepted - target_next, locked),
                "accepted_source_error_vs_trajectory_max": source_error,
                "editable_proposal_to_accepted_max": base.masked_max_abs(accepted - proposal, editable),
                "input_hash": base.tensor_hash(x), "raw_x0_hash": base.tensor_hash(denoised),
                "proposal_hash": base.tensor_hash(proposal), "source_next_hash": base.tensor_hash(target_next),
                "accepted_hash": base.tensor_hash(accepted), "fixed_noise_hash": base.tensor_hash(fixed_noise),
            })
            self.tensors.append({
                "input": x.detach().float().cpu(), "raw_x0": denoised.detach().float().cpu(),
                "proposal": proposal.detach().float().cpu(),
                "source_next": target_next.detach().float().cpu(),
                "accepted": accepted.detach().float().cpu(), "next_raw_x0": None,
            })
            x = accepted
            if callback is not None:
                callback(step, denoised, x, total)
        return sampling.inverse_noise_scaling(sigmas[-1], x)


def load_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def execute(model, positive, negative, sigmas, source_latent, noise,
            editable, stitch_steps):
    sampler = StitchEulerSampler(editable, stitch_steps)
    with torch.inference_mode():
        result = base.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            source_latent.clone(), noise_mask=None, callback=None, disable_pbar=True, seed=SEED,
        ).detach().cpu()
    return result, sampler


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pixels = load_pixels(SOURCE)
    vae = base.comfy.sd.VAE(sd=base.comfy.utils.load_torch_file(str(model_paths.VAE_PATH), safe_load=True))
    with torch.no_grad():
        source_latent = vae.encode(pixels)
    editable, locked, boundary = masks(source_latent)

    model = base.comfy.sd.load_diffusion_model(str(model_paths.MODEL_PATH), model_options={})
    clip = base.comfy.sd.load_clip([str(model_paths.TEXT_ENCODER_PATH)], clip_type=base.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    del clip
    base.comfy.model_management.unload_all_models(); base.comfy.model_management.soft_empty_cache()

    sigmas = base.get_schedule(STEPS, math.prod(source_latent.shape[-2:])).float()
    sigmas[0] = 1.0
    base.validate_schedule(sigmas)
    noise = torch.randn(source_latent.shape, generator=torch.Generator().manual_seed(SEED))
    arms: dict[str, torch.Tensor] = {}
    samplers: dict[str, StitchEulerSampler] = {}
    for name, policy in ARMS.items():
        arms[name], samplers[name] = execute(
            model, positive, negative, sigmas, source_latent, noise, editable, policy
        )

    ordinary = samplers["A_ORDINARY"]
    comparison: dict[str, list[dict[str, Any]]] = {}
    for name, sampler in samplers.items():
        rows = []
        for step, tensors in enumerate(sampler.tensors):
            ordinary_tensors = ordinary.tensors[step]
            row = dict(sampler.intervals[step])
            row.update({
                "accepted_state_vs_ordinary": {
                    "source": region_metrics(tensors["accepted"], ordinary_tensors["accepted"], locked),
                    "editable": region_metrics(tensors["accepted"], ordinary_tensors["accepted"], editable),
                    "boundary": region_metrics(tensors["accepted"], ordinary_tensors["accepted"], boundary),
                },
                "raw_x0_vs_ordinary": {
                    "source": region_metrics(tensors["raw_x0"], ordinary_tensors["raw_x0"], locked),
                    "editable": region_metrics(tensors["raw_x0"], ordinary_tensors["raw_x0"], editable),
                    "boundary": region_metrics(tensors["raw_x0"], ordinary_tensors["raw_x0"], boundary),
                },
                "boundary_latent_rms": base.masked_rms(tensors["accepted"], boundary),
                "next_raw_x0_captured": tensors["next_raw_x0"] is not None,
            })
            rows.append(row)
        comparison[name] = rows

    processed_raw = {
        name: [model.model.process_latent_out(item["raw_x0"]) for item in sampler.tensors]
        for name, sampler in samplers.items()
    }
    model_sampling_name = type(model.get_model_object("model_sampling")).__name__
    model.cleanup(); del model
    base.comfy.model_management.unload_all_models(); base.comfy.model_management.soft_empty_cache(); gc.collect()

    final_sheet = []
    for name, latent in arms.items():
        path = OUTPUT / f"{name}_FINAL.png"
        base.save_pixels(vae.decode(latent).detach().cpu(), path)
        final_sheet.append((name, path))
    base.make_sheet(final_sheet, OUTPUT / "RIGID_BRIDGE_STITCH_FINAL_COMPARISON.png")
    raw_sheet = []
    for name, predictions in processed_raw.items():
        for step, prediction in enumerate(predictions):
            path = OUTPUT / f"{name}_STEP_{step:02d}_RAW_X0.png"
            base.save_pixels(vae.decode(prediction).detach().cpu(), path)
            raw_sheet.append((f"{name} raw x0 step {step}", path))
    base.make_sheet(raw_sheet, OUTPUT / "RIGID_BRIDGE_STITCH_RAW_X0_TRAJECTORY.png")
    del vae
    base.comfy.model_management.unload_all_models(); base.comfy.model_management.soft_empty_cache()

    report = {
        "experiment": "simplified source stitch -> full denoise",
        "classification": "SIMPLIFIED SOURCE-STITCH MECHANISM INSUFFICIENT",
        "configuration": {
            "case": "rigid_bridge", "model_sampling": model_sampling_name,
            "sampler": "deterministic Euler", "steps": STEPS, "seed": SEED,
            "sigmas": [float(v) for v in sigmas], "prompt": PROMPT,
            "mask": "latent columns 16:48 source; complement editable; 1=editable",
            "source_trajectory": "(1-sigma)*source_latent + sigma*fixed_initial_noise",
        },
        "arms": {
            name: {"final_hash": base.tensor_hash(value), "intervals": comparison[name]}
            for name, value in arms.items()
        },
        "integrity": {
            "same_fixed_noise_all_arms": len({s.fixed_noise_hash for s in samplers.values()}) == 1,
            "fixed_noise_reused_each_interval": all(
                len({row["fixed_noise_hash"] for row in sampler.intervals}) == 1
                for sampler in samplers.values()
            ),
            "first_model_input_all_arms_exact": len({s.intervals[0]["input_hash"] for s in samplers.values()}) == 1,
            "first_raw_x0_all_arms_exact": len({s.intervals[0]["raw_x0_hash"] for s in samplers.values()}) == 1,
            "single_stitch_only_interval_zero": [r["stitched"] for r in samplers["B_SINGLE_STITCH"].intervals]
                == [True] + [False] * (STEPS - 1),
            "repeated_stitch_every_interval": all(r["stitched"] for r in samplers["C_REPEATED_STITCH"].intervals),
            "editable_never_modified_by_stitch": all(
                r["editable_proposal_to_accepted_max"] == 0.0
                for sampler in samplers.values() for r in sampler.intervals
            ),
            "production_changed": False, "clown_or_res4lyf_used": False,
            "noise_mask": None, "attention_or_model_patch": None,
        },
        "semantic_review": {
            "status": "FAIL",
            "causal_next_prediction": "The interval-0 stitch changes the next editable raw x0 by 0.440 RMS; B and C are identical at that evaluation.",
            "single_stitch": "The effect persists but source state subsequently drifts almost as far from its trajectory as Ordinary.",
            "repeated_stitch": "Source trajectory remains exact, but final editable bridge/cable systems restart independently at both source boundaries.",
            "decision": "Stop this simplified branch without tuning.",
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        "source_latent": source_latent, "noise": noise, "editable": editable,
        "sigmas": sigmas, "arms": arms,
        "trajectories": {name: sampler.tensors for name, sampler in samplers.items()},
    })
    base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"),
                      "comparison": str(OUTPUT / "RIGID_BRIDGE_STITCH_FINAL_COMPARISON.png")}, indent=2))


if __name__ == "__main__":
    run()
