"""Experiment-only hard-source Local Edit discriminator for FLUX.2 Klein."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import comfy.model_management  # noqa: E402
import comfy.model_sampling  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_flux import get_schedule  # noqa: E402

import flux2_coarse_global_local_falsification as phase2  # noqa: E402


OUTPUT = ROOT / "experiments" / "local_edit_hard_source_klein_results"
SOURCE_IMAGE = (
    ROOT
    / "experiments"
    / "terminal_resampling_geometry_qualification_results"
    / "LANDSCAPE_BRIDGE"
    / "selected_0.png"
)
WIDTH = 1024
HEIGHT = 512
SOURCE_WIDTH = 768
LATENT_HW = (HEIGHT // 16, WIDTH // 16)
LOCKED_LATENT_WIDTH = SOURCE_WIDTH // 16
TRANSITION_LATENT_WIDTH = 2
STEPS = 4
SEED = 20260909
PROMPT = (
    "A wide cinematic photograph of exactly one long red suspension bridge "
    "crossing calm water continuously from left to right, one yellow passenger "
    "train on the bridge, controlled towers and continuous cables, one coherent "
    "horizon, sunset light, extend the same bridge landscape naturally to the right"
)
TOLERANCE = 1e-6


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu()
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def masked_values(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(value).bool()
    return value[expanded]


def masked_rms(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = masked_values(value, mask).detach()
    return float(selected.float().square().mean().sqrt()) if selected.numel() else 0.0


def masked_max_abs(value: torch.Tensor, mask: torch.Tensor) -> float:
    selected = masked_values(value, mask).detach()
    return float(selected.float().abs().max()) if selected.numel() else 0.0


def validate_mask(editable_mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if editable_mask.shape != (reference.shape[0], 1, *reference.shape[-2:]):
        raise ValueError("Local Edit mask must be [B,1,H,W] at latent geometry.")
    if editable_mask.dtype == torch.bool:
        editable_mask = editable_mask.to(reference.dtype)
    if not torch.isfinite(editable_mask).all():
        raise ValueError("Local Edit editable mask must be finite.")
    if not torch.all((editable_mask == 0) | (editable_mask == 1)):
        raise ValueError("Local Edit hard-source experiment requires a binary mask with 1=editable.")
    return editable_mask.to(device=reference.device, dtype=reference.dtype)


def reject_noise_mask(denoise_mask: torch.Tensor | None) -> None:
    if denoise_mask is not None:
        raise ValueError(
            "Local Edit hard-source experiment rejects incoming noise_mask; "
            "its editable mask is the sole state-ownership mask."
        )


def source_trajectory(source: torch.Tensor, fixed_noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (1.0 - sigma) * source + sigma * fixed_noise


def hard_source_derivative(
    x_sigma: torch.Tensor,
    denoised: torch.Tensor,
    source: torch.Tensor,
    editable_mask: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if float(sigma) <= 0.0:
        raise ValueError("Derivative evaluation requires sigma > 0.")
    editable = validate_mask(editable_mask, x_sigma)
    locked = 1.0 - editable
    d_model = (x_sigma - denoised) / sigma
    d_source = (x_sigma - source) / sigma
    d_guided = editable * d_model + locked * d_source
    return d_guided, d_model, d_source


def restore_after_euler(
    x_star: torch.Tensor,
    source: torch.Tensor,
    fixed_noise: torch.Tensor,
    editable_mask: torch.Tensor,
    sigma_next: torch.Tensor,
) -> torch.Tensor:
    editable = validate_mask(editable_mask, x_star)
    locked = 1.0 - editable
    y_next = source_trajectory(source, fixed_noise, sigma_next)
    return editable * x_star + locked * y_next


def derive_transition_band(editable_mask: torch.Tensor, width: int) -> torch.Tensor:
    if width < 1:
        raise ValueError("Transition width must be positive.")
    editable = validate_mask(editable_mask, editable_mask)
    locked = 1.0 - editable
    near_lock = F.max_pool2d(locked, kernel_size=2 * width + 1, stride=1, padding=width)
    transition = editable * (near_lock > 0).to(editable.dtype)
    if torch.any(transition > editable):
        raise RuntimeError("Derived transition band escaped the editable mask.")
    return transition


def validate_schedule(sigmas: torch.Tensor) -> None:
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("Local Edit requires a one-dimensional sigma schedule.")
    if not torch.isfinite(sigmas).all() or float(sigmas[-1]) != 0.0:
        raise ValueError("Local Edit requires finite sigmas ending exactly at zero.")
    if not bool(torch.all(sigmas[:-1] > sigmas[1:])):
        raise ValueError("Local Edit requires a strictly decreasing sigma schedule.")


def interval_record(
    step: int,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
    x: torch.Tensor,
    x_next: torch.Tensor,
    source: torch.Tensor,
    fixed_noise: torch.Tensor,
    editable: torch.Tensor,
    transition: torch.Tensor,
    d_model: torch.Tensor,
    d_source: torch.Tensor,
    d_guided: torch.Tensor,
) -> dict[str, Any]:
    locked = 1.0 - editable
    y_sigma = source_trajectory(source, fixed_noise, sigma)
    y_next = source_trajectory(source, fixed_noise, sigma_next)
    return {
        "step": step,
        "sigma": float(sigma),
        "sigma_next": float(sigma_next),
        "locked_state_rms_error_vs_y_sigma": masked_rms(x - y_sigma, locked),
        "locked_state_max_error_vs_y_sigma": masked_max_abs(x - y_sigma, locked),
        "accepted_locked_rms_error_vs_y_sigma_next": masked_rms(x_next - y_next, locked),
        "accepted_locked_max_error_vs_y_sigma_next": masked_max_abs(x_next - y_next, locked),
        "editable_derivative_rms": masked_rms(d_guided, editable),
        "locked_raw_model_derivative_rms": masked_rms(d_model, locked),
        "locked_replacement_derivative_rms": masked_rms(d_source, locked),
        "replacement_increment_rms": masked_rms(d_guided - d_model, locked),
        "boundary_band_latent_rms": masked_rms(x_next, transition),
        "editable_derivative_change_max_abs": masked_max_abs(d_guided - d_model, editable),
        "accepted_hash": tensor_hash(x_next),
    }


class OrdinaryEulerSampler(comfy.samplers.Sampler):
    def __init__(self) -> None:
        self.intervals: list[dict[str, Any]] = []
        self.accepted: list[torch.Tensor] = []
        self.first_raw_prediction_hash: str | None = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        reject_noise_mask(denoise_mask)
        validate_schedule(sigmas)
        if noise.shape[0] != 1 or noise.shape[1] != 128:
            raise ValueError("Local Edit discriminator requires batch-one Klein 128-channel latents.")
        sampling = model.inner_model.model_sampling
        if not isinstance(sampling, comfy.model_sampling.CONST):
            raise TypeError(f"Local Edit discriminator requires CONST sampling, got {type(sampling).__name__}.")
        x = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        total = len(sigmas) - 1
        for step, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            denoised = model(x, sigma.expand(1), **extra_args)
            if self.first_raw_prediction_hash is None:
                self.first_raw_prediction_hash = tensor_hash(denoised)
            derivative = (x - denoised) / sigma
            x_next = x + (sigma_next - sigma) * derivative
            self.intervals.append({
                "step": step,
                "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "accepted_hash": tensor_hash(x_next),
            })
            self.accepted.append(x_next.detach().float().cpu())
            x = x_next
            if callback is not None:
                callback(step, denoised, x, total)
        return sampling.inverse_noise_scaling(sigmas[-1], x)


class HardSourceEulerSampler(comfy.samplers.Sampler):
    def __init__(self, editable_mask: torch.Tensor) -> None:
        self.editable_mask_cpu = editable_mask.detach().float().cpu()
        self.intervals: list[dict[str, Any]] = []
        self.accepted: list[torch.Tensor] = []
        self.first_raw_prediction_hash: str | None = None
        self.source_noise_hash: str | None = None
        self.source_noise_step_hashes: list[str] = []

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        reject_noise_mask(denoise_mask)
        validate_schedule(sigmas)
        if latent_image is None or noise.shape != latent_image.shape:
            raise ValueError("Local Edit requires source latent and noise with identical shape.")
        if noise.shape[0] != 1 or noise.shape[1] != 128:
            raise ValueError("Local Edit discriminator requires batch-one Klein 128-channel latents.")
        sampling = model.inner_model.model_sampling
        if not isinstance(sampling, comfy.model_sampling.CONST):
            raise TypeError(f"Local Edit discriminator requires CONST sampling, got {type(sampling).__name__}.")
        source = latent_image
        fixed_noise = noise.clone()
        self.source_noise_hash = tensor_hash(fixed_noise)
        editable = validate_mask(self.editable_mask_cpu.to(noise.device), noise)
        locked = 1.0 - editable
        transition = derive_transition_band(editable, TRANSITION_LATENT_WIDTH)
        if torch.any(transition * locked):
            raise RuntimeError("Transition band must lie entirely inside the editable mask.")
        x = sampling.noise_scaling(sigmas[0], noise, source, self.max_denoise(model, sigmas))
        x = restore_after_euler(x, source, fixed_noise, editable, sigmas[0])
        total = len(sigmas) - 1
        for step, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            self.source_noise_step_hashes.append(tensor_hash(fixed_noise))
            denoised = model(x, sigma.expand(1), **extra_args)
            if self.first_raw_prediction_hash is None:
                self.first_raw_prediction_hash = tensor_hash(denoised)
            d_guided, d_model, d_source = hard_source_derivative(x, denoised, source, editable, sigma)
            x_star = x + (sigma_next - sigma) * d_guided
            x_next = restore_after_euler(x_star, source, fixed_noise, editable, sigma_next)
            record = interval_record(
                step, sigma, sigma_next, x, x_next, source, fixed_noise,
                editable, transition, d_model, d_source, d_guided,
            )
            self.intervals.append(record)
            self.accepted.append(x_next.detach().float().cpu())
            if record["accepted_locked_max_error_vs_y_sigma_next"] > TOLERANCE:
                raise RuntimeError(
                    "S1 failed at accepted step "
                    f"{step}: {record['accepted_locked_max_error_vs_y_sigma_next']:.9g}"
                )
            x = x_next
            if callback is not None:
                callback(step, denoised, x, total)
        if masked_max_abs(x - source, locked) > TOLERANCE:
            raise RuntimeError("S1 failed at terminal sigma=0.")
        return sampling.inverse_noise_scaling(sigmas[-1], x)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def save_pixels(pixels: torch.Tensor, path: Path) -> None:
    phase2.save_pixels(pixels, path)


def prepare_source() -> tuple[torch.Tensor, torch.Tensor]:
    with Image.open(SOURCE_IMAGE) as source_file:
        source = source_file.convert("RGB").resize((SOURCE_WIDTH, HEIGHT), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "black")
    canvas.paste(source, (0, 0))
    canvas.save(OUTPUT / "SOURCE_PADDED.png")
    source.save(OUTPUT / "SOURCE_ORIGINAL_SIDE.png")
    array = torch.from_numpy(__import__("numpy").array(canvas)).float().div(255.0).unsqueeze(0)
    editable = torch.zeros((1, 1, *LATENT_HW), dtype=torch.float32)
    editable[..., LOCKED_LATENT_WIDTH:] = 1.0
    return array, editable


def pixel_metrics(candidate: torch.Tensor, baseline: torch.Tensor, padded: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float()
    baseline = baseline.detach().float()
    padded = padded.detach().float()
    boundary = SOURCE_WIDTH
    locked = candidate[:, :, :, :boundary]
    base_locked = baseline[:, :, :, :boundary]
    input_locked = padded[:, :, :, :boundary]
    safe_end = boundary - 32
    locked_error = (locked - base_locked).float()
    self_error = (base_locked - input_locked).float()
    strip_locked = candidate[:, :, :, boundary - 32:boundary]
    strip_locked_base = baseline[:, :, :, boundary - 32:boundary]
    strip_edit = candidate[:, :, :, boundary:boundary + 32]
    strip_edit_base = baseline[:, :, :, boundary:boundary + 32]
    seam_gradient = candidate[:, :, :, boundary] - candidate[:, :, :, boundary - 1]
    left_gradient = candidate[:, :, :, boundary - 1] - candidate[:, :, :, boundary - 2]
    right_gradient = candidate[:, :, :, boundary + 1] - candidate[:, :, :, boundary]
    edit_change = candidate[:, :, :, boundary:] - baseline[:, :, :, boundary:]
    mse = float(locked_error.square().mean())
    return {
        "locked_vs_source_reconstruction_mae": float(locked_error.abs().mean()),
        "locked_vs_source_reconstruction_max_abs": float(locked_error.abs().max()),
        "locked_vs_source_reconstruction_psnr": float("inf") if mse == 0 else -10.0 * math.log10(mse),
        "safe_locked_interior_mae": float((candidate[:, :, :, :safe_end] - baseline[:, :, :, :safe_end]).abs().mean()),
        "vae_self_reconstruction_locked_mae": float(self_error.abs().mean()),
        "locked_boundary_strip_mae": float((strip_locked - strip_locked_base).abs().mean()),
        "editable_boundary_strip_change_rms": rms(strip_edit - strip_edit_base),
        "seam_gradient_rms": rms(seam_gradient),
        "left_adjacent_gradient_rms": rms(left_gradient),
        "right_adjacent_gradient_rms": rms(right_gradient),
        "editable_region_change_rms": rms(edit_change),
    }


def make_sheet(paths: list[tuple[str, Path]], output: Path) -> None:
    panels = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        panel = Image.new("RGB", (image.width, image.height + 34), "white")
        panel.paste(image, (0, 34))
        ImageDraw.Draw(panel).text((8, 9), label, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height for p in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(output)


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pixels, editable = prepare_source()
    transition = derive_transition_band(editable, TRANSITION_LATENT_WIDTH)
    if torch.any(transition > editable):
        raise RuntimeError("Transition derivation failed containment.")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    source_latent = vae.encode(pixels)
    source_reconstruction = vae.decode(source_latent).cpu()
    save_pixels(source_reconstruction, OUTPUT / "SOURCE_RECONSTRUCTION.png")
    del vae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    model = comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    sigmas = get_schedule(STEPS, math.prod(LATENT_HW)).float()
    sigmas[0] = 1.0
    validate_schedule(sigmas)
    noise = torch.randn(source_latent.shape, generator=torch.Generator().manual_seed(SEED))
    arms: dict[str, torch.Tensor] = {}
    samplers: dict[str, Any] = {}
    started = time.perf_counter()

    ordinary = OrdinaryEulerSampler()
    with torch.inference_mode():
        arms["A_ordinary"] = comfy.sample.sample_custom(
            model, noise.clone(), 1.0, ordinary, sigmas, positive, negative,
            source_latent.clone(), noise_mask=None, callback=None, disable_pbar=True, seed=SEED,
        ).cpu()
    samplers["A_ordinary"] = ordinary

    native_mask = editable[:, 0]
    with torch.inference_mode():
        arms["B_native_noise_mask"] = comfy.sample.sample_custom(
            model, noise.clone(), 1.0, comfy.samplers.ksampler("euler"), sigmas,
            positive, negative, source_latent.clone(), noise_mask=native_mask,
            callback=None, disable_pbar=True, seed=SEED,
        ).cpu()

    hard = HardSourceEulerSampler(editable)
    with torch.inference_mode():
        arms["C_hard_source"] = comfy.sample.sample_custom(
            model, noise.clone(), 1.0, hard, sigmas, positive, negative,
            source_latent.clone(), noise_mask=None, callback=None, disable_pbar=True, seed=SEED,
        ).cpu()
    samplers["C_hard_source"] = hard

    repeat = HardSourceEulerSampler(editable)
    with torch.inference_mode():
        arms["C_hard_source_repeat"] = comfy.sample.sample_custom(
            model, noise.clone(), 1.0, repeat, sigmas, positive, negative,
            source_latent.clone(), noise_mask=None, callback=None, disable_pbar=True, seed=SEED,
        ).cpu()
    samplers["C_hard_source_repeat"] = repeat

    atomic_torch(OUTPUT / "runtime_tensors.pt", {
        "arms": arms,
        "hard_intervals": hard.intervals,
        "hard_accepted": hard.accepted,
        "ordinary_first_raw_prediction_hash": ordinary.first_raw_prediction_hash,
        "hard_first_raw_prediction_hash": hard.first_raw_prediction_hash,
        "source_noise_hash": hard.source_noise_hash,
        "source_noise_step_hashes": hard.source_noise_step_hashes,
        "source_latent": source_latent,
        "editable_mask": editable,
        "sigmas": sigmas,
    })

    model_sampling_name = type(model.get_model_object("model_sampling")).__name__
    processed_intermediates = [model.model.process_latent_out(value) for value in hard.accepted]
    model.cleanup()
    del model
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded: dict[str, torch.Tensor] = {}
    image_paths: list[tuple[str, Path]] = [
        ("SOURCE PADDED", OUTPUT / "SOURCE_PADDED.png"),
        ("SOURCE VAE RECONSTRUCTION", OUTPUT / "SOURCE_RECONSTRUCTION.png"),
    ]
    for name, latent in arms.items():
        decoded[name] = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        save_pixels(decoded[name], path)
        image_paths.append((name, path))
    for step, latent in enumerate(processed_intermediates):
        preview = vae.decode(latent).cpu()
        save_pixels(preview, OUTPUT / f"C_STEP_{step:02d}_ACCEPTED.png")
    del vae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    make_sheet(image_paths, OUTPUT / "COMPARISON.png")

    padded_chw = pixels.permute(0, 3, 1, 2)
    metrics = {
        name: pixel_metrics(value.permute(0, 3, 1, 2), source_reconstruction.permute(0, 3, 1, 2), padded_chw)
        for name, value in decoded.items()
    }
    locked = 1.0 - editable
    report = {
        "experiment": "Local Edit hard-source Klein discriminator",
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "source_image": str(SOURCE_IMAGE),
            "canvas_wh": [WIDTH, HEIGHT],
            "source_wh": [SOURCE_WIDTH, HEIGHT],
            "latent_hw": list(LATENT_HW),
            "locked_latent_width": LOCKED_LATENT_WIDTH,
            "mask_convention": "1=editable, 0=locked",
            "transition_band_latent_width": TRANSITION_LATENT_WIDTH,
            "transition_band_active": False,
            "prompt": PROMPT,
            "seed": SEED,
            "steps": STEPS,
            "sigmas": [float(value) for value in sigmas],
            "model_sampling": model_sampling_name,
            "sampler": "deterministic Euler",
        },
        "unit_tests": "run separately before inference; see test_local_edit_hard_source_klein.py",
        "arms": {
            name: {"latent_hash": tensor_hash(value), "pixel_metrics": metrics[name]}
            for name, value in arms.items()
        },
        "hard_source": {
            "intervals": hard.intervals,
            "source_noise_hash": hard.source_noise_hash,
            "source_noise_step_hashes": hard.source_noise_step_hashes,
            "fixed_source_noise_reused": len(set(hard.source_noise_step_hashes)) == 1,
            "first_raw_prediction_hash": hard.first_raw_prediction_hash,
            "terminal_locked_max_abs_vs_source": masked_max_abs(arms["C_hard_source"] - source_latent, locked),
        },
        "first_divergence": {
            "ordinary_first_raw_prediction_hash": ordinary.first_raw_prediction_hash,
            "hard_source_first_raw_prediction_hash": hard.first_raw_prediction_hash,
            "identical": ordinary.first_raw_prediction_hash == hard.first_raw_prediction_hash,
            "intended_boundary": "hard_source_derivative on locked coordinates",
        },
        "comparisons": {
            "C_repeat_bit_exact": bool(torch.equal(arms["C_hard_source"], arms["C_hard_source_repeat"])),
            "B_vs_C_bit_exact": bool(torch.equal(arms["B_native_noise_mask"], arms["C_hard_source"])),
            "B_vs_C_max_abs": float((arms["B_native_noise_mask"] - arms["C_hard_source"]).abs().max()),
            "A_vs_C_rms": rms(arms["A_ordinary"] - arms["C_hard_source"]),
        },
        "integrity": {
            "S1_pass": all(
                item["accepted_locked_max_error_vs_y_sigma_next"] <= TOLERANCE
                for item in hard.intervals
            ) and masked_max_abs(arms["C_hard_source"] - source_latent, locked) <= TOLERANCE,
            "transition_inside_editable": bool(torch.all(transition <= editable)),
            "transition_active": False,
            "incoming_noise_mask_local_edit": None,
            "res4lyf_runtime_dependency": False,
            "production_changes": False,
        },
        "runtime_seconds": time.perf_counter() - started,
        "semantic_review": {"status": "PENDING"},
        "decision": "PENDING",
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "comparison": str(OUTPUT / "COMPARISON.png"),
        "S1_pass": report["integrity"]["S1_pass"],
        "B_vs_C_bit_exact": report["comparisons"]["B_vs_C_bit_exact"],
        "repeat_bit_exact": report["comparisons"]["C_repeat_bit_exact"],
    }, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run the fixed diffusion discriminator.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not arguments.run:
        raise SystemExit("Use --run after the algebra tests pass.")
    run()
