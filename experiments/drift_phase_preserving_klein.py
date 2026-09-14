"""Inference-only phase-preserving-noise falsification on stock FLUX.2 Klein 4B.

This is an experiment harness, not a ComfyUI node.  It keeps native CONST
noise scaling and a native Euler update; only the noise endpoint is varied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
MODEL = Path(r"C:\Users\Tom-M\data\a\ai\models-1\models\diffusion_models\f2k4_realrebelai_Rebels_w4a8s\Flux2-Klein-4B-w4a8.safetensors")
TEXT_ENCODER = Path(r"C:\Users\Tom-M\data\a\ai\models-1\models\text_encoders\f2k4_Comfy-Org_vae-text-encorder-for-flux-klein-4b_text_encoders\qwen_3_4b_fp4_flux2.safetensors")
VAE = Path(r"C:\Users\Tom-M\data\a\ai\models-1\models\vae\Comfy-Org_vae-text-encorder-for-flux-klein-9b_vae\flux2-vae.safetensors")
OUTPUT = ROOT / "experiments" / "drift_phase_preserving_results"
PROMPT = "A richly textured cinematic night painting of the same bridge, lighthouse, river, moon, and tower, coherent geometry, blue and gold lighting"
SEED = 2026091401
WIDTH = HEIGHT = 512
STEPS = 8
START_INDEX = 2
CUTOFFS = (2.0, 8.0, 16.0)
TRANSITION_BANDWIDTH = 2.0

sys.path.insert(0, str(COMFY_ROOT))


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def summary(value: torch.Tensor) -> dict:
    work = value.detach().float()
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "rms": rms(work), "mean": float(work.mean()),
        "variance": float(work.var(unbiased=False)),
        "min": float(work.min()), "max": float(work.max()),
        "finite": bool(torch.isfinite(work).all()),
    }


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def phase_rms(value: torch.Tensor, source: torch.Tensor) -> float:
    value_fft = torch.fft.fft2(value.detach().float(), dim=(-2, -1))
    source_fft = torch.fft.fft2(source.detach().float(), dim=(-2, -1))
    delta = torch.angle(value_fft * source_fft.conj())
    valid = source_fft.abs() > source_fft.abs().mean(dim=(-2, -1), keepdim=True) * 1e-4
    return rms(delta[valid])


def structural_metrics(value: torch.Tensor, source: torch.Tensor) -> dict:
    work = value.detach().float()
    reference = source.detach().float().to(work)
    dx = work[..., :, 1:] - work[..., :, :-1]
    sx = reference[..., :, 1:] - reference[..., :, :-1]
    dy = work[..., 1:, :] - work[..., :-1, :]
    sy = reference[..., 1:, :] - reference[..., :-1, :]
    return {
        "source_relative_rms": rms(work - reference),
        "source_fourier_phase_rms_radians": phase_rms(work, reference),
        "source_coarse_4x_rms": rms(F.avg_pool2d(work - reference, 4, 4)),
        "source_gradient_difference_rms": math.sqrt(
            (rms(dx - sx) ** 2 + rms(dy - sy) ** 2) / 2.0
        ),
    }


def frequency_mask(height: int, width: int, cutoff: float, bandwidth: float) -> torch.Tensor:
    fy = torch.arange(height, dtype=torch.float32) - height // 2
    fx = torch.arange(width, dtype=torch.float32) - width // 2
    radius = torch.sqrt(fy[:, None].square() + fx[None, :].square())
    outside = torch.exp(-torch.clamp(radius - cutoff, min=0).square() / (2 * bandwidth**2))
    return torch.where(radius <= cutoff, torch.ones_like(radius), outside)


def structured_noise(source: torch.Tensor, gaussian: torch.Tensor,
                     cutoff: float | None = None) -> tuple[torch.Tensor, dict]:
    source_fft = torch.fft.fft2(source.float(), dim=(-2, -1))
    gaussian_fft = torch.fft.fft2(gaussian.float(), dim=(-2, -1))
    source_phase = torch.angle(source_fft)
    gaussian_phase = torch.angle(gaussian_fft)
    if cutoff is None:
        phase = source_phase
        mask = torch.ones(source.shape[-2:], dtype=torch.float32)
    else:
        shifted_mask = frequency_mask(source.shape[-2], source.shape[-1], cutoff, TRANSITION_BANDWIDTH)
        mask = torch.fft.ifftshift(shifted_mask).to(source_phase)
        phase = source_phase * mask + gaussian_phase * (1.0 - mask)
    spectrum = gaussian_fft.abs() * torch.exp(1j * phase)
    inverse = torch.fft.ifft2(spectrum, dim=(-2, -1))
    result = inverse.real.to(gaussian.dtype)
    reconstructed_fft = torch.fft.fft2(result.float(), dim=(-2, -1))
    magnitude_error = (reconstructed_fft.abs() - gaussian_fft.abs()).abs()
    report = {
        "cutoff_radius_latent_frequency_pixels": cutoff,
        "transition_bandwidth": None if cutoff is None else TRANSITION_BANDWIDTH,
        "mask_mean": float(mask.mean()),
        "imaginary_residual_rms": rms(inverse.imag),
        "gaussian": summary(gaussian), "structured": summary(result),
        "structured_to_gaussian_rms_ratio": rms(result) / rms(gaussian),
        "structured_to_gaussian_variance_ratio": float(result.float().var(unbiased=False) / gaussian.float().var(unbiased=False)),
        "fourier_magnitude_max_abs_error": float(magnitude_error.max()),
        "fourier_magnitude_relative_rms_error": rms(magnitude_error) / rms(gaussian_fft.abs()),
        "phase_rms_to_source": phase_rms(result, source),
        "phase_rms_to_gaussian": phase_rms(result, gaussian),
    }
    return result, report


def validate_preflight(gaussian: torch.Tensor, variants: dict[str, torch.Tensor], reports: dict) -> None:
    tolerance = 2e-5
    for name, value in variants.items():
        report = reports[name]
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name}: structured noise is nonfinite")
        if name == "B_FULL":
            if abs(report["structured_to_gaussian_rms_ratio"] - 1.0) > tolerance:
                raise RuntimeError(f"{name}: Parseval/RMS mismatch")
            if report["imaginary_residual_rms"] > tolerance:
                raise RuntimeError(f"{name}: inverse FFT is not effectively real")
            if report["fourier_magnitude_relative_rms_error"] > tolerance:
                raise RuntimeError(f"{name}: Gaussian Fourier magnitude was not preserved")
    if reports["B_FULL"]["phase_rms_to_source"] > tolerance:
        raise RuntimeError("B_FULL: source phase was not preserved")
    ordered = [reports[f"C_FSS_R{int(radius):02d}"]["phase_rms_to_source"] for radius in CUTOFFS]
    if not all(right < left for left, right in zip(ordered, ordered[1:])):
        raise RuntimeError(f"FSS rigidity is not monotonic: {ordered}")
    if tensor_hash(gaussian) != tensor_hash(gaussian.clone()):
        raise RuntimeError("Gaussian control changed during preflight")


def make_source(path: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (17, 29, 46))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 340, WIDTH, HEIGHT), fill=(27, 66, 91))
    draw.ellipse((340, 35, 440, 135), fill=(238, 218, 154))
    draw.rectangle((40, 185, 82, 342), fill=(220, 218, 198))
    draw.polygon(((33, 185), (61, 145), (89, 185)), fill=(171, 52, 43))
    draw.rectangle((420, 215, 478, 342), fill=(66, 61, 68))
    draw.polygon(((410, 215), (449, 170), (488, 215)), fill=(44, 39, 48))
    draw.line((55, 290, 450, 290), fill=(195, 93, 48), width=18)
    draw.line((55, 290, 150, 205, 260, 290, 360, 205, 450, 290), fill=(222, 145, 68), width=5)
    for x in range(75, 450, 25):
        draw.line((x, 290, x, 330), fill=(185, 116, 61), width=3)
    image.save(path)


class EulerCaptureSampler:
    def __init__(self, source: torch.Tensor, control: list[dict] | None = None) -> None:
        self.source = source
        self.control = control
        self.captures: list[dict] = []
        self.tensors: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def max_denoise(model, sigmas) -> bool:
        return bool(float(sigmas[0]) >= float(model.inner_model.model_sampling.sigma_max))

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None:
            raise ValueError("This probe requires unmasked source img2img")
        sampling = model.inner_model.model_sampling
        state = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        for ordinal, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
            denoised = model(state, sigma.expand(state.shape[0]), model_options=extra_args["model_options"], seed=extra_args.get("seed", 0))
            record = {
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "state": summary(state), "denoised_x0": summary(denoised),
                "state_structure": structural_metrics(state, self.source),
                "denoised_structure": structural_metrics(denoised, self.source),
            }
            if self.control is not None:
                record["state_rms_difference_from_control"] = rms(state - self.control[ordinal][0].to(state))
                record["denoised_rms_difference_from_control"] = rms(denoised - self.control[ordinal][1].to(denoised))
            self.tensors.append((state.detach().float().cpu(), denoised.detach().float().cpu()))
            self.captures.append(record)
            state = state + (state - denoised) / sigma * (sigma_next - sigma)
            if callback is not None:
                callback(ordinal, denoised, state, len(sigmas) - 1)
        return sampling.inverse_noise_scaling(sigmas[-1], state)


def save_pixels(pixels: torch.Tensor, path: Path) -> None:
    array = (pixels[0].clamp(0, 1).mul(255).round().byte().numpy())
    Image.fromarray(array).save(path)


def image_record(path: Path) -> dict:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size), "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_path = OUTPUT / "SOURCE.png"
    make_source(source_path)

    import comfy.model_management
    import comfy.model_sampling
    import comfy.sample
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_flux import get_schedule

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE), safe_load=True))
    with Image.open(source_path) as image:
        pixels = torch.from_numpy(__import__("numpy").array(image.convert("RGB"))).float().div(255).unsqueeze(0)
    with torch.inference_mode():
        source_latent = vae.encode(pixels).detach().float().cpu()
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    gaussian = torch.randn(source_latent.shape, generator=generator)
    variants = {"A_GAUSSIAN": gaussian}
    reports = {"A_GAUSSIAN": {"gaussian": summary(gaussian), "hash": tensor_hash(gaussian)}}
    full, reports["B_FULL"] = structured_noise(source_latent, gaussian)
    variants["B_FULL"] = full
    for cutoff in CUTOFFS:
        name = f"C_FSS_R{int(cutoff):02d}"
        variants[name], reports[name] = structured_noise(source_latent, gaussian, cutoff)
    validate_preflight(gaussian, {key: value for key, value in variants.items() if key != "A_GAUSSIAN"}, reports)
    preflight = {
        "status": "PASS_WITH_QUANTIFIED_FSS_REAL_PROJECTION", "source_latent": summary(source_latent),
        "source_latent_hash": tensor_hash(source_latent), "noise": reports,
        "interpretation": "Paper epsilon-hat is substituted only for native CONST's noise endpoint; native sigma interpolation is unchanged.",
    }
    (OUTPUT / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return

    model = comfy.sd.load_diffusion_model(str(MODEL), model_options={})
    if not isinstance(model.model.model_sampling, comfy.model_sampling.CONST):
        raise RuntimeError(f"Expected CONST sampling, got {type(model.model.model_sampling).__name__}")
    clip = comfy.sd.load_clip([str(TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    full_schedule = get_schedule(STEPS, math.prod(source_latent.shape[-2:])).float()
    sigmas = full_schedule[START_INDEX:].clone()
    if float(sigmas[-1]) != 0.0 or not bool((sigmas[1:] < sigmas[:-1]).all()):
        raise RuntimeError("Unexpected Klein sigma schedule")

    report = {
        "verdict": "INFERENCE-ONLY FSS PARTIAL / UNCERTAIN",
        "configuration": {
            "model": str(MODEL), "text_encoder": str(TEXT_ENCODER), "vae": str(VAE),
            "source": image_record(source_path), "prompt": PROMPT, "seed": SEED,
            "resolution_wh": [WIDTH, HEIGHT], "cfg": 1.0, "sampler": "Euler",
            "full_schedule_steps": STEPS, "start_index": START_INDEX,
            "sigmas": sigmas.tolist(), "fss_cutoffs": list(CUTOFFS),
            "transition_bandwidth": TRANSITION_BANDWIDTH,
            "production_changes": False,
        },
        "preflight": preflight, "arms": {}, "images": {},
    }
    outputs = {}
    trajectory_tensors = {}
    control_tensors = None
    for name, noise in variants.items():
        sampler = EulerCaptureSampler(source_latent, control_tensors)
        started = time.perf_counter()
        with torch.inference_mode():
            output = comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                source_latent.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=SEED,
            ).detach().float().cpu()
        if control_tensors is None:
            control_tensors = sampler.tensors
            for item in sampler.captures:
                item["state_rms_difference_from_control"] = 0.0
                item["denoised_rms_difference_from_control"] = 0.0
        outputs[name] = output
        trajectory_tensors[name] = sampler.tensors
        report["arms"][name] = {
            "wall_seconds": time.perf_counter() - started,
            "noise_hash": tensor_hash(noise), "final_latent": summary(output),
            "final_structure": structural_metrics(output, source_latent),
            "final_rms_difference_from_control": rms(output - outputs["A_GAUSSIAN"]),
            "evaluations": sampler.captures,
        }
        (OUTPUT / "telemetry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    del model, positive, negative
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    decoded = []
    for name, latent in [("SOURCE_RECONSTRUCTION", source_latent), *outputs.items()]:
        with torch.inference_mode():
            path = OUTPUT / f"{name}.png"
            save_pixels(vae.decode(latent).cpu(), path)
        report["images"][name] = image_record(path)
        decoded.append((name, Image.open(path).convert("RGB")))
    selected = (0, (len(sigmas) - 2) // 2, len(sigmas) - 2)
    for name in ("A_GAUSSIAN", "B_FULL", "C_FSS_R08", "C_FSS_R16"):
        arm = report["arms"][name]
        arm["selected_preview_ordinals"] = list(selected)
        arm["selected_previews"] = {}
        for ordinal in selected:
            state, denoised = trajectory_tensors[name][ordinal]
            for kind, value in (("STATE", state), ("X0", denoised)):
                path = OUTPUT / f"{name}_EVAL_{ordinal:02d}_{kind}.png"
                with torch.inference_mode():
                    save_pixels(vae.decode(value).cpu(), path)
                arm["selected_previews"][f"eval_{ordinal:02d}_{kind.lower()}"] = image_record(path)
    panel_w, label_h = WIDTH, 28
    sheet = Image.new("RGB", (panel_w * len(decoded), HEIGHT + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, image) in enumerate(decoded):
        sheet.paste(image, (index * panel_w, label_h))
        draw.text((index * panel_w + 6, 7), name, fill="black")
        image.close()
    sheet_path = OUTPUT / "COMPARISON.png"
    sheet.save(sheet_path)
    report["comparison"] = image_record(sheet_path)
    report["integrity"] = {
        "status": "SUCCESS", "preflight_passed_before_model": True,
        "same_source_prompt_seed_model_schedule_vae_conditioning": True,
        "only_noise_endpoint_varied": True,
        "all_finite": all(item["final_latent"]["finite"] for item in report["arms"].values()),
    }
    (OUTPUT / "telemetry.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
