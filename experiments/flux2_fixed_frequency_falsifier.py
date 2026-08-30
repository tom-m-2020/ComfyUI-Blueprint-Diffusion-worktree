"""Fixed-frequency local-correction falsifier for FLUX.2 Klein.

Reuses the exact Phase 2 model, prompt, seed, geometry, conditioning, and Euler
lifecycle. This is an experiment-only harness, not reusable production code.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import flux2_coarse_global_local_falsification as phase2


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_RESULTS = ROOT / "experiments" / "flux2_coarse_global_local_results"
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_fixed_frequency_results",
    )
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--sigmas", default="1.0,2.0", help="Two fixed Gaussian sigmas in latent-token units.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.sigmas = tuple(float(item.strip()) for item in args.sigmas.split(",") if item.strip())
    if not 0.0 < args.alpha <= 1.0:
        parser.error("--alpha must be in (0,1].")
    if not 1 <= len(args.sigmas) <= 2 or any(value <= 0 for value in args.sigmas):
        parser.error("--sigmas must contain one or two positive values.")
    return args


def gaussian_kernel(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = math.ceil(3.0 * sigma)
    positions = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (positions / sigma).square())
    kernel /= kernel.sum()
    kernel_2d = kernel[:, None] * kernel[None, :]
    return kernel_2d.to(dtype=dtype)


def low_pass(value: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(sigma, value.device, value.dtype)
    radius = kernel.shape[-1] // 2
    if radius >= min(value.shape[-2:]):
        raise ValueError(f"Gaussian radius {radius} is too large for correction {tuple(value.shape[-2:])}.")
    padded = F.pad(value, (radius, radius, radius, radius), mode="reflect")
    weights = kernel[None, None].expand(value.shape[1], 1, -1, -1)
    return F.conv2d(padded, weights, groups=value.shape[1])


@dataclass
class FrequencyTrace(phase2.VariantTrace):
    filter_sigma: float | None = None
    low_components: dict[int, torch.Tensor] = field(default_factory=dict)
    high_components: dict[int, torch.Tensor] = field(default_factory=dict)
    mapped_globals: dict[int, torch.Tensor] = field(default_factory=dict)


class FrequencyEulerSampler(phase2.comfy.samplers.Sampler):
    def __init__(
        self,
        mode: str,
        target_hw: tuple[int, int],
        global_hw: tuple[int, int],
        crops: list[phase2.Crop],
        trace: FrequencyTrace,
        seed: int,
        alpha: float,
        filter_sigma: float | None = None,
    ) -> None:
        self.mode = mode
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crops = crops
        self.trace = trace
        self.seed = seed
        self.alpha = alpha
        self.filter_sigma = filter_sigma

    def predict(
        self,
        model,
        x: torch.Tensor,
        sigma: torch.Tensor,
        model_options: dict[str, Any],
        step: int,
    ) -> torch.Tensor:
        target_h, target_w = self.target_hw
        global_h, global_w = self.global_hw
        sigma_batch = sigma.expand(x.shape[0])
        calls: list[dict[str, Any]] = []

        def evaluate(
            value: torch.Tensor,
            role: str,
            rope: dict[str, float],
            geometry: dict[str, Any],
        ) -> torch.Tensor:
            started = time.perf_counter()
            prediction = model(
                value,
                sigma_batch,
                model_options=phase2.clone_options(model_options, rope),
                seed=self.seed,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            calls.append(
                {
                    "role": role,
                    "input_latent_hw": list(value.shape[-2:]),
                    "image_tokens": int(value.shape[-2] * value.shape[-1]),
                    "rope_options": rope,
                    "geometry": geometry,
                    "prediction": phase2.stats(prediction),
                    "seconds": time.perf_counter() - started,
                }
            )
            return prediction

        global_input = F.interpolate(
            x, size=self.global_hw, mode="bilinear", align_corners=False
        )
        global_prediction = evaluate(
            global_input,
            "global",
            phase2.rope_for_global(target_h, target_w, global_h, global_w),
            {"target_extent_hw": [target_h, target_w]},
        )
        mapped_global = F.interpolate(
            global_prediction,
            size=self.target_hw,
            mode="bilinear",
            align_corners=False,
        )
        coverage = torch.ones_like(x[:, :1])
        low_component = None
        high_component = None

        if self.mode == "global":
            assembled = mapped_global
        else:
            coverage = torch.zeros_like(x[:, :1])
            residual_accumulation = torch.zeros_like(x)
            low_accumulation = torch.zeros_like(x)
            high_accumulation = torch.zeros_like(x)
            for crop in self.crops:
                local = evaluate(
                    x[:, :, crop.y : crop.y2, crop.x : crop.x2],
                    "local",
                    phase2.rope_for_crop(crop),
                    {
                        "index": crop.index,
                        "y": crop.y,
                        "x": crop.x,
                        "height": crop.height,
                        "width": crop.width,
                    },
                )
                global_region = mapped_global[
                    :, :, crop.y : crop.y2, crop.x : crop.x2
                ]
                residual = local - global_region
                weight = phase2.crop_weight(crop, self.crops, x.device)[None, None]
                coverage[:, :, crop.y : crop.y2, crop.x : crop.x2] += weight
                residual_accumulation[
                    :, :, crop.y : crop.y2, crop.x : crop.x2
                ] += residual * weight
                if self.mode == "highpass":
                    residual_low = low_pass(residual, float(self.filter_sigma))
                    residual_high = residual - residual_low
                    low_accumulation[
                        :, :, crop.y : crop.y2, crop.x : crop.x2
                    ] += residual_low * weight
                    high_accumulation[
                        :, :, crop.y : crop.y2, crop.x : crop.x2
                    ] += residual_high * weight

            if float(coverage.min()) <= 0:
                raise RuntimeError(
                    f"Incomplete assembled prediction coverage: {float(coverage.min())}."
                )
            residual = residual_accumulation / coverage
            if self.mode == "unfiltered":
                assembled = mapped_global + self.alpha * residual
            else:
                low_component = low_accumulation / coverage
                high_component = high_accumulation / coverage
                reconstruction_error = phase2.stats(
                    residual - (low_component + high_component)
                )
                if reconstruction_error["max"] > 2e-5 or reconstruction_error["min"] < -2e-5:
                    raise AssertionError(
                        f"Low/high reconstruction failed at step {step}: {reconstruction_error}."
                    )
                assembled = mapped_global + self.alpha * high_component

        if not bool(assembled.isfinite().all()):
            raise FloatingPointError(
                f"Nonfinite assembled prediction for {self.trace.name} step {step}."
            )
        record: dict[str, Any] = {
            "step": step,
            "sigma": float(sigma),
            "evaluation_identity": (
                f"{self.trace.name}:euler:{step}:sigma={float(sigma):.9g}"
            ),
            "model_forwards": len(calls),
            "calls": calls,
            "executed_image_tokens": sum(call["image_tokens"] for call in calls),
            "global_prediction": phase2.stats(global_prediction),
            "mapped_global_prediction": phase2.stats(mapped_global),
            "assembled_prediction": phase2.stats(assembled),
            "coverage": phase2.stats(coverage),
        }
        if low_component is not None and high_component is not None:
            low_norm = phase2.norm(low_component)
            high_norm = phase2.norm(high_component)
            record.update(
                {
                    "filter_sigma_latent_tokens": self.filter_sigma,
                    "filter_kernel_size": int(
                        gaussian_kernel(
                            float(self.filter_sigma), x.device, x.dtype
                        ).shape[-1]
                    ),
                    "correction_low": phase2.stats(low_component),
                    "correction_high": phase2.stats(high_component),
                    "low_to_high_norm_ratio": low_norm / max(high_norm, 1e-12),
                    "low_fraction_of_split_norm_sum": low_norm
                    / max(low_norm + high_norm, 1e-12),
                    "high_fraction_of_split_norm_sum": high_norm
                    / max(low_norm + high_norm, 1e-12),
                }
            )
            self.trace.low_components[step] = low_component.detach().float().cpu()
            self.trace.high_components[step] = high_component.detach().float().cpu()
            self.trace.mapped_globals[step] = mapped_global.detach().float().cpu()
        self.trace.evaluations.append(record)
        self.trace.intermediates[step] = assembled.detach().float().cpu()
        return assembled

    def sample(
        self,
        model,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if denoise_mask is not None:
            raise ValueError("This T2I frequency falsifier does not use a mask.")
        x = noise
        total = len(sigmas) - 1
        for step in range(total):
            sigma = sigmas[step]
            denoised = self.predict(
                model, x, sigma, extra_args["model_options"], step
            )
            derivative = (x - denoised) / sigma
            x = x + derivative * (sigmas[step + 1] - sigma)
            if callback is not None:
                callback(step, denoised, x, total)
        return x


def run_variant(
    model,
    noise,
    latent_image,
    positive,
    negative,
    sigmas,
    sampler: FrequencyEulerSampler,
    seed: int,
) -> torch.Tensor:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    else:
        before = 0
    started = time.perf_counter()
    output = phase2.comfy.sample.sample_custom(
        model,
        noise.clone(),
        1.0,
        sampler,
        sigmas.clone(),
        positive,
        negative,
        latent_image.clone(),
        disable_pbar=True,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        sampler.trace.peak_allocated_gib = peak / GIB
        sampler.trace.peak_increment_gib = (peak - before) / GIB
    sampler.trace.seconds = time.perf_counter() - started
    return output.detach().cpu()


def dry_run(args: argparse.Namespace) -> None:
    torch.manual_seed(20260830)
    value = torch.randn(1, 4, 32, 32)
    checks = []
    for sigma in args.sigmas:
        low = low_pass(value, sigma)
        high = value - low
        checks.append(
            {
                "sigma": sigma,
                "kernel_size": gaussian_kernel(
                    sigma, value.device, value.dtype
                ).shape[-1],
                "low": phase2.stats(low),
                "high": phase2.stats(high),
                "reconstruction_max_abs": float((value - low - high).abs().max()),
            }
        )
    print(json.dumps(checks, indent=2))
    if any(check["reconstruction_max_abs"] > 1e-6 for check in checks):
        raise AssertionError("Frequency split dry run failed.")


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    for path in (
        phase2.MODEL_PATH,
        phase2.TEXT_ENCODER_PATH,
        phase2.VAE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    global_width, global_height = 512, 256
    crop_width = crop_height = 512
    overlap_pixels = 128
    steps = 4
    seed = 20260829
    target_hw = (height // 16, width // 16)
    global_hw = (global_height // 16, global_width // 16)
    crop_hw = (crop_height // 16, crop_width // 16)
    overlap = overlap_pixels // 16
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap)

    model = phase2.comfy.sd.load_diffusion_model(
        str(phase2.MODEL_PATH), model_options={}
    )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(
            str(phase2.VAE_PATH), safe_load=True
        )
    )
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)],
        clip_type=phase2.comfy.sd.CLIPType.FLUX2,
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase2.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    latent_image = torch.zeros((1, 128, *target_hw), dtype=torch.float32)
    noise = torch.randn(
        latent_image.shape, generator=torch.Generator().manual_seed(seed)
    )
    sigmas = phase2.get_schedule(steps, math.prod(target_hw)).float().clone()
    variants: list[tuple[str, str, float | None]] = [
        ("A_GLOBAL_ONLY", "global", None),
        (f"B_UNFILTERED_alpha_{args.alpha:g}", "unfiltered", None),
    ]
    variants.extend(
        (
            f"C_HIGHPASS_sigma_{sigma:g}_alpha_{args.alpha:g}",
            "highpass",
            sigma,
        )
        for sigma in args.sigmas
    )

    outputs: dict[str, torch.Tensor] = {}
    traces: dict[str, FrequencyTrace] = {}
    for name, mode, filter_sigma in variants:
        print(f"Running {name}...", flush=True)
        trace = FrequencyTrace(
            name=name,
            correction_strength=args.alpha if mode != "global" else None,
            filter_sigma=filter_sigma,
        )
        sampler = FrequencyEulerSampler(
            mode,
            target_hw,
            global_hw,
            crops,
            trace,
            seed,
            args.alpha,
            filter_sigma,
        )
        with torch.inference_mode():
            outputs[name] = run_variant(
                model,
                noise,
                latent_image,
                positive,
                negative,
                sigmas,
                sampler,
                seed,
            )
        traces[name] = trace

    selected_steps = (0, 2, 3)
    for name, latent in outputs.items():
        print(f"Decoding {name}...", flush=True)
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")
        for step in selected_steps:
            with torch.inference_mode():
                estimate = vae.decode(traces[name].intermediates[step]).cpu()
            phase2.save_pixels(
                estimate, args.output_dir / f"{name}_step_{step:02d}_estimate.png"
            )

        if traces[name].filter_sigma is not None:
            for step in selected_steps:
                low = traces[name].low_components[step]
                high = traces[name].high_components[step]
                mapped = traces[name].mapped_globals[step]
                with torch.inference_mode():
                    decoded_low = vae.decode(low).cpu()
                    decoded_high = vae.decode(high).cpu()
                    decoded_global_plus_low = vae.decode(
                        mapped + args.alpha * low
                    ).cpu()
                phase2.save_pixels(
                    decoded_low,
                    args.output_dir / f"{name}_step_{step:02d}_r_low_decoded.png",
                )
                phase2.save_pixels(
                    decoded_high,
                    args.output_dir / f"{name}_step_{step:02d}_r_high_decoded.png",
                )
                phase2.save_pixels(
                    decoded_global_plus_low,
                    args.output_dir
                    / f"{name}_step_{step:02d}_global_plus_low_decoded.png",
                )
                phase2.save_heatmap(
                    low.square().mean(dim=1, keepdim=True).sqrt(),
                    args.output_dir / f"{name}_step_{step:02d}_r_low_magnitude.png",
                    (height, width),
                )
                phase2.save_heatmap(
                    high.square().mean(dim=1, keepdim=True).sqrt(),
                    args.output_dir / f"{name}_step_{step:02d}_r_high_magnitude.png",
                    (height, width),
                )

    global_output = outputs["A_GLOBAL_ONLY"]
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": phase2.PROMPT,
            "seed": seed,
            "sampler": "Euler, zero churn, one assembled prediction per accepted step",
            "cfg": 1.0,
            "steps": steps,
            "sigmas": sigmas.tolist(),
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "global_image_hw": [global_height, global_width],
            "global_latent_hw": list(global_hw),
            "crop_image_hw": [crop_height, crop_width],
            "crop_latent_hw": list(crop_hw),
            "overlap_pixels": overlap_pixels,
            "crops": [crop.__dict__ for crop in crops],
            "alpha": args.alpha,
            "filter_sigmas_latent_tokens": list(args.sigmas),
            "fusion": "mapped_global + alpha * (residual - GaussianLowPass(residual))",
            "previous_tiled_only": str(PREVIOUS_RESULTS / "B_TILED_ONLY.png"),
        },
        "comparisons": {
            name: phase2.low_high_metrics(value, global_output)
            for name, value in outputs.items()
        },
        "variants": {
            name: {
                "seconds": trace.seconds,
                "peak_allocated_gib": trace.peak_allocated_gib,
                "peak_increment_gib": trace.peak_increment_gib,
                "correction_strength": trace.correction_strength,
                "filter_sigma": trace.filter_sigma,
                "final_latent": phase2.stats(outputs[name]),
                "final_image": str(args.output_dir / f"{name}.png"),
                "evaluations": trace.evaluations,
            }
            for name, trace in traces.items()
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "seconds": {
                    name: trace.seconds for name, trace in traces.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
