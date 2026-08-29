"""FLUX.2 coarse-global/local-fusion falsification experiment.

This is an experiment-only native ComfyUI Euler harness. It intentionally does
not provide nodes, caches, selected-token execution, or a reusable engine.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
MODEL_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\diffusion_models"
    r"\f2k4_realrebelai_Rebels_w4a8s\Flux2-Klein-4B-w4a8.safetensors"
)
TEXT_ENCODER_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\text_encoders"
    r"\f2k4_Comfy-Org_vae-text-encorder-for-flux-klein-4b_text_encoders"
    r"\qwen_3_4b_fp4_flux2.safetensors"
)
VAE_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\vae"
    r"\Comfy-Org_vae-text-encorder-for-flux-klein-9b_vae\flux2-vae.safetensors"
)

sys.path.insert(0, str(COMFY_ROOT))

import comfy.model_management  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
from comfy_extras.nodes_flux import get_schedule  # noqa: E402


PROMPT = (
    "A cinematic wide-angle photograph of one enormous red suspension bridge "
    "forming a single continuous horizontal structure from the far left edge to "
    "the far right edge, crossing behind one yellow vintage train centered on the "
    "bridge; exactly one white lighthouse stands on the left shore and exactly one "
    "dark stone tower stands on the right shore, asymmetric composition, continuous "
    "bridge cables and deck, coherent perspective, sunset light"
)
GIB = 1024**3


def parse_strengths(value: str) -> tuple[float, ...]:
    strengths = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not strengths or any(not 0.0 <= item <= 1.0 for item in strengths):
        raise argparse.ArgumentTypeError("Correction strengths must be comma-separated values in [0,1].")
    if len(set(strengths)) != len(strengths):
        raise argparse.ArgumentTypeError("Correction strengths must be unique.")
    return strengths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments" / "flux2_coarse_global_local_results")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--global-width", type=int, default=512)
    parser.add_argument("--global-height", type=int, default=256)
    parser.add_argument("--crop-width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128, help="Pixel overlap in each spatial dimension.")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--correction-strengths", type=parse_strengths, default=(0.25, 0.5, 0.75, 1.0))
    parser.add_argument("--dry-run", action="store_true", help="Validate geometry/fusion identities without loading models.")
    args = parser.parse_args()
    dimensions = (args.width, args.height, args.global_width, args.global_height, args.crop_width, args.crop_height)
    if any(value <= 0 or value % 16 for value in dimensions):
        parser.error("All image dimensions must be positive multiples of 16.")
    if args.global_width > args.width or args.global_height > args.height:
        parser.error("Global dimensions must not exceed target dimensions.")
    if args.crop_width > args.width or args.crop_height > args.height:
        parser.error("Crop dimensions must not exceed target dimensions.")
    if args.overlap < 0 or args.overlap >= min(args.crop_width, args.crop_height):
        parser.error("Overlap must be nonnegative and smaller than both crop dimensions.")
    if args.overlap % 16:
        parser.error("Overlap must be divisible by 16.")
    if args.steps < 2:
        parser.error("At least two Euler steps are required.")
    return args


def positions(length: int, window: int, overlap: int) -> list[int]:
    if window == length:
        return [0]
    stride = window - overlap
    result = list(range(0, length - window + 1, stride))
    last = length - window
    if result[-1] != last:
        result.append(last)
    return result


@dataclass(frozen=True)
class Crop:
    index: int
    y: int
    x: int
    height: int
    width: int

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def x2(self) -> int:
        return self.x + self.width


def crops_for_canvas(height: int, width: int, crop_height: int, crop_width: int, overlap: int) -> list[Crop]:
    ys = positions(height, crop_height, overlap)
    xs = positions(width, crop_width, overlap)
    return [
        Crop(index, y, x, crop_height, crop_width)
        for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
    ]


def axis_weight(start: int, end: int, length: int, starts: list[int], ends: list[int], device: torch.device) -> torch.Tensor:
    weight = torch.ones(end - start, dtype=torch.float32, device=device)
    previous_ends = [value for value in ends if value > start and value < end]
    next_starts = [value for value in starts if value > start and value < end]
    if previous_ends:
        ramp = min(previous_ends) - start
        if ramp > 0:
            weight[:ramp] = torch.linspace(0.0, 1.0, ramp + 2, device=device)[1:-1]
    if next_starts:
        ramp = end - max(next_starts)
        if ramp > 0:
            weight[-ramp:] = torch.linspace(1.0, 0.0, ramp + 2, device=device)[1:-1]
    return weight


def crop_weight(crop: Crop, all_crops: list[Crop], device: torch.device) -> torch.Tensor:
    row = [item for item in all_crops if item.y == crop.y]
    column = [item for item in all_crops if item.x == crop.x]
    wy = axis_weight(crop.y, crop.y2, max(item.y2 for item in all_crops), [item.y for item in column], [item.y2 for item in column], device)
    wx = axis_weight(crop.x, crop.x2, max(item.x2 for item in all_crops), [item.x for item in row], [item.x2 for item in row], device)
    return wy[:, None] * wx[None, :]


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()))


def stats(value: torch.Tensor) -> dict[str, Any]:
    data = value.float()
    return {
        "shape": list(value.shape),
        "norm": norm(data),
        "rms": float(data.square().mean().sqrt()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "min": float(data.min()),
        "max": float(data.max()),
        "finite": bool(data.isfinite().all()),
    }


def clone_options(options: dict[str, Any], rope: dict[str, float]) -> dict[str, Any]:
    copied = options.copy()
    transformer = options.get("transformer_options", {}).copy()
    copied["transformer_options"] = transformer
    transformer["rope_options"] = rope
    return copied


def rope_for_crop(crop: Crop) -> dict[str, float]:
    return {"shift_y": float(crop.y), "shift_x": float(crop.x)}


def rope_for_global(target_h: int, target_w: int, global_h: int, global_w: int) -> dict[str, float]:
    return {
        "scale_y": 1.0 if global_h == 1 else (target_h - 1.0) / (global_h - 1.0),
        "scale_x": 1.0 if global_w == 1 else (target_w - 1.0) / (global_w - 1.0),
    }


@dataclass
class VariantTrace:
    name: str
    correction_strength: float | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    intermediates: dict[int, torch.Tensor] = field(default_factory=dict)
    correction_maps: dict[int, torch.Tensor] = field(default_factory=dict)
    seconds: float = 0.0
    peak_allocated_gib: float | None = None
    peak_increment_gib: float | None = None


class BlueprintEulerSampler(comfy.samplers.Sampler):
    def __init__(
        self,
        variant: str,
        target_hw: tuple[int, int],
        global_hw: tuple[int, int],
        crops: list[Crop],
        trace: VariantTrace,
        seed: int,
        correction_strength: float | None = None,
    ) -> None:
        self.variant = variant
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crops = crops
        self.trace = trace
        self.seed = seed
        self.correction_strength = correction_strength

    def predict(self, model, x: torch.Tensor, sigma: torch.Tensor, model_options: dict[str, Any], step: int) -> torch.Tensor:
        target_h, target_w = self.target_hw
        global_h, global_w = self.global_hw
        sigma_batch = sigma.expand(x.shape[0])
        calls: list[dict[str, Any]] = []

        def evaluate(value: torch.Tensor, role: str, rope: dict[str, float], geometry: dict[str, Any]) -> torch.Tensor:
            started = time.perf_counter()
            prediction = model(value, sigma_batch, model_options=clone_options(model_options, rope), seed=self.seed)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            calls.append({
                "role": role,
                "input_latent_hw": list(value.shape[-2:]),
                "image_tokens": int(value.shape[-2] * value.shape[-1]),
                "rope_options": rope,
                "geometry": geometry,
                "prediction": stats(prediction),
                "seconds": time.perf_counter() - started,
            })
            return prediction

        mapped_global = None
        global_prediction = None
        if self.variant in ("global", "fused"):
            global_input = F.interpolate(x, size=self.global_hw, mode="bilinear", align_corners=False)
            global_prediction = evaluate(
                global_input,
                "global",
                rope_for_global(target_h, target_w, global_h, global_w),
                {"target_extent_hw": [target_h, target_w]},
            )
            mapped_global = F.interpolate(global_prediction, size=self.target_hw, mode="bilinear", align_corners=False)

        if self.variant == "dense":
            assembled = evaluate(x, "dense", {}, {"y": 0, "x": 0, "height": target_h, "width": target_w})
            coverage = torch.ones_like(x[:, :1])
            correction = None
        elif self.variant == "global":
            assembled = mapped_global
            coverage = torch.ones_like(x[:, :1])
            correction = None
        else:
            accumulation = torch.zeros_like(x)
            coverage = torch.zeros_like(x[:, :1])
            correction_accumulation = torch.zeros_like(x)
            for crop in self.crops:
                local = evaluate(
                    x[:, :, crop.y:crop.y2, crop.x:crop.x2],
                    "local",
                    rope_for_crop(crop),
                    {"index": crop.index, "y": crop.y, "x": crop.x, "height": crop.height, "width": crop.width},
                )
                weight = crop_weight(crop, self.crops, x.device).unsqueeze(0).unsqueeze(0)
                accumulation[:, :, crop.y:crop.y2, crop.x:crop.x2] += local * weight
                coverage[:, :, crop.y:crop.y2, crop.x:crop.x2] += weight
                if self.variant == "fused":
                    global_region = mapped_global[:, :, crop.y:crop.y2, crop.x:crop.x2]
                    correction_accumulation[:, :, crop.y:crop.y2, crop.x:crop.x2] += (local - global_region) * weight
            if float(coverage.min()) <= 0:
                raise RuntimeError(f"Incomplete assembled prediction coverage: min={float(coverage.min())}.")
            tiled = accumulation / coverage
            if self.variant == "tiled":
                assembled = tiled
                correction = None
            else:
                correction = correction_accumulation / coverage
                assembled = mapped_global + float(self.correction_strength) * correction

        if not bool(assembled.isfinite().all()):
            raise FloatingPointError(f"Nonfinite assembled prediction for {self.trace.name} step {step}.")
        record: dict[str, Any] = {
            "step": step,
            "sigma": float(sigma),
            "evaluation_identity": f"{self.trace.name}:euler:{step}:sigma={float(sigma):.9g}",
            "model_forwards": len(calls),
            "calls": calls,
            "executed_image_tokens": sum(call["image_tokens"] for call in calls),
            "assembled_prediction": stats(assembled),
            "coverage": stats(coverage),
        }
        if global_prediction is not None:
            record["global_prediction"] = stats(global_prediction)
            record["mapped_global_prediction"] = stats(mapped_global)
        if correction is not None:
            record["correction"] = stats(correction)
            record["correction_to_global_norm_ratio"] = norm(correction) / max(norm(mapped_global), 1e-12)
            self.trace.correction_maps[step] = correction.detach().float().square().mean(dim=1, keepdim=True).sqrt().cpu()
        self.trace.evaluations.append(record)
        self.trace.intermediates[step] = assembled.detach().float().cpu()
        return assembled

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("This T2I falsification experiment does not use a denoise mask.")
        x = noise
        total = len(sigmas) - 1
        for step in range(total):
            sigma = sigmas[step]
            denoised = self.predict(model, x, sigma, extra_args["model_options"], step)
            derivative = (x - denoised) / sigma
            x = x + derivative * (sigmas[step + 1] - sigma)
            if callback is not None:
                callback(step, denoised, x, total)
        return x


def save_pixels(pixels: torch.Tensor, path: Path) -> None:
    array = pixels.detach().clamp(0, 1).mul(255).round().to(torch.uint8)[0].cpu().numpy()
    Image.fromarray(array).save(path)


def save_heatmap(magnitude: torch.Tensor, path: Path, output_hw: tuple[int, int]) -> None:
    value = F.interpolate(magnitude.float(), size=output_hw, mode="bilinear", align_corners=False)
    scale = torch.quantile(value.flatten(), 0.99).clamp_min(1e-12)
    value = (value / scale).clamp(0, 1)[0, 0]
    red = value
    green = (1.0 - (2.0 * value - 1.0).abs()).clamp(0, 1)
    blue = 1.0 - value
    rgb = torch.stack((red, green, blue), dim=-1).mul(255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(rgb).save(path)


def low_high_metrics(value: torch.Tensor, global_only: torch.Tensor) -> dict[str, float]:
    low_value = F.interpolate(F.interpolate(value.float(), scale_factor=0.25, mode="bilinear", align_corners=False), size=value.shape[-2:], mode="bilinear", align_corners=False)
    low_global = F.interpolate(F.interpolate(global_only.float(), scale_factor=0.25, mode="bilinear", align_corners=False), size=value.shape[-2:], mode="bilinear", align_corners=False)
    high = value.float() - low_value
    global_high = global_only.float() - low_global
    return {
        "low_frequency_mae_vs_global_only": float((low_value - low_global).abs().mean()),
        "high_frequency_rms": float(high.square().mean().sqrt()),
        "high_frequency_gain_over_global_only": float(high.square().mean().sqrt() - global_high.square().mean().sqrt()),
    }


def dry_run(args: argparse.Namespace) -> None:
    target_h, target_w = args.height // 16, args.width // 16
    crop_h, crop_w = args.crop_height // 16, args.crop_width // 16
    overlap = args.overlap // 16
    crops = crops_for_canvas(target_h, target_w, crop_h, crop_w, overlap)
    canvas = torch.zeros(1, 2, target_h, target_w)
    coverage = torch.zeros_like(canvas[:, :1])
    local_accum = torch.zeros_like(canvas)
    global_value = torch.randn_like(canvas)
    for crop in crops:
        weight = crop_weight(crop, crops, canvas.device)[None, None]
        local = torch.randn_like(canvas[:, :, crop.y:crop.y2, crop.x:crop.x2])
        local_accum[:, :, crop.y:crop.y2, crop.x:crop.x2] += local * weight
        coverage[:, :, crop.y:crop.y2, crop.x:crop.x2] += weight
    tiled = local_accum / coverage
    correction = tiled - global_value
    alpha_one = global_value + correction
    report = {
        "target_latent_hw": [target_h, target_w],
        "crops": [crop.__dict__ for crop in crops],
        "coverage": stats(coverage),
        "alpha_one_equals_tiled": bool(torch.equal(alpha_one, tiled)),
        "alpha_one_max_abs": float((alpha_one - tiled).abs().max()),
        "global_rope": rope_for_global(target_h, target_w, args.global_height // 16, args.global_width // 16),
    }
    print(json.dumps(report, indent=2))
    if float(coverage.min()) <= 0 or report["alpha_one_max_abs"] > 1e-6:
        raise AssertionError("Dry-run geometry/fusion validation failed.")


def run_variant(model, noise, latent_image, conditioning, negative, sigmas, sampler: BlueprintEulerSampler, seed: int) -> torch.Tensor:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    else:
        before = 0
    started = time.perf_counter()
    output = comfy.sample.sample_custom(
        model,
        noise.clone(),
        1.0,
        sampler,
        sigmas.clone(),
        conditioning,
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
    return output


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run(args)
        return
    for path in (MODEL_PATH, TEXT_ENCODER_PATH, VAE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_hw = (args.height // 16, args.width // 16)
    global_hw = (args.global_height // 16, args.global_width // 16)
    crop_hw = (args.crop_height // 16, args.crop_width // 16)
    overlap = args.overlap // 16
    crops = crops_for_canvas(*target_hw, *crop_hw, overlap)
    target_tokens = math.prod(target_hw)

    model = comfy.sd.load_diffusion_model(str(MODEL_PATH), model_options={})
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE_PATH), safe_load=True))
    clip = comfy.sd.load_clip([str(TEXT_ENCODER_PATH)], clip_type=comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(args.prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    latent_image = torch.zeros((1, 128, *target_hw), dtype=torch.float32)
    noise = torch.randn(latent_image.shape, generator=torch.Generator().manual_seed(args.seed))
    sigmas = get_schedule(args.steps, target_tokens).float().clone()

    variants: list[tuple[str, str, float | None]] = [
        ("A_DENSE", "dense", None),
        ("B_TILED_ONLY", "tiled", None),
        ("C_GLOBAL_ONLY", "global", None),
    ]
    variants.extend((f"D_GLOBAL_LOCAL_alpha_{strength:g}", "fused", strength) for strength in args.correction_strengths)

    outputs: dict[str, torch.Tensor] = {}
    traces: dict[str, VariantTrace] = {}
    for name, kind, strength in variants:
        print(f"Running {name}...", flush=True)
        trace = VariantTrace(name, strength)
        sampler = BlueprintEulerSampler(kind, target_hw, global_hw, crops, trace, args.seed, strength)
        with torch.inference_mode():
            outputs[name] = run_variant(model, noise, latent_image, positive, negative, sigmas, sampler, args.seed).detach().cpu()
        traces[name] = trace

    # Alpha=1 must reproduce tiled-only at every evaluation up to floating-point
    # accumulation order. This is an algebraic control, not a semantic variant.
    alpha_one_name = next((name for name in outputs if name.endswith("alpha_1")), None)
    alpha_control = None
    if alpha_one_name is not None:
        difference = outputs[alpha_one_name].float() - outputs["B_TILED_ONLY"].float()
        alpha_control = {"max_abs": float(difference.abs().max()), "mean_abs": float(difference.abs().mean())}

    # Decode final and selected intermediate denoised estimates after sampling so
    # VAE allocations do not contaminate sampling peak-memory measurements.
    selected_steps = sorted(set((0, max(0, args.steps // 2), args.steps - 1)))
    decoded_outputs: dict[str, torch.Tensor] = {}
    for name, latent in outputs.items():
        print(f"Decoding {name}...", flush=True)
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        decoded_outputs[name] = pixels
        save_pixels(pixels, args.output_dir / f"{name}.png")
        for step in selected_steps:
            estimate = traces[name].intermediates[step]
            with torch.inference_mode():
                estimate_pixels = vae.decode(estimate).cpu()
            save_pixels(estimate_pixels, args.output_dir / f"{name}_step_{step:02d}_estimate.png")
        for step, magnitude in traces[name].correction_maps.items():
            if step in selected_steps:
                save_heatmap(magnitude, args.output_dir / f"{name}_step_{step:02d}_correction_magnitude.png", (args.height, args.width))

    global_output = outputs["C_GLOBAL_ONLY"]
    comparisons = {name: low_high_metrics(value, global_output) for name, value in outputs.items()}
    report = {
        "configuration": {
            "model": str(MODEL_PATH),
            "text_encoder": str(TEXT_ENCODER_PATH),
            "vae": str(VAE_PATH),
            "prompt": args.prompt,
            "seed": args.seed,
            "sampler": "native-compatible Euler, zero churn, one model evaluation per accepted step",
            "cfg": 1.0,
            "steps": args.steps,
            "sigmas": sigmas.tolist(),
            "target_image_hw": [args.height, args.width],
            "target_latent_hw": list(target_hw),
            "target_image_tokens": target_tokens,
            "global_image_hw": [args.global_height, args.global_width],
            "global_latent_hw": list(global_hw),
            "global_image_tokens": math.prod(global_hw),
            "global_rope_options": rope_for_global(*target_hw, *global_hw),
            "crop_image_hw": [args.crop_height, args.crop_width],
            "crop_latent_hw": list(crop_hw),
            "crop_image_tokens": math.prod(crop_hw),
            "overlap_pixels": args.overlap,
            "overlap_latent_tokens": overlap,
            "crops": [crop.__dict__ for crop in crops],
            "correction_strengths": list(args.correction_strengths),
            "fusion": "mapped_global + alpha * normalized_overlap_sum(local - mapped_global_crop)",
        },
        "alpha_one_control_vs_tiled": alpha_control,
        "comparisons": comparisons,
        "variants": {
            name: {
                "seconds": trace.seconds,
                "peak_allocated_gib": trace.peak_allocated_gib,
                "peak_increment_gib": trace.peak_increment_gib,
                "correction_strength": trace.correction_strength,
                "final_latent": stats(outputs[name]),
                "final_image": str(args.output_dir / f"{name}.png"),
                "evaluations": trace.evaluations,
            }
            for name, trace in traces.items()
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output_dir / "report.json"),
        "alpha_one_control_vs_tiled": alpha_control,
        "seconds": {name: trace.seconds for name, trace in traces.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
