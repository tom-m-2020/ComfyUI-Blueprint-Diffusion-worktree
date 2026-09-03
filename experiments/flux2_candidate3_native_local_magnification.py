"""Phase 9b: native-resolution local magnification falsifier."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_bounded_global_density as phase8b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_normalized_local_working_canvas as phase9
import blueprint_diffusion.sampling.euler as production_euler
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_local_magnification_results"
REPORT = OUTPUT / "report.json"
H_HW = (64, 128)
WORK_HW = (64, 64)
STEPS = 4
SEED = 20260901
PROMPT = phase9.PROMPT
VARIANTS = {
    "A_DESTINATION_SCALE": "direct",
    "B_NAIVE_MAGNIFIED": "naive",
    "C_SIGMA_CONSISTENT_MAGNIFIED": "sigma_consistent",
}


class DestinationPlanner(FixedCropPlanner):
    QUALIFIED_GEOMETRIES = {}


def tensor_summary(value: torch.Tensor) -> dict:
    work = value.detach().float()
    return {
        "shape": list(value.shape), "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()), "variance": float(work.var(unbiased=False)),
        "max_abs": float(work.abs().max()), "finite": bool(work.isfinite().all()),
    }


class Capture:
    def __init__(self):
        self.values = {}

    def __call__(self, name, ordinal, value):
        if name in {"initial_H", "initial_G", "x0_G", "assembled_x0_H", "accepted_H", "accepted_G"}:
            self.values.setdefault(name, {})[ordinal] = value.detach().float().cpu()


class Probe:
    def __init__(self, sigmas):
        self.sigmas = [float(x) for x in sigmas[:-1]]
        self.calls = []
        self.overlaps = []
        self.representative = {}

    def ordinal(self, sigma) -> int:
        value = float(sigma)
        return min(range(len(self.sigmas)), key=lambda i: abs(self.sigmas[i] - value))

    def timed(self, kind, tokens, fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall = time.perf_counter()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        self.calls.append({
            "kind": kind, "tokens": tokens,
            "cuda_ms": float(start.elapsed_time(end)),
            "wall_seconds": time.perf_counter() - wall,
        })
        return result

    @staticmethod
    def overlap(predictions, regions):
        square_sum = 0.0
        count = 0
        for i, left in enumerate(regions):
            for j in range(i + 1, len(regions)):
                right = regions[j]
                y0, y1 = max(left.y, right.y), min(left.y2, right.y2)
                x0, x1 = max(left.x, right.x), min(left.x2, right.x2)
                if y0 >= y1 or x0 >= x1:
                    continue
                a = predictions[i][..., y0-left.y:y1-left.y, x0-left.x:x1-left.x]
                b = predictions[j][..., y0-right.y:y1-right.y, x0-right.x:x1-right.x]
                delta = a.float() - b.float()
                square_sum += float(delta.square().sum())
                count += delta.numel()
        return math.sqrt(square_sum / count) if count else 0.0


def restrict2(value):
    return F.avg_pool2d(value, 2, 2)


def prolong2(value):
    return F.interpolate(value, scale_factor=2, mode="nearest")


@contextlib.contextmanager
def scoped_variant(mode: str, probe: Probe):
    original_geometry = production_euler.BlockDCTGeometry
    original_planner = production_euler.FixedCropPlanner
    original_validate = Flux2Adapter.validate_run
    original_global = Flux2Adapter.predict_global
    original_region = Flux2Adapter.predict_region
    original_assemble = OverlapAssembler.assemble

    production_euler.BlockDCTGeometry = lambda high_hw: phase8b.ExperimentalBlockDCTGeometry(high_hw, 8, 3)
    production_euler.FixedCropPlanner = DestinationPlanner

    def validate(adapter, **kwargs):
        forwarded = dict(kwargs)
        high_shape = kwargs["high_shape"]
        forwarded["global_shape"] = (
            *kwargs["global_shape"][:-2],
            high_shape[-2] // 4 * 3, high_shape[-1] // 4 * 3,
        )
        return original_validate(adapter, **forwarded)

    def predict_global(adapter, **kwargs):
        return probe.timed(
            "global", kwargs["g"].shape[-2] * kwargs["g"].shape[-1],
            lambda: original_global(adapter, **kwargs),
        )

    def predict_region(adapter, **kwargs):
        h_view = kwargs["h_view"]
        region = kwargs["region"]
        ordinal = probe.ordinal(kwargs["sigma"])
        if mode == "direct":
            return probe.timed(
                "local", 32 * 32, lambda: original_region(adapter, **kwargs)
            )

        if mode == "naive":
            working = F.interpolate(h_view, size=WORK_HW, mode="bilinear", align_corners=True)
        else:
            base = prolong2(h_view)
            generator = torch.Generator(device="cpu").manual_seed(
                SEED + 100_003 * ordinal + 1_009 * region.index
            )
            noise = torch.randn(tuple(base.shape), generator=generator).to(
                device=base.device, dtype=base.dtype
            )
            high = noise - prolong2(restrict2(noise))
            working = base + kwargs["sigma"] * high

        options = adapter._options(kwargs["model_options"], {
            "shift_y": float(region.y), "shift_x": float(region.x),
            "scale_y": (region.height - 1.0) / (WORK_HW[0] - 1.0),
            "scale_x": (region.width - 1.0) / (WORK_HW[1] - 1.0),
        })
        x0_w = probe.timed(
            "local", math.prod(WORK_HW),
            lambda: kwargs["guider"](
                working, kwargs["sigma"].expand(1), model_options=options,
                seed=kwargs["seed"],
            ),
        )
        restricted = restrict2(x0_w)
        coarse_error = restrict2(working).float() - h_view.float()
        high_component = working.float() - prolong2(restrict2(working)).float()
        record = {
            "ordinal": ordinal, "region": region.index,
            "working": tensor_summary(working), "x0_W": tensor_summary(x0_w),
            "restricted_x0": tensor_summary(restricted),
            "working_coarse_max_abs": float(coarse_error.abs().max()),
            "working_coarse_rms": float(coarse_error.square().mean().sqrt()),
            "working_high_rms": float(high_component.square().mean().sqrt()),
        }
        probe.calls[-1]["region"] = region.index
        probe.calls[-1]["ordinal"] = ordinal
        probe.calls[-1]["state"] = record
        if region.index == 7:
            probe.representative[ordinal] = {
                "W": working.detach().float().cpu(),
                "x0_W": x0_w.detach().float().cpu(),
                "restricted_x0": restricted.detach().float().cpu(),
            }
        return restricted

    def assemble(assembler, predictions, regions, target_hw):
        probe.overlaps.append(Probe.overlap(predictions, regions))
        return original_assemble(assembler, predictions, regions, target_hw)

    Flux2Adapter.validate_run = validate
    Flux2Adapter.predict_global = predict_global
    Flux2Adapter.predict_region = predict_region
    OverlapAssembler.assemble = assemble
    try:
        yield
    finally:
        production_euler.BlockDCTGeometry = original_geometry
        production_euler.FixedCropPlanner = original_planner
        Flux2Adapter.validate_run = original_validate
        Flux2Adapter.predict_global = original_global
        Flux2Adapter.predict_region = original_region
        OverlapAssembler.assemble = original_assemble


def decode_artifacts(name, capture, representative, final):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    values = {f"assembled_x0_{i}": v for i, v in capture.values["assembled_x0_H"].items()}
    values["final"] = final.detach().float().cpu()
    for ordinal, tensors in representative.items():
        for key, value in tensors.items():
            values[f"region7_step{ordinal}_{key}"] = value
    images = {}
    for label, value in values.items():
        with torch.inference_mode():
            pixels = vae.decode(value).cpu()
        path = OUTPUT / f"{name}_{label}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images[label] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return images


def run_variant(name):
    mode = VARIANTS[name]
    geometry = phase8b.ExperimentalBlockDCTGeometry(H_HW, 8, 3)
    random_g = torch.randn(1, 3, *geometry.GLOBAL_HW, generator=torch.Generator().manual_seed(9012))
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *H_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(H_HW)).float().clone()
    sigmas[0], sigmas[-1] = 1.0, 0.0
    capture = Capture()
    probe = Probe(sigmas)
    sampler = BlueprintEulerSampler(capture=capture)
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with scoped_variant(mode, probe), torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    regions = DestinationPlanner().plan(H_HW)
    accepted = [x for x in sampler.last_telemetry if x["event"] == "accepted_interval"]
    result = {
        "variant": name,
        "configuration": {
            "H": list(H_HW), "H_tokens": math.prod(H_HW),
            "G": list(geometry.GLOBAL_HW), "G_tokens": math.prod(geometry.GLOBAL_HW),
            "destination_crop": [32, 32], "working_canvas": [32, 32] if mode == "direct" else [64, 64],
            "crop_count": len(regions),
            "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
            "coordinate_scale": [1.0, 1.0] if mode == "direct" else [31/63, 31/63],
            "sigmas": sigmas.tolist(), "seed": SEED,
        },
        "algebra": {"global_D_U_max_abs": geometry.max_right_inverse_error(random_g)},
        "runtime": {
            "sampling_wall_seconds": wall,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "global_cuda_ms": sum(x["cuda_ms"] for x in probe.calls if x["kind"] == "global"),
            "local_cuda_ms": sum(x["cuda_ms"] for x in probe.calls if x["kind"] == "local"),
        },
        "work": {
            "global_forwards": sum(x["kind"] == "global" for x in probe.calls),
            "local_forwards": sum(x["kind"] == "local" for x in probe.calls),
            "global_token_executions": sum(x["tokens"] for x in probe.calls if x["kind"] == "global"),
            "local_token_executions": sum(x["tokens"] for x in probe.calls if x["kind"] == "local"),
            "tokens_per_local_forward": 1024 if mode == "direct" else 4096,
            "destination_overlap_redundancy": sum(r.height*r.width for r in regions)/math.prod(H_HW),
        },
        "trajectory": {
            "overlap_rms": probe.overlaps,
            "projection_rms": [x["projection_rms"] for x in accepted],
            "invariant_max_abs": [x["invariant_max_abs"] for x in accepted],
            "final": tensor_summary(output),
            "local_state_records": [x["state"] for x in probe.calls if "state" in x],
        },
        "images": decode_artifacts(name, capture, probe.representative, output),
    }
    (OUTPUT / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate():
    results = {
        name: json.loads((OUTPUT / f"{name}.json").read_text(encoding="utf-8"))
        for name in VARIANTS
    }
    REPORT.write_text(json.dumps({
        "variants": results,
        "semantic_result": (
            "Sigma-consistent magnification restores useful detail and coherent "
            "whole-scene structure relative to naive magnification, but retains "
            "duplicate/ghosted train and bridge elements and is not qualified."
        ),
        "verdict": "NATIVE LOCAL MAGNIFICATION PARTIALLY SUPPORTED",
    }, indent=2), encoding="utf-8")
    sheet = Image.new("RGB", (768, 420 * len(results)), "white")
    for i, (name, result) in enumerate(results.items()):
        with Image.open(result["images"]["final"]["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
        sheet.paste(thumb, (0, i * 420 + 36))
        ImageDraw.Draw(sheet).text((10, i * 420 + 10), name, fill="black")
    sheet.save(OUTPUT / "FINAL_COMPARISON.png")
    print(json.dumps({name: {
        "wall": r["runtime"]["sampling_wall_seconds"],
        "local_cuda_ms": r["runtime"]["local_cuda_ms"],
        "overlap": r["trajectory"]["overlap_rms"],
        "final": r["trajectory"]["final"],
    } for name, r in results.items()}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
    elif args.variant:
        result = run_variant(args.variant)
        print(json.dumps({"variant": args.variant, "runtime": result["runtime"],
                          "final": result["trajectory"]["final"]}, indent=2))
    else:
        parser.error("choose --variant or --aggregate")


if __name__ == "__main__":
    main()
