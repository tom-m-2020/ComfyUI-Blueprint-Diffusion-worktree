"""Phase 9: normalized local working-canvas Candidate-3 falsifier."""

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
from PIL import Image, ImageOps, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_bounded_global_density as phase8b
import flux2_candidate3_performance_characterization as perf
import blueprint_diffusion.sampling.euler as production_euler
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.regions import OverlapAssembler
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


OUTPUT = ROOT / "experiments" / "flux2_candidate3_normalized_local_working_canvas_results"
REPORT = OUTPUT / "report.json"
H_HW = (64, 128)
OUTPUT_PIXELS = (1024, 2048)
WORK_HW = (32, 32)
STEPS = 4
SEED = 20260901
PROMPT = (
    "A single long red suspension bridge crossing the entire wide frame from left "
    "to right over calm blue water, exactly one yellow train centered on the bridge, "
    "one white lighthouse on the far left shore, one dark stone tower on the far "
    "right shore, continuous bridge deck and cables, coherent horizon, panoramic "
    "cinematic photograph, no duplicate bridges, no duplicate trains, no floating structures"
)
VARIANTS = {
    "A_CURRENT_DIRECT": {"geometry": (4, 3), "normalized": False},
    "B_FIXED_GLOBAL_DIRECT": {"geometry": (8, 3), "normalized": False},
    "C_FIXED_GLOBAL_NORMALIZED": {"geometry": (8, 3), "normalized": True},
}


def summary(value: torch.Tensor) -> dict:
    work = value.detach().float()
    return {
        "shape": list(value.shape), "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()), "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


class Capture:
    def __init__(self):
        self.values = {}

    def __call__(self, name, ordinal, value):
        if name in {"initial_H", "initial_G", "x0_G", "assembled_x0_H", "accepted_H", "accepted_G"}:
            self.values.setdefault(name, {})[ordinal] = value.detach().float().cpu()


class Measurements:
    def __init__(self):
        self.calls = []
        self.overlaps = []

    def timed(self, kind, tokens, call):
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start = end = None
        wall = time.perf_counter()
        result = call()
        if end is not None:
            end.record()
            torch.cuda.synchronize()
            cuda_ms = float(start.elapsed_time(end))
        else:
            cuda_ms = 0.0
        self.calls.append({
            "kind": kind, "tokens": tokens, "cuda_ms": cuda_ms,
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


@contextlib.contextmanager
def scoped_variant(block_high: int, block_global: int, normalized: bool, measurements: Measurements):
    original_geometry = production_euler.BlockDCTGeometry
    original_validate = Flux2Adapter.validate_run
    original_global = Flux2Adapter.predict_global
    original_region = Flux2Adapter.predict_region
    original_assemble = OverlapAssembler.assemble

    def geometry_factory(high_hw):
        return phase8b.ExperimentalBlockDCTGeometry(high_hw, block_high, block_global)

    def validate(adapter, **kwargs):
        forwarded = dict(kwargs)
        high_shape = kwargs["high_shape"]
        forwarded["global_shape"] = (
            *kwargs["global_shape"][:-2],
            high_shape[-2] // 4 * 3, high_shape[-1] // 4 * 3,
        )
        return original_validate(adapter, **forwarded)

    def predict_global(adapter, **kwargs):
        g = kwargs["g"]
        return measurements.timed(
            "global", g.shape[-2] * g.shape[-1],
            lambda: original_global(adapter, **kwargs),
        )

    def predict_region(adapter, **kwargs):
        h_view = kwargs["h_view"]
        region = kwargs["region"]
        if not normalized:
            return measurements.timed(
                "local", h_view.shape[-2] * h_view.shape[-1],
                lambda: original_region(adapter, **kwargs),
            )
        working = F.interpolate(h_view, size=WORK_HW, mode="bilinear", align_corners=True)
        options = adapter._options(kwargs["model_options"], {
            "shift_y": float(region.y), "shift_x": float(region.x),
            "scale_y": (region.height - 1.0) / (WORK_HW[0] - 1.0),
            "scale_x": (region.width - 1.0) / (WORK_HW[1] - 1.0),
        })
        prediction = measurements.timed(
            "local", math.prod(WORK_HW),
            lambda: kwargs["guider"](
                working, kwargs["sigma"].expand(1), model_options=options,
                seed=kwargs["seed"],
            ),
        )
        return F.interpolate(
            prediction, size=(region.height, region.width),
            mode="bilinear", align_corners=True,
        )

    def assemble(assembler, predictions, regions, target_hw):
        measurements.overlaps.append(Measurements.overlap(predictions, regions))
        return original_assemble(assembler, predictions, regions, target_hw)

    production_euler.BlockDCTGeometry = geometry_factory
    Flux2Adapter.validate_run = validate
    Flux2Adapter.predict_global = predict_global
    Flux2Adapter.predict_region = predict_region
    OverlapAssembler.assemble = assemble
    try:
        yield
    finally:
        production_euler.BlockDCTGeometry = original_geometry
        Flux2Adapter.validate_run = original_validate
        Flux2Adapter.predict_global = original_global
        Flux2Adapter.predict_region = original_region
        OverlapAssembler.assemble = original_assemble


def decode(capture: Capture, final: torch.Tensor, name: str) -> dict:
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    images = {}
    selected = {f"x0_step_{i}": value for i, value in capture.values["assembled_x0_H"].items()}
    selected["final"] = final.detach().float().cpu()
    for label, latent in selected.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}_{label}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images[label] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return images


def run_variant(name: str) -> dict:
    config = VARIANTS[name]
    block_high, block_global = config["geometry"]
    geometry = phase8b.ExperimentalBlockDCTGeometry(H_HW, block_high, block_global)
    random_g = torch.randn(1, 3, *geometry.GLOBAL_HW, generator=torch.Generator().manual_seed(9001))
    algebra = {
        "D_U_max_abs": geometry.max_right_inverse_error(random_g),
        "G_shape": list(geometry.GLOBAL_HW),
    }
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
    sampler = BlueprintEulerSampler(capture=capture)
    measurements = Measurements()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with scoped_variant(block_high, block_global, config["normalized"], measurements), torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    sampling_wall = time.perf_counter() - started
    regions = production_euler.FixedCropPlanner().plan(H_HW)
    accepted = [item for item in sampler.last_telemetry if item["event"] == "accepted_interval"]
    result = {
        "variant": name,
        "configuration": {
            "H": list(H_HW), "H_tokens": math.prod(H_HW),
            "G": list(geometry.GLOBAL_HW), "G_tokens": math.prod(geometry.GLOBAL_HW),
            "destination_regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
            "destination_crop_tokens": regions[0].height * regions[0].width,
            "working_canvas": list(WORK_HW) if config["normalized"] else [regions[0].height, regions[0].width],
            "per_local_forward_tokens": math.prod(WORK_HW) if config["normalized"] else regions[0].height * regions[0].width,
            "crop_count": len(regions), "seed": SEED, "sigmas": sigmas.tolist(),
            "global_coordinate_scale": [(H_HW[0]-1)/(geometry.GLOBAL_HW[0]-1), (H_HW[1]-1)/(geometry.GLOBAL_HW[1]-1)],
            "local_coordinate_scale": [(regions[0].height-1)/(WORK_HW[0]-1), (regions[0].width-1)/(WORK_HW[1]-1)] if config["normalized"] else [1.0, 1.0],
        },
        "algebra": algebra,
        "runtime": {
            "sampling_wall_seconds": sampling_wall,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "global_cuda_ms": sum(x["cuda_ms"] for x in measurements.calls if x["kind"] == "global"),
            "local_cuda_ms": sum(x["cuda_ms"] for x in measurements.calls if x["kind"] == "local"),
            "calls": measurements.calls,
        },
        "work": {
            "global_forwards": sum(x["kind"] == "global" for x in measurements.calls),
            "local_forwards": sum(x["kind"] == "local" for x in measurements.calls),
            "global_token_executions": sum(x["tokens"] for x in measurements.calls if x["kind"] == "global"),
            "local_token_executions": sum(x["tokens"] for x in measurements.calls if x["kind"] == "local"),
            "destination_coverage_tokens_per_interval": sum(r.height*r.width for r in regions),
            "destination_overlap_redundancy": sum(r.height*r.width for r in regions)/math.prod(H_HW),
        },
        "trajectory": {
            "overlap_rms": measurements.overlaps,
            "projection_rms": [x["projection_rms"] for x in accepted],
            "projection_over_H_rms": [None if x["projection_rms"] is None else x["projection_rms"]/x["accepted_H"]["rms"] for x in accepted],
            "invariant_max_abs": [x["invariant_max_abs"] for x in accepted],
            "final": summary(output),
            "finite": bool(torch.isfinite(output).all()),
        },
        "images": decode(capture, output, name),
    }
    path = OUTPUT / f"{name}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def aggregate() -> None:
    results = {}
    for name in VARIANTS:
        path = OUTPUT / f"{name}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        results[name] = json.loads(path.read_text(encoding="utf-8"))
    REPORT.write_text(json.dumps({
        "variants": results,
        "semantic_result": (
            "C preserves a coarse single bridge but loses useful local detail and "
            "retains train/structure ghosting; the naive normalized path fails."
        ),
        "verdict": "NORMALIZED LOCAL WORKING CANVAS FAILS LOCAL-FIDELITY GATE",
    }, indent=2), encoding="utf-8")
    thumbs = []
    for name, result in results.items():
        with Image.open(result["images"]["final"]["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
            canvas = Image.new("RGB", (768, 420), "white")
            canvas.paste(thumb, (0, 36))
            ImageDraw.Draw(canvas).text((12, 10), name, fill="black")
            thumbs.append(canvas)
    sheet = Image.new("RGB", (768, 420 * len(thumbs)), "white")
    for i, image in enumerate(thumbs):
        sheet.paste(image, (0, i * 420))
    sheet.save(OUTPUT / "FINAL_COMPARISON.png")
    rows = []
    for name, result in results.items():
        cells = []
        for label in ("x0_step_0", "x0_step_1", "x0_step_2", "x0_step_3", "final"):
            with Image.open(result["images"][label]["path"]) as image:
                cell = ImageOps.fit(image.convert("RGB"), (384, 192))
                framed = Image.new("RGB", (384, 218), "white")
                framed.paste(cell, (0, 26))
                ImageDraw.Draw(framed).text((8, 7), label, fill="black")
                cells.append(framed)
        row = Image.new("RGB", (384 * len(cells), 244), "white")
        ImageDraw.Draw(row).text((8, 7), name, fill="black")
        for i, cell in enumerate(cells):
            row.paste(cell, (i * 384, 26))
        rows.append(row)
    trajectory_sheet = Image.new("RGB", (384 * 5, 244 * len(rows)), "white")
    for i, row in enumerate(rows):
        trajectory_sheet.paste(row, (0, i * 244))
    trajectory_sheet.save(OUTPUT / "TRAJECTORY_COMPARISON.png")
    print(json.dumps({
        name: {"wall": value["runtime"]["sampling_wall_seconds"],
               "global_ms": value["runtime"]["global_cuda_ms"],
               "local_ms": value["runtime"]["local_cuda_ms"],
               "final": value["trajectory"]["final"]}
        for name, value in results.items()
    }, indent=2))


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
