"""Phase 8a: unchanged-production Candidate-3 practical scaling frontier."""

from __future__ import annotations

import contextlib
import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_global_refresh_cadence as phase6i
import flux2_candidate3_performance_characterization as perf

from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.regions import FixedCropPlanner
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


OUTPUT = ROOT / "experiments" / "flux2_candidate3_practical_scaling_frontier_results"
STEPS = 4
SEED = 20260901
LADDER = (
    ("1536x2048", 1536, 2048),
    ("2048x2048", 2048, 2048),
    ("1536x3072", 1536, 3072),
    ("2048x3072", 2048, 3072),
    ("2048x4096", 2048, 4096),
)
BRIDGE_PROMPT = next(item[3] for item in phase6i.CASES if item[0] == "h64_bridge_train")
SUBJECT_PROMPT = next(item[3] for item in phase6i.CASES if item[0] == "h64_centered_astronaut")
PRACTICAL_WALL_LIMIT_SECONDS = 180.0


def geometry_record(height: int, width: int) -> dict:
    high_hw = (height // 16, width // 16)
    geometry = BlockDCTGeometry(high_hw)
    regions = FixedCropPlanner().plan(high_hw)
    unique = math.prod(high_hw)
    local = sum(region.height * region.width for region in regions)
    return {
        "target_pixels_hw": [height, width],
        "target_pixels_wh": [width, height],
        "H_shape": list(high_hw),
        "unique_H_tokens": unique,
        "G_shape": list(geometry.GLOBAL_HW),
        "global_tokens": math.prod(geometry.GLOBAL_HW),
        "crop_size": [regions[0].height, regions[0].width],
        "stride_policy": 24,
        "y_starts": sorted({region.y for region in regions}),
        "x_starts": sorted({region.x for region in regions}),
        "crop_count": len(regions),
        "summed_local_tokens_per_interval": local,
        "overlap_redundancy": local / unique,
        "blueprint_global_forwards": STEPS - 1,
        "blueprint_local_forwards": STEPS * len(regions),
        "blueprint_total_forwards": (STEPS - 1) + STEPS * len(regions),
        "dense_total_forwards": STEPS,
    }


class AdapterForwardTimer:
    def __init__(self) -> None:
        self.records = []
        self.failed_call = None
        self._global = Flux2Adapter.predict_global
        self._region = Flux2Adapter.predict_region

    def _wrap(self, category, original):
        timer = self

        def wrapped(adapter_self, **kwargs):
            detail = {
                "sigma": float(kwargs["sigma"]),
                "region": None if category == "global" else kwargs["region"].index,
            }
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                result = original(adapter_self, **kwargs)
            except BaseException:
                timer.failed_call = {"category": category, **detail}
                raise
            end.record()
            timer.records.append({
                "category": category, **detail, "start": start, "end": end,
                "output_shape": list(result.shape),
            })
            return result

        return wrapped

    @contextlib.contextmanager
    def patch(self):
        Flux2Adapter.predict_global = self._wrap("global", self._global)
        Flux2Adapter.predict_region = self._wrap("local", self._region)
        try:
            yield
        finally:
            Flux2Adapter.predict_global = self._global
            Flux2Adapter.predict_region = self._region

    def resolve(self) -> list[dict]:
        torch.cuda.synchronize()
        result = []
        for item in self.records:
            result.append({
                key: value for key, value in item.items() if key not in {"start", "end"}
            } | {"cuda_ms": float(item["start"].elapsed_time(item["end"]))})
        return result


def run_blueprint(model, positive, negative, noise, sigmas, seed):
    sampler = BlueprintEulerSampler()
    timer = AdapterForwardTimer()
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with timer.patch(), torch.inference_mode():
            output = phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                torch.zeros_like(noise), callback=lambda *args: None,
                disable_pbar=True, seed=seed,
            )
        torch.cuda.synchronize()
        forwards = timer.resolve()
        accepted = [item for item in sampler.last_telemetry if item["event"] == "accepted_interval"]
        return output.detach().cpu(), {
            "status": "SUCCESS",
            "sampling_wall_seconds": time.perf_counter() - started,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "forward_timings": forwards,
            "global_cuda_ms": sum(item["cuda_ms"] for item in forwards if item["category"] == "global"),
            "local_cuda_ms": sum(item["cuda_ms"] for item in forwards if item["category"] == "local"),
            "global_forward_count": sum(item["category"] == "global" for item in forwards),
            "local_forward_count": sum(item["category"] == "local" for item in forwards),
            "telemetry": sampler.last_telemetry,
            "integrity": {
                "finite": bool(torch.isfinite(output).all()),
                "terminal_only_final": [item["terminal_release"] for item in accepted]
                == [False, False, False, True],
                "only_final_omits_global": [item["global_forward_performed"] for item in accepted]
                == [True, True, True, False],
                "nonterminal_invariants": all(
                    item["invariant_max_abs"] is None or item["invariant_max_abs"] <= 2e-6
                    for item in accepted
                ),
            },
        }
    except BaseException as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, failure_record(exc, baseline_allocated, baseline_reserved, timer.failed_call)


def run_dense(model, positive, negative, noise, sigmas, seed):
    sampler = perf.MeasuredDenseSampler()
    gc.collect()
    try:
        with torch.inference_mode():
            output = phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                torch.zeros_like(noise), callback=lambda *args: None,
                disable_pbar=True, seed=seed,
            )
        torch.cuda.synchronize()
        measurement = sampler.measurement
        timings = [
            item for item in measurement["timing_ranges"]
            if item["category"] == "dense_forward"
        ]
        measurement.update({
            "status": "SUCCESS", "forward_timings": timings,
            "dense_forward_count": len(timings),
        })
        return output.detach().cpu(), measurement
    except BaseException as exc:
        allocated = sampler.measurement.get(
            "baseline_allocated_bytes",
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0,
        )
        reserved = sampler.measurement.get(
            "baseline_reserved_bytes",
            int(torch.cuda.memory_reserved()) if torch.cuda.is_available() else 0,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, failure_record(exc, allocated, reserved, "dense_forward")


def failure_record(exc, baseline_allocated, baseline_reserved, point):
    return {
        "status": "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "OTHER_ERROR",
        "error_type": type(exc).__name__, "error_message": str(exc),
        "failure_point": point,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
        "traceback": traceback.format_exc(),
    }


def prepare(model):
    perf.prepare_model_state(model)


def decode_outputs(outputs: dict) -> dict:
    decoded = {}
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    for name, latent in outputs.items():
        try:
            with torch.inference_mode():
                pixels = vae.decode(latent).cpu()
            path = OUTPUT / f"{name}.png"
            phase2.save_pixels(pixels, path)
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                digest = hashlib.sha256(rgb.tobytes()).hexdigest()
                dimensions = list(rgb.size)
            decoded[name] = {
                "status": "SUCCESS", "path": str(path),
                "dimensions_wh": dimensions, "sha256_rgb": digest,
                "finite": bool(torch.isfinite(pixels).all()),
            }
        except BaseException as exc:
            decoded[name] = {
                "status": "OTHER_ERROR", "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    return decoded


def make_review_sheets() -> None:
    cell_w, cell_h, label_h = 640, 360, 28
    sheet = Image.new("RGB", (cell_w * 2, (cell_h + label_h) * len(LADDER)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, _, _) in enumerate(LADDER):
        for col, variant in enumerate(("dense", "blueprint")):
            suffix = "_cold" if label == "2048x4096" else ""
            path = OUTPUT / f"{label}_bridge_{variant}{suffix}.png"
            image = Image.open(path).convert("RGB")
            preview = ImageOps.contain(image, (cell_w, cell_h), Image.Resampling.LANCZOS)
            x, y = col * cell_w, row * (cell_h + label_h)
            sheet.paste(preview, (x + (cell_w - preview.width) // 2, y + label_h))
            draw.text((x + 8, y + 7), f"{label} {variant}", fill="black")
            image.close()
    sheet.save(OUTPUT / "frontier_bridge_contact_sheet.png")

    semantic = Image.new("RGB", (cell_w, (cell_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(semantic)
    items = (
        ("2048x4096 dense bridge", OUTPUT / "2048x4096_bridge_dense_cold.png"),
        ("2048x4096 Blueprint bridge", OUTPUT / "2048x4096_bridge_blueprint_cold.png"),
        ("2048x4096 Blueprint boundary subject", OUTPUT / "2048x4096_subject_blueprint.png"),
    )
    for row, (label, path) in enumerate(items):
        image = Image.open(path).convert("RGB")
        preview = ImageOps.contain(image, (cell_w, cell_h), Image.Resampling.LANCZOS)
        y = row * (cell_h + label_h)
        semantic.paste(preview, ((cell_w - preview.width) // 2, y + label_h))
        draw.text((8, y + 7), label, fill="black")
        image.close()
    semantic.save(OUTPUT / "largest_semantic_contact_sheet.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-final", action="store_true")
    parser.add_argument("--subject-only", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.review_only:
        make_review_sheets()
        return
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    conditioning = {}
    for name, prompt in (("bridge", BRIDGE_PROMPT), ("subject", SUBJECT_PROMPT)):
        conditioning[name] = (
            clip.encode_from_tokens_scheduled(clip.tokenize(prompt)),
            clip.encode_from_tokens_scheduled(clip.tokenize("")),
        )
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    report_path = OUTPUT / "report.json"
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "steps": STEPS, "cfg": 1.0,
            "sampler": "production BlueprintEulerSampler / matched dense Euler",
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "production_changes": False,
            "practical_wall_limit_seconds": PRACTICAL_WALL_LIMIT_SECONDS,
            "dimension_convention": "target labels are height x width",
        },
        "cases": {},
    }
    if args.resume_final or args.subject_only:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = {}
    dense_frontier_closed = False
    blueprint_frontier_closed = False
    largest_blueprint = None
    new_largest = False
    if args.resume_final or args.subject_only:
        previous = report.get("frontier", {}).get("largest_blueprint_attempted_successfully")
        largest_blueprint = next((item for item in LADDER if item[0] == previous), None)
    if args.subject_only:
        dense_frontier_closed = report.get("frontier", {}).get("dense_frontier_closed", False)
        blueprint_frontier_closed = report.get("frontier", {}).get("blueprint_frontier_closed", False)
    bridge_positive, bridge_negative = conditioning["bridge"]

    ladder = () if args.subject_only else (LADDER[-1:] if args.resume_final else LADDER)
    for label, height, width in ladder:
        geometry = geometry_record(height, width)
        h_shape = tuple(geometry["H_shape"])
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(SEED))
        sigmas = phase2.get_schedule(STEPS, math.prod(h_shape)).float().clone()
        sigmas[0] = 1.0
        case = {"geometry": geometry, "sigmas": sigmas.tolist(), "bridge": {}}
        for variant in ("dense", "blueprint"):
            closed = dense_frontier_closed if variant == "dense" else blueprint_frontier_closed
            if closed:
                case["bridge"][variant] = {
                    "status": "SKIPPED_AFTER_FRONTIER",
                    "reason": "An earlier larger-than-qualified ladder point established this path's limit.",
                }
                continue
            print(f"{label} {variant}: cold preparation", flush=True)
            prepare(model)
            runs = {}
            warm_output = None
            successful_temperature = None
            for temperature in ("cold", "warm"):
                print(f"{label} {variant}: {temperature}", flush=True)
                if variant == "dense":
                    output, measurement = run_dense(
                        model, bridge_positive, bridge_negative, noise, sigmas, SEED
                    )
                else:
                    output, measurement = run_blueprint(
                        model, bridge_positive, bridge_negative, noise, sigmas, SEED
                    )
                runs[temperature] = measurement
                if measurement["status"] != "SUCCESS":
                    break
                warm_output = output
                successful_temperature = temperature
            case["bridge"][variant] = runs
            failure = next(
                (item for item in runs.values() if item["status"] != "SUCCESS"), None
            )
            if failure is not None:
                if warm_output is not None:
                    outputs[f"{label}_bridge_{variant}_{successful_temperature}"] = warm_output
                    if variant == "blueprint":
                        largest_blueprint = (label, height, width)
                        new_largest = True
                if variant == "dense":
                    dense_frontier_closed = True
                else:
                    blueprint_frontier_closed = True
            elif "warm" in runs:
                outputs[f"{label}_bridge_{variant}"] = warm_output
                if variant == "blueprint":
                    largest_blueprint = (label, height, width)
                    new_largest = True
                    if runs["warm"]["sampling_wall_seconds"] > PRACTICAL_WALL_LIMIT_SECONDS:
                        blueprint_frontier_closed = True
                        case["bridge"][variant]["practical_runtime_limit_reached"] = True
        report["cases"][label] = case
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if dense_frontier_closed and blueprint_frontier_closed:
            break

    if largest_blueprint is not None and (
        args.subject_only or not args.resume_final or new_largest
    ):
        label, height, width = largest_blueprint
        h_shape = (height // 16, width // 16)
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(20260902))
        sigmas = phase2.get_schedule(STEPS, math.prod(h_shape)).float().clone()
        sigmas[0] = 1.0
        subject_positive, subject_negative = conditioning["subject"]
        print(f"{label} blueprint: second-prompt semantic control", flush=True)
        prepare(model)
        output, measurement = run_blueprint(
            model, subject_positive, subject_negative, noise, sigmas, 20260902
        )
        report["second_prompt"] = {
            "geometry": label, "prompt": SUBJECT_PROMPT, "seed": 20260902,
            "blueprint": measurement,
        }
        if output is not None:
            outputs[f"{label}_subject_blueprint"] = output

    decoded = report.get("decoded_outputs", {})
    decoded.update(decode_outputs(outputs))
    report["decoded_outputs"] = decoded
    report["frontier"] = {
        "largest_blueprint_attempted_successfully": None if largest_blueprint is None else largest_blueprint[0],
        "dense_frontier_closed": dense_frontier_closed,
        "blueprint_frontier_closed": blueprint_frontier_closed,
    }
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "frontier": report["frontier"],
        "statuses": {
            label: {
                variant: {
                    temperature: data["status"]
                    for temperature, data in runs.items()
                    if isinstance(data, dict) and "status" in data
                }
                for variant, runs in case["bridge"].items()
            }
            for label, case in report["cases"].items()
        },
        "report": str(path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
