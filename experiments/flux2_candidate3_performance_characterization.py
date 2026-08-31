"""Phase 6a: measure dense FLUX.2 and Candidate-3 without optimizing either."""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_performance_results"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2


def load_production_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "blueprint_diffusion",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blueprint_diffusion"] = module
    spec.loader.exec_module(module)


load_production_package()
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler
from blueprint_diffusion.sampling.euler import validate_schedule
from blueprint_diffusion.state import BlueprintState


PROMPT = (
    "A cinematic wide-angle photograph of exactly one large full-body woman "
    "standing centered in the foreground, occupying most of the image height; "
    "exactly one red vintage car parked on the far left; exactly one tall green "
    "tree on the far right; asymmetric left-center-right composition, continuous "
    "dry ground plane, coherent perspective and scale, distant low hills, sunset "
    "light, no duplicate people, no duplicate cars, no duplicate trees"
)
SEED = 20260831
TEXT_TOKENS = 512
GIB = 1024**3
CASES = (
    ("512x512", 512, 512),
    ("1024x512", 1024, 512),
    ("1280x512", 1280, 512),
    ("1024x2048", 1024, 2048),
)


class TimingRecorder:
    def __init__(self) -> None:
        self.ranges: list[dict[str, Any]] = []

    @contextmanager
    def cuda(self, category: str, interval: int | None, detail: str = ""):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        cpu_start = time.perf_counter()
        try:
            yield
        finally:
            end.record()
            self.ranges.append({
                "category": category,
                "interval": interval,
                "detail": detail,
                "start": start,
                "end": end,
                "cpu_wall_ms": (time.perf_counter() - cpu_start) * 1000.0,
            })

    @contextmanager
    def cpu(self, category: str, interval: int | None, detail: str = ""):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.ranges.append({
                "category": category,
                "interval": interval,
                "detail": detail,
                "cuda_ms": None,
                "cpu_wall_ms": (time.perf_counter() - started) * 1000.0,
            })

    def resolve(self) -> list[dict[str, Any]]:
        torch.cuda.synchronize()
        result = []
        for item in self.ranges:
            resolved = {key: value for key, value in item.items() if key not in {"start", "end"}}
            if "start" in item:
                resolved["cuda_ms"] = float(item["start"].elapsed_time(item["end"]))
            result.append(resolved)
        return result


def summarize_ranges(ranges: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in ranges:
        if item["cuda_ms"] is not None:
            result[item["category"]] = result.get(item["category"], 0.0) + item["cuda_ms"]
    return result


class MeasuredSamplerBase(phase2.comfy.samplers.Sampler):
    def __init__(self) -> None:
        self.measurement: dict[str, Any] = {}
        self.preview_count = 0

    def begin(self) -> tuple[TimingRecorder, float]:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.measurement = {
            "baseline_allocated_bytes": int(torch.cuda.memory_allocated()),
            "baseline_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        return TimingRecorder(), time.perf_counter()

    def finish(self, recorder: TimingRecorder, started: float) -> None:
        torch.cuda.synchronize()
        self.measurement.update({
            "sampling_wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "preview_count": self.preview_count,
            "timing_ranges": recorder.resolve(),
        })
        self.measurement["cuda_category_totals_ms"] = summarize_ranges(
            self.measurement["timing_ranges"]
        )


class MeasuredDenseSampler(MeasuredSamplerBase):
    def sample(
        self, model, sigmas, extra_args, callback, noise, latent_image=None,
        denoise_mask=None, disable_pbar=False,
    ):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Dense measurement requires the same empty-latent T2I contract.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        intervals = []
        for ordinal in range(len(sigmas) - 1):
            sigma = sigmas[ordinal]
            sigma_next = sigmas[ordinal + 1]
            with recorder.cuda("dense_forward", ordinal):
                x0 = model(
                    h,
                    sigma.expand(h.shape[0]),
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed", 0),
                )
            with recorder.cuda("dense_euler", ordinal):
                h = h + (h - x0) / sigma * (sigma_next - sigma)
            finite = bool(torch.isfinite(h).all())
            if not finite:
                raise RuntimeError(f"Dense state became nonfinite at interval {ordinal}.")
            if callback is not None:
                callback(ordinal, x0, h, len(sigmas) - 1)
                self.preview_count += 1
            intervals.append({
                "ordinal": ordinal,
                "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "finite": finite,
            })
        with recorder.cuda("inverse_noise_scaling", None):
            output = model_sampling.inverse_noise_scaling(sigmas[-1], h)
        self.finish(recorder, started)
        self.measurement["intervals"] = intervals
        self.measurement["integrity"] = {
            "status": "SUCCESS",
            "finite_final": bool(torch.isfinite(output).all()),
            "expected_intervals": len(intervals) == 4,
        }
        return output


class MeasuredBlueprintSampler(MeasuredSamplerBase):
    def sample(
        self, model, sigmas, extra_args, callback, noise, latent_image=None,
        denoise_mask=None, disable_pbar=False,
    ):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Blueprint measurement requires the production empty-latent contract.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)

        with recorder.cpu("geometry_and_crop_setup_cpu", None):
            geometry = BlockDCTGeometry(tuple(h.shape[-2:]))
            planner = FixedCropPlanner()
            regions = planner.plan(tuple(h.shape[-2:]))
            assembler = OverlapAssembler()
            adapter = Flux2Adapter()
        with recorder.cuda("dct_qualification", None):
            right_inverse_error = geometry.qualify(device=h.device, dtype=h.dtype)
        with recorder.cuda("initial_D_H", None):
            g = geometry.restrict(h)
        state = BlueprintState(g, h, float(sigmas[0]), 0, uuid.uuid4().hex)
        intervals = []

        for ordinal in range(len(sigmas) - 1):
            sigma = sigmas[ordinal]
            sigma_next = sigmas[ordinal + 1]
            sigma_next_value = float(sigma_next)
            with recorder.cpu("crop_planning_cpu", ordinal):
                regions = planner.plan(tuple(state.h.shape[-2:]))
            adapter.validate_run(
                guider=model,
                high_shape=tuple(state.h.shape),
                global_shape=tuple(state.g.shape),
                crops=regions,
                sigmas=torch.stack((sigma, sigma_next)),
                latent=state.h,
            )
            accepted_h_snapshot = state.h.clone()
            accepted_g_snapshot = state.g.clone()

            with recorder.cuda("global_forward", ordinal):
                x0_g = adapter.predict_global(
                    guider=model,
                    g=state.g,
                    sigma=sigma,
                    canvas=tuple(state.h.shape[-2:]),
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed", 0),
                )
            local_predictions = []
            crop_shapes = []
            with recorder.cpu("crop_extraction_cpu", ordinal):
                views = [
                    state.h[:, :, region.y:region.y2, region.x:region.x2]
                    for region in regions
                ]
            for region, view in zip(regions, views):
                if view.untyped_storage().data_ptr() != state.h.untyped_storage().data_ptr():
                    raise RuntimeError(f"Crop {region.index} is not an immutable H view.")
                with recorder.cuda("local_forward", ordinal, f"crop_{region.index}"):
                    prediction = adapter.predict_region(
                        guider=model,
                        h_view=view,
                        sigma=sigma,
                        canvas=tuple(state.h.shape[-2:]),
                        region=region,
                        model_options=extra_args["model_options"],
                        seed=extra_args.get("seed", 0),
                    )
                local_predictions.append(prediction)
                crop_shapes.append(list(view.shape[-2:]))
            if not torch.equal(state.h, accepted_h_snapshot) or not torch.equal(state.g, accepted_g_snapshot):
                raise RuntimeError("A Blueprint model call mutated accepted state.")

            with recorder.cuda("overlap_assembly", ordinal):
                x0_h, coverage = assembler.assemble(
                    local_predictions, regions, tuple(state.h.shape[-2:])
                )
            with recorder.cuda("euler_proposals", ordinal):
                dt = sigma_next - sigma
                g_star = state.g + (state.g - x0_g) / sigma * dt
                h_star = state.h + (state.h - x0_h) / sigma * dt
            with recorder.cuda("coupling_D", ordinal):
                restricted_h_star = geometry.restrict(h_star)
            with recorder.cuda("coupling_delta", ordinal):
                coarse_delta = g_star - restricted_h_star
            with recorder.cuda("coupling_U", ordinal):
                projection = geometry.prolong(coarse_delta)
            with recorder.cuda("coupling_accept", ordinal):
                next_h = h_star if sigma_next_value == 0.0 else h_star + projection
                next_g = g_star
            projection_rms = float(projection.float().square().mean().sqrt())

            values = (x0_g, x0_h, g_star, h_star, next_g, next_h)
            if not all(bool(torch.isfinite(value).all()) for value in values):
                raise RuntimeError(f"Blueprint state became nonfinite at interval {ordinal}.")
            invariant_error = None
            if sigma_next_value > 0.0:
                with recorder.cuda("invariant_D", ordinal):
                    accepted_restriction = geometry.restrict(next_h)
                invariant_error = float((accepted_restriction.float() - next_g.float()).abs().max())
                if invariant_error > geometry.TOLERANCE:
                    raise RuntimeError(
                        f"D(H_next) != G_next at interval {ordinal}: {invariant_error}."
                    )
            coverage_min = float(coverage.min())
            coverage_max = float(coverage.max())
            state = BlueprintState(
                next_g, next_h, sigma_next_value, ordinal + 1,
                f"measured:{ordinal}",
            )
            if callback is not None:
                callback(ordinal, x0_h, state.h, len(sigmas) - 1)
                self.preview_count += 1
            intervals.append({
                "ordinal": ordinal,
                "sigma": float(sigma),
                "sigma_next": sigma_next_value,
                "terminal_release": sigma_next_value == 0.0,
                "global_tokens": int(state.g.shape[-2] * state.g.shape[-1]),
                "crop_count": len(regions),
                "crop_shapes": crop_shapes,
                "summed_local_tokens": sum(r.height * r.width for r in regions),
                "coverage_min": coverage_min,
                "coverage_max": coverage_max,
                "projection_rms": projection_rms,
                "invariant_max_abs": invariant_error,
            })

        with recorder.cuda("inverse_noise_scaling", None):
            output = model_sampling.inverse_noise_scaling(sigmas[-1], state.h)
        self.finish(recorder, started)
        self.measurement["intervals"] = intervals
        self.measurement["integrity"] = {
            "status": "SUCCESS",
            "finite_final": bool(torch.isfinite(output).all()),
            "positive_complete_coverage": all(item["coverage_min"] > 0 for item in intervals),
            "nonterminal_invariants": all(
                item["invariant_max_abs"] is None
                or item["invariant_max_abs"] <= geometry.TOLERANCE
                for item in intervals
            ),
            "terminal_release_only_last": [item["terminal_release"] for item in intervals]
            == [False, False, False, True],
            "expected_intervals": len(intervals) == 4,
        }
        self.measurement["right_inverse_max_abs"] = right_inverse_error
        return output


def preview_counter(counter: dict[str, int]):
    def callback(step, x0, x, total_steps):
        counter["count"] += 1
    return callback


def run_sample(model, positive, negative, noise, sigmas, variant: str):
    sampler = MeasuredDenseSampler() if variant == "dense" else MeasuredBlueprintSampler()
    counter = {"count": 0}
    outer_started = time.perf_counter()
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model,
            noise.clone(),
            1.0,
            sampler,
            sigmas.clone(),
            positive,
            negative,
            torch.zeros_like(noise),
            callback=preview_counter(counter),
            disable_pbar=True,
            seed=SEED,
        )
    torch.cuda.synchronize()
    sampler.measurement["outer_sample_custom_wall_seconds"] = time.perf_counter() - outer_started
    sampler.measurement["callback_count"] = counter["count"]
    sampler.measurement["integrity"]["expected_preview_count"] = counter["count"] == 4
    return output.detach().cpu(), sampler.measurement


def failure_record(exc: BaseException) -> dict[str, Any]:
    status = "OOM" if isinstance(exc, torch.cuda.OutOfMemoryError) else "OTHER ERROR"
    if isinstance(exc, (ValueError, RuntimeError)) and "Blueprint" in str(exc):
        status = "VALIDATION FAILURE"
    return {
        "integrity": {"status": status},
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }


def geometry_telemetry(width: int, height: int) -> dict[str, Any]:
    high_hw = (height // 16, width // 16)
    geometry = BlockDCTGeometry(high_hw)
    regions = FixedCropPlanner().plan(high_hw)
    unique_h = math.prod(high_hw)
    global_tokens = math.prod(geometry.GLOBAL_HW)
    local_tokens = sum(region.height * region.width for region in regions)
    return {
        "target_pixels_wh": [width, height],
        "H_shape": list(high_hw),
        "G_shape": list(geometry.GLOBAL_HW),
        "unique_H_tokens": unique_h,
        "global_tokens": global_tokens,
        "crop_count": len(regions),
        "tokens_per_crop": 1024,
        "summed_local_token_executions": local_tokens,
        "overlap_redundancy": local_tokens / unique_h,
        "blueprint_forwards_per_interval": 1 + len(regions),
        "blueprint_forwards_per_generation": 4 * (1 + len(regions)),
        "dense_forwards_per_interval": 1,
        "dense_forwards_per_generation": 4,
        "attention_dimensions": {
            "dense_QK": [TEXT_TOKENS + unique_h, TEXT_TOKENS + unique_h],
            "global_QK": [TEXT_TOKENS + global_tokens, TEXT_TOKENS + global_tokens],
            "local_QK_each": [TEXT_TOKENS + 1024, TEXT_TOKENS + 1024],
            "scope": "matrix dimensions only; not FLOP estimates",
        },
        "regions": [vars(region) for region in regions],
    }


def prepare_model_state(model) -> None:
    gc.collect()
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    phase2.comfy.model_management.load_models_gpu([model])
    torch.cuda.synchronize()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    report: dict[str, Any] = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": PROMPT,
            "seed": SEED,
            "cfg": 1.0,
            "steps": 4,
            "sampler": "deterministic Euler/CONST flow",
            "device": str(phase2.comfy.model_management.get_torch_device()),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "timing": "CUDA events for GPU ranges; synchronized sampling wall time",
            "memory": "PyTorch allocator allocated/reserved only",
        },
        "cases": {},
    }
    warm_outputs: dict[str, torch.Tensor] = {}

    for name, width, height in CASES:
        geometry = geometry_telemetry(width, height)
        high_hw = tuple(geometry["H_shape"])
        noise = torch.randn(
            (1, 128, *high_hw), generator=torch.Generator().manual_seed(SEED)
        )
        sigmas = phase2.get_schedule(4, math.prod(high_hw)).float().clone()
        case = {"geometry": geometry, "sigmas": [float(x) for x in sigmas], "variants": {}}
        for variant in ("dense", "blueprint"):
            print(f"{name} {variant}: preparing equivalent cold state", flush=True)
            runs = {}
            try:
                prepare_model_state(model)
                for temperature in ("cold", "warm"):
                    print(f"{name} {variant} {temperature}", flush=True)
                    output, measurement = run_sample(
                        model, positive, negative, noise, sigmas, variant
                    )
                    runs[temperature] = measurement
                    if temperature == "warm":
                        warm_outputs[f"{name}_{variant}"] = output
            except BaseException as exc:
                runs["failure"] = failure_record(exc)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"{name} {variant} failed: {type(exc).__name__}: {exc}", flush=True)
            case["variants"][variant] = runs
        report["cases"][name] = case

    # Decode only after every sampling/memory measurement has completed.
    decoded = {}
    try:
        vae = phase2.comfy.sd.VAE(
            sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
        )
        for name, latent in warm_outputs.items():
            with torch.inference_mode():
                pixels = vae.decode(latent).cpu()
            valid = bool(torch.isfinite(pixels).all()) and pixels.ndim == 4
            decoded[name] = {"valid": valid, "shape": list(pixels.shape)}
            phase2.save_pixels(pixels, OUTPUT / f"{name}.png")
    except BaseException as exc:
        decoded["decode_error"] = {"type": type(exc).__name__, "message": str(exc)}
    report["decoded_outputs"] = decoded

    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(path), "decoded": decoded}, indent=2), flush=True)


if __name__ == "__main__":
    main()
