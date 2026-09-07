"""Phase 46: fixed epsilon_projection_cw Blueprint discriminator."""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments"), str(PACKAGE)]

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
import flux2_configurable_local_multistep as phase40
import flux2_persistent_coarse_guidance as phase39
import flux2_terminal_resampling_four_interval_local_trajectory as phase38
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.configurable_resampling import (
    ConfigurableResamplingGeometry,
    StreamingOverlapMetric,
    bounded_blueprint_transfer,
    configurable_region_noise,
    restrict_configurable_working,
)
from blueprint_diffusion.terminal_resampling import (
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    lift_region,
    region_noise,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_phase46_epsilon_projection_cw_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
PHASE38 = ROOT / "experiments" / "flux2_terminal_resampling_four_interval_local_trajectory_results"
PHASE39 = ROOT / "experiments" / "flux2_persistent_coarse_guidance_results"
PHASE40 = ROOT / "experiments" / "flux2_configurable_local_multistep_results"
WEIGHT = 0.25
GEOMETRY = ConfigurableResamplingGeometry((45, 45), (128, 128), (32, 32), (16, 16), (64, 64))


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _overlap_rms(values, regions) -> float:
    metric = StreamingOverlapMetric()
    for value, region in zip(values, regions, strict=True):
        metric.add(value, region)
    return metric.finish()


def rowwise_collinear(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Exact RES4LYF projection shape semantics for one [H,W] channel."""
    y_flat = y.reshape(y.size(0), -1).clone()
    x_flat = x.reshape(x.size(0), -1).clone()
    y_flat /= y_flat.norm(dim=-1, keepdim=True)
    return (torch.sum(x_flat * y_flat, dim=-1, keepdim=True) * y_flat).reshape_as(x)


def epsilon_projection_cw(
    raw: torch.Tensor,
    guide_derivative: torch.Tensor,
    model_prediction: torch.Tensor,
    guide: torch.Tensor,
    *,
    weight: float = WEIGHT,
) -> tuple[torch.Tensor, dict[str, object]]:
    if raw.shape != guide_derivative.shape or raw.shape != model_prediction.shape or raw.shape != guide.shape:
        raise ValueError("epsilon_projection_cw tensors must have identical shapes.")
    if raw.ndim != 4 or raw.shape[0] != 1:
        raise ValueError("Phase 46 requires one BCHW W state.")
    if weight != WEIGHT or not all(torch.isfinite(value).all() for value in (raw, guide_derivative, model_prediction, guide)):
        raise ValueError("Phase 46 requires finite tensors and fixed weight 0.25.")

    distances = torch.stack([
        torch.norm(model_prediction[b, c] - guide[b, c])
        for b in range(raw.shape[0]) for c in range(raw.shape[1])
    ])
    average = distances.sum() / raw.shape[1]
    ratios = torch.nan_to_num(distances / average, 0.0)
    used = raw.clone()
    collinear = torch.empty_like(raw)
    orthogonal = torch.empty_like(raw)
    projection = torch.empty_like(raw)
    ratio_index = 0
    for b in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            ratio = ratios[ratio_index]
            ratio_index += 1
            first = used[b, c] + ratio * weight * (guide_derivative[b, c] - used[b, c])
            candidate = guide_derivative[b, c]  # full-one mask, no inverse guide
            col = rowwise_collinear(first, candidate)
            ortho = candidate - rowwise_collinear(candidate, first)
            projected_target = col + ortho
            used[b, c] = first + ratio * weight * (projected_target - first)
            collinear[b, c] = col
            orthogonal[b, c] = ortho
            projection[b, c] = projected_target

    increment = used - raw
    telemetry = {
        "effective_weight_ratios": [float(value) for value in ratios],
        "effective_weights": [float(value * weight) for value in ratios],
        "raw_derivative_hash": tensor_hash(raw),
        "projected_derivative_hash": tensor_hash(used),
        "collinear_hash": tensor_hash(collinear),
        "orthogonal_hash": tensor_hash(orthogonal),
        "projection_target_hash": tensor_hash(projection),
        "raw_derivative_rms": rms(raw),
        "projected_derivative_rms": rms(used),
        "collinear_rms": rms(collinear),
        "orthogonal_rms": rms(orthogonal),
        "projection_increment_rms": rms(increment),
        "increment_to_raw_ratio": rms(increment) / max(rms(raw), 1e-30),
    }
    return used, telemetry


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def execute(*, run_name, guider, adapter, geometry, mapped_cpu, seed, options, device):
    sigmas = phase38.schedule()
    regions = geometry.regions()
    control = json.loads((PHASE40 / "A_ONE_INTERVAL" / "summary.json").read_text(encoding="utf-8"))["regions"]
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw, template=mapped_cpu.to(device))
    records, restricted_predictions, barriers = [], [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for region in regions:
        guide_cpu, source_hw = bounded_blueprint_transfer(mapped_cpu, geometry, region)
        guide = guide_cpu.to(device)
        guide_snapshot = guide.clone()
        noise_cpu = configurable_region_noise(seed, region, geometry.working_hw, device="cpu", dtype=torch.float32)
        noise = noise_cpu.to(device)
        sigma0 = torch.tensor(sigmas[0], device=device, dtype=guide.dtype)
        working = guider.inner_model.model_sampling.noise_scaling(sigma0, noise, guide, False)
        expected = control[region.index]
        if tensor_hash(noise) != expected["noise_hash"] or tensor_hash(working) != expected["initial_w_hash"]:
            raise RuntimeError(f"Phase-29 initialization mismatch in region {region.index}.")
        steps = []
        with torch.inference_mode():
            for ordinal, (sigma_value, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
                sigma = torch.tensor(sigma_value, device=device, dtype=working.dtype)
                state_hash = tensor_hash(working)
                raw_prediction = adapter.predict_native(
                    guider=guider, value=working, sigma=sigma,
                    expected_hw=geometry.working_hw, model_options=options, seed=seed,
                )
                prediction_hash = tensor_hash(raw_prediction)
                if ordinal == 0:
                    expected_hash = expected["steps"][0]["prediction_hash"]
                    if prediction_hash != expected_hash:
                        raise RuntimeError(f"First raw prediction mismatch in region {region.index}.")
                raw_derivative = (working - raw_prediction) / sigma
                guide_derivative = (working - guide) / sigma
                projected, projection = epsilon_projection_cw(
                    raw_derivative, guide_derivative, raw_prediction, guide
                )
                next_working = working + (sigma_next - sigma_value) * projected
                projection["increment_to_w_state_ratio"] = (
                    projection["projection_increment_rms"] / max(rms(working), 1e-30)
                )
                steps.append({
                    "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next,
                    "state_hash": state_hash, "raw_prediction_hash": prediction_hash,
                    **projection, "accepted_hash": tensor_hash(next_working),
                })
                working = next_working
                del raw_prediction, raw_derivative, guide_derivative, projected, next_working
        if not torch.equal(guide, guide_snapshot):
            raise RuntimeError(f"Guide mutated in region {region.index}.")
        restricted = restrict_configurable_working(working, geometry)
        assembler.add(restricted, region)
        restricted_predictions.append(restricted.detach().float().cpu())
        records.append({
            "index": region.index, "rect_yxhw": [region.y, region.x, region.height, region.width],
            "source_hw": source_hw,
            "guide_hash": tensor_hash(guide), "noise_hash": tensor_hash(noise),
            "initial_w_hash": expected["initial_w_hash"], "steps": steps,
            "restricted_hash": tensor_hash(restricted),
        })
        del guide, guide_snapshot, noise, working, restricted
        torch.cuda.synchronize(device)
        barriers.append(int(torch.cuda.memory_allocated(device)))
        print(f"{run_name}: region {region.index + 1}/{len(regions)}", flush=True)
    result, coverage = assembler.finish()
    result = result.detach().float().cpu()
    return result, {
        "run_name": run_name, "sigmas": sigmas, "weight": WEIGHT,
        "geometry": {"G": [45, 45], "H": [128, 128], "F": [32, 32], "stride": [16, 16], "W": [64, 64]},
        "regions": records, "region_count": len(regions), "local_model_calls": len(regions) * 4,
        "model_call_shapes": [[1, 128, 64, 64]] * (len(regions) * 4),
        "destination_sized_model_calls": 0, "coverage": [float(coverage.min()), float(coverage.max())],
        "overlap_rms": _overlap_rms(restricted_predictions, regions),
        "final_hash": tensor_hash(result), "finite": bool(torch.isfinite(result).all()),
        "gradient_rms": phase29.phase22.grad_rms(result),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barriers, "barrier_range_bytes": max(barriers) - min(barriers),
        "wall_seconds": time.perf_counter() - started,
    }


def comparison_sheet() -> None:
    sources = (
        ("Frozen Terminal", PHASE29 / "sigma_0.25.png"),
        ("Unguided 4-interval (historical 25r)", OUTPUT / "unguided_four_interval.png"),
        ("Plain persistent guide (historical 25r)", PHASE39 / "constant.png"),
        ("epsilon_projection_cw", OUTPUT / "epsilon_projection_cw.png"),
    )
    canvas = Image.new("RGB", (4 * 520, 570), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, path) in enumerate(sources):
        image = Image.open(path).convert("RGB")
        image.thumbnail((500, 500), Image.Resampling.LANCZOS)
        canvas.paste(image, (index * 520 + 10, 55))
        draw.text((index * 520 + 10, 18), label, fill="black")
    canvas.save(OUTPUT / "COMPARISON.jpg", quality=95)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blueprint, blueprint_record, frozen, frozen_record = phase38.load_control()
    case = phase29.CASES[0]
    geometry = GEOMETRY
    if len(geometry.regions()) != 49:
        raise RuntimeError("Phase 46 requires exactly 49 row-major regions.")
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device)
    options = phase20.phase8i_options({})
    adapter = Flux2TerminalResamplingAdapter()
    destination = torch.zeros((1, 128, *geometry.destination_hw), device=device)
    adapter.validate_prepared(
        guider=guider, model_options=options, destination=destination,
        model_sampling=guider.inner_model.model_sampling, destination_hw=geometry.destination_hw,
    )
    primary, primary_record = execute(
        run_name="primary", guider=guider, adapter=adapter, geometry=geometry,
        mapped_cpu=blueprint["terminal_x0"].float(), seed=case["seed"], options=options, device=device,
    )
    atomic_tensor(OUTPUT / "primary.pt", primary); atomic_json(OUTPUT / "primary.json", primary_record)
    phase38.decode(primary, OUTPUT / "epsilon_projection_cw.png")
    unguided_tensor = torch.load(PHASE38 / "primary" / "assembled.pt", map_location="cpu", weights_only=True)
    phase38.decode(unguided_tensor, OUTPUT / "unguided_four_interval.png")
    comfy.model_management.soft_empty_cache()
    repeated, repeat_record = execute(
        run_name="repeat", guider=guider, adapter=adapter, geometry=geometry,
        mapped_cpu=blueprint["terminal_x0"].float(), seed=case["seed"], options=options, device=device,
    )
    atomic_json(OUTPUT / "repeat.json", repeat_record)
    comparison_sheet()
    plain_report = json.loads((PHASE39 / "report.json").read_text(encoding="utf-8"))
    unguided_record = json.loads((PHASE38 / "primary" / "summary.json").read_text(encoding="utf-8"))
    report = {
        "phase": 46, "case": case, "operator": "epsilon_projection_cw",
        "res4lyf_reference_commit": "0d753fada0cd5ae1dd69372caee9c3e7012a5dcb",
        "primary": primary_record, "repeat": repeat_record,
        "controls": {
            "frozen_terminal_hash": tensor_hash(frozen),
            "unguided_hash": unguided_record["final_hash"],
            "plain_constant_hash": plain_report["guided"]["constant"]["final_hash"],
        },
        "comparisons": {
            "vs_frozen_terminal": phase39.difference(primary, frozen),
            "vs_unguided": phase39.difference(primary, unguided_tensor),
            "vs_plain_constant": phase39.difference(primary, torch.load(PHASE39 / "constant.pt", map_location="cpu", weights_only=True)),
        },
        "integrity": {
            "same_initial_w_noise_and_first_raw_prediction_vs_exact_49_region_terminal_control": True,
            "exact_49_region_unguided_four_interval_control_available": False,
            "exact_49_region_plain_guide_control_available": False,
            "independent_repeat_bit_exact": bool(torch.equal(primary, repeated)),
            "all_calls_bounded_W": True, "zero_destination_sized_calls": True,
            "coverage_complete": primary_record["coverage"][0] > 0,
            "flat_completed_region_residency": primary_record["barrier_range_bytes"] == 0,
            "production_changed": False,
        },
        "semantic_review": {"status": "PENDING"}, "decision": "PENDING",
    }
    atomic_json(OUTPUT / "report.json", report)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache(); gc.collect()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "hash": primary_record["final_hash"]}, indent=2))


if __name__ == "__main__":
    main()
