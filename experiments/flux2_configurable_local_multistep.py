"""Experiment-only local schedule-depth discriminator for configurable Blueprint."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments"), str(PACKAGE)]

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.configurable_resampling import (
    ConfigurableResamplingGeometry,
    StreamingOverlapMetric,
    bounded_blueprint_transfer,
    configurable_region_noise,
    initialize_configurable_blueprint,
    restrict_configurable_working,
)
from blueprint_diffusion.terminal_resampling import (
    BlueprintRunState,
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_configurable_local_multistep_results"
BASELINE = ROOT / "experiments" / "flux2_configurable_blueprint_prototype_results" / "report.json"
GEOMETRY = ConfigurableResamplingGeometry((45, 45), (128, 128), (32, 32), (16, 16), (64, 64))
SEED = phase29.CASES[0]["seed"]
PROMPT = phase29.CASES[0]["prompt"]
SIGMA_START = 0.25
MU = 2.291179894115571


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def shift(value: float) -> float:
    if value == 0.0:
        return 0.0
    e = math.exp(MU)
    return e / (e + (1.0 / value - 1.0))


def local_schedule(intervals: int) -> tuple[float, ...]:
    if type(intervals) is not int or intervals < 1:
        raise ValueError("Local interval count must be a positive integer.")
    e = math.exp(MU)
    unshifted_start = SIGMA_START / (e * (1.0 - SIGMA_START) + SIGMA_START)
    values = [shift(unshifted_start * (1.0 - ordinal / intervals))
              for ordinal in range(intervals + 1)]
    values[0], values[-1] = SIGMA_START, 0.0
    return tuple(values)


def accepted_local_update(working: torch.Tensor, prediction: torch.Tensor,
                          sigma: torch.Tensor, sigma_next: float) -> torch.Tensor:
    if sigma_next == 0.0:
        return prediction
    return working + (sigma_next - float(sigma)) * (working - prediction) / sigma


SCHEDULES = {
    "A_ONE_INTERVAL": local_schedule(1),
    "B_TWO_INTERVALS": local_schedule(2),
    "C_THREE_INTERVALS": local_schedule(3),
}


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    delta = left.detach().float() - right.detach().float()
    return {"bit_exact": bool(torch.equal(left, right)),
            "rms": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max())}


def laplacian_rms(value: torch.Tensor) -> float:
    channels = value.shape[1]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                          dtype=torch.float32).reshape(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    response = F.conv2d(value.detach().float().cpu(), kernel, padding=1, groups=channels)
    return float(response.square().mean().sqrt())


def terminal_blueprint(guider, adapter, options, device) -> tuple[torch.Tensor, list[dict]]:
    sigmas = torch.tensor(QUALIFIED_SIGMAS, device=device)
    state = BlueprintRunState(initialize_configurable_blueprint(SEED, GEOMETRY, device=device),
                              float(sigmas[0]), 0, "multistep:initial")
    records = []
    terminal = None
    with torch.inference_mode():
        for ordinal in range(4):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            prediction = adapter.predict_native(
                guider=guider, value=state.blueprint, sigma=sigma,
                expected_hw=GEOMETRY.blueprint_hw, model_options=options, seed=SEED,
            )
            proposal = state.blueprint + (sigma_next - sigma) * (state.blueprint - prediction) / sigma
            records.append({"ordinal": ordinal, "sigma": float(sigma),
                            "sigma_next": float(sigma_next),
                            "input_hash": tensor_hash(state.blueprint),
                            "prediction_hash": tensor_hash(prediction),
                            "accepted_hash": tensor_hash(proposal)})
            terminal = prediction
            state = BlueprintRunState(proposal, float(sigma_next), ordinal + 1,
                                      f"multistep:{ordinal}:accepted")
    return terminal.detach().float().cpu(), records


def execute_arm(name, sigmas, terminal_cpu, guider, adapter, options, device):
    arm_dir = OUTPUT / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    regions = GEOMETRY.regions()
    records = []
    barriers = []
    local_cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    for region in regions:
        tensor_path = arm_dir / f"region_{region.index:03d}.pt"
        record_path = arm_dir / f"region_{region.index:03d}.json"
        if tensor_path.exists() and record_path.exists():
            restricted = torch.load(tensor_path, map_location="cpu", weights_only=True)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record["schedule"] != list(sigmas) or tensor_hash(restricted) != record["restricted_hash"]:
                raise RuntimeError(f"Resume fingerprint mismatch for {name} region {region.index}")
            records.append(record)
            barriers.append(record["barrier_allocated_bytes"])
            local_cuda_ms += record["cuda_ms"]
            print(f"resume {name} {region.index + 1}/{len(regions)}", flush=True)
            continue

        anchor_cpu, source_hw = bounded_blueprint_transfer(terminal_cpu, GEOMETRY, region)
        noise_cpu = configurable_region_noise(SEED, region, GEOMETRY.working_hw,
                                              device="cpu", dtype=torch.float32)
        anchor = anchor_cpu.to(device=device)
        noise = noise_cpu.to(device=device)
        sigma0 = torch.tensor(sigmas[0], device=device, dtype=anchor.dtype)
        working = guider.inner_model.model_sampling.noise_scaling(sigma0, noise, anchor, False)
        initial_hash = tensor_hash(working)
        steps = []
        region_cuda_ms = 0.0
        with torch.inference_mode():
            for ordinal, (sigma_value, sigma_next_value) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
                sigma = torch.tensor(sigma_value, device=device, dtype=working.dtype)
                state_hash = tensor_hash(working)
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                begin.record()
                prediction = adapter.predict_native(
                    guider=guider, value=working, sigma=sigma,
                    expected_hw=GEOMETRY.working_hw, model_options=options, seed=SEED,
                )
                end.record(); torch.cuda.synchronize(device)
                elapsed = float(begin.elapsed_time(end))
                region_cuda_ms += elapsed
                prediction_hash = tensor_hash(prediction)
                accepted = accepted_local_update(working, prediction, sigma, sigma_next_value)
                steps.append({
                    "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next_value,
                    "state_hash": state_hash, "prediction_hash": prediction_hash,
                    "accepted_hash": tensor_hash(accepted), "cuda_ms": elapsed,
                })
                working = accepted
        restricted = restrict_configurable_working(working, GEOMETRY).detach().float().cpu()
        del anchor, noise, working, prediction, accepted
        torch.cuda.synchronize(device)
        record = {
            "index": region.index, "schedule": list(sigmas), "source_hw": source_hw,
            "noise_hash": tensor_hash(noise_cpu), "initial_w_hash": initial_hash,
            "steps": steps, "restricted_hash": tensor_hash(restricted),
            "cuda_ms": region_cuda_ms,
            "barrier_allocated_bytes": int(torch.cuda.memory_allocated(device)),
        }
        atomic_tensor(tensor_path, restricted)
        atomic_json(record_path, record)
        records.append(record)
        barriers.append(record["barrier_allocated_bytes"])
        local_cuda_ms += region_cuda_ms
        print(f"persist {name} {region.index + 1}/{len(regions)}", flush=True)

    template = torch.zeros((1, 128, *GEOMETRY.destination_hw), device=device)
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=GEOMETRY.destination_hw,
                                          template=template)
    overlap = StreamingOverlapMetric()
    for region in regions:
        restricted = torch.load(arm_dir / f"region_{region.index:03d}.pt",
                                map_location="cpu", weights_only=True)
        assembler.add(restricted.to(device=device), region)
        overlap.add(restricted, region)
    assembled, coverage = assembler.finish()
    result = assembled.detach().float().cpu()
    summary = {
        "name": name, "schedule": list(sigmas), "interval_count": len(sigmas) - 1,
        "final_hash": tensor_hash(result), "regions": records,
        "model_calls": {"G_shared": 4, "W": len(regions) * (len(sigmas) - 1), "H": 0},
        "coverage": [float(coverage.min()), float(coverage.max())],
        "overlap_rms": overlap.finish(), "gradient_rms": phase22.grad_rms(result),
        "laplacian_rms": laplacian_rms(result), "local_cuda_ms": local_cuda_ms,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "region_barrier_allocated_bytes": barriers,
        "barrier_range_bytes": max(barriers) - min(barriers),
        "working_model_hw": list(GEOMETRY.working_hw),
    }
    atomic_tensor(arm_dir / "assembled.pt", result)
    atomic_json(arm_dir / "summary.json", summary)
    return result, summary


def decode(vae, latent, path):
    pixels = vae.decode(latent)
    phase2.save_pixels(pixels.cpu(), path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def make_comparison() -> None:
    sheet = Image.new("RGB", (1860, 650), "white")
    labels = (("A_ONE_INTERVAL", "A | [0.25, 0]"),
              ("B_TWO_INTERVALS", "B | two intervals"),
              ("C_THREE_INTERVALS", "C | three intervals"))
    for column, (name, label) in enumerate(labels):
        image = Image.open(OUTPUT / f"{name}.png").convert("RGB")
        image.thumbnail((600, 600), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (620, 650), "white")
        panel.paste(image, ((620 - image.width) // 2, 40))
        ImageDraw.Draw(panel).text((8, 10), label, fill="black")
        sheet.paste(panel, (620 * column, 0))
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_case = next(item for item in baseline["cases"] if item["name"] == "SQUARE_OVERLAP16")
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device)
    options = phase20.phase8i_options({})
    adapter = Flux2TerminalResamplingAdapter()
    destination = torch.zeros((1, 128, *GEOMETRY.destination_hw), device=device)
    adapter.validate_prepared(guider=guider, model_options=options, destination=destination,
                              model_sampling=guider.inner_model.model_sampling,
                              destination_hw=GEOMETRY.destination_hw)
    terminal, g_records = terminal_blueprint(guider, adapter, options, device)
    results, summaries = {}, []
    for name, sigmas in SCHEDULES.items():
        result, summary = execute_arm(name, sigmas, terminal, guider, adapter, options, device)
        results[name] = result
        summaries.append(summary)

    control = results["A_ONE_INTERVAL"]
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    for summary in summaries:
        result = results[summary["name"]]
        summary["decoded_rgb_hash"] = decode(vae, result, OUTPUT / f"{summary['name']}.png")
        summary["difference_vs_control"] = difference(result, control)
        summary["low_frequency_rms_vs_control"] = phase22.low_frequency_rms(result, control)
    make_comparison()

    by_name = {item["name"]: item for item in summaries}
    control_regions = by_name["A_ONE_INTERVAL"]["regions"]
    isolation = {}
    for name in ("B_TWO_INTERVALS", "C_THREE_INTERVALS"):
        candidate_regions = by_name[name]["regions"]
        isolation[name] = {
            "noise_hashes_identical": all(a["noise_hash"] == b["noise_hash"] for a, b in zip(control_regions, candidate_regions)),
            "initial_w_hashes_identical": all(a["initial_w_hash"] == b["initial_w_hash"] for a, b in zip(control_regions, candidate_regions)),
            "first_prediction_hashes_identical": all(a["steps"][0]["prediction_hash"] == b["steps"][0]["prediction_hash"] for a, b in zip(control_regions, candidate_regions)),
            "first_state_hashes_identical": all(a["steps"][0]["state_hash"] == b["steps"][0]["state_hash"] for a, b in zip(control_regions, candidate_regions)),
        }
    report = {
        "experiment": "configurable_local_multistep_discriminator",
        "scope": "local W schedule only; experiment-only",
        "geometry": {"G": GEOMETRY.blueprint_hw, "H": GEOMETRY.destination_hw,
                     "F": GEOMETRY.footprint_hw, "stride": GEOMETRY.stride_hw,
                     "W": GEOMETRY.working_hw, "regions": len(GEOMETRY.regions())},
        "seed": SEED, "prompt": PROMPT, "mu": MU, "schedules": SCHEDULES,
        "shared_G": {"terminal_x0_hash": tensor_hash(terminal), "steps": g_records},
        "arms": summaries, "isolation": isolation,
        "control_regression": {"expected_hash": baseline_case["latent_hash"],
                               "observed_hash": by_name["A_ONE_INTERVAL"]["final_hash"],
                               "bit_exact": baseline_case["latent_hash"] == by_name["A_ONE_INTERVAL"]["final_hash"]},
        "integrity": {
            "all_isolation_checks_pass": all(all(value.values()) for value in isolation.values()),
            "all_coverage_complete": all(item["coverage"][0] > 0 for item in summaries),
            "all_destination_calls_zero": all(item["model_calls"]["H"] == 0 for item in summaries),
            "all_working_geometry_bounded": all(item["working_model_hw"] == [64, 64] for item in summaries),
            "all_region_barriers_flat": all(item["barrier_range_bytes"] == 0 for item in summaries),
            "production_changed": False,
        },
        "semantic_review": {"status": "PENDING"}, "decision": "PENDING",
    }
    atomic_json(OUTPUT / "report.json", report)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "integrity": report["integrity"]}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["semantic_review"] = {
        "A_ONE_INTERVAL": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "horizon": "continuous", "miniature_scene_repetition": False,
            "detail": "qualified configurable-prototype control",
        },
        "B_TWO_INTERVALS": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "horizon": "continuous", "miniature_scene_repetition": False,
            "detail": "slightly stronger fine edges and texture, but no credible improvement in car contour, wheels, foliage organization, house edges, or ground structure",
        },
        "C_THREE_INTERVALS": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "horizon": "continuous", "miniature_scene_repetition": False,
            "detail": "largest line and texture increase, but it reads as repeated fine edging rather than newly resolved semantic or local structure",
        },
        "frontier": "Neither deeper schedule materially improves credible detail over the one-step control; overlap disagreement increases with depth.",
    }
    report["decision"] = "REJECT — MULTI-STEP LOCAL DEPTH DOES NOT IMPROVE THE CREDIBLE DETAIL FRONTIER"
    report["production_recommendation"] = "Do not add a local-step control to Blueprint Configurable Prototype."
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
