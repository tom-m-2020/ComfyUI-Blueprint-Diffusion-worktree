"""Fixed persistent-coarse-guide discriminator; no production integration."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch

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
import flux2_terminal_resampling_four_interval_local_trajectory as phase38
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.terminal_resampling import (
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    lift_region,
    region_noise,
    restrict_working_prediction,
    tensor_hash,
)


OUTPUT = ROOT / "experiments" / "flux2_persistent_coarse_guidance_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
PHASE38 = ROOT / "experiments" / "flux2_terminal_resampling_four_interval_local_trajectory_results"
POLICIES = {
    "constant": (0.25, 0.25, 0.25, 0.25),
    "release": (0.5, 1.0 / 3.0, 1.0 / 6.0, 0.0),
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    delta = left.detach().float() - right.detach().float()
    return {"rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def guided_prediction(model_prediction: torch.Tensor, guide: torch.Tensor, weight: float) -> torch.Tensor:
    if not 0.0 <= weight <= 1.0 or model_prediction.shape != guide.shape:
        raise ValueError("Invalid persistent coarse guide contract.")
    return model_prediction + weight * (guide - model_prediction)


def execute_policy(*, name, weights, guider, adapter, geometry, mapped_cpu, seed, options, device):
    sigmas = phase38.schedule()
    regions = geometry.regions()
    control_regions = json.loads((PHASE29 / "sigma_0.25.json").read_text(encoding="utf-8"))["regions"]
    assembler = StreamingOverlapAssembler(
        regions=regions, target_hw=geometry.destination_hw, template=mapped_cpu.to(device)
    )
    records = []
    barriers = []
    restricted_predictions = []
    total_cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for region in regions:
        guide = lift_region(mapped_cpu[:, :, region.y:region.y2, region.x:region.x2]).to(device)
        guide_hash = tensor_hash(guide)
        guide_snapshot = guide.clone()
        noise = region_noise(seed, region, device=device, dtype=guide.dtype)
        sigma0 = torch.tensor(sigmas[0], device=device, dtype=guide.dtype)
        working = guider.inner_model.model_sampling.noise_scaling(sigma0, noise, guide, False)
        control = control_regions[region.index]
        if tensor_hash(noise) != control["noise_hash"] or tensor_hash(working) != control["working_hash"]:
            raise RuntimeError(f"Initial W/noise provenance mismatch at region {region.index}.")
        initial_hash = tensor_hash(working)
        steps = []
        with torch.inference_mode():
            for ordinal, (sigma_value, sigma_next, weight) in enumerate(
                zip(sigmas[:-1], sigmas[1:], weights, strict=True)
            ):
                sigma = torch.tensor(sigma_value, device=device, dtype=working.dtype)
                state_hash = tensor_hash(working)
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                begin.record()
                model_prediction = adapter.predict_native(
                    guider=guider, value=working, sigma=sigma,
                    expected_hw=geometry.working_hw, model_options=options, seed=seed,
                )
                end.record(); torch.cuda.synchronize(device)
                elapsed = float(begin.elapsed_time(end))
                total_cuda_ms += elapsed
                model_hash = tensor_hash(model_prediction)
                used_prediction = guided_prediction(model_prediction, guide, weight)
                guide_delta = difference(used_prediction, model_prediction)
                velocity = (working - used_prediction) / sigma
                working = working + (sigma_next - sigma_value) * velocity
                steps.append({
                    "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next,
                    "weight": weight, "state_hash": state_hash,
                    "model_prediction_hash": model_hash,
                    "used_prediction_hash": tensor_hash(used_prediction),
                    "guide_delta": guide_delta, "accepted_hash": tensor_hash(working),
                    "cuda_ms": elapsed,
                })
                del model_prediction, used_prediction, velocity
        if not torch.equal(guide, guide_snapshot):
            raise RuntimeError(f"Guide mutated at region {region.index}.")
        restricted = restrict_working_prediction(working)
        assembler.add(restricted, region)
        restricted_predictions.append(restricted.detach().float().cpu())
        records.append({
            "index": region.index, "guide_hash": guide_hash,
            "initial_w_hash": initial_hash, "noise_hash": control["noise_hash"],
            "steps": steps, "restricted_hash": tensor_hash(restricted),
        })
        del guide, guide_snapshot, noise, working, restricted
        torch.cuda.synchronize(device)
        barriers.append(int(torch.cuda.memory_allocated(device)))
        print(f"{name}: region {region.index + 1}/{len(regions)}", flush=True)
    result, coverage = assembler.finish()
    result_cpu = result.detach().float().cpu()
    summary = {
        "name": name, "weights": weights, "sigmas": sigmas,
        "final_hash": tensor_hash(result_cpu), "regions": records,
        "local_model_calls": len(regions) * len(weights),
        "destination_sized_model_calls": 0,
        "coverage": [float(coverage.min()), float(coverage.max())],
        "finite": bool(torch.isfinite(result_cpu).all()),
        "gradient_rms": phase29.phase22.grad_rms(result_cpu),
        "overlap_rms": phase8d.overlap_metrics(restricted_predictions, regions)["aggregate_rms"],
        "rms_vs_blueprint": difference(result_cpu, mapped_cpu),
        "low_frequency_rms_vs_blueprint": phase29.phase22.low_frequency_rms(result_cpu, mapped_cpu),
        "cuda_ms": total_cuda_ms, "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barriers,
    }
    return result_cpu, summary


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blueprint, blueprint_record, frozen, frozen_record = phase38.load_control()
    unguided = torch.load(PHASE38 / "primary" / "assembled.pt", map_location="cpu", weights_only=True).float()
    unguided_record = json.loads((PHASE38 / "primary" / "summary.json").read_text(encoding="utf-8"))
    case = phase29.CASES[0]
    geometry = TerminalResamplingGeometry.for_destination(case["destination_hw"])
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
    arms = {}
    tensors = {}
    previous = {}
    for name in POLICIES:
        path = OUTPUT / f"{name}.pt"
        if path.exists():
            previous[name] = torch.load(path, map_location="cpu", weights_only=True).float()
    for name, weights in POLICIES.items():
        result, record = execute_policy(
            name=name, weights=weights, guider=guider, adapter=adapter, geometry=geometry,
            mapped_cpu=blueprint["mapped"].float(), seed=case["seed"], options=options, device=device,
        )
        path = OUTPUT / f"{name}.pt"
        atomic_tensor(path, result)
        phase38.decode(result, OUTPUT / f"{name}.png")
        arms[name] = record
        tensors[name] = result
    report = {
        "experiment": "persistent_coarse_guidance",
        "res4lyf_reference_commit": "0d753fada0cd5ae1dd69372caee9c3e7012a5dcb",
        "intervention": "post-model denoised prediction / sampler velocity",
        "case": case, "mapped_blueprint_hash": blueprint_record["mapped_hash"],
        "frozen_terminal": {
            "hash": tensor_hash(frozen), "expected_hash": phase38.EXPECTED_CONTROL_HASH,
            "semantic_grade": "S3 (persisted qualified review)",
            "local_calls": frozen_record["local_calls"],
        },
        "unguided_four_interval": {
            "hash": tensor_hash(unguided), "expected_hash": unguided_record["final_hash"],
            "semantic_grade": "S3 (persisted qualified review)",
            "local_calls": unguided_record["total_local_model_calls"],
        },
        "guided": arms,
        "comparisons": {
            name: {
                "vs_frozen_terminal": difference(value, frozen),
                "vs_unguided_four_interval": difference(value, unguided),
            } for name, value in tensors.items()
        },
        "integrity": {
            "production_changed": False, "comfyui_core_changed": False,
            "same_initial_w_and_noise": all(
                record["initial_w_hash"] == frozen_record["regions"][record["index"]]["working_hash"]
                and record["noise_hash"] == frozen_record["regions"][record["index"]]["noise_hash"]
                for arm in arms.values() for record in arm["regions"]
            ),
            "independent_repeat_bit_exact": {
                name: bool(name in previous and torch.equal(value, previous[name]))
                for name, value in tensors.items()
            },
        },
        "semantic_review": {
            "constant_grade": "S3",
            "release_grade": "S3",
            "composition": "Both retain exactly one red car, one central tree, and one white house on one continuous field/horizon.",
            "detail": "Neither guided arm credibly resolves wheels, windows, car contour, foliage organization, roof structure, or ground detail beyond the frozen or unguided controls.",
            "effect": "Guidance pulls the trajectory toward the soft mapped Blueprint, reducing gradient energy and overlap disagreement rather than expanding the composition/detail frontier.",
        },
        "decision": "B — S3 RETAINED, NO CREDIBLE DETAIL IMPROVEMENT; STOP PERSISTENT-GUIDE TUNING",
    }
    atomic_json(OUTPUT / "report.json", report)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "hashes": {k: v["final_hash"] for k, v in arms.items()}}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    unguided = json.loads((PHASE38 / "primary" / "summary.json").read_text(encoding="utf-8"))
    report["integrity"]["first_model_prediction_matches_unguided_all_regions"] = all(
        report["guided"]["constant"]["regions"][index]["steps"][0]["model_prediction_hash"]
        == report["guided"]["release"]["regions"][index]["steps"][0]["model_prediction_hash"]
        == unguided["regions"][index]["steps"][0]["prediction_hash"]
        for index in range(len(unguided["regions"]))
    )
    report["semantic_review"] = {
        "constant_grade": "S3",
        "release_grade": "S3",
        "composition": "Both retain exactly one red car, one central tree, and one white house on one continuous field/horizon.",
        "detail": "Neither guided arm credibly resolves wheels, windows, car contour, foliage organization, roof structure, or ground detail beyond the frozen or unguided controls.",
        "effect": "Guidance pulls the trajectory toward the soft mapped Blueprint, reducing gradient energy and overlap disagreement rather than expanding the composition/detail frontier.",
    }
    report["decision"] = "B — S3 RETAINED, NO CREDIBLE DETAIL IMPROVEMENT; STOP PERSISTENT-GUIDE TUNING"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
