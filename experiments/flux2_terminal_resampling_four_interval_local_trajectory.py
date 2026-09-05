"""Phase 38 fixed four-interval terminal-local trajectory discriminator."""
from __future__ import annotations

import hashlib
import json
import math
import os
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
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
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

OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_four_interval_local_trajectory_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
MU = 2.291179894115571
SIGMA_START = 0.25
EXPECTED_CONTROL_HASH = "1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3"
EXPECTED_MAPPED_HASH = "8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05"
REPRESENTATIVE_REGIONS = {0: "background", 6: "car", 12: "tree", 18: "house"}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_tensor(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {"rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def shift(value: float) -> float:
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    e = math.exp(MU)
    return e / (e + (1.0 / value - 1.0))


def schedule() -> tuple[float, ...]:
    e = math.exp(MU)
    u_start = SIGMA_START / (e * (1.0 - SIGMA_START) + SIGMA_START)
    values = [shift(u_start * (1.0 - index / 4.0)) for index in range(5)]
    values[0] = SIGMA_START
    values[-1] = 0.0
    return tuple(values)


def load_control() -> tuple[dict, dict, torch.Tensor, torch.Tensor]:
    blueprint = torch.load(PHASE29 / "blueprint.pt", map_location="cpu", weights_only=True)
    blueprint_record = json.loads((PHASE29 / "blueprint.json").read_text(encoding="utf-8"))
    control = torch.load(PHASE29 / "sigma_0.25.pt", map_location="cpu", weights_only=True).float()
    control_record = json.loads((PHASE29 / "sigma_0.25.json").read_text(encoding="utf-8"))
    if tensor_hash(blueprint["mapped"]) != EXPECTED_MAPPED_HASH or blueprint_record["mapped_hash"] != EXPECTED_MAPPED_HASH:
        raise RuntimeError("Mapped Blueprint fingerprint mismatch")
    if tensor_hash(control) != EXPECTED_CONTROL_HASH or control_record["final_hash"] != EXPECTED_CONTROL_HASH:
        raise RuntimeError("Qualified control fingerprint mismatch")
    if not control_record["phase28_regression"]["bit_exact"]:
        raise RuntimeError("Phase-29 control is not qualified")
    return blueprint, blueprint_record, control, control_record


def execute_regions(*, run_dir: Path, guider, adapter, geometry, mapped_cpu, seed, options, device):
    sigmas = schedule()
    schedule_hash = hashlib.sha256(json.dumps(sigmas).encode("ascii")).hexdigest()
    regions = geometry.regions()
    mapped_hash = tensor_hash(mapped_cpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    region_records = []
    barrier = []
    cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    control_regions = json.loads((PHASE29 / "sigma_0.25.json").read_text(encoding="utf-8"))["regions"]
    for region in regions:
        tensor_path = run_dir / f"region_{region.index:03d}.pt"
        record_path = run_dir / f"region_{region.index:03d}.json"
        expected_noise = region_noise(seed, region, device="cpu", dtype=torch.float32)
        expected_noise_hash = tensor_hash(expected_noise)
        crop_cpu = mapped_cpu[:, :, region.y:region.y2, region.x:region.x2]
        anchor_cpu = lift_region(crop_cpu)
        control_region = control_regions[region.index]
        expected_initial_hash = control_region["working_hash"]
        if control_region["noise_hash"] != expected_noise_hash:
            raise RuntimeError(f"Control W/noise provenance mismatch at region {region.index}")
        if tensor_path.exists() and record_path.exists():
            restricted = torch.load(tensor_path, map_location="cpu", weights_only=True)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if (record["schedule_hash"] != schedule_hash or record["mapped_hash"] != mapped_hash
                    or record["noise_hash"] != expected_noise_hash or record["initial_w_hash"] != expected_initial_hash
                    or tensor_hash(restricted) != record["restricted_hash"]):
                raise RuntimeError(f"Resume fingerprint mismatch at region {region.index}")
            region_records.append(record)
            cuda_ms += record["cuda_ms"]
            barrier.append(record["barrier_allocated_bytes"])
            print(f"resume region {region.index + 1}/{len(regions)}", flush=True)
            continue

        anchor = anchor_cpu.to(device=device)
        noise = expected_noise.to(device=device)
        sigma0 = torch.tensor(sigmas[0], device=device, dtype=anchor.dtype)
        working = guider.inner_model.model_sampling.noise_scaling(sigma0, noise, anchor, False)
        if tensor_hash(working) != expected_initial_hash:
            raise RuntimeError(f"Live initial W mismatch at region {region.index}")
        states = []
        predictions = []
        step_records = []
        region_cuda_ms = 0.0
        with torch.inference_mode():
            for ordinal, (sigma_value, sigma_next_value) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
                sigma = torch.tensor(sigma_value, device=device, dtype=working.dtype)
                if region.index in REPRESENTATIVE_REGIONS:
                    states.append(working.detach().float().cpu())
                state_hash = tensor_hash(working)
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                begin.record()
                prediction = adapter.predict_native(
                    guider=guider, value=working, sigma=sigma,
                    expected_hw=geometry.working_hw, model_options=options, seed=seed,
                )
                end.record(); torch.cuda.synchronize(device)
                elapsed = float(begin.elapsed_time(end))
                region_cuda_ms += elapsed
                prediction_hash = tensor_hash(prediction)
                if region.index in REPRESENTATIVE_REGIONS:
                    predictions.append(prediction.detach().float().cpu())
                velocity = (working - prediction) / sigma
                working = working + (sigma_next_value - sigma_value) * velocity
                step_records.append({
                    "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next_value,
                    "state_hash": state_hash, "prediction_hash": prediction_hash,
                    "accepted_hash": tensor_hash(working), "cuda_ms": elapsed,
                })
                del prediction, velocity
        restricted = restrict_working_prediction(working).detach().float().cpu()
        if region.index in REPRESENTATIVE_REGIONS:
            atomic_tensor(run_dir / f"representative_{region.index:03d}_{REPRESENTATIVE_REGIONS[region.index]}.pt", {
                "sigmas": sigmas, "states": states, "predictions": predictions,
                "final_working": working.detach().float().cpu(),
            })
        del anchor, noise, working
        torch.cuda.synchronize(device)
        allocated = int(torch.cuda.memory_allocated(device))
        record = {
            "index": region.index, "mapped_hash": mapped_hash, "schedule_hash": schedule_hash,
            "noise_hash": expected_noise_hash, "initial_w_hash": expected_initial_hash,
            "control_noise_hash_match": True, "control_initial_w_hash_match": True,
            "steps": step_records, "restricted_hash": tensor_hash(restricted),
            "cuda_ms": region_cuda_ms, "barrier_allocated_bytes": allocated,
        }
        atomic_tensor(tensor_path, restricted)
        atomic_json(record_path, record)
        region_records.append(record)
        cuda_ms += region_cuda_ms
        barrier.append(allocated)
        print(f"persist region {region.index + 1}/{len(regions)}", flush=True)

    template = mapped_cpu.to(device=device)
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw, template=template)
    restricted_cpu = []
    for region in regions:
        value = torch.load(run_dir / f"region_{region.index:03d}.pt", map_location="cpu", weights_only=True)
        restricted_cpu.append(value)
        assembler.add(value.to(device=device), region)
    assembled, coverage = assembler.finish()
    result = assembled.detach().float().cpu()
    summary = {
        "schedule": sigmas, "schedule_hash": schedule_hash, "mapped_hash": mapped_hash,
        "final_hash": tensor_hash(result), "regions": region_records,
        "model_calls_per_region": 4, "total_local_model_calls": 4 * len(regions),
        "destination_sized_model_calls": 0,
        "overlap_rms": phase8d.overlap_metrics(restricted_cpu, regions)["aggregate_rms"],
        "gradient_rms": phase22.grad_rms(result),
        "rms_vs_blueprint": difference(result, mapped_cpu),
        "low_frequency_rms_vs_blueprint": phase22.low_frequency_rms(result, mapped_cpu),
        "coverage": [float(coverage.min()), float(coverage.max())],
        "finite": bool(torch.isfinite(result).all()), "cuda_ms": cuda_ms,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barrier,
    }
    atomic_tensor(run_dir / "assembled.pt", result)
    atomic_json(run_dir / "summary.json", summary)
    return result, summary


def decode(latent: torch.Tensor, path: Path) -> str:
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def make_sheets(control_path: Path, candidate_path: Path) -> None:
    for output_name, detail in (("PHASE38_COMPARISON.jpg", False), ("DETAIL_REVIEW.jpg", True)):
        sheet = Image.new("RGB", (1240, 650), "white")
        for column, (source, label) in enumerate(((control_path, "A | [0.25, 0]"), (candidate_path, "B | four intervals"))):
            image = Image.open(source).convert("RGB")
            if detail:
                image = image.crop((image.width // 8, image.height // 8, image.width * 7 // 8, image.height * 7 // 8))
            image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (620, 650), "white")
            panel.paste(image, ((620 - image.width) // 2, 40))
            ImageDraw.Draw(panel).text((8, 10), label, fill="black")
            sheet.paste(panel, (620 * column, 0))
        sheet.save(OUTPUT / output_name, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blueprint, blueprint_record, control, control_record = load_control()
    sigmas = schedule()
    expected = (0.25, 0.1986604057521303, 0.1408223197117642, 0.0751685213650775, 0.0)
    if max(abs(a - b) for a, b in zip(sigmas, expected, strict=True)) > 1e-15:
        raise RuntimeError(f"Schedule reproduction mismatch: {sigmas}")
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
    print("[1/2] primary four-interval run", flush=True)
    primary, primary_record = execute_regions(
        run_dir=OUTPUT / "primary", guider=guider, adapter=adapter, geometry=geometry,
        mapped_cpu=blueprint["mapped"].float(), seed=case["seed"], options=options, device=device,
    )
    print("[2/2] deterministic repeat", flush=True)
    repeated, repeat_record = execute_regions(
        run_dir=OUTPUT / "repeat", guider=guider, adapter=adapter, geometry=geometry,
        mapped_cpu=blueprint["mapped"].float(), seed=case["seed"], options=options, device=device,
    )
    repeat = {"bit_exact": bool(torch.equal(primary, repeated)), "difference": difference(primary, repeated)}
    control_png = PHASE29 / "sigma_0.25.png"
    candidate_png = OUTPUT / "four_interval.png"
    decoded_hash = decode(primary, candidate_png)
    make_sheets(control_png, candidate_png)
    report = {
        "phase": 38,
        "policy": "one newly authorized empirical Blueprint schedule; not canonical Klein partial denoising",
        "case": case, "mu": MU, "sigma_start": SIGMA_START,
        "unshifted_start": SIGMA_START / (math.exp(MU) * (1.0 - SIGMA_START) + SIGMA_START),
        "executed_sigmas": sigmas,
        "control": {
            "final_hash": tensor_hash(control), "expected_hash": EXPECTED_CONTROL_HASH,
            "bit_exact": tensor_hash(control) == EXPECTED_CONTROL_HASH,
            "semantic_grade": "S3 (persisted qualified review)",
            "gradient_rms": control_record["gradient_rms"], "overlap_rms": control_record["overlap_rms"],
            "rms_vs_blueprint": control_record["rms_vs_blueprint"],
            "low_frequency_rms_vs_blueprint": control_record["low_frequency_rms_vs_blueprint"],
            "local_calls": control_record["local_calls"],
            "local_cuda_ms": control_record["local_cuda_ms"], "wall_seconds": control_record["wall_seconds"],
            "peak_allocated_bytes": control_record["peak_allocated_bytes"],
            "peak_reserved_bytes": control_record["peak_reserved_bytes"],
        },
        "experimental": {**primary_record, "decoded_rgb_hash": decoded_hash},
        "repeat": repeat,
        "comparison": {"candidate_vs_control": difference(primary, control)},
        "integrity": {
            "blueprint_hash": blueprint_record["terminal_hash"], "mapped_hash": blueprint_record["mapped_hash"],
            "control_initial_w_and_noise_match_all_regions": all(r["control_initial_w_hash_match"] and r["control_noise_hash_match"] for r in primary_record["regions"]),
            "production_changed": False, "comfyui_core_changed": False,
        },
        "semantic_review": {
            "control_grade": "S3",
            "experimental_grade": "S3",
            "object_count": "exactly one car, one dominant tree, and one house in both arms",
            "scene_integrity": "coherent shared field and horizon; no independent local recomposition",
            "detail_judgment": "The four-interval arm adds denser line/texture energy and minor contour changes, but does not credibly resolve wheels, car body/window structure, foliage organization, roof/window structure, or grass detail.",
            "artifact_judgment": "The increased gradient energy reads predominantly as fine alias/texture density rather than new semantic structure.",
        },
        "decision": "B — CHANGES OUTPUT BUT DOES NOT SOLVE SOFTNESS",
    }
    atomic_json(OUTPUT / "report.json", report)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "repeat_bit_exact": repeat["bit_exact"]}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    control_record = json.loads((PHASE29 / "sigma_0.25.json").read_text(encoding="utf-8"))
    for key in ("local_cuda_ms", "wall_seconds", "peak_allocated_bytes", "peak_reserved_bytes"):
        report["control"][key] = control_record[key]
    report["semantic_review"] = {
        "control_grade": "S3",
        "experimental_grade": "S3",
        "object_count": "exactly one car, one dominant tree, and one house in both arms",
        "scene_integrity": "coherent shared field and horizon; no independent local recomposition",
        "detail_judgment": "The four-interval arm adds denser line/texture energy and minor contour changes, but does not credibly resolve wheels, car body/window structure, foliage organization, roof/window structure, or grass detail.",
        "artifact_judgment": "The increased gradient energy reads predominantly as fine alias/texture density rather than new semantic structure.",
    }
    report["decision"] = "B — CHANGES OUTPUT BUT DOES NOT SOLVE SOFTNESS"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
