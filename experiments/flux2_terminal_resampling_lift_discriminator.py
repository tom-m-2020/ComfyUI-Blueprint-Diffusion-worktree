"""Phase 31 nearest-versus-bilinear terminal-resampling lift discriminator."""
from __future__ import annotations

import hashlib
import json
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

OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_lift_discriminator_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results"
SEMANTIC_REVIEW = {
    "SQUARE_MULTI_OBJECT": {
        "control_grade": "S3", "bilinear_grade": "S3",
        "finding": "One car/tree/house scene is retained, but car, foliage, roof/window, and grass detail is not materially improved; bilinear is slightly smoother.",
    },
    "PORTRAIT_ASTRONAUT": {
        "control_grade": "S3", "bilinear_grade": "S3",
        "finding": "One astronaut remains coherent, but helmet, suit panels, limb boundaries, shadow, and ground detail are not improved; boundaries are slightly smoother.",
    },
    "LANDSCAPE_BRIDGE": {
        "control_grade": "S3", "bilinear_grade": "S3",
        "finding": "One bridge scene remains coherent, but tower, cable, deck, secondary-tower, and water detail is not improved; fine lines are slightly smoother.",
    },
}


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {"rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def validate_control(case: dict, blueprint_record: dict) -> tuple[torch.Tensor, dict]:
    case_dir = PHASE29 / case["name"]
    record = json.loads((case_dir / "sigma_0.25.json").read_text(encoding="utf-8"))
    latent = torch.load(case_dir / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    if record["sigma"] != 0.25 or record["mapped_hash"] != blueprint_record["mapped_hash"]:
        raise RuntimeError(f"Phase-29 control fingerprint mismatch for {case['name']}")
    if not record["phase28_regression"]["bit_exact"]:
        raise RuntimeError(f"Phase-28 regression is not exact for {case['name']}")
    if tensor_hash(latent) != record["final_hash"] or not torch.isfinite(latent).all():
        raise RuntimeError(f"Phase-29 control latent invalid for {case['name']}")
    return latent, record


def anchor_diagnostics(mapped: torch.Tensor, geometry: TerminalResamplingGeometry) -> list[dict]:
    records = []
    for region in geometry.regions():
        crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
        nearest = lift_region(crop)
        bilinear = F.interpolate(crop, scale_factor=2.0, mode="bilinear", align_corners=False)
        nearest_down = F.avg_pool2d(nearest, 2, 2)
        bilinear_down = F.avg_pool2d(bilinear, 2, 2)
        records.append({
            "index": region.index,
            "crop_hash": tensor_hash(crop),
            "nearest_hash": tensor_hash(nearest),
            "bilinear_hash": tensor_hash(bilinear),
            "bilinear_vs_nearest": difference(bilinear, nearest),
            "nearest_restriction_vs_crop": difference(nearest_down, crop),
            "bilinear_restriction_vs_crop": difference(bilinear_down, crop),
            "nearest_low_frequency_error": phase22.low_frequency_rms(nearest_down, crop),
            "bilinear_low_frequency_error": phase22.low_frequency_rms(bilinear_down, crop),
            "nearest_gradient_rms": phase22.grad_rms(nearest),
            "bilinear_gradient_rms": phase22.grad_rms(bilinear),
        })
    return records


def execute_bilinear(*, guider, adapter, geometry, seed, mapped_cpu, options, device) -> tuple[torch.Tensor, dict]:
    regions = geometry.regions()
    mapped = mapped_cpu.to(device=device)
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw, template=mapped)
    sigma = torch.tensor(0.25, device=device, dtype=mapped.dtype)
    restricted_cpu = []
    region_records = []
    barrier = []
    cuda_ms = 0.0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for region in regions:
            crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            anchor = F.interpolate(crop, scale_factor=2.0, mode="bilinear", align_corners=False)
            noise = region_noise(seed, region, device=device, dtype=anchor.dtype)
            working = guider.inner_model.model_sampling.noise_scaling(sigma, noise, anchor, False)
            expected = 0.75 * anchor + 0.25 * noise
            construction_error = float((working.float() - expected.float()).abs().max())
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            prediction = adapter.predict_native(
                guider=guider, value=working, sigma=sigma,
                expected_hw=geometry.working_hw, model_options=options, seed=seed,
            )
            end.record()
            torch.cuda.synchronize(device)
            cuda_ms += float(begin.elapsed_time(end))
            restricted = restrict_working_prediction(prediction)
            assembler.add(restricted, region)
            restricted_cpu.append(restricted.detach().float().cpu())
            region_records.append({
                "index": region.index,
                "seed": seed + 22_000_003 + 1009 * region.index,
                "blueprint_reference_hash": tensor_hash(crop),
                "anchor_hash": tensor_hash(anchor),
                "noise_hash": tensor_hash(noise),
                "working_hash": tensor_hash(working),
                "free_prediction_hash": tensor_hash(prediction),
                "restricted_hash": tensor_hash(restricted),
                "construction_max_abs": construction_error,
            })
            del crop, anchor, noise, working, expected, prediction, restricted
            torch.cuda.synchronize(device)
            barrier.append(int(torch.cuda.memory_allocated(device)))
    assembled, coverage = assembler.finish()
    assembled_cpu = assembled.detach().float().cpu()
    record = {
        "arm": "B_FIXED_BILINEAR_LIFT",
        "sigma": 0.25,
        "mapped_hash": tensor_hash(mapped_cpu),
        "final_hash": tensor_hash(assembled_cpu),
        "regions": region_records,
        "local_calls": len(regions),
        "blueprint_calls_recomputed": 0,
        "destination_model_calls": 0,
        "overlap_rms": phase8d.overlap_metrics(restricted_cpu, regions)["aggregate_rms"],
        "gradient_rms": phase22.grad_rms(assembled_cpu),
        "rms_vs_blueprint": difference(assembled_cpu, mapped_cpu),
        "low_frequency_rms_vs_blueprint": phase22.low_frequency_rms(assembled_cpu, mapped_cpu),
        "coverage": [float(coverage.min()), float(coverage.max())],
        "finite": bool(torch.isfinite(assembled_cpu).all()),
        "local_cuda_ms": cuda_ms,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barrier,
    }
    return assembled_cpu, record


def decode_results() -> dict:
    vae = None
    hashes = {}
    for case in phase29.CASES:
        case_dir = OUTPUT / case["name"]
        path = case_dir / "bilinear.png"
        if not path.exists():
            if vae is None:
                vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
            latent = torch.load(case_dir / "bilinear.pt", map_location="cpu", weights_only=True)
            phase2.save_pixels(vae.decode(latent).cpu(), path)
        with Image.open(path) as image:
            hashes[case["name"]] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    return hashes


def panel(source: Path, label: str, *, detail: bool = False) -> Image.Image:
    image = Image.open(source).convert("RGB")
    if detail:
        left, top = image.width // 5, image.height // 5
        image = image.crop((left, top, image.width - left, image.height - top))
    image.thumbnail((600, 470), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (620, 515), "white")
    result.paste(image, ((620 - image.width) // 2, 38))
    ImageDraw.Draw(result).text((8, 8), label, fill="black")
    return result


def build_sheet(name: str, *, detail: bool = False) -> None:
    sheet = Image.new("RGB", (1240, 515 * len(phase29.CASES)), "white")
    for row, case in enumerate(phase29.CASES):
        control = panel(PHASE29 / case["name"] / "sigma_0.25.png", f"{case['name']} | A nearest", detail=detail)
        candidate = panel(OUTPUT / case["name"] / "bilinear.png", f"{case['name']} | B bilinear", detail=detail)
        sheet.paste(control, (0, row * 515))
        sheet.paste(candidate, (620, row * 515))
    sheet.save(OUTPUT / name, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for ordinal, case in enumerate(phase29.CASES, 1):
        print(f"[{ordinal}/3] {case['name']} validate/reuse", flush=True)
        case_dir = OUTPUT / case["name"]
        case_dir.mkdir(exist_ok=True)
        geometry = TerminalResamplingGeometry.for_destination(case["destination_hw"])
        blueprint = torch.load(PHASE29 / case["name"] / "blueprint.pt", map_location="cpu", weights_only=True)
        blueprint_record = json.loads((PHASE29 / case["name"] / "blueprint.json").read_text(encoding="utf-8"))
        mapped = blueprint["mapped"]
        if tensor_hash(mapped) != blueprint_record["mapped_hash"]:
            raise RuntimeError(f"Blueprint fingerprint mismatch for {case['name']}")
        control, control_record = validate_control(case, blueprint_record)
        diagnostics = anchor_diagnostics(mapped, geometry)
        artifact_path = case_dir / "bilinear.pt"
        record_path = case_dir / "bilinear.json"
        if artifact_path.exists() and record_path.exists():
            candidate = torch.load(artifact_path, map_location="cpu", weights_only=True)
            candidate_record = json.loads(record_path.read_text(encoding="utf-8"))
            if candidate_record["mapped_hash"] != blueprint_record["mapped_hash"]:
                raise RuntimeError(f"Resume fingerprint mismatch for {case['name']}")
            print(f"[{ordinal}/3] {case['name']} resumed", flush=True)
        else:
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
                model_sampling=guider.inner_model.model_sampling,
                destination_hw=geometry.destination_hw,
            )
            candidate, candidate_record = execute_bilinear(
                guider=guider, adapter=adapter, geometry=geometry, seed=case["seed"],
                mapped_cpu=mapped, options=options, device=device,
            )
            candidate_record["anchor_diagnostics"] = diagnostics
            candidate_record["rms_vs_control"] = difference(candidate, control)
            torch.save(candidate, artifact_path)
            atomic_json(record_path, candidate_record)
            print(f"[{ordinal}/3] {case['name']} persisted {candidate_record['wall_seconds']:.2f}s", flush=True)

            repeat, repeat_record = execute_bilinear(
                guider=guider, adapter=adapter, geometry=geometry, seed=case["seed"],
                mapped_cpu=mapped, options=options, device=device,
            )
            candidate_record["repeat"] = {
                "bit_exact": bool(torch.equal(candidate, repeat)),
                "difference": difference(candidate, repeat),
                "final_hash": repeat_record["final_hash"],
                "restricted_hashes_exact": [x["restricted_hash"] for x in candidate_record["regions"]]
                == [x["restricted_hash"] for x in repeat_record["regions"]],
            }
            atomic_json(record_path, candidate_record)
            del model, guider, destination, candidate, repeat
            comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
        results.append({
            "case": case, "geometry": {
                "destination_hw": list(geometry.destination_hw),
                "blueprint_hw": list(geometry.blueprint_hw),
                "regions": len(geometry.regions()),
            },
            "blueprint": blueprint_record,
            "control": control_record,
            "bilinear": candidate_record,
            "semantic_review": SEMANTIC_REVIEW[case["name"]],
        })

    decoded_hashes = decode_results()
    for result in results:
        result["bilinear"]["decoded_rgb_hash"] = decoded_hashes[result["case"]["name"]]
    build_sheet("PHASE31_COMPARISON.jpg")
    build_sheet("DETAIL_REVIEW.jpg", detail=True)
    atomic_json(OUTPUT / "report.json", {
        "phase": 31,
        "only_variable": "32x32 to 64x64 anchor lift: nearest versus bilinear align_corners=False",
        "control_exact": all(item["control"]["phase28_regression"]["bit_exact"] for item in results),
        "decision": "B — LIFT CHANGES OUTPUT BUT DOES NOT SOLVE SOFTNESS",
        "semantic_summary": "Both arms are S3 in all cases; bilinear provides no credible detail improvement and is slightly smoother.",
        "results": results,
    })
    print("Phase 31 execution complete.")


if __name__ == "__main__":
    main()
