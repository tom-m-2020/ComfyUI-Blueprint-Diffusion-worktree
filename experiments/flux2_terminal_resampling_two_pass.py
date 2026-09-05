"""Phase 33 fixed two-pass terminal-resampling discriminator."""
from __future__ import annotations

import hashlib
import json
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

OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_two_pass_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results"
SEMANTIC_REVIEW = {
    "SQUARE_MULTI_OBJECT": {
        "control_grade": "S3", "second_pass_grade": "S3",
        "finding": "One car/tree/house scene remains, but wheels, windows, foliage, roof, and grass structure are not credibly clearer; increased fine texture is largely alias-like.",
    },
    "PORTRAIT_ASTRONAUT": {
        "control_grade": "S3", "second_pass_grade": "S3",
        "finding": "One astronaut remains, but helmet, limbs, suit panels, shadow, and ground structure do not materially improve; stronger texture does not resolve the soft anatomy.",
    },
    "LANDSCAPE_BRIDGE": {
        "control_grade": "S3", "second_pass_grade": "S3",
        "finding": "One bridge scene remains, but cables, deck, supports, secondary tower, and water structure are not materially clearer; line texture intensifies without structural recovery.",
    },
}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    with temporary.open("ab") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {"rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def validate_control(case: dict, blueprint_record: dict) -> tuple[torch.Tensor, dict]:
    source = PHASE29 / case["name"]
    record = json.loads((source / "sigma_0.25.json").read_text(encoding="utf-8"))
    latent = torch.load(source / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    if record["sigma"] != 0.25 or record["mapped_hash"] != blueprint_record["mapped_hash"]:
        raise RuntimeError(f"Phase-29 control fingerprint mismatch: {case['name']}")
    if not record["phase28_regression"]["bit_exact"] or tensor_hash(latent) != record["final_hash"]:
        raise RuntimeError(f"Phase-29 control is not exact: {case['name']}")
    return latent.float(), record


def execute_regions(
    *, pass_dir: Path, guider, adapter, geometry, seed, h1_cpu, options, device,
) -> tuple[torch.Tensor, dict]:
    pass_dir.mkdir(parents=True, exist_ok=True)
    regions = geometry.regions()
    h1_hash_before = tensor_hash(h1_cpu)
    sigma = torch.tensor(0.25, device=device, dtype=torch.float32)
    region_records = []
    barrier = []
    cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for region in regions:
            tensor_path = pass_dir / f"region_{region.index:03d}.pt"
            record_path = pass_dir / f"region_{region.index:03d}.json"
            expected_noise = region_noise(seed, region, device="cpu", dtype=torch.float32)
            expected_noise_hash = tensor_hash(expected_noise)
            if tensor_path.exists() and record_path.exists():
                restricted = torch.load(tensor_path, map_location="cpu", weights_only=True)
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if (
                    record["index"] != region.index
                    or record["h1_hash"] != h1_hash_before
                    or record["noise_hash"] != expected_noise_hash
                    or tensor_hash(restricted) != record["restricted_hash"]
                ):
                    raise RuntimeError(f"Resume fingerprint mismatch region {region.index}")
                region_records.append(record)
                cuda_ms += record["cuda_ms"]
                barrier.append(record["barrier_allocated_bytes"])
                print(f"  resume region {region.index + 1}/{len(regions)}", flush=True)
                continue
            crop_cpu = h1_cpu[:, :, region.y:region.y2, region.x:region.x2]
            crop = crop_cpu.to(device=device)
            anchor = lift_region(crop)
            noise = expected_noise.to(device=device, dtype=anchor.dtype)
            working = guider.inner_model.model_sampling.noise_scaling(sigma, noise, anchor, False)
            expected = 0.75 * anchor + 0.25 * noise
            construction_error = float((working.float() - expected.float()).abs().max())
            working_hash = tensor_hash(working)
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            prediction = adapter.predict_native(
                guider=guider, value=working, sigma=sigma,
                expected_hw=geometry.working_hw, model_options=options, seed=seed,
            )
            end.record()
            torch.cuda.synchronize(device)
            elapsed = float(begin.elapsed_time(end))
            prediction_hash = tensor_hash(prediction)
            restricted = restrict_working_prediction(prediction).detach().float().cpu()
            del crop, anchor, noise, working, expected, prediction
            torch.cuda.synchronize(device)
            allocated = int(torch.cuda.memory_allocated(device))
            record = {
                "index": region.index,
                "h1_hash": h1_hash_before,
                "h1_crop_hash": tensor_hash(crop_cpu),
                "noise_hash": expected_noise_hash,
                "working_hash": working_hash,
                "prediction_hash": prediction_hash,
                "restricted_hash": tensor_hash(restricted),
                "construction_max_abs": construction_error,
                "cuda_ms": elapsed,
                "barrier_allocated_bytes": allocated,
            }
            atomic_tensor(tensor_path, restricted)
            atomic_json(record_path, record)
            region_records.append(record)
            cuda_ms += elapsed
            barrier.append(allocated)
            print(f"  persist region {region.index + 1}/{len(regions)}", flush=True)

    template = h1_cpu.to(device=device)
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw, template=template)
    restricted_cpu = []
    for region in regions:
        value = torch.load(pass_dir / f"region_{region.index:03d}.pt", map_location="cpu", weights_only=True)
        restricted_cpu.append(value)
        assembler.add(value.to(device=device), region)
    assembled, coverage = assembler.finish()
    h2 = assembled.detach().float().cpu()
    if tensor_hash(h1_cpu) != h1_hash_before:
        raise RuntimeError("H1 mutated during second pass")
    summary = {
        "h1_hash": h1_hash_before,
        "h1_hash_after": tensor_hash(h1_cpu),
        "h2_hash": tensor_hash(h2),
        "regions": region_records,
        "local_calls": len(regions),
        "destination_model_calls": 0,
        "overlap_rms": phase8d.overlap_metrics(restricted_cpu, regions)["aggregate_rms"],
        "gradient_rms": phase22.grad_rms(h2),
        "coverage": [float(coverage.min()), float(coverage.max())],
        "finite": bool(torch.isfinite(h2).all()),
        "cuda_ms": cuda_ms,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barrier,
    }
    atomic_tensor(pass_dir / "assembled.pt", h2)
    atomic_json(pass_dir / "summary.json", summary)
    return h2, summary


def decode_all() -> dict:
    missing = [OUTPUT / case["name"] / "H2.png" for case in phase29.CASES if not (OUTPUT / case["name"] / "H2.png").exists()]
    vae = None
    hashes = {}
    for case in phase29.CASES:
        case_dir = OUTPUT / case["name"]
        path = case_dir / "H2.png"
        if not path.exists():
            if vae is None:
                vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
            latent = torch.load(case_dir / "primary" / "assembled.pt", map_location="cpu", weights_only=True)
            phase2.save_pixels(vae.decode(latent).cpu(), path)
        with Image.open(path) as image:
            hashes[case["name"]] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    return hashes


def make_sheet(name: str, detail: bool = False) -> None:
    sheet = Image.new("RGB", (1240, 515 * len(phase29.CASES)), "white")
    for row, case in enumerate(phase29.CASES):
        paths = (PHASE29 / case["name"] / "sigma_0.25.png", OUTPUT / case["name"] / "H2.png")
        for column, (path, arm) in enumerate(zip(paths, ("A one pass", "B two pass"), strict=True)):
            image = Image.open(path).convert("RGB")
            if detail:
                left, top = image.width // 5, image.height // 5
                image = image.crop((left, top, image.width - left, image.height - top))
            image.thumbnail((600, 470), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (620, 515), "white")
            panel.paste(image, ((620 - image.width) // 2, 38))
            ImageDraw.Draw(panel).text((8, 8), f"{case['name']} | {arm}", fill="black")
            sheet.paste(panel, (column * 620, row * 515))
    sheet.save(OUTPUT / name, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for ordinal, case in enumerate(phase29.CASES, 1):
        print(f"[{ordinal}/3] {case['name']}", flush=True)
        case_dir = OUTPUT / case["name"]
        case_dir.mkdir(exist_ok=True)
        blueprint = torch.load(PHASE29 / case["name"] / "blueprint.pt", map_location="cpu", weights_only=True)
        blueprint_record = json.loads((PHASE29 / case["name"] / "blueprint.json").read_text(encoding="utf-8"))
        h1, control = validate_control(case, blueprint_record)
        mapped = blueprint["mapped"].float()
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
            model_sampling=guider.inner_model.model_sampling,
            destination_hw=geometry.destination_hw,
        )
        print(" primary", flush=True)
        h2, primary = execute_regions(
            pass_dir=case_dir / "primary", guider=guider, adapter=adapter,
            geometry=geometry, seed=case["seed"], h1_cpu=h1,
            options=options, device=device,
        )
        print(" repeat", flush=True)
        h2_repeat, repeat = execute_regions(
            pass_dir=case_dir / "repeat", guider=guider, adapter=adapter,
            geometry=geometry, seed=case["seed"], h1_cpu=h1,
            options=options, device=device,
        )
        repeat_check = {
            "bit_exact": bool(torch.equal(h2, h2_repeat)),
            "difference": difference(h2, h2_repeat),
            "region_hashes_exact": [x["restricted_hash"] for x in primary["regions"]]
            == [x["restricted_hash"] for x in repeat["regions"]],
        }
        results.append({
            "case": case,
            "blueprint": blueprint_record,
            "control": control,
            "second_pass": primary,
            "repeat": repeat_check,
            "metrics": {
                "h2_vs_h1": difference(h2, h1),
                "h1_vs_blueprint": difference(h1, mapped),
                "h2_vs_blueprint": difference(h2, mapped),
                "h1_low_frequency_vs_blueprint": phase22.low_frequency_rms(h1, mapped),
                "h2_low_frequency_vs_blueprint": phase22.low_frequency_rms(h2, mapped),
                "h1_gradient_rms": phase22.grad_rms(h1),
                "h2_gradient_rms": phase22.grad_rms(h2),
            },
            "semantic_review": SEMANTIC_REVIEW[case["name"]],
        })
        del model, guider, destination, h2, h2_repeat
        comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()

    decoded = decode_all()
    for result in results:
        result["second_pass"]["decoded_rgb_hash"] = decoded[result["case"]["name"]]
    make_sheet("PHASE33_COMPARISON.jpg")
    make_sheet("DETAIL_REVIEW.jpg", detail=True)
    atomic_json(OUTPUT / "report.json", {
        "phase": 33,
        "only_variable": "one versus exactly two sigma-0.25 model-mediated terminal passes",
        "control_exact": all(item["control"]["phase28_regression"]["bit_exact"] for item in results),
        "decision": "B — SECOND PASS RETAINS S3 BUT DOES NOT IMPROVE DETAIL",
        "semantic_summary": "The second pass remains S3 in all cases but adds mostly texture/alias energy rather than credible local structural detail.",
        "results": results,
    })
    print("Phase 33 execution complete.")


def finalize_existing_report() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["decision"] = "B — SECOND PASS RETAINS S3 BUT DOES NOT IMPROVE DETAIL"
    report["semantic_summary"] = "The second pass remains S3 in all cases but adds mostly texture/alias energy rather than credible local structural detail."
    for result in report["results"]:
        result["semantic_review"] = SEMANTIC_REVIEW[result["case"]["name"]]
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing_report()
    else:
        main()
