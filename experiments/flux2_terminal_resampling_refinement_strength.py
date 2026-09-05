"""Phase 29 terminal-resampling sigma discriminator (research only)."""
from __future__ import annotations

import hashlib
import importlib.util
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
spec = importlib.util.spec_from_file_location(
    "blueprint_diffusion", PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
module = importlib.util.module_from_spec(spec)
sys.modules["blueprint_diffusion"] = module
spec.loader.exec_module(module)

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    BlueprintRunState,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    initialize_blueprint,
    lift_region,
    map_blueprint_to_destination,
    region_noise,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results"
PHASE28 = ROOT / "experiments" / "terminal_resampling_geometry_qualification_results"
SIGMAS = (0.10, 0.15, 0.25, 0.35, 0.50)
EXECUTION_ORDER = (0.25, 0.10, 0.15, 0.35, 0.50)
CASES = (
    {
        "name": "SQUARE_MULTI_OBJECT", "destination_hw": (128, 128), "seed": 20260921,
        "prompt": "A cinematic photograph of exactly one red vintage car on the left, one tall green tree in the center, and one small white house on the right, all on one grassy field beneath one continuous horizon, coherent perspective, no duplicate objects.",
    },
    {
        "name": "PORTRAIT_ASTRONAUT", "destination_hw": (256, 128), "seed": 20260922,
        "prompt": "A full-body astronaut standing alone in the center of a tall wide desert scene, facing the camera, both arms and both legs visible, one continuous body, distant low mountains across one horizon, exactly one astronaut, no duplicate people or body parts.",
    },
    {
        "name": "LANDSCAPE_BRIDGE", "destination_hw": (128, 192), "seed": 20260923,
        "prompt": "A wide cinematic photograph of exactly one long red suspension bridge crossing calm water from left to right, one yellow passenger train on the bridge, controlled towers and continuous cables, one coherent horizon, no duplicate bridges or trains.",
    },
)


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
    }


def run_blueprint(*, guider, adapter, geometry, seed, options, device, case_dir):
    artifact = case_dir / "blueprint.pt"
    record_path = case_dir / "blueprint.json"
    if artifact.exists() and record_path.exists():
        saved = torch.load(artifact, map_location="cpu", weights_only=True)
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record["geometry"] == list(geometry.blueprint_hw) and record["seed"] == seed:
            return saved, record
    state_value = initialize_blueprint(seed, geometry=geometry, device=device)
    initial_hash = tensor_hash(state_value)
    x0_hashes = []
    accepted_hashes = []
    terminal = None
    sigmas = torch.tensor(QUALIFIED_SIGMAS, device=device)
    with torch.inference_mode():
        for ordinal in range(4):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            terminal = adapter.predict_native(
                guider=guider, value=state_value, sigma=sigma,
                expected_hw=geometry.blueprint_hw, model_options=options, seed=seed,
            )
            x0_hashes.append(tensor_hash(terminal))
            state_value = state_value + (sigma_next - sigma) * (state_value - terminal) / sigma
            accepted_hashes.append(tensor_hash(state_value))
    terminal_cpu = terminal.detach().float().cpu()
    mapped = map_blueprint_to_destination(terminal_cpu, geometry)
    saved = {"terminal_x0": terminal_cpu, "mapped": mapped}
    torch.save(saved, artifact)
    record = {
        "seed": seed, "geometry": list(geometry.blueprint_hw),
        "initial_hash": initial_hash, "x0_hashes": x0_hashes,
        "accepted_hashes": accepted_hashes, "terminal_hash": tensor_hash(terminal_cpu),
        "mapped_hash": tensor_hash(mapped), "model_calls": 4,
    }
    atomic_json(record_path, record)
    return saved, record


def run_sigma(*, guider, adapter, geometry, seed, sigma_value, mapped_cpu, options, device, case_dir):
    name = f"sigma_{sigma_value:.2f}"
    record_path = case_dir / f"{name}.json"
    latent_path = case_dir / f"{name}.pt"
    if record_path.exists() and latent_path.exists():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("sigma") == sigma_value and record.get("mapped_hash") == tensor_hash(mapped_cpu):
            return torch.load(latent_path, map_location="cpu", weights_only=True), record
    regions = geometry.regions()
    mapped = mapped_cpu.to(device=device)
    assembler = StreamingOverlapAssembler(
        regions=regions, target_hw=geometry.destination_hw, template=mapped,
    )
    sigma = torch.tensor(sigma_value, device=device, dtype=mapped.dtype)
    restricted_cpu = []
    region_records = []
    barrier = []
    cuda_ms = 0.0
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for region in regions:
            crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            anchor = lift_region(crop)
            noise = region_noise(seed, region, device=device, dtype=anchor.dtype)
            working = guider.inner_model.model_sampling.noise_scaling(
                sigma, noise, anchor, False
            )
            expected = (1.0 - sigma_value) * anchor + sigma_value * noise
            construction_error = float((working.float() - expected.float()).abs().max())
            working_hash = tensor_hash(working)
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            prediction_w = adapter.predict_native(
                guider=guider, value=working, sigma=sigma,
                expected_hw=geometry.working_hw, model_options=options, seed=seed,
            )
            end.record(); torch.cuda.synchronize(); cuda_ms += float(begin.elapsed_time(end))
            restricted = restrict_working_prediction(prediction_w)
            assembler.add(restricted, region)
            restricted_cpu.append(restricted.detach().float().cpu())
            region_records.append({
                "index": region.index, "seed": seed + 22_000_003 + 1009 * region.index,
                "noise_hash": tensor_hash(noise), "working_hash": working_hash,
                "restricted_hash": tensor_hash(restricted),
                "construction_max_abs": construction_error,
            })
            del crop, anchor, noise, working, expected, prediction_w, restricted
            torch.cuda.synchronize(device)
            barrier.append(int(torch.cuda.memory_allocated(device)))
    assembled, coverage = assembler.finish()
    assembled_cpu = assembled.detach().float().cpu()
    overlap = phase8d.overlap_metrics(restricted_cpu, regions)["aggregate_rms"]
    record = {
        "sigma": sigma_value, "mapped_hash": tensor_hash(mapped_cpu),
        "final_hash": tensor_hash(assembled_cpu), "regions": region_records,
        "blueprint_calls_reused": 4, "local_calls": len(regions),
        "destination_model_calls": 0, "overlap_rms": overlap,
        "gradient_rms": phase22.grad_rms(assembled_cpu),
        "rms_vs_blueprint": difference(assembled_cpu, mapped_cpu),
        "low_frequency_rms_vs_blueprint": phase22.low_frequency_rms(assembled_cpu, mapped_cpu),
        "coverage": [float(coverage.min()), float(coverage.max())],
        "finite": bool(torch.isfinite(assembled_cpu).all()),
        "local_cuda_ms": cuda_ms, "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barrier,
    }
    torch.save(assembled_cpu, latent_path)
    atomic_json(record_path, record)
    return assembled_cpu, record


def decode_all():
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    for case in CASES:
        case_dir = OUTPUT / case["name"]
        decoded[case["name"]] = {}
        for sigma in SIGMAS:
            stem = f"sigma_{sigma:.2f}"
            path = case_dir / f"{stem}.png"
            latent = torch.load(case_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
            phase2.save_pixels(vae.decode(latent).cpu(), path)
            with Image.open(path) as image:
                decoded[case["name"]][stem] = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    return decoded


def build_sheet():
    rows = []
    for case in CASES:
        sources = [("tiled S0", PHASE28 / case["name"] / "ordinary_tiled.png")]
        sources += [(f"sigma {sigma:.2f}", OUTPUT / case["name"] / f"sigma_{sigma:.2f}.png") for sigma in SIGMAS]
        panels = []
        for label, source in sources:
            image = Image.open(source).convert("RGB")
            image.thumbnail((500, 500), Image.Resampling.LANCZOS)
            panel = Image.new("RGB", (500, 535), "white")
            panel.paste(image, ((500 - image.width) // 2, 35))
            ImageDraw.Draw(panel).text((8, 8), f"{case['name']} | {label}", fill="black")
            panels.append(panel)
        row = Image.new("RGB", (3000, 535), "white")
        for index, panel in enumerate(panels):
            row.paste(panel, (500 * index, 0))
        rows.append(row)
    sheet = Image.new("RGB", (3000, 535 * len(rows)), "white")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, 535 * index))
    sheet.save(OUTPUT / "PHASE29_SIGMA_COMPARISON.jpg", quality=93)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_results = []
    for case_index, case in enumerate(CASES, 1):
        print(f"[{case_index}/3] {case['name']} setup", flush=True)
        case_dir = OUTPUT / case["name"]
        case_dir.mkdir(exist_ok=True)
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
        blueprint, blueprint_record = run_blueprint(
            guider=guider, adapter=adapter, geometry=geometry, seed=case["seed"],
            options=options, device=device, case_dir=case_dir,
        )
        sigma_results = {}
        for sigma_index, sigma in enumerate(EXECUTION_ORDER, 1):
            print(f"[{case_index}/3] {case['name']} sigma {sigma:.2f} ({sigma_index}/5)", flush=True)
            latent, record = run_sigma(
                guider=guider, adapter=adapter, geometry=geometry, seed=case["seed"],
                sigma_value=sigma, mapped_cpu=blueprint["mapped"], options=options,
                device=device, case_dir=case_dir,
            )
            if sigma == 0.25:
                reference = torch.load(PHASE28 / case["name"] / "selected_0.pt", map_location="cpu", weights_only=True)
                record["phase28_regression"] = {
                    "bit_exact": bool(torch.equal(latent, reference)),
                    **difference(latent, reference),
                }
                atomic_json(case_dir / "sigma_0.25.json", record)
                if not record["phase28_regression"]["bit_exact"]:
                    raise RuntimeError(f"{case['name']} sigma 0.25 failed Phase-28 regression")
            sigma_results[f"{sigma:.2f}"] = record
        all_results.append({
            "case": case, "geometry": {
                "blueprint_hw": geometry.blueprint_hw, "destination_hw": geometry.destination_hw,
                "regions": len(geometry.regions()), "working_hw": geometry.working_hw,
            }, "blueprint": blueprint_record, "sigmas": sigma_results,
        })
        model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    decoded = decode_all()
    build_sheet()
    report = {
        "phase": 29, "sigma_set": SIGMAS, "only_variable": "terminal local resampling sigma",
        "results": all_results, "decoded_hashes": decoded,
        "all_025_exact": all(item["sigmas"]["0.25"]["phase28_regression"]["bit_exact"] for item in all_results),
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / 'report.json'), "all_025_exact": report["all_025_exact"]}, indent=2))


if __name__ == "__main__":
    main()
