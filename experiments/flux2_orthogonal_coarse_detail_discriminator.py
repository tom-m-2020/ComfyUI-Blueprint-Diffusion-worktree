"""Phase 35 fixed model-mediated orthogonal coarse/detail discriminator."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

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
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.regions import OverlapAssembler
from blueprint_diffusion.terminal_resampling import (
    TerminalResamplingGeometry,
    lift_region,
    region_noise,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_orthogonal_coarse_detail_discriminator_results"
SOURCE = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
CASE = phase29.CASES[0]
EXPECTED_MAPPED_HASH = "8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05"
EXPECTED_CONTROL_HASH = "1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3"


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_tensor(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {"rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def stats(value: torch.Tensor) -> dict:
    x = value.detach().float()
    return {
        "shape": list(x.shape), "finite": bool(torch.isfinite(x).all()),
        "mean": float(x.mean()), "rms": float(x.square().mean().sqrt()),
        "norm": float(torch.linalg.vector_norm(x)), "min": float(x.min()), "max": float(x.max()),
    }


def haar2x2(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return (
        (a + b + c + d) / 2,
        (a + b - c - d) / 2,
        (a - b + c - d) / 2,
        (a - b - c + d) / 2,
    )


def inverse_haar2x2(coefficients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    coarse, f1, f2, f3 = coefficients
    out = torch.empty(
        *coarse.shape[:-2], coarse.shape[-2] * 2, coarse.shape[-1] * 2,
        dtype=coarse.dtype, device=coarse.device,
    )
    out[..., 0::2, 0::2] = (coarse + f1 + f2 + f3) / 2
    out[..., 0::2, 1::2] = (coarse + f1 - f2 - f3) / 2
    out[..., 1::2, 0::2] = (coarse - f1 + f2 - f3) / 2
    out[..., 1::2, 1::2] = (coarse - f1 - f2 + f3) / 2
    return out


def validate_sources() -> tuple[torch.Tensor, dict, dict]:
    blueprint = torch.load(SOURCE / "blueprint.pt", map_location="cpu", weights_only=True)
    blueprint_record = json.loads((SOURCE / "blueprint.json").read_text(encoding="utf-8"))
    control_record = json.loads((SOURCE / "sigma_0.25.json").read_text(encoding="utf-8"))
    control = torch.load(SOURCE / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    mapped = blueprint["mapped"].float()
    if tensor_hash(mapped) != EXPECTED_MAPPED_HASH or blueprint_record["mapped_hash"] != EXPECTED_MAPPED_HASH:
        raise RuntimeError("Phase-29 mapped Blueprint fingerprint mismatch.")
    if tensor_hash(control) != EXPECTED_CONTROL_HASH or control_record["final_hash"] != EXPECTED_CONTROL_HASH:
        raise RuntimeError("Phase-29 sigma-0.25 control fingerprint mismatch.")
    if not torch.isfinite(mapped).all() or not control_record["phase28_regression"]["bit_exact"]:
        raise RuntimeError("Qualified source artifact failed integrity checks.")
    return mapped, blueprint_record, control_record


def execute_pass(name: str, *, guider, adapter, geometry, mapped_cpu, options, device) -> dict:
    run_dir = OUTPUT / name
    run_dir.mkdir(parents=True, exist_ok=True)
    regions = geometry.regions()
    mapped = mapped_cpu.to(device=device)
    sigma = torch.tensor(0.25, device=device, dtype=mapped.dtype)
    barrier = []
    cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for ordinal, region in enumerate(regions, 1):
            tensor_path = run_dir / f"region_{region.index:03d}.pt"
            json_path = run_dir / f"region_{region.index:03d}.json"
            if tensor_path.exists() and json_path.exists():
                saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
                record = json.loads(json_path.read_text(encoding="utf-8"))
                if (
                    record["mapped_hash"] != EXPECTED_MAPPED_HASH
                    or record["index"] != region.index
                    or any(tensor_hash(saved[key]) != record[f"{key}_hash"] for key in ("model_c", "forced_c", "f1", "f2", "f3"))
                ):
                    raise RuntimeError(f"Invalid resume artifact for region {region.index}.")
                print(f"[{name}] {ordinal}/{len(regions)} resumed", flush=True)
                continue
            crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            anchor = lift_region(crop)
            noise = region_noise(CASE["seed"], region, device=device, dtype=anchor.dtype)
            working = guider.inner_model.model_sampling.noise_scaling(sigma, noise, anchor, False)
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            prediction = adapter.predict_native(
                guider=guider, value=working, sigma=sigma,
                expected_hw=geometry.working_hw, model_options=options, seed=CASE["seed"],
            )
            end.record(); torch.cuda.synchronize(device)
            cuda_ms += float(begin.elapsed_time(end))
            model_bands = haar2x2(prediction)
            reconstructed = inverse_haar2x2(model_bands)
            reconstruction = difference(reconstructed, prediction)
            forced_c = 2 * crop
            retained = (model_bands[1], model_bands[2], model_bands[3])
            retained_hashes_before = [tensor_hash(value) for value in retained]
            replaced = (forced_c, *retained)
            retained_hashes_after = [tensor_hash(value) for value in replaced[1:]]
            saved = {
                "model_c": model_bands[0].detach().float().cpu(),
                "forced_c": forced_c.detach().float().cpu(),
                "f1": retained[0].detach().float().cpu(),
                "f2": retained[1].detach().float().cpu(),
                "f3": retained[2].detach().float().cpu(),
            }
            record = {
                "index": region.index, "region": [region.y, region.x, region.height, region.width],
                "mapped_hash": EXPECTED_MAPPED_HASH, "blueprint_crop_hash": tensor_hash(crop),
                "noise_hash": tensor_hash(noise), "working_hash": tensor_hash(working),
                "prediction_hash": tensor_hash(prediction),
                "prediction_shape": list(prediction.shape), "band_shapes": [list(value.shape) for value in replaced],
                "haar_round_trip": reconstruction,
                "detail_hashes_unchanged": retained_hashes_before == retained_hashes_after,
                "detail_max_abs_change": 0.0,
                "model_c_vs_forced_c": difference(model_bands[0], forced_c),
            }
            for key, value in saved.items():
                record[f"{key}_hash"] = tensor_hash(value)
                record[f"{key}_stats"] = stats(value)
            atomic_tensor(tensor_path, saved)
            atomic_json(json_path, record)
            del crop, anchor, noise, working, prediction, model_bands, reconstructed, forced_c, retained, replaced, saved
            torch.cuda.synchronize(device)
            barrier.append(int(torch.cuda.memory_allocated(device)))
            print(f"[{name}] {ordinal}/{len(regions)} persisted", flush=True)
    return {
        "new_local_cuda_ms": cuda_ms,
        "wall_seconds_this_invocation": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barrier,
    }


def assemble(name: str, geometry: TerminalResamplingGeometry, mapped: torch.Tensor) -> tuple[torch.Tensor, dict]:
    regions = geometry.regions()
    weighted = [torch.zeros_like(mapped) for _ in range(4)]
    coverage = [torch.zeros((1, 1, *geometry.destination_hw), dtype=torch.float32) for _ in range(4)]
    weights = OverlapAssembler()
    region_records = []
    region_bands = [[], [], [], []]
    for region in regions:
        saved = torch.load(OUTPUT / name / f"region_{region.index:03d}.pt", map_location="cpu", weights_only=True)
        values = (saved["forced_c"], saved["f1"], saved["f2"], saved["f3"])
        weight = weights.weight(region, regions, torch.device("cpu"))[None, None]
        for index, value in enumerate(values):
            region_bands[index].append(value)
            weighted[index][:, :, region.y:region.y2, region.x:region.x2] += value * weight
            coverage[index][:, :, region.y:region.y2, region.x:region.x2] += weight
        region_records.append(json.loads((OUTPUT / name / f"region_{region.index:03d}.json").read_text(encoding="utf-8")))
    if any(float(value.min()) <= 0 or not torch.isfinite(value).all() for value in coverage):
        raise RuntimeError("Incomplete or nonfinite band coverage.")
    bands = tuple(value / band_coverage for value, band_coverage in zip(weighted, coverage))
    reconstructed = inverse_haar2x2(bands)
    down = torch.nn.functional.avg_pool2d(reconstructed, 2, 2)
    report = {
        "assembled_band_stats": {key: stats(value) for key, value in zip(("C", "F1", "F2", "F3"), bands)},
        "assembled_band_hashes": {key: tensor_hash(value) for key, value in zip(("C", "F1", "F2", "F3"), bands)},
        "band_overlap_rms": {key: phase8d.overlap_metrics(values, regions)["aggregate_rms"] for key, values in zip(("C", "F1", "F2", "F3"), region_bands)},
        "coverage": [{"shape": list(value.shape), "min": float(value.min()), "max": float(value.max()), "finite": bool(torch.isfinite(value).all())} for value in coverage],
        "forced_coarse_vs_2_blueprint": difference(bands[0], 2 * mapped),
        "avgpool_reconstructed_vs_blueprint": difference(down, mapped),
        "reconstructed": stats(reconstructed),
        "reconstructed_hash": tensor_hash(reconstructed),
        "reconstructed_vs_nearest_blueprint": difference(reconstructed, torch.nn.functional.interpolate(mapped, scale_factor=2, mode="nearest")),
        "storage_bytes": {
            "one_region_four_bands_fp32": sum(value.numel() * value.element_size() for value in region_bands[0][:1]) * 4,
            "assembled_four_bands_fp32": sum(value.numel() * value.element_size() for value in bands),
            "reconstructed_fp32": reconstructed.numel() * reconstructed.element_size(),
        },
        "regions": region_records,
    }
    return reconstructed, {"bands": bands, "report": report}


def save_energy(value: torch.Tensor, path: Path) -> None:
    image = value.float().square().mean(dim=1).sqrt()[0]
    lo, hi = torch.quantile(image.flatten(), torch.tensor([0.01, 0.99]))
    image = ((image - lo) / max(float(hi - lo), 1e-12)).clamp(0, 1)
    Image.fromarray(image.mul(255).round().byte().numpy(), mode="L").save(path)


def decode(latent: torch.Tensor, path: Path) -> str:
    if path.exists():
        with Image.open(path) as image:
            return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    pixels = vae.decode_tiled(latent, tile_x=64, tile_y=64, overlap=16).cpu()
    phase2.save_pixels(pixels, path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def comparison_sheet() -> None:
    sources = (
        (SOURCE / "sigma_0.25.png", "Qualified one-pass 2048x2048"),
        (OUTPUT / "RECONSTRUCTED_FINAL.png", "Phase 35 Haar output 4096x4096"),
    )
    sheet = Image.new("RGB", (2048, 1060), "white")
    for index, (path, label) in enumerate(sources):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (1024, 1060), "white")
        panel.paste(image, ((1024 - image.width) // 2, 36))
        from PIL import ImageDraw
        ImageDraw.Draw(panel).text((8, 8), label, fill="black")
        sheet.paste(panel, (1024 * index, 0))
    sheet.save(OUTPUT / "PHASE35_COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    historical = None
    historical_path = OUTPUT / "report.pre_review.json"
    if historical_path.exists():
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
    mapped, blueprint_record, control_record = validate_sources()
    geometry = TerminalResamplingGeometry.for_destination(CASE["destination_hw"])
    if len(geometry.regions()) != 25:
        raise RuntimeError("Phase-35 requires exactly 25 square-scene regions.")

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(CASE["prompt"]))
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device)
    options = phase20.phase8i_options({})
    adapter = Flux2TerminalResamplingAdapter()
    adapter.validate_prepared(
        guider=guider, model_options=options,
        destination=torch.zeros((1, 128, *geometry.destination_hw), device=device),
        model_sampling=guider.inner_model.model_sampling,
        destination_hw=geometry.destination_hw,
    )
    execution = execute_pass("primary", guider=guider, adapter=adapter, geometry=geometry, mapped_cpu=mapped, options=options, device=device)
    if execution["new_local_cuda_ms"] == 0.0 and historical is not None:
        execution = historical["execution"]
    reconstructed, assembled = assemble("primary", geometry, mapped)
    atomic_tensor(OUTPUT / "reconstructed_latent.pt", reconstructed)
    atomic_tensor(OUTPUT / "assembled_bands.pt", {key: value for key, value in zip(("C", "F1", "F2", "F3"), assembled["bands"])})
    for key, value in zip(("C", "F1", "F2", "F3"), assembled["bands"]):
        save_energy(value, OUTPUT / f"{key}_ENERGY.png")
    decoded_hash = decode(reconstructed, OUTPUT / "RECONSTRUCTED_FINAL.png")
    comparison_sheet()

    repeat_execution = execute_pass("repeat", guider=guider, adapter=adapter, geometry=geometry, mapped_cpu=mapped, options=options, device=device)
    if repeat_execution["new_local_cuda_ms"] == 0.0 and historical is not None:
        repeat_execution = historical["repeat_execution"]
    repeated, repeated_assembled = assemble("repeat", geometry, mapped)
    repeat = {
        "bit_exact": bool(torch.equal(reconstructed, repeated)),
        "difference": difference(reconstructed, repeated),
        "hash": tensor_hash(repeated),
        "region_prediction_hashes_exact": [x["prediction_hash"] for x in assembled["report"]["regions"]] == [x["prediction_hash"] for x in repeated_assembled["report"]["regions"]],
    }
    report = {
        "phase": 35,
        "case": CASE,
        "fixed_contract": {
            "sigma": 0.25, "blueprint_shape": list(mapped.shape),
            "working_shape": [1, 128, 64, 64], "band_shape": [1, 128, 32, 32],
            "assembled_band_shape": list(mapped.shape), "reconstructed_shape": list(reconstructed.shape),
            "regions": 25, "overlap": "existing normalized Phase-29 weights",
            "coarse_rule": "C = 2 * mapped Blueprint crop", "detail_rule": "unchanged Haar detail coefficients from native model prediction",
        },
        "source_integrity": {"blueprint": blueprint_record, "control": control_record, "mapped_hash": tensor_hash(mapped)},
        "execution": execution, "repeat_execution": repeat_execution,
        "assembly": assembled["report"], "decoded_rgb_hash": decoded_hash,
        "repeat": repeat,
        "model_calls": 25, "repeat_model_calls": 25, "destination_sized_model_calls": 0,
        "production_changed": False, "comfy_core_changed": False,
        "semantic_review": {
            "grade": "S3 composition with artifacted/inconclusive detail",
            "composition": "Exactly one red car, one central tree, and one white house remain in the Blueprint left/center/right arrangement under one continuous horizon.",
            "detail": "The retained bands add dense regular crosshatch/ghost texture and broadened edge echoes. Wheels, car glazing, foliage, roof/window structure, and grass do not gain credible higher-resolution structure.",
            "classification": "Coarse ownership works, but the untouched local Haar detail is not a useful compatible detail representation in this fixed test.",
        },
        "decision": "B — PARTIAL/AMBIGUOUS: COARSE OWNERSHIP WORKS, BUT RETAINED DETAIL IS ARTIFACTED AND NOT CREDIBLY USEFUL",
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "latent_hash": tensor_hash(reconstructed), "decoded_hash": decoded_hash, "repeat": repeat}, indent=2), flush=True)


if __name__ == "__main__":
    main()
