"""Literal interpolation-only alternating G/W Blueprint prototype."""
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
import flux2_terminal_resampling_four_interval_local_trajectory as phase38
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_staged_refinement_geometry import Geometry, bounded_transfer
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.configurable_resampling import StreamingOverlapMetric
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    initialize_blueprint,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_literal_alternating_resolution_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"


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


def euler_step(state: torch.Tensor, prediction: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
    if not 0.0 <= sigma_next < sigma:
        raise ValueError("Literal alternating intervals require sigma > sigma_next >= 0.")
    if state.shape != prediction.shape:
        raise ValueError("State and prediction shapes differ.")
    return state + (sigma_next - sigma) * (state - prediction) / sigma


def interval_owner(ordinal: int) -> str:
    if ordinal < 0:
        raise ValueError("Interval ordinal must be nonnegative.")
    return "G" if ordinal % 2 == 0 else "W"


def downscale_h_to_g(h: torch.Tensor, g_hw: tuple[int, int]) -> torch.Tensor:
    if h.ndim != 4:
        raise ValueError("H must be a four-dimensional latent tensor.")
    return F.interpolate(h, size=g_hw, mode="bilinear", align_corners=False)


def transfer_geometry(geometry: TerminalResamplingGeometry) -> Geometry:
    return Geometry(
        g_hw=geometry.blueprint_hw,
        h_hw=geometry.destination_hw,
        footprint_hw=geometry.region_hw,
        stride_hw=geometry.stride_hw,
        w_hw=geometry.working_hw,
    )


def run_once(*, name, guider, adapter, geometry, seed, options, device):
    sigmas = tuple(float(value) for value in QUALIFIED_SIGMAS)
    regions = geometry.regions()
    mapping = transfer_geometry(geometry)
    g = initialize_blueprint(seed, geometry=geometry, device=device)
    initial_g_hash = tensor_hash(g)
    interval_records = []
    barrier_allocations = []
    model_call_shapes = []
    final_h = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for ordinal, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            owner = interval_owner(ordinal)
            sigma_tensor = torch.tensor(sigma, device=device, dtype=g.dtype)
            if owner == "G":
                before_hash = tensor_hash(g)
                prediction = adapter.predict_native(
                    guider=guider,
                    value=g,
                    sigma=sigma_tensor,
                    expected_hw=geometry.blueprint_hw,
                    model_options=options,
                    seed=seed,
                )
                model_call_shapes.append(list(g.shape))
                accepted = euler_step(g, prediction, sigma, sigma_next)
                interval_records.append({
                    "ordinal": ordinal,
                    "owner": "G",
                    "sigma_in": sigma,
                    "sigma_out": sigma_next,
                    "state_before_hash": before_hash,
                    "raw_prediction_hash": tensor_hash(prediction),
                    "accepted_state_hash": tensor_hash(accepted),
                    "accepted_shape": list(accepted.shape),
                    "transfer_operation": "none",
                    "model_calls": 1,
                })
                g = accepted
                del prediction, accepted
            else:
                accepted_g_hash = tensor_hash(g)
                immutable_g = g.clone()
                assembler = StreamingOverlapAssembler(
                    regions=regions,
                    target_hw=geometry.destination_hw,
                    template=torch.zeros((1, g.shape[1], *geometry.destination_hw), device=device, dtype=g.dtype),
                )
                overlap = StreamingOverlapMetric()
                region_records = []
                for region in regions:
                    rect = (region.y, region.x, region.height, region.width)
                    w, source_hw = bounded_transfer(g, mapping, rect)
                    w_before_hash = tensor_hash(w)
                    prediction = adapter.predict_native(
                        guider=guider,
                        value=w,
                        sigma=sigma_tensor,
                        expected_hw=geometry.working_hw,
                        model_options=options,
                        seed=seed,
                    )
                    model_call_shapes.append(list(w.shape))
                    accepted_w = euler_step(w, prediction, sigma, sigma_next)
                    accepted_w_hash = tensor_hash(accepted_w)
                    footprint = restrict_working_prediction(accepted_w)
                    overlap.add(footprint, region)
                    assembler.add(footprint, region)
                    region_records.append({
                        "index": region.index,
                        "rect_yxhw": list(rect),
                        "bounded_source_hw": list(source_hw),
                        "shared_accepted_g_hash": accepted_g_hash,
                        "w_after_g_resize_hash": w_before_hash,
                        "raw_prediction_hash": tensor_hash(prediction),
                        "accepted_w_before_restriction_hash": accepted_w_hash,
                        "footprint_after_restriction_hash": tensor_hash(footprint),
                    })
                    del w, prediction, accepted_w, footprint
                    torch.cuda.synchronize(device)
                    barrier_allocations.append(int(torch.cuda.memory_allocated(device)))
                if not torch.equal(g, immutable_g):
                    raise RuntimeError(f"Accepted G mutated while proposing W interval {ordinal}.")
                h, coverage = assembler.finish()
                h_hash = tensor_hash(h)
                next_g = downscale_h_to_g(h, geometry.blueprint_hw)
                interval_records.append({
                    "ordinal": ordinal,
                    "owner": "W",
                    "sigma_in": sigma,
                    "sigma_out": sigma_next,
                    "accepted_g_before_map_hash": accepted_g_hash,
                    "all_w_share_accepted_g": all(
                        item["shared_accepted_g_hash"] == accepted_g_hash for item in region_records
                    ),
                    "regions": region_records,
                    "assembled_h_before_downscale_hash": h_hash,
                    "accepted_g_after_downscale_hash": tensor_hash(next_g),
                    "coverage": [float(coverage.min()), float(coverage.max())],
                    "overlap_rms": overlap.finish(),
                    "model_calls": len(regions),
                    "cross_resolution_operations": [
                        "bilinear G-to-H mapping, footprint crop, bilinear F-to-W resize",
                        "area-mean W-to-F restriction",
                        "normalized overlap assembly",
                        "bilinear H-to-G downscale and direct replacement",
                    ],
                })
                final_h = h
                g = next_g
                del immutable_g, h, next_g
            print(f"{name}: {owner} interval {ordinal + 1}/{len(sigmas) - 1}", flush=True)

    if final_h is None:
        raise RuntimeError("Literal alternating trajectory produced no H state.")
    result = final_h.detach().float().cpu()
    return result, {
        "run": name,
        "schedule": list(sigmas),
        "ownership": [interval_owner(index) for index in range(len(sigmas) - 1)],
        "initial_g_hash": initial_g_hash,
        "final_g_hash": tensor_hash(g),
        "final_h_hash": tensor_hash(result),
        "intervals": interval_records,
        "global_model_calls": sum(item["owner"] == "G" for item in interval_records),
        "local_model_calls": sum(item["model_calls"] for item in interval_records if item["owner"] == "W"),
        "model_call_shapes": model_call_shapes,
        "destination_sized_model_calls": 0,
        "noise_scaling_calls": 0,
        "regional_rng_calls": 0,
        "gradient_rms": phase22.grad_rms(result),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "region_barrier_allocated_bytes": barrier_allocations,
        "region_barrier_range_bytes": max(barrier_allocations) - min(barrier_allocations),
        "wall_seconds": time.perf_counter() - started,
    }


def decode(latent: torch.Tensor, path: Path) -> str:
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    try:
        pixels = vae.decode(latent)
    except comfy.model_management.OOM_EXCEPTION:
        comfy.model_management.soft_empty_cache()
        pixels = vae.decode_tiled(latent, tile_x=512, tile_y=512)
    phase2.save_pixels(pixels.cpu(), path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def comparison_sheet(control: Path, candidate: Path) -> None:
    sheet = Image.new("RGB", (1280, 680), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate((("Frozen Terminal", control), ("Literal alternating", candidate))):
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620), Image.Resampling.LANCZOS)
        sheet.paste(image, (column * 640 + (640 - image.width) // 2, 45))
        draw.text((column * 640 + 8, 12), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, _, frozen, frozen_record = phase38.load_control()
    case = phase29.CASES[0]
    geometry = TerminalResamplingGeometry.for_destination(case["destination_hw"])
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device)
    options = phase20.phase8i_options({})
    adapter = Flux2TerminalResamplingAdapter()
    destination = torch.zeros((1, 128, *geometry.destination_hw), device=device)
    adapter.validate_prepared(
        guider=guider,
        model_options=options,
        destination=destination,
        model_sampling=guider.inner_model.model_sampling,
        destination_hw=geometry.destination_hw,
    )
    del destination

    primary, primary_record = run_once(
        name="primary", guider=guider, adapter=adapter, geometry=geometry,
        seed=case["seed"], options=options, device=device,
    )
    atomic_tensor(OUTPUT / "primary.pt", primary)
    primary_record["decoded_rgb_hash"] = decode(primary, OUTPUT / "literal_alternating.png")
    repeat, repeat_record = run_once(
        name="repeat", guider=guider, adapter=adapter, geometry=geometry,
        seed=case["seed"], options=options, device=device,
    )
    bit_exact = bool(torch.equal(primary, repeat))
    report = {
        "experiment": "literal_interpolation_only_alternating_resolution",
        "case": case,
        "geometry": {
            "G": list(geometry.blueprint_hw), "H": list(geometry.destination_hw),
            "F": list(geometry.region_hw), "stride": list(geometry.stride_hw),
            "W": list(geometry.working_hw), "regions": len(geometry.regions()),
        },
        "contract": "single accepted trajectory alternates G/W; only interpolation, restriction, and normalized assembly cross resolutions",
        "primary": primary_record,
        "repeat": repeat_record,
        "control": {
            "artifact": str(PHASE29 / "sigma_0.25.png"),
            "latent_hash": tensor_hash(frozen),
            "grade": "S3",
            "overlap_rms": frozen_record["overlap_rms"],
        },
        "comparison": {"vs_frozen_terminal": difference(primary, frozen)},
        "integrity": {
            "bit_exact_independent_repeat": bit_exact,
            "no_x0_transfer": True,
            "no_noise_scaling": True,
            "no_regional_noise": True,
            "no_persistent_h": True,
            "no_special_coupling": True,
            "all_calls_bounded": all(shape[-2:] in ([45, 45], [64, 64]) for shape in primary_record["model_call_shapes"]),
            "zero_destination_sized_model_calls": True,
            "production_changed": False,
        },
        "semantic_review": "PENDING VISUAL REVIEW",
        "decision": "PENDING VISUAL REVIEW",
    }
    atomic_json(OUTPUT / "report.json", report)
    comparison_sheet(PHASE29 / "sigma_0.25.png", OUTPUT / "literal_alternating.png")
    model.cleanup()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "hash": primary_record["final_h_hash"]}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["semantic_review"] = {
        "grade": "FAIL-S3",
        "object_count": "Many cars, trees, and houses replace the requested exactly one of each.",
        "scene_repetition": "Prompt-complete regional interpretations repeat across the canvas.",
        "horizon": "No single continuous horizon survives; incompatible scene bands and patches dominate.",
        "credible_detail": "Recognizable local objects belong to fragmented repeated scenes and are not a credible detail improvement.",
        "failure_localization": "The literal interpolation-only alternating trajectory itself fails under this setup; no corrective coupling was tested.",
    }
    report["decision"] = "FAIL — LITERAL INTERPOLATION-ONLY ALTERNATING PROTOTYPE LOSES S3"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
