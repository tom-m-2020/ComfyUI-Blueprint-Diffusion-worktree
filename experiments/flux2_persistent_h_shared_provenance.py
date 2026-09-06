"""Persistent-H shared stochastic provenance discriminator (experiment only)."""
from __future__ import annotations

import hashlib
import json
import math
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
import flux2_recurrent_interleaved_blueprint as fresh
import flux2_terminal_resampling_four_interval_local_trajectory as phase38
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    lift_region,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_persistent_h_shared_provenance_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
PHASE38 = ROOT / "experiments" / "flux2_terminal_resampling_four_interval_local_trajectory_results"
FRESH = ROOT / "experiments" / "flux2_recurrent_interleaved_blueprint_results"
ARMS = ("g_authoritative", "h_feedback")


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


def shared_initial_states(seed: int, geometry: TerminalResamplingGeometry, device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    h = torch.randn((1, 128, *geometry.destination_hw), generator=generator, dtype=torch.float32).to(device)
    g = F.adaptive_avg_pool2d(h, geometry.blueprint_hw)
    counts = []
    for destination, blueprint in zip(geometry.destination_hw, geometry.blueprint_hw):
        values = [
            math.ceil((index + 1) * destination / blueprint)
            - math.floor(index * destination / blueprint)
            for index in range(blueprint)
        ]
        counts.append(torch.tensor(values, device=device, dtype=g.dtype))
    scale = torch.sqrt(counts[0][:, None] * counts[1][None, :])[None, None]
    return h, g * scale


def feedback_state(h: torch.Tensor, g_hw: tuple[int, int]) -> torch.Tensor:
    if h.ndim != 4 or min(g_hw) < 2:
        raise ValueError("Invalid persistent-H feedback geometry.")
    return F.interpolate(h, size=g_hw, mode="bilinear", align_corners=False)


def max_shared_overlap_error(crops: list[torch.Tensor], regions) -> float:
    maximum = 0.0
    for left_index, left in enumerate(regions):
        for right_index in range(left_index + 1, len(regions)):
            right = regions[right_index]
            y0, y1 = max(left.y, right.y), min(left.y2, right.y2)
            x0, x1 = max(left.x, right.x), min(left.x2, right.x2)
            if y0 >= y1 or x0 >= x1:
                continue
            left_view = crops[left_index][..., y0-left.y:y1-left.y, x0-left.x:x1-left.x]
            right_view = crops[right_index][..., y0-right.y:y1-right.y, x0-right.x:x1-right.x]
            maximum = max(maximum, float((left_view - right_view).abs().max()))
    return maximum


def run_arm(*, name, guider, adapter, geometry, seed, options, device):
    if name not in ARMS:
        raise ValueError(f"Unknown persistent-H arm: {name}")
    sigmas = tuple(float(value) for value in QUALIFIED_SIGMAS)
    regions = geometry.regions()
    h, g = shared_initial_states(seed, geometry, device)
    initial_h_hash, initial_g_hash = tensor_hash(h), tensor_hash(g)
    intervals, barriers = [], []
    total_cuda_ms = 0.0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for ordinal, (sigma_value, sigma_next_value) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            sigma = torch.tensor(sigma_value, device=device, dtype=h.dtype)
            input_h, input_g = h.clone(), g.clone()
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            x0_g = adapter.predict_native(
                guider=guider, value=g, sigma=sigma, expected_hw=geometry.blueprint_hw,
                model_options=options, seed=seed,
            )
            end.record(); torch.cuda.synchronize(device)
            global_ms = float(begin.elapsed_time(end))
            total_cuda_ms += global_ms
            g_proposal = fresh.euler_step(g, x0_g, sigma_value, sigma_next_value)
            assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw, template=h)
            restricted_inputs, restricted_outputs, region_records = [], [], []
            local_ms = 0.0
            for region in regions:
                crop = h[:, :, region.y:region.y2, region.x:region.x2]
                shares_storage = crop.untyped_storage().data_ptr() == h.untyped_storage().data_ptr()
                w = lift_region(crop)
                restricted_input = restrict_working_prediction(w)
                provenance_error = difference(restricted_input, crop)
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                begin.record()
                x0_w = adapter.predict_native(
                    guider=guider, value=w, sigma=sigma, expected_hw=geometry.working_hw,
                    model_options=options, seed=seed,
                )
                end.record(); torch.cuda.synchronize(device)
                elapsed = float(begin.elapsed_time(end))
                local_ms += elapsed
                total_cuda_ms += elapsed
                accepted_w = fresh.euler_step(w, x0_w, sigma_value, sigma_next_value)
                restricted = restrict_working_prediction(accepted_w)
                assembler.add(restricted, region)
                restricted_inputs.append(restricted_input.detach().float().cpu())
                restricted_outputs.append(restricted.detach().float().cpu())
                region_records.append({
                    "index": region.index, "source_crop_shares_h_storage": shares_storage,
                    "source_crop_hash": tensor_hash(crop), "initial_w_hash": tensor_hash(w),
                    "prediction_w_hash": tensor_hash(x0_w), "accepted_w_hash": tensor_hash(accepted_w),
                    "restricted_hash": tensor_hash(restricted),
                    "restriction_vs_h_crop": provenance_error, "cuda_ms": elapsed,
                })
                del crop, w, restricted_input, x0_w, accepted_w, restricted
            h_next, coverage = assembler.finish()
            feedback = feedback_state(h_next, geometry.blueprint_hw)
            g_next = g_proposal if name == "g_authoritative" else feedback
            if not torch.equal(h, input_h) or not torch.equal(g, input_g):
                raise RuntimeError(f"Model calls mutated accepted state at interval {ordinal}.")
            if not bool(h_next.isfinite().all()) or not bool(g_next.isfinite().all()):
                raise RuntimeError(f"Nonfinite persistent-H state at interval {ordinal}.")
            intervals.append({
                "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next_value,
                "accepted_input_h_hash": tensor_hash(h), "accepted_input_g_hash": tensor_hash(g),
                "prediction_g_hash": tensor_hash(x0_g), "g_proposal_hash": tensor_hash(g_proposal),
                "feedback_g_hash": tensor_hash(feedback), "accepted_output_g_hash": tensor_hash(g_next),
                "accepted_output_h_hash": tensor_hash(h_next), "regions": region_records,
                "shared_input_overlap_max_abs": max_shared_overlap_error(restricted_inputs, regions),
                "input_restriction_max_abs": max(item["restriction_vs_h_crop"]["max_abs"] for item in region_records),
                "all_source_crops_share_h_storage": all(item["source_crop_shares_h_storage"] for item in region_records),
                "output_overlap_rms": phase8d.overlap_metrics(restricted_outputs, regions)["aggregate_rms"],
                "feedback_vs_g_proposal": difference(feedback, g_proposal),
                "coverage": [float(coverage.min()), float(coverage.max())],
                "global_cuda_ms": global_ms, "local_cuda_ms": local_ms,
                "global_model_calls": 1, "local_model_calls": len(regions),
                "regional_rng_calls": 0, "noise_scaling_calls": 0,
            })
            h, g = h_next, g_next
            del input_h, input_g, x0_g, g_proposal, feedback
            torch.cuda.synchronize(device)
            barriers.append(int(torch.cuda.memory_allocated(device)))
            print(f"{name}: interval {ordinal + 1}/{len(sigmas)-1}", flush=True)

    result = h.detach().float().cpu()
    terminal_blueprint = torch.load(PHASE29 / "blueprint.pt", map_location="cpu", weights_only=True)["mapped"].float()
    return result, {
        "name": name, "sigmas": sigmas, "initial_h_hash": initial_h_hash,
        "initial_g_hash": initial_g_hash, "final_h_hash": tensor_hash(result),
        "final_g_hash": tensor_hash(g), "intervals": intervals,
        "global_model_calls": len(intervals), "local_model_calls": len(intervals) * len(regions),
        "destination_sized_model_calls": 0, "regional_rng_calls": 0, "noise_scaling_calls": 0,
        "coverage_complete": all(item["coverage"][0] > 0.0 for item in intervals),
        "interval0_shared_provenance": {
            "all_crops_share_h_storage": intervals[0]["all_source_crops_share_h_storage"],
            "restriction_max_abs": intervals[0]["input_restriction_max_abs"],
            "overlap_max_abs": intervals[0]["shared_input_overlap_max_abs"],
        },
        "gradient_rms": phase22.grad_rms(result), "overlap_rms": intervals[-1]["output_overlap_rms"],
        "rms_vs_terminal_blueprint": difference(result, terminal_blueprint),
        "low_frequency_rms_vs_terminal_blueprint": phase22.low_frequency_rms(result, terminal_blueprint),
        "cuda_ms": total_cuda_ms, "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barriers,
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


def comparison_sheet(paths) -> None:
    sheet = Image.new("RGB", (640 * len(paths), 680), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620), Image.Resampling.LANCZOS)
        sheet.paste(image, (column * 640 + (640-image.width)//2, 45))
        draw.text((column * 640 + 8, 12), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, _, frozen, frozen_record = phase38.load_control()
    unguided = torch.load(PHASE38 / "primary" / "assembled.pt", map_location="cpu", weights_only=True).float()
    unguided_record = json.loads((PHASE38 / "primary" / "summary.json").read_text(encoding="utf-8"))
    rejected = json.loads((FRESH / "report.json").read_text(encoding="utf-8"))
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
    adapter.validate_prepared(guider=guider, model_options=options, destination=destination,
                              model_sampling=guider.inner_model.model_sampling,
                              destination_hw=geometry.destination_hw)
    previous = {name: torch.load(OUTPUT / f"{name}.pt", map_location="cpu", weights_only=True).float()
                for name in ARMS if (OUTPUT / f"{name}.pt").exists()}
    values, records, decoded = {}, {}, {}
    for name in ARMS:
        value, record = run_arm(name=name, guider=guider, adapter=adapter, geometry=geometry,
                                seed=case["seed"], options=options, device=device)
        atomic_tensor(OUTPUT / f"{name}.pt", value)
        decoded[name] = decode(value, OUTPUT / f"{name}.png")
        values[name], records[name] = value, record
    report = {
        "experiment": "persistent_h_shared_stochastic_provenance", "case": case,
        "controls": {
            "frozen_terminal": {"hash": tensor_hash(frozen), "grade": "S3",
                                "gradient_rms": frozen_record["gradient_rms"], "overlap_rms": frozen_record["overlap_rms"]},
            "unguided_four_interval": {"hash": tensor_hash(unguided), "grade": "S3",
                                       "gradient_rms": unguided_record["gradient_rms"], "overlap_rms": unguided_record["overlap_rms"]},
            "rejected_fresh_w": {name: {"hash": rejected["arms"][name]["final_h_hash"],
                                         "grade": "FAIL-S3"} for name in ("g_authoritative", "bidirectional")},
        },
        "arms": records, "decoded_rgb_hashes": decoded,
        "comparisons": {name: {"vs_frozen_terminal": difference(value, frozen),
                                "vs_unguided_four_interval": difference(value, unguided)}
                        for name, value in values.items()},
        "integrity": {
            "production_changed": False, "comfyui_core_changed": False,
            "same_initial_h": records[ARMS[0]]["initial_h_hash"] == records[ARMS[1]]["initial_h_hash"],
            "same_initial_g": records[ARMS[0]]["initial_g_hash"] == records[ARMS[1]]["initial_g_hash"],
            "independent_repeat_bit_exact": {name: bool(name in previous and torch.equal(value, previous[name]))
                                              for name, value in values.items()},
        },
    }
    atomic_json(OUTPUT / "report.json", report)
    comparison_sheet([
        ("Frozen Terminal", PHASE29 / "sigma_0.25.png"),
        ("Unguided four-interval W", PHASE38 / "four_interval.png"),
        ("Rejected fresh-W G-auth", FRESH / "g_authoritative.png"),
        ("Persistent H / G-auth", OUTPUT / "g_authoritative.png"),
        ("Persistent H / H-feedback", OUTPUT / "h_feedback.png"),
    ])
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"),
                      "hashes": {name: records[name]["final_h_hash"] for name in ARMS}}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["semantic_review"] = {
        "g_authoritative_grade": "FAIL-S3",
        "h_feedback_grade": "FAIL-S3",
        "object_count": "Both arms contain many repeated cars, trees, and houses rather than exactly one of each.",
        "composition": "Both arms form five stacked local scene bands and lose the single continuous field/horizon composition.",
        "credible_detail": "Individual local objects are more developed than Terminal, but the detail belongs to duplicated local prompt interpretations and fails the joint semantic gate.",
        "fragmentation": "Shared H provenance changes local appearance relative to fresh-W reconstruction but does not prevent locally complete prompt interpretations.",
        "arm_equivalence": "Final H is bit-identical between arms because the permitted G-authoritative and H-feedback operations do not feed G back into persistent H.",
    }
    report["decision"] = "FAIL — PERSISTENT SHARED-PROVENANCE H STILL FRAGMENTS; STOP THIS DISCRIMINATOR"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
