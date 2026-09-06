"""Fixed-cadence recurrent/interleaved Blueprint discriminator (experiment only)."""
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
import flux2_terminal_resampling_four_interval_local_trajectory as phase38
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_staged_refinement_geometry import Geometry, bounded_transfer
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    initialize_blueprint,
    region_noise,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_recurrent_interleaved_blueprint_results"
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
PHASE38 = ROOT / "experiments" / "flux2_terminal_resampling_four_interval_local_trajectory_results"
ARMS = ("g_authoritative", "bidirectional")


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


def euler_step(state: torch.Tensor, prediction: torch.Tensor, sigma, sigma_next) -> torch.Tensor:
    sigma_value = float(sigma)
    sigma_next_value = float(sigma_next)
    if sigma_value <= 0.0 or sigma_next_value < 0.0 or sigma_next_value >= sigma_value:
        raise ValueError("Recurrent exchange requires one strictly decreasing nonnegative interval.")
    if state.shape != prediction.shape:
        raise ValueError("State and prediction shapes differ.")
    return state + (sigma_next_value - sigma_value) * (state - prediction) / sigma_value


def feedback_state(assembled: torch.Tensor, g_hw: tuple[int, int]) -> torch.Tensor:
    if assembled.ndim != 4 or min(g_hw) < 2:
        raise ValueError("Invalid H-to-G feedback contract.")
    return F.interpolate(assembled, size=g_hw, mode="bilinear", align_corners=False)


def transfer_geometry(geometry: TerminalResamplingGeometry) -> Geometry:
    return Geometry(
        g_hw=geometry.blueprint_hw,
        h_hw=geometry.destination_hw,
        footprint_hw=geometry.region_hw,
        stride_hw=geometry.stride_hw,
        w_hw=geometry.working_hw,
    )


def run_arm(*, name, guider, adapter, geometry, seed, options, device):
    if name not in ARMS:
        raise ValueError(f"Unknown recurrent arm: {name}")
    sigmas = tuple(float(value) for value in QUALIFIED_SIGMAS)
    regions = geometry.regions()
    transfer = transfer_geometry(geometry)
    g = initialize_blueprint(seed, geometry=geometry, device=device)
    initial_g_hash = tensor_hash(g)
    intervals = []
    barriers = []
    total_cuda_ms = 0.0
    started = time.perf_counter()
    final_h = None
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for ordinal, (sigma_value, sigma_next_value) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            sigma = torch.tensor(sigma_value, device=device, dtype=g.dtype)
            accepted_input_g = g.clone()
            begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
            begin.record()
            x0_g = adapter.predict_native(
                guider=guider, value=g, sigma=sigma, expected_hw=geometry.blueprint_hw,
                model_options=options, seed=seed,
            )
            end.record(); torch.cuda.synchronize(device)
            global_ms = float(begin.elapsed_time(end))
            total_cuda_ms += global_ms
            g_proposal = euler_step(g, x0_g, sigma_value, sigma_next_value)
            assembler = StreamingOverlapAssembler(
                regions=regions, target_hw=geometry.destination_hw,
                template=torch.zeros((1, x0_g.shape[1], *geometry.destination_hw), device=device, dtype=x0_g.dtype),
            )
            anchor_assembler = StreamingOverlapAssembler(
                regions=regions, target_hw=geometry.destination_hw,
                template=torch.zeros((1, x0_g.shape[1], *geometry.destination_hw), device=device, dtype=x0_g.dtype),
            )
            restricted_outputs = []
            region_records = []
            local_ms = 0.0
            for region in regions:
                rect = (region.y, region.x, region.height, region.width)
                anchor_w, source_hw = bounded_transfer(x0_g, transfer, rect)
                direct_footprint, _ = bounded_transfer(
                    x0_g,
                    Geometry(transfer.g_hw, transfer.h_hw, transfer.footprint_hw,
                             transfer.stride_hw, transfer.footprint_hw),
                    rect,
                )
                restricted_anchor = restrict_working_prediction(anchor_w)
                transfer_error = difference(restricted_anchor, direct_footprint)
                anchor_assembler.add(restricted_anchor, region)
                noise = region_noise(seed, region, device=device, dtype=anchor_w.dtype)
                w = guider.inner_model.model_sampling.noise_scaling(sigma, noise, anchor_w, False)
                initial_w_hash = tensor_hash(w)
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
                accepted_w = euler_step(w, x0_w, sigma_value, sigma_next_value)
                restricted = restrict_working_prediction(accepted_w)
                assembler.add(restricted, region)
                restricted_outputs.append(restricted.detach().float().cpu())
                region_records.append({
                    "index": region.index, "bounded_source_hw": source_hw,
                    "anchor_hash": tensor_hash(anchor_w), "noise_hash": tensor_hash(noise),
                    "initial_w_hash": initial_w_hash, "prediction_w_hash": tensor_hash(x0_w),
                    "accepted_w_hash": tensor_hash(accepted_w),
                    "restricted_hash": tensor_hash(restricted),
                    "transfer_roundtrip": transfer_error, "cuda_ms": elapsed,
                })
                del anchor_w, direct_footprint, restricted_anchor, noise, w, x0_w, accepted_w, restricted
            assembled, coverage = assembler.finish()
            assembled_anchors, anchor_coverage = anchor_assembler.finish()
            feedback = feedback_state(assembled, geometry.blueprint_hw)
            anchor_roundtrip = feedback_state(assembled_anchors, geometry.blueprint_hw)
            if name == "g_authoritative":
                next_g = g_proposal
            else:
                next_g = feedback
            if not torch.equal(g, accepted_input_g):
                raise RuntimeError(f"Model/local work mutated accepted G at interval {ordinal}.")
            if not all(bool(torch.isfinite(value).all()) for value in (assembled, next_g)):
                raise RuntimeError(f"Nonfinite recurrent state at interval {ordinal}.")
            intervals.append({
                "ordinal": ordinal, "sigma": sigma_value, "sigma_next": sigma_next_value,
                "accepted_input_g_hash": tensor_hash(g), "prediction_g_hash": tensor_hash(x0_g),
                "g_proposal_hash": tensor_hash(g_proposal), "feedback_g_hash": tensor_hash(feedback),
                "accepted_output_g_hash": tensor_hash(next_g), "assembled_h_hash": tensor_hash(assembled),
                "regions": region_records, "g_to_w_anchor_roundtrip": difference(anchor_roundtrip, x0_g),
                "w_to_g_feedback_vs_g_proposal": difference(feedback, g_proposal),
                "overlap_rms": phase8d.overlap_metrics(restricted_outputs, regions)["aggregate_rms"],
                "coverage": [float(coverage.min()), float(coverage.max())],
                "anchor_coverage": [float(anchor_coverage.min()), float(anchor_coverage.max())],
                "global_cuda_ms": global_ms, "local_cuda_ms": local_ms,
                "global_model_calls": 1, "local_model_calls": len(regions),
            })
            final_h = assembled
            g = next_g
            del accepted_input_g, x0_g, g_proposal, feedback, anchor_roundtrip, assembled_anchors
            torch.cuda.synchronize(device)
            barriers.append(int(torch.cuda.memory_allocated(device)))
            print(f"{name}: interval {ordinal + 1}/{len(sigmas) - 1}", flush=True)

    if final_h is None:
        raise RuntimeError("No recurrent interval executed.")
    result = final_h.detach().float().cpu()
    summary = {
        "name": name, "sigmas": sigmas, "initial_g_hash": initial_g_hash,
        "final_g_hash": tensor_hash(g), "final_h_hash": tensor_hash(result),
        "intervals": intervals, "global_model_calls": len(intervals),
        "local_model_calls": len(intervals) * len(regions), "destination_sized_model_calls": 0,
        "coverage_complete": all(item["coverage"][0] > 0.0 for item in intervals),
        "gradient_rms": phase22.grad_rms(result),
        "overlap_rms": intervals[-1]["overlap_rms"],
        "rms_vs_terminal_blueprint": difference(result, torch.load(PHASE29 / "blueprint.pt", map_location="cpu", weights_only=True)["mapped"].float()),
        "low_frequency_rms_vs_terminal_blueprint": phase22.low_frequency_rms(
            result, torch.load(PHASE29 / "blueprint.pt", map_location="cpu", weights_only=True)["mapped"].float()
        ),
        "cuda_ms": total_cuda_ms, "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barriers,
    }
    return result, summary


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


def comparison_sheet(paths: list[tuple[str, Path]]) -> None:
    sheet = Image.new("RGB", (640 * len(paths), 680), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620), Image.Resampling.LANCZOS)
        x = column * 640 + (640 - image.width) // 2
        sheet.paste(image, (x, 45))
        draw.text((column * 640 + 8, 12), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _, _, frozen, frozen_record = phase38.load_control()
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
    previous = {}
    for name in ARMS:
        path = OUTPUT / f"{name}.pt"
        if path.exists():
            previous[name] = torch.load(path, map_location="cpu", weights_only=True).float()
    results = {}
    records = {}
    decoded = {}
    for name in ARMS:
        value, record = run_arm(
            name=name, guider=guider, adapter=adapter, geometry=geometry,
            seed=case["seed"], options=options, device=device,
        )
        atomic_tensor(OUTPUT / f"{name}.pt", value)
        decoded[name] = decode(value, OUTPUT / f"{name}.png")
        results[name] = value
        records[name] = record
    report = {
        "experiment": "recurrent_interleaved_blueprint", "case": case,
        "contract": "shared-sigma G prediction anchors freshly noise-scaled W; both accept one Euler interval",
        "controls": {
            "frozen_terminal": {"hash": tensor_hash(frozen), "grade": "S3", "metrics": {
                "gradient_rms": frozen_record["gradient_rms"], "overlap_rms": frozen_record["overlap_rms"]}},
            "unguided_four_interval": {"hash": tensor_hash(unguided), "grade": "S3", "metrics": {
                "gradient_rms": unguided_record["gradient_rms"], "overlap_rms": unguided_record["overlap_rms"]}},
        },
        "arms": records, "decoded_rgb_hashes": decoded,
        "comparisons": {name: {
            "vs_frozen_terminal": difference(value, frozen),
            "vs_unguided_four_interval": difference(value, unguided),
        } for name, value in results.items()},
        "integrity": {
            "production_changed": False, "comfyui_core_changed": False,
            "independent_repeat_bit_exact": {
                name: bool(name in previous and torch.equal(value, previous[name]))
                for name, value in results.items()
            },
            "same_initial_g": records[ARMS[0]]["initial_g_hash"] == records[ARMS[1]]["initial_g_hash"],
        },
    }
    atomic_json(OUTPUT / "report.json", report)
    comparison_sheet([
        ("Frozen Terminal", PHASE29 / "sigma_0.25.png"),
        ("Unguided four-interval W", PHASE38 / "four_interval.png"),
        ("G-authoritative recurrent", OUTPUT / "g_authoritative.png"),
        ("Bidirectional recurrent", OUTPUT / "bidirectional.png"),
    ])
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "hashes": {
        name: records[name]["final_h_hash"] for name in ARMS}}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["semantic_review"] = {
        "g_authoritative_grade": "FAIL-S3",
        "bidirectional_grade": "FAIL-S3",
        "object_count": "Both recurrent arms produce a repeated 5x5-like lattice of cars, trees, and houses instead of exactly one of each.",
        "composition": "The continuous single field/horizon is replaced by stacked local scene bands; cross-tile whole-scene composition is lost.",
        "credible_detail": "Local objects are recognizable, but their detail is attached to duplicated independent local scenes and therefore does not satisfy the composition/detail gate.",
        "feedback_localization": "The two arms are identical in interval 0. Bidirectional feedback closely tracks its local assembly in middle intervals but neither creates nor repairs the repeated-object failure shared with the G-authoritative arm.",
        "cause": "Fresh W construction at the near-one Blueprint sigmas gives every bounded region enough independent generative freedom to instantiate the prompt as a local scene; normalized overlap cannot restore one global object layout.",
    }
    report["decision"] = "FAIL — FRESH SAME-SIGMA W RECONSTRUCTION FROM CLEAN G AT NEAR-ONE SIGMA LOSES S3"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
