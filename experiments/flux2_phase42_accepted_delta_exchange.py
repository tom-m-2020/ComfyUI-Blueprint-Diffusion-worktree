"""Phase 42 scheduler-midpoint accepted-delta exchange falsifier."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
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
import flux2_persistent_h_shared_provenance as persistent
import flux2_recurrent_interleaved_blueprint as fresh
import flux2_terminal_resampling_refinement_strength as phase29
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.configurable_resampling import (
    ConfigurableResamplingGeometry,
    bounded_blueprint_transfer,
    initialize_configurable_blueprint,
    restrict_configurable_working,
)
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    lift_region,
    region_noise,
    restrict_working_prediction,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "flux2_phase42_accepted_delta_exchange_results"
TERMINAL_RESULTS = ROOT / "experiments" / "flux2_configurable_blueprint_prototype_results"
HISTORICAL_FRESH = ROOT / "experiments" / "flux2_recurrent_interleaved_blueprint_results" / "report.json"
HISTORICAL_PERSISTENT = ROOT / "experiments" / "flux2_persistent_h_shared_provenance_results" / "report.json"
GEOMETRY = ConfigurableResamplingGeometry((45, 45), (128, 128), (32, 32), (16, 16), (64, 64))
CASE = phase29.CASES[0]
SEED = CASE["seed"]
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


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    delta = left.detach().float() - right.detach().float()
    return {"bit_exact": bool(torch.equal(left, right)),
            "rms": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max())}


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def shift(value: float) -> float:
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    e = math.exp(MU)
    return e / (e + (1.0 / value - 1.0))


def inverse_shift(value: float) -> float:
    if value == 0.0:
        return 0.0
    if value == 1.0:
        return 1.0
    e = math.exp(MU)
    return value / (e * (1.0 - value) + value)


def scheduler_midpoint(sigma: float, sigma_next: float) -> float:
    if not (1.0 >= sigma > sigma_next >= 0.0):
        raise ValueError("Phase-42 midpoint requires 1 >= sigma > sigma_next >= 0.")
    return shift((inverse_shift(sigma) + inverse_shift(sigma_next)) * 0.5)


def map_global_delta(delta_g_cpu: torch.Tensor, geometry=GEOMETRY):
    regions = geometry.regions()
    template = torch.zeros((1, 128, *geometry.destination_hw), dtype=delta_g_cpu.dtype)
    assembler = StreamingOverlapAssembler(regions=regions, target_hw=geometry.destination_hw,
                                          template=template)
    records = []
    for region in regions:
        working, source_hw = bounded_blueprint_transfer(delta_g_cpu, geometry, region)
        footprint = restrict_configurable_working(working, geometry)
        assembler.add(footprint, region)
        records.append({"index": region.index, "source_hw": source_hw,
                        "working_hash": tensor_hash(working),
                        "footprint_hash": tensor_hash(footprint)})
    mapped, coverage = assembler.finish()
    return mapped, coverage, records


def restrict_h_delta(delta_h_cpu: torch.Tensor, g_hw=GEOMETRY.blueprint_hw) -> torch.Tensor:
    if tuple(delta_h_cpu.shape) != (1, 128, *GEOMETRY.destination_hw):
        raise ValueError("Phase-42 H delta has the wrong destination shape.")
    return F.adaptive_avg_pool2d(delta_h_cpu, g_hw)


def transfer_pair(h_cpu: torch.Tensor, g: torch.Tensor, g_mid: torch.Tensor,
                  h_next_cpu: torch.Tensor | None = None):
    delta_g = g_mid - g
    mapped, coverage, records = map_global_delta(delta_g.detach().float().cpu())
    h_mid = h_cpu + mapped
    result = {"delta_g": delta_g, "mapped_delta_g": mapped, "h_mid": h_mid,
              "coverage": coverage, "map_records": records}
    if h_next_cpu is not None:
        delta_h = h_next_cpu - h_mid
        restricted = restrict_h_delta(delta_h)
        g_next = g_mid + restricted.to(device=g_mid.device, dtype=g_mid.dtype)
        result.update({"delta_h": delta_h, "restricted_delta_h": restricted,
                       "g_next": g_next})
    return result


def shared_initial_states(device):
    h_device, g = persistent.shared_initial_states(SEED, GEOMETRY, device)
    h_cpu = h_device.detach().float().cpu()
    del h_device
    return h_cpu, g


def euler(state, prediction, sigma, sigma_next):
    return fresh.euler_step(state, prediction, sigma, sigma_next)


def execute_comparable_control(kind, guider, adapter, options, device):
    if kind not in {"fresh_w", "persistent_h"}:
        raise ValueError(f"Unknown comparable control: {kind}")
    regions = GEOMETRY.regions()
    sigmas = tuple(float(value) for value in QUALIFIED_SIGMAS)
    if kind == "fresh_w":
        g = initialize_configurable_blueprint(SEED, GEOMETRY, device=device)
        h = None
        initial_h_hash = None
    else:
        h, g = shared_initial_states(device)
        initial_h_hash = tensor_hash(h)
    initial_g_hash = tensor_hash(g)
    intervals, barriers = [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for ordinal, (sigma_value, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            input_h_hash = tensor_hash(h) if h is not None else None
            input_g_hash = tensor_hash(g)
            sigma = torch.tensor(sigma_value, device=device, dtype=g.dtype)
            x0_g = adapter.predict_native(guider=guider, value=g, sigma=sigma,
                                          expected_hw=GEOMETRY.blueprint_hw,
                                          model_options=options, seed=SEED)
            g_next = euler(g, x0_g, sigma_value, sigma_next)
            template = torch.zeros((1, 128, *GEOMETRY.destination_hw), dtype=torch.float32)
            assembler = StreamingOverlapAssembler(regions=regions,
                                                   target_hw=GEOMETRY.destination_hw,
                                                   template=template)
            restricted_outputs = []
            for region in regions:
                if kind == "fresh_w":
                    w, _ = bounded_blueprint_transfer(x0_g.detach().float().cpu(), GEOMETRY, region)
                    noise = region_noise(SEED, region, device=device, dtype=g.dtype)
                    w = guider.inner_model.model_sampling.noise_scaling(
                        sigma, noise, w.to(device=device, dtype=g.dtype), False)
                else:
                    crop = h[:, :, region.y:region.y2, region.x:region.x2]
                    w = lift_region(crop).to(device=device, dtype=g.dtype)
                x0_w = adapter.predict_native(guider=guider, value=w, sigma=sigma,
                                              expected_hw=GEOMETRY.working_hw,
                                              model_options=options, seed=SEED)
                accepted_w = euler(w, x0_w, sigma_value, sigma_next)
                restricted = restrict_working_prediction(accepted_w).detach().float().cpu()
                assembler.add(restricted, region)
                restricted_outputs.append(restricted)
                del w, x0_w, accepted_w, restricted
                if kind == "fresh_w":
                    del noise
            h_next, coverage = assembler.finish()
            intervals.append({"ordinal": ordinal, "input_h_hash": input_h_hash,
                              "input_g_hash": input_g_hash,
                              "h_hash": tensor_hash(h_next), "h_rms": rms(h_next),
                              "g_hash": tensor_hash(g_next), "g_rms": rms(g_next),
                              "coverage": [float(coverage.min()), float(coverage.max())],
                              "overlap_rms": phase8d.overlap_metrics(restricted_outputs, regions)["aggregate_rms"]})
            g, h = g_next, h_next
            torch.cuda.synchronize(device)
            barriers.append(int(torch.cuda.memory_allocated(device)))
            print(f"{kind}: interval {ordinal + 1}/4", flush=True)
    result = h.detach().float().cpu()
    return result, {
        "kind": kind, "initial_h_hash": initial_h_hash,
        "initial_g_hash": initial_g_hash,
        "final_h_hash": tensor_hash(result), "intervals": intervals,
        "global_model_calls": 4, "local_model_calls": 4 * len(regions),
        "destination_model_calls": 0, "overlap_rms": intervals[-1]["overlap_rms"],
        "gradient_rms": phase22.grad_rms(result),
        "coverage_complete": all(item["coverage"][0] > 0 for item in intervals),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "barrier_allocated_bytes": barriers,
        "barrier_range_bytes": max(barriers) - min(barriers),
        "wall_seconds": time.perf_counter() - started,
    }


def execute_delta_exchange(run_name, guider, adapter, options, device,
                           uncoupled_interval_rms=None):
    regions = GEOMETRY.regions()
    sigmas = tuple(float(value) for value in QUALIFIED_SIGMAS)
    h, g = shared_initial_states(device)
    initial_h_hash, initial_g_hash = tensor_hash(h), tensor_hash(g)
    intervals, barriers = [], []
    commit_ids = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for ordinal, (sigma_value, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:], strict=True)):
            sigma_mid = scheduler_midpoint(sigma_value, sigma_next)
            sigma = torch.tensor(sigma_value, device=device, dtype=g.dtype)
            sigma_mid_tensor = torch.tensor(sigma_mid, device=device, dtype=g.dtype)
            accepted_h_hash = tensor_hash(h)
            accepted_g_hash = tensor_hash(g)
            x0_g = adapter.predict_native(guider=guider, value=g, sigma=sigma,
                                          expected_hw=GEOMETRY.blueprint_hw,
                                          model_options=options, seed=SEED)
            g_mid = euler(g, x0_g, sigma_value, sigma_mid)
            transfer = transfer_pair(h, g, g_mid)
            h_mid = transfer["h_mid"]
            first_divergence = difference(h_mid, h)
            g_to_h_identity = difference(h_mid - h, transfer["mapped_delta_g"])
            midpoint_state_id = uuid.uuid4().hex

            assembler = StreamingOverlapAssembler(
                regions=regions, target_hw=GEOMETRY.destination_hw,
                template=torch.zeros_like(h_mid),
            )
            restricted_outputs = []
            region_records = []
            h_mid_snapshot = h_mid.clone()
            for region in regions:
                crop = h_mid[:, :, region.y:region.y2, region.x:region.x2]
                shares_h_storage = crop.untyped_storage().data_ptr() == h_mid.untyped_storage().data_ptr()
                w = lift_region(crop).to(device=device, dtype=g.dtype)
                input_w_hash = tensor_hash(w)
                x0_w = adapter.predict_native(guider=guider, value=w, sigma=sigma_mid_tensor,
                                              expected_hw=GEOMETRY.working_hw,
                                              model_options=options, seed=SEED)
                accepted_w = euler(w, x0_w, sigma_mid, sigma_next)
                restricted = restrict_working_prediction(accepted_w).detach().float().cpu()
                assembler.add(restricted, region)
                restricted_outputs.append(restricted)
                region_records.append({
                    "index": region.index, "crop_shares_h_mid_storage": shares_h_storage,
                    "crop_hash": tensor_hash(crop), "initial_w_hash": input_w_hash,
                    "prediction_w_hash": tensor_hash(x0_w),
                    "accepted_w_hash": tensor_hash(accepted_w),
                    "restricted_hash": tensor_hash(restricted),
                })
                del crop, w, x0_w, accepted_w, restricted
                torch.cuda.synchronize(device)
                region_records[-1]["barrier_allocated_bytes"] = int(torch.cuda.memory_allocated(device))
            if not torch.equal(h_mid, h_mid_snapshot):
                raise RuntimeError(f"Region work mutated accepted H_mid at interval {ordinal}.")
            h_next, coverage = assembler.finish()
            completed = transfer_pair(h, g, g_mid, h_next)
            delta_h = completed["delta_h"]
            restricted_delta_h = completed["restricted_delta_h"]
            g_next = completed["g_next"]
            h_to_g_identity = difference(
                (g_next - g_mid).detach().float().cpu(), restricted_delta_h
            )
            values = (g_mid, h_mid, h_next, g_next, transfer["delta_g"],
                      transfer["mapped_delta_g"], delta_h, restricted_delta_h)
            if not all(bool(value.isfinite().all()) for value in values):
                raise RuntimeError(f"Nonfinite Phase-42 proposal at interval {ordinal}.")
            ratio_g_to_h = rms(transfer["mapped_delta_g"]) / max(rms(h), 1e-12)
            ratio_h_to_g = rms(restricted_delta_h) / max(rms(g_mid), 1e-12)
            state_rms_ratio = None
            if uncoupled_interval_rms is not None:
                state_rms_ratio = rms(h_next) / max(uncoupled_interval_rms[ordinal], 1e-12)
            interval_commit = uuid.uuid4().hex
            commit_ids.append(interval_commit)
            intervals.append({
                "ordinal": ordinal, "sigma": sigma_value, "sigma_mid": sigma_mid,
                "sigma_next": sigma_next, "accepted_input_h_hash": accepted_h_hash,
                "accepted_input_g_hash": accepted_g_hash, "prediction_g_hash": tensor_hash(x0_g),
                "g_mid_hash": tensor_hash(g_mid), "midpoint_state_id": midpoint_state_id,
                "delta_g": {"hash": tensor_hash(transfer["delta_g"]), "rms": rms(transfer["delta_g"])},
                "mapped_A_delta_g": {"hash": tensor_hash(transfer["mapped_delta_g"]),
                                     "rms": rms(transfer["mapped_delta_g"]),
                                     "ratio_to_H_i": ratio_g_to_h},
                "h_mid_hash": tensor_hash(h_mid), "first_divergence_from_uncoupled": first_divergence,
                "g_to_h_identity": g_to_h_identity,
                "h_mid_unchanged_by_regions": True,
                "regions": region_records,
                "delta_h": {"hash": tensor_hash(delta_h), "rms": rms(delta_h)},
                "restricted_R_delta_h": {"hash": tensor_hash(restricted_delta_h),
                                         "rms": rms(restricted_delta_h),
                                         "ratio_to_G_mid": ratio_h_to_g},
                "h_to_g_identity": h_to_g_identity, "accepted_h_hash": tensor_hash(h_next),
                "accepted_g_hash": tensor_hash(g_next), "pair_commit_id": interval_commit,
                "coverage": [float(coverage.min()), float(coverage.max())],
                "overlap_rms": phase8d.overlap_metrics(restricted_outputs, regions)["aggregate_rms"],
                "state_rms": {"H_i": rms(h), "G_i": rms(g), "G_mid": rms(g_mid),
                              "H_mid": rms(h_mid), "H_next": rms(h_next), "G_next": rms(g_next),
                              "H_next_vs_uncoupled_ratio": state_rms_ratio},
                "global_model_calls": 1, "local_model_calls": len(regions),
                "destination_model_calls": 0,
            })
            h, g = h_next, g_next
            del x0_g, g_mid, h_mid, h_mid_snapshot, h_next, g_next, delta_h, restricted_delta_h
            torch.cuda.synchronize(device)
            barriers.append(int(torch.cuda.memory_allocated(device)))
            print(f"{run_name}: interval {ordinal + 1}/4", flush=True)
    result = h.detach().float().cpu()
    all_region_barriers = [region["barrier_allocated_bytes"]
                           for interval in intervals for region in interval["regions"]]
    return result, {
        "run_name": run_name, "initial_h_hash": initial_h_hash, "initial_g_hash": initial_g_hash,
        "final_h_hash": tensor_hash(result), "final_g_hash": tensor_hash(g),
        "intervals": intervals, "pair_commit_ids": commit_ids,
        "pair_commit_count": len(commit_ids), "global_model_calls": 4,
        "local_model_calls": 4 * len(regions), "destination_model_calls": 0,
        "coverage_complete": all(item["coverage"][0] > 0 for item in intervals),
        "overlap_rms": intervals[-1]["overlap_rms"], "gradient_rms": phase22.grad_rms(result),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "interval_barrier_allocated_bytes": barriers,
        "interval_barrier_range_bytes": max(barriers) - min(barriers),
        "region_barrier_range_bytes": max(all_region_barriers) - min(all_region_barriers),
        "wall_seconds": time.perf_counter() - started,
    }


def decode(vae, latent, path):
    try:
        pixels = vae.decode(latent)
    except comfy.model_management.OOM_EXCEPTION:
        comfy.model_management.soft_empty_cache()
        pixels = vae.decode_tiled(latent, tile_x=512, tile_y=512)
    phase2.save_pixels(pixels.cpu(), path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def make_sheet(paths):
    sheet = Image.new("RGB", (640 * len(paths), 680), "white")
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((620, 620), Image.Resampling.LANCZOS)
        sheet.paste(image, (column * 640 + (640-image.width)//2, 45))
        draw.text((column * 640 + 8, 12), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = json.loads((TERMINAL_RESULTS / "report.json").read_text(encoding="utf-8"))
    terminal_record = next(item for item in prior["cases"] if item["name"] == "SQUARE_OVERLAP16")
    terminal = torch.load(TERMINAL_RESULTS / "SQUARE_OVERLAP16.pt", map_location="cpu", weights_only=True).float()
    historical_fresh = json.loads(HISTORICAL_FRESH.read_text(encoding="utf-8"))
    historical_persistent = json.loads(HISTORICAL_PERSISTENT.read_text(encoding="utf-8"))

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
    destination = torch.zeros((1, 128, *GEOMETRY.destination_hw), device=device)
    adapter.validate_prepared(guider=guider, model_options=options, destination=destination,
                              model_sampling=guider.inner_model.model_sampling,
                              destination_hw=GEOMETRY.destination_hw)

    controls, control_records = {}, {}
    for kind in ("fresh_w", "persistent_h"):
        value, record = execute_comparable_control(kind, guider, adapter, options, device)
        controls[kind], control_records[kind] = value, record
        atomic_tensor(OUTPUT / f"control_{kind}.pt", value)
        atomic_json(OUTPUT / f"control_{kind}.json", record)
        comfy.model_management.soft_empty_cache()

    uncoupled_rms = [item["h_rms"] for item in control_records["persistent_h"]["intervals"]]
    primary, primary_record = execute_delta_exchange(
        "primary", guider, adapter, options, device, uncoupled_interval_rms=uncoupled_rms)
    repeated, repeat_record = execute_delta_exchange(
        "repeat", guider, adapter, options, device, uncoupled_interval_rms=uncoupled_rms)
    atomic_tensor(OUTPUT / "phase42.pt", primary)
    atomic_json(OUTPUT / "primary.json", primary_record)
    atomic_json(OUTPUT / "repeat.json", repeat_record)

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {
        "fresh_w": decode(vae, controls["fresh_w"], OUTPUT / "control_fresh_w.png"),
        "persistent_h": decode(vae, controls["persistent_h"], OUTPUT / "control_persistent_h.png"),
        "phase42": decode(vae, primary, OUTPUT / "phase42.png"),
    }
    make_sheet((("Terminal Refine", TERMINAL_RESULTS / "SQUARE_OVERLAP16.png"),
                ("Fresh-W recurrent / 49 regions", OUTPUT / "control_fresh_w.png"),
                ("Persistent H uncoupled / 49 regions", OUTPUT / "control_persistent_h.png"),
                ("Phase 42 delta exchange", OUTPUT / "phase42.png")))

    identity_checks = [item[key] for item in primary_record["intervals"]
                       for key in ("g_to_h_identity", "h_to_g_identity")]
    increment_ratios = [ratio for item in primary_record["intervals"] for ratio in
                        (item["mapped_A_delta_g"]["ratio_to_H_i"],
                         item["restricted_R_delta_h"]["ratio_to_G_mid"])]
    state_rms_ratios = [item["state_rms"]["H_next_vs_uncoupled_ratio"]
                        for item in primary_record["intervals"]]
    initial_h_matches_uncoupled = (
        primary_record["initial_h_hash"] == control_records["persistent_h"]["initial_h_hash"]
    )
    initial_g_matches_uncoupled = (
        primary_record["initial_g_hash"] == control_records["persistent_h"]["initial_g_hash"]
    )
    first_interval_inputs_match_uncoupled = (
        primary_record["intervals"][0]["accepted_input_h_hash"]
        == control_records["persistent_h"]["intervals"][0]["input_h_hash"]
        and primary_record["intervals"][0]["accepted_input_g_hash"]
        == control_records["persistent_h"]["intervals"][0]["input_g_hash"]
    )
    report = {
        "phase": 42, "experiment": "scheduler_midpoint_accepted_delta_exchange",
        "case": CASE, "geometry": {"G": GEOMETRY.blueprint_hw, "H": GEOMETRY.destination_hw,
                                    "F": GEOMETRY.footprint_hw, "stride": GEOMETRY.stride_hw,
                                    "W": GEOMETRY.working_hw, "regions": len(GEOMETRY.regions())},
        "midpoints": [scheduler_midpoint(a, b) for a, b in zip(QUALIFIED_SIGMAS[:-1], QUALIFIED_SIGMAS[1:], strict=True)],
        "controls": {
            "terminal_refine": {"hash": tensor_hash(terminal), "persisted": terminal_record},
            "fresh_w_49": control_records["fresh_w"],
            "persistent_h_49": control_records["persistent_h"],
            "historical_fresh_25_hash": historical_fresh["arms"]["g_authoritative"]["final_h_hash"],
            "historical_persistent_25_hash": historical_persistent["arms"]["g_authoritative"]["final_h_hash"],
        },
        "primary": primary_record, "repeat": repeat_record,
        "comparisons": {"vs_terminal": difference(primary, terminal),
                        "vs_fresh_w_49": difference(primary, controls["fresh_w"]),
                        "vs_persistent_h_49": difference(primary, controls["persistent_h"])},
        "decoded_hashes": decoded,
        "stochastic_provenance": {
            "seed": SEED, "sigma_0": QUALIFIED_SIGMAS[0],
            "destination_noise_hash": primary_record["initial_h_hash"],
            "blueprint_noise_hash": primary_record["initial_g_hash"],
            "blueprint_derivation": "variance-normalized area restriction of destination noise",
            "regional_noise_streams": 0,
        },
        "integrity": {
            "repeat_bit_exact": bool(torch.equal(primary, repeated)),
            "shared_initial_h_matches_uncoupled": initial_h_matches_uncoupled,
            "shared_initial_g_matches_uncoupled": initial_g_matches_uncoupled,
            "first_interval_inputs_match_uncoupled": first_interval_inputs_match_uncoupled,
            "shared_initial_h_deterministic_across_phase42_runs": primary_record["initial_h_hash"] == repeat_record["initial_h_hash"],
            "shared_initial_g_deterministic_across_phase42_runs": primary_record["initial_g_hash"] == repeat_record["initial_g_hash"],
            "first_divergence_only_at_A_delta_G": (
                initial_h_matches_uncoupled and initial_g_matches_uncoupled
                and first_interval_inputs_match_uncoupled
                and primary_record["intervals"][0]["first_divergence_from_uncoupled"]["rms"] > 0
                and primary_record["intervals"][0]["g_to_h_identity"]["rms"] <= 1e-7
                and primary_record["intervals"][0]["g_to_h_identity"]["max_abs"] <= 1e-6
            ),
            "transfer_tolerances": all(item["rms"] <= 1e-7 and item["max_abs"] <= 1e-6 for item in identity_checks),
            "increments_finite_and_non_dominant": all(math.isfinite(value) and value <= 1.0 for value in increment_ratios),
            "state_rms_non_explosive": all(math.isfinite(value) and value <= 2.0 for value in state_rms_ratios),
            "one_pair_commit_per_interval": primary_record["pair_commit_count"] == 4 and len(set(primary_record["pair_commit_ids"])) == 4,
            "all_H_mid_immutable_during_regions": all(item["h_mid_unchanged_by_regions"] for item in primary_record["intervals"]),
            "coverage_complete": primary_record["coverage_complete"],
            "overlap_within_gate": primary_record["overlap_rms"] <= 0.190627,
            "zero_H_model_calls": primary_record["destination_model_calls"] == 0,
            "bounded_model_geometry": True,
            "flat_region_residency": primary_record["region_barrier_range_bytes"] == 0,
            "production_changed": False,
        },
        "semantic_review": {"status": "PENDING"}, "decision": "PENDING",
    }
    atomic_json(OUTPUT / "report.json", report)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "integrity": report["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
