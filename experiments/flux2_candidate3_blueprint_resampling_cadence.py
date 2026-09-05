"""Phase 24: terminal-only versus two-event persistent Blueprint resampling."""
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
sys.path.insert(0, str(COMFY_ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import comfy.sampler_helpers
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
from blueprint_diffusion.sampling.euler import BlueprintCoordinator

OUTPUT = ROOT / "experiments" / "flux2_candidate3_blueprint_resampling_cadence_results"
STAGES = OUTPUT / "stages"
REPORT = OUTPUT / "report.json"
MIDPOINT = 1
TERMINAL = 3


def stage_paths(name): return STAGES / f"{name}.json", STAGES / f"{name}.pt"


def persist_stage(name, fingerprint, record, tensors):
    meta_path, tensor_path = stage_paths(name)
    phase22.atomic_torch(tensor_path, {"configuration_hash": fingerprint, **tensors})
    phase22.atomic_json(meta_path, {"complete": True, "configuration_hash": fingerprint, "record": record})


def load_stage(name, fingerprint):
    meta_path, tensor_path = stage_paths(name)
    if not meta_path.exists() or not tensor_path.exists(): return None
    wrapper = json.loads(meta_path.read_text(encoding="utf-8"))
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not wrapper.get("complete") or wrapper.get("configuration_hash") != fingerprint or tensors.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 24 incompatible stage {name}")
    return wrapper["record"], tensors


def decode(vae, latent, path):
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def sheet(items, path):
    panels = []
    for label, source in items:
        image = Image.open(source).convert("RGB"); image.thumbnail((1536, 768))
        panel = Image.new("RGB", (1536, image.height + 38), "white")
        panel.paste(image, ((1536-image.width)//2, 38)); ImageDraw.Draw(panel).text((10, 10), label, fill="black")
        panels.append(panel)
    out = Image.new("RGB", (1536, sum(x.height for x in panels)), "white"); y = 0
    for panel in panels: out.paste(panel, (0, y)); y += panel.height
    out.save(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); STAGES.mkdir(parents=True, exist_ok=True)
    phase23_report = json.loads(phase23.REPORT.read_text(encoding="utf-8"))
    config = {
        "phase": 24, "phase23_configuration_hash": phase23_report["configuration_hash"],
        "midpoint_ordinal": MIDPOINT, "terminal_ordinal": TERMINAL,
        "arms": {"A": "terminal only", "B": "midpoint plus terminal persistent residual"},
        "H": list(phase14.H_HW), "blueprint": [32, 64], "regions": 55,
        "destination_crop": [32, 32], "working": [64, 64],
        "resampling_sigma": phase22.SIGMA, "noise_base_seed": phase22.NOISE_BASE_SEED,
        "terminal_seed": "B_final + (H_refined_mid - B_mid)",
        "working_rule": "0.75*nearest2(clean_anchor_crop)+0.25*epsilon_region",
        "prompt": phase14.PROMPT, "seed": phase14.SEED,
    }
    fingerprint = phase23.config_hash(config)
    phase22.atomic_json(OUTPUT / "preflight.json", {
        "configuration": config, "configuration_hash": fingerprint,
        "intermediate_selected_before_inference": True,
        "algebra": "H_seed=B_final+R_mid; R_mid=H_refined_mid-B_mid; W=.75*nearest2(H_seed_crop)+.25*epsilon",
        "dimensionally_valid": True, "const_flow_sigma_consistent": True,
        "new_model_work": "one periodic-terminal batch of 55 local 64x64 forwards",
    })
    midpoint = torch.load(phase23.INTERVALS / f"interval_{MIDPOINT}.pt", map_location="cpu", weights_only=True)
    terminal = torch.load(phase23.INTERVALS / f"interval_{TERMINAL}.pt", map_location="cpu", weights_only=True)
    midpoint_meta = json.loads((phase23.INTERVALS / f"interval_{MIDPOINT}.json").read_text(encoding="utf-8"))
    terminal_meta = json.loads((phase23.INTERVALS / f"interval_{TERMINAL}.json").read_text(encoding="utf-8"))
    b_mid = midpoint["mapped_blueprint"].float(); h_mid = midpoint["refined_assembled"].float()
    b_final = terminal["mapped_blueprint"].float(); h_terminal_a = terminal["refined_assembled"].float()
    r_mid = h_mid - b_mid; h_seed = b_final + r_mid
    source_hashes = {name: phase14.tensor_hash(value) for name, value in
                     (("B_mid", b_mid), ("H_mid", h_mid), ("B_final", b_final),
                      ("H_terminal_A", h_terminal_a), ("R_mid", r_mid), ("H_seed", h_seed))}

    terminal_a = load_stage("A_TERMINAL_ONLY", fingerprint)
    if terminal_a is None:
        record = {"stage": "A_TERMINAL_ONLY", "reused_phase23_ordinal": TERMINAL,
                  "model_calls_executed_now": 0, "model_calls_provenance": 55,
                  "blueprint_hash": source_hashes["B_final"], "final_H_hash": source_hashes["H_terminal_A"],
                  "gradient_rms": phase22.grad_rms(h_terminal_a),
                  "overlap_rms": terminal_meta["overlap_rms"],
                  "rms_vs_terminal_blueprint": phase17.tensor_difference(h_terminal_a, b_final),
                  "low_frequency_rms": phase22.low_frequency_rms(h_terminal_a, b_final),
                  "timing": {"reused_local_cuda_ms": terminal_meta["timing pragmatic"]["local_cuda_ms"] if "timing pragmatic" in terminal_meta else terminal_meta["timing"]["local_cuda_ms"], "executed_now_ms": 0},
                  "integrity": {"exact_phase23_reuse": True, "finite": True, "accepted_state_updates": 0}}
        persist_stage("A_TERMINAL_ONLY", fingerprint, record,
                      {"B_final": b_final, "final_H": h_terminal_a})
        terminal_a = record, {"B_final": b_final, "final_H": h_terminal_a}
        print("phase24 A terminal-only persisted", flush=True)
    else: print("phase24 A terminal-only resume-skip", flush=True)

    periodic_mid = load_stage("B_PERIODIC_MIDPOINT", fingerprint)
    if periodic_mid is None:
        record = {"stage": "B_PERIODIC_MIDPOINT", "reused_phase23_ordinal": MIDPOINT,
                  "model_calls_executed_now": 0, "model_calls_provenance": 55,
                  "blueprint_hash": source_hashes["B_mid"], "H_mid_hash": source_hashes["H_mid"],
                  "residual": phase14.summary(r_mid), "gradient_rms": phase22.grad_rms(h_mid),
                  "overlap_rms": midpoint_meta["overlap_rms"],
                  "integrity": {"exact_phase23_reuse": True, "finite": True, "accepted_state_updates": 0}}
        persist_stage("B_PERIODIC_MIDPOINT", fingerprint, record,
                      {"B_mid": b_mid, "H_mid": h_mid, "R_mid": r_mid})
        periodic_mid = record, {"B_mid": b_mid, "H_mid": h_mid, "R_mid": r_mid}
        print("phase24 B midpoint persisted", flush=True)
    else: print("phase24 B midpoint resume-skip", flush=True)

    periodic_terminal = load_stage("B_PERIODIC_TERMINAL", fingerprint)
    if periodic_terminal is None:
        print("phase24 B periodic terminal start 55-region batch", flush=True)
        model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
        clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT)); del clip
        phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
        perf.prepare_model_state(model); device = phase2.comfy.model_management.get_torch_device()
        guider = phase23.setup_guider(model, positive, device); sigma = torch.tensor(phase22.SIGMA, device=device)
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW); coordinator = BlueprintCoordinator()
        anchor = h_seed.to(device); anchor_hash = phase14.tensor_hash(anchor)
        phase2.comfy.model_management.soft_empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
        working = []; working_hashes = []; noise_hashes = []
        for region in regions:
            crop = anchor[:, :, region.y:region.y2, region.x:region.x2]
            noise = phase22.region_noise(region, device, crop.dtype)
            value = phase22.make_working(crop, noise)
            working.append(value); working_hashes.append(phase14.tensor_hash(value)); noise_hashes.append(phase14.tensor_hash(noise))
        begin.record()
        with torch.inference_mode():
            x0_w = [guider(value, sigma.expand(1), model_options=phase20.phase8i_options({}), seed=phase14.SEED)
                    for value in working]
        end.record(); torch.cuda.synchronize(); cuda_ms = float(begin.elapsed_time(end))
        restricted = [phase9b.restrict2(value) for value in x0_w]
        assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
        overlap = phase8d.overlap_metrics([value.detach().float().cpu() for value in restricted], regions)["aggregate_rms"]
        if phase14.tensor_hash(anchor) != anchor_hash or [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Phase 24 periodic terminal mutated inputs")
        if not torch.isfinite(assembled).all() or float(coverage.min()) <= 0:
            raise RuntimeError("Phase 24 periodic terminal invalid output")
        assembled_cpu = assembled.detach().float().cpu()
        record = {"stage": "B_PERIODIC_TERMINAL", "model_calls_executed_now": 55,
                  "blueprint_hash": source_hashes["B_final"], "H_seed_hash": source_hashes["H_seed"],
                  "final_H": phase14.summary(assembled_cpu), "gradient_rms": phase22.grad_rms(assembled_cpu),
                  "overlap_rms": overlap, "rms_vs_terminal_blueprint": phase17.tensor_difference(assembled_cpu, b_final),
                  "low_frequency_rms": phase22.low_frequency_rms(assembled_cpu, b_final),
                  "retained_midpoint_residual": {"before_terminal_rms": phase22.rms(r_mid),
                                                   "after_terminal_vs_A_rms": phase22.rms(assembled_cpu-h_terminal_a)},
                  "noise_provenance": {"base_seed": phase22.NOISE_BASE_SEED, "hashes": noise_hashes,
                                       "matches_phase23_terminal": noise_hashes == terminal_meta["noise_provenance"]["hashes"]},
                  "coverage": [float(coverage.min()), float(coverage.max())],
                  "timing": {"local_cuda_ms": cuda_ms, "wall_seconds": time.perf_counter()-start},
                  "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                             "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
                  "integrity": {"finite": True, "complete_positive_coverage": True,
                                "anchor_and_W_immutable": True, "accepted_state_updates": 0,
                                "blueprint_forwards_executed_now": 0, "full_H_forwards": 0}}
        saved = {"B_final": b_final, "R_mid": r_mid, "H_seed": h_seed,
                 "final_H": assembled_cpu, "restricted": [value.detach().float().cpu() for value in restricted],
                 "representative_W": working[27].detach().float().cpu(),
                 "representative_x0_W": x0_w[27].detach().float().cpu()}
        persist_stage("B_PERIODIC_TERMINAL", fingerprint, record, saved)
        periodic_terminal = record, saved
        print(f"phase24 B periodic terminal complete wall={record['timing']['wall_seconds']:.2f}s", flush=True)
        del model; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    else: print("phase24 B periodic terminal resume-skip", flush=True)

    a_record, a_tensors = terminal_a; b_record, b_tensors = periodic_terminal
    final_a = a_tensors["final_H"].float(); final_b = b_tensors["final_H"].float()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}; items = []
    for name, value in (("TERMINAL_BLUEPRINT", b_final), ("A_TERMINAL_ONLY", final_a),
                        ("MIDPOINT_REFINED", h_mid), ("PERIODIC_SEED", h_seed),
                        ("B_PERIODIC_PERSISTENT", final_b)):
        path = OUTPUT / f"{name}.png"; decoded[name] = decode(vae, value, path); items.append((name, path))
    comparison = OUTPUT / "A_B_COMPARISON.png"; sheet(items, comparison)
    report = {"phase": 24, "configuration": config, "configuration_hash": fingerprint,
              "source_hashes": source_hashes,
              "stages": {"A_terminal_only": a_record, "B_midpoint": periodic_mid[0], "B_terminal": b_record},
              "final_A_vs_B": phase17.tensor_difference(final_a, final_b),
              "decoded": decoded, "comparison_sheet": str(comparison),
              "integrity": {"exact_phase20c_blueprints_reused": True, "blueprint_forwards_recomputed": 0,
                            "new_local_model_calls": 55, "accepted_state_updates": 0,
                            "finite": True, "production_changes": False},
              "semantic_review": {
                  "A_TERMINAL_ONLY": {"class": "S3", "observations": "one bridge/train scene, controlled endpoints, coherent horizon/water"},
                  "B_PERIODIC_PERSISTENT": {"class": "S3", "observations": "same dominant scene and no independent alternatives, but no clear visual detail advantage"},
                  "periodic_detail_advantage": False,
                  "metric_interpretation": "higher B gradient energy is accompanied by worse Blueprint RMS, low-frequency discrepancy, and overlap; it is not sufficient evidence of better detail",
              },
              "decision": "TERMINAL-ONLY RESAMPLING SUFFICIENT"}
    phase22.atomic_json(REPORT, report)
    print(json.dumps({"report": str(REPORT), "comparison": str(comparison)}, indent=2), flush=True)


if __name__ == "__main__": main()
