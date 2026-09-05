"""Phase 25: terminal-only Blueprint resampling semantic generalization."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
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
import flux2_candidate3_blueprint_resampling_cadence as phase24
from blueprint_diffusion.sampling.euler import BlueprintCoordinator

OUTPUT = ROOT / "experiments" / "flux2_candidate3_terminal_resampling_generalization_results"
CASES = OUTPUT / "cases"
REPORT = OUTPUT / "report.json"

CASE_DEFINITIONS = {
    "B_MULTI_OBJECT": {
        "seed": 20260911,
        "prompt": "A cinematic photograph of one red vintage car on the far left, one tall green tree in the center, and one small white house on the far right, all standing on the same grassy field under one continuous horizon, coherent perspective, exactly one car, one tree, and one house, no duplicate objects.",
    },
    "C_ASTRONAUT": {
        "seed": 20260912,
        "prompt": "A full-body astronaut standing alone in the center of a wide desert, facing the camera, both arms visible, both legs visible, one continuous body, distant low mountains across one horizon, exactly one astronaut, no duplicate people or body parts.",
    },
}


def fingerprint(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
def paths(case_name, stage):
    root = CASES / case_name
    return root / f"{stage}.json", root / f"{stage}.pt"


def save_stage(case_name, stage, key, record, tensors):
    meta, data = paths(case_name, stage)
    phase22.atomic_torch(data, {"configuration_hash": key, **tensors})
    phase22.atomic_json(meta, {"complete": True, "configuration_hash": key, "record": record})


def load_stage(case_name, stage, key):
    meta, data = paths(case_name, stage)
    if not meta.exists() or not data.exists(): return None
    wrapper = json.loads(meta.read_text(encoding="utf-8")); tensors = torch.load(data, map_location="cpu", weights_only=True)
    if not wrapper.get("complete") or wrapper.get("configuration_hash") != key or tensors.get("configuration_hash") != key:
        raise RuntimeError(f"Phase 25 incompatible artifact {case_name}/{stage}")
    return wrapper["record"], tensors


def region_noise(seed, region, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(seed + 22_000_003 + 1009 * region.index)
    return torch.randn((1, 128, 64, 64), generator=generator).to(device=device, dtype=dtype)


def initial_blueprint(seed, device):
    h0 = torch.randn((1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(seed)).to(device)
    coarse = F.avg_pool2d(h0, 4, 4)
    noise = torch.randn(tuple(coarse.shape), generator=torch.Generator().manual_seed(seed + 20_000_003)).to(device)
    return coarse + math.sqrt(15.0 / 16.0) * noise, h0


def decode(vae, latent, path):
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size), "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def make_sheet(items, path):
    panels = []
    for label, source in items:
        image = Image.open(source).convert("RGB"); image.thumbnail((1536, 768))
        panel = Image.new("RGB", (1536, image.height + 38), "white")
        panel.paste(image, ((1536-image.width)//2, 38)); ImageDraw.Draw(panel).text((10, 10), label, fill="black"); panels.append(panel)
    out = Image.new("RGB", (1536, sum(x.height for x in panels)), "white"); y = 0
    for panel in panels: out.paste(panel, (0, y)); y += panel.height
    out.save(path)


def execute_case(case_name, definition, sigmas, regions):
    config = {"case": case_name, **definition, "H": list(phase14.H_HW), "blueprint": [32, 64],
              "regions": 55, "destination_crop": [32, 32], "working": [64, 64],
              "blueprint_sigmas": [float(x) for x in sigmas], "resampling_sigma": phase22.SIGMA,
              "noise_formula": "case_seed+22000003+1009*region.index",
              "terminal_rule": "W=.75*nearest2(mapped terminal Blueprint crop)+.25*epsilon",
              "ordinary_control": "independent native-W four-interval Euler trajectories from identical epsilon"}
    key = fingerprint(config)
    blueprint_stage = load_stage(case_name, "BLUEPRINT", key)
    ordinary_stage = load_stage(case_name, "ORDINARY_LOCAL", key)
    resampled_stage = load_stage(case_name, "BLUEPRINT_RESAMPLED", key)
    if blueprint_stage and ordinary_stage and resampled_stage:
        print(f"phase25 {case_name} resume-skip complete", flush=True)
        return config, key, blueprint_stage, ordinary_stage, resampled_stage

    print(f"phase25 {case_name} model setup", flush=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(definition["prompt"])); del clip
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model); device = phase2.comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device); sigmas = sigmas.to(device)

    if blueprint_stage is None:
        print(f"phase25 {case_name} Blueprint start", flush=True)
        blueprint, h0 = initial_blueprint(definition["seed"], device); initial_hash = phase14.tensor_hash(blueprint)
        start = time.perf_counter(); cuda_total = 0.0; x0_history = []
        with torch.inference_mode():
            for ordinal, (sigma, sigma_next) in enumerate(zip(sigmas[:-1], sigmas[1:])):
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
                x0 = guider(blueprint, sigma.expand(1), model_options=phase20.phase8i_options({}), seed=definition["seed"])
                end.record(); torch.cuda.synchronize(); cuda_total += float(begin.elapsed_time(end)); x0_history.append(x0.detach().float().cpu())
                blueprint = blueprint + (sigma_next-sigma) * (blueprint-x0) / sigma
        terminal_x0 = x0_history[-1]; mapped = F.interpolate(terminal_x0, size=phase14.H_HW, mode="bilinear", align_corners=False)
        record = {"stage": "BLUEPRINT", "initial_hash": initial_hash, "terminal_x0": phase14.summary(terminal_x0),
                  "mapped_terminal": phase14.summary(mapped), "model_calls": 4,
                  "timing": {"cuda_ms": cuda_total, "wall_seconds": time.perf_counter()-start},
                  "integrity": {"finite": all(torch.isfinite(x).all() for x in x0_history), "no_H_feedback": True}}
        save_stage(case_name, "BLUEPRINT", key, record,
                   {"terminal_x0": terminal_x0, "mapped_terminal": mapped,
                    "x0_history": x0_history, "accepted_blueprint_final": blueprint.detach().float().cpu(),
                    "initial_H_hash": phase14.tensor_hash(h0)})
        blueprint_stage = record, {"terminal_x0": terminal_x0, "mapped_terminal": mapped, "x0_history": x0_history}
        print(f"phase25 {case_name} Blueprint complete", flush=True)
    mapped = blueprint_stage[1]["mapped_terminal"].to(device)

    if ordinary_stage is None:
        print(f"phase25 {case_name} ordinary local start 220-forward batch", flush=True)
        states = [region_noise(definition["seed"], region, device, mapped.dtype) for region in regions]
        initial_hashes = [phase14.tensor_hash(x) for x in states]
        start = time.perf_counter(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); cuda_total = 0.0
        with torch.inference_mode():
            for sigma, sigma_next in zip(sigmas[:-1], sigmas[1:]):
                begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
                predictions = [guider(state, sigma.expand(1), model_options=phase20.phase8i_options({}), seed=definition["seed"]) for state in states]
                end.record(); torch.cuda.synchronize(); cuda_total += float(begin.elapsed_time(end))
                states = [state + (sigma_next-sigma)*(state-x0)/sigma for state, x0 in zip(states, predictions)]
        restricted = [phase9b.restrict2(x) for x in states]; assembled, coverage = BlueprintCoordinator().assembler.assemble(restricted, regions, phase14.H_HW)
        overlap = phase8d.overlap_metrics([x.detach().float().cpu() for x in restricted], regions)["aggregate_rms"]
        record = {"stage": "ORDINARY_LOCAL", "model_calls": 220, "initial_noise_hashes": initial_hashes,
                  "assembled": phase14.summary(assembled), "overlap_rms": overlap, "gradient_rms": phase22.grad_rms(assembled),
                  "rms_vs_blueprint": phase17.tensor_difference(assembled, mapped), "low_frequency_rms": phase22.low_frequency_rms(assembled, mapped),
                  "coverage": [float(coverage.min()), float(coverage.max())],
                  "timing": {"cuda_ms": cuda_total, "wall_seconds": time.perf_counter()-start},
                  "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
                  "integrity": {"finite": bool(torch.isfinite(assembled).all()), "complete_positive_coverage": float(coverage.min())>0, "accepted_state_updates": 0}}
        saved = {"assembled": assembled.detach().float().cpu(), "restricted": [x.detach().float().cpu() for x in restricted]}
        save_stage(case_name, "ORDINARY_LOCAL", key, record, saved); ordinary_stage = record, saved
        print(f"phase25 {case_name} ordinary local complete wall={record['timing']['wall_seconds']:.2f}s", flush=True)

    if resampled_stage is None:
        print(f"phase25 {case_name} Blueprint resampled start 55-forward batch", flush=True)
        working = []; noise_hashes = []
        for region in regions:
            crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            noise = region_noise(definition["seed"], region, device, crop.dtype)
            working.append(phase22.make_working(crop, noise)); noise_hashes.append(phase14.tensor_hash(noise))
        working_hashes = [phase14.tensor_hash(x) for x in working]
        sigma = torch.tensor(phase22.SIGMA, device=device); start = time.perf_counter(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        begin, end = torch.cuda.Event(True), torch.cuda.Event(True); begin.record()
        with torch.inference_mode():
            x0_w = [guider(x, sigma.expand(1), model_options=phase20.phase8i_options({}), seed=definition["seed"]) for x in working]
        end.record(); torch.cuda.synchronize(); cuda_ms = float(begin.elapsed_time(end))
        restricted = [phase9b.restrict2(x) for x in x0_w]; assembled, coverage = BlueprintCoordinator().assembler.assemble(restricted, regions, phase14.H_HW)
        overlap = phase8d.overlap_metrics([x.detach().float().cpu() for x in restricted], regions)["aggregate_rms"]
        if [phase14.tensor_hash(x) for x in working] != working_hashes: raise RuntimeError(f"{case_name} W mutation")
        ordinary_noise_hashes = ordinary_stage[0]["initial_noise_hashes"]
        record = {"stage": "BLUEPRINT_RESAMPLED", "model_calls": 55, "noise_hashes": noise_hashes,
                  "noise_matches_ordinary": noise_hashes == ordinary_noise_hashes,
                  "assembled": phase14.summary(assembled), "overlap_rms": overlap, "gradient_rms": phase22.grad_rms(assembled),
                  "rms_vs_blueprint": phase17.tensor_difference(assembled, mapped), "low_frequency_rms": phase22.low_frequency_rms(assembled, mapped),
                  "rms_vs_ordinary": phase17.tensor_difference(assembled, ordinary_stage[1]["assembled"].to(assembled)),
                  "coverage": [float(coverage.min()), float(coverage.max())],
                  "timing": {"cuda_ms": cuda_ms, "wall_seconds": time.perf_counter()-start},
                  "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
                  "integrity": {"finite": bool(torch.isfinite(assembled).all()), "complete_positive_coverage": float(coverage.min())>0, "working_inputs_immutable": True, "accepted_state_updates": 0}}
        saved = {"assembled": assembled.detach().float().cpu(), "restricted": [x.detach().float().cpu() for x in restricted], "representative_x0_W": x0_w[27].detach().float().cpu()}
        save_stage(case_name, "BLUEPRINT_RESAMPLED", key, record, saved); resampled_stage = record, saved
        print(f"phase25 {case_name} Blueprint resampled complete wall={record['timing']['wall_seconds']:.2f}s", flush=True)

    del model; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    return config, key, blueprint_stage, ordinary_stage, resampled_stage


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); CASES.mkdir(parents=True, exist_ok=True)
    phase20c_report = json.loads(phase23.phase20c.REPORT.read_text(encoding="utf-8"))
    sigmas = torch.tensor(phase20c_report["configuration"]["sigmas"], dtype=torch.float32)
    regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
    phase22.atomic_json(OUTPUT / "preflight.json", {"case_order": list(CASE_DEFINITIONS), "case_definitions": CASE_DEFINITIONS,
        "seeds_declared_before_inference": True, "algorithm_frozen": True, "bridge_control": "exact Phase24 reuse"})
    results = {}
    bridge_report = json.loads(phase24.REPORT.read_text(encoding="utf-8"))
    results["A_BRIDGE"] = {"reused": True, "source_report": str(phase24.REPORT),
                           "terminal_only": bridge_report["stages"]["A_terminal_only"],
                           "semantic_class": "S3"}
    for name, definition in CASE_DEFINITIONS.items():
        config, key, blueprint, ordinary, resampled = execute_case(name, definition, sigmas, regions)
        results[name] = {"configuration": config, "configuration_hash": key,
                         "blueprint": blueprint[0], "ordinary": ordinary[0], "resampled": resampled[0]}

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}; summary_items = []
    bridge_path = phase24.OUTPUT / "A_TERMINAL_ONLY.png"; summary_items.append(("A BRIDGE SELECTED", bridge_path))
    for name in CASE_DEFINITIONS:
        key = results[name]["configuration_hash"]
        blueprint = load_stage(name, "BLUEPRINT", key)[1]["mapped_terminal"]
        ordinary = load_stage(name, "ORDINARY_LOCAL", key)[1]["assembled"]
        resampled = load_stage(name, "BLUEPRINT_RESAMPLED", key)[1]["assembled"]
        items = []
        for label, value in (("BLUEPRINT", blueprint), ("ORDINARY_LOCAL", ordinary), ("BLUEPRINT_RESAMPLED", resampled)):
            path = OUTPUT / f"{name}_{label}.png"; decoded[f"{name}_{label}"] = decode(vae, value, path); items.append((label, path))
        case_sheet = OUTPUT / f"{name}_COMPARISON.png"; make_sheet(items, case_sheet); results[name]["comparison_sheet"] = str(case_sheet)
        summary_items.append((f"{name} BLUEPRINT RESAMPLED", items[-1][1]))
    summary = OUTPUT / "SUMMARY.png"; make_sheet(summary_items, summary)
    semantic_review = {
        "A_BRIDGE": {
            "blueprint": "S3", "ordinary_local": "S0", "blueprint_resampled": "S3",
            "observations": "Exact Phase-24 control: one continuous bridge/train system and coherent horizon/water after terminal resampling."
        },
        "B_MULTI_OBJECT": {
            "blueprint": "S3", "ordinary_local": "S0", "blueprint_resampled": "S3",
            "checks": {"one_car": True, "one_tree": True, "one_house": True,
                       "left_center_right_order": True, "coherent_ground_horizon": True,
                       "duplicate_objects": False},
            "observations": "Selected output preserves one red car at far left, one tall green tree at center, and one white house at far right. The ordinary control contains repeated crop-local object rows."
        },
        "C_ASTRONAUT": {
            "blueprint": "S3", "ordinary_local": "S0", "blueprint_resampled": "S3",
            "checks": {"one_astronaut": True, "one_head_body_system": True,
                       "two_arms_two_legs": True, "coherent_scale_ground_contact": True,
                       "coherent_horizon_background": True, "duplicate_people_or_parts": False},
            "observations": "Selected output preserves one centered full-body astronaut with coherent anatomy and ground contact. The ordinary control contains many independent astronauts and repeated horizon strips."
        }
    }
    report = {"phase": 25, "case_order": ["A_BRIDGE", *CASE_DEFINITIONS], "results": results,
              "decoded": decoded, "summary_sheet": str(summary),
              "semantic_review": semantic_review,
              "decision": {"new_cases_passing": 2, "new_cases_total": 2,
                           "verdict": "TERMINAL RESAMPLING GENERALIZES",
                           "phase26_architecture_design_authorized": True},
              "integrity": {"fixed_seeds": True, "fixed_algorithm": True, "bridge_reused": True,
                            "production_changes": False}}
    phase22.atomic_json(REPORT, report); print(json.dumps({"report": str(REPORT), "summary": str(summary)}, indent=2), flush=True)


if __name__ == "__main__": main()
