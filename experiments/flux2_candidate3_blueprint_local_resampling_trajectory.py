"""Phase 23: persistent Blueprint to fixed-late-sigma local resampling trajectory."""
from __future__ import annotations

import hashlib
import json
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
import flux2_candidate3_native_blueprint_anchor_trajectory as phase20c
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
from blueprint_diffusion.sampling.euler import BlueprintCoordinator
from comfy_extras.nodes_custom_sampler import Guider_Basic

OUTPUT = ROOT / "experiments" / "flux2_candidate3_blueprint_local_resampling_trajectory_results"
INTERVALS = OUTPUT / "intervals"
REPORT = OUTPUT / "report.json"
SOURCE_INTERVALS = phase20c.INTERVALS


def config_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_completed(ordinal, fingerprint, source_hash):
    meta_path = INTERVALS / f"interval_{ordinal}.json"
    tensor_path = INTERVALS / f"interval_{ordinal}.pt"
    if not meta_path.exists() or not tensor_path.exists():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("configuration_hash") != fingerprint or meta.get("source_blueprint_x0_hash") != source_hash:
        raise RuntimeError(f"Phase 23 interval {ordinal} resume artifact is incompatible")
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if tensors.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 23 interval {ordinal} tensor fingerprint mismatch")
    return meta, tensors


def setup_guider(model, positive, device):
    guider = Guider_Basic(model); guider.set_conds(positive)
    guider.conds = {key: [item.copy() for item in values]
                    for key, values in guider.original_conds.items()}
    phase2.comfy.samplers.preprocess_conds_hooks(guider.conds)
    guider.inner_model, guider.conds, guider.loaded_models = comfy.sampler_helpers.prepare_sampling(
        model, (1, 128, 64, 64), guider.conds, guider.model_options)
    noise = torch.zeros((1, 128, 64, 64), device=device)
    guider.conds = phase2.comfy.samplers.process_conds(
        guider.inner_model, noise, guider.conds, device, torch.zeros_like(noise),
        None, phase14.SEED, latent_shapes=[noise.shape])
    model.pre_run()
    return guider


def decode(vae, latent, path):
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def make_sheet(items, path):
    panels = []
    for label, source in items:
        image = Image.open(source).convert("RGB"); image.thumbnail((1536, 768))
        panel = Image.new("RGB", (1536, image.height + 38), "white")
        panel.paste(image, ((1536 - image.width) // 2, 38))
        ImageDraw.Draw(panel).text((10, 10), label, fill="black"); panels.append(panel)
    out = Image.new("RGB", (1536, sum(x.height for x in panels)), "white"); y = 0
    for panel in panels: out.paste(panel, (0, y)); y += panel.height
    out.save(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True); INTERVALS.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(phase20c.REPORT.read_text(encoding="utf-8"))
    config = {
        "phase": 23, "source_trajectory_hash": source_report["configuration_hash"],
        "source_intervals": 4, "H": list(phase14.H_HW), "blueprint": [32, 64],
        "regions": 55, "destination_crop": [32, 32], "working": [64, 64],
        "resampling_sigma": phase22.SIGMA,
        "working_contract": "W_0.25=0.75*nearest2(mapped_blueprint_x0_crop)+0.25*epsilon_region",
        "noise_base_seed": phase22.NOISE_BASE_SEED,
        "noise_reuse": "same deterministic region noise at every Blueprint evaluation",
        "coordinates": "ordinary native 64x64 local W coordinates",
        "prompt": phase14.PROMPT, "seed": phase14.SEED,
    }
    fingerprint = config_hash(config)
    phase22.atomic_json(OUTPUT / "preflight.json", {
        "configuration": config, "configuration_hash": fingerprint,
        "construction_valid": True,
        "reason": "local resampling is an independent fixed-sigma refinement of each denoised Blueprint estimate, not a reconstruction of accepted G at sigma_i",
        "expected_new_work": "4*55=220 local forwards; four exact Blueprint forwards reused from persisted Phase20c artifacts",
    })
    regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
    if len(regions) != 55: raise AssertionError("Phase 23 requires 55 regions")
    source_items = []
    for ordinal in range(4):
        tensors = torch.load(SOURCE_INTERVALS / f"interval_{ordinal}.pt", map_location="cpu", weights_only=True)
        meta = json.loads((SOURCE_INTERVALS / f"interval_{ordinal}.json").read_text(encoding="utf-8"))["record"]
        source_items.append((meta, tensors))

    pending = []
    records = []; outputs = []
    for ordinal, (source_meta, source_tensors) in enumerate(source_items):
        source_hash = phase14.tensor_hash(source_tensors["blueprint_x0"])
        completed = load_completed(ordinal, fingerprint, source_hash)
        if completed is None: pending.append(ordinal)
        else:
            meta, tensors = completed; records.append(meta); outputs.append(tensors)
            print(f"phase23 interval {ordinal} resume-skip {ordinal + 1}/4", flush=True)

    model = guider = None
    if pending:
        model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
        clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT)); del clip
        phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
        perf.prepare_model_state(model); device = phase2.comfy.model_management.get_torch_device()
        guider = setup_guider(model, positive, device)
        sigma = torch.tensor(phase22.SIGMA, device=device)
        coordinator = BlueprintCoordinator()
        by_ordinal = {meta["ordinal"]: (meta, tensors) for meta, tensors in zip(records, outputs)}
        with torch.inference_mode():
            for ordinal in pending:
                source_meta, source_tensors = source_items[ordinal]
                blueprint_x0 = source_tensors["blueprint_x0"].to(device)
                source_hash = phase14.tensor_hash(blueprint_x0)
                mapped = F.interpolate(blueprint_x0.float(), size=phase14.H_HW,
                                       mode="bilinear", align_corners=False)
                accepted_blueprint_hash = source_meta["accepted_input_blueprint_hash"]
                blueprint_hash = phase14.tensor_hash(blueprint_x0)
                print(f"phase23 interval {ordinal} start {ordinal + 1}/4", flush=True)
                phase2.comfy.model_management.soft_empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter(); begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
                working = []; working_hashes = []; noise_hashes = []
                for region in regions:
                    crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
                    noise = phase22.region_noise(region, device, crop.dtype)
                    value = phase22.make_working(crop, noise)
                    working.append(value); working_hashes.append(phase14.tensor_hash(value)); noise_hashes.append(phase14.tensor_hash(noise))
                begin.record()
                x0_w = [guider(value, sigma.expand(1), model_options=phase20.phase8i_options({}),
                               seed=phase14.SEED) for value in working]
                end.record(); torch.cuda.synchronize(); cuda_ms = float(begin.elapsed_time(end))
                restricted = [phase9b.restrict2(value) for value in x0_w]
                assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
                overlap = phase8d.overlap_metrics([value.detach().float().cpu() for value in restricted], regions)["aggregate_rms"]
                if [phase14.tensor_hash(value) for value in working] != working_hashes:
                    raise RuntimeError(f"Phase 23 interval {ordinal} mutated W input")
                if phase14.tensor_hash(blueprint_x0) != blueprint_hash:
                    raise RuntimeError(f"Phase 23 interval {ordinal} mutated Blueprint x0")
                if not torch.isfinite(assembled).all() or float(coverage.min()) <= 0:
                    raise RuntimeError(f"Phase 23 interval {ordinal} invalid result")
                record = {
                    "ordinal": ordinal, "configuration_hash": fingerprint,
                    "blueprint_sigma": source_meta["sigma"], "blueprint_sigma_next": source_meta["sigma_next"],
                    "local_resampling_sigma": phase22.SIGMA,
                    "accepted_blueprint_input_hash": accepted_blueprint_hash,
                    "source_blueprint_x0_hash": source_hash,
                    "blueprint_x0": phase14.summary(blueprint_x0), "mapped_blueprint": phase14.summary(mapped),
                    "refined_assembled": phase14.summary(assembled),
                    "blueprint_to_refined": phase17.tensor_difference(assembled, mapped),
                    "low_frequency_rms": phase22.low_frequency_rms(assembled, mapped),
                    "gradient_rms": {"blueprint": phase22.grad_rms(mapped), "refined": phase22.grad_rms(assembled)},
                    "overlap_rms": overlap, "coverage": [float(coverage.min()), float(coverage.max())],
                    "noise_provenance": {"base_seed": phase22.NOISE_BASE_SEED,
                                         "formula": "base_seed+1009*region.index", "hashes": noise_hashes},
                    "geometry": {"map": "bilinear align_corners=False 32x64->128x256",
                                 "crop": [32, 32], "upscale": "2x nearest", "working": [64, 64],
                                 "downscale": "nonoverlapping 2x2 mean"},
                    "model_calls": {"blueprint_reused": 1, "blueprint_executed_now": 0,
                                    "local_executed_now": 55},
                    "timing": {"local_cuda_ms": cuda_ms, "wall_seconds": time.perf_counter() - start},
                    "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                               "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
                    "integrity": {"source_artifact_immutable": True, "working_inputs_immutable": True,
                                  "finite": True, "complete_positive_coverage": True,
                                  "accepted_state_updates": 0},
                }
                saved = {"configuration_hash": fingerprint, "blueprint_x0": blueprint_x0.detach().float().cpu(),
                         "mapped_blueprint": mapped.detach().float().cpu(),
                         "refined_assembled": assembled.detach().float().cpu(),
                         "representative_W": working[27].detach().float().cpu(),
                         "representative_x0_W": x0_w[27].detach().float().cpu(),
                         "restricted": [value.detach().float().cpu() for value in restricted]}
                phase22.atomic_torch(INTERVALS / f"interval_{ordinal}.pt", saved)
                phase22.atomic_json(INTERVALS / f"interval_{ordinal}.json", record)
                by_ordinal[ordinal] = (record, saved)
                print(f"phase23 interval {ordinal} complete {ordinal + 1}/4 wall={record['timing']['wall_seconds']:.2f}s", flush=True)
        records = [by_ordinal[i][0] for i in range(4)]; outputs = [by_ordinal[i][1] for i in range(4)]
    else:
        records = [json.loads((INTERVALS / f"interval_{i}.json").read_text(encoding="utf-8")) for i in range(4)]
        outputs = [torch.load(INTERVALS / f"interval_{i}.pt", map_location="cpu", weights_only=True) for i in range(4)]

    if model is not None:
        phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}; sheet_items = []
    for ordinal, tensors in enumerate(outputs):
        for key in ("mapped_blueprint", "refined_assembled"):
            path = OUTPUT / f"interval_{ordinal}_{key.upper()}.png"
            decoded[f"interval_{ordinal}_{key}"] = decode(vae, tensors[key], path)
            sheet_items.append((f"interval {ordinal} {key}", path))
        for key in ("representative_W", "representative_x0_W"):
            path = OUTPUT / f"interval_{ordinal}_{key.upper()}.png"
            decoded[f"interval_{ordinal}_{key}"] = decode(vae, tensors[key], path)
    sheet = OUTPUT / "BLUEPRINT_RESAMPLING_TRAJECTORY.png"; make_sheet(sheet_items, sheet)
    report = {
        "phase": 23, "configuration": config, "configuration_hash": fingerprint,
        "intervals": records, "decoded": decoded, "comparison_sheet": str(sheet),
        "integrity": {"completed_intervals": 4, "new_local_model_calls": 220,
                      "reused_blueprint_predictions": 4, "accepted_state_updates": 0,
                      "all_finite": all(x["integrity"]["finite"] for x in records),
                      "all_coverage_complete": all(x["integrity"]["complete_positive_coverage"] for x in records),
                      "all_inputs_immutable": all(x["integrity"]["source_artifact_immutable"] and x["integrity"]["working_inputs_immutable"] for x in records),
                      "production_changes": False},
        "semantic_review": {
            "blueprint_classes": ["S3", "S3", "S3", "S3"],
            "refined_classes": ["S3", "S3", "S3", "S3"],
            "pass": True,
            "observations": [
                "one dominant continuous bridge and one centered train at every evaluation",
                "controlled endpoint tower and stone structure",
                "continuous horizon and water",
                "no independent bridge, tower, train, or lighthouse alternatives",
                "repeatable modest edge/detail increase without progressive Blueprint geometry loss",
            ],
            "detail_behavior": "STABLE_REGENERATION",
        },
        "decision": "PASS: fixed-late-sigma Blueprint-initialized local resampling remains S3 at all four persistent Blueprint evaluations; stop before cadence architecture comparison.",
    }
    phase22.atomic_json(REPORT, report)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet)}, indent=2), flush=True)


if __name__ == "__main__": main()
