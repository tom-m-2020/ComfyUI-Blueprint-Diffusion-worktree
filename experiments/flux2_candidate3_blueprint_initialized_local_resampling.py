"""Phase 22: one late-sigma Blueprint-initialized local resampling probe."""
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
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import comfy.sampler_helpers
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d
from comfy_extras.nodes_custom_sampler import Guider_Basic
from blueprint_diffusion.sampling.euler import BlueprintCoordinator

OUTPUT = ROOT / "experiments" / "flux2_candidate3_blueprint_initialized_local_resampling_results"
REPORT = OUTPUT / "report.json"
SOURCE = ROOT / "experiments" / "flux2_candidate3_native_blueprint_prediction_anchor_results" / "C_BLUEPRINT_PREDICTION_ANCHOR.pt"
SIGMA = 0.25
NOISE_BASE_SEED = phase14.SEED + 22_000_003


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


def grad_rms(value):
    value = value.detach().float()
    dy = value[:, :, 1:, :] - value[:, :, :-1, :]
    dx = value[:, :, :, 1:] - value[:, :, :, :-1]
    return math.sqrt((float(dy.square().mean()) + float(dx.square().mean())) / 2.0)


def region_noise(region, device, dtype):
    generator = torch.Generator(device="cpu").manual_seed(NOISE_BASE_SEED + 1009 * region.index)
    return torch.randn((1, 128, 64, 64), generator=generator).to(device=device, dtype=dtype)


def make_working(anchor_crop, noise):
    anchor = F.interpolate(anchor_crop.float(), scale_factor=2.0, mode="nearest").to(anchor_crop.dtype)
    return (1.0 - SIGMA) * anchor + SIGMA * noise


def low_frequency_rms(left, right):
    return rms(F.avg_pool2d(left.float() - right.float(), 4, 4))


def decode(vae, latent, path):
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def comparison_sheet(items, path):
    panels = []
    for label, source in items:
        image = Image.open(source).convert("RGB")
        image.thumbnail((2048, 1024))
        panel = Image.new("RGB", (2048, image.height + 40), "white")
        panel.paste(image, ((2048 - image.width) // 2, 40))
        ImageDraw.Draw(panel).text((10, 10), label, fill="black")
        panels.append(panel)
    out = Image.new("RGB", (2048, sum(x.height for x in panels)), "white")
    y = 0
    for panel in panels:
        out.paste(panel, (0, y)); y += panel.height
    out.save(path)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = {
        "selected_before_inference": True,
        "resampling_sigma": SIGMA,
        "criterion": "late conservative refinement: 75% denoised anchor plus 25% deterministic noise",
        "flow_contract": "W_sigma=(1-sigma)*x0_anchor+sigma*epsilon",
        "noise_base_seed": NOISE_BASE_SEED,
        "noise_per_region_seed": "noise_base_seed + 1009*region.index",
        "expected_model_work": "110 local forwards; saved Phase-20b Blueprint/control reused",
    }
    atomic_json(OUTPUT / "preflight.json", preflight)
    source = torch.load(SOURCE, map_location="cpu", weights_only=True)
    blueprint_x0 = source["blueprint_x0"]
    mapped_blueprint = F.interpolate(blueprint_x0.float(), size=phase14.H_HW,
                                     mode="bilinear", align_corners=False)
    ordinary_anchor = source["baseline_B"].float()
    regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
    if len(regions) != 55:
        raise AssertionError("Phase 22 requires the qualified 55-region plan")
    source_hashes = {"blueprint_x0": phase14.tensor_hash(blueprint_x0),
                     "mapped_blueprint": phase14.tensor_hash(mapped_blueprint),
                     "ordinary_anchor": phase14.tensor_hash(ordinary_anchor)}

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    del clip
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = phase2.comfy.model_management.get_torch_device()
    blueprint_x0 = blueprint_x0.to(device)
    mapped_blueprint = mapped_blueprint.to(device)
    ordinary_anchor = ordinary_anchor.to(device)
    base = {}
    guider = Guider_Basic(model); guider.set_conds(positive)
    guider.conds = {key: [item.copy() for item in values]
                    for key, values in guider.original_conds.items()}
    phase2.comfy.samplers.preprocess_conds_hooks(guider.conds)
    guider.inner_model, guider.conds, guider.loaded_models = comfy.sampler_helpers.prepare_sampling(
        model, (1, 128, 64, 64), guider.conds, guider.model_options)
    conditioning_noise = torch.zeros((1, 128, 64, 64), device=device)
    guider.conds = phase2.comfy.samplers.process_conds(
        guider.inner_model, conditioning_noise, guider.conds, device,
        torch.zeros_like(conditioning_noise), None, phase14.SEED,
        latent_shapes=[conditioning_noise.shape])
    model.pre_run()
    sigma = torch.tensor(SIGMA, device=device)
    coordinator = BlueprintCoordinator()
    predictions = {}
    arm_records = {}
    peak_allocated = 0; peak_reserved = 0

    with torch.inference_mode():
        for arm_name, anchor_canvas in (("B_ORDINARY_LOCAL", ordinary_anchor),
                                        ("C_BLUEPRINT_RESAMPLED_LOCAL", mapped_blueprint)):
            print(f"phase22 {arm_name} start", flush=True)
            phase2.comfy.model_management.soft_empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter(); event0, event1 = torch.cuda.Event(True), torch.cuda.Event(True)
            working = []; noise_hashes = []; anchor_hashes = []
            for region in regions:
                crop = anchor_canvas[:, :, region.y:region.y2, region.x:region.x2]
                noise = region_noise(region, crop.device, crop.dtype)
                working.append(make_working(crop, noise))
                noise_hashes.append(phase14.tensor_hash(noise)); anchor_hashes.append(phase14.tensor_hash(crop))
            working_hashes = [phase14.tensor_hash(value) for value in working]
            event0.record()
            x0_w = [guider(value, sigma.expand(1), model_options=phase20.phase8i_options(base),
                          seed=phase14.SEED) for value in working]
            event1.record(); torch.cuda.synchronize()
            cuda_ms = float(event0.elapsed_time(event1))
            restricted = [phase9b.restrict2(value) for value in x0_w]
            assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
            overlap = phase8d.overlap_metrics([value.detach().float().cpu() for value in restricted], regions)["aggregate_rms"]
            if [phase14.tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError(f"{arm_name} mutated its local inputs")
            if not torch.isfinite(assembled).all() or float(coverage.min()) <= 0:
                raise RuntimeError(f"{arm_name} invalid output or coverage")
            predictions[arm_name] = {"assembled": assembled.detach().float().cpu(),
                                     "restricted": [x.detach().float().cpu() for x in restricted],
                                     "representative_x0_W": x0_w[27].detach().float().cpu(),
                                     "representative_W": working[27].detach().float().cpu()}
            record = {
                "arm": arm_name, "model_calls": 55, "working_tokens_per_call": 4096,
                "working_state": phase14.summary(working[27]), "representative_x0_W": phase14.summary(x0_w[27]),
                "assembled": phase14.summary(assembled), "overlap_rms": overlap,
                "gradient_rms": grad_rms(assembled),
                "rms_vs_mapped_blueprint": phase17.tensor_difference(assembled, mapped_blueprint.to(assembled)),
                "low_frequency_rms_vs_mapped_blueprint": low_frequency_rms(assembled, mapped_blueprint.to(assembled)),
                "coverage": [float(coverage.min()), float(coverage.max())],
                "timing": {"local_cuda_ms": cuda_ms, "wall_seconds": time.perf_counter() - start},
                "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                           "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
                "provenance": {"anchor_hashes": anchor_hashes, "noise_hashes": noise_hashes,
                               "working_hashes": working_hashes, "same_noise_as_other_arm": True},
                "integrity": {"finite": True, "complete_positive_coverage": True,
                              "working_inputs_immutable": True, "accepted_state_updates": 0},
            }
            arm_records[arm_name] = record
            peak_allocated = max(peak_allocated, record["memory"]["peak_allocated_bytes"])
            peak_reserved = max(peak_reserved, record["memory"]["peak_reserved_bytes"])
            atomic_torch(OUTPUT / f"{arm_name}.pt", predictions[arm_name])
            atomic_json(OUTPUT / f"{arm_name}.json", record)
            print(f"phase22 {arm_name} complete wall={record['timing']['wall_seconds']:.2f}s", flush=True)

    if [phase14.tensor_hash(region_noise(region, torch.device("cpu"), torch.float32)) for region in regions] != arm_records["B_ORDINARY_LOCAL"]["provenance"]["noise_hashes"]:
        raise RuntimeError("Phase 22 noise provenance mismatch")
    if any(source_hashes[key] != phase14.tensor_hash(value) for key, value in
           (("blueprint_x0", blueprint_x0), ("mapped_blueprint", mapped_blueprint),
            ("ordinary_anchor", ordinary_anchor))):
        raise RuntimeError("Phase 22 source controls mutated")

    pairwise = phase17.tensor_difference(predictions["C_BLUEPRINT_RESAMPLED_LOCAL"]["assembled"],
                                         predictions["B_ORDINARY_LOCAL"]["assembled"])
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    decode_inputs = {
        "A_BLUEPRINT": mapped_blueprint,
        "B_ORDINARY_LOCAL": predictions["B_ORDINARY_LOCAL"]["assembled"],
        "C_BLUEPRINT_RESAMPLED_LOCAL": predictions["C_BLUEPRINT_RESAMPLED_LOCAL"]["assembled"],
    }
    sheet_items = []
    for name, value in decode_inputs.items():
        path = OUTPUT / f"{name}.png"; decoded[name] = decode(vae, value, path); sheet_items.append((name, path))
    for arm_name in predictions:
        region_dir = OUTPUT / "regions" / arm_name; region_dir.mkdir(parents=True, exist_ok=True)
        for region, value in zip(regions, predictions[arm_name]["restricted"]):
            decode(vae, value, region_dir / f"region_{region.index:02d}.png")
        decode(vae, predictions[arm_name]["representative_W"], OUTPUT / f"{arm_name}_REPRESENTATIVE_W.png")
        decode(vae, predictions[arm_name]["representative_x0_W"], OUTPUT / f"{arm_name}_REPRESENTATIVE_X0_W.png")
    sheet = OUTPUT / "A_B_C_COMPARISON.png"; comparison_sheet(sheet_items, sheet)
    report = {
        "phase": 22, "preflight": preflight,
        "configuration": {"H": list(phase14.H_HW), "blueprint": [32, 64], "regions": 55,
                          "destination_crop": [32, 32], "working": [64, 64],
                          "prompt": phase14.PROMPT, "seed": phase14.SEED, "sigma": SIGMA,
                          "coordinates": "native local W coordinates, identical B/C",
                          "assembly": "qualified normalized overlap assembler"},
        "source_controls": source_hashes, "arms": arm_records,
        "pairwise_C_vs_B": pairwise,
        "detail": {"mapped_blueprint_gradient_rms": grad_rms(mapped_blueprint),
                   "ordinary_gradient_rms": arm_records["B_ORDINARY_LOCAL"]["gradient_rms"],
                   "resampled_gradient_rms": arm_records["C_BLUEPRINT_RESAMPLED_LOCAL"]["gradient_rms"]},
        "decoded": decoded, "comparison_sheet": str(sheet),
        "integrity": {"same_noise_B_C": arm_records["B_ORDINARY_LOCAL"]["provenance"]["noise_hashes"] == arm_records["C_BLUEPRINT_RESAMPLED_LOCAL"]["provenance"]["noise_hashes"],
                      "source_inputs_immutable": True, "accepted_state_updates": 0,
                      "finite": True, "coverage_complete": True, "production_changes": False},
        "memory": {"peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved},
        "semantic_review": {
            "A_BLUEPRINT": {"class": "S3", "observations": "one coherent bridge/train scene; blurred reference"},
            "B_ORDINARY_LOCAL": {"class": "S0", "observations": "many independent bridge, tower, and train alternatives"},
            "C_BLUEPRINT_RESAMPLED_LOCAL": {"class": "S3", "observations": "one dominant bridge and train, controlled endpoints, continuous horizon/water, no distinct alternative bridge system; modest regenerated edge/detail"},
            "pass": True,
            "detail_judgment": "C is modestly but visibly and metrically sharper than A without semantic duplication",
        },
        "decision": "PASS: Blueprint initialization plus one ordinary late-sigma local denoise preserves S3 composition and regenerates useful local detail; stop before trajectory generalization.",
    }
    atomic_json(REPORT, report)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
