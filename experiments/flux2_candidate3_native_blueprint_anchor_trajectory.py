"""Phase 20c: minimal bounded-Blueprint prediction-anchor trajectory."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_blueprint_prediction_anchor as phase20b
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule

OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_blueprint_anchor_trajectory_results"
REPORT = OUTPUT / "report.json"
TENSORS = OUTPUT / "trajectory.pt"


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def euler(value, x0, sigma, sigma_next):
    return value + (sigma_next - sigma) * (value - x0) / sigma


def stats(value):
    x = value.detach().float()
    return {"rms": float(x.square().mean().sqrt()), "finite": bool(torch.isfinite(x).all()),
            "shape": list(x.shape), "hash": phase14.tensor_hash(x)}


class Phase20cSampler(phase14.Phase14Sampler):
    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 20c requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        blueprint0, blueprint_coarse, blueprint_noise = phase20.make_blueprint_state(h0, sigmas[0])
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55:
            raise AssertionError("Phase 20c requires the qualified 55-region plan.")
        accepted_h = {"B_UNANCHORED": h0.clone(), "C_ANCHORED": h0.clone()}
        accepted_blueprint = blueprint0.clone()
        initial_hashes = {"H": phase14.tensor_hash(h0), "B": phase14.tensor_hash(blueprint0)}
        base_options = extra_args["model_options"]
        steps = []
        saved = {"blueprint_x0": [], "assembled_B": [], "assembled_C": [],
                 "accepted_B": [], "accepted_C": [], "accepted_blueprint": []}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        total_started = time.perf_counter()

        for ordinal in range(len(sigmas) - 1):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            print(f"phase20c interval {ordinal + 1}/{len(sigmas) - 1} start", flush=True)
            bp_before_hash = phase14.tensor_hash(accepted_blueprint)
            h_before_hashes = {name: phase14.tensor_hash(value) for name, value in accepted_h.items()}

            torch.cuda.synchronize()
            source_start, source_end = torch.cuda.Event(True), torch.cuda.Event(True)
            source_wall_started = time.perf_counter()
            source_start.record()
            x0_blueprint = model(
                accepted_blueprint, sigma.expand(1),
                model_options=phase20.phase8i_options(base_options), seed=phase14.SEED,
            )
            source_end.record()
            torch.cuda.synchronize()
            source_cuda_ms = float(source_start.elapsed_time(source_end))
            source_wall = time.perf_counter() - source_wall_started
            mapped_blueprint = torch.nn.functional.interpolate(
                x0_blueprint.float(), size=phase14.H_HW, mode="bilinear", align_corners=False
            ).to(x0_blueprint.dtype)
            blueprint_star = euler(accepted_blueprint, x0_blueprint, sigma, sigma_next)

            proposals = {}
            step_variants = {}
            for name in ("B_UNANCHORED", "C_ANCHORED"):
                h_current = accepted_h[name]
                working = []
                for region in regions:
                    view = h_current[:, :, region.y:region.y2, region.x:region.x2]
                    value = phase9c.make_working(view, sigma, ordinal, region)
                    error = float((phase9b.restrict2(value).float() - view.float()).abs().max())
                    if error > 1.0e-6:
                        raise RuntimeError(f"Phase 20c W coarse invariant failed: {error}")
                    working.append(value)
                working_hashes = [phase14.tensor_hash(x) for x in working]
                torch.cuda.synchronize()
                local_start, local_end = torch.cuda.Event(True), torch.cuda.Event(True)
                local_wall_started = time.perf_counter()
                local_start.record()
                x0_w = [
                    model(value, sigma.expand(1), model_options=phase20.phase8i_options(base_options), seed=phase14.SEED)
                    for value in working
                ]
                local_end.record()
                torch.cuda.synchronize()
                local_cuda_ms = float(local_start.elapsed_time(local_end))
                local_wall = time.perf_counter() - local_wall_started
                before = [phase9b.restrict2(x) for x in x0_w]
                before_overlap = phase8d.overlap_metrics([x.detach().float().cpu() for x in before], regions)
                coarse_before = []
                coarse_after = []
                ratios = []
                corrected_w = []
                if name == "C_ANCHORED":
                    for value, region in zip(x0_w, regions):
                        bp_crop = mapped_blueprint[:, :, region.y:region.y2, region.x:region.x2]
                        corrected, correction = phase20b.correct_prediction(value, bp_crop)
                        coarse_before.append(float((phase20b.d_blueprint(value).float() - bp_crop.float()).square().mean().sqrt()))
                        coarse_after.append(float((phase20b.d_blueprint(corrected).float() - bp_crop.float()).abs().max()))
                        ratios.append(float(correction.float().square().mean().sqrt() / value.float().square().mean().sqrt()))
                        corrected_w.append(corrected)
                    predictions = [phase9b.restrict2(x) for x in corrected_w]
                else:
                    predictions = before
                after_overlap = phase8d.overlap_metrics([x.detach().float().cpu() for x in predictions], regions)
                assembled, coverage = BlueprintCoordinator().assembler.assemble(predictions, regions, phase14.H_HW)
                h_star = euler(h_current, assembled, sigma, sigma_next)
                proposals[name] = h_star
                if [phase14.tensor_hash(x) for x in working] != working_hashes:
                    raise RuntimeError("Phase 20c local W input mutated.")
                step_variants[name] = {
                    "assembled": stats(assembled),
                    "local_cuda_ms": local_cuda_ms,
                    "local_wall_seconds": local_wall,
                    "overlap_before": before_overlap["aggregate_rms"],
                    "overlap_after": after_overlap["aggregate_rms"],
                    "coarse_before_rms_mean": sum(coarse_before) / len(coarse_before) if coarse_before else None,
                    "coarse_after_max_abs": max(coarse_after) if coarse_after else None,
                    "correction_local_x0_ratio_mean": sum(ratios) / len(ratios) if ratios else 0.0,
                    "correction_local_x0_ratio_max": max(ratios) if ratios else 0.0,
                    "coverage": [float(coverage.min()), float(coverage.max())],
                }
                saved["assembled_B" if name == "B_UNANCHORED" else "assembled_C"].append(assembled.detach().float().cpu())

            if phase14.tensor_hash(accepted_blueprint) != bp_before_hash:
                raise RuntimeError("Phase 20c mutated accepted Blueprint during evaluation.")
            if any(phase14.tensor_hash(accepted_h[name]) != value for name, value in h_before_hashes.items()):
                raise RuntimeError("Phase 20c mutated accepted H during evaluation.")
            if not torch.isfinite(blueprint_star).all() or any(not torch.isfinite(x).all() for x in proposals.values()):
                raise RuntimeError("Phase 20c nonfinite proposal.")
            if step_variants["C_ANCHORED"]["coarse_after_max_abs"] > 1.0e-5:
                raise RuntimeError("Phase 20c anchor invariant exceeded tolerance.")

            # One atomic publication boundary after source and both local arms validate.
            accepted_blueprint = blueprint_star
            accepted_h = proposals
            saved["blueprint_x0"].append(x0_blueprint.detach().float().cpu())
            saved["accepted_blueprint"].append(accepted_blueprint.detach().float().cpu())
            saved["accepted_B"].append(accepted_h["B_UNANCHORED"].detach().float().cpu())
            saved["accepted_C"].append(accepted_h["C_ANCHORED"].detach().float().cpu())
            steps.append({
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "source_cuda_ms": source_cuda_ms, "source_wall_seconds": source_wall,
                "blueprint_x0": stats(x0_blueprint), "accepted_blueprint": stats(accepted_blueprint),
                "variants": step_variants, "atomic_acceptance": True,
                "same_sigma_source_and_locals": True,
            })
            print(f"phase20c interval {ordinal + 1} complete source={source_wall:.2f}s "
                  f"B={step_variants['B_UNANCHORED']['local_wall_seconds']:.2f}s "
                  f"C={step_variants['C_ANCHORED']['local_wall_seconds']:.2f}s", flush=True)

        torch.cuda.synchronize()
        self.report = {
            "configuration": {
                "phase": "20c", "H": list(phase14.H_HW), "blueprint": list(phase20.BLUEPRINT_HW),
                "blueprint_tokens": phase20.BLUEPRINT_TOKENS, "blueprint_coordinates": "ordinary native 0..31 x 0..63",
                "regions": 55, "destination_region": [32, 32], "W": [64, 64], "stride": 24,
                "seed": phase14.SEED, "sigmas": [float(x) for x in sigmas],
                "anchor": "x0 + U_B(mapped_blueprint_crop - D_B(x0)); D_B=mean2, U_B=nearest2",
                "anchor_applied_intervals": list(range(len(sigmas) - 1)),
            },
            "initial": {"H": stats(h0), "blueprint": stats(blueprint0),
                        "blueprint_coarse": stats(blueprint_coarse), "blueprint_noise": stats(blueprint_noise)},
            "steps": steps,
            "final": {"B": stats(accepted_h["B_UNANCHORED"]), "C": stats(accepted_h["C_ANCHORED"]),
                      "C_vs_B": phase17.tensor_difference(accepted_h["C_ANCHORED"], accepted_h["B_UNANCHORED"])},
            "timing": {"sampling_wall_seconds": time.perf_counter() - total_started,
                       "source_cuda_ms_total": sum(x["source_cuda_ms"] for x in steps),
                       "B_local_cuda_ms_total": sum(x["variants"]["B_UNANCHORED"]["local_cuda_ms"] for x in steps),
                       "C_local_cuda_ms_total": sum(x["variants"]["C_ANCHORED"]["local_cuda_ms"] for x in steps)},
            "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
            "integrity": {"initial_H_hash": initial_hashes["H"], "initial_blueprint_hash": initial_hashes["B"],
                          "accepted_updates": len(sigmas) - 1, "all_atomic": True,
                          "same_sigma": True, "finite": True, "production_changes": False},
        }
        self.saved = saved
        return sampling.inverse_noise_scaling(sigmas[-1], accepted_h["C_ANCHORED"])


def save_image(vae, latent, path):
    pixels = vae.decode(latent).cpu()
    phase2.save_pixels(pixels, path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def make_sheet(paths, destination):
    rows = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((2048, 512))
        row = Image.new("RGB", (2048, image.height + 34), "white")
        ImageDraw.Draw(row).text((8, 8), label, fill="black")
        row.paste(image, ((2048 - image.width) // 2, 34))
        rows.append(row)
    sheet = Image.new("RGB", (2048, sum(x.height for x in rows)), "white")
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = {"estimated_minutes": 18, "stop_threshold_minutes": 30, "proceed": True,
                 "work": "4 Blueprint + 440 W forwards; 12 diagnostic decodes"}
    atomic_json(OUTPUT / "preflight_cost.json", preflight)
    print(json.dumps({"preflight": preflight}), flush=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED))
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone()
    sigmas[0], sigmas[-1] = 1.0, 0.0
    sampler = Phase20cSampler()
    perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative, torch.zeros_like(noise),
            callback=lambda *args: None, disable_pbar=True, seed=phase14.SEED,
        )
    del model
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    atomic_torch(TENSORS, sampler.saved)

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    sheet_items = []
    for ordinal in range(4):
        for key, label in (("blueprint_x0", "Blueprint x0"), ("assembled_B", "B unanchored x0"),
                           ("assembled_C", "C anchored x0")):
            path = OUTPUT / f"step_{ordinal}_{key}.png"
            decoded[f"step_{ordinal}_{key}"] = save_image(vae, sampler.saved[key][ordinal], path)
            sheet_items.append((f"step {ordinal} {label}", path))
    sheet = OUTPUT / "TRAJECTORY_COMPARISON.png"
    make_sheet(sheet_items, sheet)
    sampler.report["decoded"] = decoded
    sampler.report["comparison_sheet"] = str(sheet)
    sampler.report["preflight"] = preflight
    atomic_json(REPORT, sampler.report)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
