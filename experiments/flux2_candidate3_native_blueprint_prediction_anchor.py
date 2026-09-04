"""Phase 20b: exact terminal Blueprint prediction-space coarse anchor."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_specialized_executor as phase8i
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule

OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_blueprint_prediction_anchor_results"
ARM_JSON = OUTPUT / "C_BLUEPRINT_PREDICTION_ANCHOR.json"
ARM_PT = OUTPUT / "C_BLUEPRINT_PREDICTION_ANCHOR.pt"
REPORT = OUTPUT / "report.json"
PHASE20_OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_blueprint_local_state_results"
PHASE20_REPORT = PHASE20_OUTPUT / "report.json"
PHASE20_B_PT = PHASE20_OUTPUT / "arms" / "B_BLUEPRINT_STATE_INITIALIZATION.pt"


def now():
    return datetime.now(timezone.utc).isoformat()


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


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def d_blueprint(value):
    """Restrict one 64x64 W prediction to its 32x32 destination/Blueprint crop."""
    return phase9b.restrict2(value)


def u_blueprint(value):
    """Exact nearest-neighbor right inverse from a 32x32 discrepancy to W space."""
    return phase9b.prolong2(value)


def correct_prediction(x0_w, blueprint_crop):
    correction = u_blueprint(blueprint_crop - d_blueprint(x0_w))
    corrected = x0_w + correction
    return corrected, correction


class Phase20bSampler(phase14.Phase14Sampler):
    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 20b requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        phase20_report = json.loads(PHASE20_REPORT.read_text(encoding="utf-8"))
        phase20_tensors = torch.load(PHASE20_B_PT, map_location="cpu", weights_only=True)
        blueprint_x0_cpu = phase20_tensors["blueprint_x0"].float()
        baseline_b_cpu = phase20_tensors["assembled"].float()

        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal], sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=phase14.SEED,
            )
        sigma = sigmas[3]
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55 or float(sigmas[4]) != 0.0:
            raise AssertionError("Phase 20b setup mismatch.")
        h_hash = phase14.tensor_hash(state.h)
        g_hash = phase14.tensor_hash(state.g)
        expected = phase20_report["configuration"]
        if h_hash != expected["H_hash"] or g_hash != expected["G_hash"]:
            raise RuntimeError("Phase 20b accepted state does not reproduce Phase 20.")
        if float(sigma) != expected["sigma"]:
            raise RuntimeError("Phase 20b sigma does not reproduce Phase 20.")

        blueprint_x0 = blueprint_x0_cpu.to(state.h.device, state.h.dtype)
        mapped = torch.nn.functional.interpolate(
            blueprint_x0.float(), size=phase14.H_HW, mode="bilinear", align_corners=False
        ).to(state.h.dtype)
        working = []
        for region in regions:
            blueprint_crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            value, _, _ = phase20.make_blueprint_working(
                blueprint_crop, sigma, region, blueprint_crop.device, blueprint_crop.dtype
            )
            working.append(value)
        working_hashes = [phase14.tensor_hash(value) for value in working]
        if working_hashes != expected["control_W_hashes"] and working_hashes != phase20_report["variants"]["B_BLUEPRINT_STATE_INITIALIZATION"]["invariants"]["working_hashes"]:
            raise RuntimeError("Phase 20b W construction differs from Phase 20 B.")

        configuration = {
            "phase": "20b",
            "phase20_configuration_hash": phase20_report["configuration_hash"],
            "H_hash": h_hash,
            "G_hash": g_hash,
            "sigma": float(sigma),
            "regions": len(regions),
            "W_hashes": working_hashes,
            "D_B": "nonoverlapping 2x2 arithmetic mean, W 64x64 -> destination/Blueprint crop 32x32",
            "U_B": "2x nearest neighbor, destination/Blueprint crop 32x32 -> W 64x64",
            "right_inverse": "D_B(U_B(z)) = z exactly up to floating-point arithmetic",
        }
        config_hash = fingerprint(configuration)
        if ARM_JSON.is_file() and ARM_PT.is_file():
            metadata = json.loads(ARM_JSON.read_text(encoding="utf-8"))
            if metadata.get("configuration_hash") != config_hash or not metadata.get("complete"):
                raise RuntimeError("Phase 20b persisted artifact mismatch.")
            tensors = torch.load(ARM_PT, map_location="cpu", weights_only=True)
            if tensors.get("configuration_hash") != config_hash:
                raise RuntimeError("Phase 20b tensor artifact mismatch.")
            print("phase20b C resume-skip 1/1", flush=True)
            self.result = metadata["record"]
            self.tensors = tensors
            return sampling.inverse_noise_scaling(sigmas[-1], state.h)

        print(f"phase20b C start 1/1 {now()}", flush=True)
        gc.collect()
        phase2.comfy.model_management.soft_empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(True)
        end = torch.cuda.Event(True)
        wall_started = time.perf_counter()
        start.record()
        x0_w = [
            model(value, sigma.expand(1), model_options=phase20.phase8i_options(extra_args["model_options"]), seed=phase14.SEED)
            for value in working
        ]
        end.record()
        torch.cuda.synchronize()
        local_cuda_ms = float(start.elapsed_time(end))
        local_wall_seconds = time.perf_counter() - wall_started

        corrected_w = []
        corrections = []
        original_restricted = []
        corrected_restricted = []
        coarse_before = []
        coarse_after = []
        correction_ratios = []
        for value, region in zip(x0_w, regions):
            blueprint_crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            corrected, correction = correct_prediction(value, blueprint_crop)
            before = d_blueprint(value)
            after = d_blueprint(corrected)
            coarse_before.append(float((before.float() - blueprint_crop.float()).square().mean().sqrt()))
            coarse_after.append(float((after.float() - blueprint_crop.float()).abs().max()))
            correction_ratios.append(float(correction.float().square().mean().sqrt() / value.float().square().mean().sqrt()))
            corrected_w.append(corrected)
            corrections.append(correction)
            original_restricted.append(before)
            corrected_restricted.append(after)

        baseline_recomputed, coverage_b = coordinator.assembler.assemble(original_restricted, regions, phase14.H_HW)
        assembled_c, coverage_c = coordinator.assembler.assemble(corrected_restricted, regions, phase14.H_HW)
        overlap_b = phase8d.overlap_metrics([x.detach().float().cpu() for x in original_restricted], regions)
        overlap_c = phase8d.overlap_metrics([x.detach().float().cpu() for x in corrected_restricted], regions)
        correction_canvas, _ = coordinator.assembler.assemble(
            [d_blueprint(x) for x in corrections], regions, phase14.H_HW
        )
        baseline_validation = phase17.tensor_difference(baseline_recomputed.float().cpu(), baseline_b_cpu)
        if baseline_validation["max_abs"] != 0.0:
            raise RuntimeError(f"Phase 20b ordinary B did not reproduce saved B: {baseline_validation}")
        max_after = max(coarse_after)
        tolerance = 1.0e-5
        if max_after > tolerance:
            raise RuntimeError(f"Phase 20b coarse anchor error {max_after} exceeds {tolerance}.")
        if phase14.tensor_hash(state.h) != h_hash or phase14.tensor_hash(state.g) != g_hash:
            raise RuntimeError("Phase 20b mutated accepted H/G.")
        if [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Phase 20b mutated W inputs.")
        if not torch.isfinite(assembled_c).all():
            raise RuntimeError("Phase 20b produced nonfinite output.")

        record = {
            "configuration": configuration,
            "configuration_hash": config_hash,
            "operators": {
                "D_B": "D_B(x)[c,y,x] = (1/4) sum_{dy,dx in {0,1}} x[c,2y+dy,2x+dx]",
                "U_B": "U_B(z)[c,2y+dy,2x+dx] = z[c,y,x] for dy,dx in {0,1}",
                "correction": "x0_corrected = x0_local + U_B(blueprint_crop - D_B(x0_local))",
                "identity": "D_B(x0_corrected) = blueprint_crop because D_B(U_B(z)) = z",
                "declared_tolerance": tolerance,
            },
            "baseline_B_reproduction": baseline_validation,
            "coarse_consistency": {
                "before_rms_mean": sum(coarse_before) / len(coarse_before),
                "before_rms_max": max(coarse_before),
                "after_max_abs": max_after,
            },
            "correction": {
                "rms": float(correction_canvas.float().square().mean().sqrt()),
                "relative_to_original_local_x0_rms_mean": sum(correction_ratios) / len(correction_ratios),
                "relative_to_original_local_x0_rms_max": max(correction_ratios),
            },
            "overlap": {"B": overlap_b, "C": overlap_c},
            "assembled": {
                "B": phase14.summary(baseline_recomputed),
                "C": phase14.summary(assembled_c),
                "C_vs_B": phase17.tensor_difference(assembled_c, baseline_recomputed),
            },
            "timing": {"local_cuda_ms": local_cuda_ms, "local_wall_seconds": local_wall_seconds,
                       "post_forward_correction": "tensor arithmetic measured inside total arm wall only"},
            "memory": {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                       "peak_reserved_bytes": int(torch.cuda.max_memory_reserved())},
            "integrity": {
                "accepted_H_immutable": True,
                "accepted_G_immutable": True,
                "W_inputs_immutable": True,
                "terminal_state_updates": 0,
                "model_inputs_identical_to_Phase20_B": True,
                "coverage_B": [float(coverage_b.min()), float(coverage_b.max())],
                "coverage_C": [float(coverage_c.min()), float(coverage_c.max())],
                "finite": True,
                "production_changes": False,
            },
            "completed_at": now(),
        }
        tensors = {
            "configuration_hash": config_hash,
            "baseline_B": baseline_recomputed.detach().float().cpu(),
            "assembled_C": assembled_c.detach().float().cpu(),
            "blueprint_x0": blueprint_x0_cpu,
            "correction_canvas": correction_canvas.detach().float().cpu(),
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        atomic_torch(ARM_PT, tensors)
        atomic_json(ARM_JSON, {"complete": True, "configuration_hash": config_hash, "record": record})
        self.result = record
        self.tensors = tensors
        print(f"phase20b C complete 1/1 {record['completed_at']} local={local_wall_seconds:.2f}s", flush=True)
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def save_decoded(vae, latent, path):
    pixels = vae.decode(latent).cpu()
    phase2.save_pixels(pixels, path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def make_sheet(items, destination):
    panels = []
    for name, path in items:
        image = Image.open(path).convert("RGB")
        image.thumbnail((2048, 512))
        panel = Image.new("RGB", (2048, image.height + 38), "white")
        panel.paste(image, ((2048 - image.width) // 2, 38))
        ImageDraw.Draw(panel).text((10, 10), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (2048, sum(x.height for x in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = {"estimated_minutes": 6, "stop_threshold_minutes": 30, "proceed": True,
                 "reason": "one required 55-forward B reproduction plus three decodes"}
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
    sigmas[0] = 1.0
    sampler = Phase20bSampler()
    import flux2_candidate3_performance_characterization as perf
    perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative, torch.zeros_like(noise),
            callback=lambda *args: None, disable_pbar=True, seed=phase14.SEED,
        )
    del model
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    b_path = OUTPUT / "B_BLUEPRINT_STATE_INITIALIZATION.png"
    bp_path = OUTPUT / "BLUEPRINT_X0.png"
    c_path = OUTPUT / "C_BLUEPRINT_PREDICTION_ANCHOR.png"
    decoded = {
        "B": save_decoded(vae, sampler.tensors["baseline_B"], b_path),
        "BLUEPRINT_X0": save_decoded(vae, sampler.tensors["blueprint_x0"], bp_path),
        "C": save_decoded(vae, sampler.tensors["assembled_C"], c_path),
    }
    sheet_path = OUTPUT / "B_BLUEPRINT_C_COMPARISON.png"
    make_sheet((("B ordinary local assembly", b_path), ("Blueprint x0", bp_path),
                ("C prediction anchor", c_path)), sheet_path)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet_path)
    sampler.result["preflight"] = preflight
    atomic_json(REPORT, sampler.result)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
