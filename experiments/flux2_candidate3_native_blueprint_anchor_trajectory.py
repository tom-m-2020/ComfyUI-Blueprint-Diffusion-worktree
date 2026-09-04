"""Phase 20c: four-step trajectory with the exact Phase-20b prediction anchor."""

from __future__ import annotations

import gc
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
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_native_blueprint_prediction_anchor as phase20b
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule
from blueprint_diffusion.state import BlueprintState

OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_blueprint_anchor_trajectory_results"
REPORT = OUTPUT / "report.json"
INTERVALS = OUTPUT / "intervals"


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def interval_paths(ordinal):
    return INTERVALS / f"interval_{ordinal}.json", INTERVALS / f"interval_{ordinal}.pt"


def load_interval(ordinal, config_hash):
    metadata_path, tensor_path = interval_paths(ordinal)
    if not metadata_path.is_file() or not tensor_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if not metadata.get("complete") or metadata.get("configuration_hash") != config_hash:
        raise RuntimeError(f"Phase 20c interval {ordinal} metadata mismatch.")
    if tensors.get("configuration_hash") != config_hash:
        raise RuntimeError(f"Phase 20c interval {ordinal} tensor mismatch.")
    return metadata["record"], tensors


def save_interval(ordinal, config_hash, record, tensors):
    metadata_path, tensor_path = interval_paths(ordinal)
    atomic_torch(tensor_path, {"configuration_hash": config_hash, **tensors})
    atomic_json(metadata_path, {"complete": True, "configuration_hash": config_hash, "record": record})


def tensor_detail_rms(value):
    coarse = phase9b.restrict2(value)
    detail = value - phase9b.prolong2(coarse)
    return float(detail.float().square().mean().sqrt())


class Phase20cSampler(phase14.Phase14Sampler):
    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 20c requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55 or len(sigmas) != 5:
            raise AssertionError("Phase 20c setup mismatch.")
        configuration = {
            "phase": "20c",
            "algorithm": "Phase-20 normalized W plus exact Phase-20b prediction anchor",
            "H": list(phase14.H_HW), "initial_G": list(state.g.shape[-2:]),
            "blueprint": list(phase20.BLUEPRINT_HW), "regions": len(regions),
            "seed": phase14.SEED, "sigmas": [float(x) for x in sigmas],
            "H0_hash": phase14.tensor_hash(state.h), "G0_hash": phase14.tensor_hash(state.g),
            "D_B": "2x2 mean", "U_B": "2x nearest", "anchor_strength": 1.0,
            "blueprint_coordinates": {"y": [0, 31], "x": [0, 63], "frame": "native"},
        }
        config_hash = stable_hash(configuration)
        base = extra_args["model_options"]
        records = []
        outputs = []

        for ordinal in range(4):
            loaded = load_interval(ordinal, config_hash)
            if loaded is not None:
                record, tensors = loaded
                if phase14.tensor_hash(state.h) != record["accepted_input_H_hash"] or phase14.tensor_hash(state.g) != record["accepted_input_G_hash"]:
                    raise RuntimeError(f"Phase 20c resume lineage mismatch at interval {ordinal}.")
                state = BlueprintState(
                    tensors["accepted_G"].to(state.g.device, state.g.dtype),
                    tensors["accepted_H"].to(state.h.device, state.h.dtype),
                    float(sigmas[ordinal + 1]), ordinal + 1, f"phase20c:{ordinal}",
                )
                records.append(record)
                outputs.append({key: tensors[key] for key in ("blueprint_x0", "assembled_x0_H", "accepted_H")})
                print(f"phase20c interval {ordinal} resume-skip {ordinal + 1}/4", flush=True)
                continue

            print(f"phase20c interval {ordinal} start {ordinal + 1}/4", flush=True)
            accepted_h_hash = phase14.tensor_hash(state.h)
            accepted_g_hash = phase14.tensor_hash(state.g)
            sigma = sigmas[ordinal]
            sigma_next = sigmas[ordinal + 1]
            terminal = float(sigma_next) == 0.0
            gc.collect()
            phase2.comfy.model_management.soft_empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            arm_wall = time.perf_counter()

            x0_g = None
            global_cuda_ms = 0.0
            if not terminal:
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                x0_g = coordinator.adapter.predict_global(
                    guider=model, g=state.g, sigma=sigma, canvas=phase14.H_HW,
                    model_options=base, seed=phase14.SEED,
                )
                end.record()
                torch.cuda.synchronize()
                global_cuda_ms = float(start.elapsed_time(end))

            blueprint_state, _, _ = phase20.make_blueprint_state(state.h, sigma)
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            blueprint_x0 = model(
                blueprint_state, sigma.expand(1),
                model_options=phase20.phase8i_options(base), seed=phase14.SEED,
            )
            end.record()
            torch.cuda.synchronize()
            blueprint_cuda_ms = float(start.elapsed_time(end))
            mapped = F.interpolate(
                blueprint_x0.float(), size=phase14.H_HW, mode="bilinear", align_corners=False
            ).to(blueprint_x0.dtype)

            working = []
            blueprint_crops = []
            for region in regions:
                crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
                value, _, _ = phase20.make_blueprint_working(crop, sigma, region, crop.device, crop.dtype)
                working.append(value)
                blueprint_crops.append(crop)
            working_hashes = [phase14.tensor_hash(value) for value in working]
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            x0_w = [
                model(value, sigma.expand(1), model_options=phase20.phase8i_options(base), seed=phase14.SEED)
                for value in working
            ]
            end.record()
            torch.cuda.synchronize()
            local_cuda_ms = float(start.elapsed_time(end))

            corrected = []
            corrections = []
            before_errors = []
            after_errors = []
            detail_before = []
            detail_after = []
            correction_ratios = []
            for prediction, crop in zip(x0_w, blueprint_crops):
                anchored, correction = phase20b.correct_prediction(prediction, crop)
                before_errors.append(float((phase20b.d_blueprint(prediction).float() - crop.float()).square().mean().sqrt()))
                after_errors.append(float((phase20b.d_blueprint(anchored).float() - crop.float()).abs().max()))
                detail_before.append(tensor_detail_rms(prediction))
                detail_after.append(tensor_detail_rms(anchored))
                correction_ratios.append(float(correction.float().square().mean().sqrt() / prediction.float().square().mean().sqrt()))
                corrected.append(phase20b.d_blueprint(anchored))
                corrections.append(phase20b.d_blueprint(correction))
            if max(after_errors) > 1.0e-5:
                raise RuntimeError(f"Phase 20c anchor invariant failed at {ordinal}: {max(after_errors)}")
            x0_h, coverage = coordinator.assembler.assemble(corrected, regions, phase14.H_HW)
            correction_canvas, _ = coordinator.assembler.assemble(corrections, regions, phase14.H_HW)
            overlap = phase8d.overlap_metrics([x.detach().float().cpu() for x in corrected], regions)
            dt = sigma_next - sigma
            h_star = state.h + (state.h - x0_h) / sigma * dt
            if terminal:
                acceptance = coordinator.policy.accept_terminal(
                    retained_g=state.g, h_star=h_star, sigma_next=float(sigma_next)
                )
                invariant = None
                g_star = None
            else:
                g_star = state.g + (state.g - x0_g) / sigma * dt
                acceptance = coordinator.policy.accept(
                    g_star=g_star, h_star=h_star, sigma_next=float(sigma_next), geometry=coordinator.geometry
                )
                invariant = float((coordinator.geometry.restrict(acceptance.h).float() - acceptance.g.float()).abs().max())
                if invariant > coordinator.geometry.TOLERANCE:
                    raise RuntimeError(f"Phase 20c D(H)=G failed at {ordinal}: {invariant}")
            if phase14.tensor_hash(state.h) != accepted_h_hash or phase14.tensor_hash(state.g) != accepted_g_hash:
                raise RuntimeError(f"Phase 20c mutated accepted input at interval {ordinal}.")
            if [phase14.tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError(f"Phase 20c mutated W inputs at interval {ordinal}.")
            if not torch.isfinite(acceptance.h).all() or not torch.isfinite(acceptance.g).all():
                raise RuntimeError(f"Phase 20c produced nonfinite state at interval {ordinal}.")

            record = {
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "terminal": terminal, "accepted_input_H_hash": accepted_h_hash,
                "accepted_input_G_hash": accepted_g_hash,
                "blueprint_x0": phase14.summary(blueprint_x0),
                "assembled_x0_H": phase14.summary(x0_h), "H_star": phase14.summary(h_star),
                "accepted_H": phase14.summary(acceptance.h), "accepted_G": phase14.summary(acceptance.g),
                "global_forward_performed": not terminal, "blueprint_forward_performed": True,
                "local_forward_count": 55, "terminal_release": terminal,
                "anchor": {
                    "coarse_before_rms_mean": sum(before_errors) / len(before_errors),
                    "coarse_after_max_abs": max(after_errors),
                    "correction_rms": float(correction_canvas.float().square().mean().sqrt()),
                    "correction_over_local_x0_rms_mean": sum(correction_ratios) / len(correction_ratios),
                    "detail_rms_before_mean": sum(detail_before) / len(detail_before),
                    "detail_rms_after_mean": sum(detail_after) / len(detail_after),
                },
                "overlap_rms": overlap["aggregate_rms"], "invariant_max_abs": invariant,
                "coverage": [float(coverage.min()), float(coverage.max())],
                "global_cuda_ms": global_cuda_ms, "blueprint_cuda_ms": blueprint_cuda_ms,
                "local_cuda_ms": local_cuda_ms, "wall_seconds": time.perf_counter() - arm_wall,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "working_hashes": working_hashes,
            }
            tensors = {
                "blueprint_x0": blueprint_x0.detach().float().cpu(),
                "assembled_x0_H": x0_h.detach().float().cpu(),
                "accepted_H": acceptance.h.detach().float().cpu(),
                "accepted_G": acceptance.g.detach().float().cpu(),
                "correction_canvas": correction_canvas.detach().float().cpu(),
            }
            save_interval(ordinal, config_hash, record, tensors)
            state = BlueprintState(acceptance.g, acceptance.h, float(sigma_next), ordinal + 1, f"phase20c:{ordinal}")
            records.append(record)
            outputs.append({key: tensors[key] for key in ("blueprint_x0", "assembled_x0_H", "accepted_H")})
            print(f"phase20c interval {ordinal} complete {ordinal + 1}/4 wall={record['wall_seconds']:.2f}s", flush=True)

        self.result = {
            "configuration": configuration, "configuration_hash": config_hash,
            "intervals": records,
            "integrity": {
                "accepted_intervals": 4, "atomic_acceptances": 4,
                "terminal_release_only_last": [x["terminal_release"] for x in records] == [False, False, False, True],
                "global_forward_count": sum(x["global_forward_performed"] for x in records),
                "blueprint_forward_count": sum(x["blueprint_forward_performed"] for x in records),
                "local_forward_count": sum(x["local_forward_count"] for x in records),
                "finite": all(x["accepted_H"]["finite"] and x["accepted_G"]["finite"] for x in records),
                "production_changes": False,
            },
            "final_H": phase14.summary(state.h),
        }
        self.outputs = outputs
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def save_decode(vae, latent, path):
    pixels = vae.decode(latent).cpu()
    phase2.save_pixels(pixels, path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return {"path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}


def make_sheet(paths, destination):
    panels = []
    for label, path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((2048, 512))
        panel = Image.new("RGB", (2048, image.height + 38), "white")
        panel.paste(image, ((2048 - image.width) // 2, 38))
        ImageDraw.Draw(panel).text((10, 10), label, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (2048, sum(x.height for x in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight = {"estimated_minutes": 12, "stop_threshold_minutes": 30, "proceed": True,
                 "work": "4 Blueprint + 3 Candidate-3 global + 220 local forwards, then selected tiled decodes"}
    atomic_json(OUTPUT / "preflight_cost.json", preflight)
    print(json.dumps({"preflight": preflight}), flush=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED))
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone(); sigmas[0] = 1.0
    sampler = Phase20cSampler(); perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative, torch.zeros_like(noise),
            callback=lambda *args: None, disable_pbar=True, seed=phase14.SEED,
        )
    del model
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    sheet_items = []
    for ordinal, output in enumerate(sampler.outputs):
        if ordinal in (0, 2, 3):
            bp_path = OUTPUT / f"interval_{ordinal}_BLUEPRINT_X0.png"
            h_path = OUTPUT / f"interval_{ordinal}_ACCEPTED_H.png"
            decoded[f"interval_{ordinal}_blueprint"] = save_decode(vae, output["blueprint_x0"], bp_path)
            decoded[f"interval_{ordinal}_accepted_H"] = save_decode(vae, output["accepted_H"], h_path)
            sheet_items.extend(((f"interval {ordinal} Blueprint x0", bp_path),
                                (f"interval {ordinal} accepted H", h_path)))
    sheet_path = OUTPUT / "TRAJECTORY_COMPARISON.png"
    make_sheet(sheet_items, sheet_path)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet_path)
    sampler.result["preflight"] = preflight
    atomic_json(REPORT, sampler.result)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
