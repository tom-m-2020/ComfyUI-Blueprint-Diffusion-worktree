"""Phase 20: native-coordinate Blueprint prediction to local working state."""

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
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT)); sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_joint_depth_localization as phase18
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_specialized_executor as phase8i
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule

OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_blueprint_local_state_results"
ARMS = OUTPUT / "arms"
REPORT = OUTPUT / "report.json"
BLUEPRINT_HW = (32, 64)
BLUEPRINT_TOKENS = math.prod(BLUEPRINT_HW)


def now(): return datetime.now(timezone.utc).isoformat()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact_paths(name): return ARMS / f"{name}.json", ARMS / f"{name}.pt"


def load_arm(name, fingerprint):
    metadata_path, tensor_path = artifact_paths(name)
    if not metadata_path.is_file() or not tensor_path.is_file(): return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 20 artifact configuration mismatch: {name}")
    if not metadata.get("complete"): return None
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if tensors.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 20 tensor artifact mismatch: {name}")
    return metadata["record"], tensors


def save_arm(name, fingerprint, record, tensors):
    ARMS.mkdir(parents=True, exist_ok=True)
    metadata_path, tensor_path = artifact_paths(name)
    atomic_torch(tensor_path, {"configuration_hash": fingerprint, **tensors})
    atomic_json(metadata_path, {"complete": True, "configuration_hash": fingerprint,
                                "tensor_artifact": str(tensor_path), "record": record})


def area_4x4(value):
    return value.unfold(2, 4, 4).unfold(3, 4, 4).mean(dim=(-1, -2))


def make_blueprint_state(h, sigma):
    coarse = area_4x4(h)
    generator = torch.Generator(device="cpu").manual_seed(phase14.SEED + 20_000_003)
    noise = torch.randn(tuple(coarse.shape), generator=generator).to(coarse.device, coarse.dtype)
    added = sigma * math.sqrt(15.0 / 16.0) * noise
    return coarse + added, coarse, added


def region_noise(shape, region):
    generator = torch.Generator(device="cpu").manual_seed(
        phase14.SEED + 20_100_003 + 1_009 * region.index
    )
    return torch.randn(shape, generator=generator)


def make_blueprint_working(blueprint_crop, sigma, region, device, dtype):
    anchor = phase9b.prolong2(blueprint_crop)
    noise = region_noise(tuple(anchor.shape), region).to(device=device, dtype=dtype)
    value = (1.0 - sigma) * anchor + sigma * noise
    expected = (1.0 - sigma) * blueprint_crop + sigma * phase9b.restrict2(noise)
    null_detail = sigma * (noise - phase9b.prolong2(phase9b.restrict2(noise)))
    return value, expected, null_detail


class Phase20Sampler(phase14.Phase14Sampler):
    def _ordinary_predictions(self, model, working, sigma, base, regions, coordinator):
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        wall_started = time.perf_counter(); start.record()
        x0_w = [model(value, sigma.expand(1), model_options=phase8i_options(base), seed=phase14.SEED)
                for value in working]
        end.record(); torch.cuda.synchronize()
        restricted = [phase9b.restrict2(value) for value in x0_w]
        assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
        return assembled, restricted, {
            "local_cuda_ms": float(start.elapsed_time(end)),
            "local_wall_seconds": time.perf_counter() - wall_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "coverage": [float(coverage.min()), float(coverage.max())],
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 20 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator(); state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal], sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=phase14.SEED,
            )
        sigma = sigmas[3]; regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55 or float(sigmas[4]) != 0.0: raise AssertionError("Phase 20 setup mismatch.")
        h_hash, g_hash = phase14.tensor_hash(state.h), phase14.tensor_hash(state.g)
        control_working = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            control_working.append(phase9c.make_working(view, sigma, 3, region))
        control_hashes = [phase14.tensor_hash(value) for value in control_working]
        blueprint_state, blueprint_coarse, added_noise = make_blueprint_state(state.h, sigma)
        configuration = {
            "phase": 20, "H": list(phase14.H_HW), "G": list(state.g.shape[-2:]),
            "blueprint": list(BLUEPRINT_HW), "blueprint_tokens": BLUEPRINT_TOKENS,
            "blueprint_coordinates": {"model_frame": "ordinary_native", "y": [0, 31], "x": [0, 63]},
            "destination_mapping": "bilinear align_corners=False through normalized [0,1] canvas",
            "seed": phase14.SEED, "sigma": float(sigma), "sigmas": [float(x) for x in sigmas],
            "H_hash": h_hash, "G_hash": g_hash, "control_W_hashes": control_hashes,
            "blueprint_state_hash": phase14.tensor_hash(blueprint_state),
        }
        fingerprint = stable_hash(configuration); base = extra_args["model_options"]
        variants, outputs, blueprint_x0_cpu = {}, {}, None

        for progress, name in enumerate(("A_NORMALIZED_W_CONTROL", "B_BLUEPRINT_STATE_INITIALIZATION"), 1):
            loaded = load_arm(name, fingerprint)
            if loaded is not None:
                record, tensors = loaded; assembled = tensors["assembled"]
                if "blueprint_x0" in tensors: blueprint_x0_cpu = tensors["blueprint_x0"]
                print(f"phase20 {name} resume-skip {progress}/2", flush=True)
            else:
                started_at, arm_started = now(), time.perf_counter()
                print(f"phase20 {name} start {progress}/2 {started_at}", flush=True)
                gc.collect(); phase2.comfy.model_management.soft_empty_cache()
                source_ms = source_wall = 0.0
                invariants = {}
                if progress == 1:
                    working = control_working
                else:
                    torch.cuda.synchronize(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    source_started = time.perf_counter(); a.record()
                    blueprint_x0 = model(
                        blueprint_state, sigma.expand(1), model_options=phase8i_options(base),
                        seed=phase14.SEED,
                    )
                    b.record(); torch.cuda.synchronize()
                    source_ms, source_wall = float(a.elapsed_time(b)), time.perf_counter() - source_started
                    blueprint_x0_cpu = blueprint_x0.detach().float().cpu()
                    mapped = F.interpolate(blueprint_x0.float(), size=phase14.H_HW,
                                           mode="bilinear", align_corners=False).to(blueprint_x0.dtype)
                    working, before_errors, null_errors, null_rms = [], [], [], []
                    for region in regions:
                        crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
                        value, expected, null_detail = make_blueprint_working(
                            crop, sigma, region, crop.device, crop.dtype
                        )
                        before_errors.append(float((phase9b.restrict2(value).float() - expected.float()).abs().max()))
                        null_errors.append(float(phase9b.restrict2(null_detail).float().abs().max()))
                        null_rms.append(float(null_detail.float().square().mean().sqrt()))
                        working.append(value)
                    invariants = {
                        "D_W_equation_max_abs": max(before_errors),
                        "null_detail_D_max_abs": max(null_errors),
                        "null_detail_rms_mean": sum(null_rms) / len(null_rms),
                        "working_hashes": [phase14.tensor_hash(value) for value in working],
                        "blueprint_mapped": phase14.summary(mapped),
                    }
                working_hashes = [phase14.tensor_hash(value) for value in working]
                assembled, predictions, telemetry = self._ordinary_predictions(
                    model, working, sigma, base, regions, coordinator
                )
                after_errors = []
                if progress == 2:
                    for prediction, region in zip(predictions, regions):
                        crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
                        after_errors.append(float((prediction.float() - crop.float()).square().mean().sqrt()))
                    invariants["coarse_prediction_vs_blueprint_crop_rms_mean"] = sum(after_errors) / len(after_errors)
                overlap = phase8d.overlap_metrics([value.detach().float().cpu() for value in predictions], regions)
                record = {
                    "name": name, "started_at": started_at, "completed_at": now(),
                    "elapsed_seconds": time.perf_counter() - arm_started,
                    "blueprint_tokens": 0 if progress == 1 else BLUEPRINT_TOKENS,
                    "blueprint_forward_count": 0 if progress == 1 else 1,
                    "W_forward_count": 55, "source_cuda_ms": source_ms,
                    "source_wall_seconds": source_wall, "overlap": overlap,
                    "assembled": phase14.summary(assembled), "invariants": invariants, **telemetry,
                }
                tensors = {"assembled": assembled.detach().float().cpu(),
                           "representative_prediction": predictions[phase17.REPRESENTATIVE_REGION].detach().float().cpu()}
                if progress == 2:
                    tensors["blueprint_x0"] = blueprint_x0_cpu
                save_arm(name, fingerprint, record, tensors)
                print(f"phase20 {name} complete {progress}/2 {record['completed_at']} "
                      f"source={source_wall:.2f}s local={record['local_wall_seconds']:.2f}s", flush=True)
            variants[name] = record; outputs[name] = assembled.detach().float().cpu()
        variants["B_BLUEPRINT_STATE_INITIALIZATION"]["assembled_vs_A"] = phase17.tensor_difference(
            outputs["B_BLUEPRINT_STATE_INITIALIZATION"], outputs["A_NORMALIZED_W_CONTROL"]
        )
        if phase14.tensor_hash(state.h) != h_hash or phase14.tensor_hash(state.g) != g_hash:
            raise RuntimeError("Phase 20 mutated H/G.")
        if [phase14.tensor_hash(value) for value in control_working] != control_hashes:
            raise RuntimeError("Phase 20 mutated control W.")
        self.outputs, self.blueprint_x0 = outputs, blueprint_x0_cpu
        self.result = {
            "configuration": configuration, "configuration_hash": fingerprint,
            "blueprint_state": {
                "construction": "4x4 area mean accepted H + sigma*sqrt(15/16)*independent N",
                "state": phase14.summary(blueprint_state), "coarse": phase14.summary(blueprint_coarse),
                "added_noise": phase14.summary(added_noise),
                "variance_restoration": "analytic; 1/16 retained average-noise variance plus 15/16 added",
            },
            "working_equation": "W=(1-sigma)*nearest2(mapped_blueprint_x0_crop)+sigma*N",
            "variants": variants, "provenance": phase14.provenance_summary(),
            "integrity": {"accepted_state_immutable": True, "control_W_immutable": True,
                          "terminal_state_updates": 0, "coverage_complete": all(v["coverage"][0] > 0 for v in variants.values()),
                          "all_finite": all(v["assembled"]["finite"] for v in variants.values()),
                          "no_production_changes": True},
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def phase8i_options(base):
    return phase8i.merged_options(base, {})


def make_sheet(items, destination):
    panels = []
    for name, path in items:
        image = Image.open(path).convert("RGB"); image.thumbnail((2048, 512))
        panel = Image.new("RGB", (2048, image.height + 38), "white")
        panel.paste(image, ((2048-image.width)//2, 38)); ImageDraw.Draw(panel).text((10, 10), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (2048, sum(p.height for p in panels)), "white"); y = 0
    for panel in panels: sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    estimate = {"model_setup_minutes": 2, "blueprint_and_local_model_minutes": 2,
                "tiled_decode_minutes": 4, "estimated_total_minutes": 8,
                "stop_threshold_minutes": 30, "proceed": True}
    atomic_json(OUTPUT / "preflight_cost.json", estimate); print(json.dumps({"preflight": estimate}), flush=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT)); negative = clip.encode_from_tokens_scheduled(clip.tokenize("")); del clip
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1,128,*phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED))
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone(); sigmas[0] = 1.0
    sampler = Phase20Sampler(); perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None, disable_pbar=True, seed=phase14.SEED)
    del model; phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    items, decoded = [], {}
    bp_pixels = vae.decode(sampler.blueprint_x0).cpu(); bp_path = OUTPUT / "BLUEPRINT_X0.png"; phase2.save_pixels(bp_pixels, bp_path)
    with Image.open(bp_path) as image:
        rgb=image.convert("RGB"); decoded["BLUEPRINT_X0"]={"path":str(bp_path),"dimensions_wh":list(rgb.size),"sha256_rgb":hashlib.sha256(rgb.tobytes()).hexdigest()}
    for name, latent in sampler.outputs.items():
        pixels=vae.decode(latent).cpu(); path=OUTPUT/f"{name}.png"; phase2.save_pixels(pixels,path)
        with Image.open(path) as image:
            rgb=image.convert("RGB"); decoded[name]={"path":str(path),"dimensions_wh":list(rgb.size),"sha256_rgb":hashlib.sha256(rgb.tobytes()).hexdigest()}
        items.append((name,path))
    sheet=OUTPUT/"A_B_COMPARISON.png"; make_sheet(items,sheet)
    sampler.result["decoded"]=decoded; sampler.result["comparison_sheet"]=str(sheet); sampler.result["preflight_cost"]=estimate
    atomic_json(REPORT,sampler.result); print(json.dumps({"report":str(REPORT),"sheet":str(sheet)},indent=2))


if __name__ == "__main__": main()
