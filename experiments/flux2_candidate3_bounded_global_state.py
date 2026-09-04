"""Phase 19: bounded single-scale versus hierarchical whole-canvas state."""

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
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

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

OUTPUT = ROOT / "experiments" / "flux2_candidate3_bounded_global_state_results"
ARMS = OUTPUT / "arms"
REPORT = OUTPUT / "report.json"
ARM_SPECS = (
    ("A_SIMPLE_32X128", "simple"),
    ("B_SINGLE_SCALE_64X64", "single"),
    ("C_HIERARCHICAL_512_PLUS_3584", "hierarchical"),
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def paths(name):
    return ARMS / f"{name}.json", ARMS / f"{name}.pt"


def load_arm(name, fingerprint):
    metadata_path, tensor_path = paths(name)
    if not metadata_path.is_file() or not tensor_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 19 artifact configuration mismatch: {name}")
    if not metadata.get("complete"):
        return None
    tensors = torch.load(tensor_path, map_location="cpu", weights_only=True)
    if tensors.get("configuration_hash") != fingerprint:
        raise RuntimeError(f"Phase 19 tensor artifact mismatch: {name}")
    return metadata["record"], tensors


def save_arm(name, fingerprint, record, tensors):
    ARMS.mkdir(parents=True, exist_ok=True)
    metadata_path, tensor_path = paths(name)
    atomic_torch(tensor_path, {"configuration_hash": fingerprint, **tensors})
    atomic_json(metadata_path, {
        "complete": True, "configuration_hash": fingerprint,
        "tensor_artifact": str(tensor_path), "record": record,
    })


def area_exact(value, block_y, block_x):
    return value.unfold(2, block_y, block_y).unfold(3, block_x, block_x).mean(dim=(-1, -2))


def source_specs(h):
    return {
        "simple": [
            (area_exact(h, 4, 2), {"scale_y": 4.0, "shift_y": 1.5,
                                   "scale_x": 2.0, "shift_x": 0.5}, "simple_32x128"),
        ],
        "single": [
            (area_exact(h, 2, 4), {"scale_y": 2.0, "shift_y": 0.5,
                                   "scale_x": 4.0, "shift_x": 1.5}, "single_64x64"),
        ],
        "hierarchical": [
            (area_exact(h, 8, 8), {"scale_y": 8.0, "shift_y": 3.5,
                                   "scale_x": 8.0, "shift_x": 3.5}, "coarse_16x32"),
            (F.interpolate(h.float(), size=(32, 112), mode="area").to(h.dtype),
             {"scale_y": 4.0, "shift_y": 1.5,
              "scale_x": 256.0 / 112.0, "shift_x": (256.0 / 112.0 - 1.0) / 2.0},
             "medium_32x112"),
        ],
    }


def combine_source_states(states):
    first = states[0]
    text_tokens = phase17.TEXT_TOKENS
    for state in states[1:]:
        if not torch.equal(first.txt, state.txt) or not torch.equal(first.vec_orig, state.vec_orig):
            raise AssertionError("Hierarchical source conditioning mismatch.")
    pe = torch.cat(
        [first.pe[:, :, :text_tokens]] + [state.pe[:, :, text_tokens:] for state in states], dim=2
    )
    return phase8i.ExplicitState(
        img=torch.cat([state.img for state in states], dim=1), txt=first.txt,
        vec_orig=first.vec_orig, double_vec=first.double_vec,
        single_vec=first.single_vec, pe=pe, options=first.options,
        attn_mask=first.attn_mask,
    )


class Phase19Sampler(phase14.Phase14Sampler):
    @staticmethod
    def _capture_level(model, value, sigma, base, diffusion, probe, rope, label):
        capture = phase8i.ForwardCapture(diffusion, run_native=False)
        with capture:
            model(value, sigma.expand(1), model_options=phase8i.merged_options(
                base, rope, probe, "capture", f"phase19_{label}"
            ), seed=phase14.SEED)
        if capture.inputs is None:
            raise RuntimeError(f"{label} did not reach native FLUX.")
        return capture.inputs

    def _block_major_levels(self, coordinator, model, levels, working, sigma, base,
                            regions, diffusion):
        executor = phase8i.ExplicitFluxExecutor(diffusion)
        probe = phase8i.OneBlockGPUContext()
        level_inputs = [
            self._capture_level(model, value, sigma, base, diffusion, probe, rope, label)
            for value, rope, label in levels
        ]
        prepared_levels = [executor.prepare(value) for value in level_inputs]
        source_state = combine_source_states(prepared_levels)
        local_inputs = [
            self._capture_local(model, value, sigma, base, diffusion, probe, region)
            for value, region in zip(working, regions)
        ]
        local_states = [executor.prepare(value) for value in local_inputs]
        del level_inputs, prepared_levels, local_inputs
        gc.collect()

        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        source_ms = local_ms = 0.0
        kv_peak = 0
        block_records = []
        wall_started = time.perf_counter()
        try:
            for kind, count in (("double", 5), ("single", 20)):
                if kind == "single":
                    executor.enter_single(source_state)
                    for state in local_states:
                        executor.enter_single(state)
                for index in range(count):
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    (executor.double if kind == "double" else executor.single)(source_state, index)
                    b.record(); torch.cuda.synchronize(); source_ms += float(a.elapsed_time(b))
                    expected = (kind, index)
                    if probe.current_key != expected:
                        raise AssertionError((probe.current_key, expected))
                    kv_peak = max(kv_peak, probe.current_bytes)
                    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                    a.record()
                    for state in local_states:
                        (executor.double if kind == "double" else executor.single)(state, index)
                    b.record(); torch.cuda.synchronize(); local_ms += float(a.elapsed_time(b))
                    block_records.append({
                        "block_type": kind, "block_index": index,
                        "source_tokens": 4096, "context_consumers": 55,
                        "source_kv_bytes": probe.current_bytes,
                    })
                    probe.release()
            a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record()
            x0_w = [phase8i.raw_to_x0(executor.final(state), value, sigma, diffusion)
                    for value, state in zip(working, local_states)]
            b.record(); torch.cuda.synchronize(); final_ms = float(a.elapsed_time(b))
            restricted = [phase9b.restrict2(value) for value in x0_w]
            assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
        finally:
            probe.release()
        return assembled, restricted, {
            "source_cuda_ms": source_ms, "local_cuda_ms": local_ms,
            "final_projection_cuda_ms": final_ms,
            "terminal_wall_seconds": time.perf_counter() - wall_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "source_kv_peak_bytes": kv_peak, "cpu_kv_cache_bytes": 0,
            "cpu_to_gpu_kv_transfer_bytes": 0, "source_blocks": 25,
            "context_consumptions": 1375, "coverage": [float(coverage.min()), float(coverage.max())],
            "block_records": block_records,
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 19 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator(); state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal], sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=phase14.SEED,
            )
        sigma = sigmas[3]
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55 or float(sigmas[4]) != 0.0:
            raise AssertionError("Phase 19 geometry/schedule mismatch.")
        h_hash, g_hash = phase14.tensor_hash(state.h), phase14.tensor_hash(state.g)
        working = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(view, sigma, 3, region)
            if float((phase9b.restrict2(value).float() - view.float()).abs().max()) > 1e-6:
                raise AssertionError(region.index)
            working.append(value)
        working_hashes = [phase14.tensor_hash(value) for value in working]
        levels_by_kind = source_specs(state.h)
        construction = {
            kind: [{"shape": list(value.shape), "tokens": value.shape[-2] * value.shape[-1],
                    "hash": phase14.tensor_hash(value), "rope": rope, "label": label,
                    "summary": phase14.summary(value)} for value, rope, label in levels]
            for kind, levels in levels_by_kind.items()
        }
        configuration = {
            "phase": "19", "H": list(phase14.H_HW), "G": list(state.g.shape[-2:]),
            "source_budget": 4096, "W": [64, 64], "regions": 55,
            "seed": phase14.SEED, "sigma": float(sigma),
            "sigmas": [float(value) for value in sigmas], "H_hash": h_hash, "G_hash": g_hash,
            "working_hashes": working_hashes, "construction": construction,
        }
        fingerprint = stable_hash(configuration)
        variants, outputs = {}, {}
        base, diffusion = extra_args["model_options"], model.inner_model.diffusion_model
        control = None
        for progress, (name, kind) in enumerate(ARM_SPECS, start=1):
            loaded = load_arm(name, fingerprint)
            if loaded is not None:
                record, tensors = loaded; assembled = tensors["assembled"]
                print(f"phase19 {name} resume-skip {progress}/3", flush=True)
            else:
                started_at, arm_started = utc_now(), time.perf_counter()
                print(f"phase19 {name} start {progress}/3 {started_at}", flush=True)
                gc.collect(); phase2.comfy.model_management.soft_empty_cache()
                assembled, predictions, telemetry = self._block_major_levels(
                    coordinator, model, levels_by_kind[kind], working, sigma, base,
                    regions, diffusion,
                )
                overlap = phase8d.overlap_metrics(
                    [value.detach().float().cpu() for value in predictions], regions
                )
                record = {
                    "name": name, "kind": kind, "started_at": started_at,
                    "completed_at": utc_now(), "elapsed_seconds": time.perf_counter() - arm_started,
                    "levels": construction[kind], "source_tokens": 4096,
                    "shared_source_trajectories": 1, "W_specific_source_trajectories": 0,
                    "attention_dimensions": {"text_plus_local_queries": 4608,
                                             "text_plus_local_plus_source_keys": 8704},
                    "overlap": overlap, "assembled": phase14.summary(assembled), **telemetry,
                }
                if control is not None:
                    record["assembled_vs_A"] = phase17.tensor_difference(assembled.detach().cpu(), control)
                save_arm(name, fingerprint, record, {
                    "assembled": assembled.detach().float().cpu(),
                    "representative_prediction": predictions[phase17.REPRESENTATIVE_REGION].detach().float().cpu(),
                })
                print(
                    f"phase19 {name} complete {progress}/3 {record['completed_at']} "
                    f"wall={record['terminal_wall_seconds']:.2f}s source={record['source_cuda_ms']/1000:.2f}s "
                    f"local={record['local_cuda_ms']/1000:.2f}s",
                    flush=True,
                )
            if progress == 1:
                control = assembled.detach().float().cpu()
            elif "assembled_vs_A" not in record:
                record["assembled_vs_A"] = phase17.tensor_difference(assembled, control)
            variants[name] = record; outputs[name] = assembled.detach().float().cpu()
        if phase14.tensor_hash(state.h) != h_hash or phase14.tensor_hash(state.g) != g_hash:
            raise RuntimeError("Phase 19 mutated accepted H/G.")
        if [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Phase 19 mutated W.")
        self.outputs = outputs
        self.result = {
            "configuration": configuration, "configuration_hash": fingerprint,
            "variants": variants, "provenance": phase14.provenance_summary(),
            "integrity": {
                "arms": 3, "source_tokens_each": 4096, "shared_source_each": True,
                "accepted_state_immutable": True, "working_states_immutable": True,
                "terminal_state_updates": 0,
                "complete_coverage": all(value["coverage"][0] > 0 for value in variants.values()),
                "all_finite": all(value["assembled"]["finite"] for value in variants.values()),
                "no_production_changes": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def save_sheet(items, destination):
    panels = []
    for name, path in items:
        image = Image.open(path).convert("RGB"); image.thumbnail((2048, 512))
        panel = Image.new("RGB", (2048, image.height + 38), "white")
        panel.paste(image, ((2048 - image.width) // 2, 38))
        ImageDraw.Draw(panel).text((10, 10), name, fill="black"); panels.append(panel)
    sheet = Image.new("RGB", (2048, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    estimated = {
        "reference_phase14_fixed4k_wall_seconds": 105.234,
        "planned_new_model_arms": 3,
        "estimated_model_seconds": 3 * 110,
        "estimated_total_minutes_with_setup_decode": 12,
        "cost_stop_threshold_minutes": 30,
        "proceed": True,
    }
    atomic_json(OUTPUT / "preflight_cost.json", estimated)
    print(json.dumps({"phase19_preflight": estimated}), flush=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize("")); del clip
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED))
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone(); sigmas[0] = 1.0
    sampler = Phase19Sampler(); perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative, torch.zeros_like(noise),
            callback=lambda *args: None, disable_pbar=True, seed=phase14.SEED,
        )
    del model
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded, sheet_items = {}, []
    for name, latent in sampler.outputs.items():
        pixels = vae.decode(latent).cpu(); path = OUTPUT / f"{name}.png"; phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {"path": str(path), "dimensions_wh": list(rgb.size),
                             "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}
        sheet_items.append((name, path))
    sheet = OUTPUT / "A_B_C_COMPARISON.png"; save_sheet(sheet_items, sheet)
    sampler.result["decoded"] = decoded; sampler.result["comparison_sheet"] = str(sheet)
    sampler.result["preflight_cost"] = estimated
    atomic_json(REPORT, sampler.result)
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet)}, indent=2))


if __name__ == "__main__":
    main()
