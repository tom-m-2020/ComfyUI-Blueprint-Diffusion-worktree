"""Phase 14: fixed-4K global source at the 2048x4096 destination.

Research-only terminal discriminator.  Production and ComfyUI core are not
modified.  The two context variants use the same explicit block-major FLUX.2
executor; only their source tensor and full-canvas coordinates differ.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a
import flux2_candidate3_specialized_executor as phase8i
import flux2_candidate3_terminal_context as phase8d

import comfy.samplers
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_fixed4k_large_destination_results"
REPORT = OUTPUT / "report.json"
H_HW = (128, 256)
SOURCE_FIXED_HW = (32, 128)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def summary(value):
    value = value.detach().float()
    return {
        "rms": float(value.square().mean().sqrt()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "min": float(value.min()), "max": float(value.max()),
        "finite": bool(torch.isfinite(value).all()),
        "shape": list(value.shape), "sha256": tensor_hash(value),
    }


def difference(left, right):
    delta = left.detach().float() - right.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(left, right)),
    }


def restrict_4x2(value):
    if tuple(value.shape[-2:]) != H_HW:
        raise AssertionError(f"Expected H spatial shape {H_HW}, got {tuple(value.shape[-2:])}")
    b, c, _, _ = value.shape
    return value.reshape(b, c, 32, 4, 128, 2).mean(dim=(3, 5))


def provenance_summary():
    contributor_counts = torch.zeros(H_HW, dtype=torch.int16)
    records = []
    for row in range(32):
        for column in range(128):
            contributors = []
            for y in range(4 * row, 4 * row + 4):
                for x in range(2 * column, 2 * column + 2):
                    contributor_counts[y, x] += 1
                    contributors.append([y, x])
            records.append({
                "source_index": row * 128 + column,
                "source_grid_index": [row, column],
                "full_H_coordinate": [4.0 * row + 1.5, 2.0 * column + 0.5],
                "contributors": contributors,
            })
    return {
        "H_positions": int(contributor_counts.numel()),
        "source_positions": len(records),
        "contributors_per_source": 8,
        "minimum_contribution_count": int(contributor_counts.min()),
        "maximum_contribution_count": int(contributor_counts.max()),
        "omissions": int((contributor_counts == 0).sum()),
        "duplicates": int((contributor_counts > 1).sum()),
        "records": records,
    }


class Phase14Sampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None
        self.outputs = {}

    @staticmethod
    def _capture_source(coordinator, model, source, sigma, base, diffusion, probe, kind):
        if kind == "scaled_G":
            rope = phase2.rope_for_global(*H_HW, *coordinator.geometry.GLOBAL_HW)
            capture = phase8i.ForwardCapture(diffusion, run_native=False)
            with capture:
                coordinator.adapter.predict_global(
                    guider=model, g=source, sigma=sigma, canvas=H_HW,
                    model_options=phase8i.merged_options(
                        base, rope, probe, "capture", "phase14_scaled_G_source"
                    ), seed=SEED,
                )
        else:
            rope = {"scale_y": 4.0, "shift_y": 1.5,
                    "scale_x": 2.0, "shift_x": 0.5}
            capture = phase8i.ForwardCapture(diffusion, run_native=False)
            with capture:
                model(
                    source, sigma.expand(1),
                    model_options=phase8i.merged_options(
                        base, rope, probe, "capture", "phase14_fixed4k_source"
                    ), seed=SEED,
                )
        if capture.inputs is None:
            raise RuntimeError(f"{kind} source preparation did not reach native FLUX.")
        return capture.inputs, rope

    @staticmethod
    def _capture_local(model, working, sigma, base, diffusion, probe, region):
        capture = phase8i.ForwardCapture(diffusion, run_native=False)
        with capture:
            model(
                working, sigma.expand(1),
                model_options=phase8i.merged_options(
                    base, {}, probe, "context", f"phase14_W_crop_{region.index}"
                ), seed=SEED,
            )
        if capture.inputs is None:
            raise RuntimeError(f"Crop {region.index} preparation did not reach native FLUX.")
        return capture.inputs

    @staticmethod
    def _ordinary_local(model, working, sigma, base, regions, coordinator):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        wall_started = time.perf_counter()
        start.record()
        x0_w = [
            model(value, sigma.expand(1),
                  model_options=phase8i.merged_options(base, {}), seed=SEED)
            for value in working
        ]
        end.record()
        torch.cuda.synchronize()
        restricted = [phase9b.restrict2(value) for value in x0_w]
        assembled, coverage = coordinator.assembler.assemble(restricted, regions, H_HW)
        return assembled, restricted, {
            "source_cuda_ms": 0.0,
            "local_cuda_ms": float(start.elapsed_time(end)),
            "final_projection_cuda_ms": None,
            "specialized_cuda_ms": float(start.elapsed_time(end)),
            "terminal_wall_seconds": time.perf_counter() - wall_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "coverage": [float(coverage.min()), float(coverage.max())],
        }

    def _block_major(self, coordinator, model, source, working, sigma, base,
                     regions, diffusion, kind):
        executor = phase8i.ExplicitFluxExecutor(diffusion)
        probe = phase8i.OneBlockGPUContext()
        source_inputs, rope = self._capture_source(
            coordinator, model, source, sigma, base, diffusion, probe, kind
        )
        local_inputs = [
            self._capture_local(model, value, sigma, base, diffusion, probe, region)
            for value, region in zip(working, regions)
        ]
        source_state = executor.prepare(source_inputs)
        local_states = [executor.prepare(value) for value in local_inputs]
        source_input_img_summary = summary(source_inputs.img)
        source_after_img_in_summary = summary(source_state.img)
        del source_inputs, local_inputs
        gc.collect()

        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        source_cuda_ms = 0.0
        local_cuda_ms = 0.0
        source_kv_peak = 0
        barriers = []
        expected_key = None
        wall_started = time.perf_counter()
        try:
            for kind_name, block_count in (
                ("double", len(diffusion.double_blocks)),
                ("single", len(diffusion.single_blocks)),
            ):
                if kind_name == "single":
                    executor.enter_single(source_state)
                    for state in local_states:
                        executor.enter_single(state)
                for index in range(block_count):
                    source_start = torch.cuda.Event(enable_timing=True)
                    source_end = torch.cuda.Event(enable_timing=True)
                    source_start.record()
                    if kind_name == "double":
                        executor.double(source_state, index)
                    else:
                        executor.single(source_state, index)
                    source_end.record()
                    torch.cuda.synchronize()
                    source_cuda_ms += float(source_start.elapsed_time(source_end))
                    expected_key = (kind_name, index)
                    if probe.current_key != expected_key:
                        raise AssertionError((probe.current_key, expected_key))
                    source_kv_peak = max(source_kv_peak, probe.current_bytes)
                    representative = (
                        (kind_name == "double" and index == 0)
                        or (kind_name == "single" and index in (9, 19))
                    )
                    source_activation = None
                    if representative:
                        current = probe.global_kv[expected_key]
                        source_activation = {
                            "hidden": summary(source_state.img),
                            "K": summary(current["k"]),
                            "V": summary(current["v"]),
                        }

                    local_start = torch.cuda.Event(enable_timing=True)
                    local_end = torch.cuda.Event(enable_timing=True)
                    local_start.record()
                    for state in local_states:
                        if kind_name == "double":
                            executor.double(state, index)
                        else:
                            executor.single(state, index)
                    local_end.record()
                    torch.cuda.synchronize()
                    local_cuda_ms += float(local_start.elapsed_time(local_end))
                    barriers.append({
                        "block_type": kind_name, "block_index": index,
                        "source_kv_bytes": probe.current_bytes,
                        "source_hidden_bytes": phase8i.tensor_bytes(source_state.img),
                        "source_text_bytes": phase8i.tensor_bytes(source_state.txt),
                        "all_local_hidden_bytes": sum(
                            phase8i.tensor_bytes(state.img) + phase8i.tensor_bytes(state.txt)
                            for state in local_states
                        ),
                        "allocated_bytes": int(torch.cuda.memory_allocated()),
                        "reserved_bytes": int(torch.cuda.memory_reserved()),
                        "representative_activation": source_activation,
                    })
                    probe.release()

            final_start = torch.cuda.Event(enable_timing=True)
            final_end = torch.cuda.Event(enable_timing=True)
            final_start.record()
            x0_w = []
            for value, state in zip(working, local_states):
                raw = executor.final(state)
                x0_w.append(phase8i.raw_to_x0(raw, value, sigma, diffusion))
            final_end.record()
            torch.cuda.synchronize()
            final_ms = float(final_start.elapsed_time(final_end))
            restricted = [phase9b.restrict2(value) for value in x0_w]
            assembled, coverage = coordinator.assembler.assemble(restricted, regions, H_HW)
        finally:
            probe.release()
        return assembled, restricted, {
            "source_cuda_ms": source_cuda_ms,
            "local_cuda_ms": local_cuda_ms,
            "final_projection_cuda_ms": final_ms,
            "specialized_cuda_ms": source_cuda_ms + local_cuda_ms + final_ms,
            "terminal_wall_seconds": time.perf_counter() - wall_started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "source_kv_peak_bytes": source_kv_peak,
            "source_hidden_state_peak_bytes": max(item["source_hidden_bytes"] for item in barriers),
            "source_input_img": source_input_img_summary,
            "source_after_img_in": source_after_img_in_summary,
            "cpu_kv_cache_bytes": 0,
            "cpu_to_gpu_kv_transfer_bytes": 0,
            "source_final_projection_performed": False,
            "source_prediction_performed": False,
            "source_persistent_sampler_state": False if kind == "fixed_H" else True,
            "source_blocks": len(barriers),
            "context_consumptions": len(barriers) * len(local_states),
            "coordinate_options": rope,
            "barriers": barriers,
            "coverage": [float(coverage.min()), float(coverage.max())],
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 14 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1], model_options=extra_args["model_options"],
                seed=SEED,
            )
        sigma = sigmas[3]
        if float(sigmas[4]) != 0.0:
            raise AssertionError("Phase 14 must be a terminal sigma discriminator.")
        regions = phase9b.DestinationPlanner().plan(H_HW)
        if len(regions) != 55:
            raise AssertionError(f"Expected 55 regions, got {len(regions)}")
        h_hash, g_hash = tensor_hash(state.h), tensor_hash(state.g)
        working = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(view, sigma, 3, region)
            if float((phase9b.restrict2(value).float() - view.float()).abs().max()) > 1e-6:
                raise AssertionError(f"Working state coarse mismatch at region {region.index}")
            working.append(value)
        working_hashes = [tensor_hash(value) for value in working]
        fixed_source = restrict_4x2(state.h)
        h_stats, source_stats = summary(state.h), summary(fixed_source)
        state_stats = {
            "H": h_stats, "fixed_source": source_stats,
            "variance_ratio": source_stats["std"] ** 2 / (h_stats["std"] ** 2),
            "rms_ratio": source_stats["rms"] / h_stats["rms"],
            "normalization": "none",
        }
        base = extra_args["model_options"]
        diffusion = model.inner_model.diffusion_model
        variants = {}
        predictions = {}
        specs = (
            ("A_LOCAL_ONLY", None, None),
            ("B_DESTINATION_SCALED_CONTEXT", state.g, "scaled_G"),
            ("C_FIXED_DIRECT_4096", fixed_source, "fixed_H"),
        )
        for name, source, kind in specs:
            gc.collect()
            phase2.comfy.model_management.soft_empty_cache()
            if kind is None:
                assembled, crop_predictions, telemetry = self._ordinary_local(
                    model, working, sigma, base, regions, coordinator
                )
                source_tokens = 0
            else:
                assembled, crop_predictions, telemetry = self._block_major(
                    coordinator, model, source, working, sigma, base, regions,
                    diffusion, kind,
                )
                source_tokens = int(source.shape[-2] * source.shape[-1])
            overlap = phase8d.overlap_metrics(
                [item.detach().float().cpu() for item in crop_predictions], regions
            )
            variants[name] = {
                "source_shape": None if source is None else list(source.shape),
                "source_tokens": source_tokens,
                "attention_source_key_cardinality": source_tokens,
                "local_generated_tokens_per_region": 4096,
                "local_generated_token_work": 55 * 4096,
                "regions": 55, "overlap": overlap,
                "assembled": summary(assembled), **telemetry,
            }
            predictions[name] = assembled.detach().float().cpu()
            if tensor_hash(state.h) != h_hash or tensor_hash(state.g) != g_hash:
                raise RuntimeError(f"{name} mutated accepted H/G.")
            if [tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError(f"{name} mutated terminal working states.")
            self.outputs[name] = assembled.detach().float().cpu()

        variants["C_FIXED_DIRECT_4096"]["assembled_vs_B"] = difference(
            predictions["C_FIXED_DIRECT_4096"], predictions["B_DESTINATION_SCALED_CONTEXT"]
        )
        variants["C_FIXED_DIRECT_4096"]["assembled_vs_A"] = difference(
            predictions["C_FIXED_DIRECT_4096"], predictions["A_LOCAL_ONLY"]
        )
        self.result = {
            "configuration": {
                "output_pixels_hw": [2048, 4096], "H": list(H_HW),
                "accepted_G": list(state.g.shape[-2:]),
                "fixed_source": list(SOURCE_FIXED_HW), "seed": SEED,
                "prompt": PROMPT, "terminal_sigma": float(sigma),
                "regions": 55, "destination_crop": [32, 32], "stride": 24,
                "working_W": [64, 64], "steps": 4,
            },
            "sigmas": [float(value) for value in sigmas],
            "accepted_state": {"H_hash": h_hash, "G_hash": g_hash},
            "working_hashes": working_hashes,
            "fixed_source_hash": tensor_hash(fixed_source),
            "fixed_source_statistics": state_stats,
            "provenance": provenance_summary(),
            "variants": variants,
            "integrity": {
                "same_accepted_state": True, "same_working_states": True,
                "all_results_finite": all(item["assembled"]["finite"] for item in variants.values()),
                "complete_coverage": all(item["coverage"][0] > 0 for item in variants.values()),
                "no_production_changes": True, "terminal_state_updates": 0,
                "B_and_C_same_executor": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def make_sheet(paths, destination):
    images = [(name, Image.open(path).convert("RGB")) for name, path in paths]
    width = max(image.width for _, image in images)
    thumb_height = 512
    panels = []
    for name, image in images:
        copy = image.copy()
        copy.thumbnail((width, thumb_height))
        panel = Image.new("RGB", (width, copy.height + 44), "white")
        panel.paste(copy, ((width - copy.width) // 2, 44))
        ImageDraw.Draw(panel).text((12, 12), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (width, sum(panel.height for panel in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *H_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(H_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = Phase14Sampler()
    perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    paths = []
    decoded = {}
    for name, latent in sampler.outputs.items():
        pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
        paths.append((name, path))
    sheet = OUTPUT / "A_B_C_comparison.png"
    make_sheet(paths, sheet)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet)
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet), "decoded": decoded}, indent=2))


if __name__ == "__main__":
    main()
