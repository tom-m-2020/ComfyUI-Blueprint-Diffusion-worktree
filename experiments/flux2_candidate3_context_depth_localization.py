"""Phase 8e: local block-depth localization for terminal current-G context."""

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
import flux2_candidate2_one_eval_probe as candidate2
import flux2_candidate2_four_step_trajectory as candidate2_trajectory
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a
import flux2_candidate3_terminal_context as phase8d

import comfy.samplers

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_context_depth_localization_results"
REPORT = OUTPUT / "report.json"
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT
ALL_DOUBLE = frozenset(("double", index) for index in range(5))
ALL_SINGLE = frozenset(("single", index) for index in range(20))
VARIANTS = {
    "A_FULL_CONTEXT": ALL_DOUBLE | ALL_SINGLE,
    "B_DOUBLE_ONLY": ALL_DOUBLE,
    "C_SINGLE_ONLY": ALL_SINGLE,
    "D_EARLY_10": ALL_DOUBLE | frozenset(("single", index) for index in range(5)),
    "E_LATE_10": frozenset(("single", index) for index in range(10, 20)),
}


class SelectiveConsumptionProbe(phase8d.CPUOffloadedContextProbe):
    def __init__(self):
        super().__init__()
        self.enabled = frozenset()

    def add_global_context(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        if key not in self.enabled:
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
        return super().add_global_context(q, k, v, pe, attn_mask, extra_options)

    def restore_text_attention(self, attention_output, extra_options):
        key = self.block_key(extra_options)
        if key not in self.enabled:
            return attention_output
        return super().restore_text_attention(attention_output, extra_options)

    def assert_variant_complete(self, start_record):
        if self.pending_normal_attention:
            raise AssertionError(f"Unconsumed ordinary outputs: {tuple(self.pending_normal_attention)}")
        records = self.context_records[start_record:]
        consumed = {(item["block_type"], item["block_index"]) for item in records}
        if consumed != set(self.enabled):
            raise AssertionError(f"Enabled/consumed mismatch: {set(self.enabled) ^ consumed}")


class DepthLocalizationSampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None
        self.outputs = {}

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 8e requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h, sigmas[0])
        initial_hashes = {
            "H": hashlib.sha256(state.h.detach().float().cpu().numpy().tobytes()).hexdigest(),
            "G": hashlib.sha256(state.g.detach().float().cpu().numpy().tobytes()).hexdigest(),
        }
        accepted_hashes = []
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal], sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=extra_args.get("seed", 0),
            )
            accepted_hashes.append({
                "ordinal": ordinal,
                "H": hashlib.sha256(state.h.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "G": hashlib.sha256(state.g.detach().float().cpu().numpy().tobytes()).hexdigest(),
            })

        sigma = sigmas[3]
        terminal_h, terminal_g = state.h, state.g
        h_snapshot, g_snapshot = terminal_h.clone(), terminal_g.clone()
        regions = coordinator.planner.plan(HIGH_HW)
        base_options = extra_args["model_options"]
        probe = SelectiveConsumptionProbe()
        capture_options = candidate2_trajectory.merge_options(
            base_options,
            candidate2.model_options(
                "phase8e_current_G_capture",
                phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
                probe, "capture",
            ),
        )
        capture_start = torch.cuda.Event(enable_timing=True)
        capture_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats()
        capture_wall_start = time.perf_counter()
        capture_start.record()
        x0_g = coordinator.adapter.predict_global(
            guider=model, g=terminal_g, sigma=sigma, canvas=HIGH_HW,
            model_options=capture_options, seed=extra_args.get("seed", 0),
        )
        capture_end.record()
        torch.cuda.synchronize()
        capture_wall = time.perf_counter() - capture_wall_start
        capture_cuda = float(capture_start.elapsed_time(capture_end))
        capture_peak = {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }
        if len(probe.capture_records) != 25:
            raise AssertionError(f"Expected 25 captured blocks, got {len(probe.capture_records)}")
        g_star = terminal_g + (terminal_g - x0_g) / sigma * (sigmas[4] - sigma)
        captured_ids = {key: (id(item["k"]), id(item["v"])) for key, item in probe.global_kv.items()}
        crop_input_hashes = [
            hashlib.sha256(terminal_h[:, :, r.y:r.y2, r.x:r.x2].detach().float().cpu().numpy().tobytes()).hexdigest()
            for r in regions
        ]

        variants = {}
        full_predictions_cpu = None
        full_assembled_cpu = None
        for name, enabled in VARIANTS.items():
            probe.enabled = enabled
            record_start = len(probe.context_records)
            transfer_start = probe.transfer_bytes
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = int(torch.cuda.memory_allocated())
            baseline_reserved = int(torch.cuda.memory_reserved())
            wall_start = time.perf_counter()
            events = []
            predictions = []
            for region in regions:
                options = candidate2_trajectory.merge_options(
                    base_options,
                    candidate2.model_options(
                        f"phase8e_{name}_crop_{region.index}",
                        phase2.rope_for_crop(region), probe, "context",
                    ),
                )
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                prediction = coordinator.adapter.predict_region(
                    guider=model,
                    h_view=terminal_h[:, :, region.y:region.y2, region.x:region.x2],
                    sigma=sigma, canvas=HIGH_HW, region=region,
                    model_options=options, seed=extra_args.get("seed", 0),
                )
                end.record()
                events.append((start, end))
                predictions.append(prediction)
            torch.cuda.synchronize()
            local_cuda = sum(float(start.elapsed_time(end)) for start, end in events)
            local_wall = time.perf_counter() - wall_start
            probe.assert_variant_complete(record_start)
            assembled, coverage = coordinator.assembler.assemble(predictions, regions, HIGH_HW)
            h_star = terminal_h + (terminal_h - assembled) / sigma * (sigmas[4] - sigma)
            projection = coordinator.geometry.prolong(g_star - coordinator.geometry.restrict(h_star))
            predictions_cpu = [item.detach().float().cpu() for item in predictions]
            if name == "A_FULL_CONTEXT":
                full_predictions_cpu = predictions_cpu
                full_assembled_cpu = assembled.detach().float().cpu()
                crop_differences = [phase8d.difference(item, item) for item in predictions_cpu]
                assembled_difference = phase8d.difference(full_assembled_cpu, full_assembled_cpu)
            else:
                crop_differences = [
                    phase8d.difference(value, reference)
                    for value, reference in zip(predictions_cpu, full_predictions_cpu)
                ]
                assembled_difference = phase8d.difference(assembled.detach().float().cpu(), full_assembled_cpu)
            transfer_bytes = probe.transfer_bytes - transfer_start
            records = probe.context_records[record_start:]
            expected_records = len(enabled) * len(regions)
            if len(records) != expected_records:
                raise AssertionError((name, len(records), expected_records))
            if {key: (id(item["k"]), id(item["v"])) for key, item in probe.global_kv.items()} != captured_ids:
                raise AssertionError(f"Captured K/V identity changed in {name}")
            projection_rms = phase8d.summary(projection)["rms"]
            variants[name] = {
                "enabled_blocks": sorted([list(item) for item in enabled]),
                "enabled_block_count": len(enabled),
                "context_records": len(records),
                "augmented_q_by_k": [1536, 19968],
                "total_augmented_qk_elements": len(enabled) * len(regions) * 1536 * 19968,
                "consumed_KV_bytes": transfer_bytes,
                "terminal_local_cuda_ms": local_cuda,
                "terminal_local_wall_seconds": local_wall,
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "assembled_x0_H": phase8d.summary(assembled),
                "final_H": phase8d.summary(h_star),
                "overlap": phase8d.overlap_metrics(predictions, regions),
                "projection_needed": {
                    "rms": projection_rms,
                    "over_H_star": projection_rms / phase8d.summary(h_star)["rms"],
                    "low_frequency_rms": phase8d.summary(coordinator.geometry.restrict(projection))["rms"],
                },
                "D_final_vs_terminal_G_star": phase8d.difference(coordinator.geometry.restrict(h_star), g_star),
                "assembled_vs_full": assembled_difference,
                "per_crop_vs_full": crop_differences,
                "coverage": [float(coverage.min()), float(coverage.max())],
            }
            self.outputs[name] = h_star.detach().float().cpu()
            self.outputs[f"{name}_X0_H"] = assembled.detach().float().cpu()
            del predictions, predictions_cpu, assembled, h_star, projection
            torch.cuda.empty_cache()

        if not torch.equal(terminal_h, h_snapshot) or not torch.equal(terminal_g, g_snapshot):
            raise RuntimeError("Depth-localization calls mutated accepted H3/G3")
        self.result = {
            "sigmas": [float(value) for value in sigmas],
            "shared_controls": {
                "initial_hashes": initial_hashes,
                "accepted_nonterminal_hashes": accepted_hashes,
                "H3_hash": hashlib.sha256(terminal_h.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "G3_hash": hashlib.sha256(terminal_g.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "terminal_crop_input_hashes": crop_input_hashes,
                "crop_rectangles": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "source_capture_blocks": len(probe.capture_records),
                "source_KV_bytes": probe.capture_bytes,
                "source_capture_cuda_ms": capture_cuda,
                "source_capture_wall_seconds": capture_wall,
                "source_capture_peak": capture_peak,
                "same_source_KV_all_variants": True,
            },
            "variants": variants,
        }
        if callback is not None:
            callback(3, full_assembled_cpu, self.outputs["A_FULL_CONTEXT"], STEPS)
        return model_sampling.inverse_noise_scaling(sigmas[-1], self.outputs["A_FULL_CONTEXT"].to(noise.device, noise.dtype))


def decode(outputs):
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    for name, latent in outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {"path": str(path), "dimensions_wh": list(rgb.size),
                             "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}
    return decoded


def make_sheet(names, suffix, filename):
    width, height, label = 800, 400, 28
    canvas = Image.new("RGB", (width, (height + label) * len(names)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        key = name + suffix
        with Image.open(OUTPUT / f"{key}.png") as source:
            preview = ImageOps.contain(source.convert("RGB"), (width, height), Image.Resampling.LANCZOS)
        y = row * (height + label)
        canvas.paste(preview, ((width - preview.width) // 2, y + label))
        draw.text((8, y + 7), key, fill="black")
    canvas.save(OUTPUT / filename)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *HIGH_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(HIGH_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = DepthLocalizationSampler()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    started = time.perf_counter()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    data = sampler.result
    data["configuration"] = {
        "target_pixels_hw": [2048, 4096], "H": [128, 256], "G": [96, 192],
        "prompt": PROMPT, "seed": SEED, "cfg": 1.0,
        "production_changes": False,
        "D_early_range": "double 0-4 plus single 0-4",
        "E_late_range": "single 10-19",
    }
    data["performance"] = {
        "whole_experiment_wall_seconds": time.perf_counter() - started,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
    }
    data["decoded"] = decode(sampler.outputs)
    for name in VARIANTS:
        data["variants"][name]["decoded"] = data["decoded"][name]
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    names = tuple(VARIANTS)
    make_sheet(names, "", "CONTEXT_DEPTH_FINAL_COMPARISON.png")
    make_sheet(names, "_X0_H", "CONTEXT_DEPTH_TERMINAL_X0_COMPARISON.png")
    print(json.dumps({
        "report": str(REPORT),
        "variants": {
            name: {
                "blocks": value["enabled_block_count"],
                "overlap": value["overlap"]["aggregate_rms"],
                "projection_ratio": value["projection_needed"]["over_H_star"],
                "cuda_ms": value["terminal_local_cuda_ms"],
                "transfer_bytes": value["consumed_KV_bytes"],
                "hash": value["decoded"]["sha256_rgb"],
            }
            for name, value in data["variants"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
