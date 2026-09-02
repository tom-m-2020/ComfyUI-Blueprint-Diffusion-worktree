"""Phase 8d: terminal local/global context discriminator for Candidate-3."""

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

import comfy.samplers
from comfy.ldm.flux import math as flux_math

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_terminal_context_results"
REPORT = OUTPUT / "report.json"
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT


class CPUOffloadedContextProbe(candidate2.OneEvaluationContextProbe):
    """Same Candidate-2 K/V representation, with experiment-only CPU storage."""

    def __init__(self):
        super().__init__()
        self.capture_bytes = 0
        self.transfer_bytes = 0

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        positioned_k = flux_math.apply_rope1(k, pe)
        cached_k = positioned_k[:, :, text_tokens:].detach().cpu()
        cached_v = v[:, :, text_tokens:].detach().cpu()
        self.global_kv[key] = {
            "k": cached_k,
            "v": cached_v,
            "tokens": sequence_end - text_tokens,
        }
        self.capture_bytes += cached_k.numel() * cached_k.element_size()
        self.capture_bytes += cached_v.numel() * cached_v.element_size()
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens,
            "global_generated_tokens": sequence_end - text_tokens,
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
            "pe_shape": list(pe.shape), "storage_device": "cpu",
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def add_global_context(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        if key not in self.global_kv:
            raise KeyError(f"No same-evaluation global K/V captured for {key}.")
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        global_entry = self.global_kv[key]
        self.pending_normal_attention[key] = flux_math.attention(
            q, k, v, pe=pe, mask=attn_mask, transformer_options=extra_options
        )
        positioned_q = flux_math.apply_rope1(q, pe)
        positioned_k = flux_math.apply_rope1(k, pe)
        global_k = global_entry["k"].to(device=k.device)
        global_v = global_entry["v"].to(device=v.device)
        self.transfer_bytes += global_k.numel() * global_k.element_size()
        self.transfer_bytes += global_v.numel() * global_v.element_size()
        augmented_k = torch.cat((positioned_k, global_k), dim=2)
        augmented_v = torch.cat((v, global_v), dim=2)
        self.context_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens,
            "local_generated_queries": sequence_end - text_tokens,
            "global_generated_kv": int(global_entry["tokens"]),
            "query_tokens_total": int(q.shape[2]),
            "kv_tokens_total": int(augmented_k.shape[2]),
            "q_by_k": [int(q.shape[2]), int(augmented_k.shape[2])],
            "ordering": {
                "queries": "text, local-generated",
                "keys_values": "text, local-generated,current-G-generated",
            },
            "local_pe_shape": list(pe.shape),
            "cache_storage": "cpu; transferred unchanged per consuming block",
        })
        return {
            "q": positioned_q, "k": augmented_k, "v": augmented_v,
            "pe": None, "attn_mask": attn_mask,
        }


def summary(value):
    work = value.detach().float()
    return {
        "shape": list(value.shape), "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()), "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


def difference(left, right):
    delta = left.detach().float() - right.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(left, right)),
    }


def overlap_metrics(predictions, regions):
    square_sum = 0.0
    count = 0
    pairs = []
    for left_index, left in enumerate(regions):
        for right_index in range(left_index + 1, len(regions)):
            right = regions[right_index]
            y0, y1 = max(left.y, right.y), min(left.y2, right.y2)
            x0, x1 = max(left.x, right.x), min(left.x2, right.x2)
            if y0 >= y1 or x0 >= x1:
                continue
            a = predictions[left_index][..., y0-left.y:y1-left.y, x0-left.x:x1-left.x]
            b = predictions[right_index][..., y0-right.y:y1-right.y, x0-right.x:x1-right.x]
            delta = a.float() - b.float()
            squared = float(delta.square().sum())
            values = delta.numel()
            rms = math.sqrt(squared / values)
            pairs.append({"left": left.index, "right": right.index, "rms": rms, "values": values})
            square_sum += squared
            count += values
    return {
        "aggregate_rms": math.sqrt(square_sum / count),
        "compared_values": count,
        "pairs": pairs,
    }


class EventTimer:
    def __init__(self):
        self.records = []

    def call(self, category, ordinal, function, **kwargs):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(**kwargs)
        end.record()
        self.records.append({"category": category, "ordinal": ordinal, "start": start, "end": end})
        return result

    def resolve(self):
        torch.cuda.synchronize()
        return [
            {"category": item["category"], "ordinal": item["ordinal"],
             "cuda_ms": float(item["start"].elapsed_time(item["end"]))}
            for item in self.records
        ]


class TerminalContextSampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 8d requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        timer = EventTimer()
        state = coordinator.initialize(h, sigmas[0])
        initial_h = state.h.detach().float().cpu()
        initial_g = state.g.detach().float().cpu()
        original_global = coordinator.adapter.predict_global
        original_region = coordinator.adapter.predict_region

        def timed_global(**kwargs):
            return timer.call("nonterminal_global", state.ordinal, original_global, **kwargs)

        def timed_region(**kwargs):
            return timer.call("nonterminal_local", state.ordinal, original_region, **kwargs)

        coordinator.adapter.predict_global = timed_global
        coordinator.adapter.predict_region = timed_region
        accepted_hashes = []
        for ordinal in range(3):
            state, x0_h = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=extra_args.get("seed", 0),
            )
            accepted_hashes.append({
                "ordinal": ordinal,
                "H": hashlib.sha256(state.h.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "G": hashlib.sha256(state.g.detach().float().cpu().numpy().tobytes()).hexdigest(),
            })

        ordinal = 3
        sigma = sigmas[ordinal]
        terminal_h, terminal_g = state.h, state.g
        regions = coordinator.planner.plan(tuple(terminal_h.shape[-2:]))
        base_options = extra_args["model_options"]
        h_snapshot, g_snapshot = terminal_h.clone(), terminal_g.clone()
        crop_input_hashes = []

        # A: ordinary production terminal crop predictions.
        ordinary_predictions = []
        for region in regions:
            view = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            crop_input_hashes.append(hashlib.sha256(view.detach().float().cpu().numpy().tobytes()).hexdigest())
            ordinary_predictions.append(timer.call(
                "A_local", ordinal, original_region,
                guider=model, h_view=view, sigma=sigma,
                canvas=tuple(terminal_h.shape[-2:]), region=region,
                model_options=base_options, seed=extra_args.get("seed", 0),
            ))
        a_x0, a_coverage = coordinator.assembler.assemble(ordinary_predictions, regions, HIGH_HW)

        # One fresh current-G source forward captures same-sigma all-block K/V
        # and supplies the terminal G_star used only for projection diagnostics.
        probe = CPUOffloadedContextProbe()
        global_options = candidate2_trajectory.merge_options(
            base_options,
            candidate2.model_options(
                "phase8d_current_G_capture",
                phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
                probe,
                "capture",
            ),
        )
        x0_g = timer.call(
            "terminal_global_context_capture", ordinal, original_global,
            guider=model, g=terminal_g, sigma=sigma, canvas=HIGH_HW,
            model_options=global_options, seed=extra_args.get("seed", 0),
        )
        g_star = terminal_g + (terminal_g - x0_g) / sigma * (sigmas[4] - sigma)

        # B: identical crops/order with the same captured K/V objects.
        captured_ids = {key: (id(item["k"]), id(item["v"])) for key, item in probe.global_kv.items()}
        context_predictions = []
        per_crop_context = []
        for region in regions:
            context_options = candidate2_trajectory.merge_options(
                base_options,
                candidate2.model_options(
                    f"phase8d_context_crop_{region.index}",
                    phase2.rope_for_crop(region), probe, "context",
                ),
            )
            view = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            context_predictions.append(timer.call(
                "B_context_local", ordinal, original_region,
                guider=model, h_view=view, sigma=sigma, canvas=HIGH_HW,
                region=region, model_options=context_options,
                seed=extra_args.get("seed", 0),
            ))
            current_ids = {key: (id(item["k"]), id(item["v"])) for key, item in probe.global_kv.items()}
            per_crop_context.append({
                "crop": region.index,
                "same_KV_objects": current_ids == captured_ids,
                "context_records_after": len(probe.context_records),
            })
        probe.assert_complete()
        b_x0, b_coverage = coordinator.assembler.assemble(context_predictions, regions, HIGH_HW)
        if not torch.equal(terminal_h, h_snapshot) or not torch.equal(terminal_g, g_snapshot):
            raise RuntimeError("Terminal context calls mutated accepted H3/G3.")

        dt = sigmas[4] - sigma
        a_h_star = terminal_h + (terminal_h - a_x0) / sigma * dt
        b_h_star = terminal_h + (terminal_h - b_x0) / sigma * dt
        a_projection = coordinator.geometry.prolong(g_star - coordinator.geometry.restrict(a_h_star))
        b_projection = coordinator.geometry.prolong(g_star - coordinator.geometry.restrict(b_h_star))
        local_differences = [difference(b, a) for a, b in zip(ordinary_predictions, context_predictions)]
        timings = timer.resolve()
        context_layout = {
            "capture_blocks": len(probe.capture_records),
            "context_records": len(probe.context_records),
            "generated_global_tokens": int(probe.capture_records[0]["global_generated_tokens"]),
            "q_by_k": probe.context_records[0]["q_by_k"],
            "same_KV_all_crops": all(item["same_KV_objects"] for item in per_crop_context),
            "capture_storage": "CPU-offloaded, same tensor values/dtypes and integration",
            "captured_KV_bytes": probe.capture_bytes,
            "aggregate_KV_transfer_bytes": probe.transfer_bytes,
            "per_crop_identity": per_crop_context,
            "modified_blocks": "all 5 double-stream and 20 single-stream blocks",
            "integration": "append positioned current-G generated K/V; preserve local/text Q; restore ordinary text-query attention output",
        }
        self.outputs = {
            "A_TERMINAL_X0_H": a_x0.detach().float().cpu(),
            "B_CONTEXT_TERMINAL_X0_H": b_x0.detach().float().cpu(),
            "A_FINAL_H": a_h_star.detach().float().cpu(),
            "B_CONTEXT_FINAL_H": b_h_star.detach().float().cpu(),
            "TERMINAL_X0_G": x0_g.detach().float().cpu(),
        }
        self.result = {
            "sigmas": [float(value) for value in sigmas],
            "shared_controls": {
                "initial_H_hash": hashlib.sha256(initial_h.numpy().tobytes()).hexdigest(),
                "initial_G_hash": hashlib.sha256(initial_g.numpy().tobytes()).hexdigest(),
                "accepted_nonterminal_hashes": accepted_hashes,
                "H3_hash": hashlib.sha256(terminal_h.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "G3_hash": hashlib.sha256(terminal_g.detach().float().cpu().numpy().tobytes()).hexdigest(),
                "terminal_crop_input_hashes": crop_input_hashes,
                "crop_rectangles": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "same_sigma": True, "same_crop_order": True,
                "coverage_A": [float(a_coverage.min()), float(a_coverage.max())],
                "coverage_B": [float(b_coverage.min()), float(b_coverage.max())],
            },
            "context": context_layout,
            "variants": {
                "A_PRODUCTION_LOCAL_ONLY": self.metrics(a_h_star, a_x0, a_projection, g_star, coordinator.geometry),
                "B_CURRENT_G_CONTEXT": self.metrics(b_h_star, b_x0, b_projection, g_star, coordinator.geometry),
            },
            "A_vs_B": {
                "assembled_x0_H": difference(a_x0, b_x0),
                "final_H": difference(a_h_star, b_h_star),
                "per_crop": local_differences,
            },
            "overlap": {
                "A": overlap_metrics(ordinary_predictions, regions),
                "B": overlap_metrics(context_predictions, regions),
            },
            "timings": timings,
            "cuda_ms": {
                "nonterminal_global": sum(x["cuda_ms"] for x in timings if x["category"] == "nonterminal_global"),
                "nonterminal_local": sum(x["cuda_ms"] for x in timings if x["category"] == "nonterminal_local"),
                "terminal_global_context_capture": sum(x["cuda_ms"] for x in timings if x["category"] == "terminal_global_context_capture"),
                "A_terminal_local": sum(x["cuda_ms"] for x in timings if x["category"] == "A_local"),
                "B_terminal_context_local": sum(x["cuda_ms"] for x in timings if x["category"] == "B_context_local"),
            },
        }
        if callback is not None:
            callback(ordinal, a_x0, a_h_star, STEPS)
        return model_sampling.inverse_noise_scaling(sigmas[-1], a_h_star)

    @staticmethod
    def metrics(final, assembled, projection, g_star, geometry):
        projection_rms = summary(projection)["rms"]
        return {
            "assembled_x0_H": summary(assembled), "final_H": summary(final),
            "D_final_vs_terminal_G_star": difference(geometry.restrict(final), g_star),
            "projection_needed": {
                "rms": projection_rms,
                "over_H_star": projection_rms / summary(final)["rms"],
                "low_frequency_rms": summary(geometry.restrict(projection))["rms"],
            },
        }


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


def sheet(names, filename):
    w, h, label = 800, 400, 28
    canvas = Image.new("RGB", (w, (h + label) * len(names)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(names):
        with Image.open(OUTPUT / f"{name}.png") as source:
            preview = ImageOps.contain(source.convert("RGB"), (w, h), Image.Resampling.LANCZOS)
        y = row * (h + label)
        canvas.paste(preview, ((w - preview.width) // 2, y + label))
        draw.text((8, y + 7), name, fill="black")
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
    sampler = TerminalContextSampler()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
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
        "production_changes": False, "dense_context_run": False,
        "dense_context_reason": "Optional dense-H K/V would add 32,768-token all-block cache and 55 larger-attention calls; compact current-G result is the primary discriminator.",
    }
    data["performance"] = {
        "shared_wall_seconds": time.perf_counter() - started,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    data["decoded"] = decode(sampler.outputs)
    data["variants"]["A_PRODUCTION_LOCAL_ONLY"]["decoded"] = data["decoded"]["A_FINAL_H"]
    data["variants"]["B_CURRENT_G_CONTEXT"]["decoded"] = data["decoded"]["B_CONTEXT_FINAL_H"]
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    sheet(("A_TERMINAL_X0_H", "B_CONTEXT_TERMINAL_X0_H"), "TERMINAL_X0_H_COMPARISON.png")
    sheet(("A_FINAL_H", "B_CONTEXT_FINAL_H"), "FINAL_OUTPUT_COMPARISON.png")
    print(json.dumps({
        "report": str(REPORT), "context": data["context"],
        "projection": {name: item["projection_needed"] for name, item in data["variants"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
