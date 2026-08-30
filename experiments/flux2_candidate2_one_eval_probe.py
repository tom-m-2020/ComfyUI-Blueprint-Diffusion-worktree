"""One-evaluation FLUX.2 compact-global attention-context probe.

Experiment only. Captures reduced-global generated K/V at the current latent
and sigma, then exposes them to ordinary local generated queries in every FLUX
double/single block. It deliberately does not implement sparse execution,
caching, a sampler lifecycle, or a reusable model abstraction.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import flux2_coarse_global_local_falsification as phase2
from comfy.ldm.flux import math as flux_math


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate2_one_eval_results",
    )
    parser.add_argument("--crop-index", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tensor_difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = (value.float() - reference.float()).abs()
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": float(difference.square().mean().sqrt()),
    }


def low_frequency_difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value_low = F.interpolate(
        F.interpolate(value.float(), scale_factor=0.5, mode="bilinear", align_corners=False),
        size=value.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    reference_low = F.interpolate(
        F.interpolate(reference.float(), scale_factor=0.5, mode="bilinear", align_corners=False),
        size=reference.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return tensor_difference(value_low, reference_low)


def append_patch(options: dict[str, Any], name: str, patch) -> None:
    transformer = options.setdefault("transformer_options", {})
    patches = transformer.setdefault("patches", {})
    patches.setdefault(name, []).append(patch)


class OneEvaluationContextProbe:
    def __init__(self) -> None:
        self.global_kv: dict[tuple[str, int], dict[str, torch.Tensor | int]] = {}
        self.pending_normal_attention: dict[tuple[str, int], torch.Tensor] = {}
        self.capture_records: list[dict[str, Any]] = []
        self.context_records: list[dict[str, Any]] = []
        self.position_records: dict[str, dict[str, Any]] = {}

    @staticmethod
    def block_key(extra_options: dict[str, Any]) -> tuple[str, int]:
        return str(extra_options["block_type"]), int(extra_options["block_index"])

    def position_capture(self, role: str):
        def patch(args: dict[str, Any]) -> dict[str, Any]:
            img_ids = args["img_ids"].detach().float().cpu()
            txt_ids = args["txt_ids"].detach().float().cpu()
            self.position_records[role] = {
                "generated_ids_shape": list(img_ids.shape),
                "generated_ids_first": img_ids[0, 0].tolist(),
                "generated_ids_last": img_ids[0, -1].tolist(),
                "generated_ids_axis_min": img_ids.amin(dim=(0, 1)).tolist(),
                "generated_ids_axis_max": img_ids.amax(dim=(0, 1)).tolist(),
                "text_ids_shape": list(txt_ids.shape),
                "text_ids_first": txt_ids[0, 0].tolist(),
                "text_ids_last": txt_ids[0, -1].tolist(),
            }
            return args

        return patch

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        if sequence_end != k.shape[2]:
            raise AssertionError(f"Unexpected global image slice {extra_options['img_slice']} for {tuple(k.shape)}.")
        positioned_k = flux_math.apply_rope1(k, pe)
        self.global_kv[key] = {
            "k": positioned_k[:, :, text_tokens:].detach(),
            "v": v[:, :, text_tokens:].detach(),
            "tokens": sequence_end - text_tokens,
        }
        self.capture_records.append(
            {
                "block_type": key[0],
                "block_index": key[1],
                "text_tokens": text_tokens,
                "global_generated_tokens": sequence_end - text_tokens,
                "heads": int(k.shape[1]),
                "head_dim": int(k.shape[-1]),
                "pe_shape": list(pe.shape),
            }
        )
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def add_global_context(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        if key not in self.global_kv:
            raise KeyError(f"No same-evaluation global K/V captured for {key}.")
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        global_entry = self.global_kv[key]

        # Preserve the ordinary attention result for text queries. The augmented
        # call below changes only generated-query attention output.
        self.pending_normal_attention[key] = flux_math.attention(
            q, k, v, pe=pe, mask=attn_mask, transformer_options=extra_options
        )
        positioned_q = flux_math.apply_rope1(q, pe)
        positioned_k = flux_math.apply_rope1(k, pe)
        augmented_k = torch.cat((positioned_k, global_entry["k"]), dim=2)
        augmented_v = torch.cat((v, global_entry["v"]), dim=2)
        self.context_records.append(
            {
                "block_type": key[0],
                "block_index": key[1],
                "text_tokens": text_tokens,
                "local_generated_queries": sequence_end - text_tokens,
                "global_generated_kv": int(global_entry["tokens"]),
                "query_tokens_total": int(q.shape[2]),
                "kv_tokens_total": int(augmented_k.shape[2]),
                "q_by_k": [int(q.shape[2]), int(augmented_k.shape[2])],
                "ordering": {
                    "queries": "text, local-generated",
                    "keys_values": "text, local-generated, compact-global-generated",
                },
                "local_pe_shape": list(pe.shape),
            }
        )
        return {
            "q": positioned_q,
            "k": augmented_k,
            "v": augmented_v,
            "pe": None,
            "attn_mask": attn_mask,
        }

    def restore_text_attention(self, attention_output, extra_options):
        key = self.block_key(extra_options)
        normal = self.pending_normal_attention.pop(key)
        text_tokens = int(extra_options["img_slice"][0])
        output = attention_output.clone()
        output[:, :text_tokens] = normal[:, :text_tokens]
        return output

    def assert_complete(self) -> None:
        if self.pending_normal_attention:
            raise AssertionError(f"Unconsumed ordinary attention outputs: {tuple(self.pending_normal_attention)}")
        captured = {(item["block_type"], item["block_index"]) for item in self.capture_records}
        consumed = {(item["block_type"], item["block_index"]) for item in self.context_records}
        if captured != consumed:
            raise AssertionError(f"Captured/context block mismatch: {captured ^ consumed}")


def model_options(role: str, rope: dict[str, float], probe: OneEvaluationContextProbe, mode: str) -> dict[str, Any]:
    options: dict[str, Any] = {"transformer_options": {"rope_options": dict(rope)}}
    append_patch(options, "post_input", probe.position_capture(role))
    if mode == "capture":
        append_patch(options, "attn1_patch", probe.capture_global)
    elif mode == "context":
        append_patch(options, "attn1_patch", probe.add_global_context)
        append_patch(options, "attn1_output_patch", probe.restore_text_attention)
    elif mode != "ordinary":
        raise ValueError(mode)
    return options


def evaluate(model, value, sigma, options, seed) -> tuple[torch.Tensor, dict[str, Any]]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    else:
        before = 0
    started = time.perf_counter()
    prediction = model(value, sigma.expand(value.shape[0]), model_options=options, seed=seed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
    else:
        peak = 0
    return prediction.detach().float().cpu(), {
        "seconds": time.perf_counter() - started,
        "peak_allocated_gib": peak / GIB,
        "peak_increment_gib": (peak - before) / GIB,
        "input": phase2.stats(value),
        "prediction": phase2.stats(prediction),
    }


class OneEvaluationSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, global_hw, crop, probe, seed) -> None:
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crop = crop
        self.probe = probe
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}

    def sample(
        self,
        model,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if denoise_mask is not None:
            raise ValueError("This one-evaluation T2I probe does not use a mask.")
        sigma = sigmas[0]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
        global_input = F.interpolate(noise, size=self.global_hw, mode="bilinear", align_corners=False)
        sequence = (
            ("C_FULL_DENSE_REFERENCE", noise, model_options("dense", {}, self.probe, "ordinary")),
            (
                "A_LOCAL_ONLY",
                local_input,
                model_options("local_only", phase2.rope_for_crop(crop), self.probe, "ordinary"),
            ),
            (
                "COMPACT_GLOBAL_CAPTURE",
                global_input,
                model_options(
                    "compact_global",
                    phase2.rope_for_global(*self.target_hw, *self.global_hw),
                    self.probe,
                    "capture",
                ),
            ),
            (
                "B_LOCAL_COMPACT_GLOBAL_CONTEXT",
                local_input,
                model_options("local_context", phase2.rope_for_crop(crop), self.probe, "context"),
            ),
        )
        for name, value, options in sequence:
            # Preserve sampling-owned options such as the sigma schedule while
            # making this call's experimental patches explicit.
            merged = phase2.comfy.model_patcher.create_model_options_clone(extra_args["model_options"])
            merged_transformer = merged.setdefault("transformer_options", {})
            merged_transformer.update(options["transformer_options"])
            self.outputs[name], self.calls[name] = evaluate(
                model, value, sigma, merged, self.seed
            )
        self.probe.assert_complete()
        return noise


def dry_run() -> None:
    local = torch.randn(1, 3, 7, 4)
    global_k = torch.randn(1, 3, 5, 4)
    augmented = torch.cat((local, global_k), dim=2)
    if augmented.shape != (1, 3, 12, 4):
        raise AssertionError(augmented.shape)
    print(json.dumps({"local_kv": 7, "global_kv": 5, "augmented_shape": list(augmented.shape)}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    global_width, global_height = 512, 256
    crop_width = crop_height = 512
    overlap_pixels = 128
    seed = 20260829
    target_hw = (height // 16, width // 16)
    global_hw = (global_height // 16, global_width // 16)
    crop_hw = (crop_height // 16, crop_width // 16)
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap_pixels // 16)
    if not 0 <= args.crop_index < len(crops):
        raise ValueError(f"--crop-index must be in [0,{len(crops) - 1}].")
    crop = crops[args.crop_index]

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase2.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    latent = torch.randn((1, 128, *target_hw), generator=torch.Generator().manual_seed(seed))
    sigma = phase2.get_schedule(4, math.prod(target_hw)).float()[0]
    probe = OneEvaluationContextProbe()
    sampler = OneEvaluationSampler(target_hw, global_hw, crop, probe, seed)

    # All calls share one native conditioning/model lifecycle and exact high-noise
    # sigma. Global capture immediately precedes consumption; there is no update.
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model,
            latent.clone(),
            1.0,
            sampler,
            torch.stack((sigma, torch.zeros_like(sigma))),
            positive,
            negative,
            torch.zeros_like(latent),
            disable_pbar=True,
            seed=seed,
        )
    calls = sampler.calls
    dense = sampler.outputs["C_FULL_DENSE_REFERENCE"]
    local_only = sampler.outputs["A_LOCAL_ONLY"]
    global_prediction = sampler.outputs["COMPACT_GLOBAL_CAPTURE"]
    local_context = sampler.outputs["B_LOCAL_COMPACT_GLOBAL_CONTEXT"]

    dense_crop = dense[:, :, crop.y:crop.y2, crop.x:crop.x2]
    variants = {
        "A_LOCAL_ONLY": local_only,
        "B_LOCAL_COMPACT_GLOBAL_CONTEXT": local_context,
        "C_FULL_DENSE_REFERENCE_CROP": dense_crop,
        "COMPACT_GLOBAL": global_prediction,
    }
    for name, prediction in variants.items():
        with torch.inference_mode():
            pixels = vae.decode(prediction).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")

    # Put each local prediction back into the same global-only diagnostic so the
    # crop can be read with surrounding whole-canvas context without implying a
    # sampler update or fusion result.
    mapped_global = F.interpolate(global_prediction, size=target_hw, mode="bilinear", align_corners=False)
    for name, prediction in {
        "A_LOCAL_ONLY": local_only,
        "B_LOCAL_COMPACT_GLOBAL_CONTEXT": local_context,
        "C_FULL_DENSE_REFERENCE_CROP": dense_crop,
    }.items():
        canvas = mapped_global.clone()
        canvas[:, :, crop.y:crop.y2, crop.x:crop.x2] = prediction
        with torch.inference_mode():
            pixels = vae.decode(canvas).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}_IN_GLOBAL_CONTEXT.png")

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": phase2.PROMPT,
            "seed": seed,
            "sigma": float(sigma),
            "evaluation": "first/high-noise evaluation only; no sampler update",
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "global_image_hw": [global_height, global_width],
            "global_latent_hw": list(global_hw),
            "crop": crop.__dict__,
            "crop_image_hw": [crop_height, crop_width],
            "local_rope_options": phase2.rope_for_crop(crop),
            "global_rope_options": phase2.rope_for_global(*target_hw, *global_hw),
            "modified_blocks": "all 5 double-stream and all 20 single-stream blocks",
            "intervention": "append same-sigma compact-global generated K/V after RoPE; retain local/text Q; restore ordinary text-query attention output",
            "normal_token_order": "double: attention over text,generated; single: text,generated",
            "context_token_order": "Q=text,local-generated; K/V=text,local-generated,compact-global-generated",
        },
        "positions": probe.position_records,
        "global_capture_blocks": probe.capture_records,
        "context_blocks": probe.context_records,
        "calls": calls,
        "comparisons": {
            "context_vs_local_only": {
                "absolute": tensor_difference(local_context, local_only),
                "low_frequency": low_frequency_difference(local_context, local_only),
                "residual_norm": phase2.norm(local_context - local_only),
                "relative_residual_to_local_norm": phase2.norm(local_context - local_only) / phase2.norm(local_only),
            },
            "local_only_vs_dense_crop": {
                "absolute": tensor_difference(local_only, dense_crop),
                "low_frequency": low_frequency_difference(local_only, dense_crop),
                "residual_norm": phase2.norm(local_only - dense_crop),
                "relative_residual_to_dense_norm": phase2.norm(local_only - dense_crop) / phase2.norm(dense_crop),
            },
            "context_vs_dense_crop": {
                "absolute": tensor_difference(local_context, dense_crop),
                "low_frequency": low_frequency_difference(local_context, dense_crop),
                "residual_norm": phase2.norm(local_context - dense_crop),
                "relative_residual_to_dense_norm": phase2.norm(local_context - dense_crop) / phase2.norm(dense_crop),
            },
        },
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in variants},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "comparisons": report["comparisons"]}, indent=2))


if __name__ == "__main__":
    main()
