"""Phase 8g: mean aggregation versus selection at 4,608 consumer K/V tokens."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_postinteraction_kv_density as phase8f
from comfy.ldm.flux import math as flux_math


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_postinteraction_kv_aggregation_results"
REPORT = OUTPUT / "report.json"


class ModeFactor(int):
    def __new__(cls, value: int, mode: str):
        instance = int.__new__(cls, value)
        instance.mode = mode
        return instance


VARIANTS = {
    "A_FULL_18432": ModeFactor(1, "full"),
    "B_DECIMATE_2X2_4608": ModeFactor(2, "decimate"),
    "C_MEAN_POOL_2X2_4608": ModeFactor(2, "mean_pool"),
}


def pool_2x2(value: torch.Tensor) -> torch.Tensor:
    if value.shape[2] != math.prod(phase8f.GLOBAL_HW):
        raise AssertionError(f"Expected full 96x192 source field, got {tuple(value.shape)}")
    batch, heads, _, width = value.shape
    work = value.reshape(batch, heads, 48, 2, 96, 2, width)
    return work.mean(dim=(3, 5)).reshape(batch, heads, 4608, width)


def pool_ids_2x2(ids: torch.Tensor) -> torch.Tensor:
    if ids.shape[:2] != (1, math.prod(phase8f.GLOBAL_HW)):
        raise AssertionError(f"Unexpected full-source IDs: {tuple(ids.shape)}")
    axes = ids.shape[-1]
    return ids.reshape(1, 48, 2, 96, 2, axes).mean(dim=(2, 4)).reshape(1, 4608, axes)


class AggregationProbe(phase8f.PostInteractionDensityProbe):
    pe_embedder = None
    last_instance = None

    def __init__(self):
        super().__init__()
        type(self).last_instance = self
        self.pooled_cache_bytes = 0
        self.pool_records = []

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        if sequence_end - text_tokens != 18432:
            raise AssertionError("Aggregation source was not the full 96x192 field.")
        positioned_k = flux_math.apply_rope1(k, pe)
        raw_generated_k = k[:, :, text_tokens:]
        generated_v = v[:, :, text_tokens:]

        role = "phase8f_full_current_G_capture"
        source_ids = self.position_records[role]["generated_ids_tensor"].to(
            device=k.device, dtype=torch.float32
        )
        pooled_ids = pool_ids_2x2(source_ids)
        pooled_raw_k = pool_2x2(raw_generated_k)
        pooled_v = pool_2x2(generated_v)
        pooled_pe = self.pe_embedder(pooled_ids)
        pooled_positioned_k = flux_math.apply_rope1(pooled_raw_k, pooled_pe)

        full_k_cpu = positioned_k[:, :, text_tokens:].detach().cpu()
        full_v_cpu = generated_v.detach().cpu()
        pooled_k_cpu = pooled_positioned_k.detach().cpu()
        pooled_v_cpu = pooled_v.detach().cpu()
        self.global_kv[key] = {
            "k": full_k_cpu, "v": full_v_cpu, "tokens": 18432,
            "pooled_k": pooled_k_cpu, "pooled_v": pooled_v_cpu,
        }
        full_bytes = full_k_cpu.numel() * full_k_cpu.element_size()
        full_bytes += full_v_cpu.numel() * full_v_cpu.element_size()
        pooled_bytes = pooled_k_cpu.numel() * pooled_k_cpu.element_size()
        pooled_bytes += pooled_v_cpu.numel() * pooled_v_cpu.element_size()
        self.capture_bytes += full_bytes
        self.pooled_cache_bytes += pooled_bytes
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens, "global_generated_tokens": 18432,
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
            "pe_shape": list(pe.shape), "storage_device": "cpu",
        })
        self.pool_records.append({
            "block_type": key[0], "block_index": key[1],
            "source_tokens": 18432, "pooled_tokens": 4608,
            "contributors_per_pooled_token": 4,
            "source_contribution_count": 4608 * 4,
            "all_source_positions_used_once": True,
            "k_pool_boundary": "generated pre-RoPE K",
            "v_pool_boundary": "generated V",
            "pooled_rope": "pe_embedder(arithmetic-mean source IDs)",
            "pooled_ids_first": pooled_ids[0, 0].detach().cpu().tolist(),
            "pooled_ids_last": pooled_ids[0, -1].detach().cpu().tolist(),
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def position_capture(self, role: str):
        parent_patch = super().position_capture(role)

        def patch(args):
            result = parent_patch(args)
            self.position_records[role]["generated_ids_tensor"] = args["img_ids"].detach().cpu()
            return result

        return patch

    def selected(self, entry, name):
        mode = getattr(self.factor, "mode", "decimate")
        if mode == "mean_pool":
            return entry[f"pooled_{name}"]
        return super().selected(entry, name)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase8f.OUTPUT = OUTPUT
    phase8f.REPORT = REPORT
    phase8f.VARIANTS = VARIANTS
    phase8f.PostInteractionDensityProbe = AggregationProbe

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    AggregationProbe.pe_embedder = model.model.diffusion_model.pe_embedder
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase8f.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn(
        (1, 128, *phase8f.HIGH_HW), generator=torch.Generator().manual_seed(phase8f.SEED)
    )
    sigmas = phase2.get_schedule(phase8f.STEPS, math.prod(phase8f.HIGH_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = phase8f.DensityDiscriminatorSampler()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    started = phase8f.time.perf_counter()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=phase8f.SEED,
        )
    torch.cuda.synchronize()
    data = sampler.result
    probe = AggregationProbe.last_instance
    data["configuration"] = {
        "target_pixels_hw": [2048, 4096], "H": [128, 256], "G": [96, 192],
        "prompt": phase8f.PROMPT, "seed": phase8f.SEED, "cfg": 1.0,
        "production_changes": False,
        "source": "one ordinary full 96x192 current-G forward",
        "aggregation_stage": "generated K pre-RoPE and generated V after full-source block projection",
        "pooled_coordinate": "arithmetic mean/geometric center of four original full-source IDs",
    }
    data["performance"] = {
        "whole_experiment_wall_seconds_before_decode": phase8f.time.perf_counter() - started,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
    }
    data["shared_controls"]["pooled_consumer_KV_cache_bytes"] = probe.pooled_cache_bytes
    first_pool = probe.pool_records[0]
    data["shared_controls"]["pooling_records"] = {
        "blocks": len(probe.pool_records), "source_tokens_per_block": 18432,
        "pooled_tokens_per_block": 4608, "contributors_per_token": 4,
        "all_source_positions_contribute_once": all(
            item["all_source_positions_used_once"] for item in probe.pool_records
        ),
        "K_pooled_pre_RoPE": True,
        "coordinate_convention": "geometric center from mean of original live four-axis IDs",
        "first_pooled_id": first_pool["pooled_ids_first"],
        "last_pooled_id": first_pool["pooled_ids_last"],
        "per_block_boundaries": probe.pool_records,
    }
    for name, value in data["variants"].items():
        value["consumer_strategy"] = getattr(VARIANTS[name], "mode")
        if name == "C_MEAN_POOL_2X2_4608":
            value["selected_source_indices"] = None
            value["aggregation"] = {
                "source_positions_per_token": 4,
                "all_18432_positions_contribute_once": True,
                "K_arithmetic_mean_before_RoPE": True,
                "V_arithmetic_mean": True,
                "pooled_coordinate": "geometric cell center",
            }
    data["decoded"] = phase8f.decode(sampler.outputs)
    for name in VARIANTS:
        data["variants"][name]["decoded"] = data["decoded"][name]
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    names = tuple(VARIANTS)
    phase8f.make_sheet(names, "", "POSTINTERACTION_KV_AGGREGATION_FINAL_COMPARISON.png")
    phase8f.make_sheet(names, "_X0_H", "POSTINTERACTION_KV_AGGREGATION_TERMINAL_X0_COMPARISON.png")
    print(json.dumps({
        name: {
            "strategy": data["variants"][name]["consumer_strategy"],
            "overlap": data["variants"][name]["overlap"]["aggregate_rms"],
            "projection_ratio": data["variants"][name]["projection_needed"]["over_H_star"],
            "assembled_vs_full": data["variants"][name]["assembled_vs_full"]["rms"],
            "cuda_ms": data["variants"][name]["terminal_local_cuda_ms"],
            "hash": data["variants"][name]["decoded"]["sha256_rgb"],
        } for name in VARIANTS
    }, indent=2))


if __name__ == "__main__":
    main()
