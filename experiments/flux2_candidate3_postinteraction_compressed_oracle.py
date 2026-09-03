"""Phase 12: post-interaction compressed full-H context discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

import flux2_candidate3_normalized_w_shared_context as phase11
from comfy.ldm.flux import math as flux_math


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_postinteraction_compressed_oracle_results"
REPORT = OUTPUT / "report.json"
SOURCE_HW = (64, 128)
CONSUMER_HW = (24, 48)


def pool_spatial(value: torch.Tensor) -> torch.Tensor:
    """Adaptive-area mean [B,H,Y*X,D] from 64x128 to 24x48."""
    batch, heads, tokens, width = value.shape
    if tokens != SOURCE_HW[0] * SOURCE_HW[1]:
        raise AssertionError(f"Expected {SOURCE_HW} source, got {tokens} tokens.")
    work = value.reshape(batch, heads, *SOURCE_HW, width)
    work = work.permute(0, 1, 4, 2, 3).reshape(batch * heads, width, *SOURCE_HW)
    pooled = F.adaptive_avg_pool2d(work, CONSUMER_HW)
    pooled = pooled.reshape(batch, heads, width, *CONSUMER_HW).permute(0, 1, 3, 4, 2)
    return pooled.reshape(batch, heads, CONSUMER_HW[0] * CONSUMER_HW[1], width)


def pool_ids(value: torch.Tensor) -> torch.Tensor:
    batch, tokens, axes = value.shape
    if (batch, tokens) != (1, SOURCE_HW[0] * SOURCE_HW[1]):
        raise AssertionError(f"Unexpected source IDs: {tuple(value.shape)}")
    work = value.reshape(batch, *SOURCE_HW, axes).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(work.float(), CONSUMER_HW)
    return pooled.permute(0, 2, 3, 1).reshape(batch, -1, axes)


class PostInteractionCompressedProbe(phase11.phase8d.CPUOffloadedContextProbe):
    consumer_tokens = CONSUMER_HW[0] * CONSUMER_HW[1]

    def __init__(self):
        super().__init__()
        self.compression_records = []

    def position_capture(self, role):
        parent = super().position_capture(role)

        def patch(args):
            result = parent(args)
            self.position_records[role]["generated_ids_tensor"] = args["img_ids"].detach().cpu()
            return result

        return patch

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        source_tokens = sequence_end - text_tokens
        if source_tokens != SOURCE_HW[0] * SOURCE_HW[1]:
            raise AssertionError(f"Compressed oracle source has {source_tokens} tokens.")

        # The Phase-11 oracle exposes generated K only after native source RoPE.
        # Pool that exact positioned field, so every contributing vector retains
        # its original full-H positional phase; no compact-grid renumbering occurs.
        positioned_k = flux_math.apply_rope1(k, pe)[:, :, text_tokens:]
        generated_v = v[:, :, text_tokens:]
        pooled_k = pool_spatial(positioned_k).detach().cpu()
        pooled_v = pool_spatial(generated_v).detach().cpu()
        self.global_kv[key] = {
            "k": pooled_k,
            "v": pooled_v,
            "tokens": self.consumer_tokens,
        }
        byte_count = pooled_k.numel() * pooled_k.element_size()
        byte_count += pooled_v.numel() * pooled_v.element_size()
        self.capture_bytes += byte_count

        role = next(iter(self.position_records))
        source_ids = self.position_records[role]["generated_ids_tensor"]
        pooled_coordinates = pool_ids(source_ids)
        record = {
            "block_type": key[0],
            "block_index": key[1],
            "text_tokens": text_tokens,
            "source_generated_shape": [int(x) for x in positioned_k.shape],
            "source_generated_tokens": source_tokens,
            "consumer_k_shape": [int(x) for x in pooled_k.shape],
            "consumer_v_shape": [int(x) for x in pooled_v.shape],
            "consumer_tokens": self.consumer_tokens,
            "compression": "adaptive-area arithmetic mean 64x128 -> 24x48",
            "K_boundary": "after full-source interaction and native full-H RoPE",
            "V_boundary": "after full-source interaction",
            "position_policy": "average of full-H source IDs for telemetry; K retains constituent native RoPE phases",
            "pooled_coordinate_first": pooled_coordinates[0, 0].tolist(),
            "pooled_coordinate_last": pooled_coordinates[0, -1].tolist(),
            "storage_device": "cpu",
        }
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens, "global_generated_tokens": source_tokens,
            "consumer_generated_tokens": self.consumer_tokens,
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
            "pe_shape": list(pe.shape), "storage_device": "cpu",
        })
        self.compression_records.append(record)
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}


class Phase12Sampler(phase11.SharedContextSampler):
    def make_context_probe(self):
        if self.context_source == "H_POSTINTERACTION_COMPRESSED":
            return PostInteractionCompressedProbe()
        return super().make_context_probe()

    def context_source_tensor_and_rope(self, state):
        if self.context_source == "H_POSTINTERACTION_COMPRESSED":
            return state.h, {}
        return super().context_source_tensor_and_rope(state)


def main():
    phase11.OUTPUT = OUTPUT
    phase11.REPORT = REPORT
    phase11.VARIANTS = (
        ("A_LOCAL_ONLY", None),
        ("B_FIXED_G_CONTEXT", "G"),
        ("D_POST_INTERACTION_COMPRESSED_H_CONTEXT", "H_POSTINTERACTION_COMPRESSED"),
        ("C_FULL_H_CONTEXT_ORACLE", "H"),
    )
    phase11.SAMPLER_CLASS = Phase12Sampler
    phase11.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["phase12_isolation"] = {
        "source_execution": "D uses ordinary full accepted-H 64x128 source through all 25 native blocks",
        "compression_boundary": "per-block generated K after native full-H RoPE and generated V",
        "compression_rule": "torch adaptive_avg_pool2d area mean from 64x128 to 24x48",
        "consumer_positions": 1152,
        "fixed_G_consumer_positions": 1152,
        "source_position_provenance": "pooled K averages full-H-positioned K; no compact-grid RoPE renumbering",
        "production_changes": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
