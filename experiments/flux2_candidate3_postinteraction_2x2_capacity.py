"""Phase 12b: 2,048-token post-interaction context capacity discriminator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import flux2_candidate3_normalized_w_shared_context as phase11
import flux2_candidate3_postinteraction_compressed_oracle as phase12
from comfy.ldm.flux import math as flux_math


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_postinteraction_2x2_capacity_results"
REPORT = OUTPUT / "report.json"
SOURCE_HW = (64, 128)
CONSUMER_HW = (32, 64)


def pool_2x2(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    if tokens != SOURCE_HW[0] * SOURCE_HW[1]:
        raise AssertionError(f"Expected {SOURCE_HW} source, got {tokens} tokens.")
    work = value.reshape(batch, heads, 32, 2, 64, 2, width)
    return work.mean(dim=(3, 5)).reshape(batch, heads, 2048, width)


def coordinate_provenance(source_ids: torch.Tensor):
    if tuple(source_ids.shape[:2]) != (1, 8192):
        raise AssertionError(f"Unexpected source IDs: {tuple(source_ids.shape)}")
    axes = int(source_ids.shape[-1])
    quads = source_ids.reshape(1, 32, 2, 64, 2, axes)
    quads = quads.permute(0, 1, 3, 2, 4, 5).reshape(2048, 4, axes)
    return quads.float().tolist()


class PostInteraction2x2Probe(phase11.phase8d.CPUOffloadedContextProbe):
    consumer_tokens = 2048

    def __init__(self):
        super().__init__()
        self.compression_records = []
        self.coordinate_provenance = None

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
        if sequence_end - text_tokens != 8192:
            raise AssertionError("2x2 source is not the full accepted-H field.")
        positioned_k = flux_math.apply_rope1(k, pe)[:, :, text_tokens:]
        generated_v = v[:, :, text_tokens:]
        pooled_k = pool_2x2(positioned_k).detach().cpu()
        pooled_v = pool_2x2(generated_v).detach().cpu()
        self.global_kv[key] = {"k": pooled_k, "v": pooled_v, "tokens": 2048}
        byte_count = pooled_k.numel() * pooled_k.element_size()
        byte_count += pooled_v.numel() * pooled_v.element_size()
        self.capture_bytes += byte_count

        role = next(iter(self.position_records))
        source_ids = self.position_records[role]["generated_ids_tensor"]
        if self.coordinate_provenance is None:
            self.coordinate_provenance = coordinate_provenance(source_ids)
        provenance_bytes = json.dumps(self.coordinate_provenance, separators=(",", ":")).encode()
        provenance_hash = hashlib.sha256(provenance_bytes).hexdigest()
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens, "global_generated_tokens": 8192,
            "consumer_generated_tokens": 2048,
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
            "pe_shape": list(pe.shape), "storage_device": "cpu",
        })
        self.compression_records.append({
            "block_type": key[0], "block_index": key[1],
            "source_generated_shape": list(positioned_k.shape),
            "consumer_k_shape": list(pooled_k.shape),
            "consumer_v_shape": list(pooled_v.shape),
            "source_tokens": 8192, "consumer_tokens": 2048,
            "compression": "nonoverlapping 2x2 arithmetic mean 64x128 -> 32x64",
            "contributors_per_consumer": 4,
            "K_boundary": "after native full-H RoPE",
            "V_boundary": "generated V after full-H source interaction",
            "position_policy": "mean of four already native-RoPE-positioned K vectors; no renumbering",
            "coordinate_provenance_sha256": provenance_hash,
            "first_source_quad": self.coordinate_provenance[0],
            "last_source_quad": self.coordinate_provenance[-1],
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}


class Phase12bSampler(phase11.SharedContextSampler):
    probes = []

    def make_context_probe(self):
        if self.context_source == "H_POSTINTERACTION_COMPRESSED":
            probe = phase12.PostInteractionCompressedProbe()
        elif self.context_source == "H_POSTINTERACTION_2X2":
            probe = PostInteraction2x2Probe()
        else:
            probe = super().make_context_probe()
        type(self).probes.append(probe)
        return probe

    def context_source_tensor_and_rope(self, state):
        if self.context_source in {
            "H_POSTINTERACTION_COMPRESSED", "H_POSTINTERACTION_2X2"
        }:
            return state.h, {}
        return super().context_source_tensor_and_rope(state)


def main():
    phase11.OUTPUT = OUTPUT
    phase11.REPORT = REPORT
    phase11.VARIANTS = (
        ("B_FIXED_G_CONTEXT", "G"),
        ("D_POST_INTERACTION_COMPRESSED", "H_POSTINTERACTION_COMPRESSED"),
        ("E_POST_INTERACTION_2X2_2048", "H_POSTINTERACTION_2X2"),
        ("C_FULL_H_CONTEXT_ORACLE", "H"),
    )
    phase11.SAMPLER_CLASS = Phase12bSampler
    Phase12bSampler.probes = []
    phase11.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    two_by_two = [p for p in Phase12bSampler.probes if isinstance(p, PostInteraction2x2Probe)]
    if len(two_by_two) != 4:
        raise AssertionError(f"Expected four fresh 2x2 probes, got {len(two_by_two)}.")
    hashes = {
        p.compression_records[0]["coordinate_provenance_sha256"] for p in two_by_two
    }
    if len(hashes) != 1:
        raise AssertionError("2x2 coordinate provenance changed across intervals.")
    report["phase12b_isolation"] = {
        "source_execution": "ordinary full accepted-H 64x128 source at each current sigma",
        "compression_boundary": "generated K after native full-H RoPE and generated V",
        "compression_rule": "nonoverlapping 2x2 arithmetic mean",
        "consumer_grid": [32, 64],
        "consumer_tokens": 2048,
        "contributors_per_consumer": 4,
        "all_source_positions_used_exactly_once": True,
        "coordinate_provenance_sha256": next(iter(hashes)),
        "coordinate_provenance": two_by_two[0].coordinate_provenance,
        "fresh_probe_count": len(two_by_two),
        "production_changes": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
