"""Phase 12c: final 4,096-token post-interaction capacity discriminator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

import flux2_candidate3_normalized_w_shared_context as phase11
import flux2_candidate3_postinteraction_compressed_oracle as phase12
import flux2_candidate3_postinteraction_2x2_capacity as phase12b
from comfy.ldm.flux import math as flux_math


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_postinteraction_2x1_capacity_results"
REPORT = OUTPUT / "report.json"
SOURCE_HW = (64, 128)
CONSUMER_HW = (32, 128)


def pool_2x1(value: torch.Tensor) -> torch.Tensor:
    batch, heads, tokens, width = value.shape
    if tokens != SOURCE_HW[0] * SOURCE_HW[1]:
        raise AssertionError(f"Expected {SOURCE_HW} source, got {tokens} tokens.")
    work = value.reshape(batch, heads, 32, 2, 128, width)
    return work.mean(dim=3).reshape(batch, heads, 4096, width)


def coordinate_provenance(source_ids: torch.Tensor):
    if tuple(source_ids.shape[:2]) != (1, 8192):
        raise AssertionError(f"Unexpected source IDs: {tuple(source_ids.shape)}")
    axes = int(source_ids.shape[-1])
    pairs = source_ids.reshape(1, 32, 2, 128, axes)
    pairs = pairs.permute(0, 1, 3, 2, 4).reshape(4096, 2, axes)
    return pairs.float().tolist()


class PostInteraction2x1Probe(phase11.phase8d.CPUOffloadedContextProbe):
    consumer_tokens = 4096

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
            raise AssertionError("2x1 source is not the full accepted-H field.")
        positioned_k = flux_math.apply_rope1(k, pe)[:, :, text_tokens:]
        generated_v = v[:, :, text_tokens:]
        pooled_k = pool_2x1(positioned_k).detach().cpu()
        pooled_v = pool_2x1(generated_v).detach().cpu()
        self.global_kv[key] = {"k": pooled_k, "v": pooled_v, "tokens": 4096}
        byte_count = pooled_k.numel() * pooled_k.element_size()
        byte_count += pooled_v.numel() * pooled_v.element_size()
        self.capture_bytes += byte_count

        role = next(iter(self.position_records))
        source_ids = self.position_records[role]["generated_ids_tensor"]
        if self.coordinate_provenance is None:
            self.coordinate_provenance = coordinate_provenance(source_ids)
        encoded = json.dumps(self.coordinate_provenance, separators=(",", ":")).encode()
        provenance_hash = hashlib.sha256(encoded).hexdigest()
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_tokens, "global_generated_tokens": 8192,
            "consumer_generated_tokens": 4096,
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
            "pe_shape": list(pe.shape), "storage_device": "cpu",
        })
        self.compression_records.append({
            "block_type": key[0], "block_index": key[1],
            "source_generated_shape": list(positioned_k.shape),
            "consumer_k_shape": list(pooled_k.shape),
            "consumer_v_shape": list(pooled_v.shape),
            "source_tokens": 8192, "consumer_tokens": 4096,
            "compression": "nonoverlapping vertical 2x1 arithmetic mean 64x128 -> 32x128",
            "anisotropy": "vertical pairs; full horizontal density retained",
            "contributors_per_consumer": 2,
            "K_boundary": "after native full-H RoPE",
            "V_boundary": "generated V after full-H source interaction",
            "position_policy": "mean of two already native-RoPE-positioned K vectors; no renumbering",
            "coordinate_provenance_sha256": provenance_hash,
            "first_source_pair": self.coordinate_provenance[0],
            "last_source_pair": self.coordinate_provenance[-1],
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}


class Phase12cSampler(phase11.SharedContextSampler):
    probes = []

    def make_context_probe(self):
        if self.context_source == "H_POSTINTERACTION_COMPRESSED":
            probe = phase12.PostInteractionCompressedProbe()
        elif self.context_source == "H_POSTINTERACTION_2X2":
            probe = phase12b.PostInteraction2x2Probe()
        elif self.context_source == "H_POSTINTERACTION_2X1":
            probe = PostInteraction2x1Probe()
        else:
            probe = super().make_context_probe()
        type(self).probes.append(probe)
        return probe

    def context_source_tensor_and_rope(self, state):
        if self.context_source in {
            "H_POSTINTERACTION_COMPRESSED",
            "H_POSTINTERACTION_2X2",
            "H_POSTINTERACTION_2X1",
        }:
            return state.h, {}
        return super().context_source_tensor_and_rope(state)


def main():
    phase11.OUTPUT = OUTPUT
    phase11.REPORT = REPORT
    phase11.VARIANTS = (
        ("A_FIXED_G_1152", "G"),
        ("B_POST_INTERACTION_1152", "H_POSTINTERACTION_COMPRESSED"),
        ("C_POST_INTERACTION_2048", "H_POSTINTERACTION_2X2"),
        ("D_POST_INTERACTION_4096", "H_POSTINTERACTION_2X1"),
        ("E_FULL_H_8192_ORACLE", "H"),
    )
    phase11.SAMPLER_CLASS = Phase12cSampler
    Phase12cSampler.probes = []
    phase11.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    probes = [p for p in Phase12cSampler.probes if isinstance(p, PostInteraction2x1Probe)]
    if len(probes) != 4:
        raise AssertionError(f"Expected four fresh 2x1 probes, got {len(probes)}.")
    hashes = {p.compression_records[0]["coordinate_provenance_sha256"] for p in probes}
    if len(hashes) != 1:
        raise AssertionError("2x1 coordinate provenance changed across intervals.")
    provenance = probes[0].coordinate_provenance
    flattened = [tuple(position) for pair in provenance for position in pair]
    if len(flattened) != 8192 or len(set(flattened)) != 8192:
        raise AssertionError("2x1 mapping omitted or duplicated full-H source positions.")
    report["phase12c_isolation"] = {
        "source_execution": "ordinary full accepted-H 64x128 source at each current sigma",
        "compression_boundary": "generated K after native full-H RoPE and generated V",
        "compression_rule": "nonoverlapping vertical 2x1 arithmetic mean",
        "anisotropy": "32x128 retains full horizontal sampling for the bridge/train scene",
        "consumer_grid": [32, 128], "consumer_tokens": 4096,
        "contributors_per_consumer": 2,
        "source_positions": 8192,
        "all_source_positions_used_exactly_once": True,
        "omitted_source_positions": 0, "duplicated_source_positions": 0,
        "coordinate_provenance_sha256": next(iter(hashes)),
        "coordinate_provenance": provenance,
        "fresh_probe_count": len(probes),
        "representation_scope": "one deterministic anisotropic 4096-token representation, not all representations",
        "production_changes": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
