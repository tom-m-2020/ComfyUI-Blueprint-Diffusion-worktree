"""Phase 13: direct-4K global interaction versus post-interaction 4K."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

import flux2_candidate3_normalized_w_shared_context as phase11
import flux2_candidate3_postinteraction_2x1_capacity as phase12c


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_direct4k_source_results"
REPORT = OUTPUT / "report.json"


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def comparison(value, reference):
    left = value.detach().float()
    right = reference.detach().float()
    delta = left - right
    left_norm = left.square().mean().sqrt()
    right_norm = right.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        left.reshape(1, -1), right.reshape(1, -1), dim=1
    )
    return {
        "rms_difference": float(delta.square().mean().sqrt()),
        "max_abs_difference": float(delta.abs().max()),
        "direct_rms": float(left_norm),
        "postinteraction_rms": float(right_norm),
        "direct_over_postinteraction_norm_ratio": float(left_norm / right_norm),
        "cosine_similarity": float(cosine),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def restrict_state_2x1(value):
    if tuple(value.shape[-2:]) != (64, 128):
        raise AssertionError(f"Expected accepted H 64x128, got {tuple(value.shape)}")
    return value.reshape(value.shape[0], value.shape[1], 32, 2, 128).mean(dim=3)


def source_coordinate_provenance():
    return [
        {
            "direct_index": row * 128 + column,
            "direct_coordinate": [0.0, 2.0 * row + 0.5, float(column), 0.0],
            "source_coordinates": [
                [0.0, float(2 * row), float(column), 0.0],
                [0.0, float(2 * row + 1), float(column), 0.0],
            ],
        }
        for row in range(32)
        for column in range(128)
    ]


class Direct4KProbe(phase11.phase8d.CPUOffloadedContextProbe):
    consumer_tokens = 4096

    def __init__(self):
        super().__init__()
        self.reference_probe = None
        self.divergence_records = []

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        result = super().capture_global(q, k, v, pe, attn_mask, extra_options)
        key = self.block_key(extra_options)
        if self.reference_probe is None or key not in self.reference_probe.global_kv:
            raise AssertionError(f"Missing same-state post-interaction reference for {key}.")
        direct = self.global_kv[key]
        reference = self.reference_probe.global_kv[key]
        if direct["tokens"] != 4096 or reference["tokens"] != 4096:
            raise AssertionError(f"Unexpected 4K comparison cardinality at {key}.")
        self.divergence_records.append({
            "block_type": key[0], "block_index": key[1],
            "K": comparison(direct["k"], reference["k"]),
            "V": comparison(direct["v"], reference["v"]),
        })
        return result


class Phase13Sampler(phase11.SharedContextSampler):
    direct_probes = []

    def __init__(self, context_source, outer_probe):
        super().__init__(context_source, outer_probe)
        self.restriction_record = None

    def make_context_probe(self):
        if self.context_source == "H_POSTINTERACTION_2X1":
            return phase12c.PostInteraction2x1Probe()
        if self.context_source == "H_DIRECT_2X1":
            probe = Direct4KProbe()
            type(self).direct_probes.append(probe)
            return probe
        return super().make_context_probe()

    def context_source_tensor_and_rope(self, state):
        if self.context_source == "H_POSTINTERACTION_2X1":
            return state.h, {}
        if self.context_source == "H_DIRECT_2X1":
            direct = restrict_state_2x1(state.h)
            source_variance = float(state.h.detach().float().var(unbiased=False))
            direct_variance = float(direct.detach().float().var(unbiased=False))
            self.restriction_record = {
                "accepted_H_shape": list(state.h.shape),
                "direct_source_shape": list(direct.shape),
                "accepted_H_hash": tensor_hash(state.h),
                "direct_source_hash": tensor_hash(direct),
                "accepted_H_variance": source_variance,
                "direct_source_variance": direct_variance,
                "variance_ratio": direct_variance / source_variance,
                "restriction": "nonoverlapping vertical 2x1 arithmetic mean",
                "normalization": "none",
            }
            return direct, {"scale_y": 2.0, "shift_y": 0.5}
        return super().context_source_tensor_and_rope(state)

    def prepare_context_reference(
        self, model, state, sigma, model_options, ordinal, context_probe
    ):
        if self.context_source != "H_DIRECT_2X1":
            return None
        reference = phase12c.PostInteraction2x1Probe()
        before_hash = tensor_hash(state.h)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start.record()
        _ = model(
            state.h,
            sigma.expand(1),
            model_options=phase11.phase9c.merge_options(
                model_options, {}, reference, "capture",
                f"phase13_direct_reference_{ordinal}",
            ),
            seed=phase11.SEED,
        )
        end.record()
        torch.cuda.synchronize()
        if tensor_hash(state.h) != before_hash:
            raise RuntimeError("Phase 13 diagnostic reference mutated accepted H.")
        context_probe.reference_probe = reference
        return {
            "source": "same accepted H post-interaction 2x1 reference",
            "accepted_H_hash": before_hash,
            "sigma": float(sigma),
            "cuda_ms": float(start.elapsed_time(end)),
            "wall_seconds": time.perf_counter() - wall_started,
            "blocks": len(reference.capture_records),
            "consumer_tokens": 4096,
        }

    def augment_context_source_record(self, source_record, context_probe, reference_record):
        super().augment_context_source_record(source_record, context_probe, reference_record)
        if self.context_source != "H_DIRECT_2X1":
            return
        if len(context_probe.divergence_records) != 25:
            raise AssertionError("Direct/reference K/V comparison did not cover 25 blocks.")
        source_record["state_restriction"] = dict(self.restriction_record)
        source_record["per_block_direct_vs_postinteraction"] = list(
            context_probe.divergence_records
        )
        context_probe.reference_probe.global_kv.clear()
        context_probe.reference_probe = None


def main():
    phase11.OUTPUT = OUTPUT
    phase11.REPORT = REPORT
    phase11.VARIANTS = (
        ("A_POST_INTERACTION_4096", "H_POSTINTERACTION_2X1"),
        ("B_DIRECT_INTERACTION_4096", "H_DIRECT_2X1"),
        ("C_FULL_H_8192_ORACLE", "H"),
    )
    phase11.SAMPLER_CLASS = Phase13Sampler
    Phase13Sampler.direct_probes = []
    phase11.main()

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    direct = report["variants"]["B_DIRECT_INTERACTION_4096"]
    if len(Phase13Sampler.direct_probes) != 4:
        raise AssertionError("Direct source was not rebuilt exactly once per interval.")
    provenance = source_coordinate_provenance()
    flattened = [tuple(x) for item in provenance for x in item["source_coordinates"]]
    if len(flattened) != 8192 or len(set(flattened)) != 8192:
        raise AssertionError("Direct-state restriction omitted or duplicated H positions.")
    report["phase13_isolation"] = {
        "decisive_difference": {
            "A": "8192-token full-H interaction then 2x1 K/V aggregation",
            "B": "2x1 accepted-H state restriction then ordinary 4096-token interaction",
        },
        "direct_source_persistent": False,
        "direct_source_rebuilt_count": len(Phase13Sampler.direct_probes),
        "source_positions": 8192,
        "direct_positions": 4096,
        "all_H_positions_used_exactly_once": True,
        "omitted_positions": 0, "duplicated_positions": 0,
        "coordinate_policy": "row r at y=2r+0.5; x native 0..127",
        "coordinate_provenance": provenance,
        "variance_ratios": [
            item["state_restriction"]["variance_ratio"]
            for item in direct["context_generations"]
        ],
        "diagnostic_reference_cuda_ms": sum(
            item["diagnostic_reference"]["cuda_ms"]
            for item in direct["context_generations"]
        ),
        "production_changes": False,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
