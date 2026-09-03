"""Phase 10b: persistent native W with accepted-H coarse synchronization."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_persistent_native_local_trajectory as phase10


OUTPUT = ROOT / "experiments" / "flux2_candidate3_persistent_coarse_sync_results"
REPORT = OUTPUT / "report.json"
VARIANTS = (
    ("A_RECONSTRUCTED_W", "reconstructed"),
    ("B_PERSISTENT_W", "persistent"),
    ("C_PERSISTENT_COARSE_SYNC", "coarse_sync"),
)


def compare_outputs(left, right):
    result = {"accepted_H": {}, "assembled_x0_H": {}, "representative_W": {}}
    for key in sorted(set(left["accepted_H"]) & set(right["accepted_H"])):
        result["accepted_H"][str(key)] = phase9c.difference(
            left["accepted_H"][key], right["accepted_H"][key]
        )
    for key in sorted(set(left["assembled_x0_H"]) & set(right["assembled_x0_H"])):
        result["assembled_x0_H"][str(key)] = phase9c.difference(
            left["assembled_x0_H"][key], right["assembled_x0_H"][key]
        )
    for key in sorted(set(left["representative_W"]) & set(right["representative_W"])):
        result["representative_W"][str(key)] = phase9c.difference(
            left["representative_W"][key], right["representative_W"][key]
        )
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase10.OUTPUT = OUTPUT
    phase10.REPORT = REPORT
    results = {}
    outputs = {}
    for name, mode in VARIANTS:
        result, tensors = phase10.run_variant(name, mode, return_outputs=True)
        results[name] = result
        outputs[name] = tensors

    pairwise = {}
    names = list(results)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise[f"{left}_vs_{right}"] = compare_outputs(outputs[left], outputs[right])
    phase10.comparison_sheet(results)
    REPORT.write_text(json.dumps({
        "variants": results,
        "pairwise": pairwise,
        "terminal_W_diagnostic": (
            "C stores both unmodified terminal W_star and coarse-synchronized "
            "terminal W. Returned H is identical because neither diagnostic "
            "feeds terminal H acceptance."
        ),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "final_H": result["final_H"],
            "prediction_overlap": [x["prediction_overlap_rms"] for x in result["trajectory"]],
            "W_H_drift_max": [
                max((item["rms"] for item in step["W_H_drift"]), default=0.0)
                for step in result["trajectory"]
            ],
            "sync_correction_ratio_max": [
                max((item["correction_over_W_star_rms"] for item in step["synchronization"]), default=0.0)
                for step in result["trajectory"]
            ],
            "local_cuda_ms": result["work"]["local_cuda_ms"],
            "wall_seconds": result["runtime"]["sampling_wall_seconds"],
        }
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
