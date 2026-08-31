"""Live four-step qualification for Candidate-3 arbitrary target geometry."""

from __future__ import annotations

import json
from pathlib import Path

from live_comfyui_candidate3_qualification import base_prompt, queue


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "live_comfyui_candidate3_geometry_results"

CASES = {
    "square_512x512": (512, 512, (32, 32), (24, 24), 1),
    "portrait_512x1024": (512, 1024, (64, 32), (48, 24), 3),
    "wide_1280x512": (1280, 512, (32, 80), (24, 60), 3),
}


def geometry_record(width: int, height: int, high_hw, global_hw, crops: int):
    scale_y = (high_hw[0] - 1.0) / (global_hw[0] - 1.0)
    scale_x = (high_hw[1] - 1.0) / (global_hw[1] - 1.0)
    return {
        "pixels": [height, width],
        "high_hw": list(high_hw),
        "global_hw": list(global_hw),
        "global_tokens": global_hw[0] * global_hw[1],
        "crop_count": crops,
        "local_tokens": crops * 32 * 32,
        "model_predictions_per_interval": 1 + crops,
        "global_rope": {"scale_y": scale_y, "scale_x": scale_x},
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, (width, height, high_hw, global_hw, crops) in CASES.items():
        prompt = base_prompt(f"BlueprintPhase5_{name}")
        prompt["6"]["inputs"].update({"width": width, "height": height})
        prompt["7"]["inputs"].update({"width": width, "height": height})
        result = queue(prompt)
        result["geometry"] = geometry_record(
            width, height, high_hw, global_hw, crops
        )
        result["qualified"] = bool(
            result["completed"]
            and result["binary_previews"] == 4
            and result["output_sha256"]
        )
        results[name] = result

    report = {
        "cases": results,
        "qualified": all(result["qualified"] for result in results.values()),
    }
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"qualified": report["qualified"], "report": str(path)}, indent=2))
    if not report["qualified"]:
        raise AssertionError("Arbitrary-geometry live qualification failed")


if __name__ == "__main__":
    main()
