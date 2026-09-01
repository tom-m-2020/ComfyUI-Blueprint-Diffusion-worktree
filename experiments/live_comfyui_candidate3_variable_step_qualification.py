"""Fresh live ComfyUI qualification for production variable-step support."""

from __future__ import annotations

import json
from pathlib import Path

from live_comfyui_candidate3_qualification import base_prompt, queue, request_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "live_comfyui_candidate3_variable_step_results"
GEOMETRIES = {
    "h64x128": (2048, 1024),
    "h48x96": (1536, 768),
}
STEPS = (4, 8, 20)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    object_info = request_json("/object_info/BlueprintCandidate3EulerSampler")
    results = {}
    for geometry, (width, height) in GEOMETRIES.items():
        for steps in STEPS:
            name = f"{geometry}_{steps:02d}_steps"
            runs = []
            for repeat in range(2):
                prompt = base_prompt(f"BlueprintPhase7b_{name}_r{repeat}")
                prompt["6"]["inputs"].update({
                    "width": width, "height": height, "steps": steps,
                })
                prompt["7"]["inputs"].update({"width": width, "height": height})
                runs.append(queue(prompt))
            hashes = [run["output_sha256"] for run in runs]
            results[name] = {
                "pixels_wh": [width, height], "H_shape": [height // 16, width // 16],
                "steps": steps, "runs": runs,
                "qualified": bool(
                    all(run["completed"] for run in runs)
                    and runs[0]["binary_previews"] == steps
                    and runs[1]["binary_previews"] in (0, steps)
                    and hashes[0] == hashes[1]
                ),
            }
    description = object_info["BlueprintCandidate3EulerSampler"]["description"]
    report = {
        "node_registered": "BlueprintCandidate3EulerSampler" in object_info,
        "node_description": description,
        "description_not_four_step_specific": "four" not in description.lower(),
        "workflow": [
            "BasicGuider", "BlueprintCandidate3EulerSampler",
            "SamplerCustomAdvanced", "VAEDecode", "SaveImage",
        ],
        "cases": results,
        "qualified": all(item["qualified"] for item in results.values()),
    }
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "qualified": report["qualified"],
        "node_registered": report["node_registered"],
        "description_not_four_step_specific": report["description_not_four_step_specific"],
        "cases": {
            name: {
                "qualified": item["qualified"],
                "previews": [run["binary_previews"] for run in item["runs"]],
                "hashes": [run["output_sha256"] for run in item["runs"]],
            }
            for name, item in results.items()
        },
        "report": str(path),
    }, indent=2))
    if not (
        report["qualified"] and report["node_registered"]
        and report["description_not_four_step_specific"]
    ):
        raise AssertionError("Phase-7b live qualification failed")


if __name__ == "__main__":
    main()
