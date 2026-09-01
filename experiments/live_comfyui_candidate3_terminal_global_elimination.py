"""Live ComfyUI qualification after terminal global-forward elimination."""

from __future__ import annotations

import json
from pathlib import Path

from live_comfyui_candidate3_qualification import base_prompt, queue, request_json


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "live_comfyui_candidate3_terminal_global_elimination_results"
CASES = {
    "h64x128": (2048, 1024),
    "h48x96": (1536, 768),
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    object_info = request_json("/object_info/BlueprintCandidate3EulerSampler")
    results = {}
    for name, (width, height) in CASES.items():
        runs = []
        for repeat in range(2):
            prompt = base_prompt(f"BlueprintPhase6j_{name}_r{repeat}")
            prompt["6"]["inputs"].update({"width": width, "height": height})
            prompt["7"]["inputs"].update({"width": width, "height": height})
            runs.append(queue(prompt))
        hashes = [run["output_sha256"] for run in runs]
        results[name] = {
            "pixels_wh": [width, height],
            "H_shape": [height // 16, width // 16],
            "runs": runs,
            "deterministic_hash": hashes[0] == hashes[1],
            "qualified": bool(
                all(run["completed"] for run in runs)
                and runs[0]["binary_previews"] == 4
                and runs[1]["binary_previews"] in (0, 4)
                and hashes[0] == hashes[1]
            ),
        }
    report = {
        "node_registered": "BlueprintCandidate3EulerSampler" in object_info,
        "normal_workflow": [
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
        "cases": {
            name: {
                "qualified": value["qualified"],
                "previews": [run["binary_previews"] for run in value["runs"]],
                "hashes": [run["output_sha256"] for run in value["runs"]],
            }
            for name, value in results.items()
        },
        "report": str(path),
    }, indent=2))
    if not report["qualified"] or not report["node_registered"]:
        raise AssertionError("Phase-6j live qualification failed")


if __name__ == "__main__":
    main()
