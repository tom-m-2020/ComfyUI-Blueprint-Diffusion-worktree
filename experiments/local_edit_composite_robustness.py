"""Fixed 24 px post-decode compositing robustness discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import local_edit_postdecode_composite as policy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "local_edit_composite_robustness_results"
BRIDGE = ROOT / "experiments" / "local_edit_late_source_restoration_results"

CASES = {
    "rigid_bridge": {
        "kind": "rigid geometry crossing both boundaries",
        "source": BRIDGE / "PADDED_RESIZED.png",
        "generated": BRIDGE / "A_frozen_D_reference_FINAL.png",
        "epsilon_screen": "pass: one bridge, compatible deck/cable scale, no independent bridge",
    },
    "organic_tree": {
        "kind": "organic tree contour crossing right boundary",
        "source": OUTPUT / "organic_contour_SOURCE.png",
        "generated": OUTPUT / "organic_contour_A_GENERATED.png",
        "epsilon_screen": "pass: one continuous tree and one intact woman; no duplicate tree/person",
    },
    "photometric_desert": {
        "kind": "sky, horizon, sand, and shadow texture across both boundaries",
        "source": OUTPUT / "photometric_texture_SOURCE.png",
        "generated": OUTPUT / "photometric_texture_A_GENERATED.png",
        "epsilon_screen": "pass: one astronaut; continuous horizon and ground with no structural restart",
    },
}


def make_case_sheet(case: str, arms: dict[str, np.ndarray]) -> None:
    label_height = 30
    height, width = next(iter(arms.values())).shape[:2]
    canvas = Image.new("RGB", (width, len(arms) * (height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(arms.items()):
        y = index * (height + label_height)
        draw.text((8, y + 7), f"{case}: {name}", fill="black")
        canvas.paste(Image.fromarray(image), (0, y + label_height))
    canvas.save(OUTPUT / f"{case}_COMPARISON.png")


def run() -> None:
    hard_weight = np.zeros(1024, dtype=np.float64)
    hard_weight[policy.LEFT:policy.RIGHT] = 1.0
    inset_weight = policy.source_weight()
    report_cases = {}
    for case, config in CASES.items():
        source = policy.load_rgb(config["source"])
        generated = policy.load_rgb(config["generated"])
        if source.shape != (512, 1024, 3) or generated.shape != source.shape:
            raise ValueError(f"{case}: expected matching 1024x512 RGB images")
        arms = {
            "A_GENERATED": generated,
            "B_HARD": policy.composite(source, generated, hard_weight),
            "C_INSET_24PX": policy.composite(source, generated, inset_weight),
        }
        arm_metrics = {}
        for arm, image in arms.items():
            Image.fromarray(image).save(OUTPUT / f"{case}_{arm}.png")
            arm_metrics[arm] = policy.metrics(image, source, generated)
        make_case_sheet(case, arms)
        report_cases[case] = {
            "kind": config["kind"],
            "source": str(config["source"]),
            "generated": str(config["generated"]),
            "epsilon_screen": config["epsilon_screen"],
            "arms": arm_metrics,
        }
    report = {
        "experiment": "Fixed-policy post-decode compositing robustness",
        "policy": {
            "source_footprint": [policy.LEFT, policy.RIGHT - 1],
            "transition_width_pixels_per_boundary": policy.TRANSITION_WIDTH,
            "transition": [[policy.LEFT, policy.LEFT + policy.TRANSITION_WIDTH - 1],
                           [policy.RIGHT - policy.TRANSITION_WIDTH, policy.RIGHT - 1]],
            "exact_source_interior": [policy.LEFT + policy.TRANSITION_WIDTH,
                                      policy.RIGHT - policy.TRANSITION_WIDTH - 1],
            "falloff": "unchanged raised cosine",
        },
        "cases": report_cases,
        "diffusion_runs": {
            "new_eligible": 2,
            "excluded_pre_composite_generation_attempts": 2,
            "compositing_stage": 0,
        },
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
