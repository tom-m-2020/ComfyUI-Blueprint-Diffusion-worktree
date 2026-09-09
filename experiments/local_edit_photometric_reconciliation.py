"""Zero-diffusion affine photometric reconciliation for the fixed compositor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import local_edit_composite_robustness as robustness
import local_edit_postdecode_composite as policy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "local_edit_photometric_reconciliation_results"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
BASELINE_NEIGHBORHOOD = 8
MIN_SCALE = 0.5
MAX_SCALE = 2.0


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb.astype(np.float64) @ LUMA


def robust_luminance_affine(source: np.ndarray, generated: np.ndarray) -> tuple[float, float]:
    source_y = luminance(source).reshape(-1)
    generated_y = luminance(generated).reshape(-1)
    source_median = float(np.median(source_y))
    generated_median = float(np.median(generated_y))
    source_mad = float(np.median(np.abs(source_y - source_median)))
    generated_mad = float(np.median(np.abs(generated_y - generated_median)))
    if generated_mad < 1e-6:
        scale = 1.0
    else:
        scale = float(np.clip(source_mad / generated_mad, MIN_SCALE, MAX_SCALE))
    offset = source_median - scale * generated_median
    return scale, offset


def reconcile_generated_luminance(
    source: np.ndarray, generated: np.ndarray, source_weight: np.ndarray
) -> tuple[np.ndarray, dict]:
    corrected = generated.astype(np.float64).copy()
    transforms = {}
    strips = {
        "left": (policy.LEFT, policy.LEFT + policy.TRANSITION_WIDTH),
        "right": (policy.RIGHT - policy.TRANSITION_WIDTH, policy.RIGHT),
    }
    for name, (start, stop) in strips.items():
        source_strip = source[:, start:stop]
        generated_strip = generated[:, start:stop]
        scale, offset = robust_luminance_affine(source_strip, generated_strip)
        generated_y = luminance(generated_strip)
        target_y = scale * generated_y + offset
        correction_strength = source_weight[start:stop][None, :]
        delta = correction_strength * (target_y - generated_y)
        corrected[:, start:stop] = np.clip(
            generated_strip.astype(np.float64) + delta[..., None], 0, 255
        )
        transforms[name] = {
            "scale": scale,
            "offset_8bit_luminance": offset,
            "maximum_applied_luminance_delta_8bit": float(np.abs(delta).max()),
        }
    return corrected, transforms


def reconciled_composite(source: np.ndarray, generated: np.ndarray) -> tuple[np.ndarray, dict]:
    weight = policy.source_weight()
    corrected, transforms = reconcile_generated_luminance(source, generated, weight)
    mixed = corrected * (1.0 - weight[None, :, None])
    mixed += source.astype(np.float64) * weight[None, :, None]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8), transforms


def strip_profile(image: np.ndarray, boundary: int, side: str) -> dict:
    if side == "left":
        start = boundary
        stop = boundary + policy.TRANSITION_WIDTH
        outside = luminance(image[:, boundary - BASELINE_NEIGHBORHOOD:boundary]).mean()
        inside = luminance(image[:, stop:stop + BASELINE_NEIGHBORHOOD]).mean()
    else:
        start = boundary - policy.TRANSITION_WIDTH
        stop = boundary
        inside = luminance(image[:, start - BASELINE_NEIGHBORHOOD:start]).mean()
        outside = luminance(image[:, boundary:boundary + BASELINE_NEIGHBORHOOD]).mean()
    profile = luminance(image[:, start:stop]).mean(axis=0) / 255.0
    if side == "left":
        baseline = np.linspace(outside, inside, policy.TRANSITION_WIDTH + 2)[1:-1] / 255.0
    else:
        baseline = np.linspace(inside, outside, policy.TRANSITION_WIDTH + 2)[1:-1] / 255.0
    residual = profile - baseline
    derivative = np.diff(profile)
    return {
        "mean_luminance_by_column": [float(v) for v in profile],
        "derivative_by_column": [float(v) for v in derivative],
        "derivative_rms": float(np.sqrt(np.mean(derivative * derivative))),
        "derivative_max_abs": float(np.abs(derivative).max()),
        "smooth_baseline_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "maximum_low_frequency_deviation": float(np.abs(residual).max()),
    }


def extended_metrics(image: np.ndarray, source: np.ndarray, generated: np.ndarray) -> dict:
    result = policy.metrics(image, source, generated)
    result["luminance_strips"] = {
        "left": strip_profile(image, policy.LEFT, "left"),
        "right": strip_profile(image, policy.RIGHT, "right"),
    }
    return result


def make_sheet(case: str, arms: dict[str, np.ndarray]) -> None:
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
    OUTPUT.mkdir(parents=True, exist_ok=True)
    hard_weight = np.zeros(1024, dtype=np.float64)
    hard_weight[policy.LEFT:policy.RIGHT] = 1.0
    fixed_weight = policy.source_weight()
    cases = {}
    for case, config in robustness.CASES.items():
        source = policy.load_rgb(config["source"])
        generated = policy.load_rgb(config["generated"])
        reconciled, transforms = reconciled_composite(source, generated)
        arms = {
            "A_GENERATED": generated,
            "B_HARD": policy.composite(source, generated, hard_weight),
            "C_INSET_24PX": policy.composite(source, generated, fixed_weight),
            "D_AFFINE_24PX": reconciled,
        }
        arm_metrics = {}
        for arm, image in arms.items():
            Image.fromarray(image).save(OUTPUT / f"{case}_{arm}.png")
            arm_metrics[arm] = extended_metrics(image, source, generated)
        make_sheet(case, arms)
        cases[case] = {
            "kind": config["kind"],
            "transforms": transforms,
            "arms": arm_metrics,
        }
    report = {
        "experiment": "Zero-diffusion fixed affine photometric reconciliation",
        "policy": {
            "ownership": "unchanged 24 px inward raised-cosine composite",
            "transform": "per-boundary robust luminance affine from corresponding transition pixels",
            "scale_clip": [MIN_SCALE, MAX_SCALE],
            "application": "luminance delta applied to generated contribution with source-weight strength",
            "smooth_baseline_neighborhood_pixels": BASELINE_NEIGHBORHOOD,
        },
        "cases": cases,
        "diffusion_or_vae_executed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "transforms": {case: value["transforms"] for case, value in cases.items()},
        "strip_metrics": {
            case: {
                arm: data["luminance_strips"]
                for arm, data in value["arms"].items() if arm in {"C_INSET_24PX", "D_AFFINE_24PX"}
            } for case, value in cases.items()
        },
    }, indent=2))


if __name__ == "__main__":
    run()
