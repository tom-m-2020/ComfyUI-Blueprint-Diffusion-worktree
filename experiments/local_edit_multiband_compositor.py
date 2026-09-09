"""Zero-diffusion fixed multiband Local Edit compositor discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import local_edit_composite_robustness as robustness
import local_edit_postdecode_composite as narrow


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "local_edit_multiband_compositor_results"
HIGH_SUPPORT = 24
MID_SUPPORT = 64
LOW_SUPPORT = 128
HIGH_BLUR_RADIUS = 2.0
LOW_BLUR_RADIUS = 8.0
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
BASELINE_NEIGHBORHOOD = 8


def inward_weight(support: int, width: int = 1024) -> np.ndarray:
    weight = np.zeros(width, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0.0, 1.0, support))
    weight[narrow.LEFT:narrow.LEFT + support] = ramp
    weight[narrow.RIGHT - support:narrow.RIGHT] = ramp[::-1]
    weight[narrow.LEFT + support:narrow.RIGHT - support] = 1.0
    return weight


def edge_extend_source(source: np.ndarray) -> np.ndarray:
    extended = source.copy()
    extended[:, :narrow.LEFT] = source[:, narrow.LEFT:narrow.LEFT + 1]
    extended[:, narrow.RIGHT:] = source[:, narrow.RIGHT - 1:narrow.RIGHT]
    return extended


def gaussian(image: np.ndarray, radius: float) -> np.ndarray:
    clipped = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    return np.asarray(Image.fromarray(clipped).filter(ImageFilter.GaussianBlur(radius)), dtype=np.float64)


def bands(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = image.astype(np.float64)
    blur_high = gaussian(base, HIGH_BLUR_RADIUS)
    blur_low = gaussian(base, LOW_BLUR_RADIUS)
    return base - blur_high, blur_high - blur_low, blur_low


def multiband_composite(source: np.ndarray, generated: np.ndarray) -> np.ndarray:
    source_bands = bands(edge_extend_source(source))
    generated_bands = bands(generated)
    weights = (
        inward_weight(HIGH_SUPPORT),
        inward_weight(MID_SUPPORT),
        inward_weight(LOW_SUPPORT),
    )
    output = generated.astype(np.float64).copy()
    for source_band, generated_band, weight in zip(source_bands, generated_bands, weights):
        output += weight[None, :, None] * (source_band - generated_band)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def luminance(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) @ LUMA / 255.0


def strip_profile(image: np.ndarray, side: str) -> dict:
    y = luminance(image)
    if side == "left":
        start, stop = narrow.LEFT, narrow.LEFT + LOW_SUPPORT
        outside = y[:, narrow.LEFT - BASELINE_NEIGHBORHOOD:narrow.LEFT].mean()
        inside = y[:, stop:stop + BASELINE_NEIGHBORHOOD].mean()
        baseline = np.linspace(outside, inside, LOW_SUPPORT + 2)[1:-1]
    else:
        start, stop = narrow.RIGHT - LOW_SUPPORT, narrow.RIGHT
        inside = y[:, start - BASELINE_NEIGHBORHOOD:start].mean()
        outside = y[:, narrow.RIGHT:narrow.RIGHT + BASELINE_NEIGHBORHOOD].mean()
        baseline = np.linspace(inside, outside, LOW_SUPPORT + 2)[1:-1]
    profile = y[:, start:stop].mean(axis=0)
    derivative = np.diff(profile)
    residual = profile - baseline
    return {
        "mean_luminance_by_column": [float(value) for value in profile],
        "derivative_by_column": [float(value) for value in derivative],
        "derivative_rms": float(np.sqrt(np.mean(derivative * derivative))),
        "derivative_max_abs": float(np.abs(derivative).max()),
        "low_frequency_residual_rms": float(np.sqrt(np.mean(residual * residual))),
        "maximum_low_frequency_deviation": float(np.abs(residual).max()),
    }


def contour_sharpness(image: np.ndarray, side: str) -> dict:
    y = luminance(image)
    if side == "left":
        region = y[:, narrow.LEFT - 2:narrow.LEFT + HIGH_SUPPORT + 2]
    else:
        region = y[:, narrow.RIGHT - HIGH_SUPPORT - 2:narrow.RIGHT + 2]
    gx = np.diff(region, axis=1)
    gy = np.diff(region, axis=0)
    values = np.concatenate((np.abs(gx).reshape(-1), np.abs(gy).reshape(-1)))
    return {
        "gradient_rms": float(np.sqrt(np.mean(values * values))),
        "gradient_p95": float(np.percentile(values, 95)),
        "gradient_p99": float(np.percentile(values, 99)),
    }


def metrics(image: np.ndarray, source: np.ndarray, generated: np.ndarray) -> dict:
    inner_left = narrow.LEFT + LOW_SUPPORT
    inner_right = narrow.RIGHT - LOW_SUPPORT
    transition = np.concatenate((image[:, narrow.LEFT:inner_left], image[:, inner_right:narrow.RIGHT]), axis=1)
    source_transition = np.concatenate((source[:, narrow.LEFT:inner_left], source[:, inner_right:narrow.RIGHT]), axis=1)
    exterior = np.concatenate((image[:, :narrow.LEFT], image[:, narrow.RIGHT:]), axis=1)
    generated_exterior = np.concatenate((generated[:, :narrow.LEFT], generated[:, narrow.RIGHT:]), axis=1)
    interior_exact = np.all(image[:, inner_left:inner_right] == source[:, inner_left:inner_right], axis=-1)
    return {
        "exact_source_interior_pixel_fraction": float(interior_exact.mean()),
        "generated_exterior_bit_exact": bool(np.array_equal(exterior, generated_exterior)),
        "generated_exterior_max_abs": int(np.abs(exterior.astype(np.int16) - generated_exterior.astype(np.int16)).max()),
        "transition_mae_vs_source": float(np.abs(transition.astype(np.float64) - source_transition).mean() / 255.0),
        "strip_luminance": {"left": strip_profile(image, "left"), "right": strip_profile(image, "right")},
        "contour_sharpness": {"left": contour_sharpness(image, "left"), "right": contour_sharpness(image, "right")},
    }


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
    hard = np.zeros(1024, dtype=np.float64)
    hard[narrow.LEFT:narrow.RIGHT] = 1.0
    narrow_weight = narrow.source_weight()
    report_cases = {}
    for case, config in robustness.CASES.items():
        source = narrow.load_rgb(config["source"])
        generated = narrow.load_rgb(config["generated"])
        arms = {
            "A_GENERATED": generated,
            "B_HARD": narrow.composite(source, generated, hard),
            "C_INSET_24PX": narrow.composite(source, generated, narrow_weight),
            "D_MULTIBAND": multiband_composite(source, generated),
        }
        arm_metrics = {}
        for arm, image in arms.items():
            Image.fromarray(image).save(OUTPUT / f"{case}_{arm}.png")
            arm_metrics[arm] = metrics(image, source, generated)
        make_sheet(case, arms)
        report_cases[case] = {"kind": config["kind"], "arms": arm_metrics}
    report = {
        "experiment": "Zero-diffusion fixed multiband compositor",
        "policy": {
            "gaussian_radii": [HIGH_BLUR_RADIUS, LOW_BLUR_RADIUS],
            "high_mid_low_support_pixels": [HIGH_SUPPORT, MID_SUPPORT, LOW_SUPPORT],
            "source_footprint": [narrow.LEFT, narrow.RIGHT - 1],
            "exact_source_interior": [narrow.LEFT + LOW_SUPPORT, narrow.RIGHT - LOW_SUPPORT - 1],
            "source_pyramid_boundary": "edge extended; no gray padded exterior contribution",
        },
        "cases": report_cases,
        "diffusion_or_vae_executed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "summary": {
            case: {
                arm: {
                    "transition_mae": data["transition_mae_vs_source"],
                    "left_residual": data["strip_luminance"]["left"]["low_frequency_residual_rms"],
                    "right_residual": data["strip_luminance"]["right"]["low_frequency_residual_rms"],
                    "left_p95": data["contour_sharpness"]["left"]["gradient_p95"],
                    "right_p95": data["contour_sharpness"]["right"]["gradient_p95"],
                } for arm, data in value["arms"].items() if arm in {"C_INSET_24PX", "D_MULTIBAND"}
            } for case, value in report_cases.items()
        },
    }, indent=2))


if __name__ == "__main__":
    run()
