"""Zero-diffusion fixed gradient-domain Local Edit compositor discriminator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized

import local_edit_composite_robustness as robustness
import local_edit_multiband_compositor as prior
import local_edit_postdecode_composite as narrow


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "local_edit_gradient_compositor_results"
SUPPORT = 64


def _factor_poisson(height: int, width: int):
    count = height * width
    matrix = lil_matrix((count, count), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            index = y * width + x
            degree = 2 + int(y > 0) + int(y + 1 < height)
            matrix[index, index] = degree
            if x > 0:
                matrix[index, index - 1] = -1.0
            if x + 1 < width:
                matrix[index, index + 1] = -1.0
            if y > 0:
                matrix[index, index - width] = -1.0
            if y + 1 < height:
                matrix[index, index + width] = -1.0
    return factorized(matrix.tocsc())


def _mixed_gradient(source_p: np.ndarray, source_q: np.ndarray,
                    generated_p: np.ndarray, generated_q: np.ndarray) -> np.ndarray:
    source_gradient = np.asarray(source_p, dtype=np.float64) - np.asarray(source_q, dtype=np.float64)
    generated_gradient = np.asarray(generated_p, dtype=np.float64) - np.asarray(generated_q, dtype=np.float64)
    return np.where(np.abs(source_gradient) >= np.abs(generated_gradient), source_gradient, generated_gradient)


def _solve_side(source: np.ndarray, generated: np.ndarray, start: int, stop: int,
                solve) -> np.ndarray:
    """Solve [start, stop), with lateral generated/source Dirichlet values."""
    height = source.shape[0]
    width = stop - start
    source_guidance = source.astype(np.float64).copy()
    generated_guidance = generated.astype(np.float64)
    # The source has no ownership outside its footprint. Edge extension prevents
    # padded exterior values from becoming an artificial source boundary gradient.
    source_guidance[:, :narrow.LEFT] = source[:, narrow.LEFT:narrow.LEFT + 1]
    source_guidance[:, narrow.RIGHT:] = source[:, narrow.RIGHT - 1:narrow.RIGHT]
    output = np.empty((height, width, 3), dtype=np.float64)

    for channel in range(3):
        rhs = np.zeros((height, width), dtype=np.float64)
        for y in range(height):
            for local_x in range(width):
                x = start + local_x
                p_source = source_guidance[y, x, channel]
                p_generated = generated_guidance[y, x, channel]
                for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                    ny, nx = y + dy, x + dx
                    if ny < 0 or ny >= height:
                        continue
                    gradient = _mixed_gradient(
                        p_source, source_guidance[ny, nx, channel],
                        p_generated, generated_guidance[ny, nx, channel],
                    )
                    rhs[y, local_x] += gradient
                    if nx < start:
                        rhs[y, local_x] += generated[ny, nx, channel]
                    elif nx >= stop:
                        rhs[y, local_x] += source[ny, nx, channel]
        output[:, :, channel] = solve(rhs.reshape(-1)).reshape(height, width)
    return output


def gradient_composite(source: np.ndarray, generated: np.ndarray) -> np.ndarray:
    if source.shape != generated.shape or source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected matching RGB inputs, got {source.shape} and {generated.shape}")
    solve = _factor_poisson(source.shape[0], SUPPORT)
    result = generated.astype(np.float64).copy()
    left_start, left_stop = narrow.LEFT, narrow.LEFT + SUPPORT
    right_start, right_stop = narrow.RIGHT - SUPPORT, narrow.RIGHT
    result[:, left_start:left_stop] = _solve_side(source, generated, left_start, left_stop, solve)
    result[:, right_start:right_stop] = _solve_side(source, generated, right_start, right_stop, solve)
    result[:, left_stop:right_start] = source[:, left_stop:right_start]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def metrics(image: np.ndarray, source: np.ndarray, generated: np.ndarray) -> dict:
    inner_left = narrow.LEFT + SUPPORT
    inner_right = narrow.RIGHT - SUPPORT
    transition = np.concatenate((image[:, narrow.LEFT:inner_left], image[:, inner_right:narrow.RIGHT]), axis=1)
    target = np.concatenate((source[:, narrow.LEFT:inner_left], source[:, inner_right:narrow.RIGHT]), axis=1)
    exterior = np.concatenate((image[:, :narrow.LEFT], image[:, narrow.RIGHT:]), axis=1)
    generated_exterior = np.concatenate((generated[:, :narrow.LEFT], generated[:, narrow.RIGHT:]), axis=1)
    exact = np.all(image[:, inner_left:inner_right] == source[:, inner_left:inner_right], axis=-1)
    return {
        "exact_source_interior_pixel_fraction": float(exact.mean()),
        "generated_exterior_bit_exact": bool(np.array_equal(exterior, generated_exterior)),
        "generated_exterior_max_abs": int(np.abs(exterior.astype(np.int16) - generated_exterior.astype(np.int16)).max()),
        "transition_mae_vs_source": float(np.abs(transition.astype(np.float64) - target).mean() / 255.0),
        "strip_luminance": {"left": prior.strip_profile(image, "left"), "right": prior.strip_profile(image, "right")},
        "contour_sharpness": {"left": prior.contour_sharpness(image, "left"), "right": prior.contour_sharpness(image, "right")},
        "nominal_boundary": narrow.metrics(image, source, generated),
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
    report_cases = {}
    for case, config in robustness.CASES.items():
        source = narrow.load_rgb(config["source"])
        generated = narrow.load_rgb(config["generated"])
        arms = {
            "A_GENERATED": generated,
            "B_HARD": narrow.composite(source, generated, hard),
            "C_INSET_24PX": narrow.composite(source, generated, narrow.source_weight()),
            "D_GRADIENT": gradient_composite(source, generated),
        }
        arm_metrics = {}
        for name, image in arms.items():
            Image.fromarray(image).save(OUTPUT / f"{case}_{name}.png")
            arm_metrics[name] = metrics(image, source, generated)
        make_sheet(case, arms)
        report_cases[case] = {"kind": config["kind"], "arms": arm_metrics}
    report = {
        "experiment": "Zero-diffusion fixed gradient-domain compositor",
        "policy": {
            "support_pixels_per_boundary": SUPPORT,
            "gradient_rule": "per-channel larger absolute source/generated forward difference",
            "lateral_dirichlet": "generated exterior and exact source interior",
            "image_edge_boundary": "zero normal derivative",
            "source_outside_footprint_for_guidance": "edge extended",
            "source_footprint": [narrow.LEFT, narrow.RIGHT - 1],
            "exact_source_interior": [narrow.LEFT + SUPPORT, narrow.RIGHT - SUPPORT - 1],
        },
        "cases": report_cases,
        "diffusion_or_vae_executed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "summary": {
            case: {name: {
                "transition_mae": arm["transition_mae_vs_source"],
                "left_residual": arm["strip_luminance"]["left"]["low_frequency_residual_rms"],
                "right_residual": arm["strip_luminance"]["right"]["low_frequency_residual_rms"],
                "left_p95": arm["contour_sharpness"]["left"]["gradient_p95"],
                "right_p95": arm["contour_sharpness"]["right"]["gradient_p95"],
            } for name, arm in data["arms"].items() if name in {"C_INSET_24PX", "D_GRADIENT"}}
            for case, data in report_cases.items()
        },
    }, indent=2))


if __name__ == "__main__":
    run()
