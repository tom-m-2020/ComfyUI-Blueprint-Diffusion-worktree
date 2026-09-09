"""Zero-diffusion post-decode compositing discriminator for frozen Clown arm D."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "experiments" / "local_edit_late_source_restoration_results"
OUTPUT = ROOT / "experiments" / "local_edit_postdecode_composite_results"
SOURCE_PATH = INPUT / "PADDED_RESIZED.png"
GENERATED_PATH = INPUT / "A_frozen_D_reference_FINAL.png"
LEFT = 256
RIGHT = 768
TRANSITION_WIDTH = 24


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def source_weight(width: int = 1024) -> np.ndarray:
    """Raised-cosine source weight; transition lies strictly inside [LEFT, RIGHT)."""
    weight = np.zeros(width, dtype=np.float64)
    inner_left = LEFT + TRANSITION_WIDTH
    inner_right = RIGHT - TRANSITION_WIDTH
    weight[inner_left:inner_right] = 1.0
    t = np.linspace(0.0, 1.0, TRANSITION_WIDTH, dtype=np.float64)
    ramp = 0.5 - 0.5 * np.cos(np.pi * t)
    weight[LEFT:inner_left] = ramp
    weight[inner_right:RIGHT] = ramp[::-1]
    return weight


def composite(source: np.ndarray, generated: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mixed = generated.astype(np.float64) * (1.0 - weight[None, :, None])
    mixed += source.astype(np.float64) * weight[None, :, None]
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value.astype(np.float64)))))


def metrics(image: np.ndarray, source: np.ndarray, generated: np.ndarray) -> dict:
    inner_left = LEFT + TRANSITION_WIDTH
    inner_right = RIGHT - TRANSITION_WIDTH
    source_interior = image[:, inner_left:inner_right]
    target_interior = source[:, inner_left:inner_right]
    transition = np.concatenate((image[:, LEFT:inner_left], image[:, inner_right:RIGHT]), axis=1)
    target_transition = np.concatenate((source[:, LEFT:inner_left], source[:, inner_right:RIGHT]), axis=1)
    footprint = image[:, LEFT:RIGHT]
    target_footprint = source[:, LEFT:RIGHT]
    exact_interior = np.all(source_interior == target_interior, axis=-1)
    exact_footprint = np.all(footprint == target_footprint, axis=-1)

    def boundary(name: str, boundary: int, exterior_direction: int) -> dict:
        seam = image[:, boundary].astype(np.float64) - image[:, boundary - 1].astype(np.float64)
        if exterior_direction < 0:
            adjacent = image[:, boundary - 1].astype(np.float64) - image[:, boundary - 2].astype(np.float64)
        else:
            adjacent = image[:, boundary + 1].astype(np.float64) - image[:, boundary].astype(np.float64)
        return {
            f"{name}_seam_gradient_rms": rms(seam) / 255.0,
            f"{name}_first_derivative_discontinuity_rms": rms(seam - adjacent) / 255.0,
        }

    outside = np.ones(image.shape[:2], dtype=bool)
    outside[:, LEFT:RIGHT] = False
    # P2 requires generated identity outside the transition, not within it.
    unchanged_generated = outside

    result = {
        "source_interior_mae": float(np.abs(source_interior.astype(np.float64) - target_interior).mean() / 255.0),
        "transition_strip_mae_vs_source": float(np.abs(transition.astype(np.float64) - target_transition).mean() / 255.0),
        "source_interior_exact_pixel_fraction": float(exact_interior.mean()),
        "source_footprint_exact_pixel_fraction": float(exact_footprint.mean()),
        "generated_exterior_bit_exact": bool(np.array_equal(image[unchanged_generated], generated[unchanged_generated])),
        "generated_exterior_max_abs": int(np.abs(image[unchanged_generated].astype(np.int16) - generated[unchanged_generated].astype(np.int16)).max()),
    }
    result.update(boundary("left", LEFT, -1))
    result.update(boundary("right", RIGHT, 1))
    return result


def make_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    label_height = 32
    canvas = Image.new("RGB", (items[0][1].shape[1], len(items) * (items[0][1].shape[0] + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(items):
        y = index * (array.shape[0] + label_height)
        draw.text((8, y + 8), label, fill="black")
        canvas.paste(Image.fromarray(array), (0, y + label_height))
    canvas.save(path)


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = load_rgb(SOURCE_PATH)
    generated = load_rgb(GENERATED_PATH)
    if source.shape != generated.shape or source.shape != (512, 1024, 3):
        raise ValueError(f"Expected matching 1024x512 RGB inputs, got {source.shape} and {generated.shape}")

    hard_weight = np.zeros(1024, dtype=np.float64)
    hard_weight[LEFT:RIGHT] = 1.0
    feather_weight = source_weight()
    arms = {
        "A_frozen_D_decode": generated,
        "B_hard_pixel_composite": composite(source, generated, hard_weight),
        "C_inset_24px_composite": composite(source, generated, feather_weight),
    }

    report_arms = {}
    for name, image in arms.items():
        Image.fromarray(image).save(OUTPUT / f"{name}.png")
        report_arms[name] = metrics(image, source, generated)
    Image.fromarray(np.rint(feather_weight[None, :] * 255).astype(np.uint8).repeat(64, axis=0)).save(
        OUTPUT / "C_SOURCE_WEIGHT.png"
    )
    make_sheet(list(arms.items()), OUTPUT / "COMPARISON.png")

    report = {
        "experiment": "Zero-diffusion post-decode source compositing",
        "source": str(SOURCE_PATH),
        "frozen_generated": str(GENERATED_PATH),
        "source_footprint": {"left_inclusive": LEFT, "right_exclusive": RIGHT},
        "transition": {
            "width_pixels_per_boundary": TRANSITION_WIDTH,
            "left_columns": [LEFT, LEFT + TRANSITION_WIDTH - 1],
            "right_columns": [RIGHT - TRANSITION_WIDTH, RIGHT - 1],
            "exact_source_interior": [LEFT + TRANSITION_WIDTH, RIGHT - TRANSITION_WIDTH - 1],
            "falloff": "raised cosine, source weight 0 at nominal footprint boundary and 1 at exact-interior edge",
        },
        "arms": report_arms,
        "diffusion_or_vae_executed": False,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
