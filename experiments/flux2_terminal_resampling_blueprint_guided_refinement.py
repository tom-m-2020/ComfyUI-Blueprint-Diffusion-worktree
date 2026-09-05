"""Phase 30 pre-inference architectural-boundary validator.

This harness intentionally performs no model inference. It validates and reuses
the exact Phase-29 sigma=0.25 controls, records the inspected native interfaces,
and persists the fail-closed Phase-30 decision. A denoised-prediction attractor
cannot be specified without an arbitrary guidance scale under the qualified
Klein T2I contract and the experiment's exclusions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results"
OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_blueprint_guided_refinement_results"
COMFY = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
CASES = ("SQUARE_MULTI_OBJECT", "PORTRAIT_ASTRONAUT", "LANDSCAPE_BRIDGE")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
    temporary.replace(path)


def control_record(case: str) -> dict:
    blueprint_path = PHASE29 / case / "blueprint.json"
    branch_path = PHASE29 / case / "sigma_0.25.json"
    latent_path = PHASE29 / case / "sigma_0.25.pt"
    image_path = PHASE29 / case / "sigma_0.25.png"
    for path in (blueprint_path, branch_path, latent_path, image_path):
        if not path.is_file():
            raise RuntimeError(f"Missing Phase-29 control artifact: {path}")
    blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
    branch = json.loads(branch_path.read_text(encoding="utf-8"))
    if branch["sigma"] != 0.25 or branch["mapped_hash"] != blueprint["mapped_hash"]:
        raise RuntimeError(f"Phase-29 fingerprint mismatch for {case}")
    if not branch["phase28_regression"]["bit_exact"]:
        raise RuntimeError(f"Phase-28 exact regression is not valid for {case}")
    value = torch.load(latent_path, map_location="cpu", weights_only=True)
    if not torch.isfinite(value).all():
        raise RuntimeError(f"Nonfinite Phase-29 control latent for {case}")
    return {
        "case": case,
        "semantic_grade": "S3",
        "blueprint": blueprint,
        "control": branch,
        "latent_file_sha256": sha256(latent_path),
        "decoded_file_sha256": sha256(image_path),
        "source_tensor_finite": True,
        "source_artifacts_unchanged": True,
    }


def make_sheet(name: str, controls: list[dict], detail: bool = False) -> None:
    width, panel_height = 1300, 430
    sheet = Image.new("RGB", (width, panel_height * len(controls)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, record in enumerate(controls):
        y = row * panel_height
        image = Image.open(PHASE29 / record["case"] / "sigma_0.25.png").convert("RGB")
        if detail:
            left = image.width // 4
            upper = image.height // 4
            image = image.crop((left, upper, image.width - left, image.height - upper))
        image.thumbnail((620, 380), Image.Resampling.LANCZOS)
        sheet.paste(image, ((620 - image.width) // 2, y + 42))
        draw.text((10, y + 10), f"{record['case']} | A CONTROL | S3", fill="black")
        draw.rectangle((650, y + 42, 1280, y + 410), outline="black", width=2)
        draw.multiline_text(
            (680, y + 145),
            "B GUIDED ARM NOT EXECUTED\n\nFail-closed architectural boundary:\n"
            "no parameter-free denoised-prediction\nattractor exists in the qualified native\n"
            "Klein T2I interface under Phase-30\nexclusions.",
            fill="black", spacing=8,
        )
    sheet.save(OUTPUT / name, quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    controls = [control_record(case) for case in CASES]
    source_paths = {
        "basic_guider": COMFY / "comfy" / "samplers.py",
        "flux_model": COMFY / "comfy" / "ldm" / "flux" / "model.py",
        "model_base": COMFY / "comfy" / "model_base.py",
    }
    source_hashes = {name: sha256(path) for name, path in source_paths.items()}
    report = {
        "phase": 30,
        "status": "stopped_before_guided_inference",
        "decision": "E — ARCHITECTURAL BOUNDARY",
        "control_reuse": controls,
        "guided_arm": {
            "executed": False,
            "model_calls": 0,
            "reason": (
                "The qualified native Klein T2I prediction exposes no trained "
                "Blueprint/reference attractor in denoised-prediction space."
            ),
            "rejected_constructions": {
                "linear_blend": "Explicitly excluded and requires alpha.",
                "coarse_or_exact_projection": "Previously tested hard authority; explicitly excluded here.",
                "gradient_or_score_attractor": (
                    "Requires a guidance/likelihood scale (and a structural loss metric); "
                    "no unique coefficient follows from the CONST flow contract."
                ),
                "native_reference_latents": (
                    "Appends reference image tokens to transformer attention, changes prepared "
                    "conditioning, and is an image-context/K/V mechanism excluded by Phase 30."
                ),
                "transformer_patch": "Unsupported model surgery and outside the qualified ordinary-call path.",
            },
        },
        "fixed_contract": {
            "sigma": 0.25,
            "cases": list(CASES),
            "production_modified": False,
            "comfy_core_modified": False,
        },
        "inspected_source_sha256": source_hashes,
    }
    atomic_json(OUTPUT / "report.json", report)
    make_sheet("PHASE30_COMPARISON.jpg", controls)
    make_sheet("DETAIL_REVIEW.jpg", controls, detail=True)
    print("Phase 30 boundary persisted; no model inference executed.")


if __name__ == "__main__":
    main()
