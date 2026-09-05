"""Phase 32 fail-closed restriction-nullspace construction validator."""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PHASE29 = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results"
OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_restriction_nullspace_results"
sys.path[:0] = [str(COMFY_ROOT), str(PACKAGE.parent)]
spec = importlib.util.spec_from_file_location(
    "blueprint_diffusion", PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
module = importlib.util.module_from_spec(spec)
sys.modules["blueprint_diffusion"] = module
spec.loader.exec_module(module)

from blueprint_diffusion.terminal_resampling import (  # noqa: E402
    TerminalResamplingGeometry,
    lift_region,
    region_noise,
    tensor_hash,
)

CASES = (
    {"name": "SQUARE_MULTI_OBJECT", "destination_hw": (128, 128), "seed": 20260921},
    {"name": "PORTRAIT_ASTRONAUT", "destination_hw": (256, 128), "seed": 20260922},
    {"name": "LANDSCAPE_BRIDGE", "destination_hw": (128, 192), "seed": 20260923},
)


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
    temporary.replace(path)


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in CASES:
        source = PHASE29 / case["name"]
        blueprint = torch.load(source / "blueprint.pt", map_location="cpu", weights_only=True)
        blueprint_record = json.loads((source / "blueprint.json").read_text(encoding="utf-8"))
        control = json.loads((source / "sigma_0.25.json").read_text(encoding="utf-8"))
        mapped = blueprint["mapped"].float()
        if tensor_hash(mapped) != blueprint_record["mapped_hash"]:
            raise RuntimeError(f"Blueprint fingerprint mismatch: {case['name']}")
        if control["mapped_hash"] != blueprint_record["mapped_hash"]:
            raise RuntimeError(f"Control mapping mismatch: {case['name']}")
        if not control["phase28_regression"]["bit_exact"]:
            raise RuntimeError(f"Control is not Phase-28 exact: {case['name']}")
        geometry = TerminalResamplingGeometry.for_destination(case["destination_hw"])
        regions = []
        for region, control_region in zip(geometry.regions(), control["regions"], strict=True):
            if region.index != control_region["index"]:
                raise RuntimeError(f"Region order mismatch: {case['name']}")
            crop = mapped[:, :, region.y:region.y2, region.x:region.x2]
            anchor = lift_region(crop)
            noise = region_noise(case["seed"], region, device="cpu", dtype=torch.float32)
            if tensor_hash(noise) != control_region["noise_hash"]:
                raise RuntimeError(f"Noise provenance mismatch: {case['name']} region {region.index}")
            coarse_noise = lift_region(F.avg_pool2d(noise, 2, 2))
            noise_null = noise - coarse_noise
            restricted_null = F.avg_pool2d(noise_null, 2, 2)
            regions.append({
                "index": region.index,
                "blueprint_crop_hash": tensor_hash(crop),
                "nearest_anchor_hash": tensor_hash(anchor),
                "original_noise_hash": tensor_hash(noise),
                "null_noise_hash": tensor_hash(noise_null),
                "restricted_null_rms": rms(restricted_null),
                "restricted_null_max_abs": float(restricted_null.abs().max()),
                "noise_rms": rms(noise),
                "null_noise_rms": rms(noise_null),
                "null_to_noise_variance_ratio": float(noise_null.var(unbiased=False) / noise.var(unbiased=False)),
                "working_state_hash": None,
                "free_prediction_hash": None,
                "restricted_prediction_hash": None,
                "inference_executed": False,
            })
        cases.append({
            "case": case,
            "blueprint": blueprint_record,
            "control": {
                "phase28_bit_exact": True,
                "final_hash": control["final_hash"],
                "semantic_grade": "S3",
                "regions": len(control["regions"]),
            },
            "nullspace_diagnostics": regions,
            "all_restricted_null_within_tolerance": all(
                item["restricted_null_max_abs"] <= 1e-6 for item in regions
            ),
        })

    report = {
        "phase": 32,
        "decision": "D — NO PARAMETER-FREE SIGMA-CONSISTENT CONSTRUCTION EXISTS",
        "status": "stopped_before_inference",
        "sigma": 0.25,
        "operator": {
            "D": "avg_pool2d(kernel=2,stride=2)",
            "U": "nearest2",
            "P": "U o D",
            "noise_null": "(I-P) noise",
        },
        "covariance_proof_per_2x2_block": {
            "input_covariance": "I_4",
            "null_covariance": "I_4 - (1/4) 11^T",
            "rank": 3,
            "diagonal_variance": 0.75,
            "off_diagonal_covariance": -0.25,
            "variance_only_scale": 2.0 / math.sqrt(3.0),
            "scaled_off_diagonal_covariance": -1.0 / 3.0,
            "conclusion": (
                "No scalar maps the singular correlated nullspace law to iid N(0,I). "
                "The variance-only scale is therefore not sigma-consistent."
            ),
        },
        "construction_analysis": {
            "replace_noise_with_noise_null": "Changes the qualified Gaussian marginal and joint covariance.",
            "scale_noise_null": "Requires renormalization and still cannot restore iid covariance.",
            "add_noise_null_to_control": "Double-counts the existing null component and requires a detail-strength coefficient.",
            "retain_original_noise": "Exactly reproduces control A and introduces no new discriminator.",
        },
        "guided_arm_executed": False,
        "model_calls": 0,
        "destination_sized_forwards": 0,
        "production_modified": False,
        "comfy_core_modified": False,
        "cases": cases,
    }
    atomic_json(OUTPUT / "report.json", report)
    print("Phase 32 nullspace diagnostics persisted; stopped before inference.")


if __name__ == "__main__":
    main()
