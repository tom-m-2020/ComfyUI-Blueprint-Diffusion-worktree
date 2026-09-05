"""Phase 36 schedule preflight. Intentionally performs no diffusion forward."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import comfy.samplers
import comfy.sd
import flux2_coarse_global_local_falsification as phase2

OUTPUT = ROOT / "experiments" / "flux2_terminal_resampling_short_local_trajectory_results"
SOURCE = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
QUALIFIED_BLUEPRINT_SIGMAS = (1.0, 0.9991771578788757, 0.9975355267524719, 0.9926428198814392, 0.0)
EXPECTED_MAPPED_HASH = "8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05"
EXPECTED_CONTROL_HASH = "1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3"


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    blueprint = torch.load(SOURCE / "blueprint.pt", map_location="cpu", weights_only=True)
    blueprint_record = json.loads((SOURCE / "blueprint.json").read_text(encoding="utf-8"))
    control = torch.load(SOURCE / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    control_record = json.loads((SOURCE / "sigma_0.25.json").read_text(encoding="utf-8"))
    mapped_hash = tensor_hash(blueprint["mapped"])
    control_hash = tensor_hash(control)
    if mapped_hash != EXPECTED_MAPPED_HASH or blueprint_record["mapped_hash"] != mapped_hash:
        raise RuntimeError("Persisted mapped Blueprint fingerprint mismatch.")
    if control_hash != EXPECTED_CONTROL_HASH or control_record["final_hash"] != control_hash:
        raise RuntimeError("Persisted sigma-0.25 control fingerprint mismatch.")
    if not control_record["phase28_regression"]["bit_exact"]:
        raise RuntimeError("Persisted control is not the qualified Phase-28/29 regression.")

    # Loading model metadata/sampling is allowed for schedule inspection. The
    # transformer is never invoked.
    model = comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    model_sampling = model.get_model_object("model_sampling")
    schedules = {}
    for steps in (3, 4):
        schedules[str(steps)] = {
            name: [float(value) for value in comfy.samplers.calculate_sigmas(model_sampling, name, steps)]
            for name in comfy.samplers.SCHEDULER_NAMES
        }

    positive_qualified = [value for value in QUALIFIED_BLUEPRINT_SIGMAS if value > 0]
    report = {
        "phase": 36,
        "status": "stopped_before_diffusion_inference",
        "decision": "D — NO UNIQUE SHORT LOCAL SCHEDULE",
        "control": {
            "case": "SQUARE_MULTI_OBJECT",
            "seed": 20260921,
            "mapped_blueprint_hash": mapped_hash,
            "qualified_sigma_0.25_hash": control_hash,
            "phase28_regression_bit_exact": True,
            "semantic_grade": "S3 (persisted qualified evidence)",
            "model_work_reused": True,
        },
        "qualified_contract": {
            "blueprint_sigmas": list(QUALIFIED_BLUEPRINT_SIGMAS),
            "blueprint_schedule_source": "fixed ManualSigmas / production QUALIFIED_SIGMAS",
            "terminal_local_schedule": [0.25, 0.0],
            "terminal_local_model_calls_per_region": 1,
            "named_stage2_scheduler_family": None,
            "positive_blueprint_sigma_at_or_below_0.25": any(0 < value <= 0.25 for value in positive_qualified),
            "last_positive_blueprint_sigma": positive_qualified[-1],
        },
        "live_scheduler_audit": {
            "scheduler_names": list(comfy.samplers.SCHEDULER_NAMES),
            "scheduler_count": len(comfy.samplers.SCHEDULER_NAMES),
            "schedules_by_interval_count": schedules,
            "transformer_forwards": 0,
        },
        "ambiguities": [
            "The qualified Stage-2 contract names no scheduler family; it defines only sigma 0.25 followed by zero.",
            "The fixed Blueprint schedule contains no positive point at or below 0.25 and therefore supplies no multi-step late tail to reuse.",
            "Choosing a native scheduler family changes the sigma path materially.",
            "Choosing three versus four or another number of intervals is an independent unqualified choice.",
            "Choosing where to truncate or splice a full schedule above 0.25 is another unqualified choice.",
        ],
        "integrity": {
            "diffusion_model_forward_calls": 0,
            "local_model_calls": 0,
            "destination_sized_model_calls": 0,
            "decoded_images_generated": 0,
            "blueprint_recomputed": False,
            "accepted_state_mutated": False,
            "production_changed": False,
            "comfy_core_changed": False,
        },
        "artifacts_not_applicable": {
            "comparison_image": "not created because experimental arm B was stopped before inference",
            "detail_review": "not created because no candidate output exists",
            "representative_regions": "not created because no local trajectory executed",
        },
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "decision": report["decision"]}, indent=2))


if __name__ == "__main__":
    main()
