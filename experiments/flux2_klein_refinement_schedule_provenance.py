"""Phase 37 source/formula audit. This file performs no diffusion inference."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
OUTPUT = ROOT / "experiments" / "flux2_klein_refinement_schedule_provenance_results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    if image_seq_len > 4300:
        return a2 * image_seq_len + b2
    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1
    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    return a * num_steps + b


def shifted_schedule(image_seq_len: int, num_steps: int) -> list[float]:
    mu = empirical_mu(image_seq_len, num_steps)
    values = []
    for index in range(num_steps + 1):
        t = 1.0 - index / num_steps
        if t in (0.0, 1.0):
            values.append(t)
        else:
            values.append(math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0)))
    return values


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    model_sampling = COMFY_ROOT / "comfy" / "model_sampling.py"
    samplers = COMFY_ROOT / "comfy" / "samplers.py"
    schedulers = COMFY_ROOT / "comfy_extras" / "nodes_custom_sampler.py"
    supported_models = COMFY_ROOT / "comfy" / "supported_models.py"
    required = {
        model_sampling: ("class CONST:", "class ModelSamplingFlux"),
        samplers: ("SCHEDULER_HANDLERS", "new_steps = int(steps/denoise)"),
        schedulers: ("total_steps = int(steps/denoise)", "sigmas = sigmas[-(steps + 1):]"),
        supported_models: ('class Flux2(Flux):', '"shift": 2.02'),
    }
    for path, needles in required.items():
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                raise RuntimeError(f"Expected source marker {needle!r} missing from {path}")

    schedules = {
        str(tokens): {
            str(steps): {
                "mu": empirical_mu(tokens, steps),
                "sigmas": shifted_schedule(tokens, steps),
            }
            for steps in (4, 8, 50)
        }
        for tokens in (2048, 4096)
    }
    report = {
        "phase": 37,
        "status": "audit_complete_no_model_inference",
        "verdict": "B — CANONICAL FULL/PARTIAL SCHEDULE EXISTS, BUT LOCAL STEP COUNT REMAINS FREE",
        "final_classification": "SHORT LOCAL TRAJECTORY REQUIRES AN EXPLICIT EMPIRICAL SCHEDULE POLICY",
        "conclusion": {
            "canonical_short_schedule_found": False,
            "official_full_schedule": "BFL image-sequence-length and step-count shifted linear flow schedule",
            "official_solver": "first-order Euler",
            "official_partial_policy": "Diffusers Klein inpaint truncates an already discretized schedule by strength/index",
            "scheduler_family_prescribed": "BFL code fixes its shifted flow curve and Euler; Diffusers model package names FlowMatchEulerDiscreteScheduler",
            "local_step_count_prescribed": False,
            "sigma_0_25_is_valid_flow_coordinate": True,
            "sigma_0_25_is_official_refinement_start": False,
            "intermediate_points_uniquely_determined_from_sigma_0_25": False,
            "phase38_schedule": None,
        },
        "authoritative_sources": [
            {
                "authority": "Black Forest Labs official inference repository",
                "url": "https://github.com/black-forest-labs/flux2",
                "facts": ["Klein 4B is step-distilled", "distilled default/fixed num_steps is 4"],
            },
            {
                "authority": "Black Forest Labs official sampling implementation",
                "url": "https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py",
                "facts": ["shifted-linear full schedule", "mu depends on image sequence length and step count", "Euler update"],
            },
            {
                "authority": "BFL Klein 4B Diffusers scheduler configuration",
                "url": "https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/scheduler/scheduler_config.json",
                "facts": ["FlowMatchEulerDiscreteScheduler", "dynamic shifting", "1000 training timesteps"],
            },
            {
                "authority": "Hugging Face Diffusers Klein inpaint pipeline",
                "url": "https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py",
                "facts": ["strength truncates a precomputed schedule by index", "num_inference_steps remains caller supplied"],
            },
        ],
        "bfl_formula_audit": {
            "formula": "linspace(1,0,num_steps+1), then exp(mu)/(exp(mu)+(1/t-1)); mu depends on image_seq_len and num_steps",
            "computed_examples": schedules,
            "four_step_4096_contains_0_25": 0.25 in schedules["4096"]["4"]["sigmas"],
        },
        "comfyui_audit": {
            "const_denoised_relation": "x0 = x_t - sigma * model_output",
            "const_noise_relation": "x_t = sigma * noise_scale * noise + (1-sigma) * latent",
            "flux2_shift": 2.02,
            "scheduler_selection": "caller-selected through calculate_sigmas/SCHEDULER_HANDLERS",
            "denoise_policy": "compute a longer caller-selected schedule and retain its final steps",
            "source_files": [
                {"path": str(path), "sha256": sha256(path)} for path in required
            ],
        },
        "integrity": {
            "diffusion_model_forwards": 0,
            "local_forwards": 0,
            "destination_sized_forwards": 0,
            "decoded_images": 0,
            "production_changed": False,
            "comfyui_core_changed": False,
        },
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
