"""Phase 7b boundary regression: production versus Phase-7a coordinator path."""

from __future__ import annotations

import gc
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_global_refresh_cadence as phase6i
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_step_count_generalization as phase7a

from blueprint_diffusion.sampling.euler import (
    BlueprintCoordinator,
    BlueprintEulerSampler,
    validate_schedule,
)


OUTPUT = ROOT / "experiments" / "flux2_candidate3_variable_step_production_results"
STEPS = (1, 2, 4, 8, 20)
BOUNDARIES = (
    "initial_H", "initial_G", "x0_G", "assembled_x0_H",
    "G_star", "H_star", "accepted_G", "accepted_H",
)


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = value.float() - reference.float()
    return {
        "bit_exact": bool(torch.equal(value, reference)),
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


class CoordinatorReferenceSampler(phase2.comfy.samplers.Sampler):
    """Exact Phase-7a lifecycle path, bypassing only production front-door validation."""

    def __init__(self, capture):
        self.capture = capture
        self.last_telemetry = ()

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        phase7a.validate_experiment_schedule(sigmas, len(sigmas) - 1)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        coordinator = BlueprintCoordinator(capture=self.capture)
        state = coordinator.initialize(h, sigmas[0])
        total = len(sigmas) - 1
        for ordinal in range(total):
            state, x0_h = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            )
            if callback is not None:
                callback(ordinal, x0_h, state.h, total)
        self.last_telemetry = tuple(coordinator.telemetry)
        return model_sampling.inverse_noise_scaling(sigmas[-1], state.h)


def run(model, positive, negative, noise, sigmas, sampler, seed, previews):
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise),
            callback=lambda ordinal, *args: previews.append(ordinal),
            disable_pbar=True, seed=seed,
        )
    torch.cuda.synchronize()
    return output.detach().cpu()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case = next(item for item in phase6i.CASES if item[0] == "h48_person_car_tree")
    name, width, height, prompt, seed = case
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    h_shape = (height // 16, width // 16)
    noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(seed))
    perf.prepare_model_state(model)

    report = {
        "configuration": {
            "case": name, "H_shape": list(h_shape), "steps": list(STEPS),
            "reference": "Phase-7a direct BlueprintCoordinator lifecycle",
            "production": "BlueprintEulerSampler through sample_custom",
        },
        "steps": {},
    }
    for steps in STEPS:
        sigmas = phase2.get_schedule(steps, math.prod(h_shape)).float().clone()
        validate_schedule(sigmas)
        reference = {}

        def capture_reference(boundary, ordinal, value):
            reference.setdefault(boundary, {})[ordinal] = value.detach().cpu().clone()

        reference_previews = []
        reference_sampler = CoordinatorReferenceSampler(capture_reference)
        print(f"{steps} steps: Phase-7a reference", flush=True)
        reference_output = run(
            model, positive, negative, noise, sigmas, reference_sampler, seed,
            reference_previews,
        )

        comparisons = {}
        exact = True

        def compare_production(boundary, ordinal, value):
            nonlocal exact
            result = difference(value.detach().cpu(), reference[boundary][ordinal])
            comparisons.setdefault(boundary, {})[str(ordinal)] = result
            exact &= result["bit_exact"]

        production_previews = []
        production_sampler = BlueprintEulerSampler(capture=compare_production)
        print(f"{steps} steps: production", flush=True)
        production_output = run(
            model, positive, negative, noise, sigmas, production_sampler, seed,
            production_previews,
        )
        final = difference(production_output, reference_output)
        exact &= final["bit_exact"]
        accepted = [
            item for item in production_sampler.last_telemetry
            if item["event"] == "accepted_interval"
        ]
        crop_count = accepted[0]["model_predictions"]
        if accepted[0]["global_forward_performed"]:
            crop_count -= 1
        measured_global_forwards = sum(item["global_forward_performed"] for item in accepted)
        measured_local_forwards = sum(item["model_predictions"] for item in accepted) - measured_global_forwards
        lifecycle = {
            "reference_previews": reference_previews,
            "production_previews": production_previews,
            "preview_count_exact": production_previews == list(range(steps)),
            "global_forward_flags": [item["global_forward_performed"] for item in accepted],
            "only_terminal_omits_global": [item["global_forward_performed"] for item in accepted]
            == [True] * (steps - 1) + [False],
            "global_forward_count": measured_global_forwards,
            "expected_global_forward_count": steps - 1,
            "local_forward_count": measured_local_forwards,
            "expected_local_forward_count": crop_count * steps,
            "terminal_release_flags": [item["terminal_release"] for item in accepted],
            "nonterminal_invariants": all(
                item["invariant_max_abs"] is None or item["invariant_max_abs"] <= 2e-6
                for item in accepted
            ),
        }
        lifecycle_pass = (
            lifecycle["preview_count_exact"]
            and lifecycle["only_terminal_omits_global"]
            and lifecycle["global_forward_count"] == steps - 1
            and lifecycle["nonterminal_invariants"]
        )
        report["steps"][str(steps)] = {
            "sigmas": sigmas.tolist(), "bit_exact": exact,
            "boundary_comparisons": comparisons, "final_H": final,
            "lifecycle": lifecycle, "lifecycle_pass": lifecycle_pass,
        }
        if not exact or not lifecycle_pass:
            (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            raise AssertionError(f"First production divergence at {steps} steps")
        del reference
        gc.collect()
    report["qualified"] = all(
        item["bit_exact"] and item["lifecycle_pass"] for item in report["steps"].values()
    )
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "qualified": report["qualified"],
        "steps": {
            steps: {
                "bit_exact": item["bit_exact"],
                "previews": len(item["lifecycle"]["production_previews"]),
                "global_forwards": item["lifecycle"]["global_forward_count"],
                "local_forwards": item["lifecycle"]["local_forward_count"],
            }
            for steps, item in report["steps"].items()
        },
        "report": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
