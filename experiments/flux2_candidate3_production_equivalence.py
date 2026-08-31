"""Real Candidate-3 production-slice equivalence regression.

Runs the qualified Phase-3c terminal-release experiment path and the production
sampler path from identical inputs, then compares every state boundary required
by Phase 4b. This is a fixed regression workflow, not a parameterized sampler.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_production_equivalence_results"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_candidate3_hard_global_anchor as research
import flux2_candidate3_mapped_variance_discriminator as phase3c
import flux2_coarse_global_local_falsification as phase2


def load_production_package():
    spec = importlib.util.spec_from_file_location(
        "blueprint_diffusion",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blueprint_diffusion"] = module
    spec.loader.exec_module(module)


load_production_package()
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


SEED = phase3c.SEED
PROMPT = phase3c.PROMPT
BOUNDARIES = (
    "x0_G",
    "assembled_x0_H",
    "G_star",
    "H_star",
    "accepted_G",
    "accepted_H",
)


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    delta = value.float() - reference.float()
    max_abs = float(delta.abs().max())
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": max_abs,
        "bit_exact": bool(torch.equal(value, reference)),
        "within_tolerance": max_abs <= 2e-6,
    }


def run_production(model, noise, positive, negative, sigmas, sampler):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model,
            noise.clone(),
            1.0,
            sampler,
            sigmas.clone(),
            positive,
            negative,
            torch.zeros_like(noise),
            disable_pbar=True,
            seed=SEED,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return output.detach().float().cpu(), time.perf_counter() - started


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target_hw = (32, 64)
    crop_hw = (32, 32)
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, 8)
    sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn(
        (1, 128, *target_hw), generator=torch.Generator().manual_seed(SEED)
    )
    restrict_fn, prolong_fn, global_hw = research.geometry_operators("block_dct_24x48")
    trace = research.TrajectoryTrace("RESEARCH_REFERENCE", "terminal_release")
    reference_sampler = research.Candidate3Sampler(
        "terminal_release",
        target_hw,
        global_hw,
        crops,
        trace,
        SEED,
        restrict_fn,
        prolong_fn,
        "block_dct_24x48",
    )
    print("Running research reference...", flush=True)
    reference = research.run_trajectory(
        model, noise, positive, negative, sigmas, reference_sampler, SEED
    )

    captured: dict[str, dict[int, torch.Tensor]] = {}

    def capture(name: str, ordinal: int, value: torch.Tensor) -> None:
        captured.setdefault(name, {})[ordinal] = value.detach().float().cpu()

    production_sampler = BlueprintEulerSampler(capture=capture)
    print("Running production slice...", flush=True)
    production, production_seconds = run_production(
        model, noise, positive, negative, sigmas, production_sampler
    )

    references = {
        "initial_H": {-1: trace.h_before[0]},
        "initial_G": {-1: trace.g_before[0]},
        "x0_G": trace.g_predictions,
        "assembled_x0_H": trace.h_predictions,
        "G_star": trace.g_proposals,
        "H_star": trace.h_proposals,
        "accepted_G": trace.g_after,
        "accepted_H": trace.h_after,
    }
    comparisons = {}
    first_divergence = None
    for name in ("initial_H", "initial_G", *BOUNDARIES):
        comparisons[name] = {}
        for ordinal, reference_value in references[name].items():
            result = difference(captured[name][ordinal], reference_value)
            comparisons[name][str(ordinal)] = result
            if not result["within_tolerance"] and first_divergence is None:
                first_divergence = {"boundary": name, "ordinal": ordinal, **result}
    final = difference(production, reference)
    comparisons["final_H"] = final
    if not final["within_tolerance"] and first_divergence is None:
        first_divergence = {"boundary": "final_H", "ordinal": 3, **final}

    telemetry = production_sampler.last_telemetry
    accepted = [item for item in telemetry if item["event"] == "accepted_interval"]
    lifecycle = {
        "accepted_intervals": len(accepted),
        "all_have_four_predictions": all(item["model_predictions"] == 4 for item in accepted),
        "all_have_1152_global_tokens": all(item["global_tokens"] == 1152 for item in accepted),
        "all_have_3072_local_tokens": all(item["local_tokens"] == 3072 for item in accepted),
        "nonterminal_invariants_pass": all(
            item["invariant_max_abs"] is None or item["invariant_max_abs"] <= 2e-6
            for item in accepted
        ),
        "terminal_release_only_last": [item["terminal_release"] for item in accepted]
        == [False, False, False, True],
    }
    equivalent = first_divergence is None and all(lifecycle.values())
    report = {
        "prompt": PROMPT,
        "seed": SEED,
        "sigmas": [float(value) for value in sigmas],
        "tolerance": 2e-6,
        "equivalent": equivalent,
        "first_divergence": first_divergence,
        "comparisons": comparisons,
        "lifecycle": lifecycle,
        "reference_seconds": trace.seconds,
        "production_seconds": production_seconds,
        "production_telemetry": telemetry,
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "equivalent": equivalent,
        "first_divergence": first_divergence,
        "final_H": final,
        "lifecycle": lifecycle,
        "report": str(OUTPUT / "report.json"),
    }, indent=2), flush=True)
    if not equivalent:
        raise AssertionError(f"Production slice diverged: {first_divergence or lifecycle}")


if __name__ == "__main__":
    main()
