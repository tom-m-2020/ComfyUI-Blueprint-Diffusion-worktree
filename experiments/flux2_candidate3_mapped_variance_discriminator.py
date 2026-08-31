"""Phase-3e mapped-noise-variance discriminator.

Runs one dense metric reference and exactly three fixed terminal-release
Candidate-3 geometries. Experiment only; no production integration or sweep.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch

import flux2_candidate3_hard_global_anchor as c3
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_candidate3_mapped_variance_results"
PROMPT = (
    "A cinematic wide-angle photograph of exactly one large full-body woman "
    "standing centered in the foreground, occupying most of the image height; "
    "exactly one red vintage car parked on the far left; exactly one tall green "
    "tree on the far right; asymmetric left-center-right composition, continuous "
    "dry ground plane, coherent perspective and scale, distant low hills, sunset "
    "light, no duplicate people, no duplicate cars, no duplicate trees"
)
SEED = 20260831
GEOMETRIES = (
    ("A_ORIGINAL_16X32", "mean_16x32"),
    ("B_VARIANCE_MATCHED_16X32", "scaled_mean_16x32"),
    ("C_BLOCK_DCT_24X48", "block_dct_24x48"),
)


def synthetic_right_inverse(name: str) -> dict:
    restrict_fn, prolong_fn, global_hw = c3.geometry_operators(name)
    generator = torch.Generator().manual_seed(271828)
    value = torch.randn((1, 5, *global_hw), generator=generator)
    difference = c3.tensor_difference(restrict_fn(prolong_fn(value)), value)
    if difference["max_abs"] > 2e-6:
        raise AssertionError(f"{name} synthetic D(U(G)) failed: {difference}")
    return {"global_hw": list(global_hw), **difference}


def save_decoded(vae, value: torch.Tensor, path: Path) -> None:
    with torch.inference_mode():
        pixels = vae.decode(value).cpu()
    phase2.save_pixels(pixels, path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    synthetic = {name: synthetic_right_inverse(name) for _, name in GEOMETRIES}

    width, height = 1024, 512
    target_hw = (height // 16, width // 16)
    crop_hw = (512 // 16, 512 // 16)
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, 128 // 16)
    sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
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

    dense_trace = c3.TrajectoryTrace("DENSE_REFERENCE", "dense")
    dense_restrict, dense_prolong, dense_hw = c3.geometry_operators("mean_16x32")
    dense_sampler = c3.Candidate3Sampler(
        "dense", target_hw, dense_hw, crops, dense_trace, SEED,
        dense_restrict, dense_prolong, "dense_reference",
    )
    print("Running DENSE_REFERENCE...", flush=True)
    dense = c3.run_trajectory(
        model, noise, positive, negative, sigmas, dense_sampler, SEED
    )

    outputs = {}
    traces = {}
    initialization = {}
    for label, geometry in GEOMETRIES:
        restrict_fn, prolong_fn, global_hw = c3.geometry_operators(geometry)
        mapped = restrict_fn(noise)
        initialization[label] = {
            "geometry": geometry,
            "H_0": phase2.stats(noise),
            "G_0": phase2.stats(mapped),
            "variance_ratio_G_to_H": float(mapped.float().var() / noise.float().var()),
            "predicted_variance_ratio": 0.25 if geometry == "mean_16x32" else 0.5625,
            "adjacent_correlation_G": c3.adjacent_correlation(mapped),
            "D_H_identity": c3.tensor_difference(mapped, restrict_fn(noise)),
        }
        trace = c3.TrajectoryTrace(label, "terminal_release")
        sampler = c3.Candidate3Sampler(
            "terminal_release", target_hw, global_hw, crops, trace, SEED,
            restrict_fn, prolong_fn, geometry,
        )
        print(f"Running {label}...", flush=True)
        outputs[label] = c3.run_trajectory(
            model, noise, positive, negative, sigmas, sampler, SEED
        )
        traces[label] = trace

    final_paths = []
    dense_path = OUTPUT / "DENSE_REFERENCE_FINAL.png"
    save_decoded(vae, dense, dense_path)
    for label, _ in GEOMETRIES:
        final_path = OUTPUT / f"{label}_FINAL.png"
        global_path = OUTPUT / f"{label}_GLOBAL_FINAL.png"
        save_decoded(vae, outputs[label], final_path)
        save_decoded(vae, traces[label].g_after[3], global_path)
        final_paths.append((label, final_path))
        for step in range(4):
            c3.save_projection_heatmap(
                traces[label].projections[step],
                OUTPUT / f"{label}_STEP_{step:02d}_APPLIED_PROJECTION.png",
            )
    c3.save_contact_sheet(
        [("DENSE REFERENCE", dense_path), *final_paths],
        OUTPUT / "FINAL_COMPARISON.png",
    )
    c3.save_contact_sheet(
        [
            (label, OUTPUT / f"{label}_GLOBAL_FINAL.png")
            for label, _ in GEOMETRIES
        ],
        OUTPUT / "GLOBAL_COMPARISON.png",
    )

    variants = {}
    for label, geometry in GEOMETRIES:
        trace = traces[label]
        restrict_fn, _, global_hw = c3.geometry_operators(geometry)
        projection_records = []
        for record in trace.evaluations:
            projection_records.append(
                {
                    "step": record["step"],
                    "sigma": record["sigma"],
                    "sigma_next": record["sigma_next"],
                    "projection_available": record["projection_available"],
                    "projection_applied": record["projection_applied"],
                    "projection_rms_to_h_proposal_rms": record[
                        "projection_rms_to_h_proposal_rms"
                    ],
                    "consistency_after_acceptance": record[
                        "consistency_after_acceptance"
                    ],
                    "overlap_disagreement": record["overlap_disagreement"],
                    "model_forwards": record["model_forwards"],
                    "executed_image_tokens": record["executed_image_tokens"],
                    "global_rope_options": record["global_rope_options"],
                }
            )
        terminal_consistency = phase2d.comparison(
            restrict_fn(outputs[label]), trace.g_after[3]
        )
        variants[label] = {
            "geometry": geometry,
            "global_hw": list(global_hw),
            "global_tokens": math.prod(global_hw),
            "seconds": trace.seconds,
            "peak_allocated_gib": trace.peak_allocated_gib,
            "atomic_acceptances": trace.atomic_acceptances,
            "final_H": phase2.stats(outputs[label]),
            "final_G": phase2.stats(trace.g_after[3]),
            "vs_dense": phase2d.comparison(outputs[label], dense),
            "terminal_D_H_minus_G": terminal_consistency,
            "projection_records": projection_records,
            "final_output": str(OUTPUT / f"{label}_FINAL.png"),
            "global_output": str(OUTPUT / f"{label}_GLOBAL_FINAL.png"),
        }

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": PROMPT,
            "seed": SEED,
            "cfg": 1.0,
            "sigmas": sigmas.tolist(),
            "target_hw": list(target_hw),
            "crop_hw": list(crop_hw),
            "crops": [crop.__dict__ for crop in crops],
            "overlap_pixels": 128,
            "lifecycle": "hard nonterminal acceptance; terminal local-H release",
            "dense_reference_is_metric_target_not_candidate_variant": True,
        },
        "synthetic_right_inverse_before_model": synthetic,
        "initialization": initialization,
        "dense_reference": {
            "seconds": dense_trace.seconds,
            "peak_allocated_gib": dense_trace.peak_allocated_gib,
            "final": phase2.stats(dense),
            "output": str(dense_path),
        },
        "variants": variants,
        "outputs": {
            "final_comparison": str(OUTPUT / "FINAL_COMPARISON.png"),
            "global_comparison": str(OUTPUT / "GLOBAL_COMPARISON.png"),
        },
    }
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(OUTPUT / "report.json"),
                "variants": {
                    label: {
                        "variance_ratio": initialization[label]["variance_ratio_G_to_H"],
                        "rms_vs_dense": variants[label]["vs_dense"]["absolute"]["rms"],
                        "low_frequency_rms": variants[label]["vs_dense"]["low_frequency"]["rms"],
                        "terminal_overlap": variants[label]["projection_records"][3][
                            "overlap_disagreement"
                        ]["aggregate_rms"],
                    }
                    for label, _ in GEOMETRIES
                },
            },
            indent=2,
        )
    )
    gc.collect()


if __name__ == "__main__":
    main()
