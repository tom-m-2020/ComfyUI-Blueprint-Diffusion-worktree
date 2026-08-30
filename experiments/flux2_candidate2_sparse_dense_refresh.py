"""Zero-update sparse dense-refresh maintenance probe for Candidate 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as phase2c
import flux2_candidate2_four_step_trajectory as phase2e
import flux2_candidate2_dense_vs_compact_kv as phase2f
import flux2_candidate2_equal_budget_multiscale as phase2h
import flux2_candidate2_dense_block_necessity as phase2m


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_dense_block_necessity_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_sparse_dense_refresh_results"
VARIANTS = (
    "A_ALL_DENSE",
    "B_EARLY_ONLY",
    "C_EARLY_PERIODIC_REFRESH",
    "D_NO_IMAGE_IMAGE",
)
EARLY_ORDINALS = set(range(5))
REFRESH_ORDINALS = (10, 15, 20)
REFRESH_SCHEDULE = EARLY_ORDINALS | set(REFRESH_ORDINALS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class RefreshSampler(phase2m.DenseBlockSampler):
    def run_schedule(self, model, noise, local_input, sigma, base_options, name, dense_ordinals):
        probe = phase2m.SelectiveDenseProbe(set(dense_ordinals))
        self.probes[name] = probe
        source = self.run_call(
            model, noise, sigma,
            phase2m.selective_capture_options(f"{name}_source", probe),
            base_options, f"{name}_GLOBAL_CAPTURE", "global_capture",
        )
        output = self.run_call(
            model, local_input, sigma,
            phase2c.model_options(
                f"{name}_local", phase2.rope_for_crop(self.crop), probe, "context"
            ),
            base_options, f"{name}_LOCAL_CONTEXT", "local_with_external_kv",
        )
        probe.assert_complete()
        probe.assert_policy_complete()
        first = probe.context_records[0]
        total_dense_edges = len(dense_ordinals) * phase2m.DENSE_IMAGE_EDGES_PER_BLOCK
        self.layouts[name] = {
            "source_generated_tokens": first["global_generated_kv"],
            "source_sequence_tokens": 2560,
            "source_q_by_k": [2560, 2560],
            "local_external_q_by_k": first["q_by_k"],
            "dense_execution_ordinals": sorted(dense_ordinals),
            "dense_block_names": [
                ("double", ordinal) if ordinal < 5 else ("single", ordinal - 5)
                for ordinal in sorted(dense_ordinals)
            ],
            "dense_block_count": len(dense_ordinals),
            "dense_image_image_edges_per_dense_block": phase2m.DENSE_IMAGE_EDGES_PER_BLOCK,
            "total_dense_image_image_edges": total_dense_edges,
            "fraction_all_dense_image_edge_budget": total_dense_edges / phase2m.ALL_DENSE_IMAGE_EDGES,
            "source_policy_records": sorted(probe.policy_records, key=lambda item: item["execution_ordinal"]),
            "source_capture_records": probe.capture_records,
            "local_context_records": probe.context_records,
        }
        self.positions[name] = probe.position_records
        self.source_outputs[name] = source.detach().float().cpu()
        return output.detach().float().cpu()

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("This zero-update diagnostic does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
        self.outputs["A_ALL_DENSE"] = self.run_variant(
            model, noise, local_input, sigma, base_options, "A_ALL_DENSE", "all"
        )
        self.outputs["B_EARLY_ONLY"] = self.run_schedule(
            model, noise, local_input, sigma, base_options, "B_EARLY_ONLY", EARLY_ORDINALS
        )
        self.outputs["C_EARLY_PERIODIC_REFRESH"] = self.run_schedule(
            model, noise, local_input, sigma, base_options,
            "C_EARLY_PERIODIC_REFRESH", REFRESH_SCHEDULE,
        )
        self.outputs["D_NO_IMAGE_IMAGE"] = self.run_variant(
            model, noise, local_input, sigma, base_options, "D_NO_IMAGE_IMAGE", "none"
        )
        ordinary = self.layouts[VARIANTS[0]]
        invariant_keys = ("source_generated_tokens", "source_sequence_tokens", "source_q_by_k", "local_external_q_by_k")
        for name in VARIANTS[1:]:
            if any(self.layouts[name][key] != ordinary[key] for key in invariant_keys):
                raise AssertionError(f"Token geometry changed for {name}.")
        return noise


def difference_by_ordinal(reference_probe, candidate_probe) -> list[dict[str, Any]]:
    return phase2m.kv_differences(reference_probe, candidate_probe)


def refresh_effects(all_dense_probe, early_probe, refresh_probe) -> list[dict[str, Any]]:
    refresh_vs_dense = difference_by_ordinal(all_dense_probe, refresh_probe)
    early_vs_dense = difference_by_ordinal(all_dense_probe, early_probe)
    refresh_vs_early = difference_by_ordinal(early_probe, refresh_probe)
    records = []
    for index, ordinal in enumerate(REFRESH_ORDINALS):
        next_refresh = REFRESH_ORDINALS[index + 1] if index + 1 < len(REFRESH_ORDINALS) else 24
        after = ordinal + 1
        records.append({
            "refresh_ordinal": ordinal,
            "capture_before_refresh": ordinal,
            "capture_after_refresh": after,
            "survival_checked_at_capture": next_refresh,
            "refresh_vs_early_before_refresh": refresh_vs_early[ordinal],
            "after_refresh": {
                "refresh_vs_all_dense": refresh_vs_dense[after],
                "early_only_vs_all_dense": early_vs_dense[after],
                "refresh_vs_early_only": refresh_vs_early[after],
                "k_rms_reduction_vs_early": (
                    early_vs_dense[after]["positioned_k_difference"]["rms"]
                    - refresh_vs_dense[after]["positioned_k_difference"]["rms"]
                ),
                "v_rms_reduction_vs_early": (
                    early_vs_dense[after]["v_difference"]["rms"]
                    - refresh_vs_dense[after]["v_difference"]["rms"]
                ),
            },
            "before_next_refresh_or_final_capture": {
                "refresh_vs_all_dense": refresh_vs_dense[next_refresh],
                "early_only_vs_all_dense": early_vs_dense[next_refresh],
                "refresh_vs_early_only": refresh_vs_early[next_refresh],
                "k_rms_reduction_vs_early": (
                    early_vs_dense[next_refresh]["positioned_k_difference"]["rms"]
                    - refresh_vs_dense[next_refresh]["positioned_k_difference"]["rms"]
                ),
                "v_rms_reduction_vs_early": (
                    early_vs_dense[next_refresh]["v_difference"]["rms"]
                    - refresh_vs_dense[next_refresh]["v_difference"]["rms"]
                ),
            },
        })
    if refresh_vs_early[10]["positioned_k_difference"]["rms"] != 0.0:
        raise AssertionError("Refresh variant diverged from early-only before first refresh result.")
    if refresh_vs_early[11]["positioned_k_difference"]["rms"] == 0.0:
        raise AssertionError("First refresh did not change the following K/V capture.")
    return records


def dry_run() -> None:
    schedules = {
        "A_ALL_DENSE": set(range(25)),
        "B_EARLY_ONLY": EARLY_ORDINALS,
        "C_EARLY_PERIODIC_REFRESH": REFRESH_SCHEDULE,
        "D_NO_IMAGE_IMAGE": set(),
    }
    print(json.dumps({name: {
        "dense_ordinals": sorted(ordinals),
        "dense_block_count": len(ordinals),
        "dense_edges_per_block": phase2m.DENSE_IMAGE_EDGES_PER_BLOCK,
        "total_dense_edges": len(ordinals) * phase2m.DENSE_IMAGE_EDGES_PER_BLOCK,
        "fraction_all_dense_edge_budget": len(ordinals) / 25,
        "source_q_by_k": [2560, 2560],
        "local_q_by_k": [1536, 3584],
        "sampler_updates": 0,
    } for name, ordinals in schedules.items()}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    if not PREVIOUS_REPORT.is_file():
        raise FileNotFoundError(PREVIOUS_REPORT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_hw = (32, 64)
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, 32, 32, 8)
    crop = crops[1]
    sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()
    diagnostic_step = 2
    diagnostic_sigma = sigmas[diagnostic_step]

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase2.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn((1, 128, *target_hw), generator=torch.Generator().manual_seed(seed))
    trace = phase2e.TrajectoryTrace("STATE_REPRODUCTION")
    reproduction_sampler = phase2e.FourStepSampler("context", target_hw, phase2h.COARSE_HW, crops, trace, seed)
    accepted = phase2e.run_trajectory(model, noise, positive, negative, sigmas[: diagnostic_step + 1], reproduction_sampler, seed)
    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    expected_stats = previous["state_reproduction"]["reproduced_stats"]
    reproduced_stats = trace.evaluations[-1]["accepted_latent_after"]
    stat_differences = {key: abs(float(reproduced_stats[key]) - float(expected_stats[key])) for key in ("norm", "rms", "mean", "std", "min", "max")}
    if trace.updates != diagnostic_step or max(stat_differences.values()) > 1e-6:
        raise AssertionError((trace.updates, stat_differences))

    sampler = RefreshSampler(target_hw, crop, seed)
    with torch.inference_mode():
        unchanged = phase2.comfy.sample.sample_custom(
            model, accepted.clone(), 1.0, sampler,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive, negative, torch.zeros_like(accepted), disable_pbar=True, seed=seed,
        ).detach().float().cpu()
    no_update_difference = phase2c.tensor_difference(unchanged, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = sampler.source_outputs[VARIANTS[0]]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in sampler.outputs.items():
        comparisons[name] = {
            "absolute": phase2c.tensor_difference(output, dense_crop),
            "low_frequency": phase2c.low_frequency_difference(output, dense_crop),
            "prediction": phase2.stats(output),
            "difference_from_A_all_dense_local": phase2c.tensor_difference(output, sampler.outputs[VARIANTS[0]]),
            "difference_from_B_early_only_local": phase2c.tensor_difference(output, sampler.outputs[VARIANTS[1]]),
        }
        with torch.inference_mode():
            phase2.save_pixels(vae.decode(output).cpu(), args.output_dir / f"{name}.png")
        canvas = dense_full.clone()
        canvas[:, :, crop.y:crop.y2, crop.x:crop.x2] = output
        with torch.inference_mode():
            phase2.save_pixels(vae.decode(canvas).cpu(), args.output_dir / f"{name}_IN_DENSE_CONTEXT.png")
        phase2.save_heatmap(
            (output - dense_crop).square().mean(dim=1, keepdim=True).sqrt(),
            args.output_dir / f"{name}_ERROR_MAGNITUDE.png", (512, 512),
        )
    with torch.inference_mode():
        phase2.save_pixels(vae.decode(dense_crop).cpu(), args.output_dir / "E_DENSE_REFERENCE_CROP.png")
        phase2.save_pixels(vae.decode(dense_full).cpu(), args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")
        for name in VARIANTS[1:]:
            phase2.save_pixels(vae.decode(sampler.source_outputs[name]).cpu(), args.output_dir / f"{name}_FULL_SOURCE.png")

    all_dense_probe = sampler.probes[VARIANTS[0]]
    early_probe = sampler.probes[VARIANTS[1]]
    refresh_probe = sampler.probes[VARIANTS[2]]
    effects = refresh_effects(all_dense_probe, early_probe, refresh_probe)
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "source_grid_hw": list(target_hw),
            "early_dense_ordinals": sorted(EARLY_ORDINALS), "refresh_ordinals": list(REFRESH_ORDINALS),
            "combined_refresh_schedule": sorted(REFRESH_SCHEDULE),
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "local_integration_changed": False,
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "comparisons_vs_dense_crop": comparisons,
        "source_kv_differences_from_all_dense": {
            name: difference_by_ordinal(all_dense_probe, sampler.probes[name]) for name in VARIANTS[1:]
        },
        "source_kv_differences_from_early_only": {
            name: difference_by_ordinal(early_probe, sampler.probes[name]) for name in (VARIANTS[2], VARIANTS[3])
        },
        "refresh_effects": effects,
        "source_prediction_comparisons_from_all_dense": {
            name: phase2c.tensor_difference(sampler.source_outputs[name], dense_full) for name in VARIANTS[1:]
        },
        "decoded_semantic_observations": {
            "A_ALL_DENSE": {
                "duplicate_lighthouse": "suppressed; only the dense-reference distant silhouette remains",
                "stone_structure": "reduced center-right tower remains",
                "train": "closest continuity and alignment to ordinary dense crop",
                "bridge": "closest deck and cable geometry to ordinary dense crop",
                "horizon_water": "closest agreement; no distinct extra white lighthouse",
            },
            "B_EARLY_ONLY": {
                "duplicate_lighthouse": "small white center-right lighthouse remains beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "improved over no mixing but carriage extent still differs from all-dense",
                "bridge": "deck/cable organization improves over no mixing without restoring all-dense semantics",
                "horizon_water": "false lighthouse and tower pair remains",
            },
            "C_EARLY_PERIODIC_REFRESH": {
                "duplicate_lighthouse": "small white center-right lighthouse remains at early-only severity",
                "stone_structure": "prominent center-right tower remains",
                "train": "no material final improvement over early-only",
                "bridge": "deck/cable organization remains comparable to early-only rather than all-dense",
                "horizon_water": "false lighthouse and tower pair remains despite three refreshes",
            },
            "D_NO_IMAGE_IMAGE": {
                "duplicate_lighthouse": "small white center-right lighthouse remains beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation diverge from all-dense",
                "bridge": "deck remains continuous but train/cable geometry regresses",
                "horizon_water": "false lighthouse and tower pair remains",
            },
            "interpretation": "refreshes at 15 and 20 briefly reduce some K/V divergence, but gains do not survive blocked depth and the final semantic result is not improved over early-only",
            "scope": "manual inspection of decoded one-evaluation denoised estimates; not an automated object metric",
        },
        "dense_reference_crop": phase2.stats(dense_crop),
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in VARIANTS},
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(args.output_dir / "report.json"),
        "state_stat_max_difference": max(stat_differences.values()),
        "no_update_max_abs": no_update_difference["max_abs"],
        "variants": {name: {
            "dense_ordinals": sampler.layouts[name]["dense_execution_ordinals"],
            "dense_edge_budget_fraction": sampler.layouts[name]["fraction_all_dense_image_edge_budget"],
            "source_q_by_k": sampler.layouts[name]["source_q_by_k"],
            "local_q_by_k": sampler.layouts[name]["local_external_q_by_k"],
            "rms": comparisons[name]["absolute"]["rms"],
            "low_frequency_rms": comparisons[name]["low_frequency"]["rms"],
            "prediction_rms": comparisons[name]["prediction"]["rms"],
        } for name in VARIANTS},
    }, indent=2))


if __name__ == "__main__":
    main()
