"""Zero-update intermediate global-density discriminator for Candidate 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as phase2c
import flux2_candidate2_all_crop_assembly_probe as phase2d
import flux2_candidate2_four_step_trajectory as phase2e
import flux2_candidate2_dense_vs_compact_kv as phase2f


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_dense_vs_compact_results" / "report.json"
DENSITIES = (
    ("A_512_COMPACT", (16, 32)),
    ("B_1152_INTERMEDIATE", (24, 48)),
    ("C_2048_DENSE", (32, 64)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate2_intermediate_density_results",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class DensityDiagnosticSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, crop, seed) -> None:
        self.target_hw = target_hw
        self.crop = crop
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.source_outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.probes: dict[str, phase2c.OneEvaluationContextProbe] = {}
        self.layouts: dict[str, dict[str, Any]] = {}

    def run_call(self, model, value, sigma, options, base_options, name, role):
        merged = phase2e.merge_options(base_options, options)
        prediction, call = phase2e.model_call(
            model,
            value,
            sigma,
            merged,
            self.seed,
            role,
            self.crop if role == "local_with_external_kv" else None,
        )
        self.calls[name] = call
        return prediction

    def sample(
        self,
        model,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if denoise_mask is not None:
            raise ValueError("This no-update diagnostic does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
        common_invariants = None

        for name, global_hw in DENSITIES:
            probe = phase2c.OneEvaluationContextProbe()
            self.probes[name] = probe
            if global_hw == self.target_hw:
                global_input = noise
                rope = {}
            else:
                global_input = F.interpolate(
                    noise, size=global_hw, mode="bilinear", align_corners=False
                )
                rope = phase2.rope_for_global(*self.target_hw, *global_hw)
            source = self.run_call(
                model,
                global_input,
                sigma,
                phase2c.model_options(f"{name}_source", rope, probe, "capture"),
                base_options,
                f"{name}_GLOBAL_CAPTURE",
                "global_capture",
            )
            output = self.run_call(
                model,
                local_input,
                sigma,
                phase2c.model_options(
                    f"{name}_local", phase2.rope_for_crop(crop), probe, "context"
                ),
                base_options,
                f"{name}_LOCAL_CONTEXT",
                "local_with_external_kv",
            )
            probe.assert_complete()
            if len(probe.capture_records) != 25 or len(probe.context_records) != 25:
                raise AssertionError((name, len(probe.capture_records), len(probe.context_records)))
            first = probe.context_records[0]
            invariants = {
                "local_generated_queries": first["local_generated_queries"],
                "query_tokens_total": first["query_tokens_total"],
                "text_tokens": first["text_tokens"],
                "blocks": [(item["block_type"], item["block_index"]) for item in probe.context_records],
            }
            if common_invariants is None:
                common_invariants = invariants
            elif invariants != common_invariants:
                raise AssertionError(f"Integration path changed for {name}.")
            query_count, key_count = first["q_by_k"]
            self.layouts[name] = {
                "global_latent_hw": list(global_hw),
                "global_generated_tokens": first["global_generated_kv"],
                "query_tokens_total": query_count,
                "kv_tokens_total": key_count,
                "q_by_k": first["q_by_k"],
                "local_query_attention_qk_products_per_block": query_count * key_count,
                "rope_options": rope,
                "context_records": probe.context_records,
                "capture_records": probe.capture_records,
            }
            self.source_outputs[name] = source.detach().float().cpu()
            self.outputs[name] = output.detach().float().cpu()

        compact_work = self.layouts["A_512_COMPACT"]["local_query_attention_qk_products_per_block"]
        dense_work = self.layouts["C_2048_DENSE"]["local_query_attention_qk_products_per_block"]
        for layout in self.layouts.values():
            work = layout["local_query_attention_qk_products_per_block"]
            layout["relative_to_512_compact"] = work / compact_work
            layout["relative_to_2048_dense"] = work / dense_work

        # No sampler update.
        return noise


def dry_run() -> None:
    q = 1536
    values = []
    for name, global_hw in DENSITIES:
        global_tokens = math.prod(global_hw)
        k = 512 + 1024 + global_tokens
        values.append({"name": name, "global_tokens": global_tokens, "q_by_k": [q, k], "qk": q * k})
    if [item["global_tokens"] for item in values] != [512, 1152, 2048]:
        raise AssertionError(values)
    print(json.dumps({"sampler_updates": 0, "densities": values}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    if not PREVIOUS_REPORT.is_file():
        raise FileNotFoundError(PREVIOUS_REPORT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    target_hw = (height // 16, width // 16)
    compact_hw = (16, 32)
    crop_hw = (32, 32)
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, 8)
    crop = crops[1]
    full_sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()
    diagnostic_step = 2
    diagnostic_sigma = full_sigmas[diagnostic_step]

    model = phase2.comfy.sd.load_diffusion_model(
        str(phase2.MODEL_PATH), model_options={}
    )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase2.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn(
        (1, 128, *target_hw), generator=torch.Generator().manual_seed(seed)
    )
    reproduction_trace = phase2e.TrajectoryTrace("STATE_REPRODUCTION")
    reproduction_sampler = phase2e.FourStepSampler(
        "context", target_hw, compact_hw, crops, reproduction_trace, seed
    )
    accepted = phase2e.run_trajectory(
        model,
        noise,
        positive,
        negative,
        full_sigmas[: diagnostic_step + 1],
        reproduction_sampler,
        seed,
    )
    if reproduction_trace.updates != diagnostic_step:
        raise AssertionError(reproduction_trace.updates)

    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    expected_stats = previous["state_reproduction"]["reproduced_stats"]
    reproduced_stats = reproduction_trace.evaluations[-1]["accepted_latent_after"]
    stat_differences = {
        key: abs(float(reproduced_stats[key]) - float(expected_stats[key]))
        for key in ("norm", "rms", "mean", "std", "min", "max")
    }
    if max(stat_differences.values()) > 1e-6:
        raise AssertionError(f"Late state reproduction drifted: {stat_differences}")

    diagnostic = DensityDiagnosticSampler(target_hw, crop, seed)
    with torch.inference_mode():
        unchanged = phase2.comfy.sample.sample_custom(
            model,
            accepted.clone(),
            1.0,
            diagnostic,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive,
            negative,
            torch.zeros_like(accepted),
            disable_pbar=True,
            seed=seed,
        )
    no_update_difference = phase2c.tensor_difference(unchanged, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = diagnostic.source_outputs["C_2048_DENSE"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in diagnostic.outputs.items():
        comparisons[name] = phase2d.comparison(output, dense_crop)
        with torch.inference_mode():
            pixels = vae.decode(output).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")
        canvas = dense_full.clone()
        canvas[:, :, crop.y:crop.y2, crop.x:crop.x2] = output
        with torch.inference_mode():
            context_pixels = vae.decode(canvas).cpu()
        phase2.save_pixels(
            context_pixels, args.output_dir / f"{name}_IN_DENSE_CONTEXT.png"
        )
        phase2.save_heatmap(
            (output - dense_crop).square().mean(dim=1, keepdim=True).sqrt(),
            args.output_dir / f"{name}_ERROR_MAGNITUDE.png",
            (512, 512),
        )
    with torch.inference_mode():
        dense_crop_pixels = vae.decode(dense_crop).cpu()
        dense_full_pixels = vae.decode(dense_full).cpu()
    phase2.save_pixels(dense_crop_pixels, args.output_dir / "D_DENSE_REFERENCE_CROP.png")
    phase2.save_pixels(dense_full_pixels, args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": phase2.PROMPT,
            "seed": seed,
            "cfg": 1.0,
            "trajectory_updates_reproduced": diagnostic_step,
            "diagnostic_step": diagnostic_step,
            "sigma": float(diagnostic_sigma),
            "sampler_updates_during_diagnostic": 0,
            "target_latent_hw": list(target_hw),
            "crop": crop.__dict__,
            "local_rope_options": phase2.rope_for_crop(crop),
            "global_densities": {name: list(hw) for name, hw in DENSITIES},
            "modified_blocks": "all 5 double and all 20 single blocks",
            "output_space_global_fusion": False,
        },
        "state_reproduction": {
            "expected_stats": expected_stats,
            "reproduced_stats": reproduced_stats,
            "absolute_stat_differences": stat_differences,
            "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted),
            "no_update_difference": no_update_difference,
        },
        "layouts_and_attention_context_work": diagnostic.layouts,
        "calls": diagnostic.calls,
        "positions": {
            name: probe.position_records for name, probe in diagnostic.probes.items()
        },
        "comparisons_vs_dense_crop": comparisons,
        "decoded_semantic_observations": {
            "A_512_COMPACT": {
                "train_continuity": "continuous train, but carriage spacing and extent differ visibly from dense",
                "bridge_geometry": "continuous deck and cables, with local alignment differing from dense",
                "lighthouse_duplication": "extra small center-right lighthouse remains",
                "stone_structure_divergence": "prominent dark center-right stone structure remains",
                "horizon_water_agreement": "coherent but contains the false lighthouse/stone pair",
            },
            "B_1152_INTERMEDIATE": {
                "train_continuity": "modest carriage-spacing improvement; still visibly different from dense",
                "bridge_geometry": "incremental deck/cable alignment improvement over 512",
                "lighthouse_duplication": "extra small center-right lighthouse remains",
                "stone_structure_divergence": "prominent dark center-right stone structure remains",
                "horizon_water_agreement": "incremental local improvement; false object pair remains",
            },
            "C_2048_DENSE": {
                "train_continuity": "closest train alignment and continuity to dense reference",
                "bridge_geometry": "substantially closest deck/cable geometry to dense reference",
                "lighthouse_duplication": "extra center-right lighthouse removed",
                "stone_structure_divergence": "reduced dark tower-like structure remains; not identical to dense",
                "horizon_water_agreement": "best agreement of the three densities",
            },
            "scope": "manual inspection of decoded one-evaluation denoised estimates; not an automated object metric",
        },
        "dense_reference_crop": phase2.stats(dense_crop),
        "attention_work_scope": "Q*K products for the changed local attention context only; excludes global source forwards, projections, MLPs, residuals, and all other model FLOPs",
        "outputs": {
            name: str(args.output_dir / f"{name}.png")
            for name in diagnostic.outputs
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "state_stat_max_difference": max(stat_differences.values()),
                "no_update_max_abs": no_update_difference["max_abs"],
                "densities": {
                    name: {
                        "global_tokens": diagnostic.layouts[name]["global_generated_tokens"],
                        "q_by_k": diagnostic.layouts[name]["q_by_k"],
                        "relative_attention_context_work": diagnostic.layouts[name]["relative_to_512_compact"],
                        "rms": comparisons[name]["absolute"]["rms"],
                        "low_frequency_rms": comparisons[name]["low_frequency"]["rms"],
                        "prediction_rms": comparisons[name]["value_stats"]["rms"],
                    }
                    for name, _ in DENSITIES
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
