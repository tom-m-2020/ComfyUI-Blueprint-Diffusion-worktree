"""Late-evaluation dense-K/V versus compact-K/V diagnostic.

Reproduces the Candidate-2 trajectory state before evaluation 2, then performs
one no-update comparison on the affected center crop. Dense and compact context
use the identical external-K/V integration code; only global source/density and
its corresponding full-canvas RoPE differ.
"""

from __future__ import annotations

import argparse
import hashlib
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


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_four_step_results" / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate2_dense_vs_compact_results",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def tensor_hash(value: torch.Tensor) -> str:
    data = value.detach().contiguous().float().cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


class DiagnosticSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, global_hw, crop, seed) -> None:
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crop = crop
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.compact_probe = phase2c.OneEvaluationContextProbe()
        self.dense_probe = phase2c.OneEvaluationContextProbe()
        self.integration_control: dict[str, Any] = {}

    def run_call(self, model, value, sigma, options, base_options, name, role):
        merged = phase2e.merge_options(base_options, options)
        prediction, call = phase2e.model_call(
            model, value, sigma, merged, self.seed, role, self.crop if "LOCAL" in name else None
        )
        self.outputs[name] = prediction.detach().float().cpu()
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
            raise ValueError("This no-update T2I diagnostic does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]

        self.run_call(
            model,
            local_input,
            sigma,
            phase2c.model_options(
                "local_only", phase2.rope_for_crop(crop), phase2c.OneEvaluationContextProbe(), "ordinary"
            ),
            base_options,
            "A_LOCAL_ONLY",
            "local_only",
        )

        compact_input = F.interpolate(
            noise, size=self.global_hw, mode="bilinear", align_corners=False
        )
        self.run_call(
            model,
            compact_input,
            sigma,
            phase2c.model_options(
                "compact_global",
                phase2.rope_for_global(*self.target_hw, *self.global_hw),
                self.compact_probe,
                "capture",
            ),
            base_options,
            "COMPACT_GLOBAL_CAPTURE",
            "compact_global_capture",
        )
        self.run_call(
            model,
            local_input,
            sigma,
            phase2c.model_options(
                "local_compact_context",
                phase2.rope_for_crop(crop),
                self.compact_probe,
                "context",
            ),
            base_options,
            "B_LOCAL_COMPACT_GLOBAL_KV",
            "local_with_external_kv",
        )
        self.compact_probe.assert_complete()

        # This ordinary full-canvas prediction is both the dense reference and
        # the sole source of dense generated K/V.
        self.run_call(
            model,
            noise,
            sigma,
            phase2c.model_options(
                "dense_global", {}, self.dense_probe, "capture"
            ),
            base_options,
            "DENSE_FULL_CANVAS_CAPTURE_REFERENCE",
            "dense_full_canvas_capture",
        )
        self.run_call(
            model,
            local_input,
            sigma,
            phase2c.model_options(
                "local_dense_context",
                phase2.rope_for_crop(crop),
                self.dense_probe,
                "context",
            ),
            base_options,
            "C_LOCAL_DENSE_GLOBAL_KV",
            "local_with_external_kv",
        )
        self.dense_probe.assert_complete()

        compact_layouts = self.compact_probe.context_records
        dense_layouts = self.dense_probe.context_records
        if len(compact_layouts) != 25 or len(dense_layouts) != 25:
            raise AssertionError((len(compact_layouts), len(dense_layouts)))
        for compact, dense in zip(compact_layouts, dense_layouts):
            invariant_keys = (
                "block_type",
                "block_index",
                "text_tokens",
                "local_generated_queries",
                "query_tokens_total",
            )
            if any(compact[key] != dense[key] for key in invariant_keys):
                raise AssertionError(f"Integration path mismatch: {compact} vs {dense}")
        self.integration_control = {
            "same_integration_class": True,
            "same_context_patch_method": True,
            "same_text_attention_restore_method": True,
            "same_local_input_object": True,
            "same_sigma": True,
            "same_conditioning_lifecycle": True,
            "same_25_blocks": True,
            "compact_context_records": compact_layouts,
            "dense_context_records": dense_layouts,
            "compact_global_generated_tokens": compact_layouts[0]["global_generated_kv"],
            "dense_global_generated_tokens": dense_layouts[0]["global_generated_kv"],
            "compact_q_by_k": compact_layouts[0]["q_by_k"],
            "dense_q_by_k": dense_layouts[0]["q_by_k"],
        }

        # Diagnostic only: return the accepted latent unchanged.
        return noise


def dry_run() -> None:
    compact = [1536, 2048]
    dense = [1536, 3584]
    if compact[0] != dense[0] or compact[1] != 512 + 1024 + 512 or dense[1] != 512 + 1024 + 2048:
        raise AssertionError((compact, dense))
    print(json.dumps({"compact_q_by_k": compact, "dense_q_by_k": dense, "sampler_updates": 0}, indent=2))


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
    global_hw = (256 // 16, 512 // 16)
    crop_hw = (512 // 16, 512 // 16)
    overlap_pixels = 128
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap_pixels // 16)
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
        "context", target_hw, global_hw, crops, reproduction_trace, seed
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
    expected_stats = previous["trajectories"]["C_GLOBAL_CONTEXT_TILED"]["evaluation_records"][diagnostic_step]["accepted_latent_before"]
    reproduced_stats = reproduction_trace.evaluations[-1]["accepted_latent_after"]
    reproduced_output_stats = phase2.stats(accepted)
    stat_differences = {
        key: abs(float(reproduced_stats[key]) - float(expected_stats[key]))
        for key in ("norm", "rms", "mean", "std", "min", "max")
    }
    if max(stat_differences.values()) > 1e-6:
        raise AssertionError(f"Late state reproduction drifted: {stat_differences}")

    diagnostic = DiagnosticSampler(target_hw, global_hw, crop, seed)
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
        raise AssertionError(f"Diagnostic updated accepted latent: {no_update_difference}")

    dense_full = diagnostic.outputs["DENSE_FULL_CANVAS_CAPTURE_REFERENCE"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    variants = {
        "A_LOCAL_ONLY": diagnostic.outputs["A_LOCAL_ONLY"],
        "B_LOCAL_COMPACT_GLOBAL_KV": diagnostic.outputs["B_LOCAL_COMPACT_GLOBAL_KV"],
        "C_LOCAL_DENSE_GLOBAL_KV": diagnostic.outputs["C_LOCAL_DENSE_GLOBAL_KV"],
        "D_DENSE_REFERENCE_CROP": dense_crop,
    }
    for name, prediction in variants.items():
        with torch.inference_mode():
            pixels = vae.decode(prediction).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")
    with torch.inference_mode():
        dense_pixels = vae.decode(dense_full).cpu()
    phase2.save_pixels(dense_pixels, args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")

    # Same surrounding dense prediction for every crop diagnostic. The hard
    # insertion edge is visualization only, not fusion or a sampler result.
    for name, prediction in list(variants.items())[:3]:
        canvas = dense_full.clone()
        canvas[:, :, crop.y:crop.y2, crop.x:crop.x2] = prediction
        with torch.inference_mode():
            pixels = vae.decode(canvas).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}_IN_DENSE_CONTEXT.png")

    phase2.save_heatmap(
        (variants["B_LOCAL_COMPACT_GLOBAL_KV"] - dense_crop).square().mean(dim=1, keepdim=True).sqrt(),
        args.output_dir / "B_COMPACT_ERROR_MAGNITUDE.png",
        (512, 512),
    )
    phase2.save_heatmap(
        (variants["C_LOCAL_DENSE_GLOBAL_KV"] - dense_crop).square().mean(dim=1, keepdim=True).sqrt(),
        args.output_dir / "C_DENSE_KV_ERROR_MAGNITUDE.png",
        (512, 512),
    )

    comparisons = {
        "local_only_vs_dense_crop": phase2d.comparison(variants["A_LOCAL_ONLY"], dense_crop),
        "compact_kv_vs_dense_crop": phase2d.comparison(variants["B_LOCAL_COMPACT_GLOBAL_KV"], dense_crop),
        "dense_kv_vs_dense_crop": phase2d.comparison(variants["C_LOCAL_DENSE_GLOBAL_KV"], dense_crop),
        "compact_kv_vs_local_only": phase2d.comparison(variants["B_LOCAL_COMPACT_GLOBAL_KV"], variants["A_LOCAL_ONLY"]),
        "dense_kv_vs_local_only": phase2d.comparison(variants["C_LOCAL_DENSE_GLOBAL_KV"], variants["A_LOCAL_ONLY"]),
        "dense_kv_vs_compact_kv": phase2d.comparison(variants["C_LOCAL_DENSE_GLOBAL_KV"], variants["B_LOCAL_COMPACT_GLOBAL_KV"]),
    }
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
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "compact_global_latent_hw": list(global_hw),
            "crop": crop.__dict__,
            "crop_image_hw": [512, 512],
            "local_rope_options": phase2.rope_for_crop(crop),
            "compact_rope_options": phase2.rope_for_global(*target_hw, *global_hw),
            "dense_rope_options": {},
            "modified_blocks": "all 5 double and all 20 single blocks",
            "output_space_global_fusion": False,
        },
        "state_reproduction": {
            "expected_stats": expected_stats,
            "reproduced_stats": reproduced_stats,
            "reproduced_output_stats": reproduced_output_stats,
            "absolute_stat_differences": stat_differences,
            "accepted_latent_sha256_float32": tensor_hash(accepted),
            "no_update_difference": no_update_difference,
            "reproduction_evaluations": reproduction_trace.evaluations,
        },
        "integration_control": diagnostic.integration_control,
        "calls": diagnostic.calls,
        "compact_positions": diagnostic.compact_probe.position_records,
        "dense_positions": diagnostic.dense_probe.position_records,
        "comparisons": comparisons,
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in variants},
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
                "compact_q_by_k": diagnostic.integration_control["compact_q_by_k"],
                "dense_q_by_k": diagnostic.integration_control["dense_q_by_k"],
                "comparisons": {
                    key: {
                        "rms": value["absolute"]["rms"],
                        "low_frequency_rms": value["low_frequency"]["rms"],
                    }
                    for key, value in comparisons.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
