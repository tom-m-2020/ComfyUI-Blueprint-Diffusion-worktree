"""Minimal four-step FLUX.2 Candidate-2 trajectory experiment.

Experiment only: dense, ordinary tiled, and fresh same-evaluation compact-global
context tiled Euler trajectories. No stale cache, sparse execution, block
selection, output-space global fusion, or production integration.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as phase2c
import flux2_candidate2_all_crop_assembly_probe as phase2d


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate2_four_step_results",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


@dataclass
class TrajectoryTrace:
    name: str
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    predictions: dict[int, torch.Tensor] = field(default_factory=dict)
    crop_predictions: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    accepted_latents_before: dict[int, torch.Tensor] = field(default_factory=dict)
    accepted_latents_after: dict[int, torch.Tensor] = field(default_factory=dict)
    updates: int = 0
    seconds: float = 0.0
    peak_allocated_gib: float = 0.0
    final_context_capture_survived: bool | None = None


def merge_options(base: dict[str, Any], experimental: dict[str, Any]) -> dict[str, Any]:
    merged = phase2.comfy.model_patcher.create_model_options_clone(base)
    transformer = merged.setdefault("transformer_options", {})
    transformer.update(experimental["transformer_options"])
    return merged


def model_call(model, value, sigma, options, seed, role, crop=None):
    started = time.perf_counter()
    prediction = model(
        value,
        sigma.expand(value.shape[0]),
        model_options=options,
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record = {
        "role": role,
        "sigma": float(sigma),
        "input_latent_hw": list(value.shape[-2:]),
        "image_tokens": int(value.shape[-2] * value.shape[-1]),
        "input": phase2.stats(value),
        "prediction": phase2.stats(prediction),
        "seconds": time.perf_counter() - started,
    }
    if crop is not None:
        record["crop"] = crop.__dict__
    return prediction, record


def assemble_live(predictions, crops, target_hw):
    output = torch.zeros(
        (predictions[0].shape[0], predictions[0].shape[1], *target_hw),
        dtype=predictions[0].dtype,
        device=predictions[0].device,
    )
    coverage = torch.zeros(
        (predictions[0].shape[0], 1, *target_hw),
        dtype=torch.float32,
        device=predictions[0].device,
    )
    for prediction, crop in zip(predictions, crops):
        weight = phase2.crop_weight(crop, crops, output.device)[None, None]
        output[:, :, crop.y:crop.y2, crop.x:crop.x2] += prediction * weight
        coverage[:, :, crop.y:crop.y2, crop.x:crop.x2] += weight
    if float(coverage.min()) <= 0:
        raise RuntimeError(f"Incomplete live assembly coverage: {phase2.stats(coverage)}")
    return output / coverage, coverage


class FourStepSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, variant, target_hw, global_hw, crops, trace, seed) -> None:
        self.variant = variant
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crops = crops
        self.trace = trace
        self.seed = seed
        self.previous_kv_refs: list[weakref.ReferenceType] = []

    def ordinary_crop_calls(self, model, x, sigma, base_options, prefix):
        predictions = []
        calls = []
        for crop in self.crops:
            options = merge_options(
                base_options,
                phase2c.model_options(
                    f"{prefix}_crop_{crop.index}",
                    phase2.rope_for_crop(crop),
                    phase2c.OneEvaluationContextProbe(),
                    "ordinary",
                ),
            )
            prediction, call = model_call(
                model,
                x[:, :, crop.y:crop.y2, crop.x:crop.x2],
                sigma,
                options,
                self.seed,
                "local",
                crop,
            )
            predictions.append(prediction)
            calls.append(call)
        return predictions, calls

    def context_crop_calls(self, model, x, sigma, base_options, step):
        gc.collect()
        previous_capture_survived = any(reference() is not None for reference in self.previous_kv_refs)
        if previous_capture_survived:
            raise AssertionError(f"Previous-evaluation compact K/V survived into step {step}.")

        probe = phase2c.OneEvaluationContextProbe()
        global_input = F.interpolate(
            x, size=self.global_hw, mode="bilinear", align_corners=False
        )
        global_options = merge_options(
            base_options,
            phase2c.model_options(
                f"compact_global_step_{step}",
                phase2.rope_for_global(*self.target_hw, *self.global_hw),
                probe,
                "capture",
            ),
        )
        global_prediction, global_call = model_call(
            model,
            global_input,
            sigma,
            global_options,
            self.seed,
            "compact_global",
        )
        if len(probe.capture_records) != 25:
            raise AssertionError(f"Expected one 25-layer capture, got {len(probe.capture_records)}.")
        captured_ids = {
            key: (id(entry["k"]), id(entry["v"]))
            for key, entry in probe.global_kv.items()
        }
        weak_kv = [
            weakref.ref(tensor)
            for entry in probe.global_kv.values()
            for tensor in (entry["k"], entry["v"])
        ]

        predictions = []
        calls = [global_call]
        per_crop_identity = []
        for crop in self.crops:
            before_records = len(probe.context_records)
            context_options = merge_options(
                base_options,
                phase2c.model_options(
                    f"context_step_{step}_crop_{crop.index}",
                    phase2.rope_for_crop(crop),
                    probe,
                    "context",
                ),
            )
            prediction, call = model_call(
                model,
                x[:, :, crop.y:crop.y2, crop.x:crop.x2],
                sigma,
                context_options,
                self.seed,
                "local_with_compact_global_context",
                crop,
            )
            call["context_record_range"] = [before_records, len(probe.context_records)]
            current_ids = {
                key: (id(entry["k"]), id(entry["v"]))
                for key, entry in probe.global_kv.items()
            }
            same_objects = current_ids == captured_ids
            if not same_objects or len(probe.capture_records) != 25:
                raise AssertionError(f"Global K/V changed or recaptured for step {step} crop {crop.index}.")
            per_crop_identity.append(
                {
                    "crop_index": crop.index,
                    "same_kv_objects": same_objects,
                    "capture_record_count": len(probe.capture_records),
                }
            )
            predictions.append(prediction)
            calls.append(call)
            del context_options
        probe.assert_complete()
        context_layout = {
            "global_generated_tokens": 512,
            "local_generated_queries": 1024,
            "query_tokens_total": 1536,
            "kv_tokens_total": 2048,
            "q_by_k": [1536, 2048],
            "modified_blocks": {"double": 5, "single": 20},
            "capture_records": len(probe.capture_records),
            "context_records": len(probe.context_records),
            "previous_capture_survived_before_evaluation": previous_capture_survived,
            "same_capture_all_crops": all(item["same_kv_objects"] for item in per_crop_identity),
            "per_crop_identity": per_crop_identity,
            "accepted_latent_object_id": id(x),
            "global_input": phase2.stats(global_input),
            "global_prediction": phase2.stats(global_prediction),
        }

        # No cross-step cache: remove every strong reference owned by the probe.
        probe.global_kv.clear()
        del global_options, global_prediction, global_input, probe
        self.previous_kv_refs = weak_kv
        return predictions, calls, context_layout

    def predict(self, model, x, sigma, base_options, step):
        self.trace.accepted_latents_before[step] = x.detach().float().cpu()
        accepted_stats = phase2.stats(x)
        context_layout = None
        if self.variant == "dense":
            options = merge_options(
                base_options,
                phase2c.model_options(
                    f"dense_step_{step}", {}, phase2c.OneEvaluationContextProbe(), "ordinary"
                ),
            )
            assembled, call = model_call(
                model, x, sigma, options, self.seed, "dense"
            )
            calls = [call]
            crops = [
                assembled[:, :, crop.y:crop.y2, crop.x:crop.x2]
                for crop in self.crops
            ]
            coverage = torch.ones_like(x[:, :1])
            overlap = None
        elif self.variant == "tiled":
            crops, calls = self.ordinary_crop_calls(
                model, x, sigma, base_options, f"tiled_step_{step}"
            )
            assembled, coverage = assemble_live(crops, self.crops, self.target_hw)
            overlap = phase2d.overlap_disagreement(crops, self.crops)
        elif self.variant == "context":
            crops, calls, context_layout = self.context_crop_calls(
                model, x, sigma, base_options, step
            )
            assembled, coverage = assemble_live(crops, self.crops, self.target_hw)
            overlap = phase2d.overlap_disagreement(crops, self.crops)
        else:
            raise ValueError(self.variant)

        if not bool(assembled.isfinite().all()):
            raise FloatingPointError(f"Nonfinite {self.variant} prediction at step {step}.")
        record = {
            "step": step,
            "sigma": float(sigma),
            "evaluation_identity": f"{self.variant}:euler:{step}:sigma={float(sigma):.9g}",
            "accepted_latent_before": accepted_stats,
            "assembled_prediction": phase2.stats(assembled),
            "coverage": phase2.stats(coverage),
            "overlap_disagreement": overlap,
            "model_forwards": len(calls),
            "executed_image_tokens": sum(call["image_tokens"] for call in calls),
            "calls": calls,
            "one_assembled_prediction_for_update": True,
        }
        if context_layout is not None:
            record["compact_global_context"] = context_layout
        self.trace.evaluations.append(record)
        self.trace.predictions[step] = assembled.detach().float().cpu()
        self.trace.crop_predictions[step] = [value.detach().float().cpu() for value in crops]
        return assembled

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
            raise ValueError("This T2I trajectory experiment does not use a mask.")
        x = noise
        total = len(sigmas) - 1
        for step in range(total):
            sigma = sigmas[step]
            prediction = self.predict(
                model, x, sigma, extra_args["model_options"], step
            )
            derivative = (x - prediction) / sigma
            x = x + derivative * (sigmas[step + 1] - sigma)
            self.trace.updates += 1
            self.trace.accepted_latents_after[step] = x.detach().float().cpu()
            self.trace.evaluations[-1]["accepted_latent_after"] = phase2.stats(x)
            self.trace.evaluations[-1]["sampler_updates_consuming_prediction"] = 1
            if callback is not None:
                callback(step, prediction, x, total)
        if self.variant == "context":
            gc.collect()
            self.trace.final_context_capture_survived = any(
                reference() is not None for reference in self.previous_kv_refs
            )
            if self.trace.final_context_capture_survived:
                raise AssertionError("Final-evaluation compact K/V survived trajectory cleanup.")
        return x


def run_trajectory(model, noise, positive, negative, sigmas, sampler, seed):
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
            seed=seed,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        sampler.trace.peak_allocated_gib = torch.cuda.max_memory_allocated() / GIB
    sampler.trace.seconds = time.perf_counter() - started
    return output.detach().float().cpu()


def dry_run() -> None:
    sigmas = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0])
    x = torch.ones(1, 1, 2, 2)
    predictions = [torch.full_like(x, value) for value in (0.8, 0.6, 0.4, 0.2)]
    states = []
    for step, prediction in enumerate(predictions):
        sigma = sigmas[step]
        x = x + ((x - prediction) / sigma) * (sigmas[step + 1] - sigma)
        states.append(float(x.mean()))
    if len(states) != 4 or not bool(x.isfinite().all()):
        raise AssertionError(states)
    print(json.dumps({"updates": 4, "states": states}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    target_hw = (height // 16, width // 16)
    global_hw = (256 // 16, 512 // 16)
    crop_hw = (512 // 16, 512 // 16)
    overlap_pixels = 128
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap_pixels // 16)
    sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()

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
    outputs: dict[str, torch.Tensor] = {}
    traces: dict[str, TrajectoryTrace] = {}
    for name, variant in (
        ("A_DENSE", "dense"),
        ("B_TILED_ONLY", "tiled"),
        ("C_GLOBAL_CONTEXT_TILED", "context"),
    ):
        print(f"Running {name}...", flush=True)
        trace = TrajectoryTrace(name)
        sampler = FourStepSampler(variant, target_hw, global_hw, crops, trace, seed)
        outputs[name] = run_trajectory(
            model, noise, positive, negative, sigmas, sampler, seed
        )
        traces[name] = trace

    # Cross-variant diagnostics at each accepted evaluation identity.
    evaluations = []
    for step in range(4):
        dense_prediction = traces["A_DENSE"].predictions[step]
        tiled_prediction = traces["B_TILED_ONLY"].predictions[step]
        context_prediction = traces["C_GLOBAL_CONTEXT_TILED"].predictions[step]
        per_crop = []
        for crop_index, crop in enumerate(crops):
            dense_crop = dense_prediction[
                :, :, crop.y:crop.y2, crop.x:crop.x2
            ]
            tiled_crop = traces["B_TILED_ONLY"].crop_predictions[step][crop_index]
            context_crop = traces["C_GLOBAL_CONTEXT_TILED"].crop_predictions[step][crop_index]
            per_crop.append(
                {
                    "crop": crop.__dict__,
                    "tiled_only_vs_dense_crop": phase2d.comparison(tiled_crop, dense_crop),
                    "global_context_vs_dense_crop": phase2d.comparison(context_crop, dense_crop),
                    "context_vs_local_only": phase2d.comparison(context_crop, tiled_crop),
                }
            )
        evaluations.append(
            {
                "step": step,
                "sigma": float(sigmas[step]),
                "dense_prediction": phase2.stats(dense_prediction),
                "tiled_only_assembled_prediction": phase2.stats(tiled_prediction),
                "context_tiled_assembled_prediction": phase2.stats(context_prediction),
                "tiled_only_vs_dense": phase2d.comparison(tiled_prediction, dense_prediction),
                "global_context_vs_dense": phase2d.comparison(context_prediction, dense_prediction),
                "per_crop": per_crop,
                "tiled_only_overlap_disagreement": traces["B_TILED_ONLY"].evaluations[step]["overlap_disagreement"],
                "global_context_overlap_disagreement": traces["C_GLOBAL_CONTEXT_TILED"].evaluations[step]["overlap_disagreement"],
                "coverage": {
                    "tiled_only": traces["B_TILED_ONLY"].evaluations[step]["coverage"],
                    "global_context": traces["C_GLOBAL_CONTEXT_TILED"].evaluations[step]["coverage"],
                },
                "model_forwards": {
                    "dense": traces["A_DENSE"].evaluations[step]["model_forwards"],
                    "tiled_only": traces["B_TILED_ONLY"].evaluations[step]["model_forwards"],
                    "global_context": traces["C_GLOBAL_CONTEXT_TILED"].evaluations[step]["model_forwards"],
                },
                "context_layout": traces["C_GLOBAL_CONTEXT_TILED"].evaluations[step]["compact_global_context"],
            }
        )

    # Final samples and every model prediction/denoised estimate.
    for name, output in outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(output).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}_FINAL.png")
        for step in range(4):
            with torch.inference_mode():
                estimate = vae.decode(traces[name].predictions[step]).cpu()
            phase2.save_pixels(
                estimate,
                args.output_dir / f"{name}_STEP_{step:02d}_DENOISED.png",
            )

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": phase2.PROMPT,
            "seed": seed,
            "cfg": 1.0,
            "sampler": "Euler, zero churn, exactly one assembled prediction/update per evaluation",
            "steps": 4,
            "sigmas": sigmas.tolist(),
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "global_image_hw": [256, 512],
            "global_latent_hw": list(global_hw),
            "crop_image_hw": [512, 512],
            "crop_latent_hw": list(crop_hw),
            "overlap_pixels": overlap_pixels,
            "crops": [crop.__dict__ for crop in crops],
            "global_rope_options": phase2.rope_for_global(*target_hw, *global_hw),
            "crop_rope_options": [phase2.rope_for_crop(crop) for crop in crops],
            "context_policy": "fresh same-current-latent same-sigma capture every evaluation; no cross-step reuse",
            "output_space_global_fusion": False,
        },
        "evaluations": evaluations,
        "trajectories": {
            name: {
                "seconds": trace.seconds,
                "peak_allocated_gib": trace.peak_allocated_gib,
                "sampler_updates": trace.updates,
                "final_context_capture_survived": trace.final_context_capture_survived,
                "final": phase2.stats(outputs[name]),
                "evaluation_records": trace.evaluations,
            }
            for name, trace in traces.items()
        },
        "final_latent_comparisons": {
            "tiled_only_vs_dense": phase2d.comparison(
                outputs["B_TILED_ONLY"], outputs["A_DENSE"]
            ),
            "global_context_vs_dense": phase2d.comparison(
                outputs["C_GLOBAL_CONTEXT_TILED"], outputs["A_DENSE"]
            ),
        },
        "outputs": {
            name: str(args.output_dir / f"{name}_FINAL.png")
            for name in outputs
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "steps": [
                    {
                        "step": item["step"],
                        "sigma": item["sigma"],
                        "tiled_rms": item["tiled_only_vs_dense"]["absolute"]["rms"],
                        "context_rms": item["global_context_vs_dense"]["absolute"]["rms"],
                        "tiled_overlap": item["tiled_only_overlap_disagreement"]["aggregate_rms"],
                        "context_overlap": item["global_context_overlap_disagreement"]["aggregate_rms"],
                    }
                    for item in evaluations
                ],
                "finals": {name: phase2.stats(value) for name, value in outputs.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
