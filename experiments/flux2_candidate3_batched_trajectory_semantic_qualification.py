"""Phase 6d: semantic and performance qualification of B=2 local-crop scheduling."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d
import flux2_candidate3_hard_global_anchor as phase3
from flux2_candidate3_runtime_coordinate_batch_probe import (
    OPTION_KEY,
    clone_options as batch_options,
    scoped_coordinate_override,
)


OUTPUT = ROOT / "experiments" / "flux2_candidate3_batched_trajectory_results"
GIB = 1024 ** 3
TARGET_HW = (32, 64)
GLOBAL_HW = (24, 48)
SEED_BRIDGE = 20260829
SEED_PERSON = 20260831
PERSON_PROMPT = (
    "A cinematic wide-angle photograph of exactly one large full-body woman "
    "standing centered in the foreground, exactly one red vintage car on the "
    "left and exactly one tall green tree on the right, continuous ground, "
    "coherent perspective, no duplicate people, cars, or trees"
)
SCENES = (
    ("bridge_train", phase2.PROMPT, SEED_BRIDGE),
    ("person_car_tree", PERSON_PROMPT, SEED_PERSON),
)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class BatchedTerminalReleaseSampler(phase3.Candidate3Sampler):
    """Same Candidate-3 sampler; only local crop model-call scheduling changes."""

    def local_prediction(self, model, h, sigma, base_options, step):
        predictions = []
        calls = []
        batch_calls = []
        accepted_storage = h.untyped_storage().data_ptr()
        for start in range(0, len(self.crops), 2):
            batch_crops = self.crops[start:start + 2]
            views = [h[:, :, crop.y:crop.y2, crop.x:crop.x2] for crop in batch_crops]
            if any(view.untyped_storage().data_ptr() != accepted_storage for view in views):
                raise AssertionError("A batched crop did not read immutable accepted H storage.")
            batch_input = torch.cat(views, dim=0)
            offsets = [(float(crop.y), float(crop.x)) for crop in batch_crops]
            options = batch_options(base_options, {OPTION_KEY: offsets})
            cuda_sync()
            started = time.perf_counter()
            prediction_batch = model(
                batch_input,
                sigma.expand(len(batch_crops)),
                model_options=options,
                seed=self.seed,
            )
            cuda_sync()
            seconds = time.perf_counter() - started
            batch_calls.append({
                "batch_index": len(batch_calls),
                "batch_size": len(batch_crops),
                "crop_indices": [crop.index for crop in batch_crops],
                "offsets_yx": offsets,
                "seconds": seconds,
                "input_shape": list(batch_input.shape),
                "prediction_shape": list(prediction_batch.shape),
                "executed_image_tokens": int(len(batch_crops) * 32 * 32),
            })
            for index, crop in enumerate(batch_crops):
                prediction = prediction_batch[index:index + 1]
                predictions.append(prediction)
                calls.append({
                    "role": "local",
                    "sigma": float(sigma),
                    "input_latent_hw": [32, 32],
                    "image_tokens": 32 * 32,
                    "rope_options": {"shift_y": float(crop.y), "shift_x": float(crop.x)},
                    "crop": crop.__dict__,
                    "prediction": phase2.stats(prediction),
                    "batch_call_index": len(batch_calls) - 1,
                    "batch_element": index,
                    "crop_read_only_before_atomic_acceptance": True,
                })
        assembled, coverage = phase3.assemble_live(predictions, self.crops, self.target_hw)
        overlap = phase2d.overlap_disagreement(predictions, self.crops)
        self.trace.crop_predictions[step] = [item.detach().float().cpu() for item in predictions]
        self.trace.evaluations_pending_batches = batch_calls
        return assembled, calls, coverage, overlap

    def sample(self, *args, **kwargs):
        result = super().sample(*args, **kwargs)
        # The base lifecycle records three logical crop results. Add actual scheduling.
        for evaluation, batches in zip(self.trace.evaluations, self.trace.batch_calls_by_step):
            evaluation["actual_local_model_calls"] = len(batches)
            evaluation["local_batches"] = batches
            evaluation["actual_model_forwards"] = 1 + len(batches)
        return result


class BatchedTrace(phase3.TrajectoryTrace):
    def __init__(self, name: str, variant: str):
        super().__init__(name, variant)
        self.batch_calls_by_step: list[list[dict[str, Any]]] = []
        self.evaluations_pending_batches: list[dict[str, Any]] = []


def run_sampler(model, noise, positive, negative, sigmas, sampler, seed, batched):
    gc.collect()
    cuda_sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    baseline_reserved = torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
    started = time.perf_counter()
    diffusion = model.model.diffusion_model
    context = scoped_coordinate_override(diffusion) if batched else torch.no_grad()
    with context:
        # Capture each batch list immediately after the local call returns.
        original_local = sampler.local_prediction
        if batched:
            def local_and_capture(*args, **kwargs):
                result = original_local(*args, **kwargs)
                sampler.trace.batch_calls_by_step.append(sampler.trace.evaluations_pending_batches)
                return result
            sampler.local_prediction = local_and_capture
        with torch.inference_mode():
            output = phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive,
                negative, torch.zeros_like(noise), disable_pbar=True, seed=seed,
            )
    cuda_sync()
    return output.detach().float().cpu(), {
        "sampling_wall_seconds": time.perf_counter() - started,
        "baseline_allocated_gib": baseline_allocated / GIB,
        "baseline_reserved_gib": baseline_reserved / GIB,
        "peak_allocated_gib": (
            torch.cuda.max_memory_allocated() / GIB if torch.cuda.is_available() else 0.0
        ),
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved() / GIB if torch.cuda.is_available() else 0.0
        ),
    }


def compare_traces(sequential, batched):
    intervals = []
    for step in range(4):
        seq_eval = sequential.evaluations[step]
        bat_eval = batched.evaluations[step]
        intervals.append({
            "step": step,
            "sigma": seq_eval["sigma"],
            "assembled_x0_H": phase3.tensor_difference(
                batched.h_predictions[step], sequential.h_predictions[step]
            ),
            "accepted_H": phase3.tensor_difference(
                batched.h_after[step], sequential.h_after[step]
            ),
            "accepted_G": phase3.tensor_difference(
                batched.g_after[step], sequential.g_after[step]
            ),
            "batched_D_H_minus_G": phase3.tensor_difference(
                phase3.block_dct_restrict(batched.h_after[step]), batched.g_after[step]
            ),
            "sequential_projection_rms": seq_eval["projection_available"]["rms"],
            "batched_projection_rms": bat_eval["projection_available"]["rms"],
            "projection_difference": phase3.tensor_difference(
                batched.projections[step], sequential.projections[step]
            ),
            "sequential_overlap": seq_eval["overlap_disagreement"],
            "batched_overlap": bat_eval["overlap_disagreement"],
            "actual_batched_local_calls": bat_eval["actual_local_model_calls"],
        })
    return intervals


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    crops = phase2.crops_for_canvas(32, 64, 32, 32, 8)
    sigmas = phase2.get_schedule(4, math.prod(TARGET_HW)).float().clone()
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "target_latent_hw": list(TARGET_HW),
            "global_latent_hw": list(GLOBAL_HW),
            "crops": [crop.__dict__ for crop in crops],
            "batch_size": 2,
            "sigmas": sigmas.tolist(),
            "lifecycle": "hard block-DCT coupling nonterminal; terminal release",
            "coordinate_override": "scoped runtime process_img override; no core/production edit",
        },
        "scenes": {},
    }
    for scene_name, prompt, seed in SCENES:
        print(f"Encoding {scene_name}...", flush=True)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        noise = torch.randn((1, 128, *TARGET_HW), generator=torch.Generator().manual_seed(seed))

        sequential_trace = phase3.TrajectoryTrace("SEQUENTIAL", "terminal_release")
        sequential_sampler = phase3.Candidate3Sampler(
            "terminal_release", TARGET_HW, GLOBAL_HW, crops, sequential_trace, seed,
            phase3.block_dct_restrict, phase3.block_dct_prolong, "block_dct_24x48",
        )
        print(f"Running {scene_name} sequential...", flush=True)
        sequential_output, sequential_perf = run_sampler(
            model, noise, positive, negative, sigmas, sequential_sampler, seed, False
        )

        batched_trace = BatchedTrace("BATCHED_B2", "terminal_release")
        batched_sampler = BatchedTerminalReleaseSampler(
            "terminal_release", TARGET_HW, GLOBAL_HW, crops, batched_trace, seed,
            phase3.block_dct_restrict, phase3.block_dct_prolong, "block_dct_24x48",
        )
        print(f"Running {scene_name} batched...", flush=True)
        batched_output, batched_perf = run_sampler(
            model, noise, positive, negative, sigmas, batched_sampler, seed, True
        )

        with torch.inference_mode():
            sequential_pixels = vae.decode(sequential_output).cpu()
            batched_pixels = vae.decode(batched_output).cpu()
        seq_path = OUTPUT / f"{scene_name}_SEQUENTIAL.png"
        bat_path = OUTPUT / f"{scene_name}_BATCHED_B2.png"
        phase2.save_pixels(sequential_pixels, seq_path)
        phase2.save_pixels(batched_pixels, bat_path)
        intervals = compare_traces(sequential_trace, batched_trace)
        report["scenes"][scene_name] = {
            "prompt": prompt,
            "seed": seed,
            "sequential_performance": sequential_perf,
            "batched_performance": batched_perf,
            "speedup": sequential_perf["sampling_wall_seconds"] / batched_perf["sampling_wall_seconds"],
            "per_interval": intervals,
            "final_H_difference": phase3.tensor_difference(batched_output, sequential_output),
            "outputs": {"sequential": str(seq_path), "batched": str(bat_path)},
            "actual_local_calls_per_interval": [
                item["actual_local_model_calls"] for item in batched_trace.evaluations
            ],
            "actual_total_forwards_per_interval": [
                item["actual_model_forwards"] for item in batched_trace.evaluations
            ],
            "coordinate_offsets_per_interval": [
                [batch["offsets_yx"] for batch in item["local_batches"]]
                for item in batched_trace.evaluations
            ],
        }
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del clip
    print(json.dumps({
        name: {
            "speedup": item["speedup"],
            "final_difference": item["final_H_difference"],
            "sequential_peak_allocated": item["sequential_performance"]["peak_allocated_gib"],
            "batched_peak_allocated": item["batched_performance"]["peak_allocated_gib"],
        }
        for name, item in report["scenes"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
