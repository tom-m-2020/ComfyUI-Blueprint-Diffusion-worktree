"""Single-evaluation all-crop compact-global-context assembly probe.

Reuses the Phase 2 geometry/weights and Phase 2c attention intervention. This
is experiment-only and performs no sampler update or cross-step caching.
"""

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


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate2_all_crop_results",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def assemble(
    predictions: list[torch.Tensor], crops: list[phase2.Crop], target_hw: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    output = torch.zeros(
        (predictions[0].shape[0], predictions[0].shape[1], *target_hw),
        dtype=predictions[0].dtype,
    )
    coverage = torch.zeros((predictions[0].shape[0], 1, *target_hw), dtype=torch.float32)
    for prediction, crop in zip(predictions, crops):
        weight = phase2.crop_weight(crop, crops, output.device)[None, None]
        output[:, :, crop.y:crop.y2, crop.x:crop.x2] += prediction * weight
        coverage[:, :, crop.y:crop.y2, crop.x:crop.x2] += weight
    if float(coverage.min()) <= 0:
        raise RuntimeError(f"Incomplete assembly coverage: {phase2.stats(coverage)}")
    return output / coverage, coverage


def overlap_disagreement(
    predictions: list[torch.Tensor], crops: list[phase2.Crop]
) -> dict[str, Any]:
    records = []
    squared_sum = 0.0
    absolute_sum = 0.0
    element_count = 0
    for left_index, left_crop in enumerate(crops):
        for right_index in range(left_index + 1, len(crops)):
            right_crop = crops[right_index]
            y1 = max(left_crop.y, right_crop.y)
            y2 = min(left_crop.y2, right_crop.y2)
            x1 = max(left_crop.x, right_crop.x)
            x2 = min(left_crop.x2, right_crop.x2)
            if y1 >= y2 or x1 >= x2:
                continue
            left = predictions[left_index][
                :, :, y1 - left_crop.y:y2 - left_crop.y, x1 - left_crop.x:x2 - left_crop.x
            ].float()
            right = predictions[right_index][
                :, :, y1 - right_crop.y:y2 - right_crop.y, x1 - right_crop.x:x2 - right_crop.x
            ].float()
            difference = left - right
            records.append(
                {
                    "crop_pair": [left_crop.index, right_crop.index],
                    "absolute_overlap": {"y": y1, "x": x1, "height": y2 - y1, "width": x2 - x1},
                    "rms": float(difference.square().mean().sqrt()),
                    "mean_abs": float(difference.abs().mean()),
                    "max_abs": float(difference.abs().max()),
                }
            )
            squared_sum += float(difference.square().sum())
            absolute_sum += float(difference.abs().sum())
            element_count += difference.numel()
    return {
        "pairs": records,
        "aggregate_rms": math.sqrt(squared_sum / element_count),
        "aggregate_mean_abs": absolute_sum / element_count,
        "elements": element_count,
    }


def comparison(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    return {
        "absolute": phase2c.tensor_difference(value, reference),
        "low_frequency": phase2c.low_frequency_difference(value, reference),
        "value_stats": phase2.stats(value),
        "reference_stats": phase2.stats(reference),
        "residual_norm": phase2.norm(value - reference),
        "relative_residual_to_reference_norm": phase2.norm(value - reference)
        / phase2.norm(reference),
    }


class AllCropOneEvaluationSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, global_hw, crops, probe, seed) -> None:
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crops = crops
        self.probe = probe
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.shared_context_identity: dict[str, Any] = {}

    def run_call(self, model, value, sigma, options, base_options, name) -> None:
        merged = phase2.comfy.model_patcher.create_model_options_clone(base_options)
        merged_transformer = merged.setdefault("transformer_options", {})
        merged_transformer.update(options["transformer_options"])
        before_records = len(self.probe.context_records)
        self.outputs[name], self.calls[name] = phase2c.evaluate(
            model, value, sigma, merged, self.seed
        )
        self.calls[name]["context_record_range"] = [before_records, len(self.probe.context_records)]

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
            raise ValueError("This single-evaluation T2I probe does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        self.run_call(
            model,
            noise,
            sigma,
            phase2c.model_options("dense", {}, self.probe, "ordinary"),
            base_options,
            "A_DENSE_REFERENCE",
        )
        for crop in self.crops:
            value = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
            self.run_call(
                model,
                value,
                sigma,
                phase2c.model_options(
                    f"tiled_only_crop_{crop.index}", phase2.rope_for_crop(crop), self.probe, "ordinary"
                ),
                base_options,
                f"B_TILED_ONLY_CROP_{crop.index}",
            )

        global_input = F.interpolate(noise, size=self.global_hw, mode="bilinear", align_corners=False)
        self.run_call(
            model,
            global_input,
            sigma,
            phase2c.model_options(
                "compact_global",
                phase2.rope_for_global(*self.target_hw, *self.global_hw),
                self.probe,
                "capture",
            ),
            base_options,
            "COMPACT_GLOBAL_CAPTURE",
        )
        captured_ids = {
            f"{key[0]}_{key[1]}": [id(entry["k"]), id(entry["v"])]
            for key, entry in self.probe.global_kv.items()
        }
        capture_count = len(self.probe.capture_records)
        per_crop_identity = []
        for crop in self.crops:
            value = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
            self.run_call(
                model,
                value,
                sigma,
                phase2c.model_options(
                    f"global_context_crop_{crop.index}", phase2.rope_for_crop(crop), self.probe, "context"
                ),
                base_options,
                f"C_GLOBAL_CONTEXT_CROP_{crop.index}",
            )
            current_ids = {
                f"{key[0]}_{key[1]}": [id(entry["k"]), id(entry["v"])]
                for key, entry in self.probe.global_kv.items()
            }
            per_crop_identity.append(
                {
                    "crop_index": crop.index,
                    "same_objects": current_ids == captured_ids,
                    "capture_record_count": len(self.probe.capture_records),
                }
            )
            if current_ids != captured_ids or len(self.probe.capture_records) != capture_count:
                raise AssertionError(f"Compact context changed or was recaptured for crop {crop.index}.")
        self.shared_context_identity = {
            "captured_layers": capture_count,
            "one_capture_only": capture_count == 25,
            "per_crop": per_crop_identity,
        }
        self.probe.assert_complete()
        return noise


def dry_run() -> None:
    crops = phase2.crops_for_canvas(32, 64, 32, 32, 8)
    predictions = [torch.full((1, 2, 32, 32), float(crop.index)) for crop in crops]
    assembled, coverage = assemble(predictions, crops, (32, 64))
    disagreement = overlap_disagreement(predictions, crops)
    if len(crops) != 3 or float(coverage.min()) != 1.0 or float(coverage.max()) != 1.0:
        raise AssertionError((crops, phase2.stats(coverage)))
    print(
        json.dumps(
            {
                "crops": [crop.__dict__ for crop in crops],
                "assembled": phase2.stats(assembled),
                "coverage": phase2.stats(coverage),
                "overlap": disagreement,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    global_width, global_height = 512, 256
    target_hw = (height // 16, width // 16)
    global_hw = (global_height // 16, global_width // 16)
    crop_hw = (512 // 16, 512 // 16)
    overlap_pixels = 128
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap_pixels // 16)

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
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

    latent = torch.randn((1, 128, *target_hw), generator=torch.Generator().manual_seed(seed))
    sigma = phase2.get_schedule(4, math.prod(target_hw)).float()[0]
    probe = phase2c.OneEvaluationContextProbe()
    sampler = AllCropOneEvaluationSampler(target_hw, global_hw, crops, probe, seed)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model,
            latent.clone(),
            1.0,
            sampler,
            torch.stack((sigma, torch.zeros_like(sigma))),
            positive,
            negative,
            torch.zeros_like(latent),
            disable_pbar=True,
            seed=seed,
        )

    dense = sampler.outputs["A_DENSE_REFERENCE"]
    tiled_crops = [sampler.outputs[f"B_TILED_ONLY_CROP_{crop.index}"] for crop in crops]
    context_crops = [sampler.outputs[f"C_GLOBAL_CONTEXT_CROP_{crop.index}"] for crop in crops]
    tiled, tiled_coverage = assemble(tiled_crops, crops, target_hw)
    context, context_coverage = assemble(context_crops, crops, target_hw)

    full_outputs = {
        "A_DENSE_REFERENCE": dense,
        "B_TILED_ONLY_ASSEMBLED": tiled,
        "C_GLOBAL_CONTEXT_TILED_ASSEMBLED": context,
    }
    for name, prediction in full_outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(prediction).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")
    for crop, tiled_prediction, context_prediction in zip(crops, tiled_crops, context_crops):
        dense_crop = dense[:, :, crop.y:crop.y2, crop.x:crop.x2]
        for name, prediction in {
            f"CROP_{crop.index}_DENSE_REFERENCE": dense_crop,
            f"CROP_{crop.index}_TILED_ONLY": tiled_prediction,
            f"CROP_{crop.index}_GLOBAL_CONTEXT": context_prediction,
        }.items():
            with torch.inference_mode():
                pixels = vae.decode(prediction).cpu()
            phase2.save_pixels(pixels, args.output_dir / f"{name}.png")

    phase2.save_heatmap(
        (tiled - dense).square().mean(dim=1, keepdim=True).sqrt(),
        args.output_dir / "B_TILED_ONLY_ERROR_MAGNITUDE.png",
        (height, width),
    )
    phase2.save_heatmap(
        (context - dense).square().mean(dim=1, keepdim=True).sqrt(),
        args.output_dir / "C_GLOBAL_CONTEXT_ERROR_MAGNITUDE.png",
        (height, width),
    )

    per_crop = []
    for crop, tiled_prediction, context_prediction in zip(crops, tiled_crops, context_crops):
        dense_crop = dense[:, :, crop.y:crop.y2, crop.x:crop.x2]
        per_crop.append(
            {
                "crop": crop.__dict__,
                "tiled_only_vs_dense": comparison(tiled_prediction, dense_crop),
                "global_context_vs_dense": comparison(context_prediction, dense_crop),
                "context_vs_local_only": comparison(context_prediction, tiled_prediction),
                "prediction_rms": {
                    "dense_crop": phase2.stats(dense_crop)["rms"],
                    "tiled_only": phase2.stats(tiled_prediction)["rms"],
                    "global_context": phase2.stats(context_prediction)["rms"],
                },
            }
        )

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": phase2.PROMPT,
            "seed": seed,
            "sigma": float(sigma),
            "evaluation": "single first/high-noise evaluation; no sampler update",
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "global_image_hw": [global_height, global_width],
            "global_latent_hw": list(global_hw),
            "crop_image_hw": [512, 512],
            "crop_latent_hw": list(crop_hw),
            "overlap_pixels": overlap_pixels,
            "crops": [crop.__dict__ for crop in crops],
            "global_rope_options": phase2.rope_for_global(*target_hw, *global_hw),
            "crop_rope_options": [phase2.rope_for_crop(crop) for crop in crops],
            "modified_blocks_per_context_crop": "all 5 double and all 20 single blocks",
            "output_space_global_fusion": False,
        },
        "shared_global_context": sampler.shared_context_identity,
        "positions": probe.position_records,
        "global_capture_blocks": probe.capture_records,
        "context_blocks": probe.context_records,
        "calls": sampler.calls,
        "per_crop": per_crop,
        "assembled": {
            "tiled_only_vs_dense": comparison(tiled, dense),
            "global_context_vs_dense": comparison(context, dense),
            "context_vs_tiled_only": comparison(context, tiled),
            "tiled_only_overlap_disagreement": overlap_disagreement(tiled_crops, crops),
            "global_context_overlap_disagreement": overlap_disagreement(context_crops, crops),
            "tiled_only_coverage": phase2.stats(tiled_coverage),
            "global_context_coverage": phase2.stats(context_coverage),
        },
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in full_outputs},
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "per_crop_rms": [
                    {
                        "crop": item["crop"]["index"],
                        "tiled": item["tiled_only_vs_dense"]["absolute"]["rms"],
                        "context": item["global_context_vs_dense"]["absolute"]["rms"],
                    }
                    for item in per_crop
                ],
                "assembled": report["assembled"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
