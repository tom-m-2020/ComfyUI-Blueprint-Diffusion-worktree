"""Zero-update equal-budget multiscale global-context probe for Candidate 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as phase2c
import flux2_candidate2_four_step_trajectory as phase2e
import flux2_candidate2_dense_vs_compact_kv as phase2f


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_intermediate_density_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_equal_budget_multiscale_results"
VARIANTS = ("A_UNIFORM_1152", "B_MULTISCALE_1152", "C_DENSE_2048")
COARSE_HW = (16, 32)
UNIFORM_HW = (24, 48)
REGION = phase2.Crop(index=0, y=8, x=24, height=20, width=32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def combine_global_kv(
    coarse: phase2c.OneEvaluationContextProbe,
    regional: phase2c.OneEvaluationContextProbe,
    combined: phase2c.OneEvaluationContextProbe,
) -> None:
    if coarse.global_kv.keys() != regional.global_kv.keys():
        raise AssertionError("Coarse/regional block coverage differs.")
    for key in coarse.global_kv:
        coarse_entry = coarse.global_kv[key]
        regional_entry = regional.global_kv[key]
        combined.global_kv[key] = {
            "k": torch.cat((coarse_entry["k"], regional_entry["k"]), dim=2),
            "v": torch.cat((coarse_entry["v"], regional_entry["v"]), dim=2),
            "tokens": int(coarse_entry["tokens"]) + int(regional_entry["tokens"]),
        }
        combined.capture_records.append(
            {
                "block_type": key[0],
                "block_index": key[1],
                "text_tokens": 512,
                "global_generated_tokens": int(combined.global_kv[key]["tokens"]),
                "coarse_generated_tokens": int(coarse_entry["tokens"]),
                "regional_generated_tokens": int(regional_entry["tokens"]),
                "ordering": "coarse-global-generated, regional-global-generated",
            }
        )


class MultiscaleDiagnosticSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, crop, seed) -> None:
        self.target_hw = target_hw
        self.crop = crop
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.source_outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.layouts: dict[str, dict[str, Any]] = {}
        self.positions: dict[str, Any] = {}
        self.capture_records: dict[str, Any] = {}

    def run_call(self, model, value, sigma, options, base_options, name, role):
        prediction, call = phase2e.model_call(
            model,
            value,
            sigma,
            phase2e.merge_options(base_options, options),
            self.seed,
            role,
            self.crop if role == "local_with_external_kv" else None,
        )
        self.calls[name] = call
        return prediction

    def ordinary_variant(self, model, noise, local_input, sigma, base_options, name, global_hw):
        probe = phase2c.OneEvaluationContextProbe()
        if global_hw == self.target_hw:
            global_input = noise
            rope = {}
        else:
            global_input = F.interpolate(noise, size=global_hw, mode="bilinear", align_corners=False)
            rope = phase2.rope_for_global(*self.target_hw, *global_hw)
        source = self.run_call(
            model, global_input, sigma,
            phase2c.model_options(f"{name}_source", rope, probe, "capture"),
            base_options, f"{name}_GLOBAL_CAPTURE", "global_capture",
        )
        output = self.run_call(
            model, local_input, sigma,
            phase2c.model_options(f"{name}_local", phase2.rope_for_crop(self.crop), probe, "context"),
            base_options, f"{name}_LOCAL_CONTEXT", "local_with_external_kv",
        )
        probe.assert_complete()
        self.positions[name] = probe.position_records
        self.capture_records[name] = probe.capture_records
        self.source_outputs[name] = source.detach().float().cpu()
        self.record_layout(name, probe.context_records, math.prod(global_hw), math.prod(global_hw), 0)
        return output

    def multiscale_variant(self, model, noise, local_input, sigma, base_options):
        name = "B_MULTISCALE_1152"
        coarse_probe = phase2c.OneEvaluationContextProbe()
        regional_probe = phase2c.OneEvaluationContextProbe()
        combined_probe = phase2c.OneEvaluationContextProbe()
        coarse_input = F.interpolate(noise, size=COARSE_HW, mode="bilinear", align_corners=False)
        regional_input = noise[:, :, REGION.y:REGION.y2, REGION.x:REGION.x2]
        coarse_source = self.run_call(
            model, coarse_input, sigma,
            phase2c.model_options(
                f"{name}_coarse_source",
                phase2.rope_for_global(*self.target_hw, *COARSE_HW),
                coarse_probe, "capture",
            ),
            base_options, f"{name}_COARSE_CAPTURE", "global_capture",
        )
        regional_source = self.run_call(
            model, regional_input, sigma,
            phase2c.model_options(
                f"{name}_regional_source", phase2.rope_for_crop(REGION), regional_probe, "capture"
            ),
            base_options, f"{name}_REGIONAL_CAPTURE", "global_capture",
        )
        combine_global_kv(coarse_probe, regional_probe, combined_probe)
        output = self.run_call(
            model, local_input, sigma,
            phase2c.model_options(
                f"{name}_local", phase2.rope_for_crop(self.crop), combined_probe, "context"
            ),
            base_options, f"{name}_LOCAL_CONTEXT", "local_with_external_kv",
        )
        combined_probe.assert_complete()
        self.positions[name] = {
            "coarse": coarse_probe.position_records,
            "regional": regional_probe.position_records,
            "local": combined_probe.position_records,
        }
        self.capture_records[name] = {
            "coarse": coarse_probe.capture_records,
            "regional": regional_probe.capture_records,
            "combined": combined_probe.capture_records,
        }
        self.source_outputs[f"{name}_COARSE"] = coarse_source.detach().float().cpu()
        self.source_outputs[f"{name}_REGIONAL"] = regional_source.detach().float().cpu()
        self.record_layout(name, combined_probe.context_records, 1152, 512, 640)
        return output

    def record_layout(self, name, records, total_tokens, coarse_tokens, regional_tokens):
        if len(records) != 25:
            raise AssertionError((name, len(records)))
        first = records[0]
        q, k = first["q_by_k"]
        if first["global_generated_kv"] != total_tokens:
            raise AssertionError((name, first["global_generated_kv"], total_tokens))
        self.layouts[name] = {
            "total_external_global_tokens": total_tokens,
            "coarse_tokens": coarse_tokens,
            "regional_tokens": regional_tokens,
            "query_tokens_total": q,
            "kv_tokens_total": k,
            "q_by_k": [q, k],
            "local_query_attention_qk_products_per_block": q * k,
            "context_records": records,
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("This zero-update diagnostic does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
        self.outputs["A_UNIFORM_1152"] = self.ordinary_variant(
            model, noise, local_input, sigma, base_options, "A_UNIFORM_1152", UNIFORM_HW
        ).detach().float().cpu()
        self.outputs["B_MULTISCALE_1152"] = self.multiscale_variant(
            model, noise, local_input, sigma, base_options
        ).detach().float().cpu()
        self.outputs["C_DENSE_2048"] = self.ordinary_variant(
            model, noise, local_input, sigma, base_options, "C_DENSE_2048", self.target_hw
        ).detach().float().cpu()
        control_work = self.layouts["A_UNIFORM_1152"]["local_query_attention_qk_products_per_block"]
        dense_work = self.layouts["C_DENSE_2048"]["local_query_attention_qk_products_per_block"]
        for layout in self.layouts.values():
            work = layout["local_query_attention_qk_products_per_block"]
            layout["relative_to_uniform_1152"] = work / control_work
            layout["relative_to_dense_2048"] = work / dense_work
        return noise


def save_position_visualization(path: Path, target_hw: tuple[int, int]) -> None:
    scale = 14
    height, width = target_hw
    image = Image.new("RGB", (width * scale, height * scale), (245, 245, 245))
    draw = ImageDraw.Draw(image, "RGBA")
    for gy in range(COARSE_HW[0]):
        y = round(gy * (height - 1) / (COARSE_HW[0] - 1))
        for gx in range(COARSE_HW[1]):
            x = round(gx * (width - 1) / (COARSE_HW[1] - 1))
            draw.ellipse((x * scale - 2, y * scale - 2, x * scale + 2, y * scale + 2), fill=(30, 110, 230, 210))
    draw.rectangle(
        (REGION.x * scale, REGION.y * scale, REGION.x2 * scale - 1, REGION.y2 * scale - 1),
        fill=(255, 120, 40, 45), outline=(220, 65, 20, 255), width=2,
    )
    for y in range(REGION.y, REGION.y2):
        for x in range(REGION.x, REGION.x2):
            draw.rectangle((x * scale - 1, y * scale - 1, x * scale + 1, y * scale + 1), fill=(235, 70, 25, 210))
    draw.text((8, 8), "blue: 512 coarse whole-canvas | orange: 640 native-density regional", fill=(0, 0, 0, 255))
    image.save(path)


def dry_run() -> None:
    q = 1536
    expected = {
        "A_UNIFORM_1152": (1152, q, 512 + 1024 + 1152),
        "B_MULTISCALE_1152": (512 + REGION.height * REGION.width, q, 512 + 1024 + 1152),
        "C_DENSE_2048": (2048, q, 512 + 1024 + 2048),
    }
    if expected["B_MULTISCALE_1152"][0] != 1152:
        raise AssertionError(expected)
    print(json.dumps({name: {"global_tokens": v[0], "q_by_k": [v[1], v[2]], "qk": v[1] * v[2]} for name, v in expected.items()}, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    if not PREVIOUS_REPORT.is_file():
        raise FileNotFoundError(PREVIOUS_REPORT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_hw = (32, 64)
    crop_hw = (32, 32)
    seed = 20260829
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, 8)
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
    reproduction_sampler = phase2e.FourStepSampler("context", target_hw, COARSE_HW, crops, trace, seed)
    accepted = phase2e.run_trajectory(model, noise, positive, negative, sigmas[: diagnostic_step + 1], reproduction_sampler, seed)
    if trace.updates != diagnostic_step:
        raise AssertionError(trace.updates)

    previous = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    expected_stats = previous["state_reproduction"]["reproduced_stats"]
    reproduced_stats = trace.evaluations[-1]["accepted_latent_after"]
    stat_differences = {key: abs(float(reproduced_stats[key]) - float(expected_stats[key])) for key in ("norm", "rms", "mean", "std", "min", "max")}
    if max(stat_differences.values()) > 1e-6:
        raise AssertionError(stat_differences)

    sampler = MultiscaleDiagnosticSampler(target_hw, crop, seed)
    with torch.inference_mode():
        returned = phase2.comfy.sample.sample_custom(
            model,
            accepted.clone(),
            1.0,
            sampler,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive,
            negative,
            torch.zeros_like(accepted),
            disable_pbar=True,
            seed=seed,
        ).detach().float().cpu()
    no_update_difference = phase2c.tensor_difference(returned, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = sampler.source_outputs["C_DENSE_2048"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in sampler.outputs.items():
        comparisons[name] = {
            "absolute": phase2c.tensor_difference(output, dense_crop),
            "low_frequency": phase2c.low_frequency_difference(output, dense_crop),
            "prediction": phase2.stats(output),
        }
        with torch.inference_mode():
            pixels = vae.decode(output).cpu()
        phase2.save_pixels(pixels, args.output_dir / f"{name}.png")
        canvas = dense_full.clone()
        canvas[:, :, crop.y:crop.y2, crop.x:crop.x2] = output
        with torch.inference_mode():
            context_pixels = vae.decode(canvas).cpu()
        phase2.save_pixels(context_pixels, args.output_dir / f"{name}_IN_DENSE_CONTEXT.png")
        phase2.save_heatmap(
            (output - dense_crop).square().mean(dim=1, keepdim=True).sqrt(),
            args.output_dir / f"{name}_ERROR_MAGNITUDE.png", (512, 512),
        )
    with torch.inference_mode():
        phase2.save_pixels(vae.decode(dense_crop).cpu(), args.output_dir / "D_DENSE_REFERENCE_CROP.png")
        phase2.save_pixels(vae.decode(dense_full).cpu(), args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")
    save_position_visualization(args.output_dir / "MULTISCALE_POSITION_MAP.png", target_hw)

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "target_latent_hw": list(target_hw),
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "regional_capture": REGION.__dict__, "regional_rope": phase2.rope_for_crop(REGION),
            "coarse_hw": list(COARSE_HW), "coarse_rope": phase2.rope_for_global(*target_hw, *COARSE_HW),
            "uniform_hw": list(UNIFORM_HW), "uniform_rope": phase2.rope_for_global(*target_hw, *UNIFORM_HW),
            "modified_blocks": "all 5 double and all 20 single blocks",
            "overlap_policy": "coarse and regional K/V concatenated without coordinate deduplication",
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts_and_attention_context_work": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "capture_records": sampler.capture_records,
        "comparisons_vs_dense_crop": comparisons,
        "decoded_semantic_observations": {
            "A_UNIFORM_1152": {
                "duplicate_lighthouse": "present as a small center-right lighthouse",
                "dark_stone_structure": "prominent center-right stone structure remains",
                "train_continuity": "continuous but carriage spacing and extent differ from dense",
                "bridge_alignment": "continuous deck/cables with moderate dense-reference divergence",
                "horizon_water": "coherent horizon; false lighthouse/stone pair remains",
            },
            "B_MULTISCALE_1152": {
                "duplicate_lighthouse": "present and more visually prominent than uniform 1152",
                "dark_stone_structure": "larger, more severe center-right stone/landmass complex",
                "train_continuity": "continuous but compressed/resegmented relative to uniform and dense",
                "bridge_alignment": "deck remains continuous; cable/train geometry diverges further from dense",
                "horizon_water": "false regional landmass/object complex disrupts horizon more strongly",
            },
            "C_DENSE_2048": {
                "duplicate_lighthouse": "removed",
                "dark_stone_structure": "reduced tower-like structure remains; not identical to dense crop",
                "train_continuity": "closest alignment and continuity to dense reference",
                "bridge_alignment": "substantially closest deck/cable geometry to dense reference",
                "horizon_water": "best agreement of the three variants",
            },
            "scope": "manual inspection of decoded one-evaluation denoised estimates; not an automated object metric",
        },
        "dense_reference_crop": phase2.stats(dense_crop),
        "attention_work_scope": "Q*K products for changed local attention context only; excludes capture forwards, projections, MLPs, residuals, and total-model FLOPs",
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in VARIANTS},
        "position_visualization": str(args.output_dir / "MULTISCALE_POSITION_MAP.png"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "state_stat_max_difference": max(stat_differences.values()), "no_update_max_abs": no_update_difference["max_abs"], "variants": {name: {"tokens": sampler.layouts[name]["total_external_global_tokens"], "q_by_k": sampler.layouts[name]["q_by_k"], "relative_work": sampler.layouts[name]["relative_to_uniform_1152"], "rms": comparisons[name]["absolute"]["rms"], "low_frequency_rms": comparisons[name]["low_frequency"]["rms"], "prediction_rms": comparisons[name]["prediction"]["rms"]} for name in VARIANTS}}, indent=2))


if __name__ == "__main__":
    main()
