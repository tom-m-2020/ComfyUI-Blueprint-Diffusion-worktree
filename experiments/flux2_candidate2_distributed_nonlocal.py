"""Zero-update distributed nonlocal evidence discriminator for Candidate 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as phase2c
import flux2_candidate2_four_step_trajectory as phase2e
import flux2_candidate2_dense_vs_compact_kv as phase2f
import flux2_candidate2_equal_budget_multiscale as phase2h


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_equal_budget_multiscale_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_distributed_nonlocal_results"
VARIANTS = (
    "A_UNIFORM_1152",
    "B_FAILURE_LOCAL_1152",
    "C_NONLOCAL_1152",
    "D_DENSE_2048",
)
LEFT_STRIP = phase2.Crop(index=0, y=0, x=0, height=32, width=24)
RIGHT_STRIP = phase2.Crop(index=1, y=0, x=56, height=32, width=8)
LEFT_COUNT = 480
RIGHT_COUNT = 160


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def evenly_spaced_indices(total: int, count: int) -> torch.Tensor:
    if not 0 < count <= total:
        raise ValueError((total, count))
    # One deterministic sample per equal-width rank interval. This is content
    # blind and retains row-major coverage over the complete source strip.
    indices = torch.div(
        torch.arange(count, dtype=torch.int64) * total,
        count,
        rounding_mode="floor",
    )
    if indices.unique().numel() != count:
        raise AssertionError("Balanced selector produced duplicate indices.")
    return indices


def selected_coordinates(strip: phase2.Crop, indices: torch.Tensor) -> list[tuple[float, float]]:
    return [
        (float(strip.y + int(index) // strip.width), float(strip.x + int(index) % strip.width))
        for index in indices.tolist()
    ]


def coordinate_accounting(coarse_positions, added_positions) -> dict[str, Any]:
    combined = [tuple(position) for position in coarse_positions + added_positions]
    unique = set(combined)
    duplicated = sorted({position for position in combined if combined.count(position) > 1})
    return {
        "total_external_tokens": len(combined),
        "unique_full_canvas_rope_positions": len(unique),
        "duplicated_token_positions": len(combined) - len(unique),
        "duplicated_coordinates_yx": [list(position) for position in duplicated],
    }


def select_probe_kv(
    probe: phase2c.OneEvaluationContextProbe,
    indices: torch.Tensor,
) -> dict[tuple[str, int], dict[str, torch.Tensor | int]]:
    selected = {}
    for key, entry in probe.global_kv.items():
        device_indices = indices.to(entry["k"].device)
        selected[key] = {
            "k": entry["k"].index_select(2, device_indices),
            "v": entry["v"].index_select(2, device_indices),
            "tokens": int(indices.numel()),
        }
    return selected


def combine_selected_kv(
    coarse: phase2c.OneEvaluationContextProbe,
    left_selected,
    right_selected,
    combined: phase2c.OneEvaluationContextProbe,
) -> None:
    if not (coarse.global_kv.keys() == left_selected.keys() == right_selected.keys()):
        raise AssertionError("Nonlocal capture block coverage differs.")
    for key, coarse_entry in coarse.global_kv.items():
        entries = (coarse_entry, left_selected[key], right_selected[key])
        combined.global_kv[key] = {
            "k": torch.cat(tuple(entry["k"] for entry in entries), dim=2),
            "v": torch.cat(tuple(entry["v"] for entry in entries), dim=2),
            "tokens": sum(int(entry["tokens"]) for entry in entries),
        }
        combined.capture_records.append(
            {
                "block_type": key[0],
                "block_index": key[1],
                "text_tokens": 512,
                "global_generated_tokens": int(combined.global_kv[key]["tokens"]),
                "coarse_generated_tokens": int(coarse_entry["tokens"]),
                "left_nonlocal_tokens": int(left_selected[key]["tokens"]),
                "right_nonlocal_tokens": int(right_selected[key]["tokens"]),
                "ordering": "coarse-global, selected-left-nonlocal, selected-right-nonlocal",
            }
        )


class DistributedDiagnosticSampler(phase2h.MultiscaleDiagnosticSampler):
    def nonlocal_variant(self, model, noise, local_input, sigma, base_options):
        name = "C_NONLOCAL_1152"
        coarse_probe = phase2c.OneEvaluationContextProbe()
        left_probe = phase2c.OneEvaluationContextProbe()
        right_probe = phase2c.OneEvaluationContextProbe()
        combined_probe = phase2c.OneEvaluationContextProbe()

        coarse_input = torch.nn.functional.interpolate(
            noise, size=phase2h.COARSE_HW, mode="bilinear", align_corners=False
        )
        left_input = noise[:, :, LEFT_STRIP.y:LEFT_STRIP.y2, LEFT_STRIP.x:LEFT_STRIP.x2]
        right_input = noise[:, :, RIGHT_STRIP.y:RIGHT_STRIP.y2, RIGHT_STRIP.x:RIGHT_STRIP.x2]
        self.run_call(
            model, coarse_input, sigma,
            phase2c.model_options(
                f"{name}_coarse_source",
                phase2.rope_for_global(*self.target_hw, *phase2h.COARSE_HW),
                coarse_probe, "capture",
            ),
            base_options, f"{name}_COARSE_CAPTURE", "global_capture",
        )
        self.run_call(
            model, left_input, sigma,
            phase2c.model_options(
                f"{name}_left_source", phase2.rope_for_crop(LEFT_STRIP), left_probe, "capture"
            ),
            base_options, f"{name}_LEFT_CAPTURE", "global_capture",
        )
        self.run_call(
            model, right_input, sigma,
            phase2c.model_options(
                f"{name}_right_source", phase2.rope_for_crop(RIGHT_STRIP), right_probe, "capture"
            ),
            base_options, f"{name}_RIGHT_CAPTURE", "global_capture",
        )
        left_indices = evenly_spaced_indices(LEFT_STRIP.height * LEFT_STRIP.width, LEFT_COUNT)
        right_indices = evenly_spaced_indices(RIGHT_STRIP.height * RIGHT_STRIP.width, RIGHT_COUNT)
        left_selected = select_probe_kv(left_probe, left_indices)
        right_selected = select_probe_kv(right_probe, right_indices)
        combine_selected_kv(coarse_probe, left_selected, right_selected, combined_probe)
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
            "left_source": left_probe.position_records,
            "right_source": right_probe.position_records,
            "left_selected_indices": left_indices.tolist(),
            "right_selected_indices": right_indices.tolist(),
            "left_selected_coordinates_yx": selected_coordinates(LEFT_STRIP, left_indices),
            "right_selected_coordinates_yx": selected_coordinates(RIGHT_STRIP, right_indices),
            "local": combined_probe.position_records,
        }
        self.capture_records[name] = {
            "coarse": coarse_probe.capture_records,
            "left_source": left_probe.capture_records,
            "right_source": right_probe.capture_records,
            "combined": combined_probe.capture_records,
        }
        self.record_layout(name, combined_probe.context_records, 1152, 512, 640)
        return output

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("This zero-update diagnostic does not use a mask.")
        sigma = sigmas[0]
        base_options = extra_args["model_options"]
        crop = self.crop
        local_input = noise[:, :, crop.y:crop.y2, crop.x:crop.x2]
        self.outputs["A_UNIFORM_1152"] = self.ordinary_variant(
            model, noise, local_input, sigma, base_options, "A_UNIFORM_1152", phase2h.UNIFORM_HW
        ).detach().float().cpu()
        self.outputs["B_FAILURE_LOCAL_1152"] = self.multiscale_variant(
            model, noise, local_input, sigma, base_options
        ).detach().float().cpu()
        self.layouts["B_FAILURE_LOCAL_1152"] = self.layouts.pop("B_MULTISCALE_1152")
        self.positions["B_FAILURE_LOCAL_1152"] = self.positions.pop("B_MULTISCALE_1152")
        self.capture_records["B_FAILURE_LOCAL_1152"] = self.capture_records.pop("B_MULTISCALE_1152")
        self.outputs["C_NONLOCAL_1152"] = self.nonlocal_variant(
            model, noise, local_input, sigma, base_options
        ).detach().float().cpu()
        self.outputs["D_DENSE_2048"] = self.ordinary_variant(
            model, noise, local_input, sigma, base_options, "D_DENSE_2048", self.target_hw
        ).detach().float().cpu()
        control_work = self.layouts["A_UNIFORM_1152"]["local_query_attention_qk_products_per_block"]
        dense_work = self.layouts["D_DENSE_2048"]["local_query_attention_qk_products_per_block"]
        for layout in self.layouts.values():
            work = layout["local_query_attention_qk_products_per_block"]
            layout["relative_to_uniform_1152"] = work / control_work
            layout["relative_to_dense_2048"] = work / dense_work
        return noise


def coarse_coordinates() -> list[tuple[float, float]]:
    return [
        (y * 31.0 / 15.0, x * 63.0 / 31.0)
        for y in range(16)
        for x in range(32)
    ]


def uniform_coordinates(height: int, width: int) -> list[tuple[float, float]]:
    return [
        (y * 31.0 / (height - 1), x * 63.0 / (width - 1))
        for y in range(height)
        for x in range(width)
    ]


def position_accounting() -> dict[str, dict[str, int]]:
    coarse = coarse_coordinates()
    failure_local = [
        (float(y), float(x))
        for y in range(phase2h.REGION.y, phase2h.REGION.y2)
        for x in range(phase2h.REGION.x, phase2h.REGION.x2)
    ]
    left = selected_coordinates(
        LEFT_STRIP, evenly_spaced_indices(LEFT_STRIP.height * LEFT_STRIP.width, LEFT_COUNT)
    )
    right = selected_coordinates(
        RIGHT_STRIP, evenly_spaced_indices(RIGHT_STRIP.height * RIGHT_STRIP.width, RIGHT_COUNT)
    )
    return {
        "A_UNIFORM_1152": coordinate_accounting([], uniform_coordinates(24, 48)),
        "B_FAILURE_LOCAL_1152": coordinate_accounting(coarse, failure_local),
        "C_NONLOCAL_1152": coordinate_accounting(coarse, left + right),
        "D_DENSE_2048": coordinate_accounting([], [(float(y), float(x)) for y in range(32) for x in range(64)]),
    }


def save_position_visualization(path: Path) -> None:
    scale = 14
    image = Image.new("RGB", (64 * scale, 32 * scale), (245, 245, 245))
    draw = ImageDraw.Draw(image, "RGBA")
    for y, x in coarse_coordinates():
        draw.ellipse((x * scale - 2, y * scale - 2, x * scale + 2, y * scale + 2), fill=(30, 110, 230, 210))
    selected = (
        selected_coordinates(LEFT_STRIP, evenly_spaced_indices(768, LEFT_COUNT))
        + selected_coordinates(RIGHT_STRIP, evenly_spaced_indices(256, RIGHT_COUNT))
    )
    for y, x in selected:
        draw.rectangle((x * scale - 1, y * scale - 1, x * scale + 1, y * scale + 1), fill=(40, 175, 75, 230))
    crop = phase2h.REGION
    draw.rectangle((crop.x * scale, 0, crop.x2 * scale - 1, 32 * scale - 1), fill=(220, 40, 40, 25), outline=(200, 20, 20, 220), width=2)
    draw.text((8, 8), "blue: coarse | green: selected nonlocal | red: excluded active crop", fill=(0, 0, 0, 255))
    image.save(path)


def dry_run() -> None:
    accounting = position_accounting()
    values = {
        name: {
            "external_tokens": 1152 if name != "D_DENSE_2048" else 2048,
            "q_by_k": [1536, 2688 if name != "D_DENSE_2048" else 3584],
            "positions": accounting[name],
        }
        for name in VARIANTS
    }
    if LEFT_COUNT + RIGHT_COUNT != 640:
        raise AssertionError(values)
    print(json.dumps(values, indent=2))


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

    sampler = DistributedDiagnosticSampler(target_hw, crop, seed)
    with torch.inference_mode():
        unchanged = phase2.comfy.sample.sample_custom(
            model, accepted.clone(), 1.0, sampler,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive, negative, torch.zeros_like(accepted), disable_pbar=True, seed=seed,
        ).detach().float().cpu()
    no_update_difference = phase2c.tensor_difference(unchanged, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = sampler.source_outputs["D_DENSE_2048"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in sampler.outputs.items():
        comparisons[name] = {
            "absolute": phase2c.tensor_difference(output, dense_crop),
            "low_frequency": phase2c.low_frequency_difference(output, dense_crop),
            "prediction": phase2.stats(output),
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
    save_position_visualization(args.output_dir / "NONLOCAL_POSITION_MAP.png")

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "target_latent_hw": list(target_hw),
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "nonlocal_exclusion": crop.__dict__,
            "left_source_strip": LEFT_STRIP.__dict__, "right_source_strip": RIGHT_STRIP.__dict__,
            "selection": "floor(i * source_token_count / selected_count), independently per strip",
            "selection_counts": {"left": LEFT_COUNT, "right": RIGHT_COUNT, "total": LEFT_COUNT + RIGHT_COUNT},
            "modified_blocks": "all 5 double and all 20 single blocks",
            "overlap_policy": "coarse and selected native K/V concatenated without coordinate deduplication",
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "position_accounting": position_accounting(),
        "layouts_and_attention_context_work": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "capture_records": sampler.capture_records,
        "comparisons_vs_dense_crop": comparisons,
        "decoded_semantic_observations": {
            "A_UNIFORM_1152": {
                "duplicate_lighthouse": "present as a small center-right lighthouse",
                "dark_stone_structure": "prominent center-right stone structure remains",
                "train": "continuous, with carriage spacing and extent differing from dense",
                "bridge": "continuous deck/cables with moderate dense-reference divergence",
                "horizon_water": "coherent, but false lighthouse/stone pair remains",
                "new_failure": "none identified beyond the known center-right alternatives",
            },
            "B_FAILURE_LOCAL_1152": {
                "duplicate_lighthouse": "present and amplified relative to uniform",
                "dark_stone_structure": "enlarged into a stronger stone/landmass complex",
                "train": "compressed/resegmented relative to uniform and dense",
                "bridge": "nearby train/cable geometry diverges further from dense",
                "horizon_water": "false object complex disrupts the horizon more strongly",
                "new_failure": "amplified regional landmass attached to the false objects",
            },
            "C_NONLOCAL_1152": {
                "duplicate_lighthouse": "present at approximately uniform-control severity",
                "dark_stone_structure": "prominent center-right stone structure remains",
                "train": "continuous and broadly comparable to uniform, but not closer to dense overall",
                "bridge": "continuous deck/cables; no material semantic improvement over uniform",
                "horizon_water": "coherent, but false lighthouse/stone pair remains",
                "new_failure": "none clearly introduced; failure-local amplification is avoided",
            },
            "D_DENSE_2048": {
                "duplicate_lighthouse": "removed",
                "dark_stone_structure": "reduced tower-like structure remains; not identical to dense crop",
                "train": "closest continuity and alignment to dense reference",
                "bridge": "substantially closest deck/cable geometry to dense reference",
                "horizon_water": "best agreement of the four variants",
                "new_failure": "residual stone tower remains despite dense context",
            },
            "scope": "manual inspection of decoded one-evaluation denoised estimates; not an automated object metric",
        },
        "dense_reference_crop": phase2.stats(dense_crop),
        "attention_work_scope": "Q*K products for changed local attention context only; excludes capture forwards, selection, projections, MLPs, residuals, and total-model FLOPs",
        "outputs": {name: str(args.output_dir / f"{name}.png") for name in VARIANTS},
        "position_visualization": str(args.output_dir / "NONLOCAL_POSITION_MAP.png"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(args.output_dir / "report.json"), "state_stat_max_difference": max(stat_differences.values()), "no_update_max_abs": no_update_difference["max_abs"], "variants": {name: {"tokens": sampler.layouts[name]["total_external_global_tokens"], "q_by_k": sampler.layouts[name]["q_by_k"], "rms": comparisons[name]["absolute"]["rms"], "low_frequency_rms": comparisons[name]["low_frequency"]["rms"], "prediction_rms": comparisons[name]["prediction"]["rms"], "position_accounting": position_accounting()[name]} for name in VARIANTS}}, indent=2))


if __name__ == "__main__":
    main()
