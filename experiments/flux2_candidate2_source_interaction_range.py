"""Zero-update source interaction range discriminator for Candidate 2."""

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
import flux2_candidate2_dense_source_interaction_ablation as phase2j


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_dense_source_interaction_ablation_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_source_interaction_range_results"
VARIANTS = (
    "A_ORDINARY_DENSE_SOURCE",
    "B_WINDOWED_SOURCE",
    "C_NO_IMAGE_IMAGE_SOURCE",
)
SOURCE_HW = (32, 64)
WINDOW_HW = (16, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def window_ids(device: torch.device) -> torch.Tensor:
    height, width = SOURCE_HW
    window_height, window_width = WINDOW_HW
    y = torch.arange(height, device=device)[:, None].expand(height, width)
    x = torch.arange(width, device=device)[None, :].expand(height, width)
    windows_x = width // window_width
    return ((y // window_height) * windows_x + (x // window_width)).reshape(-1)


class WindowedSourceProbe(phase2c.OneEvaluationContextProbe):
    def __init__(self) -> None:
        super().__init__()
        self.restriction_records: list[dict[str, Any]] = []

    def restrict_to_windows(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected no pre-existing FLUX source attention mask.")
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        image_tokens = sequence_end - text_tokens
        if image_tokens != math.prod(SOURCE_HW) or q.shape[2] != sequence_end or k.shape[2] != sequence_end:
            raise AssertionError((key, q.shape, k.shape, extra_options["img_slice"]))

        ids = window_ids(q.device)
        image_mask = ids[:, None] == ids[None, :]
        mask = torch.zeros((sequence_end, sequence_end), dtype=torch.bool, device=q.device)
        mask[:text_tokens, :] = True
        mask[text_tokens:, :text_tokens] = True
        mask[text_tokens:, text_tokens:] = image_mask
        allowed_image_edges = int(image_mask.sum())
        dense_image_edges = image_tokens * image_tokens
        self.restriction_records.append({
            "block_type": key[0],
            "block_index": key[1],
            "source_query_tokens": sequence_end,
            "source_key_tokens": sequence_end,
            "source_q_by_k": [sequence_end, sequence_end],
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "source_grid_hw": list(SOURCE_HW),
            "window_hw": list(WINDOW_HW),
            "number_of_windows": int(ids.max()) + 1,
            "tokens_per_window": WINDOW_HW[0] * WINDOW_HW[1],
            "allowed_image_image_edges": allowed_image_edges,
            "dense_image_image_edges": dense_image_edges,
            "fraction_dense_image_connectivity": allowed_image_edges / dense_image_edges,
            "blocked_image_image_edges": dense_image_edges - allowed_image_edges,
            "image_image_communication": "same fixed nonoverlapping spatial window only",
            "text_query_key_policy": "unchanged; text queries attend text and all image keys",
            "image_to_text_policy": "unchanged; image queries attend all text keys",
            "mask_dtype": str(mask.dtype),
            "mask_shape": list(mask.shape),
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": mask}

    def assert_restriction_complete(self) -> None:
        if len(self.restriction_records) != 25:
            raise AssertionError(len(self.restriction_records))
        restricted = {(item["block_type"], item["block_index"]) for item in self.restriction_records}
        captured = {(item["block_type"], item["block_index"]) for item in self.capture_records}
        if restricted != captured:
            raise AssertionError(restricted ^ captured)


def windowed_capture_options(role: str, probe: WindowedSourceProbe) -> dict[str, Any]:
    options = phase2c.model_options(role, {}, probe, "capture")
    phase2c.append_patch(options, "attn1_patch", probe.restrict_to_windows)
    return options


class InteractionRangeSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, target_hw, crop, seed) -> None:
        self.target_hw = target_hw
        self.crop = crop
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.source_outputs: dict[str, torch.Tensor] = {}
        self.calls: dict[str, dict[str, Any]] = {}
        self.layouts: dict[str, dict[str, Any]] = {}
        self.positions: dict[str, Any] = {}
        self.probes: dict[str, phase2c.OneEvaluationContextProbe] = {}

    def run_call(self, model, value, sigma, options, base_options, name, role):
        prediction, call = phase2e.model_call(
            model, value, sigma, phase2e.merge_options(base_options, options),
            self.seed, role, self.crop if role == "local_with_external_kv" else None,
        )
        self.calls[name] = call
        return prediction

    def run_variant(self, model, noise, local_input, sigma, base_options, name, mode):
        if mode == "ordinary":
            probe = phase2c.OneEvaluationContextProbe()
            capture_options = phase2c.model_options(f"{name}_source", {}, probe, "capture")
            source_structure = "ordinary dense all-token attention"
            ablated_blocks = "none"
        elif mode == "windowed":
            probe = WindowedSourceProbe()
            capture_options = windowed_capture_options(f"{name}_source", probe)
            source_structure = "image queries attend same 16x16 window and all text; text queries attend all tokens"
            ablated_blocks = "all 5 double and all 20 single blocks"
        elif mode == "none":
            probe = phase2j.RestrictedSourceProbe()
            capture_options = phase2j.restricted_capture_options(f"{name}_source", probe)
            source_structure = "image queries attend text only; image-to-image blocked; text queries attend all tokens"
            ablated_blocks = "all 5 double and all 20 single blocks"
        else:
            raise ValueError(mode)
        self.probes[name] = probe
        source = self.run_call(
            model, noise, sigma, capture_options, base_options,
            f"{name}_GLOBAL_CAPTURE", "global_capture",
        )
        output = self.run_call(
            model, local_input, sigma,
            phase2c.model_options(f"{name}_local", phase2.rope_for_crop(self.crop), probe, "context"),
            base_options, f"{name}_LOCAL_CONTEXT", "local_with_external_kv",
        )
        probe.assert_complete()
        if mode != "ordinary":
            probe.assert_restriction_complete()
        first = probe.context_records[0]
        self.layouts[name] = {
            "source_generated_tokens": first["global_generated_kv"],
            "source_sequence_tokens": 2560,
            "source_q_by_k": [2560, 2560],
            "local_external_q_by_k": first["q_by_k"],
            "source_attention_structure": source_structure,
            "ablated_blocks": ablated_blocks,
            "source_restriction_records": getattr(probe, "restriction_records", []),
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
        for name, mode in zip(VARIANTS, ("ordinary", "windowed", "none")):
            self.outputs[name] = self.run_variant(
                model, noise, local_input, sigma, base_options, name, mode
            )
        invariant_keys = ("source_generated_tokens", "source_sequence_tokens", "source_q_by_k", "local_external_q_by_k")
        ordinary = self.layouts[VARIANTS[0]]
        for name in VARIANTS[1:]:
            if any(self.layouts[name][key] != ordinary[key] for key in invariant_keys):
                raise AssertionError(f"Token geometry changed for {name}.")
        return noise


def kv_differences(reference_probe, candidate_probe) -> list[dict[str, Any]]:
    differences = []
    for key in sorted(reference_probe.global_kv, key=lambda item: (item[0], item[1])):
        reference = reference_probe.global_kv[key]
        candidate = candidate_probe.global_kv[key]
        differences.append({
            "block_type": key[0],
            "block_index": key[1],
            "positioned_k_difference": phase2c.tensor_difference(reference["k"], candidate["k"]),
            "v_difference": phase2c.tensor_difference(reference["v"], candidate["v"]),
        })
    return differences


def dry_run() -> None:
    ids = window_ids(torch.device("cpu"))
    same = ids[:, None] == ids[None, :]
    allowed = int(same.sum())
    dense = math.prod(SOURCE_HW) ** 2
    if int(ids.max()) + 1 != 8 or allowed != 8 * 256 * 256:
        raise AssertionError((ids, allowed))
    print(json.dumps({
        "source_tokens": 2048,
        "source_grid_hw": list(SOURCE_HW),
        "window_hw": list(WINDOW_HW),
        "number_of_windows": 8,
        "tokens_per_window": 256,
        "allowed_image_image_edges": allowed,
        "dense_image_image_edges": dense,
        "fraction_dense_connectivity": allowed / dense,
        "source_q_by_k": [2560, 2560],
        "local_q_by_k": [1536, 3584],
        "sampler_updates": 0,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    if not PREVIOUS_REPORT.is_file():
        raise FileNotFoundError(PREVIOUS_REPORT)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    target_hw = SOURCE_HW
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

    sampler = InteractionRangeSampler(target_hw, crop, seed)
    with torch.inference_mode():
        unchanged = phase2.comfy.sample.sample_custom(
            model, accepted.clone(), 1.0, sampler,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive, negative, torch.zeros_like(accepted), disable_pbar=True, seed=seed,
        ).detach().float().cpu()
    no_update_difference = phase2c.tensor_difference(unchanged, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = sampler.source_outputs["A_ORDINARY_DENSE_SOURCE"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in sampler.outputs.items():
        comparisons[name] = {
            "absolute": phase2c.tensor_difference(output, dense_crop),
            "low_frequency": phase2c.low_frequency_difference(output, dense_crop),
            "prediction": phase2.stats(output),
            "difference_from_A_ordinary_local": phase2c.tensor_difference(output, sampler.outputs[VARIANTS[0]]),
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
        phase2.save_pixels(vae.decode(dense_crop).cpu(), args.output_dir / "D_DENSE_REFERENCE_CROP.png")
        phase2.save_pixels(vae.decode(dense_full).cpu(), args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")
        for name in VARIANTS[1:]:
            phase2.save_pixels(vae.decode(sampler.source_outputs[name]).cpu(), args.output_dir / f"{name}_FULL_SOURCE.png")

    ordinary_probe = sampler.probes[VARIANTS[0]]
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "source_grid_hw": list(SOURCE_HW),
            "window_hw": list(WINDOW_HW), "number_of_windows": 8,
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "modified_source_blocks": "all 5 double and all 20 single blocks for B and C",
            "local_integration_changed": False,
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "comparisons_vs_dense_crop": comparisons,
        "source_kv_differences_from_ordinary": {
            name: kv_differences(ordinary_probe, sampler.probes[name])
            for name in VARIANTS[1:]
        },
        "source_prediction_comparisons_from_ordinary": {
            name: phase2c.tensor_difference(sampler.source_outputs[name], dense_full)
            for name in VARIANTS[1:]
        },
        "decoded_semantic_observations": {
            "A_ORDINARY_DENSE_SOURCE": {
                "duplicate_lighthouse": "suppressed; only the dense-reference distant silhouette remains",
                "stone_structure": "reduced center-right tower remains",
                "train": "closest local train continuity and alignment to ordinary dense crop",
                "bridge": "closest deck and cable geometry to ordinary dense crop",
                "horizon_water": "closest agreement; no distinct extra white lighthouse",
            },
            "B_WINDOWED_SOURCE": {
                "duplicate_lighthouse": "small white center-right lighthouse returns beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation regress toward the no-image-image control",
                "bridge": "deck remains continuous, but train and cable geometry remain far from ordinary dense",
                "horizon_water": "false lighthouse and tower pair returns on the horizon",
            },
            "C_NO_IMAGE_IMAGE_SOURCE": {
                "duplicate_lighthouse": "small white center-right lighthouse returns beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation diverge from ordinary dense",
                "bridge": "deck remains continuous, but train and cable geometry regress",
                "horizon_water": "false lighthouse and tower pair returns on the horizon",
            },
            "windowed_vs_no_image_image": "windowed is only marginally better numerically and does not change the known semantic failure classification",
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
        "variants": {
            name: {
                "source_tokens": sampler.layouts[name]["source_generated_tokens"],
                "source_q_by_k": sampler.layouts[name]["source_q_by_k"],
                "local_q_by_k": sampler.layouts[name]["local_external_q_by_k"],
                "rms": comparisons[name]["absolute"]["rms"],
                "low_frequency_rms": comparisons[name]["low_frequency"]["rms"],
                "prediction_rms": comparisons[name]["prediction"]["rms"],
            }
            for name in VARIANTS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
