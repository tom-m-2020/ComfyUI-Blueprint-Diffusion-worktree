"""Zero-update cross-window propagation discriminator for Candidate 2."""

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
import flux2_candidate2_source_interaction_range as phase2k


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_source_interaction_range_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_cross_window_propagation_results"
VARIANTS = (
    "A_ORDINARY_DENSE_SOURCE",
    "B_FIXED_WINDOWS",
    "C_ALTERNATING_SHIFTED_WINDOWS",
    "D_NO_IMAGE_IMAGE_SOURCE",
)
SOURCE_HW = (32, 64)
WINDOW_HW = (16, 16)
SHIFT_YX = (8, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def execution_ordinal(block_type: str, block_index: int) -> int:
    if block_type == "double":
        return block_index
    if block_type == "single":
        return 5 + block_index
    raise ValueError(block_type)


def partition_ids(shifted: bool, device: torch.device) -> torch.Tensor:
    height, width = SOURCE_HW
    window_height, window_width = WINDOW_HW
    shift_y, shift_x = SHIFT_YX if shifted else (0, 0)
    y = torch.arange(height, device=device)[:, None].expand(height, width)
    x = torch.arange(width, device=device)[None, :].expand(height, width)
    windows_x = math.ceil((width + shift_x) / window_width)
    return (((y + shift_y) // window_height) * windows_x + ((x + shift_x) // window_width)).reshape(-1)


def partition_groups(shifted: bool) -> list[list[int]]:
    ids = partition_ids(shifted, torch.device("cpu")).tolist()
    groups: dict[int, list[int]] = {}
    for token, group in enumerate(ids):
        groups.setdefault(group, []).append(token)
    return list(groups.values())


def reachable_by_depth(alternating: bool) -> list[dict[str, Any]]:
    token_count = math.prod(SOURCE_HW)
    reachable = [1 << token for token in range(token_count)]
    records = []
    for ordinal in range(25):
        shifted = alternating and ordinal % 2 == 1
        for group in partition_groups(shifted):
            merged = 0
            for token in group:
                merged |= reachable[token]
            for token in group:
                reachable[token] = merged
        counts = [value.bit_count() for value in reachable]
        records.append({
            "execution_ordinal": ordinal,
            "partition": "shifted_clipped" if shifted else "unshifted",
            "minimum_reachable_tokens": min(counts),
            "maximum_reachable_tokens": max(counts),
            "mean_reachable_tokens": sum(counts) / token_count,
            "minimum_reachable_fraction": min(counts) / token_count,
            "maximum_reachable_fraction": max(counts) / token_count,
            "tokens_reaching_full_canvas": sum(count == token_count for count in counts),
        })
    return records


class AlternatingShiftedProbe(phase2c.OneEvaluationContextProbe):
    def __init__(self) -> None:
        super().__init__()
        self.restriction_records: list[dict[str, Any]] = []

    def restrict_alternating_windows(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected no pre-existing FLUX source attention mask.")
        key = self.block_key(extra_options)
        ordinal = execution_ordinal(*key)
        shifted = ordinal % 2 == 1
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        image_tokens = sequence_end - text_tokens
        if image_tokens != math.prod(SOURCE_HW) or q.shape[2] != sequence_end or k.shape[2] != sequence_end:
            raise AssertionError((key, q.shape, k.shape, extra_options["img_slice"]))

        ids = partition_ids(shifted, q.device)
        image_mask = ids[:, None] == ids[None, :]
        mask = torch.zeros((sequence_end, sequence_end), dtype=torch.bool, device=q.device)
        mask[:text_tokens, :] = True
        mask[text_tokens:, :text_tokens] = True
        mask[text_tokens:, text_tokens:] = image_mask
        allowed = int(image_mask.sum())
        dense = image_tokens * image_tokens
        groups = torch.bincount(ids)
        self.restriction_records.append({
            "block_type": key[0],
            "block_index": key[1],
            "execution_ordinal": ordinal,
            "partition": "shifted_clipped" if shifted else "unshifted",
            "shift_yx": list(SHIFT_YX if shifted else (0, 0)),
            "boundary_rule": "non-wrapping clipped windows; bin=floor((coordinate+shift)/16)",
            "source_query_tokens": sequence_end,
            "source_key_tokens": sequence_end,
            "source_q_by_k": [sequence_end, sequence_end],
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "source_grid_hw": list(SOURCE_HW),
            "window_hw": list(WINDOW_HW),
            "number_of_nonempty_windows": int((groups > 0).sum()),
            "minimum_tokens_per_window": int(groups[groups > 0].min()),
            "maximum_tokens_per_window": int(groups.max()),
            "allowed_image_image_edges": allowed,
            "dense_image_image_edges": dense,
            "fraction_dense_image_connectivity": allowed / dense,
            "blocked_image_image_edges": dense - allowed,
            "image_image_communication": "same current-depth window only",
            "text_query_key_policy": "unchanged; text queries attend text and all image keys",
            "image_to_text_policy": "unchanged; image queries attend all text keys",
            "mask_dtype": str(mask.dtype),
            "mask_shape": list(mask.shape),
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": mask}

    def assert_restriction_complete(self) -> None:
        if len(self.restriction_records) != 25:
            raise AssertionError(len(self.restriction_records))
        ordered = sorted(self.restriction_records, key=lambda item: item["execution_ordinal"])
        if [item["execution_ordinal"] for item in ordered] != list(range(25)):
            raise AssertionError("Source block execution schedule is incomplete.")
        if [item["partition"] for item in ordered] != [
            "shifted_clipped" if ordinal % 2 else "unshifted" for ordinal in range(25)
        ]:
            raise AssertionError("Unexpected alternating partition schedule.")


def alternating_capture_options(role: str, probe: AlternatingShiftedProbe) -> dict[str, Any]:
    options = phase2c.model_options(role, {}, probe, "capture")
    phase2c.append_patch(options, "attn1_patch", probe.restrict_alternating_windows)
    return options


class CrossWindowSampler(phase2.comfy.samplers.Sampler):
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
            structure = "ordinary dense all-token attention"
            reachability = None
        elif mode == "fixed":
            probe = phase2k.WindowedSourceProbe()
            capture_options = phase2k.windowed_capture_options(f"{name}_source", probe)
            structure = "fixed nonoverlapping 16x16 windows in all source blocks"
            reachability = reachable_by_depth(False)
        elif mode == "alternating":
            probe = AlternatingShiftedProbe()
            capture_options = alternating_capture_options(f"{name}_source", probe)
            structure = "alternating unshifted and clipped-shifted-by-(8,8) 16x16 windows"
            reachability = reachable_by_depth(True)
        elif mode == "none":
            probe = phase2j.RestrictedSourceProbe()
            capture_options = phase2j.restricted_capture_options(f"{name}_source", probe)
            structure = "image queries attend text only; image-image blocked"
            reachability = [{
                "execution_ordinal": ordinal,
                "partition": "none",
                "minimum_reachable_tokens": 1,
                "maximum_reachable_tokens": 1,
                "mean_reachable_tokens": 1.0,
                "minimum_reachable_fraction": 1 / 2048,
                "maximum_reachable_fraction": 1 / 2048,
                "tokens_reaching_full_canvas": 0,
            } for ordinal in range(25)]
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
            "source_attention_structure": structure,
            "source_restriction_records": getattr(probe, "restriction_records", []),
            "source_capture_records": probe.capture_records,
            "local_context_records": probe.context_records,
            "analytical_image_token_reachability_by_depth": reachability,
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
        for name, mode in zip(VARIANTS, ("ordinary", "fixed", "alternating", "none")):
            self.outputs[name] = self.run_variant(
                model, noise, local_input, sigma, base_options, name, mode
            )
        ordinary = self.layouts[VARIANTS[0]]
        invariant_keys = ("source_generated_tokens", "source_sequence_tokens", "source_q_by_k", "local_external_q_by_k")
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
            "block_type": key[0], "block_index": key[1],
            "execution_ordinal": execution_ordinal(*key),
            "positioned_k_difference": phase2c.tensor_difference(reference["k"], candidate["k"]),
            "v_difference": phase2c.tensor_difference(reference["v"], candidate["v"]),
        })
    return sorted(differences, key=lambda item: item["execution_ordinal"])


def cross_variant_kv_differences(first_probe, second_probe) -> list[dict[str, Any]]:
    differences = kv_differences(first_probe, second_probe)
    return differences


def dry_run() -> None:
    unshifted = partition_ids(False, torch.device("cpu"))
    shifted = partition_ids(True, torch.device("cpu"))
    unshifted_edges = int((unshifted[:, None] == unshifted[None, :]).sum())
    shifted_edges = int((shifted[:, None] == shifted[None, :]).sum())
    dense_edges = math.prod(SOURCE_HW) ** 2
    reachability = reachable_by_depth(True)
    if unshifted_edges != 524288 or shifted_edges != 344064:
        raise AssertionError((unshifted_edges, shifted_edges))
    print(json.dumps({
        "source_tokens": 2048,
        "window_hw": list(WINDOW_HW),
        "shift_yx": list(SHIFT_YX),
        "boundary_rule": "non-wrapping clipped shifted windows",
        "unshifted_windows": int(unshifted.max()) + 1,
        "shifted_nonempty_windows": int(torch.unique(shifted).numel()),
        "unshifted_image_edges": unshifted_edges,
        "shifted_image_edges": shifted_edges,
        "unshifted_fraction": unshifted_edges / dense_edges,
        "shifted_fraction": shifted_edges / dense_edges,
        "reachability_after_selected_depths": [reachability[index] for index in (0, 1, 2, 3, 4, 24)],
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

    sampler = CrossWindowSampler(target_hw, crop, seed)
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
        phase2.save_pixels(vae.decode(dense_crop).cpu(), args.output_dir / "E_DENSE_REFERENCE_CROP.png")
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
            "window_hw": list(WINDOW_HW), "shift_yx": list(SHIFT_YX),
            "shift_schedule": "execution ordinals 0,2,... unshifted; 1,3,... clipped shifted by (8,8)",
            "boundary_rule": "non-wrapping clipped windows; each token assigned by floor((coordinate+shift)/16)",
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "modified_source_blocks": "all 5 double and all 20 single blocks for B/C/D",
            "local_integration_changed": False,
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "comparisons_vs_dense_crop": comparisons,
        "source_kv_differences_from_ordinary": {
            name: kv_differences(ordinary_probe, sampler.probes[name]) for name in VARIANTS[1:]
        },
        "shifted_vs_fixed_source_kv_differences": cross_variant_kv_differences(
            sampler.probes["B_FIXED_WINDOWS"], sampler.probes["C_ALTERNATING_SHIFTED_WINDOWS"]
        ),
        "source_prediction_comparisons_from_ordinary": {
            name: phase2c.tensor_difference(sampler.source_outputs[name], dense_full) for name in VARIANTS[1:]
        },
        "decoded_semantic_observations": {
            "A_ORDINARY_DENSE_SOURCE": {
                "duplicate_lighthouse": "suppressed; only the dense-reference distant silhouette remains",
                "stone_structure": "reduced center-right tower remains",
                "train": "closest continuity and alignment to ordinary dense crop",
                "bridge": "closest deck and cable geometry to ordinary dense crop",
                "horizon_water": "closest agreement; no distinct extra white lighthouse",
            },
            "B_FIXED_WINDOWS": {
                "duplicate_lighthouse": "small white center-right lighthouse returns beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation regress from ordinary dense",
                "bridge": "deck remains continuous but train/cable geometry is inconsistent with ordinary dense",
                "horizon_water": "false lighthouse and tower pair returns on the horizon",
            },
            "C_ALTERNATING_SHIFTED_WINDOWS": {
                "duplicate_lighthouse": "small white center-right lighthouse remains at restricted-control severity",
                "stone_structure": "prominent center-right tower remains",
                "train": "continuous but does not recover ordinary dense placement or segmentation",
                "bridge": "deck remains continuous; cable/train geometry does not improve over fixed windows",
                "horizon_water": "false lighthouse and tower pair remains despite analytical full-canvas reachability",
            },
            "D_NO_IMAGE_IMAGE_SOURCE": {
                "duplicate_lighthouse": "small white center-right lighthouse returns beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation diverge from ordinary dense",
                "bridge": "deck remains continuous but train/cable geometry regresses",
                "horizon_water": "false lighthouse and tower pair returns on the horizon",
            },
            "shifted_interpretation": "graph reachability across depth is not sufficient to preserve the dense-source semantic constraint under this alternating clipped-window topology",
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
            } for name in VARIANTS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
