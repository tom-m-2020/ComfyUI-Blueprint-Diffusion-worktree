"""Zero-update dense-block necessity discriminator for Candidate 2."""

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
import flux2_candidate2_cross_window_propagation as phase2l


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_cross_window_propagation_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_dense_block_necessity_results"
VARIANTS = (
    "A_ALL_DENSE",
    "B_EARLY_DENSE_0_4",
    "C_LATE_DENSE_20_24",
    "D_NO_IMAGE_IMAGE",
)
IMAGE_TOKENS = 2048
TEXT_TOKENS = 512
DENSE_IMAGE_EDGES_PER_BLOCK = IMAGE_TOKENS * IMAGE_TOKENS
ALL_DENSE_IMAGE_EDGES = 25 * DENSE_IMAGE_EDGES_PER_BLOCK


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class SelectiveDenseProbe(phase2c.OneEvaluationContextProbe):
    def __init__(self, dense_ordinals: set[int]) -> None:
        super().__init__()
        self.dense_ordinals = set(dense_ordinals)
        self.policy_records: list[dict[str, Any]] = []

    def apply_source_policy(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected no pre-existing FLUX source attention mask.")
        key = self.block_key(extra_options)
        ordinal = phase2l.execution_ordinal(*key)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        image_tokens = sequence_end - text_tokens
        if text_tokens != TEXT_TOKENS or image_tokens != IMAGE_TOKENS:
            raise AssertionError((key, extra_options["img_slice"]))
        dense = ordinal in self.dense_ordinals
        self.policy_records.append({
            "block_type": key[0],
            "block_index": key[1],
            "execution_ordinal": ordinal,
            "image_image_policy": "ordinary_dense" if dense else "blocked",
            "source_q_by_k": [sequence_end, sequence_end],
            "dense_image_image_edges": DENSE_IMAGE_EDGES_PER_BLOCK if dense else 0,
            "blocked_image_image_edges": 0 if dense else DENSE_IMAGE_EDGES_PER_BLOCK,
            "fraction_dense_connectivity": 1.0 if dense else 0.0,
            "text_query_key_policy": "unchanged; text queries attend text and all image keys",
            "image_to_text_policy": "unchanged; image queries attend all text keys",
        })
        if dense:
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": None}
        mask = torch.ones((sequence_end, sequence_end), dtype=torch.bool, device=q.device)
        mask[text_tokens:, text_tokens:] = False
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": mask}

    def assert_policy_complete(self) -> None:
        ordered = sorted(self.policy_records, key=lambda item: item["execution_ordinal"])
        if [item["execution_ordinal"] for item in ordered] != list(range(25)):
            raise AssertionError("Incomplete source block policy schedule.")
        actual_dense = {item["execution_ordinal"] for item in ordered if item["image_image_policy"] == "ordinary_dense"}
        if actual_dense != self.dense_ordinals:
            raise AssertionError((actual_dense, self.dense_ordinals))


def selective_capture_options(role: str, probe: SelectiveDenseProbe) -> dict[str, Any]:
    options = phase2c.model_options(role, {}, probe, "capture")
    phase2c.append_patch(options, "attn1_patch", probe.apply_source_policy)
    return options


class DenseBlockSampler(phase2.comfy.samplers.Sampler):
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
        if mode == "all":
            probe = phase2c.OneEvaluationContextProbe()
            capture_options = phase2c.model_options(f"{name}_source", {}, probe, "capture")
            dense_ordinals = set(range(25))
            policy_records = [{
                "block_type": "double" if ordinal < 5 else "single",
                "block_index": ordinal if ordinal < 5 else ordinal - 5,
                "execution_ordinal": ordinal,
                "image_image_policy": "ordinary_dense",
                "dense_image_image_edges": DENSE_IMAGE_EDGES_PER_BLOCK,
                "fraction_dense_connectivity": 1.0,
            } for ordinal in range(25)]
        elif mode == "early":
            dense_ordinals = set(range(5))
            probe = SelectiveDenseProbe(dense_ordinals)
            capture_options = selective_capture_options(f"{name}_source", probe)
            policy_records = None
        elif mode == "late":
            dense_ordinals = set(range(20, 25))
            probe = SelectiveDenseProbe(dense_ordinals)
            capture_options = selective_capture_options(f"{name}_source", probe)
            policy_records = None
        elif mode == "none":
            dense_ordinals = set()
            probe = phase2j.RestrictedSourceProbe()
            capture_options = phase2j.restricted_capture_options(f"{name}_source", probe)
            policy_records = [{
                "block_type": "double" if ordinal < 5 else "single",
                "block_index": ordinal if ordinal < 5 else ordinal - 5,
                "execution_ordinal": ordinal,
                "image_image_policy": "blocked",
                "dense_image_image_edges": 0,
                "fraction_dense_connectivity": 0.0,
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
        if isinstance(probe, SelectiveDenseProbe):
            probe.assert_policy_complete()
            policy_records = sorted(probe.policy_records, key=lambda item: item["execution_ordinal"])
        elif mode == "none":
            probe.assert_restriction_complete()
        first = probe.context_records[0]
        total_dense_edges = len(dense_ordinals) * DENSE_IMAGE_EDGES_PER_BLOCK
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
            "dense_image_image_edges_per_dense_block": DENSE_IMAGE_EDGES_PER_BLOCK,
            "total_dense_image_image_edges": total_dense_edges,
            "fraction_all_dense_image_edge_budget": total_dense_edges / ALL_DENSE_IMAGE_EDGES,
            "source_policy_records": policy_records,
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
        for name, mode in zip(VARIANTS, ("all", "early", "late", "none")):
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
    for key in sorted(reference_probe.global_kv, key=lambda item: phase2l.execution_ordinal(*item)):
        reference = reference_probe.global_kv[key]
        candidate = candidate_probe.global_kv[key]
        differences.append({
            "block_type": key[0], "block_index": key[1],
            "execution_ordinal": phase2l.execution_ordinal(*key),
            "positioned_k_difference": phase2c.tensor_difference(reference["k"], candidate["k"]),
            "v_difference": phase2c.tensor_difference(reference["v"], candidate["v"]),
        })
    return differences


def validate_causal_kv(ordinary, early, late, restricted) -> dict[str, Any]:
    early_vs_ordinary = kv_differences(ordinary, early)
    late_vs_restricted = kv_differences(restricted, late)
    early_tracking = [item for item in early_vs_ordinary if item["execution_ordinal"] <= 5]
    late_tracking = [item for item in late_vs_restricted if item["execution_ordinal"] <= 20]
    if any(item["positioned_k_difference"]["rms"] != 0.0 or item["v_difference"]["rms"] != 0.0 for item in early_tracking):
        raise AssertionError("Early-dense stopped tracking ordinary before the first restricted result.")
    if any(item["positioned_k_difference"]["rms"] != 0.0 or item["v_difference"]["rms"] != 0.0 for item in late_tracking):
        raise AssertionError("Late-dense stopped tracking restricted before the first dense result.")
    if early_vs_ordinary[6]["positioned_k_difference"]["rms"] == 0.0:
        raise AssertionError("Early-dense did not diverge after restriction began.")
    if late_vs_restricted[21]["positioned_k_difference"]["rms"] == 0.0:
        raise AssertionError("Late-dense did not diverge after dense attention began.")
    return {
        "early_dense_tracks_ordinary_through_capture_ordinal": 5,
        "early_dense_first_divergent_capture_ordinal": 6,
        "late_dense_tracks_restricted_through_capture_ordinal": 20,
        "late_dense_first_divergent_capture_ordinal": 21,
        "early_vs_ordinary": early_vs_ordinary,
        "late_vs_restricted": late_vs_restricted,
    }


def dry_run() -> None:
    variants = {
        "A_ALL_DENSE": set(range(25)),
        "B_EARLY_DENSE_0_4": set(range(5)),
        "C_LATE_DENSE_20_24": set(range(20, 25)),
        "D_NO_IMAGE_IMAGE": set(),
    }
    print(json.dumps({name: {
        "dense_ordinals": sorted(ordinals),
        "dense_blocks": [("double", ordinal) if ordinal < 5 else ("single", ordinal - 5) for ordinal in sorted(ordinals)],
        "dense_edges_per_block": DENSE_IMAGE_EDGES_PER_BLOCK,
        "total_dense_edges": len(ordinals) * DENSE_IMAGE_EDGES_PER_BLOCK,
        "fraction_all_dense_edge_budget": len(ordinals) / 25,
        "source_q_by_k": [2560, 2560],
        "local_q_by_k": [1536, 3584],
        "sampler_updates": 0,
    } for name, ordinals in variants.items()}, indent=2))


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

    sampler = DenseBlockSampler(target_hw, crop, seed)
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
    restricted_probe = sampler.probes[VARIANTS[3]]
    causal = validate_causal_kv(
        ordinary_probe, sampler.probes[VARIANTS[1]], sampler.probes[VARIANTS[2]], restricted_probe
    )
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "source_grid_hw": list(target_hw),
            "execution_order": "double blocks 0..4 are ordinals 0..4; single blocks 0..19 are ordinals 5..24",
            "early_dense_ordinals": list(range(5)), "late_dense_ordinals": list(range(20, 25)),
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "local_integration_changed": False,
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "comparisons_vs_dense_crop": comparisons,
        "source_kv_differences_from_all_dense": {
            name: kv_differences(ordinary_probe, sampler.probes[name]) for name in VARIANTS[1:]
        },
        "causal_kv_checks": causal,
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
            "B_EARLY_DENSE_0_4": {
                "duplicate_lighthouse": "small white center-right lighthouse remains beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "materially closer to all-dense than the restricted control, but carriage extent still differs",
                "bridge": "deck/cable organization improves over restricted, without restoring all-dense semantics",
                "horizon_water": "false lighthouse and tower pair remains despite early dense interaction",
            },
            "C_LATE_DENSE_20_24": {
                "duplicate_lighthouse": "small white center-right lighthouse remains beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "only modestly improved from the restricted control",
                "bridge": "deck remains continuous, but train/cable geometry remains restricted-like",
                "horizon_water": "false lighthouse and tower pair remains",
            },
            "D_NO_IMAGE_IMAGE": {
                "duplicate_lighthouse": "small white center-right lighthouse remains beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation diverge from all-dense",
                "bridge": "deck remains continuous, but train/cable geometry regresses",
                "horizon_water": "false lighthouse and tower pair remains",
            },
            "interpretation": "early dense blocks carry more numerical and geometric benefit than late dense blocks, but neither five-block subset preserves object uniqueness",
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
