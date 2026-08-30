"""Zero-update dense-source interaction ablation for Candidate 2."""

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


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS_REPORT = ROOT / "experiments" / "flux2_candidate2_distributed_nonlocal_results" / "report.json"
OUTPUT_DIR = ROOT / "experiments" / "flux2_candidate2_dense_source_interaction_ablation_results"
VARIANTS = ("A_DENSE_SOURCE", "B_RESTRICTED_DENSE_SOURCE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class RestrictedSourceProbe(phase2c.OneEvaluationContextProbe):
    def __init__(self) -> None:
        super().__init__()
        self.restriction_records: list[dict[str, Any]] = []

    def restrict_image_image_attention(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected no pre-existing FLUX source attention mask.")
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        if q.shape[2] != sequence_end or k.shape[2] != sequence_end:
            raise AssertionError((key, q.shape, k.shape, sequence_end))
        mask = torch.ones(
            (sequence_end, sequence_end), dtype=torch.bool, device=q.device
        )
        mask[text_tokens:, text_tokens:] = False
        image_tokens = sequence_end - text_tokens
        self.restriction_records.append(
            {
                "block_type": key[0],
                "block_index": key[1],
                "source_query_tokens": sequence_end,
                "source_key_tokens": sequence_end,
                "source_q_by_k": [sequence_end, sequence_end],
                "text_tokens": text_tokens,
                "image_tokens": image_tokens,
                "allowed_text_query_to_text_keys": text_tokens * text_tokens,
                "allowed_text_query_to_image_keys": text_tokens * image_tokens,
                "allowed_image_query_to_text_keys": image_tokens * text_tokens,
                "blocked_image_query_to_image_keys": image_tokens * image_tokens,
                "image_image_communication": "none within this attention operation",
                "text_query_key_policy": "unchanged; text queries attend text and image keys",
                "image_to_text_policy": "unchanged; image queries attend all text keys",
                "mask_dtype": str(mask.dtype),
                "mask_shape": list(mask.shape),
            }
        )
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": mask}

    def assert_restriction_complete(self) -> None:
        if len(self.restriction_records) != 25:
            raise AssertionError(len(self.restriction_records))
        keys = {(item["block_type"], item["block_index"]) for item in self.restriction_records}
        expected = {(item["block_type"], item["block_index"]) for item in self.capture_records}
        if keys != expected:
            raise AssertionError(keys ^ expected)


def restricted_capture_options(role: str, probe: RestrictedSourceProbe) -> dict[str, Any]:
    options = phase2c.model_options(role, {}, probe, "capture")
    # Capture stores the ordinary projected K/V entering this block. The mask
    # then ablates only the source attention result that forms later hidden
    # states. Thus block 0 K/V is an expected control equality, while blocks
    # 1..24 reflect the accumulated interaction ablation.
    phase2c.append_patch(
        options,
        "attn1_patch",
        probe.restrict_image_image_attention,
    )
    return options


class SourceInteractionDiagnosticSampler(phase2.comfy.samplers.Sampler):
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

    def run_variant(self, model, noise, local_input, sigma, base_options, name, restricted):
        if restricted:
            probe = RestrictedSourceProbe()
            capture_options = restricted_capture_options(f"{name}_source", probe)
        else:
            probe = phase2c.OneEvaluationContextProbe()
            capture_options = phase2c.model_options(f"{name}_source", {}, probe, "capture")
        self.probes[name] = probe
        source = self.run_call(
            model, noise, sigma, capture_options, base_options,
            f"{name}_GLOBAL_CAPTURE", "global_capture",
        )
        output = self.run_call(
            model, local_input, sigma,
            phase2c.model_options(
                f"{name}_local", phase2.rope_for_crop(self.crop), probe, "context"
            ),
            base_options, f"{name}_LOCAL_CONTEXT", "local_with_external_kv",
        )
        probe.assert_complete()
        if restricted:
            probe.assert_restriction_complete()
        first = probe.context_records[0]
        if first["global_generated_kv"] != 2048:
            raise AssertionError(first)
        self.layouts[name] = {
            "source_generated_tokens": 2048,
            "source_sequence_tokens": 2560,
            "source_q_by_k": [2560, 2560],
            "local_external_q_by_k": first["q_by_k"],
            "local_context_records": probe.context_records,
            "source_capture_records": probe.capture_records,
            "source_restriction_records": (
                probe.restriction_records if restricted else []
            ),
            "source_attention_structure": (
                "image queries attend text only; image-to-image blocked; text queries attend all tokens"
                if restricted
                else "ordinary dense all-token attention"
            ),
            "ablated_blocks": (
                "all 5 double and all 20 single blocks" if restricted else "none"
            ),
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
        self.outputs["A_DENSE_SOURCE"] = self.run_variant(
            model, noise, local_input, sigma, base_options, "A_DENSE_SOURCE", False
        )
        self.outputs["B_RESTRICTED_DENSE_SOURCE"] = self.run_variant(
            model, noise, local_input, sigma, base_options, "B_RESTRICTED_DENSE_SOURCE", True
        )
        ordinary = self.layouts["A_DENSE_SOURCE"]
        restricted = self.layouts["B_RESTRICTED_DENSE_SOURCE"]
        invariant_keys = ("source_generated_tokens", "source_sequence_tokens", "source_q_by_k", "local_external_q_by_k")
        if any(ordinary[key] != restricted[key] for key in invariant_keys):
            raise AssertionError("Source/local token geometry changed under ablation.")
        return noise


def dry_run() -> None:
    text_tokens = 512
    image_tokens = 2048
    sequence = text_tokens + image_tokens
    mask = torch.ones((sequence, sequence), dtype=torch.bool)
    mask[text_tokens:, text_tokens:] = False
    if mask[:text_tokens].logical_not().any():
        raise AssertionError("Text queries were restricted.")
    if mask[text_tokens:, :text_tokens].logical_not().any():
        raise AssertionError("Image-to-text attention was restricted.")
    if mask[text_tokens:, text_tokens:].any():
        raise AssertionError("Image-to-image attention remains.")
    print(json.dumps({
        "source_tokens": image_tokens,
        "source_q_by_k": [sequence, sequence],
        "local_external_q_by_k": [1536, 3584],
        "allowed_edges": int(mask.sum()),
        "blocked_image_image_edges": image_tokens * image_tokens,
        "text_queries_unchanged": True,
        "image_queries_retain_text_keys": True,
        "ablated_blocks": 25,
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

    sampler = SourceInteractionDiagnosticSampler(target_hw, crop, seed)
    with torch.inference_mode():
        unchanged = phase2.comfy.sample.sample_custom(
            model, accepted.clone(), 1.0, sampler,
            torch.stack((diagnostic_sigma, torch.zeros_like(diagnostic_sigma))),
            positive, negative, torch.zeros_like(accepted), disable_pbar=True, seed=seed,
        ).detach().float().cpu()
    no_update_difference = phase2c.tensor_difference(unchanged, accepted)
    if no_update_difference["max_abs"] != 0.0:
        raise AssertionError(no_update_difference)

    dense_full = sampler.source_outputs["A_DENSE_SOURCE"]
    dense_crop = dense_full[:, :, crop.y:crop.y2, crop.x:crop.x2]
    comparisons = {}
    for name, output in sampler.outputs.items():
        comparisons[name] = {
            "absolute": phase2c.tensor_difference(output, dense_crop),
            "low_frequency": phase2c.low_frequency_difference(output, dense_crop),
            "prediction": phase2.stats(output),
            "difference_from_A_dense_source_local": phase2c.tensor_difference(
                output, sampler.outputs["A_DENSE_SOURCE"]
            ),
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
    restricted_source = sampler.source_outputs["B_RESTRICTED_DENSE_SOURCE"]
    ordinary_probe = sampler.probes["A_DENSE_SOURCE"]
    restricted_probe = sampler.probes["B_RESTRICTED_DENSE_SOURCE"]
    source_kv_differences = []
    for key in sorted(ordinary_probe.global_kv, key=lambda item: (item[0], item[1])):
        ordinary_entry = ordinary_probe.global_kv[key]
        restricted_entry = restricted_probe.global_kv[key]
        source_kv_differences.append({
            "block_type": key[0],
            "block_index": key[1],
            "positioned_k_difference": phase2c.tensor_difference(
                ordinary_entry["k"], restricted_entry["k"]
            ),
            "v_difference": phase2c.tensor_difference(
                ordinary_entry["v"], restricted_entry["v"]
            ),
        })
    with torch.inference_mode():
        phase2.save_pixels(vae.decode(dense_crop).cpu(), args.output_dir / "C_DENSE_REFERENCE_CROP.png")
        phase2.save_pixels(vae.decode(dense_full).cpu(), args.output_dir / "DENSE_FULL_CANVAS_REFERENCE.png")
        phase2.save_pixels(vae.decode(restricted_source).cpu(), args.output_dir / "RESTRICTED_SOURCE_FULL_CANVAS.png")

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH), "prompt": phase2.PROMPT, "seed": seed, "cfg": 1.0,
            "sigma": float(diagnostic_sigma), "trajectory_updates_reproduced": diagnostic_step,
            "sampler_updates_during_diagnostic": 0, "target_latent_hw": list(target_hw),
            "local_crop": crop.__dict__, "local_rope": phase2.rope_for_crop(crop),
            "source_rope": "ordinary native 32x64 full-canvas coordinates",
            "intervention": "boolean attention mask blocks source image-query to image-key edges only",
            "modified_source_blocks": "all 5 double and all 20 single blocks",
            "local_integration_changed": False,
        },
        "state_reproduction": {"expected_stats": expected_stats, "reproduced_stats": reproduced_stats, "absolute_stat_differences": stat_differences, "accepted_latent_sha256_float32": phase2f.tensor_hash(accepted), "no_update_difference": no_update_difference},
        "layouts": sampler.layouts,
        "calls": sampler.calls,
        "positions": sampler.positions,
        "comparisons_vs_dense_crop": comparisons,
        "source_kv_differences_ordinary_vs_restricted": source_kv_differences,
        "source_prediction_comparison": {
            "restricted_vs_ordinary": phase2c.tensor_difference(restricted_source, dense_full),
            "ordinary": phase2.stats(dense_full),
            "restricted": phase2.stats(restricted_source),
        },
        "decoded_semantic_observations": {
            "A_DENSE_SOURCE": {
                "duplicate_lighthouse": "suppressed; only the dense-reference distant silhouette remains",
                "stone_structure": "reduced center-right tower remains, as in Phase 2f",
                "train": "closest local train continuity/alignment to ordinary dense crop",
                "bridge": "closest deck/cable geometry to ordinary dense crop",
                "horizon_water": "closest agreement; no distinct extra white lighthouse",
            },
            "B_RESTRICTED_DENSE_SOURCE": {
                "duplicate_lighthouse": "small white center-right lighthouse reappears beside the stone tower",
                "stone_structure": "prominent center-right tower remains",
                "train": "carriage extent and segmentation diverge from ordinary dense crop",
                "bridge": "deck remains continuous but train/cable geometry regresses",
                "horizon_water": "duplicate lighthouse/stone pair returns on the horizon",
            },
            "restricted_source_full_canvas": "remains globally interpretable but changes bridge/train geometry; source prediction coherence alone does not preserve the successful local K/V behavior",
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
