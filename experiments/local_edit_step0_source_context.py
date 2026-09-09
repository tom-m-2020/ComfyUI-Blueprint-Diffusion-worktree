"""Step-0 dense source-context diagnostic for Local Edit."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import local_edit_prediction_state_ownership as previous
import flux2_candidate2_four_step_trajectory as trajectory
import flux2_candidate2_one_eval_probe as context_base
import flux2_coarse_global_local_falsification as phase2
from comfy.ldm.flux import math as flux_math


OUTPUT = previous.clown.ROOT / "experiments" / "local_edit_step0_source_context_results"
LATENT_HW = (32, 64)
TEXT_TOKENS = 512
GENERATED_TOKENS = math.prod(LATENT_HW)
SOURCE_TOKENS = 32 * 32
EDITABLE_TOKENS = GENERATED_TOKENS - SOURCE_TOKENS


def tensor_difference(value: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    delta = value.detach().float() - reference.detach().float()
    if mask is not None:
        delta = delta[mask.expand_as(delta).bool()]
    return {
        "rms": float(delta.square().mean().sqrt()),
        "mean_abs": float(delta.abs().mean()),
        "max_abs": float(delta.abs().max()),
    }


def generated_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    columns = torch.arange(LATENT_HW[1], device=device)[None, :].expand(LATENT_HW)
    locked = ((columns >= 16) & (columns < 48)).reshape(-1)
    return ~locked, locked


class SourceContextProbe:
    def __init__(self) -> None:
        self.source_kv: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        self.normal_attention: dict[tuple[str, int], torch.Tensor] = {}
        self.capture_records: list[dict[str, Any]] = []
        self.context_records: list[dict[str, Any]] = []

    @staticmethod
    def block_key(extra_options: dict[str, Any]) -> tuple[str, int]:
        return str(extra_options["block_type"]), int(extra_options["block_index"])

    def capture_source(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected native dense Klein attention without a mask")
        key = self.block_key(extra_options)
        text_start, image_end = map(int, extra_options["img_slice"])
        if text_start != TEXT_TOKENS or image_end - text_start != GENERATED_TOKENS:
            raise AssertionError((extra_options["img_slice"], q.shape, k.shape))
        _, locked = generated_masks(k.device)
        positioned_k = flux_math.apply_rope1(k, pe)
        self.source_kv[key] = {
            "k": positioned_k[:, :, text_start:image_end][:, :, locked].detach(),
            "v": v[:, :, text_start:image_end][:, :, locked].detach(),
        }
        self.capture_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_tokens": text_start, "full_generated_tokens": GENERATED_TOKENS,
            "source_generated_tokens": int(locked.sum()),
            "heads": int(k.shape[1]), "head_dim": int(k.shape[-1]),
        })
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def add_source_context(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected native dense Klein attention without a mask")
        key = self.block_key(extra_options)
        if key not in self.source_kv:
            raise KeyError(f"Missing same-sigma source K/V for {key}")
        text_start, image_end = map(int, extra_options["img_slice"])
        editable, locked = generated_masks(q.device)
        self.normal_attention[key] = flux_math.attention(
            q, k, v, pe=pe, mask=attn_mask, transformer_options=extra_options
        )
        positioned_q = flux_math.apply_rope1(q, pe)
        positioned_k = flux_math.apply_rope1(k, pe)
        source = self.source_kv[key]
        augmented_k = torch.cat((positioned_k, source["k"]), dim=2)
        augmented_v = torch.cat((v, source["v"]), dim=2)
        self.context_records.append({
            "block_type": key[0], "block_index": key[1],
            "text_queries": text_start,
            "editable_generated_queries": int(editable.sum()),
            "locked_generated_queries_restored": int(locked.sum()),
            "ordinary_kv_tokens": int(k.shape[2]),
            "added_source_kv_tokens": int(source["k"].shape[2]),
            "augmented_kv_tokens": int(augmented_k.shape[2]),
            "q_by_k": [int(q.shape[2]), int(augmented_k.shape[2])],
        })
        return {"q": positioned_q, "k": augmented_k, "v": augmented_v, "pe": None, "attn_mask": None}

    def restore_noneditable_queries(self, attention_output, extra_options):
        key = self.block_key(extra_options)
        normal = self.normal_attention.pop(key)
        text_start, image_end = map(int, extra_options["img_slice"])
        _, locked = generated_masks(attention_output.device)
        output = attention_output.clone()
        output[:, :text_start] = normal[:, :text_start]
        image_output = output[:, text_start:image_end]
        normal_image = normal[:, text_start:image_end]
        image_output[:, locked] = normal_image[:, locked]
        return output

    def assert_complete(self) -> None:
        if self.normal_attention:
            raise AssertionError(f"Unrestored attention outputs: {tuple(self.normal_attention)}")
        captured = {(r["block_type"], r["block_index"]) for r in self.capture_records}
        consumed = {(r["block_type"], r["block_index"]) for r in self.context_records}
        expected = {("double", i) for i in range(5)} | {("single", i) for i in range(20)}
        if captured != expected or consumed != expected:
            raise AssertionError((captured ^ expected, consumed ^ expected))


def options(probe: SourceContextProbe, mode: str) -> dict[str, Any]:
    result: dict[str, Any] = {"transformer_options": {}}
    if mode == "capture":
        context_base.append_patch(result, "attn1_patch", probe.capture_source)
    elif mode == "context":
        context_base.append_patch(result, "attn1_patch", probe.add_source_context)
        context_base.append_patch(result, "attn1_output_patch", probe.restore_noneditable_queries)
    elif mode != "ordinary":
        raise ValueError(mode)
    return result


class StepZeroSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, source_latent: torch.Tensor, fixed_noise: torch.Tensor, seed: int) -> None:
        self.source_latent = source_latent
        self.fixed_noise = fixed_noise
        self.seed = seed
        self.outputs: dict[str, torch.Tensor] = {}
        self.source_capture_output: torch.Tensor | None = None
        self.probe = SourceContextProbe()
        self.records: dict[str, Any] = {}

    def call(self, model, value, sigma, base_options, experimental, role):
        started = time.perf_counter()
        output = model(
            value, sigma.expand(value.shape[0]),
            model_options=trajectory.merge_options(base_options, experimental), seed=self.seed,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.records[role] = {
            "seconds": time.perf_counter() - started,
            "input": phase2.stats(value), "output": phase2.stats(output),
        }
        return output

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("Step-0 source-context diagnostic rejects noise masks")
        if len(sigmas) != 2 or float(sigmas[0]) != 1.0 or float(sigmas[1]) != 0.0:
            raise ValueError(f"Expected diagnostic sigmas [1,0], got {sigmas.tolist()}")
        sampling = model.inner_model.model_sampling
        sigma = sigmas[0]
        source = self.source_latent.to(device=noise.device, dtype=noise.dtype)
        fixed_noise = self.fixed_noise.to(device=noise.device, dtype=noise.dtype)
        accepted = sampling.noise_scaling(sigma, fixed_noise, source, self.max_denoise(model, sigmas))
        source_at_sigma = sampling.noise_scaling(sigma, fixed_noise, source, self.max_denoise(model, sigmas))
        base_options = extra_args["model_options"]
        self.outputs["D0_STATE_RESTORATION"] = self.call(
            model, accepted, sigma, base_options, options(self.probe, "ordinary"), "D0_CONTROL"
        ).detach().float().cpu()
        self.source_capture_output = self.call(
            model, source_at_sigma, sigma, base_options, options(self.probe, "capture"), "SOURCE_CAPTURE"
        ).detach().float().cpu()
        self.outputs["D1_SOURCE_CONTEXT"] = self.call(
            model, accepted, sigma, base_options, options(self.probe, "context"), "D1_CONTEXT"
        ).detach().float().cpu()
        self.probe.assert_complete()
        self.records["state"] = {
            "accepted_vs_source_sigma": tensor_difference(accepted, source_at_sigma),
            "source_sigma_vs_fixed_noise": tensor_difference(source_at_sigma, fixed_noise),
            "returned_state_vs_accepted": tensor_difference(accepted, accepted),
            "sampler_updates": 0,
        }
        return accepted


def editable_mask(value: torch.Tensor) -> torch.Tensor:
    mask = torch.ones_like(value[:, :1], dtype=torch.bool)
    mask[..., 16:48] = False
    return mask


def decoded_boundary_metrics(image: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    value = image.float().permute(0, 3, 1, 2)
    target = source.float().permute(0, 3, 1, 2)
    low_value = F.avg_pool2d(value, 33, stride=1, padding=16)
    low_target = F.avg_pool2d(target, 33, stride=1, padding=16)
    result = {}
    for name, boundary in (("left", 256), ("right", 768)):
        strip = slice(boundary - 24, boundary + 24)
        delta = low_value[..., strip] - low_target[..., strip]
        seam = value[..., boundary] - value[..., boundary - 1]
        result[f"{name}_lf_source_discrepancy_rms"] = float(delta.square().mean().sqrt())
        result[f"{name}_seam_gradient_rms"] = float(seam.square().mean().sqrt())
    return result


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = previous.clown.comfy.sd.VAE(sd=previous.clown.comfy.utils.load_torch_file(
        str(previous.clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    for case, config in previous.CASES.items():
        pixels = previous.clown.load_pixels(Path(config["source"]))
        with torch.no_grad():
            latent = vae.encode(pixels)
            source_decode = vae.decode(latent).detach().cpu()
        prepared[case] = {"latent": latent.detach().clone(), "source_decode": source_decode}
        previous.clown.metrics_base.save_pixels(source_decode, OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png")
    del vae
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()

    model = previous.clown.comfy.sd.load_diffusion_model(str(previous.clown.model_paths.MODEL_PATH), model_options={})
    clip = previous.clown.comfy.sd.load_clip(
        [str(previous.clown.model_paths.TEXT_ENCODER_PATH)], clip_type=previous.clown.comfy.sd.CLIPType.FLUX2
    )
    sigma = previous.clown.get_schedule(previous.clown.STEPS, GENERATED_TOKENS).float()[0]
    report_cases = {}
    outputs = {}
    for case, config in previous.CASES.items():
        item = prepared[case]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = previous.clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        noise = phase2.comfy.sample.prepare_noise(item["latent"], previous.clown.SEED)
        sampler = StepZeroSampler(item["latent"], noise, previous.clown.SEED)
        with torch.inference_mode():
            returned = phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, torch.stack((sigma, torch.zeros_like(sigma))),
                positive, negative, item["latent"].clone(), disable_pbar=True, seed=previous.clown.SEED,
            ).detach().float().cpu()
        outputs[case] = {"sampler": sampler, "returned": returned}
    del clip
    model.cleanup()
    del model
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = previous.clown.comfy.sd.VAE(sd=previous.clown.comfy.utils.load_torch_file(
        str(previous.clown.model_paths.VAE_PATH), safe_load=True
    ))
    for case, values in outputs.items():
        sampler = values["sampler"]
        decoded = {}
        for arm, prediction in sampler.outputs.items():
            pixels = vae.decode(prediction).detach().cpu()
            decoded[arm] = pixels
            previous.clown.metrics_base.save_pixels(pixels, OUTPUT / f"{case}_{arm}_RAW_X0.png")
        previous.clown.metrics_base.make_sheet([
            ("SOURCE RECONSTRUCTION", OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png"),
            ("D0 STATE RESTORATION", OUTPUT / f"{case}_D0_STATE_RESTORATION_RAW_X0.png"),
            ("D1 SOURCE CONTEXT", OUTPUT / f"{case}_D1_SOURCE_CONTEXT_RAW_X0.png"),
        ], OUTPUT / f"{case}_STEP0_COMPARISON.png")
        d0 = sampler.outputs["D0_STATE_RESTORATION"]
        d1 = sampler.outputs["D1_SOURCE_CONTEXT"]
        edit = editable_mask(d0)
        report_cases[case] = {
            "prompt": previous.CASES[case]["prompt"],
            "editable_raw_x0_D1_vs_D0": tensor_difference(d1, d0, edit),
            "locked_raw_x0_D1_vs_D0": tensor_difference(d1, d0, ~edit),
            "decoded": {
                arm: decoded_boundary_metrics(image, prepared[case]["source_decode"])
                for arm, image in decoded.items()
            },
            "calls": sampler.records,
        }
    del vae
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()

    first_layout = outputs["rigid_bridge"]["sampler"].probe
    report = {
        "experiment": "Local Edit step-0 native dense source K/V context",
        "configuration": {
            "sigma": float(sigma), "seed": previous.clown.SEED, "cfg": 1.0,
            "latent_hw": list(LATENT_HW), "text_tokens": TEXT_TOKENS,
            "ordinary_generated_tokens": GENERATED_TOKENS,
            "editable_query_tokens": EDITABLE_TOKENS,
            "added_source_kv_tokens": SOURCE_TOKENS,
            "attention_q_by_k": [TEXT_TOKENS + GENERATED_TOKENS,
                                  TEXT_TOKENS + GENERATED_TOKENS + SOURCE_TOKENS],
            "blocks_modified": "all 5 double and all 20 single attention blocks",
            "text_queries": "ordinary attention output restored exactly",
            "locked_generated_queries": "ordinary attention output restored exactly",
            "editable_generated_queries": "attend ordinary text+generated K/V plus source-region K/V",
            "source_coordinates": "native full-canvas RoPE retained before source K capture",
            "source_state": "live CONST noise_scaling(sigma=1,z,y)=z",
            "accepted_state_updates": 0,
        },
        "capture_layout": first_layout.capture_records,
        "context_layout": first_layout.context_records,
        "cases": report_cases,
        "diffusion_trajectory_executed": False,
        "production_changed": False,
    }
    previous.clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "cases": {case: value["editable_raw_x0_D1_vs_D0"] for case, value in report_cases.items()},
    }, indent=2))


if __name__ == "__main__":
    run()
