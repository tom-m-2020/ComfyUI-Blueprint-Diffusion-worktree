"""Exact Clown epsilon mechanism and editable-only causal discriminator."""

from __future__ import annotations

import gc
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

import local_edit_prediction_state_ownership as previous
import local_edit_user_clown_epsilon_reproduction as clown
from RES4LYF.beta.rk_guide_func_beta import LatentGuide
from RES4LYF.latents import get_collinear, get_orthogonal


OUTPUT = clown.ROOT / "experiments" / "local_edit_epsilon_mechanism_results"
REGION_NAMES = ("source", "editable", "boundary", "whole")


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def masks(value: torch.Tensor) -> dict[str, torch.Tensor]:
    editable, source = previous.latent_masks(value)
    return {
        "source": source.bool(),
        "editable": editable.bool(),
        "boundary": previous.latent_boundary_mask(value).bool(),
        "whole": torch.ones_like(source, dtype=torch.bool),
    }


def region_stats(value: torch.Tensor, region_masks: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    result = {}
    for name in REGION_NAMES:
        selected = value.detach().float().masked_select(region_masks[name].expand_as(value))
        result[name] = {
            "rms": float(selected.square().mean().sqrt()),
            "mean_abs": float(selected.abs().mean()),
            "max_abs": float(selected.abs().max()),
        }
    return result


class Telemetry:
    def __init__(self) -> None:
        self.calls_per_step: dict[int, int] = {}
        self.intervals: list[dict[str, Any]] = []
        self.tensors: list[dict[str, torch.Tensor]] = []
        self.accepted: list[torch.Tensor] = []
        self.acceptance_records: list[dict[str, Any]] = []


def editable_only_correction(raw: torch.Tensor, guided: torch.Tensor,
                             editable: torch.Tensor) -> torch.Tensor:
    return raw + editable.to(raw) * (guided - raw)


@contextmanager
def instrument(telemetry: Telemetry, editable_only: bool):
    original_guidance = LatentGuide.process_guides_substep
    original_preview = clown.rk_sampler_beta.preview_callback

    def guide_wrapper(self, *args, **kwargs):
        x_0, x_rows, eps_rows, data_rows = args[:4]
        row, step, step_sched = int(args[5]), int(args[6]), int(args[7])
        sigma, sigma_down, s_rows, epsilon_scale, rk = args[8], args[10], args[11], args[12], args[13]
        raw_derivative = eps_rows[row].detach().clone()
        raw_x0 = data_rows[row].detach().clone()
        guide_quantity = torch.zeros_like(raw_derivative)
        weight_mask = torch.zeros_like(raw_derivative[:, :1])
        plain = raw_derivative.clone()
        projected = raw_derivative.clone()
        if self.HAS_LATENT_GUIDE:
            y0, _, weight_mask, _ = self.get_cossim_adjusted_lgw_masks(raw_x0, step_sched)
            guide_quantity = rk.get_guide_epsilon(
                x_0, x_rows[row], y0, sigma, s_rows[row], sigma_down, epsilon_scale
            )
            plain = raw_derivative + weight_mask * (guide_quantity - raw_derivative)
            masked_guide = self.mask * guide_quantity
            projection_target = (
                get_collinear(raw_derivative, masked_guide)
                + get_orthogonal(masked_guide, raw_derivative)
            )
            projected = raw_derivative + weight_mask * (projection_target - raw_derivative)

        result = original_guidance(self, *args, **kwargs)
        exact_guided = result[0][row].detach().clone()
        if editable_only:
            edit, _ = previous.latent_masks(exact_guided)
            result[0][row].copy_(editable_only_correction(raw_derivative, exact_guided, edit))
        final_guided = result[0][row].detach().clone()

        call_index = telemetry.calls_per_step.get(step, 0)
        telemetry.calls_per_step[step] = call_index + 1
        if call_index == 0:
            state = x_rows[row].detach().clone()
            region_masks = masks(state)
            correction_before_projection = plain - raw_derivative
            projection_delta = projected - plain
            final_correction = final_guided - raw_derivative
            telemetry.intervals.append({
                "step": step, "step_sched": step_sched, "sigma": float(sigma),
                "sigma_row": float(s_rows[row]), "mode": self.guide_mode,
                "accepted_x_sigma": region_stats(state, region_masks),
                "raw_model_x0": region_stats(raw_x0, region_masks),
                "raw_sampler_derivative": region_stats(raw_derivative, region_masks),
                "guide_reference_quantity": region_stats(guide_quantity, region_masks),
                "epsilon_correction_before_projection": region_stats(correction_before_projection, region_masks),
                "projection_change_vs_plain": region_stats(projection_delta, region_masks),
                "exact_clown_correction": region_stats(exact_guided - raw_derivative, region_masks),
                "final_guided_derivative": region_stats(final_guided, region_masks),
                "final_correction": region_stats(final_correction, region_masks),
                "guide_weight_mask": region_stats(weight_mask, {k: v[:, :1] for k, v in region_masks.items()}),
            })
            telemetry.tensors.append({
                "accepted_x_sigma": state.cpu(), "raw_model_x0": raw_x0.cpu(),
                "raw_sampler_derivative": raw_derivative.cpu(),
                "guide_reference_quantity": guide_quantity.cpu(),
                "epsilon_correction_before_projection": correction_before_projection.cpu(),
                "projected_correction": (projected - raw_derivative).cpu(),
                "final_guided_derivative": final_guided.cpu(),
            })
        return result

    def preview_wrapper(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
                        callback, extra_options, **kwargs):
        final = bool(kwargs.get("final", False))
        if not final:
            accepted = x.detach().cpu().clone()
            telemetry.accepted.append(accepted)
            index = len(telemetry.acceptance_records)
            region_masks = masks(accepted)
            telemetry.acceptance_records.append({
                "step": int(kwargs.get("step_sched", step)), "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "euler_proposal": region_stats(accepted, region_masks),
                "accepted_next_state": region_stats(accepted, region_masks),
                "proposal_equals_accepted": True,
            })
            if index < len(telemetry.tensors):
                telemetry.tensors[index]["euler_proposal"] = accepted
                telemetry.tensors[index]["accepted_next_state"] = accepted.clone()
        return original_preview(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma,
                                sigma_next, callback, extra_options, **kwargs)

    LatentGuide.process_guides_substep = guide_wrapper
    clown.rk_sampler_beta.preview_callback = preview_wrapper
    try:
        yield
    finally:
        LatentGuide.process_guides_substep = original_guidance
        clown.rk_sampler_beta.preview_callback = original_preview


def execute(model, positive, negative, sigmas, latent, guide, editable_only=False):
    telemetry = Telemetry()
    with instrument(telemetry, editable_only), torch.inference_mode():
        result = clown.ClownsharKSampler_Beta.execute(
            model=model, positive=positive, negative=negative,
            latent_image={"samples": latent["samples"].clone()}, sigmas=sigmas.clone(),
            guides=guide, eta=0.5, sampler_name="linear/euler", scheduler="simple",
            steps=clown.STEPS, steps_to_run=clown.STEPS, denoise=1.0, cfg=1.0,
            seed=clown.SEED, sampler_mode="standard", bongmath=True,
        )
    return result[0]["samples"].detach().cpu(), telemetry


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    for case, config in previous.CASES.items():
        pixels = clown.load_pixels(Path(config["source"]))
        with torch.no_grad():
            latent = vae.encode(pixels)
            source_decode = vae.decode(latent).detach().cpu()
        prepared[case] = {"latent": {"samples": latent.detach().clone()}, "source": source_decode}
        clown.metrics_base.save_pixels(source_decode, OUTPUT / f"{case}_SOURCE.png")
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    outputs = {}
    started = time.perf_counter()
    for case, config in previous.CASES.items():
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        item = prepared[case]
        direct_mask = torch.ones((1, clown.HEIGHT, clown.WIDTH), dtype=torch.float32)
        direct_mask[:, :, clown.LEFT_BOUNDARY:clown.RIGHT_BOUNDARY] = 0
        plain = clown.make_guides(item["latent"], direct_mask, False)
        projection = clown.make_guides(item["latent"], direct_mask, True)
        outputs[case] = {}
        for arm, guide, edit_only in (
            ("A_ORDINARY", None, False),
            ("B_PLAIN_EPSILON", plain, False),
            ("C_EPSILON_PROJECTION", projection, False),
            ("D_PROJECTION_EDITABLE_ONLY", projection, True),
        ):
            sampled, telemetry = execute(model, positive, negative, sigmas, item["latent"], guide, edit_only)
            outputs[case][arm] = {"sampled": sampled, "telemetry": telemetry}
    del clip
    model.cleanup()
    del model
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    report_cases = {}
    tensor_archive = {}
    for case, arms in outputs.items():
        sheet = [("SOURCE", OUTPUT / f"{case}_SOURCE.png")]
        report_arms = {}
        tensor_archive[case] = {}
        for arm, values in arms.items():
            final = vae.decode(values["sampled"]).detach().cpu()
            final_path = OUTPUT / f"{case}_{arm}_FINAL.png"
            clown.metrics_base.save_pixels(final, final_path)
            sheet.append((arm, final_path))
            for step, tensors in enumerate(values["telemetry"].tensors):
                raw = vae.decode(tensors["raw_model_x0"]).detach().cpu()
                accepted = vae.decode(values["telemetry"].accepted[step]).detach().cpu()
                clown.metrics_base.save_pixels(raw, OUTPUT / f"{case}_{arm}_STEP_{step:02d}_RAW_X0.png")
                clown.metrics_base.save_pixels(accepted, OUTPUT / f"{case}_{arm}_STEP_{step:02d}_ACCEPTED.png")
            report_arms[arm] = {
                "intervals": values["telemetry"].intervals,
                "acceptances": values["telemetry"].acceptance_records,
                "final": clown.boundary_metrics(final, prepared[case]["source"]),
            }
            tensor_archive[case][arm] = values["telemetry"].tensors
        clown.metrics_base.make_sheet(sheet, OUTPUT / f"{case}_FINAL_COMPARISON.png")
        report_cases[case] = {"prompt": previous.CASES[case]["prompt"], "arms": report_arms}
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    report = {
        "experiment": "Exact Clown epsilon mechanism and editable-only causal discriminator",
        "configuration": {"steps": clown.STEPS, "sigmas": [float(v) for v in sigmas],
                          "cfg": 1.0, "sampler": "RES4LYF linear/euler",
                          "noise_mask": None, "hard_restoration": False},
        "order": ["accepted x_sigma", "Klein/CFG raw x0", "raw CONST derivative",
                  "exact Clown guide epsilon", "plain/projection correction",
                  "final guided derivative", "Euler proposal", "accepted next state"],
        "cases": report_cases, "runtime_seconds": time.perf_counter() - started,
        "production_changed": False,
    }
    clown.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", tensor_archive)
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"),
                      "runtime_seconds": report["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    run()
