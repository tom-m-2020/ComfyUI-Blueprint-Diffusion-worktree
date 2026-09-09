"""Prediction consistency versus accepted-state ownership for Local Edit."""

from __future__ import annotations

import gc
import builtins
import json
import os
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_real_open = builtins.open
_res4lyf_config = str(Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev\custom_nodes\RES4LYF\res4lyf.config.json"))


def _read_only_res4lyf_config_open(file, mode="r", *args, **kwargs):
    if os.path.normcase(str(file)) == os.path.normcase(_res4lyf_config) and any(flag in mode for flag in "wax+"):
        return _real_open(os.devnull, "w", *args, **kwargs)
    return _real_open(file, mode, *args, **kwargs)


builtins.open = _read_only_res4lyf_config_open
try:
    import local_edit_composite_robustness as qualified
    import local_edit_composite_robustness_generate as generated_cases
    import local_edit_user_clown_epsilon_reproduction as clown
    from RES4LYF.beta.rk_guide_func_beta import LatentGuide
finally:
    builtins.open = _real_open


OUTPUT = clown.ROOT / "experiments" / "local_edit_prediction_state_ownership_results"
LATENT_BOUNDARY_WIDTH = 2
PIXEL_BOUNDARY_WIDTH = 24
LF_KERNEL = 33

CASES = {
    "rigid_bridge": {"source": qualified.CASES["rigid_bridge"]["source"], "prompt": clown.PROMPT},
    "organic_tree": {
        "source": qualified.CASES["organic_tree"]["source"],
        "prompt": generated_cases.CASES["organic_contour"]["prompt"],
    },
    "photometric_desert": {
        "source": qualified.CASES["photometric_desert"]["source"],
        "prompt": generated_cases.CASES["photometric_texture"]["prompt"],
    },
}


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def latent_masks(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edit = torch.ones_like(source[:, :1])
    edit[..., :, 16:48] = 0
    return edit, 1 - edit


def source_state(source: torch.Tensor, fixed_noise: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (1 - sigma) * source + sigma * fixed_noise


def latent_boundary_mask(value: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros_like(value[:, :1])
    w = LATENT_BOUNDARY_WIDTH
    mask[..., 16 - w:16 + w] = 1
    mask[..., 48 - w:48 + w] = 1
    return mask


class Telemetry(clown.GuideTelemetry):
    def __init__(self) -> None:
        super().__init__()
        self.raw_x0: list[torch.Tensor] = []
        self.effective_x0: list[torch.Tensor] = []
        self.derivatives: list[torch.Tensor] = []
        self.prediction_records: list[dict[str, Any]] = []


@contextmanager
def prediction_policy(telemetry: Telemetry, source: torch.Tensor, edit: torch.Tensor,
                      locked: torch.Tensor, authoritative: bool):
    original = LatentGuide.process_guides_substep

    def wrapped(self, *args, **kwargs):
        x_0, x_rows, eps_rows, data_rows = args[:4]
        row, step, sigma = int(args[5]), int(args[6]), args[8]
        s_rows = args[11]
        raw_derivative = eps_rows[row].detach().clone()
        raw_prediction = data_rows[row].detach().clone()
        result = original(self, *args, **kwargs)
        guided_derivative = result[0][row]
        sigma_row = s_rows[row].to(device=guided_derivative.device, dtype=guided_derivative.dtype)
        state = x_rows[row]
        source_on_device = source.to(device=state.device, dtype=state.dtype)
        edit_on_device = edit.to(device=state.device, dtype=state.dtype)
        locked_on_device = locked.to(device=state.device, dtype=state.dtype)
        exact_derivative = (state - source_on_device) / sigma_row
        if authoritative:
            guided_derivative.copy_(edit_on_device * raw_derivative + locked_on_device * exact_derivative)

        call_index = telemetry.calls_per_step.get(step, 0)
        telemetry.calls_per_step[step] = call_index + 1
        if call_index == 0:
            if telemetry.initial_state is None:
                telemetry.initial_state = x_0.detach().clone()
            effective_x0 = state - sigma_row * guided_derivative
            boundary = latent_boundary_mask(state).to(state)
            telemetry.raw_x0.append(raw_prediction.detach().cpu().clone())
            telemetry.effective_x0.append(effective_x0.detach().cpu().clone())
            telemetry.derivatives.append(guided_derivative.detach().cpu().clone())
            telemetry.prediction_records.append({
                "step": step,
                "sigma": float(sigma),
                "sigma_row": float(sigma_row),
                "locked_predicted_x0_error_rms": rms(locked_on_device * (effective_x0 - source_on_device)),
                "locked_derivative_error_rms": rms(locked_on_device * (guided_derivative - exact_derivative)),
                "boundary_predicted_x0_error_rms": rms(boundary * (effective_x0 - source_on_device)),
                "editable_raw_to_effective_prediction_rms": rms(edit_on_device * (effective_x0 - raw_prediction)),
            })
        return result

    LatentGuide.process_guides_substep = wrapped
    try:
        yield
    finally:
        LatentGuide.process_guides_substep = original


@contextmanager
def accepted_state_policy(telemetry: Telemetry, source: torch.Tensor, locked: torch.Tensor,
                          enabled: bool):
    original = clown.rk_sampler_beta.preview_callback

    def wrapped(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
                callback, extra_options, **kwargs):
        final = bool(kwargs.get("final", False))
        step_index = int(kwargs.get("step_sched", step))
        if telemetry.initial_state is None:
            raise RuntimeError("Initial CONST state was not captured")
        source_on_device = source.to(device=x.device, dtype=x.dtype)
        locked_on_device = locked.to(device=x.device, dtype=x.dtype)
        fixed_noise = telemetry.initial_state.to(device=x.device, dtype=x.dtype)
        target = source_on_device if final else source_state(source_on_device, fixed_noise, sigma_next.to(x))
        before = rms(locked_on_device * (x - target))
        if enabled:
            x.copy_((1 - locked_on_device) * x + locked_on_device * target)
        after = rms(locked_on_device * (x - target))
        record = {
            "step": "terminal" if final else step_index,
            "sigma": float(sigma),
            "sigma_next": float(sigma_next),
            "locked_state_error_before_rms": before,
            "locked_state_error_after_rms": after,
            "restored": enabled,
        }
        telemetry.restorations.append(record)
        if final:
            telemetry.terminal_state = x.detach().cpu().clone()
        else:
            telemetry.accepted_states.append(x.detach().cpu().clone())
        return original(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
                        callback, extra_options, **kwargs)

    clown.rk_sampler_beta.preview_callback = wrapped
    try:
        yield
    finally:
        clown.rk_sampler_beta.preview_callback = original


def execute(model, positive, negative, sigmas, latent, guide, edit, locked,
            restore: bool, authoritative: bool):
    telemetry = Telemetry()
    with prediction_policy(telemetry, latent["samples"], edit, locked, authoritative), \
            accepted_state_policy(telemetry, latent["samples"], locked, restore), \
            torch.inference_mode():
        result = clown.ClownsharKSampler_Beta.execute(
            model=model, positive=positive, negative=negative,
            latent_image={"samples": latent["samples"].clone()}, sigmas=sigmas.clone(),
            guides=guide, eta=0.5, sampler_name="linear/euler", scheduler="simple",
            steps=clown.STEPS, steps_to_run=clown.STEPS, denoise=1.0, cfg=1.0,
            seed=clown.SEED, sampler_mode="standard", bongmath=True,
        )
    return result[0]["samples"].detach().cpu(), telemetry


def low_frequency_strip(decoded: torch.Tensor, source: torch.Tensor) -> dict[str, float]:
    value = decoded.detach().float().permute(0, 3, 1, 2)
    target = source.detach().float().permute(0, 3, 1, 2)
    value = F.avg_pool2d(value, LF_KERNEL, stride=1, padding=LF_KERNEL // 2)
    target = F.avg_pool2d(target, LF_KERNEL, stride=1, padding=LF_KERNEL // 2)
    result = {}
    for name, boundary in (("left", clown.LEFT_BOUNDARY), ("right", clown.RIGHT_BOUNDARY)):
        delta = value[..., boundary - PIXEL_BOUNDARY_WIDTH:boundary + PIXEL_BOUNDARY_WIDTH] - \
                target[..., boundary - PIXEL_BOUNDARY_WIDTH:boundary + PIXEL_BOUNDARY_WIDTH]
        result[f"{name}_decoded_boundary_lf_rms"] = rms(delta)
        result[f"{name}_decoded_boundary_lf_max_abs"] = float(delta.abs().max())
    return result


def prediction_difference(left: Telemetry, right: Telemetry, edit: torch.Tensor) -> list[dict[str, float]]:
    records = []
    edit_cpu = edit.cpu()
    for step, (a, c) in enumerate(zip(left.effective_x0, right.effective_x0)):
        delta = edit_cpu * (c.float() - a.float())
        records.append({"step": step, "editable_prediction_A_to_C_rms": rms(delta),
                        "editable_prediction_A_to_C_max_abs": float(delta.abs().max())})
    return records


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    for case, config in CASES.items():
        pixels = clown.load_pixels(Path(config["source"]))
        with torch.no_grad():
            encoded = vae.encode(pixels)
            reconstruction = vae.decode(encoded).detach().cpu()
        edit, locked = latent_masks(encoded)
        prepared[case] = {
            "pixels": pixels, "latent": {"samples": encoded.detach().clone()},
            "source_decode": reconstruction, "edit": edit, "locked": locked,
        }
        clown.metrics_base.save_pixels(pixels, OUTPUT / f"{case}_SOURCE.png")
        clown.metrics_base.save_pixels(reconstruction, OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png")
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
    for case, config in CASES.items():
        item = prepared[case]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        mask = torch.ones((1, clown.HEIGHT, clown.WIDTH), dtype=torch.float32)
        mask[:, :, clown.LEFT_BOUNDARY:clown.RIGHT_BOUNDARY] = 0
        guide = clown.make_guides(item["latent"], mask, True)
        outputs[case] = {}
        for arm, restore, authoritative in (
            ("A_EPSILON_BASELINE", False, False),
            ("B_STATE_RESTORATION", True, False),
            ("C_PREDICTION_AND_STATE", True, True),
        ):
            sampled, telemetry = execute(
                model, positive, negative, sigmas, item["latent"], guide,
                item["edit"], item["locked"], restore, authoritative,
            )
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
    for case, arms in outputs.items():
        item = prepared[case]
        report_arms = {}
        sheet = [("SOURCE", OUTPUT / f"{case}_SOURCE.png")]
        for arm, values in arms.items():
            telemetry = values["telemetry"]
            final = vae.decode(values["sampled"]).detach().cpu()
            final_path = OUTPUT / f"{case}_{arm}_FINAL.png"
            clown.metrics_base.save_pixels(final, final_path)
            sheet.append((arm, final_path))
            timestep = []
            for step, (raw_x0, effective_x0) in enumerate(zip(telemetry.raw_x0, telemetry.effective_x0)):
                raw_decode = vae.decode(raw_x0).detach().cpu()
                effective_decode = vae.decode(effective_x0).detach().cpu()
                clown.metrics_base.save_pixels(raw_decode, OUTPUT / f"{case}_{arm}_STEP_{step:02d}_RAW_X0.png")
                clown.metrics_base.save_pixels(effective_decode, OUTPUT / f"{case}_{arm}_STEP_{step:02d}_EFFECTIVE_X0.png")
                record = dict(telemetry.prediction_records[step])
                record.update(low_frequency_strip(effective_decode, item["source_decode"]))
                timestep.append(record)
            report_arms[arm] = {
                "timesteps": timestep,
                "acceptances": telemetry.restorations,
                "final_decoded": clown.boundary_metrics(final, item["source_decode"]),
                "terminal_locked_latent_error_rms": rms(item["locked"].cpu() * (
                    values["sampled"].float() - item["latent"]["samples"].cpu().float()
                )),
            }
        report_arms["C_PREDICTION_AND_STATE"]["editable_prediction_difference_vs_A"] = \
            prediction_difference(arms["A_EPSILON_BASELINE"]["telemetry"],
                                  arms["C_PREDICTION_AND_STATE"]["telemetry"], item["edit"])
        clown.metrics_base.make_sheet(sheet, OUTPUT / f"{case}_FINAL_COMPARISON.png")
        report_cases[case] = {"prompt": CASES[case]["prompt"], "arms": report_arms}
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    report = {
        "experiment": "Prediction consistency versus accepted-state ownership",
        "ordering": ["accepted x_sigma", "Klein prediction and CFG", "CONST derivative",
                     "RES4LYF epsilon_projection (A/B only on locked mask)",
                     "C locked derivative replacement", "Euler proposal",
                     "B/C accepted source-state restoration"],
        "configuration": {
            "model": str(clown.model_paths.MODEL_PATH), "seed": clown.SEED,
            "steps": clown.STEPS, "sigmas": [float(value) for value in sigmas],
            "sampler": "RES4LYF linear/euler, deterministic CONST", "cfg": 1.0,
            "noise_mask": None, "mask": "M=1 editable side latents; L=1 center source latents",
            "source_trajectory": "(1-sigma)*y + sigma*z_L; fixed z_L from initial accepted state",
        },
        "cases": report_cases,
        "runtime_seconds": time.perf_counter() - started,
        "diffusion_executed": True,
        "production_changed": False,
    }
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"), "runtime_seconds": report["runtime_seconds"],
        "final": {case: {arm: data["final_decoded"] for arm, data in item["arms"].items()}
                  for case, item in report_cases.items()},
    }, indent=2))


if __name__ == "__main__":
    run()
