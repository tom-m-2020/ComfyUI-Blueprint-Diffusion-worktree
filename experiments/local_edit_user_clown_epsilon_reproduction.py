"""Exact installed-RES4LYF reproduction of the uploaded Clown workflow."""

from __future__ import annotations

import asyncio
import gc
import json
import math
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
CUSTOM_ROOT = COMFY_ROOT / "custom_nodes"
sys.path[:0] = [str(COMFY_ROOT), str(CUSTOM_ROOT), str(ROOT / "experiments")]

import comfy.model_management  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import nodes  # noqa: E402
from comfy_extras.nodes_flux import get_schedule  # noqa: E402
from comfy_extras.nodes_post_processing import scale_dimensions  # noqa: E402
from server import PromptServer  # noqa: E402

if not hasattr(PromptServer, "instance"):
    PromptServer(asyncio.new_event_loop())

from RES4LYF.beta.rk_guide_func_beta import LatentGuide  # noqa: E402
from RES4LYF.beta import rk_sampler_beta  # noqa: E402
from RES4LYF.beta.samplers import ClownsharKSampler_Beta  # noqa: E402
from RES4LYF.beta.samplers_extensions import ClownGuide_Beta  # noqa: E402
from RES4LYF.latents import get_collinear, get_orthogonal  # noqa: E402

import flux2_coarse_global_local_falsification as model_paths  # noqa: E402
import local_edit_hard_source_klein as metrics_base  # noqa: E402


OUTPUT = ROOT / "experiments" / "local_edit_user_clown_epsilon_results"
SOURCE_PATH = Path(r"C:\Users\Tom-M\Downloads\ComfyUI_temp_jzdml_00001_.png")
REFERENCE_PATH = Path(r"C:\Users\Tom-M\Downloads\ComfyUI_temp_fpgkr_00001_.png")
PROMPT = (
    "A wide cinematic photograph of one single long red suspension bridge stretching "
    "continuously from the far left edge to the far right edge over calm water, one "
    "yellow passenger train centered on the bridge, one white lighthouse at the far "
    "left, one dark stone tower at the far right, continuous bridge deck and cables, "
    "coherent perspective, no duplicate bridges, trains, lighthouses, or towers"
)
SEED = 0
STEPS = 8
WIDTH = 1024
HEIGHT = 512
LEFT_BOUNDARY = 256
RIGHT_BOUNDARY = 768


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def load_pixels(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def blur_mask_like_kjnodes(mask: torch.Tensor, radius: float) -> torch.Tensor:
    output = []
    for item in mask.reshape((-1, mask.shape[-2], mask.shape[-1])):
        array = np.clip(item.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        blurred = Image.fromarray(array, mode="L").filter(ImageFilter.GaussianBlur(radius))
        output.append(torch.from_numpy(np.asarray(blurred, dtype=np.float32).copy() / 255.0))
    return torch.stack(output)


def prepare_inputs() -> dict[str, torch.Tensor]:
    source = load_pixels(SOURCE_PATH)
    padded, outpaint_mask = nodes.ImagePadForOutpaint().expand_image(
        source, left=512, top=0, right=512, bottom=0, feathering=0
    )
    resized = scale_dimensions(padded, WIDTH, HEIGHT, "lanczos", "disabled")
    direct_mask = scale_dimensions(outpaint_mask, WIDTH, HEIGHT, "nearest-exact", "disabled")
    blurred = blur_mask_like_kjnodes(outpaint_mask, 32)
    blurred_mask = scale_dimensions(blurred, WIDTH, HEIGHT, "nearest-exact", "disabled")
    return {
        "source": source,
        "padded": padded,
        "resized": resized,
        "direct_mask": direct_mask,
        "blurred_mask": blurred_mask,
    }


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().float().cpu()
    unique = torch.unique(value)
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "nonzero_fraction": float((value != 0).float().mean()),
        "one_fraction": float((value == 1).float().mean()),
        "unique_count": int(unique.numel()),
        "first_unique_values": [float(v) for v in unique[:16]],
    }


class GuideTelemetry:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.predictions: list[torch.Tensor] = []
        self.calls_per_step: dict[int, int] = {}
        self.initial_state: torch.Tensor | None = None
        self.restorations: list[dict[str, Any]] = []
        self.accepted_states: list[torch.Tensor] = []
        self.terminal_state: torch.Tensor | None = None


@contextmanager
def instrument_guidance(telemetry: GuideTelemetry):
    original = LatentGuide.process_guides_substep

    def wrapped(self, *args, **kwargs):
        x_0, x_rows, eps_rows, data_rows = args[:4]
        row, step, step_sched = args[5], args[6], args[7]
        sigma = args[8]
        sigma_down = args[10]
        s_rows = args[11]
        epsilon_scale = args[12]
        rk = args[13]
        before = eps_rows[row].detach().clone()
        data = data_rows[row].detach().clone()
        plain = before.clone()
        projection_candidate = before.clone()
        guide_derivative = torch.zeros_like(before)
        weight_mask = torch.zeros_like(before[:, :1])
        if self.HAS_LATENT_GUIDE:
            y0, _, weight_mask, _ = self.get_cossim_adjusted_lgw_masks(data_rows[row], step_sched)
            guide_derivative = rk.get_guide_epsilon(
                x_0, x_rows[row], y0, sigma, s_rows[row], sigma_down, epsilon_scale
            )
            plain = before + weight_mask * (guide_derivative - before)
            if self.guide_mode == "epsilon_projection":
                masked_guide = self.mask * guide_derivative
                projection_candidate = (
                    get_collinear(before, masked_guide)
                    + get_orthogonal(masked_guide, before)
                )
        result = original(self, *args, **kwargs)
        after = result[0][row].detach().clone()
        call_index = telemetry.calls_per_step.get(int(step), 0)
        telemetry.calls_per_step[int(step)] = call_index + 1
        if call_index != 0:
            return result
        if telemetry.initial_state is None:
            telemetry.initial_state = x_0.detach().clone()
        correction = after - before
        hard_source_correction = weight_mask * (guide_derivative - before)
        telemetry.records.append({
            "step": int(step),
            "step_sched": int(step_sched),
            "sigma": float(sigma),
            "mode": self.guide_mode,
            "model_derivative_rms": rms(before),
            "guide_derivative_rms": rms(guide_derivative),
            "guide_correction_rms": rms(correction),
            "correction_to_model_ratio": rms(correction) / max(rms(before), 1e-12),
            "plain_epsilon_correction_rms": rms(plain - before),
            "projection_candidate_correction_rms": rms(projection_candidate - before),
            "projection_effect_vs_plain_rms": rms(after - plain),
            "exact_correction_vs_hard_source_rms": rms(correction - hard_source_correction),
            "mask_mean": float(weight_mask.detach().float().mean()),
        })
        telemetry.predictions.append(data.cpu())
        return result

    LatentGuide.process_guides_substep = wrapped
    try:
        yield
    finally:
        LatentGuide.process_guides_substep = original


@contextmanager
def restore_accepted_source(
    telemetry: GuideTelemetry,
    source_latent: torch.Tensor,
    source_mask: torch.Tensor,
    restore_steps: set[int],
):
    """Experiment-only accepted-state hook; model-visible input is never changed pre-call."""
    original = rk_sampler_beta.preview_callback

    def wrapped(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
                callback, extra_options, **kwargs):
        step_index = int(kwargs.get("step_sched", step))
        final = bool(kwargs.get("final", False))
        before = x.detach().clone()
        restored = False
        target_error_before = None
        target_error_after = None
        if final and restore_steps:
            y = source_latent.to(device=x.device, dtype=x.dtype)
            mask = source_mask.to(device=x.device, dtype=x.dtype)
            target_error_before = rms(mask * (x - y))
            x.copy_((1 - mask) * x + mask * y)
            target_error_after = rms(mask * (x - y))
            restored = True
        elif step_index in restore_steps:
            if telemetry.initial_state is None:
                raise RuntimeError("Initial CONST state was not captured before restoration")
            sigma_next_value = sigma_next.to(device=x.device, dtype=x.dtype)
            y = source_latent.to(device=x.device, dtype=x.dtype)
            z = telemetry.initial_state.to(device=x.device, dtype=x.dtype)
            mask = source_mask.to(device=x.device, dtype=x.dtype)
            source_state = (1 - sigma_next_value) * y + sigma_next_value * z
            target_error_before = rms(mask * (x - source_state))
            x.copy_((1 - mask) * x + mask * source_state)
            target_error_after = rms(mask * (x - source_state))
            restored = True
        if final:
            telemetry.terminal_state = x.detach().cpu().clone()
        else:
            telemetry.accepted_states.append(x.detach().cpu().clone())
        telemetry.restorations.append({
            "step": "terminal" if final else step_index,
            "sigma": float(sigma),
            "sigma_next": float(sigma_next),
            "restored": restored,
            "accepted_change_rms": rms(x - before),
            "source_target_error_before_rms": target_error_before,
            "source_target_error_after_rms": target_error_after,
        })
        return original(
            x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
            callback, extra_options, **kwargs
        )

    rk_sampler_beta.preview_callback = wrapped
    try:
        yield
    finally:
        rk_sampler_beta.preview_callback = original


def make_guides(guide_latent: dict[str, torch.Tensor], direct_mask: torch.Tensor, projection: bool):
    return ClownGuide_Beta().main(
        guide={"samples": guide_latent["samples"]},
        mask=direct_mask,
        guide_mode="epsilon",
        channelwise_mode=False,
        projection_mode=projection,
        weight=1.0,
        cutoff=1.0,
        weight_scheduler="constant",
        start_step=0,
        end_step=-1,
        invert_mask=False,
    )[0]


def execute_arm(
    model, positive, negative, sigmas, latent, guide_config, noise_mask,
    source_mask=None, restore_steps=None,
):
    latent_input = {"samples": latent["samples"].clone()}
    if noise_mask is not None:
        latent_input["noise_mask"] = noise_mask.clone()
    telemetry = GuideTelemetry()
    restoration = (
        restore_accepted_source(telemetry, latent["samples"], source_mask, set(restore_steps))
        if source_mask is not None else nullcontext()
    )
    with instrument_guidance(telemetry), restoration, torch.inference_mode():
        result = ClownsharKSampler_Beta.execute(
            model=model,
            positive=positive,
            negative=negative,
            latent_image=latent_input,
            sigmas=sigmas.clone(),
            guides=guide_config,
            eta=0.5,
            sampler_name="linear/euler",
            scheduler="simple",
            steps=8,
            steps_to_run=8,
            denoise=1.0,
            cfg=1.0,
            seed=SEED,
            sampler_mode="standard",
            bongmath=True,
        )
    output = result[0]
    denoised = result[1]
    return output["samples"].detach().cpu(), denoised["samples"].detach().cpu(), telemetry


def boundary_metrics(image: torch.Tensor, source_reconstruction: torch.Tensor) -> dict[str, float]:
    chw = image.detach().float().permute(0, 3, 1, 2)
    source = source_reconstruction.detach().float().permute(0, 3, 1, 2)
    center = chw[..., LEFT_BOUNDARY:RIGHT_BOUNDARY]
    source_center = source[..., LEFT_BOUNDARY:RIGHT_BOUNDARY]
    editable = torch.cat((chw[..., :LEFT_BOUNDARY], chw[..., RIGHT_BOUNDARY:]), dim=-1)
    source_editable = torch.cat((source[..., :LEFT_BOUNDARY], source[..., RIGHT_BOUNDARY:]), dim=-1)
    result = {
        "locked_source_mae": float((center - source_center).abs().mean()),
        "locked_source_max_abs": float((center - source_center).abs().max()),
        "editable_change_rms": rms(editable - source_editable),
    }
    for name, boundary in (("left", LEFT_BOUNDARY), ("right", RIGHT_BOUNDARY)):
        result[f"{name}_seam_gradient_rms"] = rms(chw[..., boundary] - chw[..., boundary - 1])
        result[f"{name}_inside_adjacent_gradient_rms"] = rms(chw[..., boundary - 1] - chw[..., boundary - 2])
        result[f"{name}_outside_adjacent_gradient_rms"] = rms(chw[..., boundary + 1] - chw[..., boundary])
    return result


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = prepare_inputs()
    metrics_base.save_pixels(prepared["resized"], OUTPUT / "PADDED_RESIZED.png")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(model_paths.VAE_PATH), safe_load=True))
    with torch.no_grad():
        encoded = vae.encode(prepared["resized"])
    latent = {"samples": encoded.detach().clone()}
    source_reconstruction = vae.decode(latent["samples"]).detach().cpu()
    metrics_base.save_pixels(source_reconstruction, OUTPUT / "SOURCE_RECONSTRUCTION.png")
    del vae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()

    model = comfy.sd.load_diffusion_model(str(model_paths.MODEL_PATH), model_options={})
    clip = comfy.sd.load_clip([str(model_paths.TEXT_ENCODER_PATH)], clip_type=comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = nodes.ConditioningZeroOut().zero_out(positive)[0]
    del clip
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(STEPS, round(WIDTH * HEIGHT / (16 * 16))).float()

    projection_guides = make_guides(latent, prepared["direct_mask"], True)
    plain_guides = make_guides(latent, prepared["direct_mask"], False)
    arms = {
        "A_exact_reference": (projection_guides, prepared["blurred_mask"]),
        "B_no_projection": (plain_guides, prepared["blurred_mask"]),
        "C_guide_disabled": (None, prepared["blurred_mask"]),
        "D_noise_mask_disabled": (projection_guides, None),
        "E_unblurred_noise_mask": (projection_guides, prepared["direct_mask"]),
        "F_no_projection_no_noise_mask": (plain_guides, None),
        "G_no_guide_no_noise_mask": (None, None),
    }
    outputs: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for name, (guides, noise_mask) in arms.items():
        sampled, denoised, telemetry = execute_arm(
            model, positive, negative, sigmas, latent, guides, noise_mask
        )
        outputs[name] = {
            "sampled": sampled,
            "denoised": denoised,
            "telemetry": telemetry,
        }
    model.cleanup()
    del model
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(model_paths.VAE_PATH), safe_load=True))
    image_paths = [("PADDED SOURCE", OUTPUT / "PADDED_RESIZED.png")]
    report_arms = {}
    for name, values in outputs.items():
        sampled_pixels = vae.decode(values["sampled"]).detach().cpu()
        denoised_pixels = vae.decode(values["denoised"]).detach().cpu()
        metrics_base.save_pixels(sampled_pixels, OUTPUT / f"{name}_OUTPUT0.png")
        metrics_base.save_pixels(denoised_pixels, OUTPUT / f"{name}_OUTPUT1_DENOISED.png")
        image_paths.append((name, OUTPUT / f"{name}_OUTPUT1_DENOISED.png"))
        for index, prediction in enumerate(values["telemetry"].predictions):
            processed = prediction
            if processed.ndim == 4:
                processed = processed.float()
            preview = vae.decode(processed).detach().cpu()
            metrics_base.save_pixels(preview, OUTPUT / f"{name}_STEP_{index:02d}_PREDICTION.png")
        report_arms[name] = {
            "metrics_output1_denoised": boundary_metrics(denoised_pixels, source_reconstruction),
            "metrics_output0_sampled": boundary_metrics(sampled_pixels, source_reconstruction),
            "guide_telemetry": values["telemetry"].records,
            "output_slots_pixel_rms": rms(sampled_pixels - denoised_pixels),
        }
    del vae
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    metrics_base.make_sheet(image_paths, OUTPUT / "COMPARISON.png")

    reference = load_pixels(REFERENCE_PATH)
    reference_errors = {}
    for name in outputs:
        reproduced = load_pixels(OUTPUT / f"{name}_OUTPUT1_DENOISED.png")
        reference_errors[name] = {
            "decoded_png_bit_exact": bool(torch.equal(reference, reproduced)),
            "decoded_png_mae": float((reference - reproduced).abs().mean()),
            "decoded_png_max_abs": float((reference - reproduced).abs().max()),
        }
    report = {
        "experiment": "Uploaded Clown epsilon_projection workflow reproduction",
        "workflow": "C:/Users/Tom-M/Downloads/local_edit_test.json",
        "source": str(SOURCE_PATH),
        "uploaded_reference": str(REFERENCE_PATH),
        "configuration": {
            "model": str(model_paths.MODEL_PATH),
            "text_encoder": str(model_paths.TEXT_ENCODER_PATH),
            "vae": str(model_paths.VAE_PATH),
            "prompt": PROMPT,
            "seed": SEED,
            "steps": STEPS,
            "sigmas": [float(v) for v in sigmas],
            "sampler": "installed RES4LYF linear/euler",
            "cfg": 1.0,
            "decoded_workflow_output_slot": 1,
        },
        "mask_provenance": {
            "outpaint_2048x1024": tensor_stats(nodes.ImagePadForOutpaint().expand_image(
                prepared["source"], 512, 0, 512, 0, 0
            )[1]),
            "guide_direct_1024x512_before_clown_inversion": tensor_stats(prepared["direct_mask"]),
            "guide_internal_1024x512_after_clown_inversion": tensor_stats(1 - prepared["direct_mask"]),
            "noise_blurred_1024x512": tensor_stats(prepared["blurred_mask"]),
        },
        "serialized_graph_finding": "SetLatentNoiseMask node 1069 has mode=4 (bypass); D, not user-declared A, is the actual serialized reference.",
        "reference_equivalence_by_arm": reference_errors,
        "arms": report_arms,
        "runtime_seconds": time.perf_counter() - started,
        "semantic_review": {
            "reference": "D projection-only reproduces the uploaded PNG within one 8-bit code value and preserves one continuous bridge with small seams.",
            "noise_mask": "Enabling either blurred or binary native noise masking materially changes composition and creates larger boundary discontinuities; it is not active in the uploaded serialized graph.",
            "projection": "Projection is decisive in the actual no-mask state: compare D against F and G. Under native masking, A/B/C are much closer and all show boundary breaks.",
            "plain_epsilon": "The non-projection correction equals the masked hard-source derivative correction at the primary Euler row; projection adds the measured batch-global orthogonal component.",
        },
        "decision": "Treat uploaded epsilon_projection with bypassed noise mask as positive counterevidence. Narrow the prior decision to the two-column scalar transition only; do not generalize it to Clown epsilon/projection.",
    }
    metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        name: {"sampled": value["sampled"], "denoised": value["denoised"]}
        for name, value in outputs.items()
    })
    metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "reference_equivalence": report["reference_equivalence_by_arm"],
    }, indent=2))


if __name__ == "__main__":
    run()
