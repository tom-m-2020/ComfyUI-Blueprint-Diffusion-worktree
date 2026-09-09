"""Late accepted-state source restoration starting from reproduced Clown arm D."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import local_edit_user_clown_epsilon_reproduction as reference


OUTPUT = reference.ROOT / "experiments" / "local_edit_late_source_restoration_results"
FROZEN = reference.OUTPUT / "runtime_tensors.pt"


def latent_masks(direct_mask: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    edit = F.interpolate(
        direct_mask.unsqueeze(1).float(), size=latent.shape[-2:], mode="nearest-exact"
    ).to(dtype=latent.dtype)
    source = 1 - edit
    return edit, source


def latent_metrics(value: torch.Tensor, source: torch.Tensor, edit_mask: torch.Tensor,
                   source_mask: torch.Tensor) -> dict[str, float]:
    delta = value.float() - source.float()
    return {
        "source_latent_rms": reference.rms(source_mask * delta),
        "source_latent_max_abs": float((source_mask * delta).abs().max()),
        "editable_change_rms": reference.rms(edit_mask * delta),
    }


def first_divergence(candidate: list[torch.Tensor], baseline: list[torch.Tensor]) -> dict:
    records = []
    first = None
    for step, (left, right) in enumerate(zip(candidate[:reference.STEPS], baseline[:reference.STEPS])):
        delta = left.float() - right.float()
        maximum = float(delta.abs().max())
        records.append({"step": step, "rms": reference.rms(delta), "max_abs": maximum})
        if first is None and maximum != 0:
            first = step
    return {"first_step": first, "per_step": records}


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prepared = reference.prepare_inputs()
    reference.metrics_base.save_pixels(prepared["resized"], OUTPUT / "PADDED_RESIZED.png")

    vae = reference.comfy.sd.VAE(sd=reference.comfy.utils.load_torch_file(
        str(reference.model_paths.VAE_PATH), safe_load=True
    ))
    with torch.no_grad():
        encoded = vae.encode(prepared["resized"])
    latent = {"samples": encoded.detach().clone()}
    edit_mask, source_mask = latent_masks(prepared["direct_mask"], latent["samples"])
    source_reconstruction = vae.decode(latent["samples"]).detach().cpu()
    reference.metrics_base.save_pixels(source_reconstruction, OUTPUT / "SOURCE_RECONSTRUCTION.png")
    del vae
    reference.comfy.model_management.unload_all_models()
    reference.comfy.model_management.soft_empty_cache()

    model = reference.comfy.sd.load_diffusion_model(str(reference.model_paths.MODEL_PATH), model_options={})
    clip = reference.comfy.sd.load_clip(
        [str(reference.model_paths.TEXT_ENCODER_PATH)], clip_type=reference.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(reference.PROMPT))
    negative = reference.nodes.ConditioningZeroOut().zero_out(positive)[0]
    del clip
    reference.comfy.model_management.unload_all_models()
    reference.comfy.model_management.soft_empty_cache()
    sigmas = reference.get_schedule(
        reference.STEPS, round(reference.WIDTH * reference.HEIGHT / (16 * 16))
    ).float()
    guides = reference.make_guides(latent, prepared["direct_mask"], True)

    arm_steps = {
        "A_frozen_D_reference": set(),
        "B_final_accepted_only": {reference.STEPS - 1},
        "C_final_two_accepted": {reference.STEPS - 2, reference.STEPS - 1},
    }
    outputs = {}
    started = time.perf_counter()
    for name, steps in arm_steps.items():
        sampled, denoised, telemetry = reference.execute_arm(
            model, positive, negative, sigmas, latent, guides, None,
            source_mask=source_mask, restore_steps=steps,
        )
        outputs[name] = {"sampled": sampled, "denoised": denoised, "telemetry": telemetry}
    model.cleanup()
    del model
    reference.comfy.model_management.unload_all_models()
    reference.comfy.model_management.soft_empty_cache()
    gc.collect()

    frozen = torch.load(FROZEN, map_location="cpu", weights_only=True)["D_noise_mask_disabled"]
    frozen_delta = outputs["A_frozen_D_reference"]["sampled"] - frozen["sampled"]

    vae = reference.comfy.sd.VAE(sd=reference.comfy.utils.load_torch_file(
        str(reference.model_paths.VAE_PATH), safe_load=True
    ))
    sheet = [("PADDED SOURCE", OUTPUT / "PADDED_RESIZED.png")]
    report_arms = {}
    baseline_states = outputs["A_frozen_D_reference"]["telemetry"].accepted_states
    for name, values in outputs.items():
        pixels = vae.decode(values["sampled"]).detach().cpu()
        reference.metrics_base.save_pixels(pixels, OUTPUT / f"{name}_FINAL.png")
        sheet.append((name, OUTPUT / f"{name}_FINAL.png"))
        for step, state in enumerate(values["telemetry"].accepted_states[:reference.STEPS]):
            preview = vae.decode(state).detach().cpu()
            reference.metrics_base.save_pixels(preview, OUTPUT / f"{name}_ACCEPTED_{step:02d}.png")
        report_arms[name] = {
            "latent": latent_metrics(values["sampled"], latent["samples"].cpu(), edit_mask.cpu(), source_mask.cpu()),
            "decoded": reference.boundary_metrics(pixels, source_reconstruction),
            "first_divergence_vs_A": first_divergence(
                values["telemetry"].accepted_states, baseline_states
            ),
            "restoration_records": values["telemetry"].restorations,
            "returned_equals_terminal_callback": bool(torch.equal(
                values["sampled"], values["telemetry"].terminal_state
            )),
        }
    del vae
    reference.comfy.model_management.unload_all_models()
    reference.comfy.model_management.soft_empty_cache()
    reference.metrics_base.make_sheet(sheet, OUTPUT / "COMPARISON.png")

    report = {
        "experiment": "Late accepted-state source restoration from reproduced arm D",
        "invariant": "Identical full-canvas epsilon_projection and pre-model states until an arm's declared post-Euler restoration step",
        "configuration": {
            "prompt": reference.PROMPT,
            "seed": reference.SEED,
            "steps": reference.STEPS,
            "sigmas": [float(v) for v in sigmas],
            "sampler": "installed RES4LYF linear/euler",
            "guide": "epsilon_projection, non-channelwise, weight=1, cutoff=1, constant",
            "native_noise_mask": None,
            "source_mask": "binary center source at latent columns 16:48; 1=source",
        },
        "frozen_D_integrity": {
            "sampled_bit_exact": bool(torch.equal(outputs["A_frozen_D_reference"]["sampled"], frozen["sampled"])),
            "sampled_rms": reference.rms(frozen_delta),
            "sampled_max_abs": float(frozen_delta.abs().max()),
        },
        "arms": report_arms,
        "runtime_seconds": time.perf_counter() - started,
    }
    reference.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        name: {"sampled": value["sampled"], "denoised": value["denoised"]}
        for name, value in outputs.items()
    })
    reference.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "frozen_D_integrity": report["frozen_D_integrity"],
        "arms": report["arms"],
    }, indent=2))


if __name__ == "__main__":
    run()
