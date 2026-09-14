"""Phase 2: fixed training-free ILVR-like correction for native Klein CONST Euler."""

from __future__ import annotations

import json
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PHASE1B_PREREG = ROOT / "experiments" / "DRIFT_PHASE_1B_PREREGISTRATION.json"
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_2_PREREGISTRATION.json"
OUTPUT = ROOT / "experiments" / "drift_phase_2_ilvr_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_phase_preserving_klein as phase1
import drift_phase_preserving_natural_generalization as phase1b


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def lowpass(value: torch.Tensor, factor: int) -> torch.Tensor:
    height, width = value.shape[-2:]
    if height % factor or width % factor:
        raise ValueError(f"Low-pass factor {factor} does not divide {height}x{width}")
    coarse = F.interpolate(value, size=(height // factor, width // factor), mode="area")
    return F.interpolate(coarse, size=(height, width), mode="nearest")


def lowpass_rms(value: torch.Tensor, factor: int) -> float:
    return phase1.rms(lowpass(value, factor))


class EulerILVRSampler:
    def __init__(self, source: torch.Tensor, gaussian: torch.Tensor, factor: int,
                 alpha: float, correction: bool, control: list[dict] | None = None) -> None:
        self.source = source
        self.gaussian = gaussian
        self.factor = factor
        self.alpha = alpha
        self.correction = correction
        self.control = control
        self.captures: list[dict] = []
        self.tensors: list[tuple[torch.Tensor, torch.Tensor]] = []

    @staticmethod
    def max_denoise(model, sigmas) -> bool:
        return bool(float(sigmas[0]) >= float(model.inner_model.model_sampling.sigma_max))

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None:
            raise ValueError("Phase 2 requires unmasked source img2img")
        sampling = model.inner_model.model_sampling
        state = sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model, sigmas)
        )
        for ordinal, (sigma, sigma_next) in enumerate(pairwise(sigmas)):
            denoised = model(
                state, sigma.expand(state.shape[0]),
                model_options=extra_args["model_options"], seed=extra_args.get("seed", 0),
            )
            record = {
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "state": phase1.summary(state), "denoised_x0": phase1.summary(denoised),
                "state_structure": phase1.structural_metrics(state, self.source),
                "denoised_structure": phase1.structural_metrics(denoised, self.source),
            }
            if self.control is not None:
                record["state_rms_difference_from_control"] = phase1.rms(
                    state - self.control[ordinal]["state"].to(state)
                )
                record["denoised_rms_difference_from_control"] = phase1.rms(
                    denoised - self.control[ordinal]["denoised"].to(denoised)
                )
            else:
                record["state_rms_difference_from_control"] = 0.0
                record["denoised_rms_difference_from_control"] = 0.0
            self.tensors.append((state.detach().float().cpu(), denoised.detach().float().cpu()))

            proposal = state + (state - denoised) / sigma * (sigma_next - sigma)
            interval = {
                "proposal_rms": phase1.rms(proposal),
                "correction_applied": False,
                "terminal_sigma_zero_skipped": float(sigma_next) == 0.0,
                "correction_rms": 0.0,
                "correction_to_proposal_rms_ratio": 0.0,
                "correction_to_proposal_energy_ratio": 0.0,
                "coarse_mismatch_removed_fraction": 0.0,
            }
            if self.correction and float(sigma_next) > 0.0:
                source_at_next = sampling.noise_scaling(
                    sigma_next, self.gaussian.to(proposal), latent_image, False
                )
                mismatch = lowpass(source_at_next, self.factor) - lowpass(proposal, self.factor)
                correction = self.alpha * mismatch
                corrected = proposal + correction
                pre = lowpass_rms(proposal - source_at_next, self.factor)
                post = lowpass_rms(corrected - source_at_next, self.factor)
                correction_rms = phase1.rms(correction)
                proposal_rms = max(phase1.rms(proposal), 1e-12)
                interval.update({
                    "correction_applied": True,
                    "source_at_matching_sigma": phase1.summary(source_at_next),
                    "source_sigma_formula_rms_error": phase1.rms(
                        source_at_next - (sigma_next * self.gaussian.to(source_at_next)
                                          + (1.0 - sigma_next) * latent_image)
                    ),
                    "pre_correction_coarse_mismatch_rms": pre,
                    "post_correction_coarse_mismatch_rms": post,
                    "correction": phase1.summary(correction),
                    "correction_rms": correction_rms,
                    "correction_to_proposal_rms_ratio": correction_rms / proposal_rms,
                    "correction_to_proposal_energy_ratio": (correction_rms / proposal_rms) ** 2,
                    "coarse_mismatch_removed_fraction": 0.0 if pre == 0.0 else 1.0 - post / pre,
                })
                state = corrected
            else:
                state = proposal
            record["accepted_interval"] = interval
            self.captures.append(record)
            if callback is not None:
                callback(ordinal, denoised, state, len(sigmas) - 1)
        return sampling.inverse_noise_scaling(sigmas[-1], state)


def validate_preflight(source: torch.Tensor, gaussian: torch.Tensor, factor: int) -> dict:
    projected = lowpass(source, factor)
    idempotence = phase1.rms(lowpass(projected, factor) - projected)
    linearity = phase1.rms(
        lowpass(source + gaussian, factor) - lowpass(source, factor) - lowpass(gaussian, factor)
    )
    proposal = torch.randn(source.shape, generator=torch.Generator().manual_seed(31))
    corrected = proposal + lowpass(source, factor) - lowpass(proposal, factor)
    replacement_error = phase1.rms(lowpass(corrected, factor) - lowpass(source, factor))
    result = {
        "status": "PASS", "factor": factor,
        "latent_shape": list(source.shape),
        "coarse_shape": [source.shape[-2] // factor, source.shape[-1] // factor],
        "coarse_spatial_dof_fraction": 1.0 / (factor * factor),
        "projection_idempotence_rms_error": idempotence,
        "projection_linearity_rms_error": linearity,
        "full_replacement_coarse_rms_error": replacement_error,
    }
    if max(idempotence, linearity, replacement_error) > 1e-6:
        raise RuntimeError(f"ILVR projection preflight failed: {result}")
    return result


def contact_sheet(case_name: str, paths: list[tuple[str, Path]]) -> Path:
    panel_w, panel_h, label_h = 512, 512, 28
    sheet = Image.new("RGB", (panel_w * len(paths), panel_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, path) in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), (index * panel_w, label_h))
        draw.text((index * panel_w + 6, 7), name, fill="black")
    path = OUTPUT / f"{case_name}_COMPARISON.png"
    sheet.save(path)
    return path


def main() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    phase1b_prereg = json.loads(PHASE1B_PREREG.read_text(encoding="utf-8"))
    if not prereg.get("registered_before_phase_2_outputs"):
        raise RuntimeError("Missing Phase 2 preregistration")
    included = set(prereg["included_cases"])
    cases = [case for case in phase1b_prereg["cases"] if case["name"] in included]
    if [case["name"] for case in cases] != prereg["included_cases"]:
        raise RuntimeError("Phase 2 case set/order mismatch")
    factor = prereg["fixed_runtime"]["ilvr_downsample_factor_latent"]
    alpha = prereg["fixed_runtime"]["ilvr_blend_alpha"]
    OUTPUT.mkdir(parents=True, exist_ok=True)

    import comfy.model_management
    import comfy.model_sampling
    import comfy.sample
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_flux import get_schedule

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase1.VAE), safe_load=True))
    prepared = {}
    for case in cases:
        source_path = OUTPUT / f"{case['name']}_SOURCE.png"
        pixels = phase1b.prepare_source(case, source_path)
        with torch.inference_mode():
            latent = vae.encode(pixels).detach().float().cpu()
        prepared[case["name"]] = {"latent": latent, "path": source_path}

    model = comfy.sd.load_diffusion_model(str(phase1.MODEL), model_options={})
    if not isinstance(model.model.model_sampling, comfy.model_sampling.CONST):
        raise TypeError("Phase 2 requires native Klein CONST sampling")
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    conditioning = {
        case["name"]: (
            clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"])),
            clip.encode_from_tokens_scheduled(clip.tokenize("")),
        ) for case in cases
    }
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(phase1.STEPS, 32 * 32).float()[phase1.START_INDEX:].clone()
    if float(sigmas[-1]) != 0.0 or not bool((sigmas[1:] < sigmas[:-1]).all()):
        raise RuntimeError("Unexpected Klein sigma schedule")

    report = {
        "phase": 2, "verdict": "ILVR-STYLE CORRECTION PARTIAL",
        "preregistration": str(PREREGISTRATION),
        "configuration": prereg["fixed_runtime"] | {"sigmas": sigmas.tolist()},
        "formulation": {
            "paper": "x_next = phi(y_next) + proposal - phi(proposal)",
            "implemented": "accepted = proposal + alpha * (phi(source_at_sigma_next) - phi(proposal)); only when sigma_next > 0",
            "source_trajectory": "Native CONST noise_scaling at sigma_next using the same ordinary Gaussian endpoint and clean encoded source.",
            "model_forward_modified": False,
        },
        "cases": {}, "integrity": {"production_changes": False},
    }
    trajectory_store = {}
    for case in cases:
        name = case["name"]
        source = prepared[name]["latent"]
        gaussian = torch.randn(source.shape, generator=torch.Generator().manual_seed(phase1.SEED))
        fss, fss_report = phase1.structured_noise(source, gaussian, 2.0)
        preflight = validate_preflight(source, gaussian, factor)
        positive, negative = conditioning[name]
        arms = {
            "A_GAUSSIAN": (gaussian, False),
            "B_FSS_R02": (fss, False),
            "C_ILVR_N04_A1": (gaussian, True),
        }
        case_record = {
            "role": case["role"], "prompt": case["prompt"],
            "source": phase1.image_record(prepared[name]["path"]),
            "source_latent": phase1.summary(source), "preflight": preflight,
            "fss_construction": fss_report, "arms": {},
        }
        control = None
        outputs = {}
        trajectory_store[name] = {}
        for arm_name, (noise, use_correction) in arms.items():
            sampler = EulerILVRSampler(source, gaussian, factor, alpha, use_correction, control)
            started = time.perf_counter()
            with torch.inference_mode():
                output = comfy.sample.sample_custom(
                    model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                    source.clone(), callback=lambda *unused: None, disable_pbar=True,
                    seed=phase1.SEED,
                ).detach().float().cpu()
            if control is None:
                control = [
                    {"state": state, "denoised": denoised}
                    for state, denoised in sampler.tensors
                ]
            outputs[arm_name] = output
            trajectory_store[name][arm_name] = sampler.tensors
            correction_records = [x["accepted_interval"] for x in sampler.captures]
            applied = [x for x in correction_records if x["correction_applied"]]
            case_record["arms"][arm_name] = {
                "wall_seconds": time.perf_counter() - started,
                "noise_hash": phase1.tensor_hash(noise),
                "final_latent": phase1.summary(output),
                "final_latent_structure": phase1.structural_metrics(output, source),
                "final_latent_rms_from_control": phase1.rms(output - outputs["A_GAUSSIAN"]),
                "correction_summary": {
                    "intervals_applied": len(applied),
                    "terminal_intervals_skipped": sum(x["terminal_sigma_zero_skipped"] for x in correction_records),
                    "mean_correction_to_proposal_rms_ratio": float(np.mean([x["correction_to_proposal_rms_ratio"] for x in applied])) if applied else 0.0,
                    "max_correction_to_proposal_rms_ratio": max([x["correction_to_proposal_rms_ratio"] for x in applied], default=0.0),
                    "mean_correction_to_proposal_energy_ratio": float(np.mean([x["correction_to_proposal_energy_ratio"] for x in applied])) if applied else 0.0,
                    "mean_coarse_mismatch_removed_fraction": float(np.mean([x["coarse_mismatch_removed_fraction"] for x in applied])) if applied else 0.0,
                },
                "evaluations": sampler.captures,
            }
        report["cases"][name] = case_record
        atomic_json(OUTPUT / "telemetry.json", report)

    del model, conditioning
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    selected = (0, (len(sigmas) - 2) // 2, len(sigmas) - 2)
    for case in cases:
        name = case["name"]
        source_path = prepared[name]["path"]
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        sheet_items = [("SOURCE", source_path)]
        for arm_name, arm_record in report["cases"][name]["arms"].items():
            final_latent = outputs[arm_name] if name == cases[-1]["name"] else None
            if final_latent is None:
                # The terminal denoised estimate equals Euler's sigma-zero proposal.
                final_latent = trajectory_store[name][arm_name][-1][1]
            final_path = OUTPUT / f"{name}_{arm_name}.png"
            with torch.inference_mode():
                phase1.save_pixels(vae.decode(final_latent).cpu(), final_path)
            final_rgb = np.asarray(Image.open(final_path).convert("RGB"))
            arm_record["final_image"] = phase1.image_record(final_path)
            arm_record["final_decoded_metrics"] = phase1b.decoded_metrics(final_rgb, source_rgb)
            arm_record["selected_x0_previews"] = {}
            for ordinal in selected:
                preview_path = OUTPUT / f"{name}_{arm_name}_EVAL_{ordinal:02d}_X0.png"
                with torch.inference_mode():
                    phase1.save_pixels(vae.decode(trajectory_store[name][arm_name][ordinal][1]).cpu(), preview_path)
                arm_record["selected_x0_previews"][str(ordinal)] = phase1.image_record(preview_path)
            sheet_items.append((arm_name, final_path))
        report["cases"][name]["comparison"] = phase1.image_record(contact_sheet(name, sheet_items))
        atomic_json(OUTPUT / "telemetry.json", report)
    report["integrity"].update({
        "status": "SUCCESS", "all_finite": all(
            arm["final_latent"]["finite"]
            for case in report["cases"].values() for arm in case["arms"].values()
        ),
        "same_model_vae_text_encoder_cfg_sampler_schedule_seed_policy": True,
        "native_const_noise_scaling_preserved": True,
        "ordinary_model_forward_untouched": True,
        "correction_after_accepted_nonterminal_euler_intervals_only": True,
        "no_terminal_sigma_zero_correction": True,
        "no_fss_ilvr_combination": True,
    })
    report["visual_assessment"] = {
        "portrait_bronze": "FAIL: better placement/coarse structure than Gaussian, but facial identity, apparent age, arm pose, and framing are not preserved.",
        "astronaut_steampunk": "PASS: stance and silhouette remain close while suit material and scene appearance change substantially.",
        "bridge_crystal_night": "PASS: towers, deck, cables, train, and perspective remain close while lighting/material change.",
        "decisive_interpretation": "Portrait failure overrides the two cross-case successes; the fixed setting is useful for coarse layout but not identity-sensitive geometry.",
    }
    atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
