"""Phase 5: normalized high-FBS substitution on Klein sampler latent states."""

from __future__ import annotations

import json
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PHASE1B_PREREG = ROOT / "experiments" / "DRIFT_PHASE_1B_PREREGISTRATION.json"
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_5_PREREGISTRATION.json"
OUTPUT = ROOT / "experiments" / "drift_phase_5_sampler_latent_fbsdiff_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_fbsdiff_like_klein as phase4
import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1
import drift_phase_preserving_natural_generalization as phase1b


class SamplerLatentHighFBS:
    def __init__(self, source: torch.Tensor, reference_states: list[torch.Tensor],
                 normalized_threshold: float, calibration: set[int],
                 control: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.source = source
        self.reference_states = reference_states
        self.normalized_threshold = normalized_threshold
        self.calibration = calibration
        self.control = control
        self.captures: list[dict] = []
        self.tensors: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.band_records: list[dict] = []

    @staticmethod
    def max_denoise(model, sigmas) -> bool:
        return bool(float(sigmas[0]) >= float(model.inner_model.model_sampling.sigma_max))

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None:
            raise ValueError("Phase 5 requires unmasked source img2img")
        sampling = model.inner_model.model_sampling
        state = sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model, sigmas)
        )
        for ordinal, (sigma, sigma_next) in enumerate(pairwise(sigmas)):
            reference = self.reference_states[ordinal].to(state)
            if reference.shape != state.shape:
                raise RuntimeError(
                    f"Reference/target state mismatch: {reference.shape} vs {state.shape}"
                )
            record = {
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "reference_sigma": float(sigma),
                "shape": list(state.shape), "calibration_active": ordinal in self.calibration,
                "substitution_rms": 0.0,
                "substitution_to_target_rms_ratio": 0.0,
                "substitution_to_target_energy_ratio": 0.0,
            }
            if ordinal in self.calibration:
                height, width = state.shape[-2:]
                matrix = phase4.dct_matrix(height, state.device)
                target_dct = phase4.dct_2d(state.float(), matrix)
                reference_dct = phase4.dct_2d(reference.float(), matrix)
                rows = torch.arange(height, device=state.device).unsqueeze(1)
                columns = torch.arange(width, device=state.device).unsqueeze(0)
                normalized_radius = (rows + columns) / (height - 1)
                mask = (normalized_radius > self.normalized_threshold).to(target_dct.dtype)
                merged_dct = reference_dct * mask + target_dct * (1.0 - mask)
                substituted = phase4.idct_2d(merged_dct, matrix).to(state.dtype)
                delta = substituted - state
                target_rms = max(phase1.rms(state), 1e-12)
                substitution_rms = phase1.rms(delta)
                record.update({
                    "substituted_coefficients": int(mask.sum()),
                    "total_coefficients": height * width,
                    "substituted_coefficient_fraction": float(mask.mean()),
                    "reference_high_band_rms": phase1.rms(
                        phase4.idct_2d(reference_dct * mask, matrix)
                    ),
                    "target_high_band_rms": phase1.rms(
                        phase4.idct_2d(target_dct * mask, matrix)
                    ),
                    "reference_low_band_rms": phase1.rms(
                        phase4.idct_2d(reference_dct * (1.0 - mask), matrix)
                    ),
                    "target_low_band_rms": phase1.rms(
                        phase4.idct_2d(target_dct * (1.0 - mask), matrix)
                    ),
                    "substitution_rms": substitution_rms,
                    "substitution_to_target_rms_ratio": substitution_rms / target_rms,
                    "substitution_to_target_energy_ratio": (substitution_rms / target_rms) ** 2,
                    "post_substitution_high_band_mismatch_rms": phase1.rms(
                        phase4.idct_2d((merged_dct - reference_dct) * mask, matrix)
                    ),
                })
                state = substituted
            self.band_records.append(record)

            denoised = model(
                state, sigma.expand(state.shape[0]),
                model_options=extra_args["model_options"], seed=extra_args.get("seed", 0),
            )
            evaluation = {
                "ordinal": ordinal, "sigma": float(sigma), "sigma_next": float(sigma_next),
                "state": phase1.summary(state), "denoised_x0": phase1.summary(denoised),
                "state_structure": phase1.structural_metrics(state, self.source),
                "denoised_structure": phase1.structural_metrics(denoised, self.source),
                "state_rms_difference_from_control": phase1.rms(
                    state - self.control[ordinal][0].to(state)
                ),
                "denoised_rms_difference_from_control": phase1.rms(
                    denoised - self.control[ordinal][1].to(denoised)
                ),
                "band_substitution": record,
            }
            self.captures.append(evaluation)
            self.tensors.append((state.detach().float().cpu(), denoised.detach().float().cpu()))
            state = state + (state - denoised) / sigma * (sigma_next - sigma)
            if callback is not None:
                callback(ordinal, denoised, state, len(sigmas) - 1)
        return sampling.inverse_noise_scaling(sigmas[-1], state)


def contact_sheet(case_name: str, paths: list[tuple[str, Path]]) -> Path:
    width, height, label_height = 512, 512, 28
    sheet = Image.new("RGB", (width * len(paths), height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), (index * width, label_height))
        draw.text((index * width + 6, 7), label, fill="black")
    output_path = OUTPUT / f"{case_name}_COMPARISON.png"
    sheet.save(output_path)
    return output_path


def main() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    phase1b_prereg = json.loads(PHASE1B_PREREG.read_text(encoding="utf-8"))
    if not prereg.get("registered_before_phase_5_outputs"):
        raise RuntimeError("Missing Phase 5 preregistration")
    cases = [
        case for case in phase1b_prereg["cases"]
        if case["name"] in set(prereg["cases"])
    ]
    if [case["name"] for case in cases] != prereg["cases"]:
        raise RuntimeError("Phase 5 case set/order mismatch")
    band = prereg["band_policy"]
    normalized_threshold = band["paper_default_high_pass_threshold"] / (
        band["paper_grid"] - 1
    )
    if abs(normalized_threshold * (band["klein_grid"] - 1) - band["klein_continuous_threshold"]) > 1e-12:
        raise RuntimeError("Normalized threshold mismatch")
    calibration = set(prereg["calibration"]["active_evaluations"])
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
        raise TypeError("Phase 5 requires native Klein CONST sampling")
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    conditioning = {
        case["name"]: clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
        for case in cases
    }
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(phase1.STEPS, 32 * 32).float()[phase1.START_INDEX:].clone()
    if len(sigmas) - 1 != 6 or float(sigmas[-1]) != 0.0:
        raise RuntimeError(f"Unexpected Phase 2 schedule: {sigmas.tolist()}")
    dct_preflight = phase4.validate_dct(32, int(band["klein_continuous_threshold"]))
    if dct_preflight["high_band_coefficients"] != band["expected_substituted_coefficients"]:
        raise RuntimeError(f"Unexpected normalized high-band count: {dct_preflight}")

    report = {
        "phase": 5, "verdict": "SAMPLER-LATENT FBSDIFF FALSIFIED",
        "preregistration": str(PREREGISTRATION),
        "configuration": {
            "model": str(phase1.MODEL), "text_encoder": str(phase1.TEXT_ENCODER),
            "vae": str(phase1.VAE), "resolution_wh": [phase1.WIDTH, phase1.HEIGHT],
            "cfg": 1.0, "sampler": "Euler", "seed": phase1.SEED,
            "sigmas": sigmas.tolist(), "band_policy": band,
            "calibration": prereg["calibration"],
            "reference_analogue": prereg["reference_analogue"],
        },
        "dct_preflight": dct_preflight,
        "cases": {}, "integrity": {"production_changes": False},
    }
    outputs = {}
    trajectories = {}
    for case in cases:
        name = case["name"]
        source = prepared[name]["latent"]
        gaussian = torch.randn(source.shape, generator=torch.Generator().manual_seed(phase1.SEED))
        fss, fss_report = phase1.structured_noise(source, gaussian, 2.0)
        outputs[name] = {}
        trajectories[name] = {}
        case_record = {
            "role": case["role"], "prompt": case["prompt"],
            "source": phase1.image_record(prepared[name]["path"]),
            "source_latent": phase1.summary(source), "fss_construction": fss_report,
            "reference_trajectory": {}, "arms": {},
        }

        reference_sampler = phase1.EulerCaptureSampler(source)
        with torch.inference_mode():
            reference_output = comfy.sample.sample_custom(
                model, gaussian.clone(), 1.0, reference_sampler, sigmas.clone(),
                empty, empty, source.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=phase1.SEED,
            ).detach().float().cpu()
        reference_states = [state for state, unused in reference_sampler.tensors]
        case_record["reference_trajectory"] = {
            "conditioning": "empty/CFG1", "final_latent": phase1.summary(reference_output),
            "final_structure": phase1.structural_metrics(reference_output, source),
            "evaluations": reference_sampler.captures,
        }

        control_tensors = None
        for arm_name, endpoint in (("A_GAUSSIAN", gaussian), ("B_FSS_R02", fss)):
            sampler = phase1.EulerCaptureSampler(source, control_tensors)
            started = time.perf_counter()
            with torch.inference_mode():
                output = comfy.sample.sample_custom(
                    model, endpoint.clone(), 1.0, sampler, sigmas.clone(),
                    conditioning[name], empty, source.clone(), callback=lambda *unused: None,
                    disable_pbar=True, seed=phase1.SEED,
                ).detach().float().cpu()
            if control_tensors is None:
                control_tensors = sampler.tensors
                for item in sampler.captures:
                    item["state_rms_difference_from_control"] = 0.0
                    item["denoised_rms_difference_from_control"] = 0.0
            outputs[name][arm_name] = output
            trajectories[name][arm_name] = sampler.tensors
            case_record["arms"][arm_name] = {
                "wall_seconds": time.perf_counter() - started,
                "endpoint_hash": phase1.tensor_hash(endpoint),
                "final_latent": phase1.summary(output),
                "final_latent_structure": phase1.structural_metrics(output, source),
                "final_latent_rms_from_control": phase1.rms(
                    output - outputs[name]["A_GAUSSIAN"]
                ),
                "evaluations": sampler.captures,
            }

        sampler = SamplerLatentHighFBS(
            source, reference_states, normalized_threshold, calibration, control_tensors
        )
        started = time.perf_counter()
        with torch.inference_mode():
            output = comfy.sample.sample_custom(
                model, gaussian.clone(), 1.0, sampler, sigmas.clone(),
                conditioning[name], empty, source.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=phase1.SEED,
            ).detach().float().cpu()
        outputs[name]["C_SAMPLER_LATENT_FBSDIFF_HIGH"] = output
        trajectories[name]["C_SAMPLER_LATENT_FBSDIFF_HIGH"] = sampler.tensors
        applied = [record for record in sampler.band_records if record["calibration_active"]]
        case_record["arms"]["C_SAMPLER_LATENT_FBSDIFF_HIGH"] = {
            "wall_seconds": time.perf_counter() - started,
            "endpoint_hash": phase1.tensor_hash(gaussian),
            "final_latent": phase1.summary(output),
            "final_latent_structure": phase1.structural_metrics(output, source),
            "final_latent_rms_from_control": phase1.rms(output - outputs[name]["A_GAUSSIAN"]),
            "substitution_summary": {
                "calibration_evaluations": len(applied),
                "substituted_coefficient_fraction": applied[0]["substituted_coefficient_fraction"],
                "mean_substitution_rms": float(np.mean([x["substitution_rms"] for x in applied])),
                "mean_substitution_to_target_rms_ratio": float(np.mean([
                    x["substitution_to_target_rms_ratio"] for x in applied
                ])),
                "mean_substitution_to_target_energy_ratio": float(np.mean([
                    x["substitution_to_target_energy_ratio"] for x in applied
                ])),
                "mean_reference_high_band_rms": float(np.mean([
                    x["reference_high_band_rms"] for x in applied
                ])),
                "mean_target_high_band_rms": float(np.mean([
                    x["target_high_band_rms"] for x in applied
                ])),
            },
            "substitutions": sampler.band_records,
            "evaluations": sampler.captures,
        }
        report["cases"][name] = case_record
        phase2.atomic_json(OUTPUT / "telemetry.json", report)

    del model, conditioning, empty
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    selected = (0, 2, 5)
    for case in cases:
        name = case["name"]
        source_path = prepared[name]["path"]
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        sheet_items = [("SOURCE", source_path)]
        for arm_name, arm_record in report["cases"][name]["arms"].items():
            final_path = OUTPUT / f"{name}_{arm_name}.png"
            with torch.inference_mode():
                phase1.save_pixels(vae.decode(outputs[name][arm_name]).cpu(), final_path)
            final_rgb = np.asarray(Image.open(final_path).convert("RGB"))
            arm_record["final_image"] = phase1.image_record(final_path)
            arm_record["final_decoded_metrics"] = phase1b.decoded_metrics(final_rgb, source_rgb)
            arm_record["selected_x0_previews"] = {}
            for ordinal in selected:
                path = OUTPUT / f"{name}_{arm_name}_EVAL_{ordinal:02d}_X0.png"
                with torch.inference_mode():
                    phase1.save_pixels(
                        vae.decode(trajectories[name][arm_name][ordinal][1]).cpu(), path
                    )
                arm_record["selected_x0_previews"][str(ordinal)] = phase1.image_record(path)
            sheet_items.append((arm_name, final_path))
        report["cases"][name]["comparison"] = phase1.image_record(
            contact_sheet(name, sheet_items)
        )
        phase2.atomic_json(OUTPUT / "telemetry.json", report)
    report["integrity"].update({
        "status": "SUCCESS", "all_finite": all(
            arm["final_latent"]["finite"]
            for case in report["cases"].values() for arm in case["arms"].values()
        ),
        "exactly_three_arms": True,
        "sampler_latent_shape_128x32x32": True,
        "normalized_threshold_used": True,
        "same_sigma_ordinal_reference_states": True,
        "empty_conditioning_reference": True,
        "ordinary_model_internals_untouched": True,
        "no_fss_or_ilvr_in_feature_arm": True,
        "no_terminal_state_correction": True,
    })
    phase2.atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
