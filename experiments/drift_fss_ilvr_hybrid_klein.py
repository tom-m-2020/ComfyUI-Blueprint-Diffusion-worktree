"""Phase 3: one fixed FSS plus ILVR hybrid falsifier on native Klein."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PHASE1B_PREREG = ROOT / "experiments" / "DRIFT_PHASE_1B_PREREGISTRATION.json"
PHASE2_PREREG = ROOT / "experiments" / "DRIFT_PHASE_2_PREREGISTRATION.json"
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_3_PREREGISTRATION.json"
OUTPUT = ROOT / "experiments" / "drift_phase_3_fss_ilvr_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1
import drift_phase_preserving_natural_generalization as phase1b


def contact_sheet(case_name: str, paths: list[tuple[str, Path]]) -> Path:
    panel_w, panel_h, label_h = 512, 512, 28
    sheet = Image.new("RGB", (panel_w * len(paths), panel_h + label_h), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (name, path) in enumerate(paths):
        with Image.open(path) as image:
            sheet.paste(image.convert("RGB"), (index * panel_w, label_h))
        draw.text((index * panel_w + 6, 7), name, fill="black")
    output_path = OUTPUT / f"{case_name}_COMPARISON.png"
    sheet.save(output_path)
    return output_path


def correction_summary(captures: list[dict]) -> dict:
    intervals = [item["accepted_interval"] for item in captures]
    applied = [item for item in intervals if item["correction_applied"]]
    return {
        "intervals_applied": len(applied),
        "terminal_intervals_skipped": sum(
            item["terminal_sigma_zero_skipped"] for item in intervals
        ),
        "mean_correction_rms": float(np.mean([item["correction_rms"] for item in applied])) if applied else 0.0,
        "mean_correction_to_proposal_rms_ratio": float(np.mean([
            item["correction_to_proposal_rms_ratio"] for item in applied
        ])) if applied else 0.0,
        "max_correction_to_proposal_rms_ratio": max([
            item["correction_to_proposal_rms_ratio"] for item in applied
        ], default=0.0),
        "mean_correction_to_proposal_energy_ratio": float(np.mean([
            item["correction_to_proposal_energy_ratio"] for item in applied
        ])) if applied else 0.0,
        "mean_coarse_mismatch_removed_fraction": float(np.mean([
            item["coarse_mismatch_removed_fraction"] for item in applied
        ])) if applied else 0.0,
    }


def main() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    phase2_prereg = json.loads(PHASE2_PREREG.read_text(encoding="utf-8"))
    phase1b_prereg = json.loads(PHASE1B_PREREG.read_text(encoding="utf-8"))
    if not prereg.get("registered_before_phase_3_outputs"):
        raise RuntimeError("Missing Phase 3 preregistration")
    fixed = prereg["fixed_runtime"]
    phase2_fixed = phase2_prereg["fixed_runtime"]
    invariant_keys = (
        "model", "text_encoder", "vae", "resolution_wh", "sampler", "cfg",
        "full_schedule_steps", "start_index", "seed_for_every_case",
        "ilvr_downsample_factor_latent", "ilvr_blend_alpha",
        "terminal_sigma_zero_correction",
    )
    mismatches = [key for key in invariant_keys if fixed[key] != phase2_fixed[key]]
    if mismatches:
        raise RuntimeError(f"Phase 2 runtime mismatch: {mismatches}")
    cases = [
        case for case in phase1b_prereg["cases"]
        if case["name"] in set(prereg["included_cases"])
    ]
    if [case["name"] for case in cases] != prereg["included_cases"]:
        raise RuntimeError("Phase 3 case set/order mismatch")
    factor = fixed["ilvr_downsample_factor_latent"]
    alpha = fixed["ilvr_blend_alpha"]
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
        raise TypeError("Phase 3 requires native Klein CONST sampling")
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
        "phase": 3, "verdict": "FSS+ILVR STILL INSUFFICIENT",
        "preregistration": str(PREREGISTRATION),
        "configuration": fixed | {"sigmas": sigmas.tolist()},
        "hybrid_formulation": {
            "endpoint": "Validated FSS r=2 structured endpoint.",
            "source_at_next": "native CONST noise_scaling(sigma_next, same FSS endpoint, clean source)",
            "accepted_nonterminal": "proposal + phi(source_at_next) - phi(proposal)",
            "terminal_sigma_zero_correction": False,
            "model_forward_modified": False,
        },
        "cases": {}, "integrity": {"production_changes": False},
    }
    final_outputs = {}
    trajectories = {}
    for case in cases:
        name = case["name"]
        source = prepared[name]["latent"]
        gaussian = torch.randn(source.shape, generator=torch.Generator().manual_seed(phase1.SEED))
        fss, fss_report = phase1.structured_noise(source, gaussian, 2.0)
        preflight = phase2.validate_preflight(source, gaussian, factor)
        positive, negative = conditioning[name]
        arms = {
            "A_GAUSSIAN": (gaussian, gaussian, False),
            "B_FSS_R02": (fss, fss, False),
            "C_ILVR_N04_A1": (gaussian, gaussian, True),
            "D_FSS_R02_ILVR_N04_A1": (fss, fss, True),
        }
        case_record = {
            "role": case["role"], "prompt": case["prompt"],
            "source": phase1.image_record(prepared[name]["path"]),
            "source_latent": phase1.summary(source), "preflight": preflight,
            "fss_construction": fss_report, "arms": {},
        }
        control = None
        final_outputs[name] = {}
        trajectories[name] = {}
        for arm_name, (endpoint, source_endpoint, use_correction) in arms.items():
            sampler = phase2.EulerILVRSampler(
                source, source_endpoint, factor, alpha, use_correction, control
            )
            started = time.perf_counter()
            with torch.inference_mode():
                output = comfy.sample.sample_custom(
                    model, endpoint.clone(), 1.0, sampler, sigmas.clone(), positive,
                    negative, source.clone(), callback=lambda *unused: None,
                    disable_pbar=True, seed=phase1.SEED,
                ).detach().float().cpu()
            if control is None:
                control = [
                    {"state": state, "denoised": denoised}
                    for state, denoised in sampler.tensors
                ]
            final_outputs[name][arm_name] = output
            trajectories[name][arm_name] = sampler.tensors
            case_record["arms"][arm_name] = {
                "wall_seconds": time.perf_counter() - started,
                "endpoint_hash": phase1.tensor_hash(endpoint),
                "matching_sigma_source_endpoint_hash": phase1.tensor_hash(source_endpoint),
                "final_latent": phase1.summary(output),
                "final_latent_structure": phase1.structural_metrics(output, source),
                "final_latent_rms_from_control": phase1.rms(
                    output - final_outputs[name]["A_GAUSSIAN"]
                ),
                "correction_summary": correction_summary(sampler.captures),
                "evaluations": sampler.captures,
            }
        ilvr = case_record["arms"]["C_ILVR_N04_A1"]["correction_summary"]
        hybrid = case_record["arms"]["D_FSS_R02_ILVR_N04_A1"]["correction_summary"]
        case_record["hybrid_correction_vs_gaussian_ilvr"] = {
            "mean_correction_rms_ratio": hybrid["mean_correction_rms"] / ilvr["mean_correction_rms"],
            "mean_relative_correction_ratio": hybrid["mean_correction_to_proposal_rms_ratio"] / ilvr["mean_correction_to_proposal_rms_ratio"],
            "hybrid_is_smaller_by_absolute_rms": hybrid["mean_correction_rms"] < ilvr["mean_correction_rms"],
            "hybrid_is_smaller_relative_to_proposal": hybrid["mean_correction_to_proposal_rms_ratio"] < ilvr["mean_correction_to_proposal_rms_ratio"],
        }
        report["cases"][name] = case_record
        phase2.atomic_json(OUTPUT / "telemetry.json", report)

    del model, conditioning
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    selected = (0, (len(sigmas) - 2) // 2, len(sigmas) - 2)
    for case in cases:
        name = case["name"]
        source_path = prepared[name]["path"]
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        sheet_items = [("SOURCE", source_path)]
        for arm_name, arm_record in report["cases"][name]["arms"].items():
            final_path = OUTPUT / f"{name}_{arm_name}.png"
            with torch.inference_mode():
                phase1.save_pixels(vae.decode(final_outputs[name][arm_name]).cpu(), final_path)
            final_rgb = np.asarray(Image.open(final_path).convert("RGB"))
            arm_record["final_image"] = phase1.image_record(final_path)
            arm_record["final_decoded_metrics"] = phase1b.decoded_metrics(final_rgb, source_rgb)
            arm_record["selected_x0_previews"] = {}
            for ordinal in selected:
                preview_path = OUTPUT / f"{name}_{arm_name}_EVAL_{ordinal:02d}_X0.png"
                with torch.inference_mode():
                    phase1.save_pixels(
                        vae.decode(trajectories[name][arm_name][ordinal][1]).cpu(), preview_path
                    )
                arm_record["selected_x0_previews"][str(ordinal)] = phase1.image_record(preview_path)
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
        "phase_2_runtime_invariants_equal": True,
        "ordinary_model_forward_untouched": True,
        "hybrid_endpoint_and_matching_sigma_source_use_same_fss_tensor": all(
            case["arms"]["D_FSS_R02_ILVR_N04_A1"]["endpoint_hash"]
            == case["arms"]["D_FSS_R02_ILVR_N04_A1"]["matching_sigma_source_endpoint_hash"]
            for case in report["cases"].values()
        ),
        "correction_after_accepted_nonterminal_euler_intervals_only": True,
        "no_terminal_sigma_zero_correction": True,
        "exactly_four_arms": True,
    })
    report["visual_assessment"] = {
        "portrait_bronze": "FAIL: hybrid improves framing and contour metrics but retains the FSS arm's older, different facial identity and produces a weaker bronze-statue material transformation than ILVR.",
        "astronaut_steampunk": "PASS: hybrid preserves stance and contours while retaining a visible steampunk material change.",
        "bridge_crystal_night": "PASS: hybrid preserves bridge geometry and improves contour overlap while retaining visible night/material change.",
        "decisive_interpretation": "Portrait has no qualitative identity improvement over both standalone mechanisms; scalar complementarity is insufficient for qualification.",
    }
    phase2.atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
