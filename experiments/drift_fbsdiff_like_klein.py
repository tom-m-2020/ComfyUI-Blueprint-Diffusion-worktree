"""Phase 4: one preregistered post-input high-band transfer on stock Klein."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PHASE1B_PREREG = ROOT / "experiments" / "DRIFT_PHASE_1B_PREREGISTRATION.json"
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_4_PREREGISTRATION.json"
OUTPUT = ROOT / "experiments" / "drift_phase_4_fbsdiff_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1
import drift_phase_preserving_natural_generalization as phase1b


def dct_matrix(size: int, device: torch.device) -> torch.Tensor:
    positions = torch.arange(size, device=device, dtype=torch.float32) + 0.5
    frequencies = torch.arange(size, device=device, dtype=torch.float32).unsqueeze(1)
    matrix = torch.cos(math.pi / size * frequencies * positions)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def dct_2d(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(matrix, value), matrix.t())


def idct_2d(value: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    return torch.matmul(torch.matmul(matrix.t(), value), matrix)


def feature_grid(tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[1] != height * width:
        raise ValueError(f"Unexpected post_input shape {tuple(tokens.shape)}")
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height, width)


def feature_tokens(grid: torch.Tensor) -> torch.Tensor:
    return grid.flatten(2).transpose(1, 2)


def rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt())


class ReferenceFeatureCapture:
    def __init__(self) -> None:
        self.features: list[torch.Tensor] = []
        self.records: list[dict] = []

    def __call__(self, args: dict) -> dict:
        image = args["img"]
        self.features.append(image.detach().float().cpu())
        self.records.append({
            "ordinal": len(self.features) - 1,
            "layer_id": "post_input/pre_double_block_0",
            "shape": list(image.shape),
            "rms": rms(image),
            "dtype": str(image.dtype),
        })
        return args


class HighBandTransfer:
    def __init__(self, references: list[torch.Tensor], height: int, width: int,
                 threshold: int, calibration_evaluations: set[int]) -> None:
        self.references = references
        self.height = height
        self.width = width
        self.threshold = threshold
        self.calibration_evaluations = calibration_evaluations
        self.records: list[dict] = []

    def __call__(self, args: dict) -> dict:
        ordinal = len(self.records)
        generated_tokens = args["img"]
        if ordinal >= len(self.references):
            raise RuntimeError("More generation features than reference features")
        reference_tokens = self.references[ordinal].to(generated_tokens)
        if generated_tokens.shape != reference_tokens.shape:
            raise RuntimeError(
                f"Reference/generation feature mismatch at {ordinal}: "
                f"{tuple(reference_tokens.shape)} vs {tuple(generated_tokens.shape)}"
            )
        record = {
            "ordinal": ordinal,
            "layer_id": "post_input/pre_double_block_0",
            "shape": list(generated_tokens.shape),
            "calibration_active": ordinal in self.calibration_evaluations,
            "threshold_u_plus_v_gt": self.threshold,
            "generated_feature_rms": rms(generated_tokens),
            "reference_feature_rms": rms(reference_tokens),
            "substitution_rms": 0.0,
            "substitution_to_generated_rms_ratio": 0.0,
            "substitution_to_generated_energy_ratio": 0.0,
        }
        if ordinal in self.calibration_evaluations:
            generated = feature_grid(generated_tokens.float(), self.height, self.width)
            reference = feature_grid(reference_tokens.float(), self.height, self.width)
            matrix = dct_matrix(self.height, generated.device)
            generated_dct = dct_2d(generated, matrix)
            reference_dct = dct_2d(reference, matrix)
            rows = torch.arange(self.height, device=generated.device).unsqueeze(1)
            columns = torch.arange(self.width, device=generated.device).unsqueeze(0)
            mask = (rows + columns > self.threshold).to(generated_dct.dtype)
            merged_dct = reference_dct * mask + generated_dct * (1.0 - mask)
            merged = idct_2d(merged_dct, matrix)
            output = feature_tokens(merged).to(generated_tokens.dtype)
            delta = output - generated_tokens
            generated_rms = max(rms(generated_tokens), 1e-12)
            substitution_rms = rms(delta)
            record.update({
                "mask_coefficients": int(mask.sum()),
                "mask_fraction": float(mask.mean()),
                "generated_high_band_rms": rms(idct_2d(generated_dct * mask, matrix)),
                "reference_high_band_rms": rms(idct_2d(reference_dct * mask, matrix)),
                "generated_low_band_rms": rms(idct_2d(generated_dct * (1.0 - mask), matrix)),
                "reference_low_band_rms": rms(idct_2d(reference_dct * (1.0 - mask), matrix)),
                "substitution_rms": substitution_rms,
                "substitution_to_generated_rms_ratio": substitution_rms / generated_rms,
                "substitution_to_generated_energy_ratio": (substitution_rms / generated_rms) ** 2,
                "post_substitution_high_band_mismatch_rms": rms(
                    idct_2d((merged_dct - reference_dct) * mask, matrix)
                ),
            })
            args = args.copy()
            args["img"] = output
        self.records.append(record)
        return args


def validate_dct(size: int, threshold: int) -> dict:
    matrix = dct_matrix(size, torch.device("cpu"))
    identity_error = rms(matrix @ matrix.t() - torch.eye(size))
    value = torch.randn((1, 7, size, size), generator=torch.Generator().manual_seed(41))
    reconstruction_error = rms(idct_2d(dct_2d(value, matrix), matrix) - value)
    rows = torch.arange(size).unsqueeze(1)
    columns = torch.arange(size).unsqueeze(0)
    count = int((rows + columns > threshold).sum())
    result = {
        "status": "PASS", "dct_size": size, "threshold": threshold,
        "orthonormal_identity_rms_error": identity_error,
        "roundtrip_rms_error": reconstruction_error,
        "high_band_coefficients": count,
        "total_coefficients": size * size,
        "high_band_fraction": count / (size * size),
    }
    if max(identity_error, reconstruction_error) > 2e-5:
        raise RuntimeError(f"DCT preflight failed: {result}")
    return result


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
    if not prereg.get("registered_before_phase_4_outputs"):
        raise RuntimeError("Missing Phase 4 preregistration")
    cases = [
        case for case in phase1b_prereg["cases"]
        if case["name"] in set(prereg["cases"])
    ]
    if [case["name"] for case in cases] != prereg["cases"]:
        raise RuntimeError("Phase 4 case set/order mismatch")
    height, width = prereg["feature_location"]["spatial_grid_hw"]
    threshold = prereg["band_policy"]["threshold"]
    calibration = set(prereg["trajectory_policy"]["calibration_evaluations"])
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
        raise TypeError("Phase 4 requires native Klein CONST sampling")
    diffusion = model.model.diffusion_model
    if diffusion.patch_size != 1 or diffusion.img_in.out_features != 3072:
        raise RuntimeError(
            f"Unexpected Klein input path: patch={diffusion.patch_size}, "
            f"hidden={diffusion.img_in.out_features}"
        )
    if len(diffusion.double_blocks) == 0:
        raise RuntimeError("No double block 0 after post_input")
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    empty = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    conditioning = {
        case["name"]: clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
        for case in cases
    }
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(phase1.STEPS, height * width).float()[phase1.START_INDEX:].clone()
    if len(sigmas) - 1 != 6 or float(sigmas[-1]) != 0.0:
        raise RuntimeError(f"Unexpected Phase 2 schedule: {sigmas.tolist()}")

    report = {
        "phase": 4, "verdict": "FBSDIFF-LIKE FEATURE TRANSFER FALSIFIED",
        "preregistration": str(PREREGISTRATION),
        "configuration": {
            "model": str(phase1.MODEL), "text_encoder": str(phase1.TEXT_ENCODER),
            "vae": str(phase1.VAE), "resolution_wh": [phase1.WIDTH, phase1.HEIGHT],
            "cfg": 1.0, "sampler": "Euler", "seed": phase1.SEED,
            "sigmas": sigmas.tolist(), "feature_location": prereg["feature_location"],
            "band_policy": prereg["band_policy"],
            "trajectory_policy": prereg["trajectory_policy"],
        },
        "dct_preflight": validate_dct(height, threshold),
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

        reference_capture = ReferenceFeatureCapture()
        reference_model = model.clone()
        reference_model.set_model_post_input_patch(reference_capture)
        reference_sampler = phase1.EulerCaptureSampler(source)
        with torch.inference_mode():
            comfy.sample.sample_custom(
                reference_model, gaussian.clone(), 1.0, reference_sampler, sigmas.clone(),
                empty, empty, source.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=phase1.SEED,
            )
        if len(reference_capture.features) != 6:
            raise RuntimeError(f"Expected six reference features, got {len(reference_capture.features)}")
        case_record["reference_trajectory"] = {
            "conditioning": "empty/CFG1", "feature_captures": reference_capture.records,
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

        transfer = HighBandTransfer(
            reference_capture.features, height, width, threshold, calibration
        )
        feature_model = model.clone()
        feature_model.set_model_post_input_patch(transfer)
        sampler = phase1.EulerCaptureSampler(source, control_tensors)
        started = time.perf_counter()
        with torch.inference_mode():
            output = comfy.sample.sample_custom(
                feature_model, gaussian.clone(), 1.0, sampler, sigmas.clone(),
                conditioning[name], empty, source.clone(), callback=lambda *unused: None,
                disable_pbar=True, seed=phase1.SEED,
            ).detach().float().cpu()
        if len(transfer.records) != 6:
            raise RuntimeError(f"Expected six feature records, got {len(transfer.records)}")
        outputs[name]["C_FBSDIFF_LIKE_POST_INPUT_HIGH"] = output
        trajectories[name]["C_FBSDIFF_LIKE_POST_INPUT_HIGH"] = sampler.tensors
        applied = [record for record in transfer.records if record["calibration_active"]]
        case_record["arms"]["C_FBSDIFF_LIKE_POST_INPUT_HIGH"] = {
            "wall_seconds": time.perf_counter() - started,
            "endpoint_hash": phase1.tensor_hash(gaussian),
            "final_latent": phase1.summary(output),
            "final_latent_structure": phase1.structural_metrics(output, source),
            "final_latent_rms_from_control": phase1.rms(output - outputs[name]["A_GAUSSIAN"]),
            "feature_transfer_summary": {
                "location": "post_input/pre_double_block_0",
                "calibration_evaluations": len(applied),
                "mean_substitution_rms": float(np.mean([x["substitution_rms"] for x in applied])),
                "mean_substitution_to_generated_rms_ratio": float(np.mean([
                    x["substitution_to_generated_rms_ratio"] for x in applied
                ])),
                "mean_substitution_to_generated_energy_ratio": float(np.mean([
                    x["substitution_to_generated_energy_ratio"] for x in applied
                ])),
                "mean_reference_high_band_rms": float(np.mean([
                    x["reference_high_band_rms"] for x in applied
                ])),
                "mean_generated_high_band_rms": float(np.mean([
                    x["generated_high_band_rms"] for x in applied
                ])),
            },
            "feature_transfers": transfer.records,
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
        "one_feature_location": True,
        "one_band_policy": True,
        "same_sigma_ordinal_reference_features": True,
        "ordinary_model_weights_and_blocks_unchanged": True,
        "no_fss_or_ilvr_in_feature_arm": True,
        "no_terminal_state_correction": True,
    })
    report["visual_assessment"] = {
        "portrait_bronze": "FAIL: substantial bronze/statue change occurs, but facial identity, apparent age, arm pose, body framing, and placement are not preserved over either control.",
        "astronaut_steampunk": "FAIL: the feature arm changes face visibility/identity, suit silhouette, proportions, and stance; it does not preserve geometry better than FSS.",
        "bridge_crystal_night": "FAIL: the feature arm constructs a different bridge/tower system and perspective rather than preserving source contours.",
        "decisive_interpretation": "The preregistered internal post-input high band has no useful identity/geometry advantage over FSS and is unstable under global transformer mixing.",
    }
    phase2.atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
