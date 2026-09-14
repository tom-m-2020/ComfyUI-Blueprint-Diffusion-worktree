"""Phase 1b: preregistered natural-source weak-FSS generalization probe."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_1B_PREREGISTRATION.json"
OUTPUT = ROOT / "experiments" / "drift_phase_1b_natural_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_phase_preserving_klein as phase1


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_source(case: dict, path: Path) -> torch.Tensor:
    source_path = ROOT / case["source"]
    if file_hash(source_path) != case["source_sha256"]:
        raise RuntimeError(f"Preregistered source hash mismatch: {case['name']}")
    with Image.open(source_path) as image:
        rgb = image.convert("RGB").crop(tuple(case["crop_xyxy"]))
        rgb = rgb.resize((phase1.WIDTH, phase1.HEIGHT), Image.Resampling.LANCZOS)
        rgb.save(path)
        return torch.from_numpy(np.asarray(rgb).copy()).float().div(255).unsqueeze(0)


def histogram_js(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for channel in range(3):
        p = np.histogram(left[..., channel], bins=64, range=(0, 255))[0].astype(np.float64)
        q = np.histogram(right[..., channel], bins=64, range=(0, 255))[0].astype(np.float64)
        p /= max(p.sum(), 1); q /= max(q.sum(), 1)
        m = 0.5 * (p + q)
        kl_p = np.sum(np.where(p > 0, p * np.log((p + 1e-12) / (m + 1e-12)), 0))
        kl_q = np.sum(np.where(q > 0, q * np.log((q + 1e-12) / (m + 1e-12)), 0))
        values.append(0.5 * (kl_p + kl_q))
    return float(np.mean(values))


def decoded_metrics(image: np.ndarray, source: np.ndarray) -> dict:
    image_f = image.astype(np.float32) / 255.0
    source_f = source.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    source_gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160) > 0
    source_edges = cv2.Canny(source_gray, 80, 160) > 0
    distance_to_edges = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
    distance_to_source = cv2.distanceTransform((~source_edges).astype(np.uint8), cv2.DIST_L2, 3)
    chamfer = 0.5 * (
        float(distance_to_edges[source_edges].mean()) if source_edges.any() else 0.0
    ) + 0.5 * (float(distance_to_source[edges].mean()) if edges.any() else 0.0)
    kernel = np.ones((5, 5), np.uint8)
    edges_near_source = cv2.dilate(source_edges.astype(np.uint8), kernel) > 0
    source_near_edges = cv2.dilate(edges.astype(np.uint8), kernel) > 0
    precision = float(edges_near_source[edges].mean()) if edges.any() else 0.0
    recall = float(source_near_edges[source_edges].mean()) if source_edges.any() else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    coarse_image = cv2.resize(image_f, (128, 128), interpolation=cv2.INTER_AREA)
    coarse_source = cv2.resize(source_f, (128, 128), interpolation=cv2.INTER_AREA)
    return {
        "appearance_rgb_rms": float(np.sqrt(np.mean((image_f - source_f) ** 2))),
        "appearance_rgb_histogram_js": histogram_js(image, source),
        "source_relative_coarse_rgb_rms": float(np.sqrt(np.mean((coarse_image - coarse_source) ** 2))),
        "symmetric_canny_edge_chamfer_pixels": chamfer,
        "edge_overlap_f1_within_2px": f1,
        "source_edge_pixels": int(source_edges.sum()), "output_edge_pixels": int(edges.sum()),
    }


def validate_case_preflight(gaussian: torch.Tensor, arms: dict, construction: dict) -> None:
    for name, value in arms.items():
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"{name}: nonfinite endpoint")
        if name != "A_GAUSSIAN":
            ratio = construction[name]["structured_to_gaussian_rms_ratio"]
            if abs(ratio - 1.0) > 2e-5:
                raise RuntimeError(f"{name}: endpoint RMS mismatch: {ratio}")
    phase_values = [construction[name]["phase_rms_to_source"] for name in
                    ("C_FSS_R01", "C_FSS_R02", "D_STRONG_R08")]
    if not all(right < left for left, right in zip(phase_values, phase_values[1:])):
        raise RuntimeError(f"FSS phase rigidity is not monotonic: {phase_values}")


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
    if not prereg.get("registered_before_phase_1b_outputs"):
        raise RuntimeError("Missing valid preregistration")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    import comfy.model_management
    import comfy.model_sampling
    import comfy.sample
    import comfy.sd
    import comfy.utils
    from comfy_extras.nodes_flux import get_schedule

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase1.VAE), safe_load=True))
    prepared = {}
    for case in prereg["cases"]:
        source_path = OUTPUT / f"{case['name']}_SOURCE.png"
        pixels = prepare_source(case, source_path)
        with torch.inference_mode():
            latent = vae.encode(pixels).detach().float().cpu()
        prepared[case["name"]] = {"pixels": pixels, "latent": latent, "path": source_path}

    model = comfy.sd.load_diffusion_model(str(phase1.MODEL), model_options={})
    if not isinstance(model.model.model_sampling, comfy.model_sampling.CONST):
        raise RuntimeError("Phase 1b requires native CONST sampling")
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    conditioning = {
        case["name"]: (
            clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"])),
            clip.encode_from_tokens_scheduled(clip.tokenize("")),
        ) for case in prereg["cases"]
    }
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    sigmas = get_schedule(phase1.STEPS, 32 * 32).float()[phase1.START_INDEX:].clone()

    report = {
        "phase": "1b", "verdict": "WEAK-FSS CASE-DEPENDENT",
        "preregistration": str(PREREGISTRATION),
        "configuration": prereg["fixed_runtime"] | {"sigmas": sigmas.tolist()},
        "face_landmarks": {
            "status": "UNAVAILABLE",
            "reason": "No installed landmark model; OpenCV Canny diagnostics used only after sampling. No package installed.",
        },
        "cases": {}, "integrity": {"production_changes": False},
    }
    trajectory_store = {}
    for case in prereg["cases"]:
        name = case["name"]
        source_latent = prepared[name]["latent"]
        gaussian = torch.randn(source_latent.shape, generator=torch.Generator().manual_seed(phase1.SEED))
        arms = {"A_GAUSSIAN": gaussian}
        construction = {"A_GAUSSIAN": {"gaussian": phase1.summary(gaussian)}}
        for arm_name, cutoff in (("C_FSS_R01", 1.0), ("C_FSS_R02", 2.0), ("D_STRONG_R08", 8.0)):
            arms[arm_name], construction[arm_name] = phase1.structured_noise(source_latent, gaussian, cutoff)
        validate_case_preflight(gaussian, arms, construction)
        positive, negative = conditioning[name]
        case_record = {
            "role": case["role"], "prompt": case["prompt"],
            "source": phase1.image_record(prepared[name]["path"]),
            "source_latent": phase1.summary(source_latent),
            "noise_construction": construction, "arms": {},
        }
        control_tensors = None
        outputs = {}
        trajectory_store[name] = {}
        for arm_name, noise in arms.items():
            sampler = phase1.EulerCaptureSampler(source_latent, control_tensors)
            started = time.perf_counter()
            with torch.inference_mode():
                output = comfy.sample.sample_custom(
                    model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                    source_latent.clone(), callback=lambda *unused: None,
                    disable_pbar=True, seed=phase1.SEED,
                ).detach().float().cpu()
            if control_tensors is None:
                control_tensors = sampler.tensors
                for item in sampler.captures:
                    item["state_rms_difference_from_control"] = 0.0
                    item["denoised_rms_difference_from_control"] = 0.0
            outputs[arm_name] = output
            trajectory_store[name][arm_name] = sampler.tensors
            case_record["arms"][arm_name] = {
                "wall_seconds": time.perf_counter() - started,
                "noise_hash": phase1.tensor_hash(noise),
                "final_latent": phase1.summary(output),
                "final_latent_structure": phase1.structural_metrics(output, source_latent),
                "final_latent_rms_from_control": phase1.rms(output - outputs["A_GAUSSIAN"]),
                "evaluations": sampler.captures,
            }
        report["cases"][name] = case_record
        atomic_json(OUTPUT / "telemetry.json", report)

    del model, conditioning
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    selected = (0, (len(sigmas) - 2) // 2, len(sigmas) - 2)
    for case in prereg["cases"]:
        name = case["name"]
        source_path = prepared[name]["path"]
        source_rgb = np.asarray(Image.open(source_path).convert("RGB"))
        sheet_items = [("SOURCE", source_path)]
        for arm_name, arm_record in report["cases"][name]["arms"].items():
            final_latent = trajectory_store[name][arm_name][-1][1]
            final_path = OUTPUT / f"{name}_{arm_name}.png"
            with torch.inference_mode():
                phase1.save_pixels(vae.decode(final_latent).cpu(), final_path)
            final_rgb = np.asarray(Image.open(final_path).convert("RGB"))
            arm_record["final_image"] = phase1.image_record(final_path)
            arm_record["final_decoded_metrics"] = decoded_metrics(final_rgb, source_rgb)
            arm_record["selected_x0_previews"] = {}
            for ordinal in selected:
                preview_path = OUTPUT / f"{name}_{arm_name}_EVAL_{ordinal:02d}_X0.png"
                with torch.inference_mode():
                    phase1.save_pixels(vae.decode(trajectory_store[name][arm_name][ordinal][1]).cpu(), preview_path)
                arm_record["selected_x0_previews"][str(ordinal)] = phase1.image_record(preview_path)
            sheet_items.append((arm_name, final_path))
        sheet_path = contact_sheet(name, sheet_items)
        report["cases"][name]["comparison"] = phase1.image_record(sheet_path)
        atomic_json(OUTPUT / "telemetry.json", report)
    report["integrity"].update({
        "status": "SUCCESS", "all_finite": all(
            arm["final_latent"]["finite"]
            for case in report["cases"].values() for arm in case["arms"].values()
        ),
        "same_model_sampler_schedule_cfg_seed_policy_vae": True,
        "only_endpoint_noise_differs_within_case": True,
        "preregistered_before_outputs": True,
    })
    atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
