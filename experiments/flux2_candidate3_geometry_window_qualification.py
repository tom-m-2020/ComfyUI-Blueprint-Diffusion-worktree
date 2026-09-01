"""Phase 6h exact-geometry local-window production qualification."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_local_window_scale as phase6g


OUTPUT = ROOT / "experiments" / "flux2_candidate3_geometry_window_qualification_results"

CASES = (
    {
        "name": "h64_person_car_tree", "width": 2048, "height": 1024,
        "prompt": perf.PROMPT, "seed": 20260831,
        "stress": [("vertical", 48), ("vertical", 64)],
    },
    {
        "name": "h64_long_bridge_train", "width": 2048, "height": 1024,
        "prompt": (
            "A wide cinematic photograph of one single long red suspension bridge "
            "stretching continuously from the far left edge to the far right edge "
            "over calm water, one yellow passenger train centered on the bridge, "
            "one white lighthouse at the far left, one dark stone tower at the far "
            "right, continuous bridge deck and cables, coherent perspective, no "
            "duplicate bridges, trains, lighthouses, or towers"
        ),
        "seed": 20260901, "stress": [("vertical", 48), ("vertical", 64)],
    },
    {
        "name": "h64_subject_on_boundary", "width": 2048, "height": 1024,
        "prompt": (
            "A wide cinematic full-body photograph of exactly one enormous astronaut "
            "standing exactly in the center of the image, one small red rover far "
            "left, one antenna far right, "
            "continuous desert ground and horizon, no duplicate people or body parts"
        ),
        "seed": 20260902, "stress": [("vertical", 64)],
    },
    {
        "name": "h64_subject_in_overlap", "width": 2048, "height": 1024,
        "prompt": (
            "A wide cinematic full-body photograph of exactly one large orange giraffe "
            "standing slightly left of center at about forty-four percent of the image "
            "width, centered inside the overlapping region, one blue jeep far left, "
            "one acacia tree far right, continuous savanna ground and horizon, no "
            "duplicate giraffes, heads, necks, legs, cars, or trees"
        ),
        "seed": 20260903, "stress": [("vertical", 48), ("vertical", 64)],
    },
    {
        "name": "h48_person_car_tree", "width": 1536, "height": 768,
        "prompt": perf.PROMPT, "seed": 20260831,
        "stress": [("vertical", 36), ("vertical", 48)],
    },
    {
        "name": "h48_subject_on_boundary", "width": 1536, "height": 768,
        "prompt": (
            "A wide full-body photograph of exactly one large blue humanoid robot "
            "standing exactly in the center of the image, one yellow motorcycle on "
            "the far left, one "
            "small cactus on the far right, continuous ground and horizon, no "
            "duplicate robots, heads, arms, legs, motorcycles, or cacti"
        ),
        "seed": 20260904, "stress": [("vertical", 48)],
    },
    {
        "name": "h48_long_structure", "width": 1536, "height": 768,
        "prompt": (
            "A wide side-view photograph of one single continuous silver passenger "
            "train stretching horizontally across nearly the full image, all carriages "
            "connected on one straight railway track, mountains and one continuous "
            "horizon behind it, no duplicate trains, broken cars, gaps, or extra tracks"
        ),
        "seed": 20260905, "stress": [("vertical", 36), ("vertical", 48)],
    },
)

POLICIES = {
    (64, 128): (("A_BASELINE_32", 32, 24), ("B_CANDIDATE_64", 64, 48)),
    (48, 96): (("A_BASELINE_32", 32, 24), ("B_CANDIDATE_48", 48, 36)),
}


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.float() - right.float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(left, right)),
    }


def stressed_boundary(value: torch.Tensor, axis: str, coordinate: int) -> dict:
    if axis == "vertical":
        delta = value[..., :, coordinate].float() - value[..., :, coordinate - 1].float()
    elif axis == "horizontal":
        delta = value[..., coordinate, :].float() - value[..., coordinate - 1, :].float()
    else:
        raise ValueError(axis)
    return {
        "axis": axis, "coordinate": coordinate,
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "metric": "adjacent one-token strips at deliberately stressed coordinate",
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "cfg": 1.0, "steps": 4,
            "sampler": "Euler CONST flow", "production_changes": False,
            "policies": {
                "64x128": {"baseline": [32, 24], "candidate": [64, 48]},
                "48x96": {"baseline": [32, 24], "candidate": [48, 36]},
            },
            "repeatability": "each primary measured run repeated from identical noise; exact latent comparison",
        },
        "cases": {},
    }
    outputs = {}
    warmed = set()
    for case_spec in CASES:
        name = case_spec["name"]
        width, height, seed = case_spec["width"], case_spec["height"], case_spec["seed"]
        h_shape = (height // 16, width // 16)
        prompt = case_spec["prompt"]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        noise = torch.randn(
            (1, 128, *h_shape), generator=torch.Generator().manual_seed(seed)
        )
        sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
        if h_shape not in warmed:
            perf.prepare_model_state(model)
            print(f"{h_shape}: unmeasured baseline warm-up", flush=True)
            warm, _ = phase6g.run(
                model, positive, negative, noise, sigmas, 32, 24, seed
            )
            del warm
            warmed.add(h_shape)
        case = {
            "target_pixels_wh": [width, height], "H_shape": list(h_shape),
            "prompt": prompt, "seed": seed, "sigmas": sigmas.tolist(),
            "stress_boundaries": case_spec["stress"], "variants": {},
        }
        for variant, crop_size, stride in POLICIES[h_shape]:
            print(f"{name}: {variant} primary", flush=True)
            output, measurement = phase6g.run(
                model, positive, negative, noise, sigmas, crop_size, stride, seed
            )
            print(f"{name}: {variant} deterministic repeat", flush=True)
            repeated, repeated_measurement = phase6g.run(
                model, positive, negative, noise, sigmas, crop_size, stride, seed
            )
            repeat_difference = difference(output, repeated)
            outputs[f"{name}_{variant}"] = output
            case["variants"][variant] = {
                "geometry": phase6g.geometry_record(
                    width, height, crop_size, stride
                ),
                "measurement": measurement,
                "repeat_measurement_integrity": repeated_measurement["integrity"],
                "repeat_difference": repeat_difference,
                "stressed_boundary_metrics": [
                    stressed_boundary(output, axis, coordinate)
                    for axis, coordinate in case_spec["stress"]
                ],
            }
            if not repeat_difference["bit_exact"]:
                raise RuntimeError(
                    f"{name} {variant} is not deterministic: {repeat_difference}"
                )
        report["cases"][name] = case
        (OUTPUT / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    del clip

    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    decoded = {}
    for name, latent in outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        decoded[name] = {
            "path": str(path), "finite": bool(torch.isfinite(pixels).all()),
            "shape": list(pixels.shape),
        }
    report["decoded_outputs"] = decoded
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        name: {
            variant: {
                "wall_s": value["measurement"]["sampling_wall_seconds"],
                "local_cuda_ms": value["measurement"]["cuda_category_totals_ms"]["local_forward"],
                "boundary": value["measurement"]["final_H_boundary_strip"]["aggregate_rms"],
                "overlap": value["measurement"]["intervals"][-1]["ordinary_overlap_disagreement"]["aggregate_rms"],
                "repeat_exact": value["repeat_difference"]["bit_exact"],
            }
            for variant, value in case["variants"].items()
        }
        for name, case in report["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
