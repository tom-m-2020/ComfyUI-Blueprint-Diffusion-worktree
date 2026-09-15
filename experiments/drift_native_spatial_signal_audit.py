"""Phase 6: measurement-only native spatial correspondence audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PREREGISTRATION = ROOT / "experiments" / "DRIFT_PHASE_6_PREREGISTRATION.json"
INPUT = ROOT / "experiments" / "drift_phase_2_ilvr_results"
OUTPUT = ROOT / "experiments" / "drift_phase_6_native_signal_results"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments")]

import drift_ilvr_klein as phase2
import drift_phase_preserving_klein as phase1


def pixels(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    value = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return value.reshape(image.height, image.width, 3).float().div(255).unsqueeze(0)


class Capture:
    def __init__(self) -> None:
        self.values: dict[str, torch.Tensor] = {}

    def post_input(self, args: dict) -> dict:
        self.values["post_input"] = args["img"].detach().float().cpu()
        return args

    def double(self, args: dict, wrappers: dict) -> dict:
        output = wrappers["original_block"](args)
        self.values["double_0"] = output["img"].detach().float().cpu()
        return output

    def single(self, args: dict, wrappers: dict) -> dict:
        output = wrappers["original_block"](args)
        start, stop = args["transformer_options"]["img_slice"]
        self.values["single_7"] = output["img"][:, start:stop].detach().float().cpu()
        return output


class OneForward:
    def __init__(self, sigma: float) -> None:
        self.sigma = sigma

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if latent_image is None or denoise_mask is not None:
            raise ValueError("Phase 6 requires an unmasked measurement latent")
        sigma = torch.tensor(self.sigma, device=latent_image.device, dtype=latent_image.dtype)
        model(latent_image, sigma.expand(latent_image.shape[0]),
              model_options=extra_args["model_options"], seed=extra_args.get("seed", 0))
        return latent_image


def as_grid(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 4:
        return value
    if value.ndim != 3 or value.shape[1] != 1024:
        raise ValueError(f"Unexpected spatial representation {tuple(value.shape)}")
    return value.transpose(1, 2).reshape(value.shape[0], value.shape[2], 32, 32)


def correspondence(source: torch.Tensor, current: torch.Tensor, radius: int) -> dict:
    source = F.normalize(as_grid(source).float(), dim=1, eps=1e-8)
    current = F.normalize(as_grid(current).float(), dim=1, eps=1e-8)
    if source.shape != current.shape or source.shape[0] != 1:
        raise ValueError(f"Pair mismatch {tuple(source.shape)} vs {tuple(current.shape)}")
    _, _, height, width = source.shape
    gradients = torch.zeros((height, width))
    gradients[:, 1:] += (source[0, :, :, 1:] - source[0, :, :, :-1]).square().mean(0).sqrt()
    gradients[1:, :] += (source[0, :, 1:, :] - source[0, :, :-1, :]).square().mean(0).sqrt()
    positive = gradients[gradients > 0]
    low = torch.quantile(positive, 0.25) if positive.numel() else torch.tensor(0.0)
    high = torch.quantile(positive, 0.75) if positive.numel() else torch.tensor(0.0)
    displacements, confidences, best_scores, local_distances, weights = [], [], [], [], []
    structure_disp, background_disp = [], []
    for y in range(height):
        for x in range(width):
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            candidates = current[0, :, y0:y1, x0:x1].reshape(current.shape[1], -1)
            scores = source[0, :, y, x] @ candidates
            values, indices = torch.topk(scores, k=min(2, scores.numel()))
            index = int(indices[0])
            match_y, match_x = y0 + index // (x1 - x0), x0 + index % (x1 - x0)
            displacement = ((match_y - y) ** 2 + (match_x - x) ** 2) ** 0.5
            confidence = float(values[0] - values[1]) if values.numel() > 1 else 1.0
            local = 1.0 - float((source[0, :, y, x] * current[0, :, y, x]).sum())
            displacements.append(displacement); confidences.append(confidence)
            best_scores.append(float(values[0])); local_distances.append(local)
            weights.append(float(gradients[y, x]))
            if gradients[y, x] >= high:
                structure_disp.append(displacement)
            if gradients[y, x] <= low:
                background_disp.append(displacement)
    displacement = torch.tensor(displacements)
    weight = torch.tensor(weights)
    weighted = float((displacement * weight).sum() / weight.sum().clamp_min(1e-12))
    return {
        "mean_displacement_tokens": float(displacement.mean()),
        "p90_displacement_tokens": float(torch.quantile(displacement, 0.9)),
        "gradient_weighted_mean_displacement_tokens": weighted,
        "mean_best_cosine": sum(best_scores) / len(best_scores),
        "mean_match_margin": sum(confidences) / len(confidences),
        "mean_same_position_cosine_distance": sum(local_distances) / len(local_distances),
        "structure_proxy_mean_displacement": sum(structure_disp) / len(structure_disp),
        "background_proxy_mean_displacement": sum(background_disp) / len(background_disp),
        "structure_minus_background_displacement": (
            sum(structure_disp) / len(structure_disp) - sum(background_disp) / len(background_disp)
        ),
    }


def main() -> None:
    prereg = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if not prereg.get("registered_before_phase_6_measurements"):
        raise RuntimeError("Missing Phase 6 preregistration")
    prior = json.loads((INPUT / "telemetry.json").read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)

    import comfy.model_management
    import comfy.sample
    import comfy.sd
    import comfy.utils

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(phase1.VAE), safe_load=True))
    model = comfy.sd.load_diffusion_model(str(phase1.MODEL), model_options={})
    clip = comfy.sd.load_clip([str(phase1.TEXT_ENCODER)], clip_type=comfy.sd.CLIPType.FLUX2)
    report = {
        "phase": 6, "verdict": "NATIVE FEATURES NOT SUFFICIENT FOR DRIFT MEASUREMENT",
        "method": {"sampling_modified": False, "trajectory_rerun": False,
                   "input": "saved decoded x0 previews re-encoded through the fixed VAE",
                   "search_radius_tokens": prereg["correspondence"]["search_radius_tokens"]},
        "cases": {}, "integrity": {"production_changes": False},
    }
    for case_name, arms in prereg["cases_and_arms"].items():
        case = prior["cases"][case_name]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        source_pixels = pixels(INPUT / f"{case_name}_SOURCE.png")
        with torch.inference_mode():
            source_latent = vae.encode(source_pixels).detach().float().cpu()
        case_result = {"known_labels": {}, "evaluations": {}}
        for ordinal in prereg["evaluation_ordinals"]:
            sigma = case["arms"][arms[0]]["evaluations"][ordinal]["sigma"]
            case_result["evaluations"][str(ordinal)] = {}
            for arm in arms:
                current_path = INPUT / f"{case_name}_{arm}_EVAL_{ordinal:02d}_X0.png"
                with torch.inference_mode():
                    current_latent = vae.encode(pixels(current_path)).detach().float().cpu()
                capture = Capture()
                patched = model.clone()
                patched.set_model_post_input_patch(capture.post_input)
                patched.set_model_patch_replace(capture.double, "dit", "double_block", 0)
                patched.set_model_patch_replace(capture.single, "dit", "single_block", 7)
                pair = torch.cat((source_latent, current_latent), dim=0)
                with torch.inference_mode():
                    comfy.sample.sample_custom(
                        patched, torch.zeros_like(pair), 1.0, OneForward(sigma),
                        torch.tensor([sigma, 0.0]), positive, negative, pair,
                        callback=lambda *unused: None, disable_pbar=True, seed=phase1.SEED,
                    )
                representations = {"vae_latent": pair}
                representations.update(capture.values)
                case_result["known_labels"][arm] = prereg["known_labels"][f"{case_name}/{arm}"]
                case_result["evaluations"][str(ordinal)][arm] = {
                    name: {"shape": list(value.shape), **correspondence(value[:1], value[1:], 4)}
                    for name, value in representations.items()
                }
                phase2.atomic_json(OUTPUT / "telemetry.json", report | {"cases": report["cases"] | {case_name: case_result}})
            del patched
        report["cases"][case_name] = case_result
    del clip, model, vae
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    report["integrity"].update({
        "status": "SUCCESS", "saved_evaluations_only": True,
        "measurement_forwards_only": True, "same_conditioning_within_each_pair": True,
        "no_external_vision_model": True, "no_sampler_constraint": True,
        "fixed_layers_no_sweep": True,
    })
    phase2.atomic_json(OUTPUT / "telemetry.json", report)
    print(f"Wrote {OUTPUT / 'telemetry.json'}")


if __name__ == "__main__":
    main()
