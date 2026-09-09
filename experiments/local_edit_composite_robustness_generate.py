"""Generate the two additional fixed-policy Clown epsilon robustness cases."""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import local_edit_user_clown_epsilon_reproduction as clown


OUTPUT = clown.ROOT / "experiments" / "local_edit_composite_robustness_results"

CASES = {
    "organic_contour": {
        "path": clown.ROOT / "experiments" / "flux2_candidate3_geometry_window_qualification_results" / "h64_person_car_tree_A_BASELINE_32.png",
        "crop": (768, 0, 1792, 1024),
        "prompt": (
            "A wide cinematic photograph continuing one coherent dry desert scene, exactly one full-body woman "
            "in a long gray dress left of center, exactly one tall green tree crossing the right region, "
            "continuous tree canopy and branches, continuous ground plane and distant hills, warm sunset light, "
            "no duplicate people, no duplicate trees, no cropped or repeated tree"
        ),
    },
    "photometric_texture": {
        "path": clown.ROOT / "experiments" / "flux2_candidate3_global_refresh_cadence_results" / "h64_centered_astronaut_A_BASELINE.png",
        "crop": (512, 0, 1536, 1024),
        "prompt": (
            "A wide cinematic photograph of exactly one astronaut standing centered in a continuous open desert, "
            "pale clear sky, warm sand and a straight distant horizon continuing naturally to both sides, coherent "
            "lighting and texture, no duplicate astronauts, no duplicate horizon, no vertical seams"
        ),
    },
}


def pixels_from_crop(path: Path, crop: tuple[int, int, int, int]) -> torch.Tensor:
    image = Image.open(path).convert("RGB").crop(crop)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)


def prepare(source: torch.Tensor) -> dict[str, torch.Tensor]:
    padded, mask = clown.nodes.ImagePadForOutpaint().expand_image(
        source, left=512, top=0, right=512, bottom=0, feathering=0
    )
    return {
        "source_crop": source,
        "padded": padded,
        "resized": clown.scale_dimensions(padded, clown.WIDTH, clown.HEIGHT, "lanczos", "disabled"),
        "mask": clown.scale_dimensions(mask, clown.WIDTH, clown.HEIGHT, "nearest-exact", "disabled"),
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    latents = {}
    for name, config in CASES.items():
        item = prepare(pixels_from_crop(config["path"], config["crop"]))
        with torch.no_grad():
            encoded = vae.encode(item["resized"])
        prepared[name] = item
        latents[name] = {"samples": encoded.detach().clone()}
        clown.metrics_base.save_pixels(item["resized"], OUTPUT / f"{name}_SOURCE.png")
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    sigmas = clown.get_schedule(
        clown.STEPS, round(clown.WIDTH * clown.HEIGHT / (16 * 16))
    ).float()
    outputs = {}
    for name, config in CASES.items():
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        guides = clown.make_guides(latents[name], prepared[name]["mask"], True)
        sampled, denoised, telemetry = clown.execute_arm(
            model, positive, negative, sigmas, latents[name], guides, None
        )
        outputs[name] = {"sampled": sampled, "denoised": denoised}
    del clip
    model.cleanup()
    del model
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    for name, value in outputs.items():
        generated = vae.decode(value["sampled"]).detach().cpu()
        clown.metrics_base.save_pixels(generated, OUTPUT / f"{name}_A_GENERATED.png")
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    print(str(OUTPUT))


if __name__ == "__main__":
    run()
