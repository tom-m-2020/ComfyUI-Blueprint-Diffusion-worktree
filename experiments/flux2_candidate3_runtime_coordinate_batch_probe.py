"""Experiment-only per-batch FLUX.2 coordinate preparation override."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from einops import rearrange


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_runtime_coordinate_batch_results"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import comfy.ldm.common_dit
import flux2_coarse_global_local_falsification as phase2


def load_production_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "blueprint_diffusion",
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blueprint_diffusion"] = module
    spec.loader.exec_module(module)


load_production_package()
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.regions import FixedCropPlanner
from blueprint_diffusion.sampling.euler import validate_schedule


PROMPT = (
    "A cinematic wide-angle photograph of exactly one large full-body woman "
    "standing centered in the foreground, exactly one red vintage car on the "
    "left and exactly one tall green tree on the right, continuous ground, "
    "coherent perspective, no duplicate people, cars, or trees"
)
SEED = 20260831
TARGET_HW = (128, 64)
SELECTED_REGION_INDICES = (0, 4)
OPTION_KEY = "blueprint_per_batch_crop_offsets"


def clone_options(options: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    result = options.copy()
    transformer = options.get("transformer_options", {}).copy()
    transformer.update(extra)
    result["transformer_options"] = transformer
    return result


def per_batch_process_img(self, x, index=0, h_offset=0, w_offset=0, transformer_options={}):
    offsets = transformer_options.get(OPTION_KEY)
    if offsets is None:
        return self._blueprint_original_process_img(
            x,
            index=index,
            h_offset=h_offset,
            w_offset=w_offset,
            transformer_options=transformer_options,
        )
    if transformer_options.get("rope_options") is not None:
        raise ValueError("Per-batch Blueprint offsets cannot be combined with rope_options.")

    batch, channels, height, width = x.shape
    if len(offsets) != batch:
        raise ValueError(f"Expected {batch} Blueprint offsets, got {len(offsets)}.")
    patch_size = self.patch_size
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (patch_size, patch_size))
    image = rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch_size,
        pw=patch_size,
    )
    steps_h = (height + patch_size // 2) // patch_size
    steps_w = (width + patch_size // 2) // patch_size
    base_h = (h_offset + patch_size // 2) // patch_size
    base_w = (w_offset + patch_size // 2) // patch_size
    ids = torch.zeros(
        (batch, steps_h, steps_w, len(self.params.axes_dim)),
        device=x.device,
        dtype=torch.float32,
    )
    for batch_index, offset in enumerate(offsets):
        shift_y, shift_x = (float(offset[0]), float(offset[1]))
        ids[batch_index, :, :, 0] = float(index)
        ids[batch_index, :, :, 1] = torch.linspace(
            base_h + shift_y,
            base_h + shift_y + steps_h - 1,
            steps=steps_h,
            device=x.device,
            dtype=torch.float32,
        ).unsqueeze(1)
        ids[batch_index, :, :, 2] = torch.linspace(
            base_w + shift_x,
            base_w + shift_x + steps_w - 1,
            steps=steps_w,
            device=x.device,
            dtype=torch.float32,
        ).unsqueeze(0)
    return image, ids.reshape(batch, steps_h * steps_w, len(self.params.axes_dim))


@contextmanager
def scoped_coordinate_override(diffusion_model):
    if OPTION_KEY in diffusion_model.__dict__:
        raise RuntimeError(f"Unexpected existing {OPTION_KEY} attribute.")
    original = diffusion_model.process_img
    had_instance_method = "process_img" in diffusion_model.__dict__
    previous_instance_method = diffusion_model.__dict__.get("process_img")
    diffusion_model._blueprint_original_process_img = original
    diffusion_model.process_img = MethodType(per_batch_process_img, diffusion_model)
    try:
        yield
    finally:
        del diffusion_model._blueprint_original_process_img
        if had_instance_method:
            diffusion_model.process_img = previous_instance_method
        else:
            del diffusion_model.process_img


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = value.float() - reference.float()
    return {
        "shape": list(value.shape),
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "bit_exact": bool(torch.equal(value, reference)),
        "finite": bool(torch.isfinite(value).all()),
    }


class ProbeSampler(phase2.comfy.samplers.Sampler):
    def __init__(self) -> None:
        self.report: dict[str, Any] = {}

    def sample(
        self, model, sigmas, extra_args, callback, noise, latent_image=None,
        denoise_mask=None, disable_pbar=False,
    ):
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        sigma = sigmas[0]
        regions = FixedCropPlanner().plan(tuple(h.shape[-2:]))
        selected = [regions[index] for index in SELECTED_REGION_INDICES]
        adapter = Flux2Adapter()

        sequential = []
        views = []
        for region in selected:
            view = h[:, :, region.y:region.y2, region.x:region.x2]
            views.append(view)
            sequential.append(adapter.predict_region(
                guider=model,
                h_view=view,
                sigma=sigma,
                canvas=TARGET_HW,
                region=region,
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            ))

        batch_input = torch.cat(views, dim=0)
        offsets = [(region.y, region.x) for region in selected]
        diffusion_model = model.inner_model.diffusion_model
        original_bound = diffusion_model.process_img

        repeated_input = torch.cat((views[0], views[0]), dim=0)
        ordinary_repeated = model(
            repeated_input,
            sigma.expand(2),
            model_options=adapter._options(
                extra_args["model_options"],
                {"shift_y": float(selected[0].y), "shift_x": float(selected[0].x)},
            ),
            seed=extra_args.get("seed", 0),
        )
        repeated_control = [
            difference(ordinary_repeated[index:index + 1], sequential[0])
            for index in range(2)
        ]

        sequential_ids = []
        for view, region in zip(views, selected):
            _, ids = original_bound(
                view,
                transformer_options={
                    "rope_options": {
                        "shift_y": float(region.y),
                        "shift_x": float(region.x),
                    }
                },
            )
            sequential_ids.append(ids)
        with scoped_coordinate_override(diffusion_model):
            _, batch_ids = diffusion_model.process_img(
                batch_input,
                transformer_options={OPTION_KEY: offsets},
            )
            batched = model(
                batch_input,
                sigma.expand(len(selected)),
                model_options=clone_options(
                    extra_args["model_options"], {OPTION_KEY: offsets}
                ),
                seed=extra_args.get("seed", 0),
            )
        restored = (
            diffusion_model.process_img.__func__ is original_bound.__func__
            and diffusion_model.process_img.__self__ is original_bound.__self__
        )

        comparisons = [
            difference(batched[index:index + 1], sequential[index])
            for index in range(len(selected))
        ]
        tolerance = 2e-5
        id_comparisons = [
            {
                "bit_exact": bool(torch.equal(batch_ids[index:index + 1], sequential_ids[index])),
                "max_abs": float((batch_ids[index:index + 1] - sequential_ids[index]).abs().max()),
            }
            for index in range(len(selected))
        ]
        self.report = {
            "target_H_shape": list(TARGET_HW),
            "sigma": float(sigma),
            "accepted_state_source": "initial H from mapped ordinary noise at sigma[0]",
            "input_batch_shape": list(batch_input.shape),
            "crop_order": [region.index for region in selected],
            "crops": [vars(region) for region in selected],
            "absolute_coordinate_endpoints": [
                {
                    "first_yx": [region.y, region.x],
                    "last_yx": [region.y2 - 1, region.x2 - 1],
                }
                for region in selected
            ],
            "comparisons": comparisons,
            "coordinate_id_comparisons": id_comparisons,
            "ordinary_same_crop_same_coordinate_batch_control": repeated_control,
            "tolerance": tolerance,
            "within_tolerance": all(item["max_abs"] <= tolerance for item in comparisons),
            "coordinate_ids_bit_exact": all(item["bit_exact"] for item in id_comparisons),
            "ordinary_batch_within_tolerance": all(
                item["max_abs"] <= tolerance for item in repeated_control
            ),
            "original_method_restored": restored,
            "no_cross_crop_attention": "batch dimension remains independent; no spatial concatenation",
        }
        if self.report["within_tolerance"] and restored:
            self.report["verdict"] = "RUNTIME COORDINATE OVERRIDE FEASIBLE"
        elif (
            self.report["coordinate_ids_bit_exact"]
            and not self.report["ordinary_batch_within_tolerance"]
        ):
            self.report["verdict"] = "NATIVE BATCH EXECUTION NOT EQUIVALENT"
        else:
            self.report["verdict"] = "INCONCLUSIVE"
        return model_sampling.inverse_noise_scaling(sigmas[-1], h)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn(
        (1, 128, *TARGET_HW), generator=torch.Generator().manual_seed(SEED)
    )
    sigmas = phase2.get_schedule(4, math.prod(TARGET_HW)).float().clone()
    sampler = ProbeSampler()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model,
            noise,
            1.0,
            sampler,
            sigmas,
            positive,
            negative,
            torch.zeros_like(noise),
            disable_pbar=True,
            seed=SEED,
        )
    report = {
        "model": str(phase2.MODEL_PATH),
        "prompt": PROMPT,
        "seed": SEED,
        "implementation_scope": (
            "experiment-only instance method override of Flux.process_img; "
            "restored immediately after one batched call"
        ),
        **sampler.report,
    }
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
