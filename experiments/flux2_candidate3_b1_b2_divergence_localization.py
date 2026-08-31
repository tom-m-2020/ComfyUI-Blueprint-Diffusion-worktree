"""Phase 6c: localize the first B=1 versus duplicated B=2 divergence."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_b1_b2_divergence_results"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

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
REGION_INDEX = 0
TOLERANCE = 2e-5


def flatten_tensors(prefix: str, value: Any) -> dict[str, torch.Tensor]:
    if torch.is_tensor(value):
        return {prefix: value}
    result = {}
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            result.update(flatten_tensors(f"{prefix}[{index}]", item))
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(flatten_tensors(f"{prefix}.{key}", item))
    return result


class BoundaryCapture:
    def __init__(self) -> None:
        self.mode = ""
        self.references: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.comparisons: dict[str, dict[str, Any]] = {}
        self.order: list[str] = []

    def record(self, name: str, value: Any, module=None) -> None:
        for tensor_name, tensor in flatten_tensors(name, value).items():
            detached = tensor.detach()
            if self.mode == "B1":
                self.references[tensor_name] = detached.cpu().clone()
                self.metadata[tensor_name] = {
                    "dtype": str(detached.dtype),
                    "shape_B1": list(detached.shape),
                    "module": None if module is None else f"{type(module).__module__}.{type(module).__name__}",
                }
                self.order.append(tensor_name)
            elif self.mode == "B2" and tensor_name in self.references:
                reference = self.references[tensor_name]
                if detached.ndim == 0 or detached.shape[0] != 2 or reference.ndim == 0 or reference.shape[0] != 1:
                    return
                values = detached.cpu()
                elements = []
                for index in range(2):
                    delta = values[index:index + 1].float() - reference.float()
                    elements.append({
                        "batch_index": index,
                        "max_abs": float(delta.abs().max()),
                        "rms": float(delta.square().mean().sqrt()),
                        "bit_exact": bool(torch.equal(values[index:index + 1], reference)),
                    })
                self.comparisons[tensor_name] = {
                    **self.metadata[tensor_name],
                    "shape_B2": list(detached.shape),
                    "elements": elements,
                    "material": any(item["max_abs"] > TOLERANCE for item in elements),
                }

    def hook(self, name: str):
        def callback(module, inputs, output):
            self.record(f"{name}.input", inputs, module)
            self.record(f"{name}.output", output, module)
        return callback

    def pre_hook_kwargs(self, name: str):
        def callback(module, args, kwargs):
            self.record(f"{name}.args", args, module)
            self.record(f"{name}.kwargs", kwargs, module)
        return callback


class LocalizationSampler(phase2.comfy.samplers.Sampler):
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
        region = FixedCropPlanner().plan(tuple(h.shape[-2:]))[REGION_INDEX]
        crop = h[:, :, region.y:region.y2, region.x:region.x2]
        duplicated = torch.cat((crop, crop), dim=0)
        adapter = Flux2Adapter()
        diffusion = model.inner_model.diffusion_model
        capture = BoundaryCapture()
        handles = []

        original_calculate_input = model_sampling.calculate_input
        original_process_img = diffusion.process_img

        def calculate_input_wrapper(this, sigma_value, latent):
            result = original_calculate_input(sigma_value, latent)
            capture.record("calculate_input.latent", latent)
            capture.record("calculate_input.sigma", sigma_value)
            capture.record("calculate_input.output", result)
            return result

        def process_img_wrapper(this, x, index=0, h_offset=0, w_offset=0, transformer_options={}):
            result = original_process_img(
                x, index=index, h_offset=h_offset, w_offset=w_offset,
                transformer_options=transformer_options,
            )
            capture.record("process_img.input", x)
            capture.record("process_img.tokens", result[0])
            capture.record("process_img.img_ids", result[1])
            return result

        model_sampling.calculate_input = MethodType(calculate_input_wrapper, model_sampling)
        diffusion.process_img = MethodType(process_img_wrapper, diffusion)

        handles.append(diffusion.register_forward_pre_hook(
            capture.pre_hook_kwargs("model_core"), with_kwargs=True
        ))
        modules = {
            "img_in": diffusion.img_in,
            "time_in": diffusion.time_in,
            "txt_in": diffusion.txt_in,
            "pe_embedder": diffusion.pe_embedder,
            "double0": diffusion.double_blocks[0],
            "double0.img_qkv": diffusion.double_blocks[0].img_attn.qkv,
            "double0.txt_qkv": diffusion.double_blocks[0].txt_attn.qkv,
            "double0.img_attn_proj": diffusion.double_blocks[0].img_attn.proj,
            "double0.txt_attn_proj": diffusion.double_blocks[0].txt_attn.proj,
            "double0.img_mlp_in": diffusion.double_blocks[0].img_mlp[0],
            "double0.img_mlp_out": diffusion.double_blocks[0].img_mlp[2],
            "double0.txt_mlp_in": diffusion.double_blocks[0].txt_mlp[0],
            "double0.txt_mlp_out": diffusion.double_blocks[0].txt_mlp[2],
            "final_layer": diffusion.final_layer,
        }
        if diffusion.vector_in is not None:
            modules["vector_in"] = diffusion.vector_in
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture.hook(name)))

        def attention_output_patch(attention, options):
            if options.get("block_type") == "double" and options.get("block_index") == 0:
                capture.record("double0.attention_output", attention)
            return attention

        model_options = extra_args["model_options"].copy()
        transformer = model_options.get("transformer_options", {}).copy()
        patches = transformer.get("patches", {}).copy()
        attention_patches = list(patches.get("attn1_output_patch", []))
        attention_patches.append(attention_output_patch)
        patches["attn1_output_patch"] = attention_patches
        transformer["patches"] = patches
        model_options["transformer_options"] = transformer

        rope = {"shift_y": float(region.y), "shift_x": float(region.x)}
        options = adapter._options(model_options, rope)
        try:
            capture.mode = "B1"
            prediction_b1 = model(
                crop, sigma.expand(1), model_options=options,
                seed=extra_args.get("seed", 0),
            )
            capture.record("guider_prediction", prediction_b1)
            capture.mode = "B2"
            prediction_b2 = model(
                duplicated, sigma.expand(2), model_options=options,
                seed=extra_args.get("seed", 0),
            )
            capture.record("guider_prediction", prediction_b2)
        finally:
            capture.mode = ""
            for handle in handles:
                handle.remove()
            model_sampling.calculate_input = original_calculate_input
            diffusion.process_img = original_process_img

        ordered = []
        seen = set()
        for name in capture.order:
            if name in capture.comparisons and name not in seen:
                ordered.append({"name": name, **capture.comparisons[name]})
                seen.add(name)
        first_material = next((item for item in ordered if item["material"]), None)
        img_in_weight = diffusion.img_in.weight
        self.report = {
            "target_H_shape": list(TARGET_HW),
            "region": vars(region),
            "sigma": float(sigma),
            "B1_input_shape": list(crop.shape),
            "B2_input_shape": list(duplicated.shape),
            "tolerance": TOLERANCE,
            "boundaries_in_execution_order": ordered,
            "first_material_divergence": first_material,
            "img_in_operator": {
                "module": f"{type(diffusion.img_in).__module__}.{type(diffusion.img_in).__name__}",
                "weight_type": f"{type(img_in_weight).__module__}.{type(img_in_weight).__name__}",
                "weight_shape": list(img_in_weight.shape),
                "weight_dtype": str(img_in_weight.dtype),
                "quant_format": getattr(diffusion.img_in, "quant_format", None),
                "layout_type": getattr(diffusion.img_in, "layout_type", None),
                "has_input_scale": getattr(diffusion.img_in, "input_scale", None) is not None,
                "has_pre_quant_scale": getattr(diffusion.img_in, "pre_quant_scale", None) is not None,
                "full_precision_mm": bool(getattr(diffusion.img_in, "_full_precision_mm", False)),
            },
            "prediction_comparison": capture.comparisons.get("guider_prediction"),
            "restored_runtime_methods": (
                model_sampling.calculate_input == original_calculate_input
                and diffusion.process_img == original_process_img
            ),
        }
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
    sampler = LocalizationSampler()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), disable_pbar=True, seed=SEED,
        )

    report = {
        "model": str(phase2.MODEL_PATH),
        "seed": SEED,
        "prompt": PROMPT,
        "control": "B=1 crop versus B=2 identical duplicated crop",
        **sampler.report,
    }
    first = report["first_material_divergence"]
    if first is None:
        report["verdict"] = "INCONCLUSIVE"
    elif first["name"].startswith(("calculate_input", "model_core", "process_img", "img_in", "txt_in", "time_in", "vector_in")):
        module = first.get("module") or ""
        report["verdict"] = (
            "DIVERGENCE IN QUANTIZED OPERATOR"
            if "quant" in module.lower() else "DIVERGENCE IN INPUT/EMBEDDING PATH"
        )
    elif first["name"].startswith("double0"):
        report["verdict"] = "DIVERGENCE IN ATTENTION/BLOCK EXECUTION"
    elif first["name"] == "guider_prediction":
        report["verdict"] = "DIVERGENCE ABOVE MODEL CORE"
    else:
        report["verdict"] = "INCONCLUSIVE"
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "first_material_divergence": first,
        "prediction_comparison": report["prediction_comparison"],
        "verdict": report["verdict"],
        "report": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
