"""Phase 44: fixed native Z-Image-Turbo Terminal Refine discriminator."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
MODEL_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\diffusion_models"
    r"\T5B_Z-Image-Turbo-FP8\z-image-turbo-fp8-e4m3fn.safetensors"
)
TEXT_ENCODER_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\text_encoders"
    r"\f2k4_Comfy-Org_vae-text-encorder-for-flux-klein-4b_text_encoders"
    r"\qwen_3_4b_fp4_flux2.safetensors"
)
VAE_PATH = Path(
    r"C:\Users\Tom-M\data\a\ai\models-1\models\vae"
    r"\Comfy-Org_z_image_turbo_vae\ae.safetensors"
)
sys.path[:0] = [str(COMFY_ROOT), str(PACKAGE)]

if "blueprint_diffusion" not in sys.modules:
    specification = importlib.util.spec_from_file_location(
        "blueprint_diffusion", PACKAGE / "__init__.py",
        submodule_search_locations=[str(PACKAGE)],
    )
    package_module = importlib.util.module_from_spec(specification)
    sys.modules["blueprint_diffusion"] = package_module
    specification.loader.exec_module(package_module)

import comfy.model_management
import comfy.sampler_helpers
import comfy.samplers
import comfy.sd
import comfy.utils
from comfy_extras.nodes_custom_sampler import Guider_Basic
from blueprint_diffusion.regions import OverlapAssembler, Region


OUTPUT = ROOT / "experiments" / "zimage_terminal_refine_discriminator_results"
PROMPT = (
    "A cinematic photograph of exactly one red vintage car on the left, "
    "one tall green tree in the center, and one small white house on the right, "
    "all on one grassy field beneath one continuous horizon, coherent perspective, "
    "no duplicate objects."
)
SEED = 20260921
SIGMAS = (
    1.0, 0.9545454383, 0.8999999762, 0.8333333135, 0.75,
    0.6428571343, 0.5, 0.3000000119, 0.0,
)
LOCAL_SIGMA = SIGMAS[-2]
G_HW = (128, 128)
H_HW = (256, 256)
F_HW = (64, 64)
STRIDE_HW = (48, 48)
W_HW = (128, 128)
CHANNELS = 16


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().float().cpu().numpy().tobytes()).hexdigest()


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_tensor(path: Path, value: torch.Tensor) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def starts(length: int, size: int, stride: int) -> tuple[int, ...]:
    if not 0 < stride <= size <= length:
        raise ValueError("Phase-44 region axes require 0 < stride <= footprint <= destination.")
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return tuple(values)


def regions() -> tuple[Region, ...]:
    ys = starts(H_HW[0], F_HW[0], STRIDE_HW[0])
    xs = starts(H_HW[1], F_HW[1], STRIDE_HW[1])
    result = tuple(Region(index, y, x, *F_HW)
                   for index, (y, x) in enumerate((y, x) for y in ys for x in xs))
    if len(result) != 25:
        raise RuntimeError(f"Phase 44 requires exactly 25 regions, got {len(result)}.")
    return result


def axis_coordinates(start: int, count: int, source: int, destination: int,
                     device: torch.device) -> torch.Tensor:
    output = torch.arange(start, start + count, dtype=torch.float64, device=device)
    return ((output + 0.5) * source / destination - 0.5).clamp(0, source - 1)


def bounded_transfer(value: torch.Tensor, region: Region) -> tuple[torch.Tensor, tuple[int, int]]:
    if tuple(value.shape) != (1, CHANNELS, *G_HW):
        raise ValueError(f"Phase-44 terminal G must be [1,16,128,128], got {tuple(value.shape)}.")
    gy = axis_coordinates(region.y, region.height, G_HW[0], H_HW[0], value.device)
    gx = axis_coordinates(region.x, region.width, G_HW[1], H_HW[1], value.device)
    y0, y1 = int(torch.floor(gy.min())), int(torch.ceil(gy.max()))
    x0, x1 = int(torch.floor(gx.min())), int(torch.ceil(gx.max()))
    source = value[:, :, y0:y1 + 1, x0:x1 + 1]
    ny = 2.0 * (gy - y0) / (source.shape[-2] - 1) - 1.0
    nx = 2.0 * (gx - x0) / (source.shape[-1] - 1) - 1.0
    grid_y, grid_x = torch.meshgrid(ny, nx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].to(dtype=value.dtype)
    footprint = F.grid_sample(source, grid, mode="bilinear", padding_mode="border",
                              align_corners=True)
    working = F.interpolate(footprint, size=W_HW, mode="nearest")
    return working, (source.shape[-2], source.shape[-1])


def complete_map(value: torch.Tensor) -> torch.Tensor:
    return F.interpolate(value, size=H_HW, mode="bilinear", align_corners=False)


def restrict_working(value: torch.Tensor) -> torch.Tensor:
    if tuple(value.shape) != (1, CHANNELS, *W_HW):
        raise ValueError(f"Phase-44 W result has invalid shape {tuple(value.shape)}.")
    return F.avg_pool2d(value, kernel_size=2, stride=2)


def region_noise(region: Region, device, dtype) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 22_000_003 + 1009 * region.index)
    value = torch.randn((1, CHANNELS, *W_HW), generator=generator, dtype=torch.float32)
    return value.to(device=device, dtype=dtype)


class StreamingAssembler:
    def __init__(self, planned: tuple[Region, ...]):
        self.planned = planned
        self.next_index = 0
        self.weights = OverlapAssembler()
        self.weighted_sum = torch.zeros((1, CHANNELS, *H_HW), dtype=torch.float32)
        self.coverage = torch.zeros((1, 1, *H_HW), dtype=torch.float32)

    def add(self, value: torch.Tensor, region: Region) -> None:
        if region != self.planned[self.next_index]:
            raise RuntimeError("Phase-44 region order changed.")
        if tuple(value.shape) != (1, CHANNELS, *F_HW):
            raise ValueError(f"Region {region.index} has invalid restricted shape.")
        value = value.detach().float().cpu()
        weight = self.weights.weight(region, self.planned, value.device)[None, None]
        self.weighted_sum[:, :, region.y:region.y2, region.x:region.x2] += value * weight
        self.coverage[:, :, region.y:region.y2, region.x:region.x2] += weight
        self.next_index += 1

    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.next_index != len(self.planned) or float(self.coverage.min()) <= 0:
            raise RuntimeError("Phase-44 assembly is incomplete.")
        return self.weighted_sum / self.coverage, self.coverage


class StreamingOverlapRMS:
    def __init__(self):
        self.active: list[tuple[Region, torch.Tensor]] = []
        self.square_sum = 0.0
        self.count = 0

    def add(self, value: torch.Tensor, region: Region) -> None:
        value = value.detach().float().cpu()
        self.active = [(prior, tensor) for prior, tensor in self.active if prior.y2 > region.y]
        for prior, tensor in self.active:
            y0, y1 = max(prior.y, region.y), min(prior.y2, region.y2)
            x0, x1 = max(prior.x, region.x), min(prior.x2, region.x2)
            if y0 < y1 and x0 < x1:
                left = tensor[..., y0-prior.y:y1-prior.y, x0-prior.x:x1-prior.x]
                right = value[..., y0-region.y:y1-region.y, x0-region.x:x1-region.x]
                delta = left - right
                self.square_sum += float(delta.square().sum())
                self.count += delta.numel()
        self.active.append((region, value))

    def finish(self) -> float:
        return math.sqrt(self.square_sum / self.count) if self.count else 0.0


class RecordingBasicGuider(Guider_Basic):
    def __init__(self, model):
        super().__init__(model)
        self.call_records: list[dict[str, Any]] = []

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        transformer = model_options.get("transformer_options", {})
        if transformer.get("rope_options") is not None:
            raise RuntimeError("Phase 44 forbids rope_options.")
        self.call_records.append({
            "shape": list(x.shape), "sigma": [float(v) for v in timestep.detach().float().cpu()],
            "image_tokens_hw": [x.shape[-2] // 2, x.shape[-1] // 2],
            "rope_options": None,
        })
        return super().predict_noise(x, timestep, model_options=model_options, seed=seed)


class ZImageAdapter:
    @staticmethod
    def validate_model(model, model_sampling) -> None:
        base = model.model
        diffusion = base.diffusion_model
        if type(base).__module__ != "comfy.model_base" or type(base).__name__ != "Lumina2":
            raise ValueError("Phase 44 requires native comfy.model_base.Lumina2.")
        if type(diffusion).__module__ != "comfy.ldm.lumina.model" or type(diffusion).__name__ != "NextDiT":
            raise ValueError("Phase 44 requires native latent-space NextDiT.")
        expected = {
            "in_channels": 16, "out_channels": 16, "patch_size": 2,
            "dim": 3840, "n_heads": 30, "time_scale": 1000.0,
            "pad_tokens_multiple": 32,
        }
        for name, required in expected.items():
            if getattr(diffusion, name, None) != required:
                raise ValueError(f"Phase-44 Z-Image profile requires {name}={required}.")
        if list(diffusion.axes_dims) != [32, 48, 48]:
            raise ValueError("Phase-44 Z-Image profile requires axes_dims=[32,48,48].")
        if diffusion.rope_embedder.theta != 256 or len(diffusion.layers) != 30:
            raise ValueError("Phase-44 Z-Image RoPE/layer profile mismatch.")
        if len(diffusion.noise_refiner) != 2 or len(diffusion.context_refiner) != 2:
            raise ValueError("Phase-44 Z-Image refiner profile mismatch.")
        mro = {item.__name__ for item in type(model_sampling).__mro__}
        if not {"ModelSamplingDiscreteFlow", "CONST"}.issubset(mro):
            raise ValueError("Phase 44 requires ModelSamplingDiscreteFlow + CONST.")
        for name, required in (("shift", 3.0), ("multiplier", 1.0), ("noise_scale", 1.0)):
            if float(getattr(model_sampling, name, float("nan"))) != required:
                raise ValueError(f"Phase-44 sampling requires {name}={required}.")
        options = model.model_options.get("transformer_options", {})
        forbidden = {name for name in ("patches", "patches_replace", "wrappers", "callbacks",
                                        "rope_options", "control") if options.get(name)}
        if forbidden or model.model_options.get("context_handler") is not None:
            raise ValueError(f"Phase 44 forbids model/position modifications: {sorted(forbidden)}.")

    @staticmethod
    def validate_prepared(guider: RecordingBasicGuider) -> None:
        if not isinstance(guider, Guider_Basic) or float(guider.cfg) != 1.0:
            raise ValueError("Phase 44 requires BasicGuider with CFG 1.")
        if set(guider.conds) != {"positive"} or len(guider.conds["positive"]) != 1:
            raise ValueError("Phase 44 requires exactly one positive conditioning branch.")
        for entry in guider.conds["positive"]:
            if {"area", "mask", "control", "gligen"}.intersection(entry) or entry.get("hooks") is not None:
                raise ValueError("Phase 44 forbids spatial, control, and hooked conditioning.")
            model_conds = entry.get("model_conds", {})
            forbidden = {"c_concat", "concat_latent_image", "concat_mask", "reference_latents",
                         "ref_latents", "ref_contexts", "siglip_feats", "clip_vision_outputs"}
            if forbidden.intersection(model_conds):
                raise ValueError("Phase 44 forbids reference/edit/Omni conditioning.")

    @staticmethod
    def predict(guider: RecordingBasicGuider, value: torch.Tensor, sigma: torch.Tensor,
                model_options: dict[str, Any]) -> torch.Tensor:
        if tuple(value.shape) != (1, CHANNELS, *W_HW):
            raise ValueError(f"Phase-44 native prediction shape mismatch: {tuple(value.shape)}.")
        options = model_options.copy()
        options["transformer_options"] = model_options.get("transformer_options", {}).copy()
        if options["transformer_options"].get("rope_options") is not None:
            raise ValueError("Phase 44 forbids rope_options.")
        output = guider(value, sigma.expand(1), model_options=options, seed=SEED)
        if tuple(output.shape) != tuple(value.shape) or not bool(output.isfinite().all()):
            raise RuntimeError("Phase-44 native prediction returned invalid output.")
        return output


def setup_local_guider(model, positive, device) -> RecordingBasicGuider:
    guider = RecordingBasicGuider(model)
    guider.set_conds(positive)
    guider.conds = {key: [item.copy() for item in values]
                    for key, values in guider.original_conds.items()}
    comfy.samplers.preprocess_conds_hooks(guider.conds)
    guider.inner_model, guider.conds, guider.loaded_models = comfy.sampler_helpers.prepare_sampling(
        model, (1, CHANNELS, *W_HW), guider.conds, guider.model_options)
    zeros = torch.zeros((1, CHANNELS, *W_HW), device=device)
    guider.conds = comfy.samplers.process_conds(
        guider.inner_model, zeros, guider.conds, device, zeros, None, SEED,
        latent_shapes=[zeros.shape])
    model.pre_run()
    ZImageAdapter.validate_prepared(guider)
    return guider


def generate_terminal_g(model, positive) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]], float]:
    guider = RecordingBasicGuider(model)
    guider.set_conds(positive)
    noise = torch.randn((1, CHANNELS, *G_HW), generator=torch.Generator("cpu").manual_seed(SEED))
    latent = torch.zeros_like(noise)
    sampler = comfy.samplers.sampler_object("res_multistep")
    sigma_tensor = torch.tensor(SIGMAS, dtype=torch.float32)
    callback_values: list[torch.Tensor] = []
    started = time.perf_counter()
    output = guider.sample(
        noise, latent, sampler, sigma_tensor, callback=lambda i, x0, x, total: callback_values.append(x0.detach().float().cpu()),
        disable_pbar=True, seed=SEED)
    wall = time.perf_counter() - started
    if len(callback_values) != 8:
        raise RuntimeError(f"Phase 44 expected eight G callbacks, got {len(callback_values)}.")
    if any(record["shape"] != [1, CHANNELS, *G_HW] for record in guider.call_records):
        raise RuntimeError("Phase 44 observed a non-native G model call.")
    return output.detach().float().cpu(), callback_values[-1], guider.call_records, wall


def run_once(model, positive, run_name: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    device = comfy.model_management.get_torch_device()
    torch.cuda.reset_peak_memory_stats(device)
    native_g, terminal_internal, g_calls, g_wall = generate_terminal_g(model, positive)
    plain_internal = complete_map(terminal_internal)
    planned = regions()
    guider = setup_local_guider(model, positive, device)
    model_sampling = guider.inner_model.model_sampling
    ZImageAdapter.validate_model(model, model_sampling)
    assembler = StreamingAssembler(planned)
    overlap = StreamingOverlapRMS()
    records, barriers = [], []
    sigma = torch.tensor(LOCAL_SIGMA, device=device, dtype=torch.float32)
    local_started = time.perf_counter()
    try:
        with torch.inference_mode():
            for region in planned:
                anchor, source_hw = bounded_transfer(terminal_internal, region)
                anchor = anchor.to(device=device, dtype=torch.float32)
                epsilon = region_noise(region, device, anchor.dtype)
                working = model_sampling.noise_scaling(sigma, epsilon, anchor, False)
                snapshot = working.clone()
                before_calls = len(guider.call_records)
                x0_w = ZImageAdapter.predict(guider, working, sigma, guider.model_options)
                if not torch.equal(working, snapshot) or len(guider.call_records) != before_calls + 1:
                    raise RuntimeError(f"Phase-44 region {region.index} mutation/call-count failure.")
                restricted = restrict_working(x0_w).detach().float().cpu()
                assembler.add(restricted, region)
                overlap.add(restricted, region)
                records.append({
                    "index": region.index,
                    "rect_yxhw": [region.y, region.x, region.height, region.width],
                    "source_hw": list(source_hw),
                    "anchor_hash": tensor_hash(anchor), "noise_hash": tensor_hash(epsilon),
                    "working_hash": tensor_hash(working), "prediction_hash": tensor_hash(x0_w),
                    "restricted_hash": tensor_hash(restricted),
                })
                del anchor, epsilon, working, snapshot, x0_w, restricted
                torch.cuda.synchronize(device)
                barriers.append(int(torch.cuda.memory_allocated(device)))
                print(f"{run_name}: region {region.index + 1}/25", flush=True)
        terminal_internal_h, coverage = assembler.finish()
        terminal_external = model.model.process_latent_out(terminal_internal_h).detach().float().cpu()
        plain_external = model.model.process_latent_out(plain_internal).detach().float().cpu()
    finally:
        model.cleanup()
        comfy.sampler_helpers.cleanup_models(guider.conds, guider.loaded_models)
        del guider.inner_model, guider.loaded_models
    local_wall = time.perf_counter() - local_started
    calls = g_calls + guider.call_records
    result = {"native": native_g, "plain": plain_external, "terminal_refine": terminal_external}
    telemetry = {
        "run_name": run_name, "sigmas": list(SIGMAS), "local_sigma": LOCAL_SIGMA,
        "G_hw": list(G_HW), "H_hw": list(H_HW), "F_hw": list(F_HW),
        "stride_hw": list(STRIDE_HW), "W_hw": list(W_HW), "regions": records,
        "model_calls": calls, "global_model_calls": len(g_calls),
        "local_model_calls": len(guider.call_records), "destination_model_calls": 0,
        "all_calls_native_128": all(item["shape"] == [1, CHANNELS, *W_HW] for item in calls),
        "all_tokens_native_64": all(item["image_tokens_hw"] == [64, 64] for item in calls),
        "all_rope_options_absent": all(item["rope_options"] is None for item in calls),
        "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max()),
        "overlap_rms": overlap.finish(), "terminal_internal_g_hash": tensor_hash(terminal_internal),
        "native_hash": tensor_hash(native_g), "plain_hash": tensor_hash(plain_external),
        "terminal_refine_hash": tensor_hash(terminal_external),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "completed_region_barriers": barriers,
        "completed_region_barrier_range_bytes": max(barriers) - min(barriers),
        "global_wall_seconds": g_wall, "local_wall_seconds": local_wall,
        "wall_seconds": g_wall + local_wall,
        "atomic_final_publication": True,
    }
    return result, telemetry


def decode(vae, latent: torch.Tensor, path: Path) -> str:
    try:
        pixels = vae.decode(latent)
    except comfy.model_management.OOM_EXCEPTION:
        comfy.model_management.soft_empty_cache()
        pixels = vae.decode_tiled(latent, tile_x=512, tile_y=512)
    pixels = pixels.clamp(0, 1)
    array = (pixels[0].detach().cpu().numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array).save(path)
    return hashlib.sha256(array.tobytes()).hexdigest()


def decoded_file_hash(path: Path) -> str:
    return hashlib.sha256(np.asarray(Image.open(path)).tobytes()).hexdigest()


def comparison_sheet(paths: tuple[tuple[str, Path], ...]) -> None:
    sheet = Image.new("RGB", (700 * len(paths), 740), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((680, 680), Image.Resampling.LANCZOS)
        x = index * 700 + (700 - image.width) // 2
        sheet.paste(image, (x, 48))
        draw.text((index * 700 + 10, 14), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def decoded_comparison_metrics(plain_path: Path, refined_path: Path) -> dict[str, float]:
    plain = np.asarray(Image.open(plain_path), dtype=np.float64) / 255.0
    refined = np.asarray(Image.open(refined_path), dtype=np.float64) / 255.0

    def gradient_rms(value: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.diff(value, axis=0) ** 2) + np.mean(np.diff(value, axis=1) ** 2)))

    plain_gradient = gradient_rms(plain)
    refined_gradient = gradient_rms(refined)
    return {
        "pixel_rms_C_vs_B": float(np.sqrt(np.mean((refined - plain) ** 2))),
        "plain_gradient_rms": plain_gradient,
        "terminal_refine_gradient_rms": refined_gradient,
        "terminal_refine_to_plain_gradient_ratio": refined_gradient / plain_gradient,
    }


def main() -> None:
    for path in (MODEL_PATH, TEXT_ENCODER_PATH, VAE_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    resumable = all((OUTPUT / name).is_file() for name in
                    ("native.pt", "plain.pt", "terminal_refine.pt", "primary.json", "repeat.json"))
    model = None
    if resumable:
        first = {name: torch.load(OUTPUT / f"{name}.pt", map_location="cpu", weights_only=True)
                 for name in ("native", "plain", "terminal_refine")}
        first_record = json.loads((OUTPUT / "primary.json").read_text(encoding="utf-8"))
        repeat_record = json.loads((OUTPUT / "repeat.json").read_text(encoding="utf-8"))
        repeat_bit_exact = all(
            first_record[f"{name}_hash"] == repeat_record[f"{name}_hash"]
            for name in ("native", "plain", "terminal_refine")
        )
    else:
        model = comfy.sd.load_diffusion_model(str(MODEL_PATH), model_options={})
        ZImageAdapter.validate_model(model, model.get_model_object("model_sampling"))
        clip = comfy.sd.load_clip([str(TEXT_ENCODER_PATH)], clip_type=comfy.sd.CLIPType.LUMINA2)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
        del clip
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        first, first_record = run_once(model, positive, "primary")
        for name, value in first.items():
            atomic_tensor(OUTPUT / f"{name}.pt", value)
        atomic_json(OUTPUT / "primary.json", first_record)
        comfy.model_management.soft_empty_cache()

        repeated, repeat_record = run_once(model, positive, "repeat")
        atomic_json(OUTPUT / "repeat.json", repeat_record)
        repeat_bit_exact = all(torch.equal(first[name], repeated[name]) for name in first)

    decoded_paths = {name: OUTPUT / f"{name}.png" for name in first}
    if all(path.is_file() for path in decoded_paths.values()):
        decoded = {name: decoded_file_hash(path) for name, path in decoded_paths.items()}
    else:
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(VAE_PATH), safe_load=True))
        decoded = {name: decode(vae, value, decoded_paths[name]) for name, value in first.items()}
    comparison_sheet((("A — Native 1024", OUTPUT / "native.png"),
                      ("B — Plain mapped Blueprint 2048", OUTPUT / "plain.png"),
                      ("C — Terminal Refine 2048", OUTPUT / "terminal_refine.png")))
    comparison_metrics = decoded_comparison_metrics(OUTPUT / "plain.png", OUTPUT / "terminal_refine.png")
    report = {
        "phase": 44, "prompt": PROMPT, "seed": SEED,
        "algorithm": "native_zimage_turbo_terminal_refine",
        "primary": first_record, "repeat": repeat_record,
        "independent_repeat_bit_exact": repeat_bit_exact,
        "independent_repeat_hash": repeat_record["terminal_refine_hash"],
        "decoded_hashes": decoded,
        "decoded_comparison_metrics": comparison_metrics,
        "integrity": {
            "fixed_geometry": len(regions()) == 25,
            "fixed_schedule": first_record["sigmas"] == list(SIGMAS),
            "local_sigma_exact": first_record["local_sigma"] == LOCAL_SIGMA,
            "model_calls_bounded": first_record["all_calls_native_128"],
            "token_geometry_bounded": first_record["all_tokens_native_64"],
            "native_zero_origin_rope": first_record["all_rope_options_absent"],
            "global_calls_exact": first_record["global_model_calls"] == 8,
            "local_calls_exact": first_record["local_model_calls"] == 25,
            "zero_H_sized_calls": first_record["destination_model_calls"] == 0,
            "coverage_complete": first_record["coverage_min"] > 0,
            "deterministic_repeat": repeat_bit_exact,
            "flat_completed_region_residency": first_record["completed_region_barrier_range_bytes"] == 0,
            "production_changed": False,
        },
        "semantic_review": {
            "status": "FAIL",
            "object_count": {"car": 1, "tree": 1, "house": 1},
            "left_center_right_placement_preserved": True,
            "continuous_horizon": True,
            "miniature_scene_tiling": False,
            "regional_complete_prompt_reinterpretation": False,
            "duplicate_objects": False,
            "separate_regional_horizons": False,
            "inconsistent_perspective": False,
            "cross_footprint_structural_disagreement": False,
            "credible_structural_detail_gain_over_plain": False,
            "observation": (
                "C preserves the single S3 scene, but does not add credible car, tree, house, or "
                "ground structure beyond B. It is slightly smoother overall; decoded gradient RMS "
                "falls to about 0.90 times the plain mapped control."
            ),
        },
        "decision": "REJECT_DIRECT_ZIMAGE_TERMINAL_REFINE_PORT_AND_STOP",
    }
    atomic_json(OUTPUT / "report.json", report)
    if model is not None:
        model.cleanup()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "integrity": report["integrity"]}, indent=2))


if __name__ == "__main__":
    main()
