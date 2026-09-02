"""Phase 8b: fixed-canvas bounded Candidate-3 global-density discriminator."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_global_refresh_cadence as phase6i
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a

import blueprint_diffusion.sampling.euler as production_euler
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


OUTPUT = ROOT / "experiments" / "flux2_candidate3_bounded_global_density_results"
REPORT = OUTPUT / "report.json"
HEIGHT = 2048
WIDTH = 4096
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT
VARIANTS = {
    "A_CURRENT_4_TO_3": (4, 3),
    "B_BOUNDED_8_TO_4": (8, 4),
    "C_BOUNDED_8_TO_3": (8, 3),
}


class ExperimentalBlockDCTGeometry(BlockDCTGeometry):
    BLOCK_HIGH = 4
    BLOCK_GLOBAL = 3
    TOLERANCE = 3e-6

    def __init__(self, high_hw: tuple[int, int], block_high: int, block_global: int):
        self.BLOCK_HIGH = int(block_high)
        self.BLOCK_GLOBAL = int(block_global)
        self.HIGH_HW = tuple(int(value) for value in high_hw)
        if any(value <= 0 or value % self.BLOCK_HIGH for value in self.HIGH_HW):
            raise ValueError(
                f"Experimental {self.BLOCK_HIGH}->{self.BLOCK_GLOBAL} DCT requires "
                f"divisible positive H axes, got {self.HIGH_HW}."
            )
        self.GLOBAL_HW = tuple(
            value // self.BLOCK_HIGH * self.BLOCK_GLOBAL
            for value in self.HIGH_HW
        )

    def restrict(self, value: torch.Tensor) -> torch.Tensor:
        self.validate_high(value)
        original_dtype = value.dtype
        work = value.float()
        n, k = self.BLOCK_HIGH, self.BLOCK_GLOBAL
        blocks = self._grid_to_blocks(work, n)
        qn = self._matrix(n, work.device, work.dtype)
        qk = self._matrix(k, work.device, work.dtype)
        coefficients = torch.matmul(torch.matmul(qn, blocks), qn.T)
        retained = coefficients[..., :k, :k]
        global_blocks = (k / n) * torch.matmul(torch.matmul(qk.T, retained), qk)
        return self._blocks_to_grid(global_blocks).to(dtype=original_dtype)

    def prolong(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 4 or tuple(value.shape[-2:]) != self.GLOBAL_HW:
            raise ValueError(
                f"Experimental global grid must end in {self.GLOBAL_HW}, "
                f"got {tuple(value.shape)}."
            )
        original_dtype = value.dtype
        work = value.float()
        n, k = self.BLOCK_HIGH, self.BLOCK_GLOBAL
        blocks = self._grid_to_blocks(work, k)
        qk = self._matrix(k, work.device, work.dtype)
        qn = self._matrix(n, work.device, work.dtype)
        retained = torch.matmul(torch.matmul(qk, blocks), qk.T)
        coefficients = torch.zeros(
            (*retained.shape[:-2], n, n), device=work.device, dtype=work.dtype
        )
        coefficients[..., :k, :k] = retained
        high_blocks = (n / k) * torch.matmul(torch.matmul(qn.T, coefficients), qn)
        return self._blocks_to_grid(high_blocks).to(dtype=original_dtype)


def tensor_summary(value: torch.Tensor) -> dict:
    work = value.float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "variance": float(work.var(unbiased=False)),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


def preflight(block_high: int, block_global: int) -> dict:
    geometry = ExperimentalBlockDCTGeometry(HIGH_HW, block_high, block_global)
    generator = torch.Generator().manual_seed(8317 + block_high * 10 + block_global)
    global_value = torch.randn((1, 3, *geometry.GLOBAL_HW), generator=generator)
    constant_h = torch.full((1, 2, *HIGH_HW), 2.75)
    constant_g = torch.full((1, 2, *geometry.GLOBAL_HW), -1.5)
    return {
        "block_high": block_high,
        "block_global": block_global,
        "H_shape": list(HIGH_HW),
        "G_shape": list(geometry.GLOBAL_HW),
        "G_tokens": math.prod(geometry.GLOBAL_HW),
        "right_inverse_max_abs_float32": geometry.max_right_inverse_error(global_value),
        "constant_restriction_max_abs": float((geometry.restrict(constant_h) - 2.75).abs().max()),
        "constant_prolongation_max_abs": float((geometry.prolong(constant_g) + 1.5).abs().max()),
        "scale_restrict": block_global / block_high,
        "scale_prolong": block_high / block_global,
        "coordinate_scale_y": (HIGH_HW[0] - 1) / (geometry.GLOBAL_HW[0] - 1),
        "coordinate_scale_x": (HIGH_HW[1] - 1) / (geometry.GLOBAL_HW[1] - 1),
        "coordinate_first_yx": [0.0, 0.0],
        "coordinate_last_yx": [127.0, 255.0],
    }


class Capture:
    def __init__(self):
        self.values: dict[str, dict[int, torch.Tensor]] = {}

    def __call__(self, name: str, ordinal: int, value: torch.Tensor) -> None:
        if name in {"initial_H", "initial_G", "x0_G", "H_star", "accepted_G"}:
            self.values.setdefault(name, {})[ordinal] = value.detach().float().cpu()


class OverlapProbe:
    def __init__(self):
        self.records = []
        self.original = OverlapAssembler.assemble

    def wrapped(self, assembler, predictions, regions, target_hw):
        square_sum = 0.0
        count = 0
        for left_index, left in enumerate(regions):
            for right_index in range(left_index + 1, len(regions)):
                right = regions[right_index]
                y0, y1 = max(left.y, right.y), min(left.y2, right.y2)
                x0, x1 = max(left.x, right.x), min(left.x2, right.x2)
                if y0 >= y1 or x0 >= x1:
                    continue
                a = predictions[left_index][..., y0-left.y:y1-left.y, x0-left.x:x1-left.x]
                b = predictions[right_index][..., y0-right.y:y1-right.y, x0-right.x:x1-right.x]
                difference = (a.float() - b.float())
                square_sum += float(difference.square().sum())
                count += difference.numel()
        self.records.append({
            "overlap_pairwise_rms": math.sqrt(square_sum / count) if count else 0.0,
            "overlap_compared_values": count,
        })
        return self.original(assembler, predictions, regions, target_hw)


@contextlib.contextmanager
def scoped_geometry(block_high: int, block_global: int, overlap_probe: OverlapProbe):
    original_geometry = production_euler.BlockDCTGeometry
    original_validate = Flux2Adapter.validate_run
    original_assemble = OverlapAssembler.assemble

    def factory(high_hw):
        return ExperimentalBlockDCTGeometry(high_hw, block_high, block_global)

    def validate(adapter, **kwargs):
        high_shape = kwargs["high_shape"]
        expected = tuple(value // 4 * 3 for value in high_shape[-2:])
        forwarded = dict(kwargs)
        forwarded["global_shape"] = (*kwargs["global_shape"][:-2], *expected)
        return original_validate(adapter, **forwarded)

    production_euler.BlockDCTGeometry = factory
    Flux2Adapter.validate_run = validate
    def assemble(assembler, predictions, regions, target_hw):
        return overlap_probe.wrapped(assembler, predictions, regions, target_hw)

    OverlapAssembler.assemble = assemble
    try:
        yield
    finally:
        production_euler.BlockDCTGeometry = original_geometry
        Flux2Adapter.validate_run = original_validate
        OverlapAssembler.assemble = original_assemble


def decode_latents(latents: dict[str, torch.Tensor]) -> dict:
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    decoded = {}
    for name, latent in latents.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {
                "path": str(path),
                "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return decoded


def run_variant(name: str, block_high: int, block_global: int) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    algebra = preflight(block_high, block_global)
    if algebra["right_inverse_max_abs_float32"] > ExperimentalBlockDCTGeometry.TOLERANCE:
        raise RuntimeError(f"Preflight right inverse failed: {algebra}")

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn((1, 128, *HIGH_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(HIGH_HW)).float().clone()
    sigmas[0] = 1.0
    sampler_capture = Capture()
    sampler = BlueprintEulerSampler(capture=sampler_capture)
    timer = phase8a.AdapterForwardTimer()
    overlap = OverlapProbe()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with scoped_geometry(block_high, block_global, overlap), timer.patch(), torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    wall = time.perf_counter() - started
    forwards = timer.resolve()
    telemetry = list(sampler.last_telemetry)
    intervals = [item for item in telemetry if item["event"] == "accepted_interval"]
    initial_h = sampler_capture.values["initial_H"][-1]
    initial_g = sampler_capture.values["initial_G"][-1]
    projection = []
    for item in intervals:
        ordinal = item["ordinal"]
        h_star_rms = tensor_summary(sampler_capture.values["H_star"][ordinal])["rms"]
        projection.append({
            "ordinal": ordinal,
            "sigma": item["sigma"],
            "sigma_next": item["sigma_next"],
            "projection_rms": item["projection_rms"],
            "H_star_rms": h_star_rms,
            "projection_over_H_star": None if item["projection_rms"] is None else item["projection_rms"] / h_star_rms,
            "invariant_max_abs": item["invariant_max_abs"],
            **overlap.records[ordinal],
        })
    latents = {f"{name}_FINAL_H": output.detach().float().cpu()}
    for ordinal, value in sampler_capture.values.get("x0_G", {}).items():
        latents[f"{name}_STEP_{ordinal:02d}_GLOBAL_X0"] = value
    for ordinal, value in sampler_capture.values.get("accepted_G", {}).items():
        if ordinal < STEPS - 1:
            latents[f"{name}_STEP_{ordinal:02d}_ACCEPTED_G"] = value
    decoded = decode_latents(latents)
    regions = FixedCropPlanner().plan(HIGH_HW)
    local_tokens = sum(region.height * region.width for region in regions)
    return {
        "variant": name,
        "operator": algebra,
        "prompt": PROMPT,
        "seed": SEED,
        "sigmas": sigmas.tolist(),
        "initial_state": {
            "H0": tensor_summary(initial_h),
            "G0": tensor_summary(initial_g),
            "G0_over_H0_variance": float(initial_g.var(unbiased=False) / initial_h.var(unbiased=False)),
        },
        "work_integrity": {
            "crop_rectangles": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
            "crop_count": len(regions),
            "local_forwards": sum(item["category"] == "local" for item in forwards),
            "local_tokens_per_interval": local_tokens,
            "local_token_executions_generation": local_tokens * STEPS,
            "global_forwards": sum(item["category"] == "global" for item in forwards),
            "H0_sha256": hashlib.sha256(initial_h.numpy().tobytes()).hexdigest(),
            "terminal_only_final": [item["terminal_release"] for item in intervals] == [False, False, False, True],
            "all_nonterminal_invariants": all(item["invariant_max_abs"] is None or item["invariant_max_abs"] <= ExperimentalBlockDCTGeometry.TOLERANCE for item in intervals),
            "finite_final": bool(torch.isfinite(output).all()),
        },
        "performance": {
            "sampling_wall_seconds": wall,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "global_cuda_ms_by_interval": [item["cuda_ms"] for item in forwards if item["category"] == "global"],
            "global_cuda_ms_total": sum(item["cuda_ms"] for item in forwards if item["category"] == "global"),
            "local_cuda_ms_total": sum(item["cuda_ms"] for item in forwards if item["category"] == "local"),
        },
        "projection_and_overlap": projection,
        "telemetry": telemetry,
        "decoded": decoded,
    }


def merge_report(result: dict) -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {
        "configuration": {
            "target_pixels_hw": [HEIGHT, WIDTH],
            "H_shape": list(HIGH_HW),
            "steps": STEPS,
            "cfg": 1.0,
            "production_changes": False,
            "dense_reference": str(phase8a.OUTPUT / "2048x4096_bridge_dense_cold.png"),
        },
        "variants": {},
    }
    report["variants"][result["variant"]] = result
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")


def make_contact_sheet() -> None:
    names = tuple(VARIANTS)
    cell_w, cell_h, label_h = 800, 400, 30
    sheet = Image.new("RGB", (cell_w, (cell_h + label_h) * len(names)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(names):
        path = OUTPUT / f"{name}_FINAL_H.png"
        with Image.open(path) as source:
            preview = ImageOps.contain(source.convert("RGB"), (cell_w, cell_h), Image.Resampling.LANCZOS)
        y = row * (cell_h + label_h)
        sheet.paste(preview, ((cell_w - preview.width) // 2, y + label_h))
        draw.text((8, y + 8), name, fill="black")
    sheet.save(OUTPUT / "BOUNDED_GLOBAL_DENSITY_COMPARISON.png")

    global_sheet = Image.new(
        "RGB", (cell_w, (cell_h + label_h) * len(names) * 3), "white"
    )
    draw = ImageDraw.Draw(global_sheet)
    row = 0
    for name in names:
        for ordinal in range(3):
            path = OUTPUT / f"{name}_STEP_{ordinal:02d}_GLOBAL_X0.png"
            with Image.open(path) as source:
                preview = ImageOps.contain(
                    source.convert("RGB"), (cell_w, cell_h), Image.Resampling.LANCZOS
                )
            y = row * (cell_h + label_h)
            global_sheet.paste(preview, ((cell_w - preview.width) // 2, y + label_h))
            draw.text((8, y + 8), f"{name} global x0 interval {ordinal}", fill="black")
            row += 1
    global_sheet.save(OUTPUT / "GLOBAL_X0_COMPARISON.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        print(json.dumps({name: preflight(*operator) for name, operator in VARIANTS.items()}, indent=2))
        return
    if args.contact_sheet:
        make_contact_sheet()
        return
    if args.variant is None:
        raise SystemExit("Specify --variant, --preflight, or --contact-sheet")
    result = run_variant(args.variant, *VARIANTS[args.variant])
    merge_report(result)
    print(json.dumps({
        "variant": args.variant,
        "wall": result["performance"]["sampling_wall_seconds"],
        "global_ms": result["performance"]["global_cuda_ms_total"],
        "report": str(REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
