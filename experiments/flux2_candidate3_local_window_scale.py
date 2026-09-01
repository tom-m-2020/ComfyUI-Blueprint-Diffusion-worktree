"""Phase 6g: Candidate-3 local crop-size/stride discriminator."""

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
import flux2_candidate3_overlap_necessity as phase6f

from blueprint_diffusion.regions import Region


OUTPUT = ROOT / "experiments" / "flux2_candidate3_local_window_scale_results"
CASES = (
    ("person_1024x2048", 2048, 1024, perf.PROMPT, 20260831),
    ("person_768x1536", 1536, 768, perf.PROMPT, 20260831),
)
VARIANTS = (
    ("A_CURRENT_32", 32, 24),
    ("B_MEDIUM_48", 48, 36),
    ("C_LARGE_64", 64, 48),
)


class WindowPlanner:
    def __init__(self, crop_size: int, stride: int):
        self.crop_size = crop_size
        self.stride = stride

    def starts(self, length: int) -> tuple[int, ...]:
        if length < self.crop_size:
            raise ValueError(
                f"crop {self.crop_size} cannot fit latent axis {length}"
            )
        final = length - self.crop_size
        starts = list(range(0, final + 1, self.stride))
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def plan(self, target_hw: tuple[int, int]) -> tuple[Region, ...]:
        ys = self.starts(target_hw[0])
        xs = self.starts(target_hw[1])
        return tuple(
            Region(index, y, x, self.crop_size, self.crop_size)
            for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
        )


def geometry_record(width: int, height: int, crop_size: int, stride: int) -> dict:
    h_shape = (height // 16, width // 16)
    regions = WindowPlanner(crop_size, stride).plan(h_shape)
    summed = len(regions) * crop_size * crop_size
    unique = math.prod(h_shape)
    return {
        "target_pixels_wh": [width, height],
        "H_shape": list(h_shape),
        "G_shape": [h_shape[0] // 4 * 3, h_shape[1] // 4 * 3],
        "crop_size": crop_size,
        "stride": stride,
        "nominal_overlap": crop_size - stride,
        "y_starts": sorted({region.y for region in regions}),
        "x_starts": sorted({region.x for region in regions}),
        "crop_grid": [
            len({region.y for region in regions}),
            len({region.x for region in regions}),
        ],
        "crop_count": len(regions),
        "unique_H_positions": unique,
        "summed_local_token_executions": summed,
        "redundant_token_executions": summed - unique,
        "redundancy_ratio": summed / unique,
        "local_forwards_per_interval": len(regions),
        "total_model_forwards": 4 * (1 + len(regions)),
        "regions": [vars(region) for region in regions],
        "actual_pairwise_overlaps": phase6f.actual_overlap_layout(regions),
    }


def run(model, positive, negative, noise, sigmas, crop_size, stride, seed):
    original_planner = phase6f.StridePlanner
    phase6f.StridePlanner = lambda ignored_stride: WindowPlanner(crop_size, stride)
    try:
        # Phase 6f used stride<32 only as its overlap-metric switch. Every 6g
        # variant has positive nominal overlap, so keep that diagnostic enabled;
        # the experiment planner above owns the actual stride.
        return phase6f.run(
            model, positive, negative, noise, sigmas, min(stride, 31), seed
        )
    finally:
        phase6f.StridePlanner = original_planner


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "cfg": 1.0,
            "steps": 4,
            "sampler": "Euler CONST flow",
            "variants": [
                {"name": name, "crop_size": crop, "stride": stride}
                for name, crop, stride in VARIANTS
            ],
            "timing": "one unmeasured current-window warm-up per geometry, then synchronized CUDA events/wall time",
            "memory": "PyTorch peak allocated/reserved",
            "production_changes": False,
        },
        "cases": {},
    }
    outputs = {}
    for name, width, height, prompt, seed in CASES:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        h_shape = (height // 16, width // 16)
        noise = torch.randn(
            (1, 128, *h_shape), generator=torch.Generator().manual_seed(seed)
        )
        sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
        perf.prepare_model_state(model)
        print(f"{name}: unmeasured 32/24 warm-up", flush=True)
        warm_output, _ = run(
            model, positive, negative, noise, sigmas, 32, 24, seed
        )
        del warm_output
        case = {
            "prompt": prompt,
            "seed": seed,
            "sigmas": sigmas.tolist(),
            "variants": {},
        }
        for variant, crop_size, stride in VARIANTS:
            if min(h_shape) < crop_size:
                case["variants"][variant] = {
                    "status": "SKIPPED_CROP_DOES_NOT_FIT",
                    "H_shape": list(h_shape),
                    "crop_size": crop_size,
                    "stride": stride,
                }
                print(
                    f"{name}: {variant} skipped; crop {crop_size} does not fit {h_shape}",
                    flush=True,
                )
                continue
            print(
                f"{name}: {variant} crop={crop_size} stride={stride}", flush=True
            )
            output, measurement = run(
                model, positive, negative, noise, sigmas,
                crop_size, stride, seed,
            )
            outputs[f"{name}_{variant}"] = output
            case["variants"][variant] = {
                "status": "SUCCESS",
                "geometry": geometry_record(
                    width, height, crop_size, stride
                ),
                "measurement": measurement,
            }
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
            "path": str(path),
            "finite": bool(torch.isfinite(pixels).all()),
            "shape": list(pixels.shape),
        }
    report["decoded_outputs"] = decoded
    (OUTPUT / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        case_name: {
            variant: (
                {"status": data["status"]}
                if data["status"] != "SUCCESS" else {
                    "crops": data["geometry"]["crop_count"],
                    "tokens": data["geometry"]["summed_local_token_executions"],
                    "wall_s": data["measurement"]["sampling_wall_seconds"],
                    "local_cuda_ms": data["measurement"]["cuda_category_totals_ms"]["local_forward"],
                    "peak_alloc_gib": data["measurement"]["peak_allocated_bytes"] / 1024**3,
                    "peak_reserved_gib": data["measurement"]["peak_reserved_bytes"] / 1024**3,
                }
            )
            for variant, data in case["variants"].items()
        }
        for case_name, case in report["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
