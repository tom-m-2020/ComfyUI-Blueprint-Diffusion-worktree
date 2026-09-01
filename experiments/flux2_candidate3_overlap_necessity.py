"""Phase 6f: fixed Candidate-3 lifecycle with stride 24/28/32 crop plans."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
import uuid
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d
import flux2_candidate3_performance_characterization as perf

from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.regions import Region, OverlapAssembler
from blueprint_diffusion.sampling.euler import validate_schedule
from blueprint_diffusion.state import BlueprintState


OUTPUT = ROOT / "experiments" / "flux2_candidate3_overlap_necessity_results"
PERSON_PROMPT = perf.PROMPT
CASES = (
    ("bridge_1024x512", 1024, 512, phase2.PROMPT, 20260829),
    ("person_1024x2048", 2048, 1024, PERSON_PROMPT, 20260831),
)
VARIANTS = (
    ("A_CURRENT", 24),
    ("B_REDUCED_OVERLAP", 28),
    ("C_NO_OVERLAP", 32),
)


class StridePlanner:
    def __init__(self, stride: int):
        self.stride = stride

    def starts(self, length: int) -> tuple[int, ...]:
        final = length - 32
        starts = list(range(0, final + 1, self.stride))
        if starts[-1] != final:
            starts.append(final)
        return tuple(starts)

    def plan(self, target_hw: tuple[int, int]) -> tuple[Region, ...]:
        ys = self.starts(target_hw[0])
        xs = self.starts(target_hw[1])
        return tuple(
            Region(index, y, x)
            for index, (y, x) in enumerate((y, x) for y in ys for x in xs)
        )


def adjacent_boundary_metric(value: torch.Tensor, regions: tuple[Region, ...]) -> dict:
    height, width = value.shape[-2:]
    vertical = sorted({region.x for region in regions if 0 < region.x < width})
    horizontal = sorted({region.y for region in regions if 0 < region.y < height})
    entries = []
    for x in vertical:
        delta = value[..., :, x].float() - value[..., :, x - 1].float()
        entries.append({
            "axis": "vertical", "coordinate": x,
            "rms": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()),
        })
    for y in horizontal:
        delta = value[..., y, :].float() - value[..., y - 1, :].float()
        entries.append({
            "axis": "horizontal", "coordinate": y,
            "rms": float(delta.square().mean().sqrt()),
            "max_abs": float(delta.abs().max()),
        })
    return {
        "metric": "adjacent one-token strip RMS across crop boundary; not overlap RMS",
        "entries": entries,
        "aggregate_rms": (
            float(math.sqrt(sum(item["rms"] ** 2 for item in entries) / len(entries)))
            if entries else 0.0
        ),
        "aggregate_max_abs": max((item["max_abs"] for item in entries), default=0.0),
    }


def actual_overlap_layout(regions: tuple[Region, ...]) -> list[dict]:
    result = []
    for i, left in enumerate(regions):
        for right in regions[i + 1:]:
            y1, y2 = max(left.y, right.y), min(left.y2, right.y2)
            x1, x2 = max(left.x, right.x), min(left.x2, right.x2)
            if y2 > y1 and x2 > x1:
                result.append({
                    "crop_pair": [left.index, right.index],
                    "overlap_yxhw": [y1, x1, y2 - y1, x2 - x1],
                    "tokens": (y2 - y1) * (x2 - x1),
                })
    return result


class OverlapProbeSampler(perf.MeasuredSamplerBase):
    def __init__(self, stride: int):
        super().__init__()
        self.stride = stride

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 6f requires empty-latent T2I.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        planner = StridePlanner(self.stride)
        geometry = BlockDCTGeometry(tuple(h.shape[-2:]))
        regions = planner.plan(tuple(h.shape[-2:]))
        assembler = OverlapAssembler()
        adapter = Flux2Adapter()
        with recorder.cuda("dct_qualification", None):
            right_inverse = geometry.qualify(device=h.device, dtype=h.dtype)
        with recorder.cuda("initial_D_H", None):
            g = geometry.restrict(h)
        state = BlueprintState(g, h, float(sigmas[0]), 0, uuid.uuid4().hex)
        intervals = []

        for ordinal in range(4):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            accepted_h = state.h.clone()
            accepted_g = state.g.clone()
            with recorder.cuda("global_forward", ordinal):
                x0_g = adapter.predict_global(
                    guider=model, g=state.g, sigma=sigma,
                    canvas=tuple(state.h.shape[-2:]),
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed", 0),
                )
            predictions = []
            for region in regions:
                view = state.h[:, :, region.y:region.y2, region.x:region.x2]
                if view.untyped_storage().data_ptr() != state.h.untyped_storage().data_ptr():
                    raise RuntimeError("Crop is not an immutable accepted-H view.")
                with recorder.cuda("local_forward", ordinal, f"crop_{region.index}"):
                    prediction = adapter.predict_region(
                        guider=model, h_view=view, sigma=sigma,
                        canvas=tuple(state.h.shape[-2:]), region=region,
                        model_options=extra_args["model_options"],
                        seed=extra_args.get("seed", 0),
                    )
                predictions.append(prediction)
            if not torch.equal(state.h, accepted_h) or not torch.equal(state.g, accepted_g):
                raise RuntimeError("Model call mutated accepted Candidate-3 state.")
            overlap = (
                phase2d.overlap_disagreement(predictions, list(regions))
                if self.stride < 32 else None
            )
            with recorder.cuda("overlap_assembly", ordinal):
                x0_h, coverage = assembler.assemble(
                    predictions, regions, tuple(state.h.shape[-2:])
                )
            boundary = adjacent_boundary_metric(x0_h, regions)
            with recorder.cuda("euler_and_coupling", ordinal):
                dt = sigma_next - sigma
                g_star = state.g + (state.g - x0_g) / sigma * dt
                h_star = state.h + (state.h - x0_h) / sigma * dt
                projection = geometry.prolong(g_star - geometry.restrict(h_star))
                terminal = float(sigma_next) == 0.0
                next_h = h_star if terminal else h_star + projection
                next_g = g_star
            invariant = None
            if not terminal:
                invariant = float((geometry.restrict(next_h).float() - next_g.float()).abs().max())
                if invariant > geometry.TOLERANCE:
                    raise RuntimeError(f"D(H_next) != G_next: {invariant}")
            if not all(bool(torch.isfinite(v).all()) for v in (x0_g, x0_h, next_h, next_g)):
                raise RuntimeError("Phase 6f produced nonfinite state.")
            state = BlueprintState(next_g, next_h, float(sigma_next), ordinal + 1, f"6f:{ordinal}")
            if callback is not None:
                callback(ordinal, x0_h, state.h, 4)
                self.preview_count += 1
            intervals.append({
                "ordinal": ordinal,
                "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "global_proposals": 1,
                "local_crop_count": len(regions),
                "immutable_H_verified": True,
                "assembled_x0_H_count": 1,
                "atomic_pair_acceptances": 1,
                "coverage_min": float(coverage.min()),
                "coverage_max": float(coverage.max()),
                "invariant_max_abs": invariant,
                "terminal_release": terminal,
                "projection_rms": float(projection.float().square().mean().sqrt()),
                "ordinary_overlap_disagreement": overlap,
                "assembled_prediction_boundary_strip": boundary,
            })
        final_boundary = adjacent_boundary_metric(state.h, regions)
        output = model_sampling.inverse_noise_scaling(sigmas[-1], state.h)
        self.finish(recorder, started)
        self.measurement.update({
            "integrity": {
                "status": "SUCCESS",
                "finite_final": bool(torch.isfinite(output).all()),
                "complete_positive_coverage": all(x["coverage_min"] > 0 for x in intervals),
                "nonterminal_invariants": all(
                    x["invariant_max_abs"] is None or x["invariant_max_abs"] <= geometry.TOLERANCE
                    for x in intervals
                ),
                "terminal_release_only_last": [x["terminal_release"] for x in intervals]
                == [False, False, False, True],
                "one_atomic_acceptance_per_interval": all(x["atomic_pair_acceptances"] == 1 for x in intervals),
                "expected_previews": self.preview_count == 4,
            },
            "right_inverse_max_abs": right_inverse,
            "intervals": intervals,
            "final_H_boundary_strip": final_boundary,
        })
        return output


def geometry_record(width: int, height: int, stride: int) -> dict:
    h_shape = (height // 16, width // 16)
    regions = StridePlanner(stride).plan(h_shape)
    summed = len(regions) * 1024
    unique = math.prod(h_shape)
    return {
        "target_pixels_wh": [width, height],
        "H_shape": list(h_shape),
        "G_shape": [h_shape[0] // 4 * 3, h_shape[1] // 4 * 3],
        "stride": stride,
        "nominal_overlap": 32 - stride,
        "y_starts": sorted({r.y for r in regions}),
        "x_starts": sorted({r.x for r in regions}),
        "crop_grid": [len({r.y for r in regions}), len({r.x for r in regions})],
        "crop_count": len(regions),
        "unique_H_positions": unique,
        "summed_local_token_executions": summed,
        "redundant_token_executions": summed - unique,
        "redundancy_ratio": summed / unique,
        "local_forwards_per_interval": len(regions),
        "total_model_forwards": 4 * (1 + len(regions)),
        "regions": [vars(r) for r in regions],
        "actual_pairwise_overlaps": actual_overlap_layout(regions),
    }


def run(model, positive, negative, noise, sigmas, stride, seed):
    sampler = OverlapProbeSampler(stride)
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=seed,
        )
    torch.cuda.synchronize()
    return output.detach().cpu(), sampler.measurement


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "cfg": 1.0, "steps": 4,
            "sampler": "Euler CONST flow", "crop_size": 32,
            "timing": "one unmeasured geometry warm-up, then synchronized CUDA events/wall time",
            "memory": "PyTorch peak allocated/reserved",
        },
        "cases": {},
    }
    outputs = {}
    for name, width, height, prompt, seed in CASES:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        h_shape = (height // 16, width // 16)
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(seed))
        sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
        perf.prepare_model_state(model)
        print(f"{name}: unmeasured stride-24 warm-up", flush=True)
        warm_output, _ = run(model, positive, negative, noise, sigmas, 24, seed)
        del warm_output
        case = {"prompt": prompt, "seed": seed, "sigmas": sigmas.tolist(), "variants": {}}
        for variant, stride in VARIANTS:
            print(f"{name}: {variant} stride={stride}", flush=True)
            output, measurement = run(model, positive, negative, noise, sigmas, stride, seed)
            outputs[f"{name}_{variant}"] = output
            case["variants"][variant] = {
                "geometry": geometry_record(width, height, stride),
                "measurement": measurement,
            }
        report["cases"][name] = case
        (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    del clip

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    for name, latent in outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        decoded[name] = {"path": str(path), "finite": bool(torch.isfinite(pixels).all()), "shape": list(pixels.shape)}
    report["decoded_outputs"] = decoded
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        case_name: {
            variant: {
                "crops": data["geometry"]["crop_count"],
                "tokens": data["geometry"]["summed_local_token_executions"],
                "wall_s": data["measurement"]["sampling_wall_seconds"],
                "local_cuda_ms": data["measurement"]["cuda_category_totals_ms"]["local_forward"],
                "peak_alloc_gib": data["measurement"]["peak_allocated_bytes"] / 1024**3,
                "peak_reserved_gib": data["measurement"]["peak_reserved_bytes"] / 1024**3,
            }
            for variant, data in case["variants"].items()
        }
        for case_name, case in report["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
