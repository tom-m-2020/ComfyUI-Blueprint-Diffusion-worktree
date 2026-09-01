"""Phase 6i: Candidate-3 stale global-model-estimate cadence discriminator."""

from __future__ import annotations

import json
import math
import sys
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_overlap_necessity as phase6f

from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.geometry.block_dct import BlockDCTGeometry
from blueprint_diffusion.regions import FixedCropPlanner, OverlapAssembler
from blueprint_diffusion.sampling.euler import validate_schedule
from blueprint_diffusion.state import BlueprintState


OUTPUT = ROOT / "experiments" / "flux2_candidate3_global_refresh_cadence_results"
POLICIES = {
    "A_BASELINE": (0, 1, 2, 3),
    "B_EARLY_3": (0, 1, 2),
    "C_EARLY_2": (0, 1),
    "D_ALTERNATING": (0, 2),
    "E_EARLY_ONLY": (0,),
}
CASES = (
    ("h64_person_car_tree", 2048, 1024, perf.PROMPT, 20260831),
    (
        "h64_bridge_train", 2048, 1024,
        "A wide cinematic photograph of one single long red suspension bridge "
        "stretching continuously from the far left edge to the far right edge "
        "over calm water, one yellow passenger train centered on the bridge, one "
        "white lighthouse at the far left, one dark stone tower at the far right, "
        "continuous bridge deck and cables, coherent perspective, no duplicate "
        "bridges, trains, lighthouses, or towers",
        20260901,
    ),
    (
        "h64_centered_astronaut", 2048, 1024,
        "A wide cinematic full-body photograph of exactly one enormous astronaut "
        "standing exactly in the center of the image, one small red rover far left, "
        "one antenna far right, continuous desert ground and horizon, no duplicate "
        "people or body parts",
        20260902,
    ),
    ("h48_person_car_tree", 1536, 768, perf.PROMPT, 20260831),
)


def tensor_stats(value: torch.Tensor) -> dict:
    data = value.float()
    return {
        "rms": float(data.square().mean().sqrt()),
        "mean": float(data.mean()),
        "max_abs": float(data.abs().max()),
    }


class CadenceSampler(perf.MeasuredSamplerBase):
    def __init__(self, fresh_ordinals: tuple[int, ...]):
        super().__init__()
        self.fresh_ordinals = fresh_ordinals

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 6i requires empty-latent T2I.")
        validate_schedule(sigmas)
        if 0 not in self.fresh_ordinals:
            raise ValueError("Phase 6i cadence must compute interval 0 freshly.")
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        planner = FixedCropPlanner()
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
        cached_x0_g = None
        cached_source_ordinal = None

        for ordinal in range(4):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            accepted_h = state.h.clone()
            accepted_g = state.g.clone()
            fresh = ordinal in self.fresh_ordinals
            if fresh:
                with recorder.cuda("global_forward", ordinal):
                    cached_x0_g = adapter.predict_global(
                        guider=model, g=state.g, sigma=sigma,
                        canvas=tuple(state.h.shape[-2:]),
                        model_options=extra_args["model_options"],
                        seed=extra_args.get("seed", 0),
                    )
                cached_source_ordinal = ordinal
            if cached_x0_g is None:
                raise RuntimeError("No causally available global estimate.")
            x0_g = cached_x0_g

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
            overlap = phase2d.overlap_disagreement(predictions, list(regions))
            with recorder.cuda("overlap_assembly", ordinal):
                x0_h, coverage = assembler.assemble(
                    predictions, regions, tuple(state.h.shape[-2:])
                )
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
                raise RuntimeError("Phase 6i produced nonfinite state.")
            state = BlueprintState(next_g, next_h, float(sigma_next), ordinal + 1, f"6i:{ordinal}")
            if callback is not None:
                callback(ordinal, x0_h, state.h, 4)
                self.preview_count += 1
            intervals.append({
                "ordinal": ordinal, "sigma": float(sigma),
                "sigma_next": float(sigma_next), "global_fresh": fresh,
                "global_estimate_source_ordinal": cached_source_ordinal,
                "global_estimate_age_intervals": ordinal - cached_source_ordinal,
                "global_proposals": 1 if fresh else 0,
                "local_crop_count": len(regions), "immutable_H_verified": True,
                "assembled_x0_H_count": 1, "atomic_pair_acceptances": 1,
                "coverage_min": float(coverage.min()),
                "coverage_max": float(coverage.max()),
                "invariant_max_abs": invariant, "terminal_release": terminal,
                "projection_rms": float(projection.float().square().mean().sqrt()),
                "proposed_H_rms": tensor_stats(h_star)["rms"],
                "accepted_G_rms": tensor_stats(next_g)["rms"],
                "global_estimate_rms": tensor_stats(x0_g)["rms"],
                "ordinary_overlap_disagreement": overlap,
                "assembled_prediction_boundary_strip": phase6f.adjacent_boundary_metric(x0_h, regions),
            })
        output = model_sampling.inverse_noise_scaling(sigmas[-1], state.h)
        self.finish(recorder, started)
        self.measurement.update({
            "integrity": {
                "status": "SUCCESS", "finite_final": bool(torch.isfinite(output).all()),
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
            "fresh_ordinals": list(self.fresh_ordinals),
            "global_forward_count": sum(x["global_proposals"] for x in intervals),
            "intervals": intervals,
            "final_H_boundary_strip": phase6f.adjacent_boundary_metric(state.h, regions),
        })
        return output


def run(model, positive, negative, noise, sigmas, fresh_ordinals, seed):
    sampler = CadenceSampler(fresh_ordinals)
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=seed,
        )
    torch.cuda.synchronize()
    return output.detach().cpu(), sampler.measurement


def compare_to_baseline(value: torch.Tensor, baseline: torch.Tensor) -> dict:
    delta = value.float() - baseline.float()
    low_value = F.avg_pool2d(value.float(), kernel_size=4, stride=4)
    low_baseline = F.avg_pool2d(baseline.float(), kernel_size=4, stride=4)
    low_delta = low_value - low_baseline
    return {
        "final_latent_rms": float(delta.square().mean().sqrt()),
        "final_latent_max_abs": float(delta.abs().max()),
        "low_frequency_rms": float(low_delta.square().mean().sqrt()),
        "low_frequency_definition": "4x4 nonoverlapping mean of final H",
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
            "reuse_definition": (
                "On a skipped interval, use the most recent causally available x0_G "
                "as the denoised estimate in the current-state/current-sigma Euler "
                "proposal. G remains persistent and updates every interval."
            ),
            "policies": {name: list(ordinals) for name, ordinals in POLICIES.items()},
        },
        "cases": {},
    }
    outputs = {}
    warmed = set()
    for name, width, height, prompt, seed in CASES:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        h_shape = (height // 16, width // 16)
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(seed))
        sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
        if h_shape not in warmed:
            perf.prepare_model_state(model)
            print(f"{h_shape}: unmeasured baseline warm-up", flush=True)
            warm, _ = run(model, positive, negative, noise, sigmas, POLICIES["A_BASELINE"], seed)
            del warm
            warmed.add(h_shape)
        case = {
            "target_pixels_wh": [width, height], "H_shape": list(h_shape),
            "prompt": prompt, "seed": seed, "sigmas": sigmas.tolist(),
            "variants": {},
        }
        baseline = None
        for policy, fresh_ordinals in POLICIES.items():
            print(f"{name}: {policy} fresh={fresh_ordinals}", flush=True)
            output, measurement = run(
                model, positive, negative, noise, sigmas, fresh_ordinals, seed
            )
            if baseline is None:
                baseline = output
            outputs[f"{name}_{policy}"] = output
            regions = FixedCropPlanner().plan(h_shape)
            case["variants"][policy] = {
                "geometry": {
                    "crop_count": len(regions),
                    "crop_size": regions[0].height,
                    "stride_x_first": regions[1].x - regions[0].x,
                    "summed_local_tokens": sum(r.height * r.width for r in regions),
                },
                "comparison_to_baseline": compare_to_baseline(output, baseline),
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
            policy: {
                "wall_s": value["measurement"]["sampling_wall_seconds"],
                "global_cuda_ms": value["measurement"]["cuda_category_totals_ms"].get("global_forward", 0.0),
                "global_forwards": value["measurement"]["global_forward_count"],
                "final_rms": value["comparison_to_baseline"]["final_latent_rms"],
                "low_rms": value["comparison_to_baseline"]["low_frequency_rms"],
                "projection": [round(x["projection_rms"], 6) for x in value["measurement"]["intervals"]],
            }
            for policy, value in case["variants"].items()
        }
        for case_name, case in report["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
