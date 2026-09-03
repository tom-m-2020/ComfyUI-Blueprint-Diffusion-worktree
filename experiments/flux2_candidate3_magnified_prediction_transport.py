"""Phase 9e: exact CONST-flow magnified-prediction transport discriminator."""

from __future__ import annotations

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
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_terminal_context as phase8d
import flux2_candidate3_performance_characterization as perf
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_magnified_prediction_transport_results"
REPORT = OUTPUT / "report.json"
H_HW = phase9b.H_HW
STEPS = phase9b.STEPS
SEED = phase9b.SEED
PROMPT = phase9b.PROMPT
REPRESENTATIVE_REGION = 7


def summary(value):
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


class PredictionTransportSampler(phase2.comfy.samplers.Sampler):
    def __init__(self):
        self.results = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 9e requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model,
                state=state,
                sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"],
                seed=SEED,
            )

        terminal_sigma = sigmas[3]
        sigma = terminal_sigma.to(device=state.h.device, dtype=torch.float32)
        regions = phase9b.DestinationPlanner().plan(H_HW)
        accepted_h_hash = phase9c.tensor_hash(state.h)
        accepted_g_hash = phase9c.tensor_hash(state.g)
        working = []
        working_errors = []
        for region in regions:
            h_crop = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(h_crop, terminal_sigma, 3, region)
            working_errors.append(float((phase9b.restrict2(value).float() - h_crop.float()).abs().max()))
            working.append(value)
        if max(working_errors) > 1e-6:
            raise RuntimeError("Phase 9e W construction drifted from Phase 9b.")
        working_hashes = [phase9c.tensor_hash(value) for value in working]

        gc.collect()
        phase2.comfy.model_management.soft_empty_cache()
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        x0_w_values = []
        for value in working:
            x0_w_values.append(model(
                value,
                terminal_sigma.expand(1),
                model_options=phase9c.merge_options(extra_args["model_options"], {}),
                seed=SEED,
            ))
        end_event.record()
        torch.cuda.synchronize()
        model_cuda_ms = float(start_event.elapsed_time(end_event))
        model_wall_seconds = time.perf_counter() - started

        names = (
            "A_X0_MEAN",
            "B_VELOCITY_MEAN",
            "C_DELTA_TRANSPORT",
            "D_ORACLE_RENOISE_RESTRICT",
        )
        transported = {name: [] for name in names}
        per_crop = []
        for region, value, x0_w in zip(regions, working, x0_w_values):
            h_crop = state.h[:, :, region.y:region.y2, region.x:region.x2].float()
            w = value.float()
            x0 = x0_w.float()

            # Native CONST: x0 = x_sigma - sigma * model_output.
            velocity_w = (w - x0) / sigma
            a = phase9b.restrict2(x0)
            b = h_crop - sigma * phase9b.restrict2(velocity_w)
            delta_w = x0 - w
            c = h_crop + phase9b.restrict2(delta_w)
            renoised_w = x0 + sigma * velocity_w
            same_sigma_crop = phase9b.restrict2(renoised_w)
            d = same_sigma_crop - sigma * phase9b.restrict2(velocity_w)

            values = {names[0]: a, names[1]: b, names[2]: c, names[3]: d}
            for name, prediction in values.items():
                transported[name].append(prediction)
            per_crop.append({
                "region": region.index,
                "W_to_H": phase9c.difference(phase9b.restrict2(w), h_crop),
                "renoised_W_to_W": phase9c.difference(renoised_w, w),
                "restricted_renoised_to_H": phase9c.difference(same_sigma_crop, h_crop),
                "velocity_W": summary(velocity_w),
                "delta_W": summary(delta_w),
                "A_vs_B": phase9c.difference(a, b),
                "A_vs_C": phase9c.difference(a, c),
                "A_vs_D": phase9c.difference(a, d),
            })

        assembled = {}
        variants = {}
        for name in names:
            result, coverage = coordinator.assembler.assemble(transported[name], regions, H_HW)
            assembled[name] = result.detach().float().cpu()
            variants[name] = {
                "destination_predictions": [summary(item) for item in transported[name]],
                "assembled": summary(result),
                "overlap": phase8d.overlap_metrics(
                    [item.detach().float().cpu() for item in transported[name]], regions
                ),
                "coverage": [float(coverage.min()), float(coverage.max())],
            }

        pairwise = {}
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                pairwise[f"{left}_vs_{right}"] = {
                    "assembled": phase9c.difference(assembled[left], assembled[right]),
                    "representative_crop": phase9c.difference(
                        transported[left][REPRESENTATIVE_REGION],
                        transported[right][REPRESENTATIVE_REGION],
                    ),
                }
        if phase9c.tensor_hash(state.h) != accepted_h_hash or phase9c.tensor_hash(state.g) != accepted_g_hash:
            raise RuntimeError("Phase 9e transport mutated accepted state.")
        if [phase9c.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Phase 9e transport mutated W.")

        self.results = {
            "configuration": {
                "H": list(H_HW),
                "G": list(state.g.shape[-2:]),
                "terminal_sigma": float(terminal_sigma),
                "seed": SEED,
                "coordinate_convention": "native local 64x64; unit coordinates 0..63",
                "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "accepted_H_hash": accepted_h_hash,
                "accepted_G_hash": accepted_g_hash,
                "working_hashes": working_hashes,
                "working_coarse_max_error": max(working_errors),
                "model_calls": len(regions),
                "model_cuda_ms": model_cuda_ms,
                "model_wall_seconds": model_wall_seconds,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "zero_state_updates": True,
            },
            "algebra": {
                "CONST": "x0_W = W - sigma*v_W",
                "A": "D(x0_W)",
                "B": "H_crop - sigma*D(v_W) = D(W) - sigma*D(v_W) = D(x0_W)",
                "C": "H_crop + D(x0_W-W) = D(W) + D(x0_W-W) = D(x0_W)",
                "D": "D(x0_W+sigma*v_W) - sigma*D(v_W) = D(W)-sigma*D(v_W) = D(x0_W)",
                "reason": "D is linear and D(W)=H_crop",
            },
            "per_crop_consistency": per_crop,
            "variants": variants,
            "pairwise": pairwise,
        }
        self.outputs = {
            name: {
                "x0_w": [item.detach().float().cpu() for item in x0_w_values],
                "restricted": [item.detach().float().cpu() for item in transported[name]],
                "assembled": assembled[name],
            }
            for name in names
        }
        return assembled[names[0]].to(state.h.device)


def decode_outputs(outputs):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    images = {}
    for name, values in outputs.items():
        selected = {
            "representative_x0_W": values["x0_w"][REPRESENTATIVE_REGION],
            "assembled_terminal_x0_H": values["assembled"],
        }
        for index, value in enumerate(values["restricted"]):
            selected[f"transported_crop_{index:02d}"] = value
        images[name] = {}
        for label, latent in selected.items():
            with torch.inference_mode():
                pixels = vae.decode(latent).cpu()
            path = OUTPUT / f"{name}_{label}.png"
            phase2.save_pixels(pixels, path)
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                images[name][label] = {
                    "path": str(path),
                    "dimensions_wh": list(rgb.size),
                    "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
    return images


def comparison_sheet(images):
    sheet = Image.new("RGB", (768, 420 * len(images)), "white")
    for index, (name, records) in enumerate(images.items()):
        with Image.open(records["assembled_terminal_x0_H"]["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
        sheet.paste(thumb, (0, index * 420 + 36))
        ImageDraw.Draw(sheet).text((10, index * 420 + 10), name, fill="black")
    sheet.save(OUTPUT / "FINAL_COMPARISON.png")


def main():
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
    noise = torch.randn((1, 128, *H_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(H_HW)).float().clone()
    sigmas[0], sigmas[-1] = 1.0, 0.0
    preterminal_probe = phase9b.Probe(sigmas)
    sampler = PredictionTransportSampler()
    perf.prepare_model_state(model)
    with phase9b.scoped_variant("sigma_consistent", preterminal_probe), torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    if sampler.results is None or sampler.outputs is None:
        raise RuntimeError("Phase 9e sampler produced no result.")
    images = decode_outputs(sampler.outputs)
    comparison_sheet(images)
    sampler.results["images"] = images
    REPORT.write_text(json.dumps(sampler.results, indent=2), encoding="utf-8")
    print(json.dumps({
        "working_coarse_max_error": sampler.results["configuration"]["working_coarse_max_error"],
        "model_cuda_ms": sampler.results["configuration"]["model_cuda_ms"],
        "variants": {
            name: {
                "overlap_rms": value["overlap"]["aggregate_rms"],
                "assembled_rms": value["assembled"]["rms"],
            }
            for name, value in sampler.results["variants"].items()
        },
        "pairwise": sampler.results["pairwise"],
    }, indent=2))


if __name__ == "__main__":
    main()
