"""Phase 9d: native-working-coordinate terminal discriminator."""

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
import flux2_candidate2_one_eval_probe as candidate2
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_terminal_context as phase8d
import flux2_candidate3_performance_characterization as perf
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_working_coordinate_results"
REPORT = OUTPUT / "report.json"
H_HW = phase9b.H_HW
STEPS = phase9b.STEPS
SEED = phase9b.SEED
PROMPT = phase9b.PROMPT
REPRESENTATIVE_REGION = 7


def working_source_rope(region):
    """Map the complete H canvas into this crop's native W coordinate frame."""
    return {
        "scale_y": 2.0,
        "scale_x": 2.0,
        "shift_y": -2.0 * region.y,
        "shift_x": -2.0 * region.x,
    }


def stats(value):
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


class NativeWorkingCoordinateSampler(phase2.comfy.samplers.Sampler):
    def __init__(self):
        self.results = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 9d requires empty-latent T2I without masks.")
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
        regions = phase9b.DestinationPlanner().plan(H_HW)
        accepted_h_hash = phase9c.tensor_hash(state.h)
        accepted_g_hash = phase9c.tensor_hash(state.g)
        working = []
        coarse_errors = []
        for region in regions:
            h_view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(h_view, terminal_sigma, 3, region)
            error = phase9b.restrict2(value).float() - h_view.float()
            coarse_errors.append(float(error.abs().max()))
            working.append(value)
        if max(coarse_errors) > 1e-6:
            raise RuntimeError("Phase 9d W construction drifted from Phase 9b.")
        working_hashes = [phase9c.tensor_hash(value) for value in working]
        base_options = extra_args["model_options"]
        variants = {}
        predictions = {}

        for name in (
            "A_COMPRESSED_DESTINATION_COORDS",
            "B_NATIVE_LOCAL_COORDS",
            "C_NATIVE_LOCAL_COORDS_CONTEXT",
        ):
            gc.collect()
            phase2.comfy.model_management.soft_empty_cache()
            torch.cuda.synchronize()
            baseline_allocated = int(torch.cuda.memory_allocated())
            baseline_reserved = int(torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            total_started = time.perf_counter()
            source_cuda_ms = 0.0
            local_cuda_ms = 0.0
            source_wall = 0.0
            local_wall = 0.0
            cache_bytes = 0
            transfer_bytes = 0
            source_calls = 0
            x0_w_values = []
            restricted = []
            coordinate_records = []

            for region, value in zip(regions, working):
                probe = None
                if name == "C_NATIVE_LOCAL_COORDS_CONTEXT":
                    probe = phase8d.CPUOffloadedContextProbe()
                    source_options = phase9c.merge_options(
                        base_options,
                        working_source_rope(region),
                        probe,
                        "capture",
                        f"phase9d_H_source_crop_{region.index}",
                    )
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    wall_started = time.perf_counter()
                    start.record()
                    _ = model(
                        state.h,
                        terminal_sigma.expand(1),
                        model_options=source_options,
                        seed=SEED,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    source_cuda_ms += float(start.elapsed_time(end))
                    source_wall += time.perf_counter() - wall_started
                    source_calls += 1

                local_rope = (
                    {"shift_y": float(region.y), "shift_x": float(region.x),
                     "scale_y": 31.0 / 63.0, "scale_x": 31.0 / 63.0}
                    if name == "A_COMPRESSED_DESTINATION_COORDS"
                    else {}
                )
                local_options = phase9c.merge_options(
                    base_options,
                    local_rope,
                    probe,
                    "ordinary" if probe is None else "context",
                    f"phase9d_{name}_crop_{region.index}",
                )
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                wall_started = time.perf_counter()
                start.record()
                x0_w = model(
                    value,
                    terminal_sigma.expand(1),
                    model_options=local_options,
                    seed=SEED,
                )
                end.record()
                torch.cuda.synchronize()
                local_cuda_ms += float(start.elapsed_time(end))
                local_wall += time.perf_counter() - wall_started
                x0_w_values.append(x0_w)
                restricted.append(phase9b.restrict2(x0_w))

                record = {
                    "region": region.index,
                    "destination_origin": [region.y, region.x],
                    "local_rope": local_rope,
                }
                if probe is not None:
                    record["source_rope"] = working_source_rope(region)
                    record["local_ids"] = probe.position_records[
                        f"phase9d_{name}_crop_{region.index}"
                    ]
                    record["source_ids"] = probe.position_records[
                        f"phase9d_H_source_crop_{region.index}"
                    ]
                    cache_bytes += probe.capture_bytes
                    transfer_bytes += probe.transfer_bytes
                    probe.global_kv.clear()
                    probe.pending_normal_attention.clear()
                coordinate_records.append(record)

            torch.cuda.synchronize()
            assembled, coverage = coordinator.assembler.assemble(restricted, regions, H_HW)
            overlap = phase8d.overlap_metrics(
                [item.detach().float().cpu() for item in restricted], regions
            )
            if phase9c.tensor_hash(state.h) != accepted_h_hash or phase9c.tensor_hash(state.g) != accepted_g_hash:
                raise RuntimeError("Phase 9d evaluation mutated accepted state.")
            if [phase9c.tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError("Phase 9d evaluation mutated W.")
            variants[name] = {
                "source": None if name != "C_NATIVE_LOCAL_COORDS_CONTEXT" else "accepted_H",
                "source_tokens_per_call": 0 if name != "C_NATIVE_LOCAL_COORDS_CONTEXT" else math.prod(H_HW),
                "source_calls": source_calls,
                "context_blocks": 0 if name != "C_NATIVE_LOCAL_COORDS_CONTEXT" else 25,
                "local_generated_tokens": 4096,
                "local_calls": len(regions),
                "source_cuda_ms": source_cuda_ms,
                "local_cuda_ms": local_cuda_ms,
                "source_wall_seconds": source_wall,
                "local_wall_seconds": local_wall,
                "total_wall_seconds": time.perf_counter() - total_started,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "cpu_kv_cache_bytes_aggregate": cache_bytes,
                "cpu_to_gpu_transfer_bytes": transfer_bytes,
                "overlap": overlap,
                "coverage": [float(coverage.min()), float(coverage.max())],
                "assembled": stats(assembled),
                "representative_x0_W": stats(x0_w_values[REPRESENTATIVE_REGION]),
                "coordinate_records": coordinate_records,
            }
            predictions[name] = {
                "x0_w": [item.detach().float().cpu() for item in x0_w_values],
                "restricted": [item.detach().float().cpu() for item in restricted],
                "assembled": assembled.detach().float().cpu(),
            }
            del x0_w_values, restricted, assembled

        names = list(variants)
        pairwise = {}
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                pairwise[f"{left}_vs_{right}"] = {
                    "assembled": phase9c.difference(
                        predictions[left]["assembled"], predictions[right]["assembled"]
                    ),
                    "representative_x0_W": phase9c.difference(
                        predictions[left]["x0_w"][REPRESENTATIVE_REGION],
                        predictions[right]["x0_w"][REPRESENTATIVE_REGION],
                    ),
                }
        self.results = {
            "configuration": {
                "H": list(H_HW),
                "G": list(state.g.shape[-2:]),
                "terminal_sigma": float(terminal_sigma),
                "seed": SEED,
                "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "working_canvas": [64, 64],
                "accepted_H_hash": accepted_h_hash,
                "accepted_G_hash": accepted_g_hash,
                "working_hashes": working_hashes,
                "working_coarse_max_error": max(coarse_errors),
                "zero_state_updates": True,
            },
            "variants": variants,
            "pairwise": pairwise,
        }
        self.outputs = predictions
        return predictions["A_COMPRESSED_DESTINATION_COORDS"]["assembled"].to(state.h.device)


def decode_outputs(outputs):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    images = {}
    for name, values in outputs.items():
        selected = {
            "representative_x0_W": values["x0_w"][REPRESENTATIVE_REGION],
            "assembled_terminal_x0_H": values["assembled"],
            "final_terminal": values["assembled"],
        }
        for index, value in enumerate(values["restricted"]):
            selected[f"restricted_crop_{index:02d}"] = value
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
        with Image.open(records["final_terminal"]["path"]) as image:
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
    sampler = NativeWorkingCoordinateSampler()
    perf.prepare_model_state(model)
    with phase9b.scoped_variant("sigma_consistent", preterminal_probe), torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    if sampler.results is None or sampler.outputs is None:
        raise RuntimeError("Phase 9d sampler produced no result.")
    images = decode_outputs(sampler.outputs)
    comparison_sheet(images)
    sampler.results["images"] = images
    REPORT.write_text(json.dumps(sampler.results, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "overlap_rms": value["overlap"]["aggregate_rms"],
            "source_cuda_ms": value["source_cuda_ms"],
            "local_cuda_ms": value["local_cuda_ms"],
            "peak_allocated_bytes": value["peak_allocated_bytes"],
            "peak_reserved_bytes": value["peak_reserved_bytes"],
        }
        for name, value in sampler.results["variants"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
