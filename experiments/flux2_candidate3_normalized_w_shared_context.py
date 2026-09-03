"""Phase 11: reconstructed native W with shared transformer context."""

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
from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_normalized_w_shared_context_results"
REPORT = OUTPUT / "report.json"
H_HW = phase9b.H_HW
STEPS = phase9b.STEPS
SEED = phase9b.SEED
PROMPT = phase9b.PROMPT
REPRESENTATIVE_REGION = 7
VARIANTS = (
    ("A_LOCAL_ONLY", None),
    ("B_FIXED_G_CONTEXT", "G"),
    ("C_FULL_H_CONTEXT_ORACLE", "H"),
)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def summary(value):
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


class ContextRuntime:
    def __init__(self, outer_probe):
        self.outer_probe = outer_probe
        self.ordinal = 0
        self.context_probe = None
        self.x0_w = {}
        self.w_hashes = {}
        self.representative = {}

    def start(self, ordinal, context_probe):
        self.ordinal = ordinal
        self.context_probe = context_probe
        self.x0_w = {}
        self.w_hashes = {}

    def predict_region(self, adapter, **kwargs):
        region = kwargs["region"]
        working = phase9c.make_working(
            kwargs["h_view"], kwargs["sigma"], self.ordinal, region
        )
        before_hash = tensor_hash(working)
        options = phase9c.merge_options(
            kwargs["model_options"],
            {},
            self.context_probe,
            "ordinary" if self.context_probe is None else "context",
            f"phase11_crop_{self.ordinal}_{region.index}",
        )
        x0_w = self.outer_probe.timed(
            "local",
            64 * 64,
            lambda: kwargs["guider"](
                working,
                kwargs["sigma"].expand(1),
                model_options=options,
                seed=kwargs["seed"],
            ),
        )
        if tensor_hash(working) != before_hash:
            raise RuntimeError(f"Local call mutated W at {self.ordinal}/{region.index}.")
        self.x0_w[region.index] = x0_w
        self.w_hashes[region.index] = before_hash
        if region.index == REPRESENTATIVE_REGION:
            self.representative[self.ordinal] = {
                "W": working.detach().float().cpu(),
                "x0_W": x0_w.detach().float().cpu(),
                "restricted_x0": phase9b.restrict2(x0_w).detach().float().cpu(),
            }
        return phase9b.restrict2(x0_w)


class SharedContextSampler(phase2.comfy.samplers.Sampler):
    def __init__(self, context_source, outer_probe):
        self.context_source = context_source
        self.outer_probe = outer_probe
        self.results = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 11 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        regions = phase9b.DestinationPlanner().plan(H_HW)
        runtime = ContextRuntime(self.outer_probe)
        original_region = Flux2Adapter.predict_region

        def patched(adapter, **kwargs):
            return runtime.predict_region(adapter, **kwargs)

        Flux2Adapter.predict_region = patched
        trajectory = []
        images = {
            "accepted_H": {0: state.h.detach().float().cpu()},
            "assembled_x0_H": {},
            "x0_W": {},
        }
        context_generations = []
        try:
            for ordinal in range(len(sigmas) - 1):
                sigma = sigmas[ordinal]
                sigma_next = sigmas[ordinal + 1]
                h_hash = tensor_hash(state.h)
                g_hash = tensor_hash(state.g)
                context_probe = None
                source_record = None
                if self.context_source is not None:
                    context_probe = phase8d.CPUOffloadedContextProbe()
                    if self.context_source == "G":
                        source = state.g
                        rope = {
                            "scale_y": (H_HW[0] - 1.0) / (state.g.shape[-2] - 1.0),
                            "scale_x": (H_HW[1] - 1.0) / (state.g.shape[-1] - 1.0),
                        }
                    else:
                        source = state.h
                        rope = {}
                    source_hash = tensor_hash(source)
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    wall_started = time.perf_counter()
                    start.record()
                    _ = model(
                        source,
                        sigma.expand(1),
                        model_options=phase9c.merge_options(
                            extra_args["model_options"], rope, context_probe,
                            "capture", f"phase11_source_{self.context_source}_{ordinal}",
                        ),
                        seed=SEED,
                    )
                    end.record()
                    torch.cuda.synchronize()
                    if tensor_hash(source) != source_hash:
                        raise RuntimeError(f"Context source mutated at interval {ordinal}.")
                    source_record = {
                        "ordinal": ordinal,
                        "sigma": float(sigma),
                        "source": self.context_source,
                        "source_shape": list(source.shape),
                        "source_tokens": int(source.shape[-2] * source.shape[-1]),
                        "source_hash": source_hash,
                        "accepted_H_hash": h_hash,
                        "accepted_G_hash": g_hash,
                        "rope": rope,
                        "cuda_ms": float(start.elapsed_time(end)),
                        "wall_seconds": time.perf_counter() - wall_started,
                        "probe_python_id": id(context_probe),
                    }

                runtime.start(ordinal, context_probe)
                candidate_state, x0_h = coordinator.evaluate(
                    guider=model,
                    state=state,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    model_options=extra_args["model_options"],
                    seed=SEED,
                )
                if set(runtime.x0_w) != set(range(len(regions))):
                    raise RuntimeError(f"Incomplete local predictions at interval {ordinal}.")
                if tensor_hash(state.h) != h_hash or tensor_hash(state.g) != g_hash:
                    raise RuntimeError("Context/local evaluation mutated accepted state.")
                interval = coordinator.telemetry[-1]
                predictions = [phase9b.restrict2(runtime.x0_w[index]) for index in range(len(regions))]
                overlap = phase9b.Probe.overlap(predictions, regions)
                crop_summaries = [summary(item) for item in predictions]
                context_record_count = 0
                capture_bytes = 0
                transfer_bytes = 0
                if context_probe is not None:
                    context_probe.assert_complete()
                    context_record_count = len(context_probe.context_records)
                    capture_bytes = context_probe.capture_bytes
                    transfer_bytes = context_probe.transfer_bytes
                    source_record.update({
                        "capture_blocks": len(context_probe.capture_records),
                        "context_consumptions": context_record_count,
                        "captured_kv_bytes": capture_bytes,
                        "cpu_to_gpu_transfer_bytes": transfer_bytes,
                    })
                    context_generations.append(source_record)
                    context_probe.global_kv.clear()
                    context_probe.pending_normal_attention.clear()

                # Atomic Candidate-3 acceptance occurs only after the complete
                # source and all fifteen local calls have validated.
                state = candidate_state
                images["accepted_H"][ordinal + 1] = state.h.detach().float().cpu()
                images["assembled_x0_H"][ordinal] = x0_h.detach().float().cpu()
                images["x0_W"][ordinal] = [
                    runtime.x0_w[index].detach().float().cpu() for index in range(len(regions))
                ]
                trajectory.append({
                    "ordinal": ordinal,
                    "sigma": float(sigma),
                    "sigma_next": float(sigma_next),
                    "input_H_hash": h_hash,
                    "input_G_hash": g_hash,
                    "accepted_H_hash": tensor_hash(state.h),
                    "accepted_G_hash": tensor_hash(state.g),
                    "W_hashes": [runtime.w_hashes[index] for index in range(len(regions))],
                    "prediction_overlap_rms": overlap,
                    "crop_predictions": crop_summaries,
                    "assembled_x0_H": summary(x0_h),
                    "projection_rms": interval["projection_rms"],
                    "invariant_max_abs": interval["invariant_max_abs"],
                    "coverage": [interval["coverage_min"], interval["coverage_max"]],
                    "context_probe_python_id": None if context_probe is None else id(context_probe),
                    "context_consumptions": context_record_count,
                    "context_capture_bytes": capture_bytes,
                    "context_transfer_bytes": transfer_bytes,
                })
        finally:
            Flux2Adapter.predict_region = original_region

        self.results = {
            "configuration": {
                "H": list(H_HW),
                "G": list(state.g.shape[-2:]),
                "context_source": self.context_source,
                "destination_crop": [32, 32],
                "stride": 24,
                "working_canvas": [64, 64],
                "working_coordinates": "native local unit grid 0..63",
                "sigmas": sigmas.tolist(),
                "seed": SEED,
                "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
            },
            "trajectory": trajectory,
            "context_generations": context_generations,
            "work": {
                "candidate3_global_forwards": sum(x["kind"] == "global" for x in self.outer_probe.calls),
                "context_source_forwards": len(context_generations),
                "local_forwards": sum(x["kind"] == "local" for x in self.outer_probe.calls),
                "local_token_executions": sum(x["tokens"] for x in self.outer_probe.calls if x["kind"] == "local"),
                "candidate3_global_cuda_ms": sum(x["cuda_ms"] for x in self.outer_probe.calls if x["kind"] == "global"),
                "local_cuda_ms": sum(x["cuda_ms"] for x in self.outer_probe.calls if x["kind"] == "local"),
                "context_source_cuda_ms": sum(x["cuda_ms"] for x in context_generations),
                "context_kv_tokens_per_block": (
                    0 if self.context_source is None else
                    math.prod(state.g.shape[-2:]) if self.context_source == "G" else math.prod(H_HW)
                ),
                "context_blocks_per_local": 0 if self.context_source is None else 25,
                "context_cache_bytes_per_interval": [x["captured_kv_bytes"] for x in context_generations],
                "context_transfer_bytes_total": sum(x["cpu_to_gpu_transfer_bytes"] for x in context_generations),
            },
            "final_H": summary(state.h),
            "final_H_hash": tensor_hash(state.h),
            "integrity": {
                "fresh_context_probe_ids": [x["probe_python_id"] for x in context_generations],
                "fresh_context_every_interval": (
                    self.context_source is None or
                    len({x["probe_python_id"] for x in context_generations}) == len(sigmas) - 1
                ),
                "same_sigma_provenance": all(
                    x["ordinal"] == i and x["sigma"] == float(sigmas[i])
                    for i, x in enumerate(context_generations)
                ),
                "all_context_consumed_25x15": all(
                    x["context_consumptions"] == 25 * len(regions) for x in context_generations
                ),
                "finite": bool(state.h.isfinite().all()),
            },
        }
        self.outputs = {"trajectory": images, "representative": runtime.representative}
        return state.h


def decode_outputs(name, outputs):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    values = {}
    for ordinal, value in outputs["trajectory"]["accepted_H"].items():
        values[f"accepted_H_{ordinal}"] = value
    for ordinal, value in outputs["trajectory"]["assembled_x0_H"].items():
        values[f"assembled_x0_H_{ordinal}"] = value
    for ordinal, record in outputs["representative"].items():
        for key, value in record.items():
            values[f"region7_step{ordinal}_{key}"] = value
    images = {}
    for label, latent in values.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}_{label}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images[label] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return images


def run_variant(name, source):
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
    outer_probe = phase9b.Probe(sigmas)
    sampler = SharedContextSampler(source, outer_probe)
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with phase9b.scoped_variant("direct", outer_probe), torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    sampler.results["runtime"] = {
        "sampling_wall_seconds": time.perf_counter() - started,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    sampler.results["images"] = decode_outputs(name, sampler.outputs)
    (OUTPUT / f"{name}.json").write_text(json.dumps(sampler.results, indent=2), encoding="utf-8")
    return sampler.results, sampler.outputs


def difference(left, right):
    return phase9c.difference(left, right)


def pairwise_outputs(left, right):
    result = {"accepted_H": {}, "assembled_x0_H": {}, "representative_x0_W": {}, "crop_prediction_rms": {}}
    for key in sorted(left["trajectory"]["accepted_H"]):
        result["accepted_H"][str(key)] = difference(
            left["trajectory"]["accepted_H"][key], right["trajectory"]["accepted_H"][key]
        )
    for key in sorted(left["trajectory"]["assembled_x0_H"]):
        result["assembled_x0_H"][str(key)] = difference(
            left["trajectory"]["assembled_x0_H"][key], right["trajectory"]["assembled_x0_H"][key]
        )
        crop_values = [
            difference(a, b)
            for a, b in zip(left["trajectory"]["x0_W"][key], right["trajectory"]["x0_W"][key])
        ]
        if key in left["representative"] and key in right["representative"]:
            result["representative_x0_W"][str(key)] = difference(
                left["representative"][key]["x0_W"], right["representative"][key]["x0_W"]
            )
        result["crop_prediction_rms"][str(key)] = crop_values
    return result


def comparison_sheet(results):
    sheet = Image.new("RGB", (768, 420 * len(results)), "white")
    for index, (name, result) in enumerate(results.items()):
        with Image.open(result["images"]["accepted_H_4"]["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
        sheet.paste(thumb, (0, index * 420 + 36))
        ImageDraw.Draw(sheet).text((10, index * 420 + 10), name, fill="black")
    sheet.save(OUTPUT / "FINAL_COMPARISON.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    outputs = {}
    for name, source in VARIANTS:
        result, tensors = run_variant(name, source)
        results[name] = result
        outputs[name] = tensors
    pairwise = {}
    names = list(results)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            pairwise[f"{left}_vs_{right}"] = pairwise_outputs(outputs[left], outputs[right])
    comparison_sheet(results)
    REPORT.write_text(json.dumps({"variants": results, "pairwise": pairwise}, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "overlap": [x["prediction_overlap_rms"] for x in result["trajectory"]],
            "source_tokens": result["work"]["context_kv_tokens_per_block"],
            "source_cuda_ms": result["work"]["context_source_cuda_ms"],
            "local_cuda_ms": result["work"]["local_cuda_ms"],
            "wall_seconds": result["runtime"]["sampling_wall_seconds"],
            "peak_allocated_bytes": result["runtime"]["peak_allocated_bytes"],
            "peak_reserved_bytes": result["runtime"]["peak_reserved_bytes"],
        }
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
