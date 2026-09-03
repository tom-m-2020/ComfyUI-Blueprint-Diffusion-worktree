"""Phase 8h: exact block-major full-context streaming feasibility probe."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import threading
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_one_eval_probe as candidate2
import flux2_candidate2_four_step_trajectory as candidate2_trajectory
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a
import flux2_candidate3_terminal_context as phase8d

import comfy.samplers
from comfy.ldm.flux import math as flux_math

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_block_major_context_streaming_results"
REPORT = OUTPUT / "report.json"
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT
EXPECTED_HASH = "53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce"
LOCAL_COUNT = 55


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


class BlockMajorStreamingProbe(candidate2.OneEvaluationContextProbe):
    """Coordinate one native source forward with all native local forwards."""

    def __init__(self, local_count: int):
        super().__init__()
        self.local_count = local_count
        self.condition = threading.Condition()
        self.thread_state = threading.local()
        self.current_key = None
        self.current_k = None
        self.current_v = None
        self.consumed = 0
        self.source_records = []
        self.local_records = []
        self.barrier_memory = []
        self.one_block_kv_peak_bytes = 0
        self.pending = {}
        self.source_error = None

    def set_local(self, crop_index: int):
        self.thread_state.crop_index = crop_index

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        positioned_k = flux_math.apply_rope1(k, pe)
        global_k = positioned_k[:, :, text_tokens:]
        global_v = v[:, :, text_tokens:]
        kv_bytes = global_k.numel() * global_k.element_size()
        kv_bytes += global_v.numel() * global_v.element_size()
        with self.condition:
            if self.current_key is not None:
                raise AssertionError(f"Previous source block not released: {self.current_key}")
            self.current_key = key
            self.current_k = global_k
            self.current_v = global_v
            self.consumed = 0
            self.one_block_kv_peak_bytes = max(self.one_block_kv_peak_bytes, kv_bytes)
            self.source_records.append({
                "block_type": key[0], "block_index": key[1],
                "generated_tokens": sequence_end - text_tokens,
                "K_hash": tensor_hash(global_k), "V_hash": tensor_hash(global_v),
                "K_shape": list(global_k.shape), "V_shape": list(global_v.shape),
                "K_dtype": str(global_k.dtype), "V_dtype": str(global_v.dtype),
                "resident_KV_bytes": kv_bytes,
            })
            self.condition.notify_all()
            while self.consumed < self.local_count:
                if self.source_error is not None:
                    raise RuntimeError("A streaming local call failed") from self.source_error
                self.condition.wait()
            torch.cuda.synchronize()
            self.barrier_memory.append({
                "block_type": key[0], "block_index": key[1],
                "allocated_bytes_after_all_local_attention": int(torch.cuda.memory_allocated()),
                "reserved_bytes_after_all_local_attention": int(torch.cuda.memory_reserved()),
            })
            self.current_key = None
            self.current_k = None
            self.current_v = None
            self.condition.notify_all()
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def add_global_context(self, q, k, v, pe, attn_mask, extra_options):
        key = self.block_key(extra_options)
        crop_index = int(self.thread_state.crop_index)
        with self.condition:
            while self.current_key != key:
                if self.source_error is not None:
                    raise RuntimeError("Source streaming call failed") from self.source_error
                self.condition.wait()
            global_k = self.current_k
            global_v = self.current_v
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        pending_key = (threading.get_ident(), key)
        self.pending[pending_key] = flux_math.attention(
            q, k, v, pe=pe, mask=attn_mask, transformer_options=extra_options
        )
        positioned_q = flux_math.apply_rope1(q, pe)
        positioned_k = flux_math.apply_rope1(k, pe)
        augmented_k = torch.cat((positioned_k, global_k), dim=2)
        augmented_v = torch.cat((v, global_v), dim=2)
        self.local_records.append({
            "crop": crop_index, "block_type": key[0], "block_index": key[1],
            "global_tokens": int(global_k.shape[2]),
            "q_by_k": [int(q.shape[2]), int(augmented_k.shape[2])],
        })
        return {"q": positioned_q, "k": augmented_k, "v": augmented_v,
                "pe": None, "attn_mask": attn_mask}

    def restore_text_attention(self, attention_output, extra_options):
        key = self.block_key(extra_options)
        pending_key = (threading.get_ident(), key)
        normal = self.pending.pop(pending_key)
        text_tokens = int(extra_options["img_slice"][0])
        output = attention_output.clone()
        output[:, :text_tokens] = normal[:, :text_tokens]
        torch.cuda.synchronize()
        with self.condition:
            self.consumed += 1
            if self.consumed > self.local_count:
                raise AssertionError((key, self.consumed))
            self.condition.notify_all()
        return output

    def publish_source_failure(self, error):
        with self.condition:
            self.source_error = error
            self.condition.notify_all()

    def assert_complete(self):
        if self.pending:
            raise AssertionError(f"Pending local ordinary outputs: {tuple(self.pending)}")
        if self.current_key is not None:
            raise AssertionError(f"Unreleased source block: {self.current_key}")
        if len(self.source_records) != 25:
            raise AssertionError(f"Expected 25 source blocks, got {len(self.source_records)}")
        if len(self.local_records) != self.local_count * 25:
            raise AssertionError((len(self.local_records), self.local_count * 25))


class BlockMajorSampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None
        self.output = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 8h requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h, sigmas[0])
        accepted_hashes = []
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal], sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"], seed=extra_args.get("seed", 0),
            )
            accepted_hashes.append({"ordinal": ordinal, "H": tensor_hash(state.h), "G": tensor_hash(state.g)})

        sigma = sigmas[3]
        terminal_h, terminal_g = state.h, state.g
        h_snapshot, g_snapshot = terminal_h.clone(), terminal_g.clone()
        regions = coordinator.planner.plan(HIGH_HW)
        if len(regions) != LOCAL_COUNT:
            raise AssertionError((len(regions), LOCAL_COUNT))
        base_options = extra_args["model_options"]
        crop_input_hashes = [tensor_hash(terminal_h[:, :, r.y:r.y2, r.x:r.x2]) for r in regions]

        # A: exact Phase-8d CPU-offloaded reference, then release its all-block cache.
        reference_probe = phase8d.CPUOffloadedContextProbe()
        reference_capture_options = candidate2_trajectory.merge_options(
            base_options,
            candidate2.model_options(
                "phase8h_reference_source", phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
                reference_probe, "capture",
            ),
        )
        ref_source_start, ref_source_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ref_source_start.record()
        reference_x0_g = coordinator.adapter.predict_global(
            guider=model, g=terminal_g, sigma=sigma, canvas=HIGH_HW,
            model_options=reference_capture_options, seed=extra_args.get("seed", 0),
        )
        ref_source_end.record()
        reference_predictions = []
        reference_events = []
        reference_wall_start = time.perf_counter()
        for region in regions:
            options = candidate2_trajectory.merge_options(
                base_options,
                candidate2.model_options(
                    f"phase8h_reference_crop_{region.index}", phase2.rope_for_crop(region),
                    reference_probe, "context",
                ),
            )
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            prediction = coordinator.adapter.predict_region(
                guider=model, h_view=terminal_h[:, :, region.y:region.y2, region.x:region.x2],
                sigma=sigma, canvas=HIGH_HW, region=region, model_options=options,
                seed=extra_args.get("seed", 0),
            )
            end.record()
            reference_events.append((start, end))
            reference_predictions.append(prediction)
        torch.cuda.synchronize()
        reference_source_cuda = float(ref_source_start.elapsed_time(ref_source_end))
        reference_local_cuda = sum(float(a.elapsed_time(b)) for a, b in reference_events)
        reference_wall = time.perf_counter() - reference_wall_start
        reference_assembled, reference_coverage = coordinator.assembler.assemble(
            reference_predictions, regions, HIGH_HW
        )
        reference_h_star = terminal_h + (terminal_h - reference_assembled) / sigma * (sigmas[4] - sigma)
        reference_crop_cpu = [item.detach().float().cpu() for item in reference_predictions]
        reference_cache_bytes = reference_probe.capture_bytes
        reference_transfer_bytes = reference_probe.transfer_bytes
        reference_probe.global_kv.clear()
        del reference_predictions, reference_probe
        gc.collect()
        torch.cuda.empty_cache()

        # B: one native source call and all native crop calls, suspended block-major.
        streaming_probe = BlockMajorStreamingProbe(len(regions))
        source_options = candidate2_trajectory.merge_options(
            base_options,
            candidate2.model_options(
                "phase8h_stream_source", phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
                streaming_probe, "capture",
            ),
        )
        streaming_predictions = [None] * len(regions)
        errors = []
        source_result = {}

        def source_worker():
            try:
                with torch.inference_mode():
                    source_result["x0_g"] = coordinator.adapter.predict_global(
                        guider=model, g=terminal_g, sigma=sigma, canvas=HIGH_HW,
                        model_options=source_options, seed=extra_args.get("seed", 0),
                    )
            except BaseException as error:
                source_result["error"] = error
                print(f"PHASE8H_SOURCE_ERROR {type(error).__name__}: {error}", flush=True)
                streaming_probe.publish_source_failure(error)

        def local_worker(region):
            try:
                streaming_probe.set_local(region.index)
                options = candidate2_trajectory.merge_options(
                    base_options,
                    candidate2.model_options(
                        f"phase8h_stream_crop_{region.index}", phase2.rope_for_crop(region),
                        streaming_probe, "context",
                    ),
                )
                with torch.inference_mode():
                    streaming_predictions[region.index] = coordinator.adapter.predict_region(
                        guider=model, h_view=terminal_h[:, :, region.y:region.y2, region.x:region.x2],
                        sigma=sigma, canvas=HIGH_HW, region=region, model_options=options,
                        seed=extra_args.get("seed", 0),
                    )
            except BaseException as error:
                errors.append((region.index, error))
                print(f"PHASE8H_LOCAL_ERROR crop={region.index} {type(error).__name__}: {error}", flush=True)
                streaming_probe.publish_source_failure(error)

        torch.cuda.reset_peak_memory_stats()
        stream_baseline_allocated = int(torch.cuda.memory_allocated())
        stream_baseline_reserved = int(torch.cuda.memory_reserved())
        stream_start, stream_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        stream_wall_start = time.perf_counter()
        stream_start.record()
        source_thread = threading.Thread(target=source_worker, name="phase8h-source")
        local_threads = [threading.Thread(target=local_worker, args=(r,), name=f"phase8h-crop-{r.index}")
                         for r in regions]
        source_thread.start()
        for thread in local_threads:
            thread.start()
        source_thread.join()
        for thread in local_threads:
            thread.join()
        stream_end.record()
        torch.cuda.synchronize()
        stream_cuda = float(stream_start.elapsed_time(stream_end))
        stream_wall = time.perf_counter() - stream_wall_start
        if source_result.get("error") is not None or errors:
            failure_error = errors[0][1] if errors else source_result["error"]
            self.result = {
                "status": "OOM",
                "failure_point": "streaming double block 0 before all local native call frames completed",
                "error_type": type(failure_error).__name__,
                "error": str(failure_error),
                "shared_controls": {
                    "accepted_nonterminal_hashes": accepted_hashes,
                    "H3_hash": tensor_hash(terminal_h), "G3_hash": tensor_hash(terminal_g),
                    "terminal_crop_input_hashes": crop_input_hashes,
                    "crop_rectangles": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                    "full_global_tokens": 18432, "context_blocks_required": 25,
                },
                "reference": {
                    "source_x0_G": phase8d.summary(reference_x0_g),
                    "assembled_x0_H": phase8d.summary(reference_assembled),
                    "final_H": phase8d.summary(reference_h_star),
                    "overlap": phase8d.overlap_metrics(reference_crop_cpu, regions),
                    "coverage": [float(reference_coverage.min()), float(reference_coverage.max())],
                    "CPU_source_cache_bytes": reference_cache_bytes,
                    "CPU_to_GPU_KV_transfer_bytes": reference_transfer_bytes,
                    "source_cuda_ms": reference_source_cuda,
                    "terminal_local_cuda_ms": reference_local_cuda,
                    "terminal_local_wall_seconds": reference_wall,
                    "qualified_decoded_hash": EXPECTED_HASH,
                },
                "streaming_failure": {
                    "source_blocks_reached": len(streaming_probe.source_records),
                    "local_attention_calls_reached": len(streaming_probe.local_records),
                    "failed_crop_count": len(errors),
                    "failed_crops": sorted(index for index, _ in errors),
                    "one_block_GPU_KV_bytes": streaming_probe.one_block_kv_peak_bytes,
                    "CPU_source_cache_bytes": 0,
                    "CPU_to_GPU_KV_transfer_bytes": 0,
                    "baseline_allocated_bytes": stream_baseline_allocated,
                    "baseline_reserved_bytes": stream_baseline_reserved,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "peak_increment_over_baseline_bytes": int(torch.cuda.max_memory_allocated()) - stream_baseline_allocated,
                    "barrier_memory": streaming_probe.barrier_memory,
                    "wall_seconds_until_failure": stream_wall,
                    "cuda_ms_until_failure": stream_cuda,
                    "resident_tensor_interpretation": (
                        "Native call-frame suspension retains per-crop Q/K/V, normalization/modulation, "
                        "attention and residual temporaries in addition to block-to-block hidden state; "
                        "OOM occurs before a clean all-crop block barrier, so persistent hidden and "
                        "temporary attention memory cannot be isolated independently."
                    ),
                },
            }
            raise RuntimeError("Block-major native-call streaming OOM") from failure_error
        if source_result.get("error") is not None:
            raise RuntimeError("Streaming source failed") from source_result["error"]
        if errors:
            index, error = errors[0]
            raise RuntimeError(f"Streaming crop {index} failed") from error
        if any(item is None for item in streaming_predictions):
            raise RuntimeError("Streaming did not publish every crop prediction.")
        streaming_probe.assert_complete()
        streaming_assembled, streaming_coverage = coordinator.assembler.assemble(
            streaming_predictions, regions, HIGH_HW
        )
        streaming_h_star = terminal_h + (terminal_h - streaming_assembled) / sigma * (sigmas[4] - sigma)
        reference_g_star = terminal_g + (terminal_g - reference_x0_g) / sigma * (sigmas[4] - sigma)
        streaming_g_star = terminal_g + (terminal_g - source_result["x0_g"]) / sigma * (sigmas[4] - sigma)
        projection = coordinator.geometry.prolong(
            streaming_g_star - coordinator.geometry.restrict(streaming_h_star)
        )
        per_crop = [phase8d.difference(value.detach().float().cpu(), reference)
                    for value, reference in zip(streaming_predictions, reference_crop_cpu)]
        source_kv_hashes = [{"block_type": item["block_type"], "block_index": item["block_index"],
                             "K_hash": item["K_hash"], "V_hash": item["V_hash"]}
                            for item in streaming_probe.source_records]
        projection_rms = phase8d.summary(projection)["rms"]
        self.result = {
            "sigmas": [float(value) for value in sigmas],
            "shared_controls": {
                "accepted_nonterminal_hashes": accepted_hashes,
                "H3_hash": tensor_hash(terminal_h), "G3_hash": tensor_hash(terminal_g),
                "terminal_crop_input_hashes": crop_input_hashes,
                "crop_rectangles": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "full_global_tokens": 18432, "context_blocks": 25,
                "local_context_records": len(streaming_probe.local_records),
            },
            "reference": {
                "source_x0_G": phase8d.summary(reference_x0_g),
                "assembled_x0_H": phase8d.summary(reference_assembled),
                "final_H": phase8d.summary(reference_h_star),
                "overlap": phase8d.overlap_metrics(reference_crop_cpu, regions),
                "coverage": [float(reference_coverage.min()), float(reference_coverage.max())],
                "CPU_source_cache_bytes": reference_cache_bytes,
                "CPU_to_GPU_KV_transfer_bytes": reference_transfer_bytes,
                "source_cuda_ms": reference_source_cuda,
                "terminal_local_cuda_ms": reference_local_cuda,
                "terminal_local_wall_seconds": reference_wall,
            },
            "streaming": {
                "source_x0_G": phase8d.summary(source_result["x0_g"]),
                "source_x0_G_vs_reference": phase8d.difference(source_result["x0_g"], reference_x0_g),
                "G_star_vs_reference": phase8d.difference(streaming_g_star, reference_g_star),
                "assembled_x0_H": phase8d.summary(streaming_assembled),
                "assembled_vs_reference": phase8d.difference(streaming_assembled, reference_assembled),
                "per_crop_vs_reference": per_crop,
                "final_H": phase8d.summary(streaming_h_star),
                "final_H_vs_reference": phase8d.difference(streaming_h_star, reference_h_star),
                "overlap": phase8d.overlap_metrics(streaming_predictions, regions),
                "projection_needed": {"rms": projection_rms,
                    "over_H_star": projection_rms / phase8d.summary(streaming_h_star)["rms"]},
                "coverage": [float(streaming_coverage.min()), float(streaming_coverage.max())],
                "CPU_source_cache_bytes": 0,
                "CPU_to_GPU_KV_transfer_bytes": 0,
                "one_block_GPU_KV_peak_bytes": streaming_probe.one_block_kv_peak_bytes,
                "baseline_allocated_bytes": stream_baseline_allocated,
                "baseline_reserved_bytes": stream_baseline_reserved,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "barrier_memory": streaming_probe.barrier_memory,
                "source_KV_hashes": source_kv_hashes,
                "terminal_source_plus_local_cuda_ms": stream_cuda,
                "terminal_source_plus_local_wall_seconds": stream_wall,
            },
        }
        if not torch.equal(terminal_h, h_snapshot) or not torch.equal(terminal_g, g_snapshot):
            raise RuntimeError("Block-major calls mutated accepted H3/G3.")
        self.output = streaming_h_star.detach().float().cpu()
        if callback is not None:
            callback(3, streaming_assembled, streaming_h_star, STEPS)
        return model_sampling.inverse_noise_scaling(sigmas[-1], streaming_h_star)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *HIGH_HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(STEPS, math.prod(HIGH_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = BlockMajorSampler()
    perf.prepare_model_state(model)
    try:
        with torch.inference_mode():
            phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                torch.zeros_like(noise), callback=lambda *args: None,
                disable_pbar=True, seed=SEED,
            )
    except RuntimeError as error:
        if sampler.result is None:
            raise
        sampler.result["configuration"] = {
            "target_pixels_hw": [2048, 4096], "H": [128, 256], "G": [96, 192],
            "prompt": PROMPT, "seed": SEED, "cfg": 1.0, "production_changes": False,
            "execution": "native forward calls coordinated by experiment-local per-block barriers",
            "production_or_core_changes": False,
        }
        sampler.result["verdict"] = "EXACT STREAMING REQUIRES A SPECIALIZED FLUX EXECUTOR"
        REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(REPORT), "status": sampler.result["status"],
                          "failure": sampler.result["failure_point"],
                          "telemetry": sampler.result["streaming_failure"],
                          "verdict": sampler.result["verdict"]}, indent=2))
        return
    pixels = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    ).decode(sampler.output).cpu()
    image_path = OUTPUT / "BLOCK_MAJOR_STREAMING.png"
    phase2.save_pixels(pixels, image_path)
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        decoded = {"path": str(image_path), "dimensions_wh": list(rgb.size),
                   "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}
    sampler.result["configuration"] = {
        "target_pixels_hw": [2048, 4096], "H": [128, 256], "G": [96, 192],
        "prompt": PROMPT, "seed": SEED, "cfg": 1.0, "production_changes": False,
        "execution": "native forward calls coordinated by experiment-local per-block barriers",
    }
    sampler.result["streaming"]["decoded"] = decoded
    sampler.result["streaming"]["decoded_matches_qualified_hash"] = decoded["sha256_rgb"] == EXPECTED_HASH
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(REPORT), "decoded": decoded,
        "assembled_difference": sampler.result["streaming"]["assembled_vs_reference"],
        "final_difference": sampler.result["streaming"]["final_H_vs_reference"],
        "reference_cache_bytes": sampler.result["reference"]["CPU_source_cache_bytes"],
        "stream_KV_bytes": sampler.result["streaming"]["one_block_GPU_KV_peak_bytes"],
        "stream_transfer_bytes": sampler.result["streaming"]["CPU_to_GPU_KV_transfer_bytes"],
        "stream_peak_allocated": sampler.result["streaming"]["peak_allocated_bytes"],
        "stream_wall": sampler.result["streaming"]["terminal_source_plus_local_wall_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
