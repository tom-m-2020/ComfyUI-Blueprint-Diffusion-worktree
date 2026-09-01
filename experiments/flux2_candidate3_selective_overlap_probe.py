"""Phase 6e zero-update cross-crop selective-token FLUX.2 probe."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
SPOTEDIT_ROOT = ROOT.parent / "ComfyUI-SpotEdit" / "target" / "ComfyUI-SpotEdit"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(SPOTEDIT_ROOT))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_hard_global_anchor as phase3
from spotedit import flux2_native as sparse_flux
from spotedit.state import SpotEditRunState


OUTPUT = ROOT / "experiments" / "flux2_candidate3_selective_overlap_results"
PROMPT = phase2.PROMPT
SEED = 20260829
TARGET_HW = (32, 64)
CROP_A_X = 0
CROP_B_X = 24
OVERLAP_WIDTH = 8
ACTIVE_WIDTH = 24
REPEATS = 3


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = value.float() - reference.float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "bit_exact": bool(torch.equal(value, reference)),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def cuda_measure(function, baseline_tensors=()):
    torch.cuda.synchronize()
    function()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    times = []
    for _ in range(REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return {
        "cuda_ms": times,
        "cuda_ms_mean": sum(times) / len(times),
        "baseline_allocated_gib": baseline_allocated / 1024**3,
        "baseline_reserved_gib": baseline_reserved / 1024**3,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def remap_crop_a_cache_to_crop_b(cache, prepared_b, run_state):
    """Map A's rightmost 8 columns into B's leftmost 8 cache slots."""
    text_tokens = cache.metadata.text_token_count
    a_indices = torch.tensor(
        [y * 32 + x for y in range(32) for x in range(24, 32)],
        device=cache.metadata.device,
        dtype=torch.long,
    )
    b_indices = torch.tensor(
        [y * 32 + x for y in range(32) for x in range(0, 8)],
        device=cache.metadata.device,
        dtype=torch.long,
    )
    layers = []
    for layer in cache.layers:
        source = layer.dynamic
        key = torch.zeros_like(source.key)
        value = torch.zeros_like(source.value)
        key[:, :, :text_tokens] = source.key[:, :, :text_tokens]
        value[:, :, :text_tokens] = source.value[:, :, :text_tokens]
        key[:, :, text_tokens + b_indices] = source.key[:, :, text_tokens + a_indices]
        value[:, :, text_tokens + b_indices] = source.value[:, :, text_tokens + a_indices]
        layers.append(sparse_flux.Flux2LayerKV(
            layer.block_type,
            layer.block_index,
            sparse_flux.Flux2KVSegment(key, value),
            layer.reference,
        ))
    metadata = replace(
        cache.metadata,
        image_ids=prepared_b.image_ids.detach().clone(),
        text_ids=prepared_b.text_ids.detach().clone(),
        run_id=run_state.run_id,
    )
    return sparse_flux.Flux2KVCache(metadata=metadata, layers=tuple(layers))


class ProbeSampler(phase2.comfy.samplers.Sampler):
    def __init__(self):
        self.report = None
        self.diagnostic = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        sigma = sigmas[0]
        diffusion = model.inner_model.diffusion_model
        crop_a = h[:, :, :, CROP_A_X:CROP_A_X + 32]
        crop_b = h[:, :, :, CROP_B_X:CROP_B_X + 32]
        core_captures = {}
        full_double = []
        full_single = []
        handles = []

        def core_pre(module, args, kwargs):
            core_captures["args"] = args
            core_captures["kwargs"] = kwargs

        def core_post(module, args, kwargs, output):
            core_captures["output"] = output.detach().clone()

        handles.append(diffusion.register_forward_pre_hook(core_pre, with_kwargs=True))
        handles.append(diffusion.register_forward_hook(core_post, with_kwargs=True))
        for block in diffusion.double_blocks:
            handles.append(block.register_forward_hook(
                lambda module, inputs, output: full_double.append(output[0].detach().clone())
            ))
        for block in diffusion.single_blocks:
            handles.append(block.register_forward_hook(
                lambda module, inputs, output: full_single.append(output.detach().clone())
            ))

        options_a = phase2.clone_options(
            extra_args["model_options"], {"shift_y": 0.0, "shift_x": float(CROP_A_X)}
        )
        options_b = phase2.clone_options(
            extra_args["model_options"], {"shift_y": 0.0, "shift_x": float(CROP_B_X)}
        )
        with torch.inference_mode():
            ordinary_a = model(crop_a, sigma.expand(1), model_options=options_a, seed=SEED)
            full_double.clear()
            full_single.clear()
            ordinary_b = model(crop_b, sigma.expand(1), model_options=options_b, seed=SEED)
        for handle in handles:
            handle.remove()

        core_args = core_captures["args"]
        core_kwargs = core_captures["kwargs"].copy()
        raw_b = core_captures["output"]
        x_b, timestep_b = core_args[:2]
        context_b = core_kwargs.pop("context")
        transformer_b = core_kwargs.pop("transformer_options", {})
        y_b = core_kwargs.pop("y", None)
        guidance_b = core_kwargs.pop("guidance", None)
        ref_b = core_kwargs.pop("ref_latents", None)
        control_b = core_kwargs.pop("control", None)

        # Recreate the corresponding core input for crop A; model sampling is CONST.
        x_a = model_sampling.calculate_input(sigma.expand(1), crop_a).to(x_b.dtype)
        transformer_a = options_a.get("transformer_options", {})
        run_state = SpotEditRunState(source_shape=tuple(x_a.shape))
        run_state.run_metadata["source_id"] = None
        with torch.inference_mode():
            sparse_flux.full_pass_flux2_forward(
                diffusion, x_a, timestep_b, context_b, y=y_b, guidance=guidance_b,
                ref_latents=ref_b, control=control_b, transformer_options=transformer_a,
                run_state=run_state,
            )
        source_cache = run_state.candidate_cache
        prepared_b = sparse_flux.prepare_flux2_full_pass_inputs(
            diffusion, x_b, context_b, ref_b, transformer_b,
            diffusion.params.default_ref_method,
        )
        b_cache_state = SpotEditRunState(source_shape=tuple(x_b.shape))
        b_cache_state.run_metadata["source_id"] = None
        with torch.inference_mode():
            sparse_flux.full_pass_flux2_forward(
                diffusion, x_b, timestep_b, context_b, y=y_b, guidance=guidance_b,
                ref_latents=ref_b, control=control_b, transformer_options=transformer_b,
                run_state=b_cache_state,
            )
        b_full_cache = b_cache_state.candidate_cache
        text_count = source_cache.metadata.text_token_count
        a_overlap = torch.tensor(
            [y * 32 + x for y in range(32) for x in range(24, 32)],
            device=x_b.device, dtype=torch.long,
        ) + text_count
        b_overlap = torch.tensor(
            [y * 32 + x for y in range(32) for x in range(0, 8)],
            device=x_b.device, dtype=torch.long,
        ) + text_count
        source_overlap_kv = []
        for a_layer, b_layer in zip(source_cache.layers, b_full_cache.layers):
            source_overlap_kv.append({
                "block": f"{a_layer.block_type}_{a_layer.block_index}",
                "key": difference(
                    a_layer.dynamic.key.index_select(2, a_overlap),
                    b_layer.dynamic.key.index_select(2, b_overlap),
                ),
                "value": difference(
                    a_layer.dynamic.value.index_select(2, a_overlap),
                    b_layer.dynamic.value.index_select(2, b_overlap),
                ),
            })
        b_cache_state.clear()
        del b_full_cache
        run_state.accepted_cache = remap_crop_a_cache_to_crop_b(
            source_cache, prepared_b, run_state
        )
        run_state.candidate_cache = None
        active = torch.tensor(
            [y * 32 + x for y in range(32) for x in range(8, 32)],
            device=x_b.device,
            dtype=torch.long,
        )

        sparse_double = []
        sparse_single = []
        original_double = sparse_flux._sparse_double_block
        original_single = sparse_flux._sparse_single_block

        def capture_double(*args, **kwargs):
            result = original_double(*args, **kwargs)
            sparse_double.append(result[0].detach().clone())
            return result

        def capture_single(*args, **kwargs):
            result = original_single(*args, **kwargs)
            sparse_single.append(result[0].detach().clone())
            return result

        sparse_flux._sparse_double_block = capture_double
        sparse_flux._sparse_single_block = capture_single
        try:
            with torch.inference_mode():
                sparse_result = sparse_flux.sparse_transformer_forward(
                    diffusion, x_b, timestep_b, context_b, active, run_state,
                    y=y_b, guidance=guidance_b, ref_latents=ref_b,
                    control=control_b, transformer_options=transformer_b,
                )
        finally:
            sparse_flux._sparse_double_block = original_double
            sparse_flux._sparse_single_block = original_single

        text_tokens = context_b.shape[1]
        per_block = []
        for index, sparse_hidden in enumerate(sparse_double):
            per_block.append({
                "block": f"double_{index}",
                "active_hidden": difference(
                    sparse_hidden, full_double[index].index_select(1, active)
                ),
            })
        for index, sparse_hidden in enumerate(sparse_single):
            full_active = full_single[index].index_select(
                1, active + text_tokens
            )
            per_block.append({
                "block": f"single_{index}",
                "active_hidden": difference(
                    sparse_hidden[:, text_tokens:], full_active
                ),
            })

        raw_tokens_b = raw_b.permute(0, 2, 3, 1).reshape(1, 1024, 128)
        active_raw_reference = raw_tokens_b.index_select(1, active)
        raw_difference = difference(
            sparse_result.active_velocity_tokens, active_raw_reference
        )
        ordinary_active = ordinary_b.permute(0, 2, 3, 1).reshape(1, 1024, 128).index_select(1, active)
        sparse_active_x0 = (
            x_b.permute(0, 2, 3, 1).reshape(1, 1024, 128).index_select(1, active)
            - sparse_result.active_velocity_tokens * sigma.to(sparse_result.active_velocity_tokens.dtype)
        )
        x0_difference = difference(sparse_active_x0, ordinary_active)
        ordinary_active_grid = ordinary_active.reshape(1, 32, ACTIVE_WIDTH, 128).permute(0, 3, 1, 2)
        sparse_active_grid = sparse_active_x0.reshape(1, 32, ACTIVE_WIDTH, 128).permute(0, 3, 1, 2)
        low_difference = difference(
            torch.nn.functional.avg_pool2d(sparse_active_grid.float(), 4, 4),
            torch.nn.functional.avg_pool2d(ordinary_active_grid.float(), 4, 4),
        )

        hybrid = ordinary_b.clone().permute(0, 2, 3, 1).reshape(1, 1024, 128)
        hybrid.index_copy_(1, active, sparse_active_x0.to(hybrid.dtype))
        hybrid = hybrid.reshape(1, 32, 32, 128).permute(0, 3, 1, 2).contiguous()
        self.diagnostic = {"ordinary": ordinary_b.detach().float().cpu(), "selective": hybrid.detach().float().cpu()}

        def dense_core():
            diffusion(
                x_b, timestep_b, context=context_b, y=y_b, guidance=guidance_b,
                ref_latents=ref_b, control=control_b, transformer_options=transformer_b,
            )

        def sparse_core():
            sparse_flux.sparse_transformer_forward(
                diffusion, x_b, timestep_b, context_b, active, run_state,
                y=y_b, guidance=guidance_b, ref_latents=ref_b,
                control=control_b, transformer_options=transformer_b,
            )

        with torch.inference_mode():
            dense_perf = cuda_measure(dense_core)
            sparse_perf = cuda_measure(sparse_core)

        first_divergence = next(
            (item for item in per_block if item["active_hidden"]["max_abs"] != 0.0),
            None,
        )
        ds = sparse_result.double_stats
        ss = sparse_result.single_stats
        self.report = {
            "state": "initial accepted Candidate-3 H at sigma 1.0; zero sampler updates",
            "crop_A": {"x": CROP_A_X, "tokens": 1024},
            "crop_B": {"x": CROP_B_X, "tokens": 1024},
            "overlap_tokens_reused": 256,
            "active_new_tokens": int(active.numel()),
            "active_indices": {"first": int(active[0]), "last": int(active[-1])},
            "cache_provenance": (
                "same accepted H, sigma, conditioning, model and absolute positions; "
                "per-block K/V for B overlap remapped from crop A"
            ),
            "ordinary_crop_A_vs_B_overlap_input_x0": difference(
                ordinary_a[:, :, :, 24:32], ordinary_b[:, :, :, 0:8]
            ),
            "active_prediction": x0_difference,
            "active_prediction_low_frequency_4x4": low_difference,
            "active_raw_velocity": raw_difference,
            "per_block": per_block,
            "source_overlap_kv_divergence": source_overlap_kv,
            "first_divergence": first_divergence,
            "work": {
                "img_in_tokens": ds.img_in_tokens,
                "double_image_qkv_tokens_each": list(ds.image_qkv_tokens),
                "double_image_projection_tokens_each": list(ds.image_projection_tokens),
                "double_image_mlp_tokens_each": list(ds.image_mlp_tokens),
                "double_query_lengths": list(ds.query_lengths),
                "double_kv_lengths": list(ds.key_value_lengths),
                "single_linear1_tokens_each": list(ss.linear1_tokens),
                "single_mlp_tokens_each": list(ss.mlp_tokens),
                "single_linear2_tokens_each": list(ss.linear2_tokens),
                "single_query_lengths": list(ss.query_lengths),
                "single_kv_lengths": list(ss.key_value_lengths),
                "final_layer_tokens": ss.final_layer_tokens,
                "skipped_generated_token_local_fraction": 256 / 1024,
                "text_tokens_still_executed": int(text_tokens),
                "full_context_kv_tokens": int(text_tokens + 1024),
            },
            "performance": {"ordinary_dense_core": dense_perf, "selective_core": sparse_perf},
            "methods_restored": (
                sparse_flux._sparse_double_block is original_double
                and sparse_flux._sparse_single_block is original_single
            ),
        }
        return h


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    sigmas = phase2.get_schedule(4, math.prod(TARGET_HW)).float().clone()
    noise = torch.randn((1, 128, *TARGET_HW), generator=torch.Generator().manual_seed(SEED))
    sampler = ProbeSampler()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), disable_pbar=True, seed=SEED,
        )
        for name, latent in sampler.diagnostic.items():
            pixels = vae.decode(latent).cpu()
            phase2.save_pixels(pixels, OUTPUT / f"{name}_crop_B.png")
    report_path = OUTPUT / "report.json"
    report_path.write_text(json.dumps(sampler.report, indent=2), encoding="utf-8")
    print(json.dumps({
        "active_prediction": sampler.report["active_prediction"],
        "first_divergence": sampler.report["first_divergence"],
        "performance": sampler.report["performance"],
        "report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
