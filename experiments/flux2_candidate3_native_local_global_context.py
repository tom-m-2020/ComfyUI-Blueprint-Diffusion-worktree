"""Phase 9c: terminal native-local global-context discriminator."""

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
import flux2_candidate2_four_step_trajectory as candidate2_trajectory
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_terminal_context as phase8d
import flux2_candidate3_performance_characterization as perf
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_native_local_global_context_results"
REPORT = OUTPUT / "report.json"
H_HW = phase9b.H_HW
STEPS = phase9b.STEPS
SEED = phase9b.SEED
PROMPT = phase9b.PROMPT
REPRESENTATIVE_REGION = 7


def difference(left, right):
    delta = left.detach().float() - right.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(left, right)),
    }


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def merge_options(base, rope, probe=None, mode="ordinary", role="phase9c"):
    if probe is None:
        return candidate2_trajectory.merge_options(
            base, {"transformer_options": {"rope_options": dict(rope)}}
        )
    return candidate2_trajectory.merge_options(
        base, candidate2.model_options(role, rope, probe, mode)
    )


def make_working(h_view, sigma, ordinal, region):
    base = phase9b.prolong2(h_view)
    generator = torch.Generator(device="cpu").manual_seed(
        SEED + 100_003 * ordinal + 1_009 * region.index
    )
    noise = torch.randn(tuple(base.shape), generator=generator).to(
        device=base.device, dtype=base.dtype
    )
    high = noise - phase9b.prolong2(phase9b.restrict2(noise))
    return base + sigma * high


class TerminalContextSampler(phase2.comfy.samplers.Sampler):
    def __init__(self):
        self.results = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 9c requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1], model_options=extra_args["model_options"],
                seed=SEED,
            )

        terminal_sigma = sigmas[3]
        regions = phase9b.DestinationPlanner().plan(H_HW)
        accepted_h_hash = tensor_hash(state.h)
        accepted_g_hash = tensor_hash(state.g)
        working = []
        for region in regions:
            h_view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = make_working(h_view, terminal_sigma, 3, region)
            coarse_error = phase9b.restrict2(value).float() - h_view.float()
            if float(coarse_error.abs().max()) > 1e-6:
                raise RuntimeError("Phase 9c W construction drifted from Phase 9b.")
            working.append(value)
        working_hashes = [tensor_hash(value) for value in working]
        base_options = extra_args["model_options"]
        variants = {}
        predictions_by_variant = {}

        for name, source_kind in (
            ("A_MAGNIFIED_LOCAL_ONLY", None),
            ("B_MAGNIFIED_FULL_H_CONTEXT", "H"),
            ("C_MAGNIFIED_FIXED_G_CONTEXT", "G"),
        ):
            gc.collect()
            phase2.comfy.model_management.soft_empty_cache()
            torch.cuda.synchronize()
            baseline_allocated = int(torch.cuda.memory_allocated())
            baseline_reserved = int(torch.cuda.memory_reserved())
            torch.cuda.reset_peak_memory_stats()
            probe = None
            source_cuda_ms = 0.0
            source_wall = 0.0
            source_tokens = 0
            if source_kind is not None:
                probe = phase8d.CPUOffloadedContextProbe()
                if source_kind == "H":
                    source = state.h
                    rope = {}
                else:
                    source = state.g
                    rope = {
                        "scale_y": (H_HW[0] - 1.0) / (state.g.shape[-2] - 1.0),
                        "scale_x": (H_HW[1] - 1.0) / (state.g.shape[-1] - 1.0),
                    }
                source_tokens = source.shape[-2] * source.shape[-1]
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                source_started = time.perf_counter()
                start_event.record()
                _ = model(
                    source, terminal_sigma.expand(1),
                    model_options=merge_options(
                        base_options, rope, probe, "capture", f"phase9c_{source_kind}_source"
                    ),
                    seed=SEED,
                )
                end_event.record()
                torch.cuda.synchronize()
                source_cuda_ms = float(start_event.elapsed_time(end_event))
                source_wall = time.perf_counter() - source_started

            local_start = torch.cuda.Event(enable_timing=True)
            local_end = torch.cuda.Event(enable_timing=True)
            local_started = time.perf_counter()
            local_start.record()
            x0_w_values = []
            restricted = []
            for region, value in zip(regions, working):
                rope = {
                    "shift_y": float(region.y), "shift_x": float(region.x),
                    "scale_y": 31.0 / 63.0, "scale_x": 31.0 / 63.0,
                }
                options = merge_options(
                    base_options, rope, probe,
                    "ordinary" if probe is None else "context",
                    f"phase9c_{name}_crop_{region.index}",
                )
                x0_w = model(
                    value, terminal_sigma.expand(1), model_options=options, seed=SEED
                )
                x0_w_values.append(x0_w)
                restricted.append(phase9b.restrict2(x0_w))
            local_end.record()
            torch.cuda.synchronize()
            local_cuda_ms = float(local_start.elapsed_time(local_end))
            local_wall = time.perf_counter() - local_started
            assembled, coverage = coordinator.assembler.assemble(restricted, regions, H_HW)
            overlap = phase8d.overlap_metrics(
                [item.detach().float().cpu() for item in restricted], regions
            )
            if tensor_hash(state.h) != accepted_h_hash or tensor_hash(state.g) != accepted_g_hash:
                raise RuntimeError("Phase 9c context evaluation mutated accepted state.")
            if [tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError("Phase 9c context evaluation mutated W.")
            variants[name] = {
                "source": source_kind,
                "source_tokens": source_tokens,
                "context_tokens_per_block": source_tokens,
                "context_blocks": 0 if probe is None else 25,
                "local_query_tokens": 512 + 4096,
                "local_generated_tokens": 4096,
                "local_forwards": len(regions),
                "source_forwards": int(source_kind is not None),
                "source_cuda_ms": source_cuda_ms,
                "source_wall_seconds": source_wall,
                "local_cuda_ms": local_cuda_ms,
                "local_wall_seconds": local_wall,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "cpu_kv_cache_bytes": 0 if probe is None else probe.capture_bytes,
                "cpu_to_gpu_transfer_bytes": 0 if probe is None else probe.transfer_bytes,
                "overlap": overlap,
                "coverage": [float(coverage.min()), float(coverage.max())],
                "assembled": phase8d.summary(assembled),
                "representative_x0_W": phase8d.summary(x0_w_values[REPRESENTATIVE_REGION]),
            }
            predictions_by_variant[name] = {
                "x0_w": [item.detach().float().cpu() for item in x0_w_values],
                "restricted": [item.detach().float().cpu() for item in restricted],
                "assembled": assembled.detach().float().cpu(),
            }
            if probe is not None:
                probe.global_kv.clear()
                probe.pending_normal_attention.clear()
            del probe, x0_w_values, restricted, assembled

        names = list(variants)
        pairwise = {}
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                pairwise[f"{left}_vs_{right}"] = {
                    "assembled": difference(
                        predictions_by_variant[left]["assembled"],
                        predictions_by_variant[right]["assembled"],
                    ),
                    "representative_x0_W": difference(
                        predictions_by_variant[left]["x0_w"][REPRESENTATIVE_REGION],
                        predictions_by_variant[right]["x0_w"][REPRESENTATIVE_REGION],
                    ),
                }
        self.results = {
            "configuration": {
                "H": list(H_HW), "G": list(state.g.shape[-2:]),
                "terminal_sigma": float(terminal_sigma), "seed": SEED,
                "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
                "working_canvas": [64, 64], "coordinate_scale": [31/63, 31/63],
                "accepted_H_hash": accepted_h_hash, "accepted_G_hash": accepted_g_hash,
                "working_hashes": working_hashes,
            },
            "variants": variants,
            "pairwise": pairwise,
        }
        self.outputs = predictions_by_variant
        return predictions_by_variant["A_MAGNIFIED_LOCAL_ONLY"]["assembled"].to(state.h.device)


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
                    "path": str(path), "dimensions_wh": list(rgb.size),
                    "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
                }
    return images


def comparison_sheet(images):
    sheet = Image.new("RGB", (768, 420 * len(images)), "white")
    for i, (name, records) in enumerate(images.items()):
        with Image.open(records["final_terminal"]["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
        sheet.paste(thumb, (0, i * 420 + 36))
        ImageDraw.Draw(sheet).text((10, i * 420 + 10), name, fill="black")
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
    sampler = TerminalContextSampler()
    perf.prepare_model_state(model)
    with phase9b.scoped_variant("sigma_consistent", preterminal_probe), torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    if sampler.results is None or sampler.outputs is None:
        raise RuntimeError("Phase 9c sampler produced no result.")
    images = decode_outputs(sampler.outputs)
    comparison_sheet(images)
    sampler.results["images"] = images
    REPORT.write_text(json.dumps(sampler.results, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "source_tokens": value["source_tokens"],
            "overlap_rms": value["overlap"]["aggregate_rms"],
            "assembled": value["assembled"],
            "source_cuda_ms": value["source_cuda_ms"],
            "local_cuda_ms": value["local_cuda_ms"],
        }
        for name, value in sampler.results["variants"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
