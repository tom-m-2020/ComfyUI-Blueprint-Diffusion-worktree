"""Phase 8c: fixed-state terminal-authority lifecycle discriminator."""

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
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a

import comfy.samplers

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_terminal_authority_results"
REPORT = OUTPUT / "report.json"
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT


def summary(value: torch.Tensor) -> dict:
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


def difference(left: torch.Tensor, right: torch.Tensor) -> dict:
    delta = left.detach().float() - right.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(left, right)),
    }


class EventTimer:
    def __init__(self):
        self.records = []

    def call(self, category, ordinal, function, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(**kwargs)
        end.record()
        self.records.append({
            "category": category, "ordinal": ordinal,
            "start": start, "end": end,
        })
        return result

    def resolve(self):
        torch.cuda.synchronize()
        return [
            {
                "category": item["category"],
                "ordinal": item["ordinal"],
                "cuda_ms": float(item["start"].elapsed_time(item["end"])),
            }
            for item in self.records
        ]


class Capture:
    def __init__(self):
        self.values = {}

    def __call__(self, name, ordinal, value):
        self.values.setdefault(name, {})[ordinal] = value.detach().float().cpu()


class TerminalAuthoritySampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None

    def sample(
        self, model, sigmas, extra_args, callback, noise, latent_image=None,
        denoise_mask=None, disable_pbar=False,
    ):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 8c requires empty-latent T2I without a mask.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(
            sigmas[0], noise, latent_image, self.max_denoise(model, sigmas)
        )
        capture = Capture()
        coordinator = BlueprintCoordinator(capture=capture)
        timer = EventTimer()
        state = coordinator.initialize(h, sigmas[0])
        accepted_before = {0: state.h.detach().float().cpu()}

        # Production-exact nonterminal lifecycle, executed only once.
        original_global = coordinator.adapter.predict_global
        original_region = coordinator.adapter.predict_region

        def timed_global(**kwargs):
            return timer.call("global", state.ordinal, original_global, **kwargs)

        def timed_region(**kwargs):
            return timer.call("local", state.ordinal, original_region, **kwargs)

        coordinator.adapter.predict_global = timed_global
        coordinator.adapter.predict_region = timed_region
        for ordinal in range(3):
            state, x0_h = coordinator.evaluate(
                guider=model,
                state=state,
                sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1],
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            )
            accepted_before[ordinal + 1] = state.h.detach().float().cpu()
            if callback is not None:
                callback(ordinal, x0_h, state.h, STEPS)

        # One immutable terminal local evaluation shared by A/B/C.
        ordinal = 3
        sigma = sigmas[ordinal]
        sigma_next = sigmas[ordinal + 1]
        terminal_h = state.h
        terminal_g = state.g
        h_snapshot = terminal_h.clone()
        g_snapshot = terminal_g.clone()
        regions = coordinator.planner.plan(tuple(terminal_h.shape[-2:]))
        local_predictions = []
        terminal_local_inputs = []
        for region in regions:
            view = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            terminal_local_inputs.append(view.detach().float().cpu())
            local_predictions.append(timer.call(
                "local", ordinal, original_region,
                guider=model, h_view=view, sigma=sigma,
                canvas=tuple(terminal_h.shape[-2:]), region=region,
                model_options=extra_args["model_options"],
                seed=extra_args.get("seed", 0),
            ))
        if not torch.equal(terminal_h, h_snapshot) or not torch.equal(terminal_g, g_snapshot):
            raise RuntimeError("Terminal model calls mutated accepted state.")
        x0_h, coverage = coordinator.assembler.assemble(
            local_predictions, regions, tuple(terminal_h.shape[-2:])
        )
        h_star = terminal_h + (terminal_h - x0_h) / sigma * (sigma_next - sigma)
        a_final = h_star
        c_projection = coordinator.geometry.prolong(terminal_g - coordinator.geometry.restrict(h_star))
        c_final = h_star + c_projection
        peak_before_terminal_global = {
            "allocated": int(torch.cuda.max_memory_allocated()),
            "reserved": int(torch.cuda.max_memory_reserved()),
        }

        # B alone performs the deliberately restored terminal global prediction.
        x0_g = timer.call(
            "global", ordinal, original_global,
            guider=model, g=terminal_g, sigma=sigma,
            canvas=tuple(terminal_h.shape[-2:]),
            model_options=extra_args["model_options"],
            seed=extra_args.get("seed", 0),
        )
        g_star = terminal_g + (terminal_g - x0_g) / sigma * (sigma_next - sigma)
        b_projection = coordinator.geometry.prolong(g_star - coordinator.geometry.restrict(h_star))
        b_final = h_star + b_projection
        torch.cuda.synchronize()

        # Capture terminal diagnostics without publishing an experimental state.
        capture("accepted_H_before", ordinal, terminal_h)
        capture("assembled_x0_H", ordinal, x0_h)
        capture("H_star", ordinal, h_star)
        capture("x0_G", ordinal, x0_g)
        capture("G_star", ordinal, g_star)
        capture("A_FINAL", ordinal, a_final)
        capture("B_FINAL", ordinal, b_final)
        capture("C_FINAL", ordinal, c_final)
        capture("B_PROJECTION", ordinal, b_projection)
        capture("C_PROJECTION", ordinal, c_projection)

        # Shared-state controls are identities by construction and are also hashed.
        local_prediction_hashes = [
            hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()
            for value in local_predictions
        ]
        timings = timer.resolve()
        terminal_euler = difference(h_star, x0_h)
        variants = {
            "A_PRODUCTION_TERMINAL_RELEASE": self.variant_metrics(
                a_final, h_star, g_star, terminal_g, None, coordinator.geometry
            ),
            "B_TERMINAL_HARD_PROJECTION": self.variant_metrics(
                b_final, h_star, g_star, terminal_g, b_projection, coordinator.geometry
            ),
            "C_TERMINAL_RETAINED_G_PROJECTION": self.variant_metrics(
                c_final, h_star, terminal_g, terminal_g, c_projection, coordinator.geometry
            ),
        }
        variants["A_PRODUCTION_TERMINAL_RELEASE"]["global_forward_count"] = 3
        variants["B_TERMINAL_HARD_PROJECTION"]["global_forward_count"] = 4
        variants["C_TERMINAL_RETAINED_G_PROJECTION"]["global_forward_count"] = 3
        for item in variants.values():
            item["local_forward_count"] = len(regions) * STEPS

        pairs = {}
        finals = {"A": a_final, "B": b_final, "C": c_final}
        for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
            pairs[f"{left}_vs_{right}"] = difference(finals[left], finals[right])

        self.result = {
            "sigmas": [float(value) for value in sigmas],
            "terminal_sigma": float(sigma),
            "terminal_sigma_next": float(sigma_next),
            "terminal_euler_H_star_vs_x0_H": terminal_euler,
            "coverage_min": float(coverage.min()),
            "coverage_max": float(coverage.max()),
            "shared_controls": {
                "initial_H": summary(capture.values["initial_H"][-1]),
                "initial_G": summary(capture.values["initial_G"][-1]),
                "accepted_H3": summary(terminal_h),
                "accepted_G3": summary(terminal_g),
                "terminal_crop_count": len(regions),
                "terminal_crop_input_hashes": [hashlib.sha256(value.numpy().tobytes()).hexdigest() for value in terminal_local_inputs],
                "terminal_local_prediction_hashes": local_prediction_hashes,
                "assembled_terminal_x0_H": summary(x0_h),
                "terminal_H_star": summary(h_star),
                "A_B_C_share_terminal_local_tensors": True,
                "C_terminal_global_forward_performed": False,
            },
            "variants": variants,
            "pairwise_final_latent": pairs,
            "timings": timings,
            "terminal_global_cuda_ms": next(item["cuda_ms"] for item in timings if item["category"] == "global" and item["ordinal"] == 3),
            "nonterminal_global_cuda_ms": sum(item["cuda_ms"] for item in timings if item["category"] == "global" and item["ordinal"] < 3),
            "local_cuda_ms": sum(item["cuda_ms"] for item in timings if item["category"] == "local"),
            "peak_before_terminal_global": peak_before_terminal_global,
            "peak_after_terminal_global": {
                "allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved()),
            },
            "telemetry_nonterminal": coordinator.telemetry,
            "captures": capture.values,
        }
        if callback is not None:
            callback(ordinal, x0_h, a_final, STEPS)
        return model_sampling.inverse_noise_scaling(sigmas[-1], a_final)

    @staticmethod
    def variant_metrics(final, h_star, coarse_target, retained_g, projection, geometry):
        final_coarse = geometry.restrict(final)
        target_error = difference(final_coarse, coarse_target)
        retained_error = difference(final_coarse, retained_g)
        if projection is None:
            projection_stats = {"rms": 0.0, "over_H_star": 0.0, "low_frequency_rms": 0.0}
        else:
            projection_rms = summary(projection)["rms"]
            projection_stats = {
                "rms": projection_rms,
                "over_H_star": projection_rms / summary(h_star)["rms"],
                "low_frequency_rms": summary(geometry.restrict(projection))["rms"],
            }
        return {
            "final": summary(final),
            "D_final_vs_coarse_target": target_error,
            "D_final_vs_retained_G3": retained_error,
            "projection": projection_stats,
        }


def decode(latents):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    result = {}
    for name, latent in latents.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            result[name] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return result


def make_sheets():
    cell_w, cell_h, label_h = 800, 400, 28
    final_names = (
        "A_PRODUCTION_TERMINAL_RELEASE_FINAL",
        "B_TERMINAL_HARD_PROJECTION_FINAL",
        "C_TERMINAL_RETAINED_G_PROJECTION_FINAL",
    )
    sheet = Image.new("RGB", (cell_w, (cell_h + label_h) * 3), "white")
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(final_names):
        with Image.open(OUTPUT / f"{name}.png") as source:
            preview = ImageOps.contain(source.convert("RGB"), (cell_w, cell_h), Image.Resampling.LANCZOS)
        y = row * (cell_h + label_h)
        sheet.paste(preview, ((cell_w - preview.width) // 2, y + label_h))
        draw.text((8, y + 7), name, fill="black")
    sheet.save(OUTPUT / "FINAL_ABC_COMPARISON.png")

    rows = []
    for ordinal in range(4):
        if ordinal < 3:
            rows.append((f"interval {ordinal} x0_G", f"INTERVAL_{ordinal:02d}_X0_G"))
        rows.extend((
            (f"interval {ordinal} assembled x0_H", f"INTERVAL_{ordinal:02d}_X0_H"),
            (f"interval {ordinal} H_star", f"INTERVAL_{ordinal:02d}_H_STAR"),
            (f"interval {ordinal} accepted H", f"INTERVAL_{ordinal:02d}_ACCEPTED_H"),
        ))
    life_w, life_h, life_label = 480, 240, 24
    lifecycle = Image.new("RGB", (life_w, (life_h + life_label) * len(rows)), "white")
    draw = ImageDraw.Draw(lifecycle)
    for row, (label, name) in enumerate(rows):
        with Image.open(OUTPUT / f"{name}.png") as source:
            preview = ImageOps.contain(source.convert("RGB"), (life_w, life_h), Image.Resampling.LANCZOS)
        y = row * (life_h + life_label)
        lifecycle.paste(preview, ((life_w - preview.width) // 2, y + life_label))
        draw.text((8, y + 5), label, fill="black")
    lifecycle.save(OUTPUT / "LIFECYCLE_CONTACT_SHEET.jpg", quality=88)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if "--sheets-only" in sys.argv:
        make_sheets()
        return
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
    sampler = TerminalAuthoritySampler()
    perf.prepare_model_state(model)
    gc.collect()
    torch.cuda.synchronize()
    baseline_allocated = int(torch.cuda.memory_allocated())
    baseline_reserved = int(torch.cuda.memory_reserved())
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    sampling_wall = time.perf_counter() - started
    data = sampler.result
    captures = data.pop("captures")
    latents = {}
    for ordinal in range(4):
        if ordinal < 3:
            latents[f"INTERVAL_{ordinal:02d}_X0_G"] = captures["x0_G"][ordinal]
        latents[f"INTERVAL_{ordinal:02d}_X0_H"] = captures["assembled_x0_H"][ordinal]
        latents[f"INTERVAL_{ordinal:02d}_H_STAR"] = captures["H_star"][ordinal]
        if ordinal < 3:
            latents[f"INTERVAL_{ordinal:02d}_ACCEPTED_H"] = captures["accepted_H"][ordinal]
        else:
            latents[f"INTERVAL_{ordinal:02d}_ACCEPTED_H"] = captures["A_FINAL"][ordinal]
    latents.update({
        "TERMINAL_ACCEPTED_H_BEFORE": captures["accepted_H_before"][3],
        "TERMINAL_X0_G": captures["x0_G"][3],
        "A_PRODUCTION_TERMINAL_RELEASE_FINAL": captures["A_FINAL"][3],
        "B_TERMINAL_HARD_PROJECTION_FINAL": captures["B_FINAL"][3],
        "C_TERMINAL_RETAINED_G_PROJECTION_FINAL": captures["C_FINAL"][3],
    })
    data["configuration"] = {
        "target_pixels_hw": [2048, 4096], "H": [128, 256], "G": [96, 192],
        "prompt": PROMPT, "seed": SEED, "cfg": 1.0,
        "production_changes": False,
    }
    data["performance"] = {
        "shared_sampling_wall_seconds_including_terminal_B_global": sampling_wall,
        "baseline_allocated_bytes": baseline_allocated,
        "baseline_reserved_bytes": baseline_reserved,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    data["decoded"] = decode(latents)
    for name, key in (
        ("A_PRODUCTION_TERMINAL_RELEASE", "A_PRODUCTION_TERMINAL_RELEASE_FINAL"),
        ("B_TERMINAL_HARD_PROJECTION", "B_TERMINAL_HARD_PROJECTION_FINAL"),
        ("C_TERMINAL_RETAINED_G_PROJECTION", "C_TERMINAL_RETAINED_G_PROJECTION_FINAL"),
    ):
        data["variants"][name]["decoded"] = data["decoded"][key]
    REPORT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    make_sheets()
    print(json.dumps({
        "report": str(REPORT),
        "terminal_euler": data["terminal_euler_H_star_vs_x0_H"],
        "hashes": {name: item["decoded"]["sha256_rgb"] for name, item in data["variants"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
