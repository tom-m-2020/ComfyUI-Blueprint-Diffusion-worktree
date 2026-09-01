"""Phase 7a: experiment-only Candidate-3 step-count generalization."""

from __future__ import annotations

import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d
import flux2_candidate3_global_refresh_cadence as phase6i
import flux2_candidate3_performance_characterization as perf

from blueprint_diffusion.adapters.flux2 import Flux2Adapter
from blueprint_diffusion.sampling.euler import BlueprintCoordinator


OUTPUT = ROOT / "experiments" / "flux2_candidate3_step_count_generalization_results"
STEPS = (4, 8, 12, 20)
CASES = phase6i.CASES


def stats(value: torch.Tensor) -> dict:
    work = value.float()
    return {
        "rms": float(work.square().mean().sqrt()),
        "mean": float(work.mean()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
    }


def validate_experiment_schedule(sigmas: torch.Tensor, steps: int) -> None:
    if sigmas.ndim != 1 or sigmas.numel() != steps + 1:
        raise ValueError(f"Expected {steps + 1} sigma values.")
    values = sigmas.detach().float().cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Schedule is nonfinite.")
    if float(values[0]) != 1.0 or float(values[-1]) != 0.0:
        raise ValueError("Schedule must begin at 1 and terminate at exact zero.")
    if bool((values[:-1] <= 0).any()) or bool((values[1:] >= values[:-1]).any()):
        raise ValueError("Schedule must be positive then strictly decreasing.")


class CountingAdapter:
    def __init__(self) -> None:
        self.inner = Flux2Adapter()
        self.global_calls = 0
        self.local_calls = 0

    def validate_run(self, **kwargs):
        return self.inner.validate_run(**kwargs)

    def predict_global(self, **kwargs):
        self.global_calls += 1
        return self.inner.predict_global(**kwargs)

    def predict_region(self, **kwargs):
        self.local_calls += 1
        return self.inner.predict_region(**kwargs)

    def describe_work(self, **kwargs):
        return self.inner.describe_work(**kwargs)


class DiagnosticAssembler:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.overlap = []

    def assemble(self, predictions, regions, canvas):
        self.overlap.append(phase2d.overlap_disagreement(predictions, list(regions)))
        return self.inner.assemble(predictions, regions, canvas)


class StepGeneralizationSampler(perf.MeasuredSamplerBase):
    def __init__(self, steps: int) -> None:
        super().__init__()
        self.steps = steps

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 7a requires empty-latent T2I.")
        validate_experiment_schedule(sigmas, self.steps)
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)

        captures = {}

        def capture(name, ordinal, value):
            if ordinal >= 0:
                captures.setdefault(ordinal, {})[name] = stats(value)

        coordinator = BlueprintCoordinator(capture=capture)
        counting = CountingAdapter()
        coordinator.adapter = counting
        diagnostic_assembler = DiagnosticAssembler(coordinator.assembler)
        coordinator.assembler = diagnostic_assembler
        state = coordinator.initialize(h, sigmas[0])
        intervals = []
        for ordinal in range(self.steps):
            with recorder.cuda("blueprint_interval", ordinal):
                state, x0_h = coordinator.evaluate(
                    guider=model,
                    state=state,
                    sigma=sigmas[ordinal],
                    sigma_next=sigmas[ordinal + 1],
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed", 0),
                )
            accepted = coordinator.telemetry[-1]
            h_star_rms = captures[ordinal]["H_star"]["rms"]
            projection_rms = accepted["projection_rms"]
            intervals.append({
                "ordinal": ordinal,
                "sigma": float(sigmas[ordinal]),
                "sigma_next": float(sigmas[ordinal + 1]),
                "terminal_release": accepted["terminal_release"],
                "global_forward_performed": accepted["global_forward_performed"],
                "terminal_global_unused": accepted["terminal_global_unused"],
                "global_synchronized": accepted["global_synchronized"],
                "invariant_max_abs": accepted["invariant_max_abs"],
                "projection_rms": projection_rms,
                "projection_to_H_star_rms": (
                    None if projection_rms is None else projection_rms / h_star_rms
                ),
                "global_prediction": captures[ordinal].get("x0_G"),
                "local_assembled_prediction": captures[ordinal]["assembled_x0_H"],
                "H_star": captures[ordinal]["H_star"],
                "accepted_H": captures[ordinal]["accepted_H"],
                "overlap_disagreement": diagnostic_assembler.overlap[ordinal],
            })
            if callback is not None:
                callback(ordinal, x0_h, state.h, self.steps)
                self.preview_count += 1

        with recorder.cuda("inverse_noise_scaling", None):
            output = model_sampling.inverse_noise_scaling(sigmas[-1], state.h)
        self.finish(recorder, started)
        terminal_flags = [item["terminal_release"] for item in intervals]
        global_flags = [item["global_forward_performed"] for item in intervals]
        self.measurement.update({
            "steps": self.steps,
            "global_forward_count": counting.global_calls,
            "local_forward_count": counting.local_calls,
            "intervals": intervals,
            "final_latent": stats(output),
            "integrity": {
                "status": "SUCCESS",
                "finite_final": bool(torch.isfinite(output).all()),
                "expected_previews": self.preview_count == self.steps,
                "nonterminal_invariants": all(
                    item["invariant_max_abs"] is None
                    or item["invariant_max_abs"] <= coordinator.geometry.TOLERANCE
                    for item in intervals
                ),
                "terminal_release_only_final": terminal_flags
                == [False] * (self.steps - 1) + [True],
                "only_final_omits_global": global_flags
                == [True] * (self.steps - 1) + [False],
                "global_forward_count": counting.global_calls == self.steps - 1,
                "local_forward_count": counting.local_calls
                == self.steps * len(coordinator.planner.plan(tuple(h.shape[-2:]))),
            },
        })
        return output


class DenseLongEulerSampler(perf.MeasuredSamplerBase):
    def __init__(self, steps: int) -> None:
        super().__init__()
        self.steps = steps

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 7a dense reference requires empty-latent T2I.")
        validate_experiment_schedule(sigmas, self.steps)
        model_sampling = model.inner_model.model_sampling
        recorder, started = self.begin()
        with recorder.cuda("initial_noise_scaling", None):
            h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, True)
        intervals = []
        for ordinal in range(self.steps):
            sigma, sigma_next = sigmas[ordinal], sigmas[ordinal + 1]
            with recorder.cuda("dense_forward", ordinal):
                x0 = model(
                    h, sigma.expand(h.shape[0]),
                    model_options=extra_args["model_options"],
                    seed=extra_args.get("seed", 0),
                )
            with recorder.cuda("dense_euler", ordinal):
                h = h + (h - x0) / sigma * (sigma_next - sigma)
            if not bool(torch.isfinite(h).all()):
                raise RuntimeError(f"Dense state became nonfinite at {ordinal}.")
            intervals.append({
                "ordinal": ordinal, "sigma": float(sigma),
                "sigma_next": float(sigma_next), "prediction": stats(x0),
                "accepted_H": stats(h),
            })
            if callback is not None:
                callback(ordinal, x0, h, self.steps)
                self.preview_count += 1
        output = model_sampling.inverse_noise_scaling(sigmas[-1], h)
        self.finish(recorder, started)
        self.measurement.update({
            "steps": self.steps,
            "dense_forward_count": self.steps,
            "intervals": intervals,
            "final_latent": stats(output),
            "integrity": {
                "status": "SUCCESS", "finite_final": bool(torch.isfinite(output).all()),
                "expected_previews": self.preview_count == self.steps,
                "dense_forward_count": self.steps,
            },
        })
        return output


def run(model, positive, negative, noise, sigmas, sampler, seed):
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=seed,
        )
    torch.cuda.synchronize()
    return output.detach().cpu(), sampler.measurement


def save_contact_sheet(case_name: str, image_paths: dict) -> Path:
    images = {(variant, steps): Image.open(path).convert("RGB") for (variant, steps), path in image_paths.items()}
    thumb_w = 512
    first = next(iter(images.values()))
    thumb_h = round(first.height * thumb_w / first.width)
    label_h = 30
    sheet = Image.new("RGB", (thumb_w * len(STEPS), (thumb_h + label_h) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for row, variant in enumerate(("dense", "blueprint")):
        for col, steps in enumerate(STEPS):
            image = images[(variant, steps)]
            image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            x, y = col * thumb_w, row * (thumb_h + label_h)
            sheet.paste(image, (x, y + label_h))
            draw.text((x + 8, y + 7), f"{variant.upper()} {steps} steps", fill="black")
    path = OUTPUT / f"{case_name}_contact_sheet.png"
    sheet.save(path)
    for image in images.values():
        image.close()
    return path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    encoded = {}
    for name, _, _, prompt, _ in CASES:
        encoded[name] = (
            clip.encode_from_tokens_scheduled(clip.tokenize(prompt)),
            clip.encode_from_tokens_scheduled(clip.tokenize("")),
        )
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH), "cfg": 1.0,
            "steps": list(STEPS), "sampler": "Euler CONST flow",
            "production_changes": False,
            "algorithm": "current production Candidate-3 including terminal global omission",
        },
        "cases": {},
    }
    outputs = {}
    warmed_shapes = set()
    for name, width, height, prompt, seed in CASES:
        positive, negative = encoded[name]
        h_shape = (height // 16, width // 16)
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(seed))
        if h_shape not in warmed_shapes:
            perf.prepare_model_state(model)
            warm_sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
            print(f"warmup {h_shape}: dense + blueprint", flush=True)
            warm, _ = run(model, positive, negative, noise, warm_sigmas, DenseLongEulerSampler(4), seed)
            del warm
            warm, _ = run(model, positive, negative, noise, warm_sigmas, StepGeneralizationSampler(4), seed)
            del warm
            warmed_shapes.add(h_shape)
        case = {
            "target_pixels_wh": [width, height], "H_shape": list(h_shape),
            "prompt": prompt, "seed": seed, "schedules": {},
        }
        for steps in STEPS:
            sigmas = phase2.get_schedule(steps, math.prod(h_shape)).float().clone()
            validate_experiment_schedule(sigmas, steps)
            item = {"sigmas": sigmas.tolist(), "variants": {}}
            for variant in ("dense", "blueprint"):
                print(f"{name}: {steps} steps {variant}", flush=True)
                sampler = DenseLongEulerSampler(steps) if variant == "dense" else StepGeneralizationSampler(steps)
                output, measurement = run(
                    model, positive, negative, noise, sigmas, sampler, seed
                )
                outputs[(name, variant, steps)] = output
                item["variants"][variant] = measurement
            case["schedules"][str(steps)] = item
            report["cases"][name] = case
            (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    del encoded
    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = {}
    sheets = {}
    for case_name, *_ in CASES:
        paths = {}
        for variant in ("dense", "blueprint"):
            for steps in STEPS:
                latent = outputs[(case_name, variant, steps)]
                with torch.inference_mode():
                    pixels = vae.decode(latent).cpu()
                path = OUTPUT / f"{case_name}_{steps:02d}_{variant}.png"
                phase2.save_pixels(pixels, path)
                decoded[f"{case_name}_{steps}_{variant}"] = {
                    "path": str(path), "shape": list(pixels.shape),
                    "finite": bool(torch.isfinite(pixels).all()),
                }
                paths[(variant, steps)] = path
        sheets[case_name] = str(save_contact_sheet(case_name, paths))
    report["decoded_outputs"] = decoded
    report["contact_sheets"] = sheets
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        case_name: {
            steps: {
                variant: round(case["schedules"][str(steps)]["variants"][variant]["sampling_wall_seconds"], 3)
                for variant in ("dense", "blueprint")
            }
            for steps in STEPS
        }
        for case_name, case in report["cases"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
