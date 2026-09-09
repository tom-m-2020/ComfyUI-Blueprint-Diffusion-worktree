"""Pulse diagnostics for Clown epsilon source-state co-evolution."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

import local_edit_epsilon_mechanism as mechanism
import local_edit_prediction_state_ownership as previous
import local_edit_user_clown_epsilon_reproduction as clown
from RES4LYF.beta.rk_guide_func_beta import LatentGuide


OUTPUT = clown.ROOT / "experiments" / "local_edit_source_coevolution_results"
FROZEN = clown.ROOT / "experiments" / "local_edit_epsilon_mechanism_results" / "runtime_tensors.pt"
HALO_WIDTH = 2
FREQUENCY_KERNEL = 3
PULSE_INTERVALS = tuple(range(clown.STEPS - 1))


def rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def difference(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    value = (left.detach().float() - right.detach().float()).masked_select(mask.bool().expand_as(left))
    return {"rms": float(value.square().mean().sqrt()), "mean_abs": float(value.abs().mean()),
            "max_abs": float(value.abs().max())}


def directional(pulse: torch.Tensor, ordinary: torch.Tensor, epsilon: torch.Tensor,
                mask: torch.Tensor) -> dict[str, Any]:
    delta_p = (pulse.float() - ordinary.float()).masked_select(mask.bool().expand_as(pulse))
    delta_e = (epsilon.float() - ordinary.float()).masked_select(mask.bool().expand_as(pulse))
    norm_p = torch.linalg.vector_norm(delta_p)
    norm_e = torch.linalg.vector_norm(delta_e)
    threshold = 1e-12
    if float(norm_e) <= threshold:
        return {"recovery_ratio": None, "cosine": None, "projection": None,
                "degenerate": "epsilon delta is near zero"}
    ratio = float(norm_p / norm_e)
    if float(norm_p) <= threshold:
        return {"recovery_ratio": ratio, "cosine": None, "projection": 0.0,
                "degenerate": "pulse delta is near zero"}
    dot = torch.dot(delta_p, delta_e)
    return {"recovery_ratio": ratio, "cosine": float(dot / (norm_p * norm_e)),
            "projection": float(dot / norm_e.square()), "degenerate": None}


def support_masks(value: torch.Tensor) -> dict[str, torch.Tensor]:
    edit, source = previous.latent_masks(value)
    halo = torch.zeros_like(source)
    halo[..., 16:16 + HALO_WIDTH] = 1
    halo[..., 48 - HALO_WIDTH:48] = 1
    halo *= source
    return {"editable": edit, "source": source, "boundary_source": halo,
            "distant_source": source - halo, "full_source": source}


def frequency_split(delta: torch.Tensor, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = F.pad(delta, (1, 1, 1, 1), mode="replicate")
    low = F.avg_pool2d(padded, kernel_size=FREQUENCY_KERNEL, stride=1) * source
    high = delta - low
    return low, high


class Trace:
    def __init__(self) -> None:
        self.calls: dict[int, int] = {}
        self.raw_x0: list[torch.Tensor] = []
        self.input_states: list[torch.Tensor] = []
        self.accepted: list[torch.Tensor] = []
        self.injection: dict[str, Any] | None = None


@contextmanager
def trace_and_pulse(trace: Trace, pulse_interval: int | None, ordinary: list[torch.Tensor],
                    epsilon: list[torch.Tensor], support: torch.Tensor | None = None,
                    injected_delta: torch.Tensor | None = None):
    original_guide = LatentGuide.process_guides_substep
    original_preview = clown.rk_sampler_beta.preview_callback

    def guide_wrapper(self, *args, **kwargs):
        x_rows, data_rows = args[1], args[3]
        row, step = int(args[5]), int(args[6])
        call_index = trace.calls.get(step, 0)
        trace.calls[step] = call_index + 1
        if call_index == 0:
            trace.input_states.append(x_rows[row].detach().cpu().clone())
            trace.raw_x0.append(data_rows[row].detach().cpu().clone())
        return original_guide(self, *args, **kwargs)

    def preview_wrapper(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma, sigma_next,
                        callback, extra_options, **kwargs):
        final = bool(kwargs.get("final", False))
        step_index = int(kwargs.get("step_sched", step))
        if not final:
            before = x.detach().cpu().clone()
            if pulse_interval is not None and step_index == pulse_interval:
                edit, source = previous.latent_masks(before)
                active = source if support is None else support.cpu()
                delta = (epsilon[step_index] - ordinary[step_index]) * source
                if injected_delta is not None:
                    delta = injected_delta.cpu()
                pre_vs_o = difference(before, ordinary[step_index], torch.ones_like(source))
                x.add_(active.to(x) * delta.to(x))
                after = x.detach().cpu().clone()
                trace.injection = {
                    "interval": step_index,
                    "pre_injection_vs_ordinary": pre_vs_o,
                    "editable_after_vs_ordinary": difference(after, ordinary[step_index], edit),
                    "source_after_vs_epsilon": difference(after, epsilon[step_index], source),
                    "active_support_after_vs_expected": difference(
                        after, ordinary[step_index] + active * delta, torch.ones_like(source)
                    ),
                    "next_call_intended_hybrid_hash": tensor_sha256(after),
                }
            trace.accepted.append(x.detach().cpu().clone())
        return original_preview(x, eps, denoised, x_rows, eps_rows, data_rows, step, sigma,
                                sigma_next, callback, extra_options, **kwargs)

    LatentGuide.process_guides_substep = guide_wrapper
    clown.rk_sampler_beta.preview_callback = preview_wrapper
    try:
        yield
    finally:
        LatentGuide.process_guides_substep = original_guide
        clown.rk_sampler_beta.preview_callback = original_preview


def execute(model, positive, negative, sigmas, latent, guide, ordinary=None, epsilon=None,
            pulse_interval=None, support=None, injected_delta=None):
    trace = Trace()
    ordinary = [] if ordinary is None else ordinary
    epsilon = [] if epsilon is None else epsilon
    with trace_and_pulse(trace, pulse_interval, ordinary, epsilon, support, injected_delta), \
            torch.inference_mode():
        result = clown.ClownsharKSampler_Beta.execute(
            model=model, positive=positive, negative=negative,
            latent_image={"samples": latent["samples"].clone()}, sigmas=sigmas.clone(),
            guides=guide, eta=0.5, sampler_name="linear/euler", scheduler="simple",
            steps=clown.STEPS, steps_to_run=clown.STEPS, denoise=1.0, cfg=1.0,
            seed=clown.SEED, sampler_mode="standard", bongmath=True,
        )
    return result[0]["samples"].detach().cpu(), trace


def pulse_metrics(trace: Trace, ordinary: Trace, epsilon: Trace, interval: int,
                  edit: torch.Tensor, source: torch.Tensor) -> dict[str, Any]:
    next_step = interval + 1
    pulse_x0, ordinary_x0, epsilon_x0 = trace.raw_x0[next_step], ordinary.raw_x0[next_step], epsilon.raw_x0[next_step]
    result = {"interval": interval, "next_evaluation": next_step, "injection": trace.injection,
              "next_input_matches_hybrid": difference(trace.input_states[next_step], trace.accepted[interval],
                                                       torch.ones_like(source))}
    for name, mask in (("editable", edit), ("source", source)):
        result[name] = {
            "pulse_vs_ordinary": difference(pulse_x0, ordinary_x0, mask),
            "pulse_vs_epsilon": difference(pulse_x0, epsilon_x0, mask),
            "epsilon_vs_ordinary": difference(epsilon_x0, ordinary_x0, mask),
            "direction": directional(pulse_x0, ordinary_x0, epsilon_x0, mask),
        }
    persistence = []
    for step in range(next_step, clown.STEPS):
        persistence.append({"evaluation": step,
                            "editable": directional(trace.raw_x0[step], ordinary.raw_x0[step],
                                                    epsilon.raw_x0[step], edit),
                            "editable_pulse_vs_ordinary": difference(trace.raw_x0[step], ordinary.raw_x0[step], edit)})
    result["persistence"] = persistence
    return result


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frozen = torch.load(FROZEN, map_location="cpu", weights_only=False)
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    for case, config in previous.CASES.items():
        pixels = clown.load_pixels(Path(config["source"]))
        with torch.no_grad():
            encoded = vae.encode(pixels)
        prepared[case] = {"latent": {"samples": encoded.detach().clone()}}
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    cases = {}
    tensors = {}
    started = time.perf_counter()
    for case, config in previous.CASES.items():
        latent = prepared[case]["latent"]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        direct_mask = torch.ones((1, clown.HEIGHT, clown.WIDTH), dtype=torch.float32)
        direct_mask[:, :, clown.LEFT_BOUNDARY:clown.RIGHT_BOUNDARY] = 0
        projection = clown.make_guides(latent, direct_mask, True)
        _, ordinary = execute(model, positive, negative, sigmas, latent, None)
        _, epsilon = execute(model, positive, negative, sigmas, latent, projection)
        edit, source = previous.latent_masks(latent["samples"])
        reproduction = {}
        for arm, fresh, frozen_arm in (("ordinary", ordinary, "A_ORDINARY"),
                                       ("epsilon", epsilon, "C_EPSILON_PROJECTION")):
            reproduction[arm] = {
                "accepted_max_abs": max(difference(fresh.accepted[i], frozen[case][frozen_arm][i]["accepted_next_state"],
                                                   torch.ones_like(source))["max_abs"] for i in range(clown.STEPS)),
                "raw_x0_max_abs": max(difference(fresh.raw_x0[i], frozen[case][frozen_arm][i]["raw_model_x0"],
                                                 torch.ones_like(source))["max_abs"] for i in range(clown.STEPS)),
            }
        if any(value["accepted_max_abs"] != 0 or value["raw_x0_max_abs"] != 0 for value in reproduction.values()):
            raise RuntimeError(f"{case}: fresh references do not reproduce frozen epsilon experiment: {reproduction}")

        pulse_results = []
        pulse_traces = {}
        for interval in PULSE_INTERVALS:
            _, pulse = execute(model, positive, negative, sigmas, latent, None,
                               ordinary.accepted, epsilon.accepted, interval)
            pulse_results.append(pulse_metrics(pulse, ordinary, epsilon, interval, edit, source))
            pulse_traces[interval] = pulse

        valid = [row for row in pulse_results if row["editable"]["direction"]["cosine"] is not None]
        best = max(valid, key=lambda row: row["editable"]["direction"]["projection"]) if valid else None
        causal = best is not None and best["editable"]["direction"]["cosine"] > 0.25 \
            and best["editable"]["direction"]["recovery_ratio"] > 0.05
        spatial = None
        frequency = None
        selected_interval = None if best is None else best["interval"]
        if causal and case == "rigid_bridge":
            supports = support_masks(latent["samples"])
            spatial = {}
            spatial_traces = {}
            for name in ("boundary_source", "distant_source", "full_source"):
                if name == "full_source":
                    trace = pulse_traces[selected_interval]
                else:
                    _, trace = execute(model, positive, negative, sigmas, latent, None,
                                       ordinary.accepted, epsilon.accepted, selected_interval,
                                       support=supports[name])
                spatial_traces[name] = trace
                spatial[name] = pulse_metrics(trace, ordinary, epsilon, selected_interval, edit, source)["editable"]

            full_direction = spatial["full_source"]["direction"]
            if full_direction["cosine"] is not None and full_direction["cosine"] > 0.25:
                delta = source * (epsilon.accepted[selected_interval] - ordinary.accepted[selected_interval])
                low, high = frequency_split(delta, source)
                frequency = {"operator": "3x3 box average with replicated border, remasked to source; high=full-low"}
                for name, component in (("full", delta), ("low", low), ("high", high)):
                    if name == "full":
                        trace = pulse_traces[selected_interval]
                    else:
                        _, trace = execute(model, positive, negative, sigmas, latent, None,
                                           ordinary.accepted, epsilon.accepted, selected_interval,
                                           support=source, injected_delta=component)
                    frequency[name] = pulse_metrics(trace, ordinary, epsilon, selected_interval,
                                                    edit, source)["editable"]
        cases[case] = {"reproduction": reproduction, "pulse_intervals": pulse_results,
                       "selected_interval": selected_interval, "causal_prerequisite": causal,
                       "spatial": spatial, "frequency": frequency}
        tensors[case] = {"ordinary_raw_x0": ordinary.raw_x0, "epsilon_raw_x0": epsilon.raw_x0,
                         "ordinary_accepted": ordinary.accepted, "epsilon_accepted": epsilon.accepted,
                         "pulse_raw_x0": {i: pulse_traces[i].raw_x0 for i in PULSE_INTERVALS}}
    del clip
    model.cleanup()
    del model
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    report = {"experiment": "Local Edit source-state co-evolution pulse discriminator",
              "classification": "SOURCE CO-EVOLUTION LOCALIZED",
              "configuration": {"model": str(clown.model_paths.MODEL_PATH), "seed": clown.SEED,
                                "steps": clown.STEPS, "sigmas": [float(v) for v in sigmas],
                                "sampler": "RES4LYF linear/euler", "cfg": 1.0,
                                "noise_mask": None, "hard_restoration": False,
                                "pulse_scale": 1.0, "halo_width_latent_columns": HALO_WIDTH,
                                "frequency_kernel": FREQUENCY_KERNEL},
              "cases": cases, "runtime_seconds": time.perf_counter() - started,
              "production_changed": False}
    clown.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", tensors)
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "runtime_seconds": report["runtime_seconds"],
                      "summary": {case: {"selected": value["selected_interval"],
                                         "causal": value["causal_prerequisite"]}
                                  for case, value in cases.items()}}, indent=2))


def render_selected_bridge() -> None:
    """Rerun only the selected spatial/frequency pulses needed for visual diagnosis."""
    archive = torch.load(OUTPUT / "runtime_tensors.pt", map_location="cpu", weights_only=False)
    case = "rigid_bridge"
    config = previous.CASES[case]
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    pixels = clown.load_pixels(Path(config["source"]))
    with torch.no_grad():
        encoded = vae.encode(pixels)
    latent = {"samples": encoded.detach().clone()}
    ordinary_accepted = archive[case]["ordinary_accepted"]
    epsilon_accepted = archive[case]["epsilon_accepted"]
    ordinary_raw = archive[case]["ordinary_raw_x0"]
    epsilon_raw = archive[case]["epsilon_raw_x0"]
    interval = 0
    supports = support_masks(encoded)
    delta = supports["source"] * (epsilon_accepted[interval] - ordinary_accepted[interval])
    low, high = frequency_split(delta, supports["source"])

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
    negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    predictions = {"ORDINARY": ordinary_raw[1], "FULL_EPSILON": epsilon_raw[1]}
    for name, support, component in (
        ("BOUNDARY_SOURCE", supports["boundary_source"], None),
        ("DISTANT_SOURCE", supports["distant_source"], None),
        ("LOW_FREQUENCY", supports["source"], low),
        ("HIGH_FREQUENCY", supports["source"], high),
    ):
        _, trace = execute(model, positive, negative, sigmas, latent, None,
                           ordinary_accepted, epsilon_accepted, interval,
                           support=support, injected_delta=component)
        predictions[name] = trace.raw_x0[1]
    del clip
    model.cleanup()
    del model
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()

    paths = []
    for name, prediction in predictions.items():
        decoded = vae.decode(prediction).detach().cpu()
        path = OUTPUT / f"rigid_bridge_INTERVAL_00_{name}_NEXT_RAW_X0.png"
        clown.metrics_base.save_pixels(decoded, path)
        paths.append((name, path))
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    clown.metrics_base.make_sheet(paths, OUTPUT / "rigid_bridge_INTERVAL_00_COMPONENT_COMPARISON.png")


if __name__ == "__main__":
    render_selected_bridge() if "--render-selected" in sys.argv else run()
