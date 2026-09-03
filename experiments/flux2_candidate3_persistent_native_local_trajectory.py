"""Phase 10: persistent native-scale local trajectory falsifier."""

from __future__ import annotations

import argparse
import contextlib
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


OUTPUT = ROOT / "experiments" / "flux2_candidate3_persistent_native_local_trajectory_results"
REPORT = OUTPUT / "report.json"
H_HW = phase9b.H_HW
STEPS = phase9b.STEPS
SEED = phase9b.SEED
PROMPT = phase9b.PROMPT
REPRESENTATIVE_REGION = 7


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


def rms(value):
    return float(value.detach().float().square().mean().sqrt())


def build_initial_working(h_crop, sigma, region):
    return phase9c.make_working(h_crop, sigma, 0, region)


def euler_proposal(state, x0, sigma, sigma_next):
    return state + (sigma_next - sigma) * (state - x0) / sigma


def coarse_drift(w_states, h, regions):
    values = []
    for region, working in zip(regions, w_states):
        target = h[:, :, region.y:region.y2, region.x:region.x2]
        delta = phase9b.restrict2(working).float() - target.float()
        values.append({
            "region": region.index,
            "rms": rms(delta),
            "max_abs": float(delta.abs().max()),
        })
    return values


def detail_rms(working):
    detail = working.float() - phase9b.prolong2(phase9b.restrict2(working)).float()
    return rms(detail)


class LocalRuntime:
    def __init__(self, mode, regions, probe):
        self.mode = mode
        self.regions = regions
        self.probe = probe
        self.ordinal = 0
        self.accepted_w = None
        self.x0_w = {}
        self.working_cpu = {}
        self.input_records = []
        self.reconstruction_count = 0

    def start_interval(self, ordinal, accepted_w):
        self.ordinal = ordinal
        self.accepted_w = accepted_w
        self.x0_w = {}
        self.working_cpu = {}

    def predict_region(self, adapter, **kwargs):
        region = kwargs["region"]
        if self.mode == "reconstructed":
            working = phase9c.make_working(
                kwargs["h_view"], kwargs["sigma"], self.ordinal, region
            )
            self.reconstruction_count += 1
            provenance = "reconstructed_from_current_H"
        else:
            working = self.accepted_w[region.index]
            provenance = "persistent_accepted_W"
        before_hash = tensor_hash(working)
        options = adapter._options(kwargs["model_options"], {})
        x0_w = self.probe.timed(
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
            raise RuntimeError(f"Region {region.index} model call mutated accepted W.")
        self.x0_w[region.index] = x0_w
        self.working_cpu[region.index] = working.detach().float().cpu()
        self.input_records.append({
            "ordinal": self.ordinal,
            "region": region.index,
            "provenance": provenance,
            "python_id": id(working),
            "hash": before_hash,
            "working": summary(working),
            "detail_rms": detail_rms(working),
            "x0_W": summary(x0_w),
        })
        return phase9b.restrict2(x0_w)


@contextlib.contextmanager
def scoped_region_runtime(runtime):
    original = Flux2Adapter.predict_region
    def patched(adapter, **kwargs):
        return runtime.predict_region(adapter, **kwargs)

    Flux2Adapter.predict_region = patched
    try:
        yield
    finally:
        Flux2Adapter.predict_region = original


class PersistentTrajectorySampler(phase2.comfy.samplers.Sampler):
    def __init__(self, mode, probe):
        self.mode = mode
        self.probe = probe
        self.results = None
        self.outputs = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 10 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        regions = phase9b.DestinationPlanner().plan(H_HW)
        initial_w = [
            build_initial_working(
                state.h[:, :, region.y:region.y2, region.x:region.x2], sigmas[0], region
            )
            for region in regions
        ]
        initial_errors = coarse_drift(initial_w, state.h, regions)
        if max(item["max_abs"] for item in initial_errors) > 1e-6:
            raise RuntimeError("Phase 10 initialization violates D(W0)=H0_crop.")

        accepted_w = initial_w if self.mode in {"persistent", "coarse_sync"} else None
        probe = self.probe
        runtime = LocalRuntime(self.mode, regions, probe)
        trajectory = []
        images = {
            "accepted_H": {0: state.h.detach().float().cpu()},
            "representative_W": {},
            "restricted_W": {},
            "assembled_x0_H": {},
            "terminal_W_star": {},
            "terminal_W_synced": {},
        }
        if self.mode == "persistent":
            images["representative_W"][0] = accepted_w[REPRESENTATIVE_REGION].detach().float().cpu()
            images["restricted_W"][0] = [phase9b.restrict2(w).detach().float().cpu() for w in accepted_w]

        initialization_count = len(initial_w)
        euler_updates = [0 for _ in regions]
        accepted_h_hash = tensor_hash(state.h)
        accepted_g_hash = tensor_hash(state.g)

        with scoped_region_runtime(runtime):
            for ordinal in range(len(sigmas) - 1):
                sigma = sigmas[ordinal]
                sigma_next = sigmas[ordinal + 1]
                before_h_hash = tensor_hash(state.h)
                before_g_hash = tensor_hash(state.g)
                before_w_hashes = None if accepted_w is None else [tensor_hash(w) for w in accepted_w]
                before_w_ids = None if accepted_w is None else [id(w) for w in accepted_w]
                runtime.start_interval(ordinal, accepted_w)

                candidate_state, x0_h = coordinator.evaluate(
                    guider=model,
                    state=state,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    model_options=extra_args["model_options"],
                    seed=SEED,
                )
                if set(runtime.x0_w) != set(range(len(regions))):
                    raise RuntimeError(f"Incomplete W predictions at interval {ordinal}.")

                candidate_w = None
                lineage = []
                synchronization = []
                if self.mode == "persistent":
                    candidate_w = []
                    for region, working in zip(regions, accepted_w):
                        proposed = euler_proposal(
                            working, runtime.x0_w[region.index], sigma, sigma_next
                        )
                        if not bool(proposed.isfinite().all()):
                            raise RuntimeError(f"Nonfinite W proposal for region {region.index}.")
                        candidate_w.append(proposed)
                        lineage.append({
                            "region": region.index,
                            "parent_python_id": id(working),
                            "parent_hash": tensor_hash(working),
                            "proposal_python_id": id(proposed),
                            "proposal_hash": tensor_hash(proposed),
                            "x0_hash": tensor_hash(runtime.x0_w[region.index]),
                        })
                elif self.mode == "coarse_sync":
                    candidate_w = []
                    for region, working in zip(regions, accepted_w):
                        proposed = euler_proposal(
                            working, runtime.x0_w[region.index], sigma, sigma_next
                        )
                        target = candidate_state.h[
                            :, :, region.y:region.y2, region.x:region.x2
                        ]
                        before = phase9b.restrict2(proposed).float() - target.float()
                        correction = target - phase9b.restrict2(proposed)
                        synchronized = proposed + phase9b.prolong2(correction)
                        after = phase9b.restrict2(synchronized).float() - target.float()
                        proposed_detail = proposed.float() - phase9b.prolong2(
                            phase9b.restrict2(proposed)
                        ).float()
                        synchronized_detail = synchronized.float() - phase9b.prolong2(
                            phase9b.restrict2(synchronized)
                        ).float()
                        detail_delta = synchronized_detail - proposed_detail
                        if not bool(synchronized.isfinite().all()):
                            raise RuntimeError(f"Nonfinite synchronized W for region {region.index}.")
                        if not bool(sigma_next == 0) and float(after.abs().max()) > 1e-6:
                            raise RuntimeError(
                                f"D(W_next) != crop(H_next) for region {region.index}: "
                                f"{float(after.abs().max())}."
                            )
                        candidate_w.append(synchronized)
                        lineage.append({
                            "region": region.index,
                            "parent_python_id": id(working),
                            "parent_hash": tensor_hash(working),
                            "proposal_python_id": id(proposed),
                            "proposal_hash": tensor_hash(proposed),
                            "accepted_python_id": id(synchronized),
                            "accepted_hash": tensor_hash(synchronized),
                            "x0_hash": tensor_hash(runtime.x0_w[region.index]),
                        })
                        synchronization.append({
                            "region": region.index,
                            "drift_before_rms": rms(before),
                            "drift_before_max_abs": float(before.abs().max()),
                            "correction_rms": rms(phase9b.prolong2(correction)),
                            "correction_over_W_star_rms": (
                                rms(phase9b.prolong2(correction)) / max(rms(proposed), 1e-12)
                            ),
                            "drift_after_rms": rms(after),
                            "drift_after_max_abs": float(after.abs().max()),
                            "detail_rms_before": rms(proposed_detail),
                            "detail_rms_after": rms(synchronized_detail),
                            "detail_delta_rms": rms(detail_delta),
                            "detail_delta_max_abs": float(detail_delta.abs().max()),
                        })
                        if bool(sigma_next == 0):
                            images["terminal_W_star"][region.index] = proposed.detach().float().cpu()
                            images["terminal_W_synced"][region.index] = synchronized.detach().float().cpu()

                # Transactional gate: accepted objects are still unchanged here.
                if tensor_hash(state.h) != before_h_hash or tensor_hash(state.g) != before_g_hash:
                    raise RuntimeError("Coordinator mutated accepted H/G before commit.")
                if accepted_w is not None and [tensor_hash(w) for w in accepted_w] != before_w_hashes:
                    raise RuntimeError("A local evaluation mutated accepted W before commit.")

                predictions = [phase9b.restrict2(runtime.x0_w[index]) for index in range(len(regions))]
                overlap_prediction = phase9b.Probe.overlap(predictions, regions)
                images["assembled_x0_H"][ordinal] = x0_h.detach().float().cpu()
                images["representative_W"].setdefault(
                    ordinal, runtime.working_cpu[REPRESENTATIVE_REGION]
                )
                images["restricted_W"].setdefault(
                    ordinal,
                    [phase9b.restrict2(runtime.working_cpu[index]) for index in range(len(regions))],
                )

                # One atomic logical publication point for H/G and every W.
                state = candidate_state
                if candidate_w is not None:
                    accepted_w = candidate_w
                    for index in range(len(euler_updates)):
                        euler_updates[index] += 1

                accepted_h_hash = tensor_hash(state.h)
                accepted_g_hash = tensor_hash(state.g)
                interval_telemetry = coordinator.telemetry[-1]
                images["accepted_H"][ordinal + 1] = state.h.detach().float().cpu()
                drift = []
                w_overlap = None
                if accepted_w is not None:
                    drift = coarse_drift(accepted_w, state.h, regions)
                    restricted_states = [phase9b.restrict2(w) for w in accepted_w]
                    w_overlap = phase9b.Probe.overlap(restricted_states, regions)
                    images["representative_W"][ordinal + 1] = accepted_w[
                        REPRESENTATIVE_REGION
                    ].detach().float().cpu()
                    images["restricted_W"][ordinal + 1] = [
                        value.detach().float().cpu() for value in restricted_states
                    ]

                trajectory.append({
                    "ordinal": ordinal,
                    "sigma": float(sigma),
                    "sigma_next": float(sigma_next),
                    "accepted_H": summary(state.h),
                    "accepted_G": summary(state.g),
                    "accepted_H_hash": accepted_h_hash,
                    "accepted_G_hash": accepted_g_hash,
                    "W_lineage": lineage,
                    "synchronization": synchronization,
                    "W_input_ids": before_w_ids,
                    "W_input_hashes": before_w_hashes,
                    "W_H_drift": drift,
                    "W_overlap_rms": w_overlap,
                    "prediction_overlap_rms": overlap_prediction,
                    "coverage_min": interval_telemetry["coverage_min"],
                    "coverage_max": interval_telemetry["coverage_max"],
                    "projection_rms": interval_telemetry["projection_rms"],
                    "invariant_max_abs": interval_telemetry["invariant_max_abs"],
                    "terminal_release": interval_telemetry["terminal_release"],
                })

        self.results = {
            "variant": self.mode,
            "configuration": {
                "H": list(H_HW),
                "G": list(state.g.shape[-2:]),
                "destination_crop": [32, 32],
                "stride": 24,
                "working_canvas": [64, 64],
                "working_coordinates": "native local unit grid 0..63",
                "sigmas": sigmas.tolist(),
                "seed": SEED,
                "regions": [[r.index, r.y, r.x, r.height, r.width] for r in regions],
            },
            "initialization": {
                "W_constructed_once_count": initialization_count,
                "evaluation_reconstruction_count": runtime.reconstruction_count,
                "post_first_evaluation_reconstruction_count": (
                    max(0, runtime.reconstruction_count - len(regions))
                    if self.mode == "reconstructed" else 0
                ),
                "initial_W_H_drift": initial_errors,
            },
            "integrity": {
                "W_euler_updates_per_region": euler_updates,
                "all_W_updates_equal_interval_count": (
                    self.mode == "reconstructed" or all(x == len(sigmas) - 1 for x in euler_updates)
                ),
                "accepted_H_hash": accepted_h_hash,
                "accepted_G_hash": accepted_g_hash,
                "finite_final_H": bool(state.h.isfinite().all()),
                "complete_coverage": all(
                    abs(x - 1.0) < 2e-6 for item in trajectory
                    for x in [item["coverage_min"], item["coverage_max"]]
                ),
            },
            "work": {
                "global_forwards": sum(call["kind"] == "global" for call in probe.calls),
                "local_forwards": sum(call["kind"] == "local" for call in probe.calls),
                "global_token_executions": sum(call["tokens"] for call in probe.calls if call["kind"] == "global"),
                "local_token_executions": sum(call["tokens"] for call in probe.calls if call["kind"] == "local"),
                "tokens_per_local_forward": 4096,
                "global_cuda_ms": sum(call["cuda_ms"] for call in probe.calls if call["kind"] == "global"),
                "local_cuda_ms": sum(call["cuda_ms"] for call in probe.calls if call["kind"] == "local"),
            },
            "trajectory": trajectory,
            "final_H": summary(state.h),
        }
        self.outputs = images
        return state.h


def decode_outputs(name, outputs):
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    images = {}
    values = {}
    for ordinal, value in outputs["accepted_H"].items():
        values[f"accepted_H_{ordinal}"] = value
    for ordinal, value in outputs["assembled_x0_H"].items():
        values[f"assembled_x0_H_{ordinal}"] = value
    for ordinal, value in outputs["representative_W"].items():
        values[f"representative_W_{ordinal}"] = value
    for ordinal, states in outputs["restricted_W"].items():
        for region, value in enumerate(states):
            values[f"restricted_W_{ordinal}_region_{region:02d}"] = value
    for region, value in outputs["terminal_W_star"].items():
        values[f"terminal_W_star_region_{region:02d}"] = value
    for region, value in outputs["terminal_W_synced"].items():
        values[f"terminal_W_synced_region_{region:02d}"] = value
    for label, latent in values.items():
        with torch.inference_mode():
            pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}_{label}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            images[label] = {
                "path": str(path),
                "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
    return images


def run_variant(name, mode, *, return_outputs=False):
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
    sampler = PersistentTrajectorySampler(mode, outer_probe)
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
    sampler.results["final_H_hash"] = tensor_hash(output)
    path = OUTPUT / f"{name}.json"
    path.write_text(json.dumps(sampler.results, indent=2), encoding="utf-8")
    if return_outputs:
        return sampler.results, sampler.outputs
    return sampler.results


def comparison_sheet(results):
    sheet = Image.new("RGB", (768, 420 * len(results)), "white")
    for index, (name, result) in enumerate(results.items()):
        record = result["images"]["accepted_H_4"]
        with Image.open(record["path"]) as image:
            thumb = ImageOps.fit(image.convert("RGB"), (768, 384))
        sheet.paste(thumb, (0, index * 420 + 36))
        ImageDraw.Draw(sheet).text((10, index * 420 + 10), name, fill="black")
    sheet.save(OUTPUT / "FINAL_COMPARISON.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("A", "B", "C", "all"), default="all")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    requested = (
        (("A_RECONSTRUCTED_W", "reconstructed"), ("B_PERSISTENT_W", "persistent"))
        if args.variant == "all"
        else (("A_RECONSTRUCTED_W", "reconstructed"),) if args.variant == "A"
        else (("B_PERSISTENT_W", "persistent"),) if args.variant == "B"
        else (("C_PERSISTENT_COARSE_SYNC", "coarse_sync"),)
    )
    results = {name: run_variant(name, mode) for name, mode in requested}
    if len(results) == 2:
        comparison_sheet(results)
        REPORT.write_text(json.dumps({"variants": results}, indent=2), encoding="utf-8")
    print(json.dumps({
        name: {
            "final_H": result["final_H"],
            "overlap": [x["prediction_overlap_rms"] for x in result["trajectory"]],
            "drift_rms_max": [
                max((item["rms"] for item in step["W_H_drift"]), default=0.0)
                for step in result["trajectory"]
            ],
            "local_cuda_ms": result["work"]["local_cuda_ms"],
            "wall_seconds": result["runtime"]["sampling_wall_seconds"],
        }
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
