"""Phase 17: external K/V versus bidirectional joint source/W interaction."""

from __future__ import annotations

import gc
import hashlib
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
import flux2_candidate2_one_eval_probe as candidate2
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_fixed4k_source_statistics as phase15
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_specialized_executor as phase8i
import flux2_candidate3_terminal_context as phase8d

from comfy.ldm.flux import math as flux_math
from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_fixed4k_consumer_interface_results"
REPORT = OUTPUT / "report.json"
TEXT_TOKENS = 512
SOURCE_TOKENS = 4096
W_TOKENS = 4096
REPRESENTATIVE_REGION = 27


def tensor_difference(value, reference):
    delta = value.detach().float() - reference.detach().float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "left_rms": float(value.detach().float().square().mean().sqrt()),
        "right_rms": float(reference.detach().float().square().mean().sqrt()),
    }


def state_rms(value):
    return float(value.detach().float().square().mean().sqrt())


class ExternalRecordingProbe(phase8i.OneBlockGPUContext):
    def __init__(self):
        super().__init__()
        self.local_current = None

    def add_global_context(self, q, k, v, pe, attn_mask, extra_options):
        text_tokens, sequence_end = map(int, extra_options["img_slice"])
        positioned_k = flux_math.apply_rope1(k, pe)
        self.local_current = {
            "k": positioned_k[:, :, text_tokens:sequence_end],
            "v": v[:, :, text_tokens:sequence_end],
        }
        return super().add_global_context(q, k, v, pe, attn_mask, extra_options)

    def release(self):
        self.local_current = None
        super().release()


class JointRecordingPatch:
    def __init__(self):
        self.current = None

    def __call__(self, q, k, v, pe, attn_mask, extra_options):
        positioned_k = flux_math.apply_rope1(k, pe)
        if k.shape[2] != TEXT_TOKENS + SOURCE_TOKENS + W_TOKENS:
            raise AssertionError(f"Unexpected joint tokens: {k.shape[2]}")
        source_slice = slice(TEXT_TOKENS, TEXT_TOKENS + SOURCE_TOKENS)
        w_slice = slice(TEXT_TOKENS + SOURCE_TOKENS, None)
        self.current = {
            "source_k": positioned_k[:, :, source_slice],
            "source_v": v[:, :, source_slice],
            "W_k": positioned_k[:, :, w_slice],
            "W_v": v[:, :, w_slice],
        }
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}

    def release(self):
        self.current = None


def copy_options_with_patch(options, patch=None):
    copied = options.copy()
    patches = {key: list(value) for key, value in options.get("patches", {}).items()}
    copied["patches"] = patches
    if patch is not None:
        patches.setdefault("attn1_patch", []).append(patch)
    return copied


def joint_state(source_state, local_state, patch=None):
    if source_state.txt.shape != local_state.txt.shape:
        raise AssertionError("Source/local text shapes differ.")
    if float((source_state.txt.float() - local_state.txt.float()).abs().max()) != 0.0:
        raise AssertionError("Prepared source/local text is not identical.")
    if float((source_state.vec_orig.float() - local_state.vec_orig.float()).abs().max()) != 0.0:
        raise AssertionError("Prepared source/local modulation differs.")
    text_pe = source_state.pe[:, :, :TEXT_TOKENS]
    source_pe = source_state.pe[:, :, TEXT_TOKENS:]
    local_pe = local_state.pe[:, :, TEXT_TOKENS:]
    pe = torch.cat((text_pe, source_pe, local_pe), dim=2)
    return phase8i.ExplicitState(
        img=torch.cat((source_state.img, local_state.img), dim=1),
        txt=local_state.txt,
        vec_orig=local_state.vec_orig,
        double_vec=local_state.double_vec,
        single_vec=local_state.single_vec,
        pe=pe,
        options=copy_options_with_patch(local_state.options, patch),
        attn_mask=local_state.attn_mask,
    )


class Phase17Sampler(phase14.Phase14Sampler):
    @staticmethod
    def _capture_plain(model, value, sigma, base, diffusion, rope, role):
        capture = phase8i.ForwardCapture(diffusion, run_native=False)
        with capture:
            model(
                value, sigma.expand(1),
                model_options=phase8i.merged_options(base, rope, None, "ordinary", role),
                seed=phase14.SEED,
            )
        if capture.inputs is None:
            raise RuntimeError(f"{role} did not reach native FLUX.")
        return capture.inputs

    def _representative_comparison(self, coordinator, model, source, working, sigma,
                                   base, regions, diffusion):
        executor = phase8i.ExplicitFluxExecutor(diffusion)
        region = regions[REPRESENTATIVE_REGION]
        value = working[REPRESENTATIVE_REGION]

        external_probe = ExternalRecordingProbe()
        source_inputs, _ = self._capture_source(
            coordinator, model, source, sigma, base, diffusion,
            external_probe, "fixed_H",
        )
        local_inputs = self._capture_local(
            model, value, sigma, base, diffusion, external_probe, region
        )
        external_source = executor.prepare(source_inputs)
        external_local = executor.prepare(local_inputs)

        plain_source_inputs = self._capture_plain(
            model, source, sigma, base, diffusion,
            {"scale_y": 4.0, "shift_y": 1.5, "scale_x": 2.0, "shift_x": 0.5},
            "phase17_joint_source",
        )
        plain_local_inputs = self._capture_plain(
            model, value, sigma, base, diffusion, {}, "phase17_joint_W"
        )
        recorder = JointRecordingPatch()
        joint = joint_state(
            executor.prepare(plain_source_inputs), executor.prepare(plain_local_inputs), recorder
        )
        records = []
        for kind, count in (("double", len(diffusion.double_blocks)),
                            ("single", len(diffusion.single_blocks))):
            if kind == "single":
                executor.enter_single(external_source)
                executor.enter_single(external_local)
                executor.enter_single(joint)
            for index in range(count):
                if kind == "double":
                    executor.double(external_source, index)
                    executor.double(external_local, index)
                    executor.double(joint, index)
                    joint_source_hidden = joint.img[:, :SOURCE_TOKENS]
                    joint_w_hidden = joint.img[:, SOURCE_TOKENS:]
                    external_source_hidden = external_source.img
                    external_w_hidden = external_local.img
                else:
                    executor.single(external_source, index)
                    executor.single(external_local, index)
                    executor.single(joint, index)
                    joint_source_hidden = joint.img[:, TEXT_TOKENS:TEXT_TOKENS + SOURCE_TOKENS]
                    joint_w_hidden = joint.img[:, TEXT_TOKENS + SOURCE_TOKENS:]
                    external_source_hidden = external_source.img[:, TEXT_TOKENS:]
                    external_w_hidden = external_local.img[:, TEXT_TOKENS:]
                key = (kind, index)
                source_entry = external_probe.global_kv[key]
                if external_probe.local_current is None or recorder.current is None:
                    raise AssertionError(f"Missing K/V capture at {key}")
                records.append({
                    "block_type": kind, "block_index": index,
                    "hidden": {
                        "source_joint_vs_external": tensor_difference(
                            joint_source_hidden, external_source_hidden
                        ),
                        "W_joint_vs_external": tensor_difference(
                            joint_w_hidden, external_w_hidden
                        ),
                        "joint_source_rms": state_rms(joint_source_hidden),
                        "joint_W_rms": state_rms(joint_w_hidden),
                    },
                    "K": {
                        "source_joint_vs_external": tensor_difference(
                            recorder.current["source_k"], source_entry["k"]
                        ),
                        "W_joint_vs_external": tensor_difference(
                            recorder.current["W_k"], external_probe.local_current["k"]
                        ),
                    },
                    "V": {
                        "source_joint_vs_external": tensor_difference(
                            recorder.current["source_v"], source_entry["v"]
                        ),
                        "W_joint_vs_external": tensor_difference(
                            recorder.current["W_v"], external_probe.local_current["v"]
                        ),
                    },
                })
                external_probe.release()
                recorder.release()
        external_raw = executor.final(external_local)
        external_x0 = phase8i.raw_to_x0(external_raw, value, sigma, diffusion)
        joint_w_hidden = joint.img[:, TEXT_TOKENS + SOURCE_TOKENS:]
        joint_raw = diffusion.final_layer(joint_w_hidden, joint.vec_orig)
        joint_x0 = phase8i.raw_to_x0(joint_raw, value, sigma, diffusion)
        return {
            "region": {
                "index": region.index, "y": region.y, "x": region.x,
                "height": region.height, "width": region.width,
            },
            "per_block": records,
            "prediction_joint_vs_external": tensor_difference(joint_x0, external_x0),
            "external_prediction": phase14.summary(external_x0),
            "joint_prediction": phase14.summary(joint_x0),
            "joint_source_final_projection": False,
            "joint_W_final_projection": True,
        }

    def _all_joint(self, model, source, working, sigma, base, regions, diffusion):
        executor = phase8i.ExplicitFluxExecutor(diffusion)
        source_inputs = self._capture_plain(
            model, source, sigma, base, diffusion,
            {"scale_y": 4.0, "shift_y": 1.5, "scale_x": 2.0, "shift_x": 0.5},
            "phase17_all_joint_source",
        )
        source_prepared = executor.prepare(source_inputs)
        local_inputs = [
            self._capture_plain(
                model, value, sigma, base, diffusion, {}, f"phase17_all_joint_W_{region.index}"
            )
            for value, region in zip(working, regions)
        ]
        local_prepared = [executor.prepare(item) for item in local_inputs]
        states = [joint_state(source_prepared, local) for local in local_prepared]
        del local_inputs, local_prepared, source_inputs, source_prepared
        gc.collect()
        torch.cuda.synchronize()
        baseline_allocated = int(torch.cuda.memory_allocated())
        baseline_reserved = int(torch.cuda.memory_reserved())
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        wall_started = time.perf_counter()
        barriers = []
        for kind, count in (("double", len(diffusion.double_blocks)),
                            ("single", len(diffusion.single_blocks))):
            if kind == "single":
                for state in states:
                    executor.enter_single(state)
            for index in range(count):
                for state in states:
                    if kind == "double":
                        executor.double(state, index)
                    else:
                        executor.single(state, index)
                representative = states[REPRESENTATIVE_REGION]
                if kind == "double":
                    source_hidden = representative.img[:, :SOURCE_TOKENS]
                    w_hidden = representative.img[:, SOURCE_TOKENS:]
                else:
                    source_hidden = representative.img[:, TEXT_TOKENS:TEXT_TOKENS + SOURCE_TOKENS]
                    w_hidden = representative.img[:, TEXT_TOKENS + SOURCE_TOKENS:]
                torch.cuda.synchronize()
                barriers.append({
                    "block_type": kind, "block_index": index,
                    "representative_source_hidden_rms": state_rms(source_hidden),
                    "representative_W_hidden_rms": state_rms(w_hidden),
                    "allocated_bytes": int(torch.cuda.memory_allocated()),
                    "reserved_bytes": int(torch.cuda.memory_reserved()),
                })
        end.record()
        torch.cuda.synchronize()
        joint_transformer_ms = float(start.elapsed_time(end))
        final_start = torch.cuda.Event(enable_timing=True)
        final_end = torch.cuda.Event(enable_timing=True)
        final_start.record()
        x0_w = []
        for value, state in zip(working, states):
            w_hidden = state.img[:, TEXT_TOKENS + SOURCE_TOKENS:]
            raw = diffusion.final_layer(w_hidden, state.vec_orig)
            x0_w.append(phase8i.raw_to_x0(raw, value, sigma, diffusion))
        final_end.record()
        torch.cuda.synchronize()
        restricted = [phase9b.restrict2(value) for value in x0_w]
        coordinator = BlueprintCoordinator()
        assembled, coverage = coordinator.assembler.assemble(restricted, regions, phase14.H_HW)
        return assembled, restricted, {
            "joint_transformer_cuda_ms": joint_transformer_ms,
            "W_final_projection_cuda_ms": float(final_start.elapsed_time(final_end)),
            "wall_seconds": time.perf_counter() - wall_started,
            "baseline_allocated_bytes": baseline_allocated,
            "baseline_reserved_bytes": baseline_reserved,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "barriers": barriers,
            "source_local_cuda_separable": False,
            "source_local_cuda_note": (
                "Native joint attention and token-linear execution update source and W in one block; "
                "separate source/local CUDA attribution would be fictitious."
            ),
            "coverage": [float(coverage.min()), float(coverage.max())],
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 17 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1], model_options=extra_args["model_options"],
                seed=phase14.SEED,
            )
        sigma = sigmas[3]
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55 or float(sigmas[4]) != 0.0:
            raise AssertionError("Phase 17 geometry/schedule mismatch.")
        h_hash, g_hash = phase14.tensor_hash(state.h), phase14.tensor_hash(state.g)
        working = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(view, sigma, 3, region)
            if float((phase9b.restrict2(value).float() - view.float()).abs().max()) > 1e-6:
                raise AssertionError(region.index)
            working.append(value)
        working_hashes = [phase14.tensor_hash(value) for value in working]
        source = phase14.restrict_4x2(state.h)
        base = extra_args["model_options"]
        diffusion = model.inner_model.diffusion_model
        representative = self._representative_comparison(
            coordinator, model, source, working, sigma, base, regions, diffusion
        )
        gc.collect()
        phase2.comfy.model_management.soft_empty_cache()
        assembled, restricted, telemetry = self._all_joint(
            model, source, working, sigma, base, regions, diffusion
        )
        overlap = phase8d.overlap_metrics(
            [value.detach().float().cpu() for value in restricted], regions
        )
        phase15_report = json.loads(phase15.REPORT.read_text(encoding="utf-8"))
        if phase14.tensor_hash(state.h) != h_hash or phase14.tensor_hash(state.g) != g_hash:
            raise RuntimeError("Joint oracle mutated accepted H/G.")
        if [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Joint oracle mutated W.")
        self.outputs["B_JOINT_SOURCE_W_INTERACTION"] = assembled.detach().float().cpu()
        self.result = {
            "configuration": {
                **phase15_report["configuration"],
                "source": "Phase-14 4x2 area-mean fixed 4096",
                "consumer_interface_variable": "external frozen generated K/V versus bidirectional native joint interaction",
            },
            "sigmas": [float(value) for value in sigmas],
            "accepted_state": {"H_hash": h_hash, "G_hash": g_hash},
            "working_hashes": working_hashes,
            "source_hash": phase14.tensor_hash(source),
            "representative_one_region": representative,
            "joint_all_regions": {
                "assembled": phase14.summary(assembled),
                "overlap": overlap,
                "regions": 55,
                "source_tokens_per_region": SOURCE_TOKENS,
                "W_tokens_per_region": W_TOKENS,
                "text_tokens": TEXT_TOKENS,
                "joint_attention_dimensions": [8704, 8704],
                "joint_source_trajectories": 55,
                "context_blocks": 25,
                **telemetry,
            },
            "external_control": {
                "variant": phase15_report["variants"]["C0_FIXED_4096_MEAN"],
                "decoded": phase15_report["decoded"]["C0_FIXED_4096_MEAN"],
                "attention_dimensions": [4608, 8704],
                "source_attention_dimensions": [4608, 4608],
            },
            "provenance": phase14.provenance_summary(),
            "integrity": {
                "same_accepted_H_as_phase15": h_hash == phase15_report["accepted_state"]["H_hash"],
                "same_accepted_G_as_phase15": g_hash == phase15_report["accepted_state"]["G_hash"],
                "same_W_as_phase15": working_hashes == phase15_report["working_hashes"],
                "source_tokens": SOURCE_TOKENS,
                "joint_states": 55,
                "blocks_per_joint_state": 25,
                "finite": bool(torch.isfinite(assembled).all()),
                "complete_coverage": telemetry["coverage"][0] > 0,
                "terminal_state_updates": 0,
                "no_production_changes": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def make_sheet(control_path, joint_path, destination):
    images = [
        ("A_EXTERNAL_KV_CONTROL", Image.open(control_path).convert("RGB")),
        ("B_JOINT_SOURCE_W_INTERACTION", Image.open(joint_path).convert("RGB")),
    ]
    panels = []
    for name, image in images:
        image.thumbnail((4096, 640))
        panel = Image.new("RGB", (4096, image.height + 44), "white")
        panel.paste(image, ((4096 - image.width) // 2, 44))
        ImageDraw.Draw(panel).text((12, 12), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (4096, sum(item.height for item in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn(
        (1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED)
    )
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = Phase17Sampler()
    perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=phase14.SEED,
        )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    pixels = vae.decode(sampler.outputs["B_JOINT_SOURCE_W_INTERACTION"]).cpu()
    image_path = OUTPUT / "B_JOINT_SOURCE_W_INTERACTION.png"
    phase2.save_pixels(pixels, image_path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        decoded = {
            "path": str(image_path), "dimensions_wh": list(rgb.size),
            "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
        }
    control_path = Path(sampler.result["external_control"]["decoded"]["path"])
    sheet = OUTPUT / "EXTERNAL_VS_JOINT.png"
    make_sheet(control_path, image_path, sheet)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet)
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet), "decoded": decoded}, indent=2))


if __name__ == "__main__":
    main()
