"""Phase 8i: experiment-only explicit FLUX.2 block executor.

This file deliberately targets one qualified native FLUX.2 Klein terminal
state.  It does not patch ComfyUI on disk or change Blueprint production code.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import torch
from einops import rearrange


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
from comfy.ldm.flux.layers import timestep_embedding

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_specialized_executor_results"
REPORT = OUTPUT / "report.json"
HIGH_HW = (128, 256)
STEPS = 4
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT
EXPECTED_HASH = "53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce"
REFERENCE_ASSEMBLED_RMS = 0.709837
REFERENCE_OVERLAP_RMS = 0.281238
LOCAL_COUNT = 55


def tensor_bytes(value: torch.Tensor | None) -> int:
    return 0 if value is None else value.numel() * value.element_size()


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().float().cpu().numpy().tobytes()).hexdigest()


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    delta = value.detach().float().cpu() - reference.detach().float().cpu()
    return {
        "bit_exact": bool(torch.equal(value.detach().cpu(), reference.detach().cpu())),
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def state_difference(value, reference):
    if isinstance(value, tuple):
        return {
            "img": difference(value[0], reference[0]),
            "txt": difference(value[1], reference[1]),
        }
    return difference(value, reference)


def clone_cpu(value):
    if isinstance(value, tuple):
        return tuple(item.detach().cpu().clone() for item in value)
    return value.detach().cpu().clone()


class PreparedStop(BaseException):
    pass


@dataclass
class ForwardInputs:
    img: torch.Tensor
    img_ids: torch.Tensor
    txt: torch.Tensor
    txt_ids: torch.Tensor
    timesteps: torch.Tensor
    y: torch.Tensor | None
    guidance: torch.Tensor | None
    control: Any
    timestep_zero_index: Any
    transformer_options: dict[str, Any]
    attn_mask: torch.Tensor | None


class ForwardCapture:
    """Narrow runtime replacement around one Flux.forward_orig invocation."""

    def __init__(self, diffusion, *, run_native: bool):
        self.diffusion = diffusion
        self.run_native = run_native
        self.original = diffusion.forward_orig
        self.inputs: ForwardInputs | None = None

    def __enter__(self):
        owner = self

        def wrapped(this, img, img_ids, txt, txt_ids, timesteps, y,
                    guidance=None, control=None, timestep_zero_index=None,
                    transformer_options={}, attn_mask=None):
            owner.inputs = ForwardInputs(
                img=img.detach().clone(), img_ids=img_ids.detach().clone(),
                txt=txt.detach().clone(), txt_ids=txt_ids.detach().clone(),
                timesteps=timesteps.detach().clone(),
                y=None if y is None else y.detach().clone(),
                guidance=None if guidance is None else guidance.detach().clone(),
                control=control, timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options.copy(), attn_mask=attn_mask,
            )
            if not owner.run_native:
                raise PreparedStop()
            return owner.original(
                img, img_ids, txt, txt_ids, timesteps, y, guidance, control,
                timestep_zero_index=timestep_zero_index,
                transformer_options=transformer_options, attn_mask=attn_mask,
            )

        self.diffusion.forward_orig = MethodType(wrapped, self.diffusion)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.diffusion.forward_orig = self.original
        return exc_type is PreparedStop


class NativeBoundaryCapture:
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self.handles = []
        self.values: dict[str, Any] = {}

    def _hook(self, name):
        def hook(_module, _inputs, output):
            self.values[name] = clone_cpu(output)
        return hook

    def __enter__(self):
        self.handles.append(self.diffusion.img_in.register_forward_hook(self._hook("img_in")))
        self.handles.append(self.diffusion.txt_in.register_forward_hook(self._hook("txt_in")))
        for i, block in enumerate(self.diffusion.double_blocks):
            self.handles.append(block.register_forward_hook(self._hook(f"double_{i}")))
        for i, block in enumerate(self.diffusion.single_blocks):
            self.handles.append(block.register_forward_hook(self._hook(f"single_{i}")))
        self.handles.append(self.diffusion.final_layer.register_forward_hook(self._hook("final")))
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


@dataclass
class ExplicitState:
    img: torch.Tensor
    txt: torch.Tensor
    vec_orig: torch.Tensor
    double_vec: Any
    single_vec: Any
    pe: torch.Tensor | None
    options: dict[str, Any]
    attn_mask: torch.Tensor | None

    def persistent_bytes(self) -> dict[str, int]:
        return {
            "image_hidden": tensor_bytes(self.img),
            "text_hidden": tensor_bytes(self.txt),
            "positional_embedding": tensor_bytes(self.pe),
            "modulation_inputs": tensor_bytes(self.vec_orig),
        }


class ExplicitFluxExecutor:
    """Explicit orchestration around unmodified native FLUX layer modules."""

    def __init__(self, diffusion):
        self.model = diffusion

    def prepare(self, inputs: ForwardInputs) -> ExplicitState:
        if inputs.control is not None or inputs.timestep_zero_index is not None:
            raise ValueError("Phase 8i supports neither control nor timestep-zero references.")
        options = inputs.transformer_options.copy()
        patches = options.get("patches", {})
        img = self.model.img_in(inputs.img)
        vec = self.model.time_in(timestep_embedding(inputs.timesteps, 256).to(img.dtype))
        if self.model.params.guidance_embed and inputs.guidance is not None:
            vec = vec + self.model.guidance_in(
                timestep_embedding(inputs.guidance, 256).to(img.dtype)
            )
        if self.model.vector_in is not None:
            y = inputs.y
            if y is None:
                y = torch.zeros(
                    (img.shape[0], self.model.params.vec_in_dim),
                    device=img.device, dtype=img.dtype,
                )
            vec = vec + self.model.vector_in(y[:, :self.model.params.vec_in_dim])
        txt = inputs.txt
        if self.model.txt_norm is not None:
            txt = self.model.txt_norm(txt)
        txt = self.model.txt_in(txt)
        img_ids, txt_ids = inputs.img_ids, inputs.txt_ids
        for patch in patches.get("post_input", []):
            out = patch({"img": img, "txt": txt, "img_ids": img_ids,
                         "txt_ids": txt_ids, "transformer_options": options})
            img, txt = out["img"], out["txt"]
            img_ids, txt_ids = out["img_ids"], out["txt_ids"]
        pe = self.model.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        double_vec = vec
        single_vec = vec
        if self.model.params.global_modulation:
            double_vec = (
                self.model.double_stream_modulation_img(vec),
                self.model.double_stream_modulation_txt(vec),
            )
            single_vec, _ = self.model.single_stream_modulation(vec)
        return ExplicitState(img, txt, vec, double_vec, single_vec, pe, options, inputs.attn_mask)

    @staticmethod
    def _block_options(state: ExplicitState, kind: str, index: int, total: int) -> dict[str, Any]:
        options = state.options.copy()
        options["total_blocks"] = total
        options["block_type"] = kind
        options["block_index"] = index
        if kind == "single":
            options["img_slice"] = [state.txt.shape[1], state.img.shape[1]]
        return options

    def double(self, state: ExplicitState, index: int) -> None:
        options = self._block_options(state, "double", index, len(self.model.double_blocks))
        state.img, state.txt = self.model.double_blocks[index](
            img=state.img, txt=state.txt, vec=state.double_vec, pe=state.pe,
            attn_mask=state.attn_mask, transformer_options=options,
        )

    def enter_single(self, state: ExplicitState) -> None:
        if state.img.dtype == torch.float16:
            state.img = torch.nan_to_num(state.img, nan=0.0, posinf=65504, neginf=-65504)
        state.img = torch.cat((state.txt, state.img), dim=1)

    def single(self, state: ExplicitState, index: int) -> None:
        options = self._block_options(state, "single", index, len(self.model.single_blocks))
        state.img = self.model.single_blocks[index](
            state.img, vec=state.single_vec, pe=state.pe,
            attn_mask=state.attn_mask, transformer_options=options,
        )

    def final(self, state: ExplicitState) -> torch.Tensor:
        generated = state.img[:, state.txt.shape[1]:]
        return self.model.final_layer(generated, state.vec_orig)

    def run(self, inputs: ForwardInputs, capture=False):
        state = self.prepare(inputs)
        boundaries = {}
        if capture:
            boundaries["img_in"] = clone_cpu(state.img)
            boundaries["txt_in"] = clone_cpu(state.txt)
        for i in range(len(self.model.double_blocks)):
            self.double(state, i)
            if capture:
                boundaries[f"double_{i}"] = clone_cpu((state.img, state.txt))
        self.enter_single(state)
        for i in range(len(self.model.single_blocks)):
            self.single(state, i)
            if capture:
                boundaries[f"single_{i}"] = clone_cpu(state.img)
        raw = self.final(state)
        if capture:
            boundaries["final"] = clone_cpu(raw)
        return raw, boundaries


class OneBlockGPUContext(candidate2.OneEvaluationContextProbe):
    """Retain only the current source block K/V on GPU."""

    def __init__(self):
        super().__init__()
        self.current_key = None
        self.current_bytes = 0

    def capture_global(self, q, k, v, pe, attn_mask, extra_options):
        result = super().capture_global(q, k, v, pe, attn_mask, extra_options)
        self.current_key = self.block_key(extra_options)
        entry = self.global_kv[self.current_key]
        self.current_bytes = tensor_bytes(entry["k"]) + tensor_bytes(entry["v"])
        return result

    def release(self):
        self.global_kv.clear()
        self.current_key = None
        self.current_bytes = 0


def raw_to_x0(raw: torch.Tensor, latent: torch.Tensor, sigma: torch.Tensor, diffusion) -> torch.Tensor:
    h, w = latent.shape[-2:]
    model_output = rearrange(
        raw, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=h, w=w, ph=diffusion.patch_size, pw=diffusion.patch_size,
    )[..., :h, :w]
    return latent - model_output.float() * sigma


def merged_options(base, rope, probe=None, mode="ordinary", role="phase8i"):
    if probe is None:
        return candidate2_trajectory.merge_options(
            base, {"transformer_options": {"rope_options": dict(rope)}}
        )
    return candidate2_trajectory.merge_options(
        base, candidate2.model_options(role, rope, probe, mode)
    )


def cuda_snapshot() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated": int(torch.cuda.memory_allocated()),
        "reserved": int(torch.cuda.memory_reserved()),
        "peak_allocated": int(torch.cuda.max_memory_allocated()),
        "peak_reserved": int(torch.cuda.max_memory_reserved()),
    }


class SpecializedExecutorSampler(comfy.samplers.Sampler):
    def __init__(self):
        self.result = None
        self.output = None

    @staticmethod
    def _native_global(coordinator, model, g, sigma, base, probe, diffusion, capture_boundaries=False):
        options = merged_options(
            base, phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
            probe, "capture", "phase8i_global",
        )
        boundaries = NativeBoundaryCapture(diffusion) if capture_boundaries else None
        with ForwardCapture(diffusion, run_native=True) as inputs_capture:
            if boundaries:
                with boundaries:
                    x0 = coordinator.adapter.predict_global(
                        guider=model, g=g, sigma=sigma, canvas=HIGH_HW,
                        model_options=options, seed=SEED,
                    )
            else:
                x0 = coordinator.adapter.predict_global(
                    guider=model, g=g, sigma=sigma, canvas=HIGH_HW,
                    model_options=options, seed=SEED,
                )
        return x0, inputs_capture.inputs, {} if boundaries is None else boundaries.values

    @staticmethod
    def _native_region(coordinator, model, h, sigma, base, region, diffusion,
                       probe=None, mode="ordinary", capture_boundaries=False):
        options = merged_options(base, phase2.rope_for_crop(region), probe, mode,
                                 f"phase8i_crop_{region.index}")
        boundaries = NativeBoundaryCapture(diffusion) if capture_boundaries else None
        with ForwardCapture(diffusion, run_native=True) as inputs_capture:
            if boundaries:
                with boundaries:
                    x0 = coordinator.adapter.predict_region(
                        guider=model, h_view=h, sigma=sigma, canvas=HIGH_HW,
                        region=region, model_options=options, seed=SEED,
                    )
            else:
                x0 = coordinator.adapter.predict_region(
                    guider=model, h_view=h, sigma=sigma, canvas=HIGH_HW,
                    region=region, model_options=options, seed=SEED,
                )
        return x0, inputs_capture.inputs, {} if boundaries is None else boundaries.values

    @staticmethod
    def _prepare_region(coordinator, model, h, sigma, base, region, diffusion, probe, mode):
        options = merged_options(base, phase2.rope_for_crop(region), probe, mode,
                                 f"phase8i_prepare_crop_{region.index}")
        capture = ForwardCapture(diffusion, run_native=False)
        with capture:
            coordinator.adapter.predict_region(
                guider=model, h_view=h, sigma=sigma, canvas=HIGH_HW,
                region=region, model_options=options, seed=SEED,
            )
        if capture.inputs is None:
            raise RuntimeError("Local preparation did not reach Flux.forward_orig.")
        return capture.inputs

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 8i requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        model_sampling = model.inner_model.model_sampling
        h = model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h, sigmas[0])
        accepted = []
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1], model_options=extra_args["model_options"],
                seed=SEED,
            )
            accepted.append({"ordinal": ordinal, "H": tensor_hash(state.h), "G": tensor_hash(state.g)})

        terminal_h, terminal_g, sigma = state.h, state.g, sigmas[3]
        h_snapshot, g_snapshot = terminal_h.clone(), terminal_g.clone()
        regions = coordinator.planner.plan(HIGH_HW)
        if len(regions) != LOCAL_COUNT:
            raise AssertionError((len(regions), LOCAL_COUNT))
        base = extra_args["model_options"]
        diffusion = model.inner_model.diffusion_model
        executor = ExplicitFluxExecutor(diffusion)
        result = {
            "configuration": {
                "output_pixels": [2048, 4096], "H": list(HIGH_HW), "G": [96, 192],
                "seed": SEED, "prompt": PROMPT, "sigma": float(sigma),
                "crops": len(regions), "context_tokens": 18432,
                "production_or_core_changes": False,
            },
            "sigmas": [float(item) for item in sigmas],
            "accepted_nonterminal": accepted,
            "gates": {},
        }

        # Gate 1: exact ordinary crop, then explicit execution of the same modules.
        gate1_region = regions[0]
        gate1_latent = terminal_h[:, :, gate1_region.y:gate1_region.y2,
                                  gate1_region.x:gate1_region.x2]
        native_x0, gate1_inputs, native_boundaries = self._native_region(
            coordinator, model, gate1_latent, sigma, base, gate1_region,
            diffusion, capture_boundaries=True,
        )
        explicit_raw, explicit_boundaries = executor.run(gate1_inputs, capture=True)
        explicit_x0 = raw_to_x0(explicit_raw, gate1_latent, sigma, diffusion)
        gate1_diffs = {key: state_difference(explicit_boundaries[key], value)
                       for key, value in native_boundaries.items()}
        gate1_x0_diff = difference(explicit_x0, native_x0)
        gate1_first = next((key for key, value in gate1_diffs.items()
                            if (value.get("max_abs", 0.0) if isinstance(value, dict) and "max_abs" in value
                                else max(part["max_abs"] for part in value.values())) != 0.0), None)
        if gate1_first is None and gate1_x0_diff["max_abs"] != 0.0:
            gate1_first = "returned_x0"
        result["gates"]["gate1_native_local"] = {
            "passed": gate1_first is None,
            "first_divergence": gate1_first,
            "boundaries": gate1_diffs,
            "returned_x0": gate1_x0_diff,
            "input_shapes": {"img": list(gate1_inputs.img.shape), "txt": list(gate1_inputs.txt.shape)},
        }
        if gate1_first is not None:
            result["status"] = "GATE_1_FAILED"
            result["verdict"] = "SPECIALIZED EXECUTOR FAILS NATIVE EQUIVALENCE"
            self.result = result
            return model_sampling.inverse_noise_scaling(sigmas[-1], terminal_h)

        del native_boundaries, explicit_boundaries, explicit_raw, explicit_x0
        gc.collect()
        torch.cuda.empty_cache()

        # Gate 2 reference: ordinary full source capture plus one ordinary context crop.
        reference_probe = phase8d.CPUOffloadedContextProbe()
        reference_x0_g, global_inputs, _ = self._native_global(
            coordinator, model, terminal_g, sigma, base, reference_probe, diffusion
        )
        native_context_x0, local_inputs, native_context_boundaries = self._native_region(
            coordinator, model, gate1_latent, sigma, base, gate1_region, diffusion,
            probe=reference_probe, mode="context", capture_boundaries=True,
        )
        reference_crop_x0 = [native_context_x0.detach().float().cpu()]
        reference_local_start = time.perf_counter()
        for region in regions[1:]:
            latent = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            prediction, _, _ = self._native_region(
                coordinator, model, latent, sigma, base, region, diffusion,
                probe=reference_probe, mode="context", capture_boundaries=False,
            )
            reference_crop_x0.append(prediction.detach().float().cpu())
        torch.cuda.synchronize()
        reference_local_seconds = time.perf_counter() - reference_local_start
        reference_assembled, reference_coverage = coordinator.assembler.assemble(
            reference_crop_x0, regions, HIGH_HW
        )
        reference_kv = {
            key: {"k": value["k"].clone(), "v": value["v"].clone()}
            for key, value in reference_probe.global_kv.items()
        }
        reference_probe.global_kv.clear()
        del reference_probe
        gc.collect()
        torch.cuda.empty_cache()

        stream_probe = OneBlockGPUContext()
        global_inputs.transformer_options = merged_options(
            base, phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
            stream_probe, "capture", "phase8i_gate2_global",
        )["transformer_options"]
        local_inputs.transformer_options = merged_options(
            base, phase2.rope_for_crop(gate1_region), stream_probe, "context",
            "phase8i_gate2_local",
        )["transformer_options"]
        global_state = executor.prepare(global_inputs)
        local_state = executor.prepare(local_inputs)
        gate2_boundaries = {}
        source_kv_diffs = []
        for i in range(len(diffusion.double_blocks)):
            executor.double(global_state, i)
            key = ("double", i)
            source_kv_diffs.append({"block": list(key),
                "K": difference(stream_probe.global_kv[key]["k"], reference_kv[key]["k"]),
                "V": difference(stream_probe.global_kv[key]["v"], reference_kv[key]["v"])})
            executor.double(local_state, i)
            gate2_boundaries[f"double_{i}"] = clone_cpu((local_state.img, local_state.txt))
            stream_probe.release()
        executor.enter_single(global_state)
        executor.enter_single(local_state)
        for i in range(len(diffusion.single_blocks)):
            executor.single(global_state, i)
            key = ("single", i)
            source_kv_diffs.append({"block": list(key),
                "K": difference(stream_probe.global_kv[key]["k"], reference_kv[key]["k"]),
                "V": difference(stream_probe.global_kv[key]["v"], reference_kv[key]["v"])})
            executor.single(local_state, i)
            gate2_boundaries[f"single_{i}"] = clone_cpu(local_state.img)
            stream_probe.release()
        gate2_raw = executor.final(local_state)
        gate2_boundaries["final"] = clone_cpu(gate2_raw)
        gate2_x0 = raw_to_x0(gate2_raw, gate1_latent, sigma, diffusion)
        gate2_diffs = {key: state_difference(gate2_boundaries[key], native_context_boundaries[key])
                       for key in gate2_boundaries}
        gate2_x0_diff = difference(gate2_x0, native_context_x0)
        kv_max = max(max(item["K"]["max_abs"], item["V"]["max_abs"])
                     for item in source_kv_diffs)
        hidden_max = max(
            value["max_abs"] if "max_abs" in value else max(part["max_abs"] for part in value.values())
            for value in gate2_diffs.values()
        )
        gate2_passed = kv_max == 0.0 and hidden_max == 0.0 and gate2_x0_diff["max_abs"] == 0.0
        result["gates"]["gate2_full_context_one_crop"] = {
            "passed": gate2_passed, "source_kv_max_abs": kv_max,
            "local_hidden_max_abs": hidden_max, "returned_x0": gate2_x0_diff,
            "source_x0_G": phase8d.summary(reference_x0_g),
            "same_process_all_crop_reference": {
                "assembled_x0_H": phase8d.summary(reference_assembled),
                "overlap": phase8d.overlap_metrics(reference_crop_x0, regions),
                "coverage": [float(reference_coverage.min()), float(reference_coverage.max())],
                "terminal_local_wall_seconds": reference_local_seconds,
            },
            "source_kv": source_kv_diffs, "boundaries": gate2_diffs,
        }
        if not gate2_passed:
            result["status"] = "GATE_2_FAILED"
            result["verdict"] = "SPECIALIZED EXECUTOR FAILS NATIVE EQUIVALENCE"
            self.result = result
            return model_sampling.inverse_noise_scaling(sigmas[-1], terminal_h)

        del reference_kv, native_context_boundaries, gate2_boundaries
        del global_state, local_state, gate2_raw, gate2_x0, stream_probe
        gc.collect()
        torch.cuda.empty_cache()

        # Gate 3: explicit block-major source + all 55 local hidden trajectories.
        stream_probe = OneBlockGPUContext()
        global_inputs.transformer_options = merged_options(
            base, phase2.rope_for_global(*HIGH_HW, *coordinator.geometry.GLOBAL_HW),
            stream_probe, "capture", "phase8i_gate3_global",
        )["transformer_options"]
        local_inputs_all = []
        for region in regions:
            latent = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            local_inputs_all.append(self._prepare_region(
                coordinator, model, latent, sigma, base, region, diffusion,
                stream_probe, "context",
            ))
        global_state = executor.prepare(global_inputs)
        local_states = [executor.prepare(item) for item in local_inputs_all]
        del local_inputs_all
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        barrier_records = []
        temporary_records = []
        source_kv_peak = 0
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start_event.record()

        def barrier(kind, index):
            persistent = {
                "global_hidden_bytes": tensor_bytes(global_state.img) + tensor_bytes(global_state.txt),
                "all_crop_hidden_bytes": sum(tensor_bytes(s.img) + tensor_bytes(s.txt) for s in local_states),
                "text_hidden_bytes": sum(tensor_bytes(s.txt) for s in local_states) + tensor_bytes(global_state.txt),
                "positional_bytes": sum(tensor_bytes(s.pe) for s in local_states) + tensor_bytes(global_state.pe),
                "current_source_kv_bytes": stream_probe.current_bytes,
            }
            persistent.update(cuda_snapshot())
            persistent.update({"block_type": kind, "block_index": index})
            barrier_records.append(persistent)

        for kind, blocks in (("double", range(len(diffusion.double_blocks))),
                             ("single", range(len(diffusion.single_blocks)))):
            if kind == "single":
                executor.enter_single(global_state)
                for local_state in local_states:
                    executor.enter_single(local_state)
            for i in blocks:
                if kind == "double":
                    executor.double(global_state, i)
                else:
                    executor.single(global_state, i)
                source_kv_peak = max(source_kv_peak, stream_probe.current_bytes)
                before = int(torch.cuda.memory_allocated())
                for crop_index, local_state in enumerate(local_states):
                    torch.cuda.reset_peak_memory_stats()
                    if kind == "double":
                        executor.double(local_state, i)
                    else:
                        executor.single(local_state, i)
                    if crop_index == 0:
                        torch.cuda.synchronize()
                        temporary_records.append({
                            "block_type": kind, "block_index": i,
                            "before_crop_bytes": before,
                            "peak_during_crop_bytes": int(torch.cuda.max_memory_allocated()),
                            "after_crop_bytes": int(torch.cuda.memory_allocated()),
                        })
                barrier(kind, i)
                stream_probe.release()
                gc.collect()

        crop_raw = []
        crop_x0 = []
        for region, local_state in zip(regions, local_states):
            raw = executor.final(local_state)
            crop_raw.append(raw)
            latent = terminal_h[:, :, region.y:region.y2, region.x:region.x2]
            crop_x0.append(raw_to_x0(raw, latent, sigma, diffusion))
        end_event.record()
        torch.cuda.synchronize()
        cuda_ms = float(start_event.elapsed_time(end_event))
        wall_seconds = time.perf_counter() - wall_start
        assembled, coverage = coordinator.assembler.assemble(crop_x0, regions, HIGH_HW)
        final_h = assembled  # sigma_next=0 under CONST Euler terminal release.
        overlap = phase8d.overlap_metrics([item.detach().float().cpu() for item in crop_x0], regions)
        per_crop_vs_reference = [
            difference(value, reference)
            for value, reference in zip(crop_x0, reference_crop_x0)
        ]
        assembled_vs_reference = difference(assembled, reference_assembled)
        all_crop_exact = all(item["bit_exact"] for item in per_crop_vs_reference)
        assembly_numerically_exact = assembled_vs_reference["max_abs"] <= 5.0e-7
        qualified = all_crop_exact and assembly_numerically_exact
        result["gates"]["gate3_all_55"] = {
            "passed": qualified,
            "all_55_crop_predictions_bit_exact": all_crop_exact,
            "assembly_numerically_exact": assembly_numerically_exact,
            "first_non_bit_exact_boundary": (
                None if assembled_vs_reference["bit_exact"]
                else "overlap assembly: CPU-stored reference versus GPU-resident explicit crops"
            ),
            "assembly_tolerance_max_abs": 5.0e-7,
            "assembled_x0_H": phase8d.summary(assembled),
            "assembled_vs_same_process_reference": assembled_vs_reference,
            "per_crop_vs_same_process_reference": per_crop_vs_reference,
            "assembled_rms_vs_recorded_reference": abs(phase8d.summary(assembled)["rms"] - REFERENCE_ASSEMBLED_RMS),
            "overlap": overlap,
            "overlap_rms_vs_recorded_reference": abs(overlap["aggregate_rms"] - REFERENCE_OVERLAP_RMS),
            "coverage": [float(coverage.min()), float(coverage.max())],
            "source_kv_peak_bytes": source_kv_peak,
            "CPU_source_cache_bytes": 0,
            "CPU_to_GPU_KV_transfer_bytes": 0,
            "persistent_local_states": {
                "count": len(local_states),
                "requires_grad_any": any(s.img.requires_grad or s.txt.requires_grad for s in local_states),
                "inference_tensor_all": all(s.img.is_inference() and s.txt.is_inference() for s in local_states),
                "container": "one ExplicitState per crop; only block outputs replace img/txt",
            },
            "barriers": barrier_records,
            "temporary_peaks": temporary_records,
            "terminal_source_plus_local_cuda_ms": cuda_ms,
            "terminal_source_plus_local_wall_seconds": wall_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
        if not torch.equal(terminal_h, h_snapshot) or not torch.equal(terminal_g, g_snapshot):
            raise RuntimeError("Explicit execution mutated accepted H3/G3.")
        self.output = final_h.detach().float().cpu()
        self.result = result
        if callback is not None:
            callback(3, assembled, final_h, STEPS)
        return model_sampling.inverse_noise_scaling(sigmas[-1], final_h)


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
    sampler = SpecializedExecutorSampler()
    perf.prepare_model_state(model)
    try:
        with torch.inference_mode():
            phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
                torch.zeros_like(noise), callback=lambda *args: None,
                disable_pbar=True, seed=SEED,
            )
    except torch.OutOfMemoryError as error:
        if sampler.result is None:
            sampler.result = {"configuration": {"H": list(HIGH_HW), "G": [96, 192]}, "gates": {}}
        sampler.result["status"] = "GATE_3_OOM"
        sampler.result["error"] = str(error)
        sampler.result["verdict"] = "EXACT EXECUTOR IS PERSISTENT-STATE VRAM LIMITED"
        REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(REPORT), "verdict": sampler.result["verdict"],
                          "error": str(error)}, indent=2))
        return

    result = sampler.result
    if result is None:
        raise RuntimeError("Phase 8i sampler produced no result.")
    if result.get("status", "").startswith("GATE_"):
        REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"report": str(REPORT), "status": result["status"],
                          "verdict": result["verdict"]}, indent=2))
        return

    pixels = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    ).decode(sampler.output).cpu()
    image_path = OUTPUT / "SPECIALIZED_EXECUTOR.png"
    phase2.save_pixels(pixels, image_path)
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        decoded = {"path": str(image_path), "dimensions_wh": list(rgb.size),
                   "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}
    result["gates"]["gate3_all_55"]["decoded"] = decoded
    result["gates"]["gate3_all_55"]["decoded_matches_reference"] = decoded["sha256_rgb"] == EXPECTED_HASH
    if result["gates"]["gate3_all_55"]["passed"]:
        result["status"] = "SUCCESS"
        result["verdict"] = "SPECIALIZED FLUX EXECUTOR QUALIFIED"
    else:
        result["status"] = "GATE_3_SEMANTIC_MISMATCH"
        result["verdict"] = "SPECIALIZED EXECUTOR FAILS NATIVE EQUIVALENCE"
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "decoded": decoded,
                      "gate1": result["gates"]["gate1_native_local"]["passed"],
                      "gate2": result["gates"]["gate2_full_context_one_crop"]["passed"],
                      "gate3": result["gates"]["gate3_all_55"]["passed"],
                      "verdict": result["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
