"""Phase 18: semantic depth localization of the Phase-17 joint interface."""

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
import flux2_candidate3_fixed4k_consumer_interface as phase17
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_specialized_executor as phase8i
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_joint_depth_localization_results"
REPORT = OUTPUT / "report.json"
CHECKPOINTS = (
    ("D0", 0), ("D4", 4), ("S0", 5), ("S4", 9),
    ("S9", 14), ("S14", 19), ("S19", 24),
)


def clean_options(options):
    copied = options.copy()
    patches = {
        key: list(value) for key, value in options.get("patches", {}).items()
        if key not in ("attn1_patch", "attn1_output_patch")
    }
    copied["patches"] = patches
    return copied


def probe_options(options, probe, mode):
    copied = clean_options(options)
    patches = copied.setdefault("patches", {})
    if mode == "capture":
        patches.setdefault("attn1_patch", []).append(probe.capture_global)
    elif mode == "context":
        patches.setdefault("attn1_patch", []).append(probe.add_global_context)
        patches.setdefault("attn1_output_patch", []).append(probe.restore_text_attention)
    else:
        raise ValueError(mode)
    return copied


def block_name(ordinal):
    return ("double", ordinal) if ordinal < 5 else ("single", ordinal - 5)


def hidden_hash(value):
    return hashlib.sha256(
        value.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def enter_single_if_needed(executor, states, ordinal):
    if ordinal == 5:
        for state in states:
            executor.enter_single(state)


def advance_joint(executor, states, ordinal):
    enter_single_if_needed(executor, states, ordinal)
    kind, index = block_name(ordinal)
    for state in states:
        if kind == "double":
            executor.double(state, index)
        else:
            executor.single(state, index)


def advance_shared_external(executor, source, locals_, probe, ordinal):
    enter_single_if_needed(executor, [source, *locals_], ordinal)
    kind, index = block_name(ordinal)
    if kind == "double":
        executor.double(source, index)
        for state in locals_:
            executor.double(state, index)
    else:
        executor.single(source, index)
        for state in locals_:
            executor.single(state, index)
    probe.release()


def advance_independent_external(executor, pairs, ordinal):
    if ordinal == 5:
        for source, local, _probe in pairs:
            executor.enter_single(source)
            executor.enter_single(local)
    kind, index = block_name(ordinal)
    for source, local, probe in pairs:
        if kind == "double":
            executor.double(source, index)
            executor.double(local, index)
        else:
            executor.single(source, index)
            executor.single(local, index)
        probe.release()


def generated_parts(state, ordinal):
    if ordinal < 5:
        return (
            state.img[:, :phase17.SOURCE_TOKENS],
            state.img[:, phase17.SOURCE_TOKENS:],
        )
    return (
        state.img[:, phase17.TEXT_TOKENS:phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS],
        state.img[:, phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS:],
    )


def split_joint_state(state, ordinal):
    source_img, local_img = generated_parts(state, ordinal)
    text_pe = state.pe[:, :, :phase17.TEXT_TOKENS]
    source_pe = state.pe[:, :, phase17.TEXT_TOKENS:phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS]
    local_pe = state.pe[:, :, phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS:]
    current_text = state.txt if ordinal < 5 else state.img[:, :phase17.TEXT_TOKENS]
    if ordinal >= 5:
        source_img = torch.cat((current_text, source_img), dim=1)
        local_img = torch.cat((current_text, local_img), dim=1)
    probe = phase8i.OneBlockGPUContext()
    source = phase8i.ExplicitState(
        source_img.clone(), current_text.clone(), state.vec_orig, state.double_vec,
        state.single_vec, torch.cat((text_pe, source_pe), dim=2),
        probe_options(state.options, probe, "capture"), state.attn_mask,
    )
    local = phase8i.ExplicitState(
        local_img.clone(), current_text.clone(), state.vec_orig, state.double_vec,
        state.single_vec, torch.cat((text_pe, local_pe), dim=2),
        probe_options(state.options, probe, "context"), state.attn_mask,
    )
    return source, local, probe


def join_external_state(source, local, ordinal):
    if ordinal < 5:
        source_img, local_img = source.img, local.img
    else:
        source_img = source.img[:, phase17.TEXT_TOKENS:]
        local_img = local.img[:, phase17.TEXT_TOKENS:]
    text_pe = local.pe[:, :, :phase17.TEXT_TOKENS]
    source_pe = source.pe[:, :, phase17.TEXT_TOKENS:]
    local_pe = local.pe[:, :, phase17.TEXT_TOKENS:]
    current_text = local.txt if ordinal < 5 else local.img[:, :phase17.TEXT_TOKENS]
    image = torch.cat((source_img, local_img), dim=1)
    if ordinal >= 5:
        image = torch.cat((current_text, image), dim=1)
    return phase8i.ExplicitState(
        image, current_text, local.vec_orig, local.double_vec, local.single_vec,
        torch.cat((text_pe, source_pe, local_pe), dim=2), clean_options(local.options),
        local.attn_mask,
    )


def final_predictions(diffusion, states, working, mode):
    predictions = []
    for value, state in zip(working, states):
        if mode == "joint":
            hidden = state.img[:, phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS:]
        else:
            hidden = state.img[:, phase17.TEXT_TOKENS:]
        raw = diffusion.final_layer(hidden, state.vec_orig)
        predictions.append(phase8i.raw_to_x0(raw, value, torch.tensor([0.0], device=value.device), diffusion))
    return predictions


class Phase18Sampler(phase17.Phase17Sampler):
    def _prepare_plain_states(self, model, source, working, sigma, base, regions, diffusion):
        source_inputs = self._capture_plain(
            model, source, sigma, base, diffusion,
            {"scale_y": 4.0, "shift_y": 1.5, "scale_x": 2.0, "shift_x": 0.5},
            "phase18_source",
        )
        local_inputs = [
            self._capture_plain(
                model, value, sigma, base, diffusion, {}, f"phase18_W_{region.index}"
            )
            for value, region in zip(working, regions)
        ]
        executor = phase8i.ExplicitFluxExecutor(diffusion)
        return executor, executor.prepare(source_inputs), [executor.prepare(item) for item in local_inputs]

    def _assemble(self, diffusion, states, working, regions, mode, sigma):
        predictions = []
        for value, state in zip(working, states):
            hidden = (
                state.img[:, phase17.TEXT_TOKENS + phase17.SOURCE_TOKENS:]
                if mode == "joint" else state.img[:, phase17.TEXT_TOKENS:]
            )
            raw = diffusion.final_layer(hidden, state.vec_orig)
            x0 = phase8i.raw_to_x0(raw, value, sigma, diffusion)
            predictions.append(phase9b.restrict2(x0))
        assembled, coverage = BlueprintCoordinator().assembler.assemble(
            predictions, regions, phase14.H_HW
        )
        overlap = phase8d.overlap_metrics(
            [value.detach().float().cpu() for value in predictions], regions
        )
        return assembled, predictions, overlap, coverage

    def _full_joint_reference(self, model, source, working, sigma, base, regions, diffusion):
        executor, source_state, local_states = self._prepare_plain_states(
            model, source, working, sigma, base, regions, diffusion
        )
        states = [phase17.joint_state(source_state, local) for local in local_states]
        checkpoint_x0 = {}
        representative_boundaries = {}
        for ordinal in range(25):
            advance_joint(executor, states, ordinal)
            if ordinal in {value for _name, value in CHECKPOINTS}:
                name = next(name for name, value in CHECKPOINTS if value == ordinal)
                representative = states[phase17.REPRESENTATIVE_REGION]
                source_hidden, w_hidden = generated_parts(representative, ordinal)
                raw = diffusion.final_layer(w_hidden, representative.vec_orig)
                x0 = phase8i.raw_to_x0(
                    raw, working[phase17.REPRESENTATIVE_REGION], sigma, diffusion
                )
                checkpoint_x0[name] = x0.detach().float().cpu()
                representative_boundaries[name] = {
                    "source_hidden_rms": phase17.state_rms(source_hidden),
                    "W_hidden_rms": phase17.state_rms(w_hidden),
                    "source_hidden_hash": hidden_hash(source_hidden),
                    "W_hidden_hash": hidden_hash(w_hidden),
                }
        assembled, predictions, overlap, coverage = self._assemble(
            diffusion, states, working, regions, "joint", sigma
        )
        return {
            "assembled": assembled.detach().float().cpu(),
            "predictions": [value.detach().float().cpu() for value in predictions],
            "overlap": overlap, "coverage": [float(coverage.min()), float(coverage.max())],
            "checkpoints": checkpoint_x0,
            "representative_boundaries": representative_boundaries,
        }

    def _run_prefix_joint(self, model, source, working, sigma, base, regions,
                          diffusion, checkpoint, reference):
        executor, source_state, local_states = self._prepare_plain_states(
            model, source, working, sigma, base, regions, diffusion
        )
        states = [phase17.joint_state(source_state, local) for local in local_states]
        started = time.perf_counter()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for ordinal in range(checkpoint + 1):
            advance_joint(executor, states, ordinal)
        name = next(name for name, value in CHECKPOINTS if value == checkpoint)
        transition_source, transition_w = generated_parts(
            states[phase17.REPRESENTATIVE_REGION], checkpoint
        )
        transition = {
            "source_hidden_rms": phase17.state_rms(transition_source),
            "W_hidden_rms": phase17.state_rms(transition_w),
            "source_vs_full_joint": phase17.tensor_difference(
                transition_source.detach().cpu(),
                transition_source.detach().cpu(),
            ),
            "expected_full_joint_hashes": reference["representative_boundaries"][name],
        }
        pairs = [split_joint_state(state, checkpoint) for state in states]
        del states
        for ordinal in range(checkpoint + 1, 25):
            advance_independent_external(executor, pairs, ordinal)
        locals_final = [local for _source, local, _probe in pairs]
        assembled, predictions, overlap, coverage = self._assemble(
            diffusion, locals_final, working, regions, "external", sigma
        )
        end.record(); torch.cuda.synchronize()
        return assembled.detach().float().cpu(), [value.detach().float().cpu() for value in predictions], {
            "family": "prefix_joint_external_tail", "checkpoint": name,
            "transition_ordinal": checkpoint,
            "joint_blocks": checkpoint + 1, "external_tail_blocks": 24 - checkpoint,
            "transition": transition, "overlap": overlap,
            "coverage": [float(coverage.min()), float(coverage.max())],
            "cuda_ms": float(start.elapsed_time(end)),
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def _run_external_prefix(self, model, source, working, sigma, base, regions,
                             diffusion, checkpoint, reference):
        executor, source_state, local_states = self._prepare_plain_states(
            model, source, working, sigma, base, regions, diffusion
        )
        probe = phase8i.OneBlockGPUContext()
        source_state.options = probe_options(source_state.options, probe, "capture")
        for state in local_states:
            state.options = probe_options(state.options, probe, "context")
        started = time.perf_counter()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for ordinal in range(checkpoint + 1):
            advance_shared_external(executor, source_state, local_states, probe, ordinal)
        name = next(name for name, value in CHECKPOINTS if value == checkpoint)
        representative_local = local_states[phase17.REPRESENTATIVE_REGION]
        if checkpoint < 5:
            external_source_hidden = source_state.img
            external_w_hidden = representative_local.img
        else:
            external_source_hidden = source_state.img[:, phase17.TEXT_TOKENS:]
            external_w_hidden = representative_local.img[:, phase17.TEXT_TOKENS:]
        transition = {
            "source_hidden_rms": phase17.state_rms(external_source_hidden),
            "W_hidden_rms": phase17.state_rms(external_w_hidden),
            "source_hash": hidden_hash(external_source_hidden),
            "W_hash": hidden_hash(external_w_hidden),
            "full_joint_reference": reference["representative_boundaries"][name],
            "text_state_choice_at_join": "W/local external-prefix text state",
        }
        states = [join_external_state(source_state, local, checkpoint) for local in local_states]
        del source_state, local_states
        for ordinal in range(checkpoint + 1, 25):
            advance_joint(executor, states, ordinal)
        assembled, predictions, overlap, coverage = self._assemble(
            diffusion, states, working, regions, "joint", sigma
        )
        end.record(); torch.cuda.synchronize()
        return assembled.detach().float().cpu(), [value.detach().float().cpu() for value in predictions], {
            "family": "external_prefix_joint_tail", "checkpoint": name,
            "transition_ordinal": checkpoint,
            "external_prefix_blocks": checkpoint + 1, "joint_tail_blocks": 24 - checkpoint,
            "transition": transition, "overlap": overlap,
            "coverage": [float(coverage.min()), float(coverage.max())],
            "cuda_ms": float(start.elapsed_time(end)),
            "wall_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 18 requires empty-latent T2I without masks.")
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
            raise AssertionError("Phase 18 geometry/schedule mismatch.")
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
        base, diffusion = extra_args["model_options"], model.inner_model.diffusion_model

        reference = self._full_joint_reference(
            model, source, working, sigma, base, regions, diffusion
        )
        variants = {}
        outputs = {}
        for family in ("prefix", "tail"):
            for name, checkpoint in CHECKPOINTS:
                gc.collect(); phase2.comfy.model_management.soft_empty_cache()
                if family == "prefix":
                    assembled, predictions, record = self._run_prefix_joint(
                        model, source, working, sigma, base, regions, diffusion,
                        checkpoint, reference,
                    )
                    variant_name = f"A_PREFIX_JOINT_{name}_EXTERNAL_TAIL"
                else:
                    assembled, predictions, record = self._run_external_prefix(
                        model, source, working, sigma, base, regions, diffusion,
                        checkpoint, reference,
                    )
                    variant_name = f"B_EXTERNAL_PREFIX_{name}_JOINT_TAIL"
                print(
                    f"phase18 completed {record['family']} "
                    f"{record['checkpoint']} in {record['wall_seconds']:.2f}s",
                    flush=True,
                )
                per_region = [
                    phase17.tensor_difference(value, target)
                    for value, target in zip(predictions, reference["predictions"])
                ]
                record.update({
                    "assembled": phase14.summary(assembled),
                    "assembled_vs_full_joint": phase17.tensor_difference(
                        assembled, reference["assembled"]
                    ),
                    "per_region_vs_full_joint": per_region,
                    "per_region_vs_full_joint_rms_mean": sum(item["rms"] for item in per_region) / len(per_region),
                    "attention_dimensions": {
                        "joint": [8704, 8704],
                        "external_consumer": [4608, 8704],
                        "external_source": [4608, 4608],
                    },
                })
                variants[variant_name] = record
                outputs[variant_name] = assembled
        if phase14.tensor_hash(state.h) != h_hash or phase14.tensor_hash(state.g) != g_hash:
            raise RuntimeError("Phase 18 mutated accepted H/G.")
        if [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("Phase 18 mutated W.")
        self.outputs = outputs
        self.checkpoint_x0 = reference["checkpoints"]
        self.result = {
            "configuration": {
                "H": list(phase14.H_HW), "source": [32, 128], "source_tokens": 4096,
                "W": [64, 64], "regions": 55, "seed": phase14.SEED,
                "terminal_sigma": float(sigma), "checkpoints": list(CHECKPOINTS),
                "text_join_policy": "external-prefix joint tail retains W/local text state",
            },
            "sigmas": [float(value) for value in sigmas],
            "accepted_state": {"H_hash": h_hash, "G_hash": g_hash},
            "working_hashes": working_hashes,
            "source_hash": phase14.tensor_hash(source),
            "full_joint_reference": {
                "assembled": phase14.summary(reference["assembled"]),
                "overlap": reference["overlap"], "coverage": reference["coverage"],
                "representative_boundaries": reference["representative_boundaries"],
            },
            "variants": variants,
            "provenance": phase14.provenance_summary(),
            "integrity": {
                "source_tokens": 4096, "regions": 55, "variants": len(variants),
                "all_finite": all(item["assembled"]["finite"] for item in variants.values()),
                "complete_coverage": all(item["coverage"][0] > 0 for item in variants.values()),
                "accepted_state_immutable": True, "working_states_immutable": True,
                "terminal_state_updates": 0, "no_production_changes": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def save_sheet(items, destination, columns=2):
    opened = [(name, Image.open(path).convert("RGB")) for name, path in items]
    panel_w, image_h, title_h = 2048, 512, 38
    panels = []
    for name, image in opened:
        image.thumbnail((panel_w, image_h))
        panel = Image.new("RGB", (panel_w, image.height + title_h), "white")
        panel.paste(image, ((panel_w - image.width) // 2, title_h))
        ImageDraw.Draw(panel).text((10, 10), name, fill="black")
        panels.append(panel)
    rows = math.ceil(len(panels) / columns)
    row_h = max(panel.height for panel in panels)
    sheet = Image.new("RGB", (panel_w * columns, row_h * rows), "white")
    for index, panel in enumerate(panels):
        sheet.paste(panel, ((index % columns) * panel_w, (index // columns) * row_h))
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
    phase2.comfy.model_management.unload_all_models(); phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn(
        (1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED)
    )
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone(); sigmas[0] = 1.0
    sampler = Phase18Sampler(); perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=phase14.SEED,
        )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    decoded = {}
    family_paths = {"prefix": [], "tail": []}
    for name, latent in sampler.outputs.items():
        pixels = vae.decode(latent).cpu(); path = OUTPUT / f"{name}.png"; phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {"path": str(path), "dimensions_wh": list(rgb.size),
                             "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest()}
        family_paths["prefix" if name.startswith("A_") else "tail"].append((name, path))
    checkpoint_decoded = {}
    for name, latent in sampler.checkpoint_x0.items():
        pixels = vae.decode(latent).cpu(); path = OUTPUT / f"CHECKPOINT_FULL_JOINT_{name}_W.png"; phase2.save_pixels(pixels, path)
        checkpoint_decoded[name] = str(path)
    prefix_sheet = OUTPUT / "PREFIX_JOINT_EXTERNAL_TAIL.png"
    tail_sheet = OUTPUT / "EXTERNAL_PREFIX_JOINT_TAIL.png"
    checkpoint_sheet = OUTPUT / "FULL_JOINT_W_CHECKPOINTS.png"
    save_sheet(family_paths["prefix"], prefix_sheet)
    save_sheet(family_paths["tail"], tail_sheet)
    save_sheet([(name, Path(path)) for name, path in checkpoint_decoded.items()], checkpoint_sheet, columns=4)
    sampler.result["decoded"] = decoded
    sampler.result["checkpoint_decoded"] = checkpoint_decoded
    sampler.result["sheets"] = {"prefix": str(prefix_sheet), "tail": str(tail_sheet),
                                "checkpoints": str(checkpoint_sheet)}
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "sheets": sampler.result["sheets"]}, indent=2))


if __name__ == "__main__":
    main()
