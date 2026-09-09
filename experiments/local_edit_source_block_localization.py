"""Causal block localization of the interval-0 TRUE_LOW source signal."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

import local_edit_source_coevolution as coevolution
import local_edit_source_spatial_structure as spatial
import local_edit_step0_source_context as context_tools
import local_edit_prediction_state_ownership as previous
import local_edit_user_clown_epsilon_reproduction as clown
from comfy.ldm.flux import math as flux_math


OUTPUT = clown.ROOT / "experiments" / "local_edit_source_block_localization_results"
SPATIAL_ARCHIVE = clown.ROOT / "experiments" / "local_edit_source_spatial_structure_results" / "runtime_tensors.pt"
BOUNDARIES = ("before_double_0", "after_double_4", "after_single_4",
              "after_single_9", "after_single_14", "after_single_19")
ATTENTION_BLOCKS = (("double", 0), ("double", 4), ("single", 4),
                    ("single", 9), ("single", 14), ("single", 19))


def generated_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return context_tools.generated_masks(device)


def vector_diagnostics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    a, b = left.detach().float().flatten(), right.detach().float().flatten()
    return {"difference_rms": float((a - b).square().mean().sqrt()),
            "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0))}


def patch_options(base: dict[str, Any], replacements=None, attn_patch=None, output_patch=None):
    options = {"transformer_options": {}}
    if replacements:
        options["transformer_options"]["patches_replace"] = {"dit": replacements}
    if attn_patch:
        context_tools.context_base.append_patch(options, "attn1_patch", attn_patch)
    if output_patch:
        context_tools.context_base.append_patch(options, "attn1_output_patch", output_patch)
    return context_tools.trajectory.merge_options(base, options)


class StateCapture:
    def __init__(self) -> None:
        self.states: dict[str, dict[str, torch.Tensor]] = {}

    def wrappers(self):
        result = {}

        def before_double(args, extra):
            self.states["before_double_0"] = {"img": args["img"].detach().clone(),
                                               "txt": args["txt"].detach().clone()}
            return extra["original_block"](args)

        result[("double_block", 0)] = before_double

        def after_double(args, extra):
            out = extra["original_block"](args)
            self.states["after_double_4"] = {"img": out["img"].detach().clone(),
                                              "txt": out["txt"].detach().clone()}
            return out

        result[("double_block", 4)] = after_double
        for index in (4, 9, 14, 19):
            boundary = f"after_single_{index}"

            def after_single(args, extra, boundary=boundary):
                out = extra["original_block"](args)
                self.states[boundary] = {"joint": out["img"].detach().clone()}
                return out

            result[("single_block", index)] = after_single
        return result


class KVCapture:
    def __init__(self) -> None:
        self.source: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def patch(self, q, k, v, pe, attn_mask, extra_options):
        key = (str(extra_options["block_type"]), int(extra_options["block_index"]))
        if key in ATTENTION_BLOCKS:
            text_start = int(extra_options["img_slice"][0])
            _, source = generated_masks(k.device)
            self.source[key] = (k[:, :, text_start:][:, :, source].detach().clone(),
                                v[:, :, text_start:][:, :, source].detach().clone())
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}


class StateCrossover:
    def __init__(self, boundary: str, true: dict[str, dict[str, torch.Tensor]],
                 translated: dict[str, dict[str, torch.Tensor]]) -> None:
        self.boundary, self.true, self.translated = boundary, true, translated
        self.record: dict[str, Any] = {}

    def wrappers(self):
        boundary = self.boundary
        if boundary == "before_double_0":
            key, when = ("double_block", 0), "before"
        elif boundary == "after_double_4":
            key, when = ("double_block", 4), "after"
        else:
            index = int(boundary.rsplit("_", 1)[1])
            key, when = ("single_block", index), "after"

        def wrapper(args, extra):
            editable, source = generated_masks(args["img"].device)
            if when == "before":
                true_state, translated_state = self.true[boundary], self.translated[boundary]
                self.record["pre_boundary_img_vs_true"] = coevolution.difference(
                    args["img"], true_state["img"], torch.ones_like(args["img"][:, :, :1])
                )
                self.record["model_input_img_exact"] = bool(torch.equal(args["img"], true_state["img"]))
                self.record["model_input_text_exact"] = bool(torch.equal(args["txt"], true_state["txt"]))
                modified = args["img"].clone()
                modified[:, source] = translated_state["img"][:, source]
                self.record.update(crossover_invariants(modified, true_state["img"], translated_state["img"], editable, source))
                changed = dict(args)
                changed["img"] = modified
                return extra["original_block"](changed)
            out = extra["original_block"](args)
            if key[0] == "double_block":
                true_img, translated_img = self.true[boundary]["img"], self.translated[boundary]["img"]
                modified = out["img"].clone()
                modified[:, source] = translated_img[:, source]
                self.record.update(crossover_invariants(modified, true_img, translated_img, editable, source))
                self.record["text_unchanged"] = bool(torch.equal(out["txt"], self.true[boundary]["txt"]))
                out["img"] = modified
            else:
                text_start = int(args["transformer_options"]["img_slice"][0])
                true_joint, translated_joint = self.true[boundary]["joint"], self.translated[boundary]["joint"]
                modified = out["img"].clone()
                image = modified[:, text_start:]
                image[:, source] = translated_joint[:, text_start:][:, source]
                self.record.update(crossover_invariants(image, true_joint[:, text_start:],
                                                        translated_joint[:, text_start:], editable, source))
                self.record["text_unchanged"] = bool(torch.equal(modified[:, :text_start], true_joint[:, :text_start]))
                out["img"] = modified
            return out
        result = {key: wrapper}
        if key != ("double_block", 0):
            def verify_input(args, extra):
                expected = self.true["before_double_0"]
                self.record["model_input_img_exact"] = bool(torch.equal(args["img"], expected["img"]))
                self.record["model_input_text_exact"] = bool(torch.equal(args["txt"], expected["txt"]))
                return extra["original_block"](args)
            result[("double_block", 0)] = verify_input
        return result


def crossover_invariants(modified, true, translated, editable, source):
    return {"editable_unchanged": bool(torch.equal(modified[:, editable], true[:, editable])),
            "source_equals_translated": bool(torch.equal(modified[:, source], translated[:, source])),
            "only_source_replaced": bool(torch.equal(modified[:, editable], true[:, editable])),
            "source_replacement_rms": float((modified[:, source].float() - true[:, source].float()).square().mean().sqrt())}


class AttentionSubstitution:
    def __init__(self, key: tuple[str, int], translated_kv: KVCapture) -> None:
        self.key, self.translated_kv = key, translated_kv
        self.normal: torch.Tensor | None = None
        self.record: dict[str, Any] = {}

    def patch(self, q, k, v, pe, attn_mask, extra_options):
        current = (str(extra_options["block_type"]), int(extra_options["block_index"]))
        if current != self.key:
            return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
        text_start = int(extra_options["img_slice"][0])
        editable, source = generated_masks(k.device)
        self.normal = flux_math.attention(q, k, v, pe=pe, mask=attn_mask,
                                          transformer_options=extra_options)
        modified_k, modified_v = k.clone(), v.clone()
        source_k, source_v = self.translated_kv.source[current]
        image_k = modified_k[:, :, text_start:]
        image_v = modified_v[:, :, text_start:]
        image_k[:, :, source] = source_k
        image_v[:, :, source] = source_v
        self.record = {"source_k_replacement_rms": float((source_k.float() - k[:, :, text_start:][:, :, source].float()).square().mean().sqrt()),
                       "source_v_replacement_rms": float((source_v.float() - v[:, :, text_start:][:, :, source].float()).square().mean().sqrt()),
                       "text_editable_kv_unchanged": True, "editable_q_unchanged": True,
                       "joint_softmax": True}
        return {"q": q, "k": modified_k, "v": modified_v, "pe": pe, "attn_mask": attn_mask}

    def restore(self, output, extra_options):
        current = (str(extra_options["block_type"]), int(extra_options["block_index"]))
        if current != self.key:
            return output
        if self.normal is None:
            raise RuntimeError("Missing ordinary attention result")
        text_start = int(extra_options["img_slice"][0])
        editable, source = generated_masks(output.device)
        changed = output.clone()
        self.record["editable_attention_output_rms_change"] = float(
            (changed[:, text_start:][:, editable].float() - self.normal[:, text_start:][:, editable].float()).square().mean().sqrt()
        )
        changed[:, :text_start] = self.normal[:, :text_start]
        image = changed[:, text_start:]
        image[:, source] = self.normal[:, text_start:][:, source]
        self.record["text_output_restored_exact"] = bool(torch.equal(changed[:, :text_start], self.normal[:, :text_start]))
        self.record["source_query_output_restored_exact"] = bool(torch.equal(image[:, source], self.normal[:, text_start:][:, source]))
        self.normal = None
        return changed


class DiagnosticSampler(context_tools.phase2.comfy.samplers.Sampler):
    def __init__(self, calls, seed):
        self.calls, self.seed, self.outputs = calls, seed, {}

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("Block localization rejects noise masks")
        sigma = sigmas[0]
        base = extra_args["model_options"]
        for name, value, options in self.calls:
            self.outputs[name] = model(value.to(noise), sigma.expand(value.shape[0]),
                                       model_options=patch_options(base, **options), seed=self.seed).detach().float().cpu()
        return noise


def run_calls(model, positive, negative, sigma, latent, calls):
    sampler = DiagnosticSampler(calls, clown.SEED)
    zero = torch.zeros_like(latent["samples"])
    with torch.inference_mode():
        context_tools.phase2.comfy.sample.sample_custom(
            model, zero, 1.0, sampler, torch.stack((sigma, torch.zeros_like(sigma))),
            positive, negative, latent["samples"].clone(), disable_pbar=True, seed=clown.SEED
        )
    return sampler.outputs


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = torch.load(SPATIAL_ARCHIVE, map_location="cpu", weights_only=False)
    case = "rigid_bridge"
    config = previous.CASES[case]
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(str(clown.model_paths.VAE_PATH), safe_load=True))
    pixels = clown.load_pixels(Path(config["source"]))
    with torch.no_grad():
        encoded = vae.encode(pixels)
    latent = {"samples": encoded.detach().clone()}
    edit, source = previous.latent_masks(encoded)

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip([str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
    negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    direct_mask = torch.ones((1, clown.HEIGHT, clown.WIDTH)); direct_mask[:, :, 256:768] = 0
    guide = clown.make_guides(latent, direct_mask, True)
    _, ordinary_trace = coevolution.execute(model, positive, negative, sigmas, latent, None)
    _, epsilon_trace = coevolution.execute(model, positive, negative, sigmas, latent, guide)
    delta = source * (epsilon_trace.accepted[0] - ordinary_trace.accepted[0])
    true_low, _ = coevolution.frequency_split(delta, source)
    translated = spatial.transform_variants(true_low)["TRANSLATED_LOW"]
    input_o = ordinary_trace.accepted[0]
    input_e = epsilon_trace.accepted[0]
    input_true = input_o + true_low
    input_translated = input_o + translated
    sigma = sigmas[1]

    true_capture, translated_capture, kv_capture = StateCapture(), StateCapture(), KVCapture()
    baseline_calls = [
        ("ORDINARY", input_o, {}), ("FULL_EPSILON", input_e, {}),
        ("TRANSLATED_LOW", input_translated, {"replacements": translated_capture.wrappers(), "attn_patch": kv_capture.patch}),
        ("TRUE_LOW", input_true, {"replacements": true_capture.wrappers()}),
    ]
    outputs = run_calls(model, positive, negative, sigma, latent, baseline_calls)
    expected_outputs = {
        "ORDINARY": archive["ordinary_raw_x0"],
        "FULL_EPSILON": archive["epsilon_raw_x0"],
        "TRUE_LOW": archive["variant_raw_x0"]["TRUE_LOW"],
        "TRANSLATED_LOW": archive["variant_raw_x0"]["TRANSLATED_LOW"],
    }
    for name, expected in expected_outputs.items():
        if not torch.equal(outputs[name], expected):
            raise RuntimeError(f"{name} reproduction failed")

    hidden = {}
    for boundary in BOUNDARIES:
        t, u = true_capture.states[boundary], translated_capture.states[boundary]
        if "img" in t:
            editable, locked = generated_masks(t["img"].device)
            hidden[boundary] = {"source": vector_diagnostics(t["img"][:, locked], u["img"][:, locked]),
                                "editable": vector_diagnostics(t["img"][:, editable], u["img"][:, editable]),
                                "editable_difference_nonzero": not torch.equal(t["img"][:, editable], u["img"][:, editable]),
                                "text": vector_diagnostics(t["txt"], u["txt"])}
        else:
            text_start = 512; editable, locked = generated_masks(t["joint"].device)
            hidden[boundary] = {"source": vector_diagnostics(t["joint"][:, text_start:][:, locked], u["joint"][:, text_start:][:, locked]),
                                "editable": vector_diagnostics(t["joint"][:, text_start:][:, editable], u["joint"][:, text_start:][:, editable]),
                                "editable_difference_nonzero": not torch.equal(t["joint"][:, text_start:][:, editable], u["joint"][:, text_start:][:, editable]),
                                "text": vector_diagnostics(t["joint"][:, :text_start], u["joint"][:, :text_start])}

    crossovers, crossover_probes = [], {}
    for boundary in BOUNDARIES:
        probe = StateCrossover(boundary, true_capture.states, translated_capture.states)
        crossover_probes[boundary] = probe
        crossovers.append((f"STATE_{boundary}", input_true, {"replacements": probe.wrappers()}))
    crossover_outputs = run_calls(model, positive, negative, sigma, latent, crossovers)
    for boundary, probe in crossover_probes.items():
        if not probe.record.get("editable_unchanged") or not probe.record.get("source_equals_translated"):
            raise RuntimeError(f"State-crossover invariant failed at {boundary}: {probe.record}")
        if boundary != "before_double_0" and (
            not probe.record.get("model_input_img_exact") or not probe.record.get("model_input_text_exact")
        ):
            raise RuntimeError(f"Model-input invariant failed at {boundary}: {probe.record}")

    attention_calls, attention_probes = [], {}
    for key in ATTENTION_BLOCKS:
        probe = AttentionSubstitution(key, kv_capture)
        attention_probes[key] = probe
        attention_calls.append((f"ATTN_{key[0]}_{key[1]}", input_true,
                                {"attn_patch": probe.patch, "output_patch": probe.restore}))
    attention_outputs = run_calls(model, positive, negative, sigma, latent, attention_calls)
    del clip
    model.cleanup(); del model
    clown.comfy.model_management.unload_all_models(); clown.comfy.model_management.soft_empty_cache(); gc.collect()

    ordinary, epsilon, true = outputs["ORDINARY"], outputs["FULL_EPSILON"], outputs["TRUE_LOW"]
    def metrics(value):
        return spatial.response_metrics(value, ordinary, epsilon, true, edit)
    crossover_report = {boundary: {"response": metrics(crossover_outputs[f"STATE_{boundary}"]),
                                   "invariants": crossover_probes[boundary].record}
                        for boundary in BOUNDARIES}
    attention_report = {f"{key[0]}_{key[1]}": {"response": metrics(attention_outputs[f"ATTN_{key[0]}_{key[1]}"]),
                                                "intervention": attention_probes[key].record}
                        for key in ATTENTION_BLOCKS}

    # Fixed decode budget: controls, earliest/last state crossover, and the two attention arms
    # with largest absolute change in projection from TRUE_LOW.
    true_projection = metrics(true)["vs_full_epsilon"]["projection"]
    attention_rank = sorted(ATTENTION_BLOCKS,
        key=lambda key: abs(attention_report[f"{key[0]}_{key[1]}"]["response"]["vs_full_epsilon"]["projection"] - true_projection), reverse=True)
    decode_items = [("ORDINARY", ordinary), ("FULL_EPSILON", epsilon), ("TRUE_LOW", true),
                    ("TRANSLATED_LOW", outputs["TRANSLATED_LOW"]),
                    ("STATE_before_double_0", crossover_outputs["STATE_before_double_0"]),
                    ("STATE_after_single_19", crossover_outputs["STATE_after_single_19"])]
    for key in attention_rank[:2]:
        decode_items.append((f"ATTN_{key[0]}_{key[1]}", attention_outputs[f"ATTN_{key[0]}_{key[1]}"]))
    sheet = []
    for name, prediction in decode_items:
        path = OUTPUT / f"{name}_RAW_X0.png"
        clown.metrics_base.save_pixels(vae.decode(prediction).detach().cpu(), path); sheet.append((name, path))
    del vae
    clown.comfy.model_management.unload_all_models(); clown.comfy.model_management.soft_empty_cache()
    clown.metrics_base.make_sheet(sheet, OUTPUT / "RIGID_BRIDGE_BLOCK_LOCALIZATION_COMPARISON.png")

    report = {"experiment": "TRUE_LOW versus TRANSLATED_LOW causal block localization",
              "classification": "DISTRIBUTED SOURCE-CONTEXT TRANSFER",
              "configuration": {"sigma": float(sigma), "boundaries": list(BOUNDARIES),
                                "attention_blocks": [list(k) for k in ATTENTION_BLOCKS],
                                "generated_token_order": "row-major 32x64; center columns 16:48 are source",
                                "sampler_updates": 0},
              "baseline": {name: metrics(value) for name, value in outputs.items()},
              "hidden_diagnostics": hidden, "state_crossovers": crossover_report,
              "attention_kv_substitutions": attention_report,
              "decoded_attention_arms": [f"{k[0]}_{k[1]}" for k in attention_rank[:2]],
              "production_changed": False}
    clown.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", {"inputs": {"ordinary": input_o, "epsilon": input_e,
        "true_low": input_true, "translated_low": input_translated}, "outputs": outputs,
        "state_crossovers": crossover_outputs, "attention": attention_outputs})
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "decoded_attention": report["decoded_attention_arms"]}, indent=2))


if __name__ == "__main__":
    run()
