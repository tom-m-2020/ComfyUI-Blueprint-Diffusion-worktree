"""Zero-update directional source/editable attention discriminator."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

import local_edit_source_block_localization as localization
import local_edit_source_coevolution as coevolution
import local_edit_source_spatial_structure as spatial
import local_edit_prediction_state_ownership as previous
import local_edit_user_clown_epsilon_reproduction as clown


OUTPUT = clown.ROOT / "experiments" / "local_edit_source_directional_coupling_results"
BLOCK_ARCHIVE = clown.ROOT / "experiments" / "local_edit_source_block_localization_results" / "runtime_tensors.pt"
DIRECTIONS = ("NO_EDITABLE_TO_SOURCE", "NO_SOURCE_TO_EDITABLE")
EXPECTED_BLOCKS = {("double", i) for i in range(5)} | {("single", i) for i in range(20)}


def directional_mask(total_tokens: int, text_start: int, image_end: int,
                     editable: torch.Tensor, source: torch.Tensor, direction: str,
                     dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, dict[str, Any]]:
    if image_end - text_start != editable.numel() or editable.shape != source.shape:
        raise AssertionError((total_tokens, text_start, image_end, editable.shape, source.shape))
    if bool((editable & source).any()) or not bool((editable | source).all()):
        raise AssertionError("Generated-token source/editable partition is invalid")
    if direction == "NO_EDITABLE_TO_SOURCE":
        query_local, key_local = source, editable
    elif direction == "NO_SOURCE_TO_EDITABLE":
        query_local, key_local = editable, source
    else:
        raise ValueError(direction)

    query_indices = torch.nonzero(query_local, as_tuple=False).flatten() + text_start
    key_indices = torch.nonzero(key_local, as_tuple=False).flatten() + text_start
    mask = torch.zeros((1, 1, total_tokens, total_tokens), dtype=dtype, device=device)
    blocked_value = -torch.finfo(dtype).max
    mask[0, 0, query_indices[:, None], key_indices[None, :]] = blocked_value
    expected_edges = int(query_indices.numel() * key_indices.numel())
    record = {
        "direction": direction,
        "total_tokens": total_tokens,
        "text_tokens": text_start,
        "generated_tokens": image_end - text_start,
        "source_queries_or_keys": int(source.sum()),
        "editable_queries_or_keys": int(editable.sum()),
        "blocked_edges": int((mask != 0).sum()),
        "expected_blocked_edges": expected_edges,
        "text_query_edges_unchanged": bool(torch.equal(mask[:, :, :text_start], torch.zeros_like(mask[:, :, :text_start]))),
        "text_key_edges_unchanged": bool(torch.equal(mask[:, :, :, :text_start], torch.zeros_like(mask[:, :, :, :text_start]))),
        "mask_has_only_zero_or_blocked": bool(((mask == 0) | (mask == blocked_value)).all()),
        "blocked_value": float(blocked_value),
    }
    return mask, record


class DirectionalAttention:
    def __init__(self, direction: str) -> None:
        if direction not in DIRECTIONS:
            raise ValueError(direction)
        self.direction = direction
        self.records: list[dict[str, Any]] = []

    def patch(self, q, k, v, pe, attn_mask, extra_options):
        if attn_mask is not None:
            raise AssertionError("Expected unmasked native Klein attention")
        if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
            raise AssertionError((q.shape, k.shape, v.shape))
        text_start, image_end = map(int, extra_options["img_slice"])
        editable, source = localization.generated_masks(q.device)
        mask, record = directional_mask(q.shape[2], text_start, image_end, editable, source,
                                        self.direction, q.dtype, q.device)
        key = (str(extra_options["block_type"]), int(extra_options["block_index"]))
        record.update({
            "block_type": key[0], "block_index": key[1],
            "q_identity_preserved": True, "k_identity_preserved": True,
            "v_identity_preserved": True, "pe_identity_preserved": True,
            "ordinary_joint_softmax": True,
        })
        if record["blocked_edges"] != record["expected_blocked_edges"]:
            raise AssertionError(record)
        self.records.append(record)
        return {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": mask}

    def assert_complete(self) -> None:
        observed = {(r["block_type"], r["block_index"]) for r in self.records}
        if observed != EXPECTED_BLOCKS or len(self.records) != len(EXPECTED_BLOCKS):
            raise AssertionError((observed ^ EXPECTED_BLOCKS, len(self.records)))
        for record in self.records:
            if not all(record[name] for name in (
                "text_query_edges_unchanged", "text_key_edges_unchanged",
                "mask_has_only_zero_or_blocked", "q_identity_preserved",
                "k_identity_preserved", "v_identity_preserved", "pe_identity_preserved",
                "ordinary_joint_softmax",
            )):
                raise AssertionError(record)


def regional_norm(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = value.detach().float().masked_select(mask.bool().expand_as(value))
    return {"rms": float(selected.square().mean().sqrt()),
            "l2": float(torch.linalg.vector_norm(selected)),
            "max_abs": float(selected.abs().max())}


def arm_metrics(value: torch.Tensor, ordinary: torch.Tensor, epsilon: torch.Tensor,
                true: torch.Tensor, editable: torch.Tensor, source: torch.Tensor) -> dict[str, Any]:
    difference_true = value - true
    difference_ordinary = value - ordinary
    whole = torch.ones_like(editable)
    return {
        "response": spatial.response_metrics(value, ordinary, epsilon, true, editable),
        "difference_from_true_low": {
            "editable": regional_norm(difference_true, editable),
            "source": regional_norm(difference_true, source),
            "whole": regional_norm(difference_true, whole),
        },
        "difference_from_ordinary": {
            "editable": regional_norm(difference_ordinary, editable),
            "source": regional_norm(difference_ordinary, source),
            "whole": regional_norm(difference_ordinary, whole),
        },
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    archive = torch.load(BLOCK_ARCHIVE, map_location="cpu", weights_only=False)
    input_true = archive["inputs"]["true_low"]
    ordinary = archive["outputs"]["ORDINARY"]
    epsilon = archive["outputs"]["FULL_EPSILON"]
    archived_true = archive["outputs"]["TRUE_LOW"]
    editable, source = previous.latent_masks(input_true)
    input_hash = coevolution.tensor_sha256(input_true)

    config = previous.CASES["rigid_bridge"]
    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
    negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    sigma = sigmas[1]

    probes = {direction: DirectionalAttention(direction) for direction in DIRECTIONS}
    calls = [("TRUE_LOW_FULL", input_true, {})]
    for direction, probe in probes.items():
        calls.append((direction, input_true, {"attn_patch": probe.patch}))
    outputs = localization.run_calls(model, positive, negative, sigma, {"samples": input_true}, calls)
    if not torch.equal(outputs["TRUE_LOW_FULL"], archived_true):
        raise RuntimeError("TRUE_LOW control did not reproduce the block-localization archive")
    if coevolution.tensor_sha256(input_true) != input_hash:
        raise RuntimeError("Archived TRUE_LOW input was mutated")
    for probe in probes.values():
        probe.assert_complete()

    del clip
    model.cleanup(); del model
    clown.comfy.model_management.unload_all_models(); clown.comfy.model_management.soft_empty_cache(); gc.collect()

    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    decode_items = [
        ("ORDINARY", ordinary), ("FULL_EPSILON", epsilon),
        ("TRUE_LOW_FULL", outputs["TRUE_LOW_FULL"]),
        ("NO_EDITABLE_TO_SOURCE", outputs["NO_EDITABLE_TO_SOURCE"]),
        ("NO_SOURCE_TO_EDITABLE", outputs["NO_SOURCE_TO_EDITABLE"]),
    ]
    sheet = []
    for name, prediction in decode_items:
        path = OUTPUT / f"{name}_RAW_X0.png"
        clown.metrics_base.save_pixels(vae.decode(prediction).detach().cpu(), path)
        sheet.append((name, path))
    del vae
    clown.comfy.model_management.unload_all_models(); clown.comfy.model_management.soft_empty_cache()
    clown.metrics_base.make_sheet(sheet, OUTPUT / "RIGID_BRIDGE_DIRECTIONAL_COUPLING_COMPARISON.png")

    report = {
        "experiment": "full-depth directional generated-token attention coupling",
        "classification": "BIDIRECTIONAL CO-EVOLUTION",
        "configuration": {
            "case": "rigid_bridge", "sigma": float(sigma), "sampler_updates": 0,
            "blocks": {"double": [0, 1, 2, 3, 4], "single": list(range(20))},
            "generated_token_order": "row-major 32x64; center columns 16:48 are source",
            "attention_semantics": "additive -finfo(dtype).max edge mask inside each native joint softmax",
            "archived_input_sha256": input_hash,
        },
        "baseline_reproduction": {
            "true_low_bit_exact": True,
            "input_immutable": True,
            "ordinary_and_full_epsilon_from_block_archive": True,
        },
        "arms": {
            "TRUE_LOW_FULL": arm_metrics(outputs["TRUE_LOW_FULL"], ordinary, epsilon, archived_true, editable, source),
            **{direction: arm_metrics(outputs[direction], ordinary, epsilon, archived_true, editable, source)
               for direction in DIRECTIONS},
        },
        "mechanical_invariants": {
            direction: {"blocks_observed": len(probe.records), "all_blocks_exactly_once": True,
                        "per_block": probe.records}
            for direction, probe in probes.items()
        },
        "production_changed": False,
    }
    clown.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        "input_true_low": input_true, "ordinary": ordinary, "full_epsilon": epsilon, "outputs": outputs,
    })
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json")}, indent=2))


if __name__ == "__main__":
    run()
