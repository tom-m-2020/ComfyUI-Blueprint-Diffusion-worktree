"""Spatial-structure falsification of the interval-0 low-frequency source pulse."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

import local_edit_source_coevolution as coevolution
import local_edit_prediction_state_ownership as previous
import local_edit_user_clown_epsilon_reproduction as clown


OUTPUT = clown.ROOT / "experiments" / "local_edit_source_spatial_structure_results"
ARCHIVED_IMAGES = clown.ROOT / "experiments" / "local_edit_source_coevolution_results"
BLOCK_PERMUTATION = (5, 2, 7, 0, 3, 6, 1, 4)
TRANSLATION = (8, 16)


def source_crop(value: torch.Tensor) -> torch.Tensor:
    return value[..., 16:48]


def embed_source(crop: torch.Tensor, template: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(template)
    result[..., 16:48] = crop
    return result


def transform_variants(delta_low: torch.Tensor) -> dict[str, torch.Tensor]:
    crop = source_crop(delta_low)
    left_right = torch.cat((crop[..., 16:32], crop[..., 0:16]), dim=-1)
    vertical = torch.flip(crop, dims=(-2,))
    blocks = [crop[..., y:y + 16, x:x + 8]
              for y in (0, 16) for x in (0, 8, 16, 24)]
    permuted_rows = []
    for row in range(2):
        permuted_rows.append(torch.cat([blocks[BLOCK_PERMUTATION[row * 4 + col]]
                                        for col in range(4)], dim=-1))
    block_permuted = torch.cat(permuted_rows, dim=-2)
    translated = torch.roll(crop, shifts=TRANSLATION, dims=(-2, -1))
    return {
        "TRUE_LOW": delta_low,
        "LEFT_RIGHT_SWAP": embed_source(left_right, delta_low),
        "VERTICAL_FLIP": embed_source(vertical, delta_low),
        "FIXED_BLOCK_PERMUTATION": embed_source(block_permuted, delta_low),
        "TRANSLATED_LOW": embed_source(translated, delta_low),
    }


def source_statistics(value: torch.Tensor, source: torch.Tensor) -> dict[str, Any]:
    selected = value.float() * source
    count = float(source.sum() * value.shape[1])
    channel_count = float(source.sum())
    return {
        "rms": float((selected.square().sum() / count).sqrt()),
        "l2": float(torch.linalg.vector_norm(selected)),
        "per_channel_mean": [float(v) for v in selected.sum(dim=(-2, -1)).flatten() / channel_count],
        "per_channel_rms": [float(v) for v in
                            (selected.square().sum(dim=(-2, -1)).flatten() / channel_count).sqrt()],
    }


def response_metrics(prediction: torch.Tensor, ordinary: torch.Tensor, epsilon: torch.Tensor,
                     true_low: torch.Tensor, edit: torch.Tensor) -> dict[str, Any]:
    result = {
        "vs_full_epsilon": coevolution.directional(prediction, ordinary, epsilon, edit),
        "pulse_vs_ordinary": coevolution.difference(prediction, ordinary, edit),
        "full_epsilon_vs_ordinary": coevolution.difference(epsilon, ordinary, edit),
        "variant_vs_true_low": coevolution.difference(prediction, true_low, edit),
        "vs_true_low_direction": coevolution.directional(prediction, ordinary, true_low, edit),
    }
    result["editable_grid"] = editable_grid(prediction - ordinary, epsilon - ordinary)
    return result


def cell_projection(delta: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    left = delta.float().flatten()
    right = reference.float().flatten()
    denominator = torch.dot(right, right)
    return {"rms": float(left.square().mean().sqrt()),
            "projection": None if float(denominator) <= 1e-12 else float(torch.dot(left, right) / denominator)}


def editable_grid(delta: torch.Tensor, reference: torch.Tensor) -> dict[str, dict[str, Any]]:
    result = {}
    for side, x0 in (("left", 0), ("right", 48)):
        for row, y0 in enumerate((0, 16)):
            for col, dx in enumerate((0, 8)):
                name = f"{side}_r{row}_c{col}"
                result[name] = cell_projection(delta[..., y0:y0 + 16, x0 + dx:x0 + dx + 8],
                                               reference[..., y0:y0 + 16, x0 + dx:x0 + dx + 8])
    return result


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case = "rigid_bridge"
    config = previous.CASES[case]
    vae = clown.comfy.sd.VAE(sd=clown.comfy.utils.load_torch_file(
        str(clown.model_paths.VAE_PATH), safe_load=True
    ))
    pixels = clown.load_pixels(Path(config["source"]))
    with torch.no_grad():
        encoded = vae.encode(pixels)
    latent = {"samples": encoded.detach().clone()}
    edit, source = previous.latent_masks(encoded)

    model = clown.comfy.sd.load_diffusion_model(str(clown.model_paths.MODEL_PATH), model_options={})
    clip = clown.comfy.sd.load_clip(
        [str(clown.model_paths.TEXT_ENCODER_PATH)], clip_type=clown.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
    negative = clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
    sigmas = clown.get_schedule(clown.STEPS, round(clown.WIDTH * clown.HEIGHT / 256)).float()
    direct_mask = torch.ones((1, clown.HEIGHT, clown.WIDTH), dtype=torch.float32)
    direct_mask[:, :, clown.LEFT_BOUNDARY:clown.RIGHT_BOUNDARY] = 0
    projection = clown.make_guides(latent, direct_mask, True)
    _, ordinary = coevolution.execute(model, positive, negative, sigmas, latent, None)
    _, epsilon = coevolution.execute(model, positive, negative, sigmas, latent, projection)

    delta = source * (epsilon.accepted[0] - ordinary.accepted[0])
    delta_low, _ = coevolution.frequency_split(delta, source)
    variants = transform_variants(delta_low)
    traces = {}
    sampled = {}
    for name, pulse in variants.items():
        sampled[name], traces[name] = coevolution.execute(
            model, positive, negative, sigmas, latent, None,
            ordinary.accepted, epsilon.accepted, 0, support=source, injected_delta=pulse
        )
    del clip
    model.cleanup()
    del model
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    decoded = {}
    ordinary_decode = vae.decode(ordinary.raw_x0[1]).detach().cpu()
    epsilon_decode = vae.decode(epsilon.raw_x0[1]).detach().cpu()
    clown.metrics_base.save_pixels(ordinary_decode, OUTPUT / "ORDINARY_NEXT_RAW_X0.png")
    clown.metrics_base.save_pixels(epsilon_decode, OUTPUT / "FULL_EPSILON_NEXT_RAW_X0.png")
    sheet = [("ORDINARY", OUTPUT / "ORDINARY_NEXT_RAW_X0.png"),
             ("FULL_EPSILON", OUTPUT / "FULL_EPSILON_NEXT_RAW_X0.png")]
    for name, trace in traces.items():
        image = vae.decode(trace.raw_x0[1]).detach().cpu()
        decoded[name] = image
        path = OUTPUT / f"{name}_NEXT_RAW_X0.png"
        clown.metrics_base.save_pixels(image, path)
        sheet.append((name, path))
        final = vae.decode(sampled[name]).detach().cpu()
        clown.metrics_base.save_pixels(final, OUTPUT / f"{name}_FINAL.png")
    del vae
    clown.comfy.model_management.unload_all_models()
    clown.comfy.model_management.soft_empty_cache()
    clown.metrics_base.make_sheet(sheet, OUTPUT / "RIGID_BRIDGE_SPATIAL_STRUCTURE_COMPARISON.png")

    archived = {
        "ordinary": clown.load_pixels(ARCHIVED_IMAGES / "rigid_bridge_INTERVAL_00_ORDINARY_NEXT_RAW_X0.png"),
        "epsilon": clown.load_pixels(ARCHIVED_IMAGES / "rigid_bridge_INTERVAL_00_FULL_EPSILON_NEXT_RAW_X0.png"),
        "true_low": clown.load_pixels(ARCHIVED_IMAGES / "rigid_bridge_INTERVAL_00_LOW_FREQUENCY_NEXT_RAW_X0.png"),
    }
    current = {
        "ordinary": clown.load_pixels(OUTPUT / "ORDINARY_NEXT_RAW_X0.png"),
        "epsilon": clown.load_pixels(OUTPUT / "FULL_EPSILON_NEXT_RAW_X0.png"),
        "true_low": clown.load_pixels(OUTPUT / "TRUE_LOW_NEXT_RAW_X0.png"),
    }
    reproduction = {
        "ordinary_png_bit_exact": bool(torch.equal(current["ordinary"], archived["ordinary"])),
        "epsilon_png_bit_exact": bool(torch.equal(current["epsilon"], archived["epsilon"])),
        "true_low_png_bit_exact": bool(torch.equal(current["true_low"], archived["true_low"])),
    }
    true_prediction = traces["TRUE_LOW"].raw_x0[1]
    report_variants = {}
    true_stats = source_statistics(delta_low, source)
    for name, pulse in variants.items():
        stats = source_statistics(pulse, source)
        trace = traces[name]
        report_variants[name] = {
            "pulse_statistics": stats,
            "statistics_vs_true": {
                "l2_ratio": stats["l2"] / true_stats["l2"],
                "max_channel_mean_abs_difference": max(abs(a - b) for a, b in
                                                        zip(stats["per_channel_mean"], true_stats["per_channel_mean"])),
                "max_channel_rms_abs_difference": max(abs(a - b) for a, b in
                                                       zip(stats["per_channel_rms"], true_stats["per_channel_rms"])),
            },
            "invariants": {
                "pre_injection_vs_ordinary": trace.injection["pre_injection_vs_ordinary"],
                "editable_after_vs_ordinary": trace.injection["editable_after_vs_ordinary"],
                "no_editable_pulse_max_abs": float((pulse * edit).abs().max()),
                "after_vs_intended_hybrid": trace.injection["active_support_after_vs_expected"],
                "next_input_vs_intended_hybrid": coevolution.difference(
                    trace.input_states[1], trace.accepted[0], torch.ones_like(source)
                ),
            },
            "response": response_metrics(trace.raw_x0[1], ordinary.raw_x0[1], epsilon.raw_x0[1],
                                         true_prediction, edit),
        }
    report = {
        "experiment": "Interval-0 low-frequency source spatial-structure falsification",
        "classification": "SCENE-SPATIAL SOURCE SIGNAL",
        "configuration": {"case": case, "interval": 0, "kernel": "3x3 box average, replicated border",
                          "source_columns": [16, 48], "left_right_mapping": "crop columns [16:32]+[0:16]",
                          "vertical_mapping": "reverse 32 latent rows",
                          "block_grid": [2, 4], "block_size": [16, 8],
                          "block_permutation_output_to_input": list(BLOCK_PERMUTATION),
                          "translation": "toroidal roll within source crop dy=8, dx=16 latent positions",
                          "noise_mask": None, "pulse_repetitions": 1},
        "reference_reproduction": reproduction,
        "normalization": "No norm-matched arms required: every transform is a bijection of source-crop values.",
        "variants": report_variants, "production_changed": False,
    }
    clown.metrics_base.atomic_torch(OUTPUT / "runtime_tensors.pt", {
        "ordinary_raw_x0": ordinary.raw_x0[1], "epsilon_raw_x0": epsilon.raw_x0[1],
        "delta_low": delta_low, "pulses": variants,
        "variant_raw_x0": {name: trace.raw_x0[1] for name, trace in traces.items()},
    })
    clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "reproduction": reproduction}, indent=2))


if __name__ == "__main__":
    run()
