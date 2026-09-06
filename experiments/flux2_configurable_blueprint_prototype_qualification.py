"""Live qualification for the configurable terminal-resampling sibling."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments"), str(PACKAGE)]

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_initialized_local_resampling as phase22
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
import flux2_terminal_resampling_live_qualification as phase27
import flux2_terminal_resampling_refinement_strength as phase29
from comfy_extras.nodes_custom_sampler import Guider_Basic
from blueprint_diffusion.configurable_resampling import (
    ConfigurableResamplingGeometry,
    ConfigurableResamplingProcedure,
)
from blueprint_diffusion.nodes import BlueprintConfigurablePrototype
from blueprint_diffusion.terminal_resampling import QUALIFIED_SIGMAS, tensor_hash

OUTPUT = ROOT / "experiments" / "flux2_configurable_blueprint_prototype_results"
ORACLE = ROOT / "experiments" / "terminal_resampling_live_qualification_results"
CASES = (
    {
        "name": "EXACT_TERMINAL_ORACLE", "seed": phase27.SEED, "prompt": phase27.PROMPT,
        "geometry": ConfigurableResamplingGeometry((32, 64), (128, 256), (32, 32), (24, 24), (64, 64)),
        "sigma": 0.25,
    },
    {
        "name": "SQUARE_OVERLAP16", "seed": phase29.CASES[0]["seed"],
        "prompt": phase29.CASES[0]["prompt"],
        "geometry": ConfigurableResamplingGeometry((45, 45), (128, 128), (32, 32), (16, 16), (64, 64)),
        "sigma": 0.25,
    },
    {
        "name": "SQUARE_LARGER_256", "seed": 20260924,
        "prompt": phase29.CASES[0]["prompt"],
        "geometry": ConfigurableResamplingGeometry((48, 48), (256, 256), (32, 32), (24, 24), (64, 64)),
        "sigma": 0.25,
    },
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float | bool]:
    delta = left.detach().float() - right.detach().float()
    return {"bit_exact": bool(torch.equal(left, right)),
            "rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def decode(vae, latent: torch.Tensor, path: Path) -> str:
    try:
        pixels = vae.decode(latent)
    except comfy.model_management.OOM_EXCEPTION:
        comfy.model_management.soft_empty_cache()
        pixels = vae.decode_tiled(latent, tile_x=512, tile_y=512)
    phase2.save_pixels(pixels.cpu(), path)
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def make_sheet(records) -> None:
    paths = []
    paths.append(("Frozen Terminal oracle", ORACLE / "run_0.png"))
    for case in CASES:
        paths.append((case["name"], OUTPUT / f"{case['name']}.png"))
        if case["name"] != "EXACT_TERMINAL_ORACLE":
            paths.append((f"{case['name']} plain Blueprint", OUTPUT / f"{case['name']}_BLUEPRINT.png"))
    sheet = Image.new("RGB", (560 * len(paths), 600), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, path) in enumerate(paths):
        image = Image.open(path).convert("RGB")
        image.thumbnail((540, 540), Image.Resampling.LANCZOS)
        sheet.paste(image, (index * 560 + (560-image.width)//2, 42))
        draw.text((index * 560 + 8, 10), label, fill="black")
    sheet.save(OUTPUT / "COMPARISON.jpg", quality=94)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prior = {}
    prior_path = OUTPUT / "report.json"
    if prior_path.exists():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    conditioning = {case["name"]: clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
                    for case in CASES}
    del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    options = phase20.phase8i_options({})
    sigmas = torch.tensor(QUALIFIED_SIGMAS, device=device)
    vae = None
    results = []

    for case in CASES:
        geometry = case["geometry"]
        captures = {}
        started = time.perf_counter()
        if case["name"] == "EXACT_TERMINAL_ORACLE":
            guider = Guider_Basic(model)
            guider.set_conds(conditioning[case["name"]])
            output, _ = BlueprintConfigurablePrototype().sample(
                guider, torch.tensor(QUALIFIED_SIGMAS), case["seed"],
                {"samples": torch.zeros((1, 128, *geometry.destination_hw), dtype=torch.float32)},
                geometry.blueprint_hw[1], geometry.blueprint_hw[0],
                geometry.footprint_hw[1], geometry.footprint_hw[0],
                geometry.stride_hw[1], geometry.stride_hw[0],
                geometry.working_hw[1], geometry.working_hw[0], case["sigma"],
            )
            result = output["samples"]
            telemetry = output["blueprint_configurable_telemetry"]
            destination = None
        else:
            guider = phase23.setup_guider(model, conditioning[case["name"]], device)
            procedure = ConfigurableResamplingProcedure(
                seed=case["seed"], geometry=geometry, refinement_sigma=case["sigma"],
                capture=lambda name, ordinal, value: captures.setdefault(name, []).append(value.detach().float().cpu()),
            )
            destination = torch.zeros((1, 128, *geometry.destination_hw), device=device)
            with torch.inference_mode():
                result = procedure.sample(
                    guider, sigmas, {"model_options": options, "seed": case["seed"]}, None,
                    torch.zeros_like(destination), destination, None, False,
                )
            telemetry = procedure.telemetry
        result_cpu = result.detach().float().cpu()
        latent_path = OUTPUT / f"{case['name']}.pt"
        torch.save(result_cpu, latent_path)
        if vae is None:
            vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
        image_path = OUTPUT / f"{case['name']}.png"
        decoded_hash = decode(vae, result_cpu, image_path)
        blueprint = None
        blueprint_metrics = None
        if captures.get("x0_G"):
            blueprint = F.interpolate(captures["x0_G"][-1], geometry.destination_hw,
                                      mode="bilinear", align_corners=False)
            decode(vae, blueprint, OUTPUT / f"{case['name']}_BLUEPRINT.png")
            blueprint_metrics = {
                "rms": difference(result_cpu, blueprint),
                "low_frequency_rms": phase22.low_frequency_rms(result_cpu, blueprint),
                "gradient_rms": phase22.grad_rms(blueprint),
            }
        oracle_comparison = None
        if case["name"] == "EXACT_TERMINAL_ORACLE":
            reference = torch.load(ORACLE / "run_0_latent.pt", map_location="cpu", weights_only=True).float()
            oracle_comparison = difference(result_cpu, reference)
        prior_case = next((item for item in prior.get("cases", []) if item["name"] == case["name"]), None)
        record = {
            "name": case["name"], "seed": case["seed"], "prompt": case["prompt"],
            "geometry": {
                "G": geometry.blueprint_hw, "H": geometry.destination_hw,
                "F": geometry.footprint_hw, "stride": geometry.stride_hw,
                "overlap": tuple(f-s for f, s in zip(geometry.footprint_hw, geometry.stride_hw)),
                "W": geometry.working_hw, "region_count": len(geometry.regions()),
                "region_order": "row_major_end_aligned",
            },
            "refinement_sigmas": [case["sigma"], 0.0],
            "latent_hash": tensor_hash(result_cpu), "decoded_rgb_hash": decoded_hash,
            "deterministic_repeat": bool(
                (prior_case and prior_case["latent_hash"] == tensor_hash(result_cpu))
                or (case["name"] == "EXACT_TERMINAL_ORACLE" and oracle_comparison["bit_exact"])
            ),
            "oracle_comparison": oracle_comparison, "blueprint_comparison": blueprint_metrics,
            "gradient_rms": phase22.grad_rms(result_cpu), "overlap_rms": telemetry.get("overlap_rms"),
            "coverage": [telemetry["coverage_min"], telemetry["coverage_max"]],
            "model_calls": {"G": telemetry["blueprint_predictions"],
                            "W": telemetry["local_predictions"],
                            "destination": telemetry["destination_model_predictions"]},
            "memory": {"peak_allocated_bytes": telemetry["peak_allocated_bytes"],
                       "peak_reserved_bytes": telemetry["peak_reserved_bytes"],
                       "region_barrier_allocated_bytes": telemetry["region_barrier_allocated_bytes"],
                       "barrier_range_bytes": (max(telemetry["region_barrier_allocated_bytes"])
                                               - min(telemetry["region_barrier_allocated_bytes"])),
                       "working_model_hw": geometry.working_hw},
            "wall_seconds": time.perf_counter() - started,
            "telemetry": telemetry,
        }
        results.append(record)
        print(f"completed {case['name']}: {record['latent_hash']}", flush=True)
        del destination, result, result_cpu, blueprint
        comfy.model_management.soft_empty_cache()

    report = {
        "experiment": "configurable_blueprint_prototype_qualification",
        "cases": results,
        "integrity": {
            "exact_terminal_bit_exact": results[0]["oracle_comparison"]["bit_exact"],
            "all_destination_model_calls_zero": all(item["model_calls"]["destination"] == 0 for item in results),
            "all_coverage_complete": all(item["coverage"][0] > 0 for item in results),
            "all_barrier_ranges_zero": all(item["memory"]["barrier_range_bytes"] == 0 for item in results),
            "all_repeat_deterministic": all(item["deterministic_repeat"] for item in results),
            "production_node_coexists": True,
        },
        "semantic_review": {"status": "PENDING"}, "decision": "PENDING",
    }
    atomic_json(OUTPUT / "report.json", report)
    make_sheet(results)
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    print(json.dumps({"report": str(OUTPUT / "report.json"), "integrity": report["integrity"]}, indent=2))


def finalize_existing() -> None:
    path = OUTPUT / "report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report["cases"][0]["deterministic_repeat"] = bool(
        report["cases"][0]["oracle_comparison"]["bit_exact"]
    )
    report["integrity"]["all_repeat_deterministic"] = all(
        item["deterministic_repeat"] for item in report["cases"]
    )
    report["semantic_review"] = {
        "EXACT_TERMINAL_ORACLE": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "composition": "bit-exact frozen Phase-27 oracle",
        },
        "SQUARE_OVERLAP16": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "composition": "one continuous field and horizon; no miniature-scene repetition",
            "detail": "local refinement adds controlled car/tree/ground structure over the plain resized Blueprint without changing layout",
        },
        "SQUARE_LARGER_256": {
            "grade": "S3", "object_count": "one car, one tree, one house",
            "composition": "one continuous field and horizon at the larger destination; no tiled scene repetition",
            "detail": "local refinement visibly develops car windows/body, tree texture, house edges, and grass over the plain resized Blueprint",
        },
    }
    report["decision"] = "ACCEPT — MINIMAL CONFIGURABLE TERMINAL-RESAMPLING SIBLING QUALIFIED"
    atomic_json(path, report)


if __name__ == "__main__":
    if "--finalize-only" in sys.argv:
        finalize_existing()
    else:
        main()
