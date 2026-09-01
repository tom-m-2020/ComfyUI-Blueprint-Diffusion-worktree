"""Capture/compare Phase-6j production boundaries around the terminal change."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE_ROOT = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_terminal_global_elimination_results"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_global_refresh_cadence as phase6i
import flux2_candidate3_performance_characterization as perf


def load_production_package() -> None:
    spec = importlib.util.spec_from_file_location(
        "blueprint_diffusion", PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["blueprint_diffusion"] = module
    spec.loader.exec_module(module)


load_production_package()
from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


def difference(value: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = value.float() - reference.float()
    return {
        "rms": float(delta.square().mean().sqrt()),
        "max_abs": float(delta.abs().max()),
        "bit_exact": bool(torch.equal(value, reference)),
    }


def run_case(model, positive, negative, noise, sigmas, seed):
    captured = {}

    def capture(name, ordinal, value):
        captured.setdefault(name, {})[ordinal] = value.detach().cpu().clone()

    sampler = BlueprintEulerSampler(capture=capture)
    previews = []
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise),
            callback=lambda ordinal, x0, x, total: previews.append(ordinal),
            disable_pbar=True, seed=seed,
        )
    torch.cuda.synchronize()
    return {
        "captured": captured,
        "output": output.detach().cpu().clone(),
        "telemetry": sampler.last_telemetry,
        "previews": previews,
        "wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("capture-baseline", "compare-production"))
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    report = {"mode": args.mode, "cases": {}}
    warmed_shapes = set()
    for name, width, height, prompt, seed in phase6i.CASES:
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        h_shape = (height // 16, width // 16)
        noise = torch.randn((1, 128, *h_shape), generator=torch.Generator().manual_seed(seed))
        sigmas = phase2.get_schedule(4, math.prod(h_shape)).float().clone()
        perf.prepare_model_state(model)
        if h_shape not in warmed_shapes:
            print(f"{args.mode}: warmup {h_shape}", flush=True)
            warmup = run_case(model, positive, negative, noise, sigmas, seed)
            del warmup
            warmed_shapes.add(h_shape)
        print(f"{args.mode}: {name}", flush=True)
        result = run_case(model, positive, negative, noise, sigmas, seed)
        baseline_path = OUTPUT / f"{name}_four_global_baseline.pt"
        if args.mode == "capture-baseline":
            torch.save({
                "captured": result["captured"], "output": result["output"],
                "telemetry": result["telemetry"], "previews": result["previews"],
                "sigmas": sigmas,
            }, baseline_path)
            report["cases"][name] = {
                "baseline_path": str(baseline_path),
                "wall_seconds": result["wall_seconds"],
                "previews": result["previews"],
                "telemetry": result["telemetry"],
            }
            continue

        baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        comparisons = {}
        expected = {
            "initial_H": (-1,), "initial_G": (-1,),
            "x0_G": (0, 1, 2), "assembled_x0_H": (0, 1, 2, 3),
            "G_star": (0, 1, 2), "H_star": (0, 1, 2, 3),
            "accepted_G": (0, 1, 2), "accepted_H": (0, 1, 2, 3),
        }
        all_exact = True
        for boundary, ordinals in expected.items():
            comparisons[boundary] = {}
            for ordinal in ordinals:
                item = difference(
                    result["captured"][boundary][ordinal],
                    baseline["captured"][boundary][ordinal],
                )
                comparisons[boundary][str(ordinal)] = item
                all_exact &= item["bit_exact"]
        final = difference(result["output"], baseline["output"])
        all_exact &= final["bit_exact"]
        terminal_absent = all(
            3 not in result["captured"].get(name, {}) for name in ("x0_G", "G_star")
        )
        terminal_retained = difference(
            result["captured"]["accepted_G"][3],
            result["captured"]["accepted_G"][2],
        )
        accepted = [x for x in result["telemetry"] if x["event"] == "accepted_interval"]
        lifecycle = {
            "four_previews": result["previews"] == [0, 1, 2, 3],
            "global_forward_flags": [x["global_forward_performed"] for x in accepted],
            "terminal_global_unused": [x["terminal_global_unused"] for x in accepted],
            "terminal_release_only_last": [x["terminal_release"] for x in accepted]
            == [False, False, False, True],
            "three_global_forwards": sum(x["global_forward_performed"] for x in accepted) == 3,
            "terminal_capture_absent": terminal_absent,
            "terminal_g_retained_exact": terminal_retained["bit_exact"],
            "nonterminal_invariants": all(
                x["invariant_max_abs"] is None or x["invariant_max_abs"] <= 2e-6
                for x in accepted
            ),
        }
        equivalent = all_exact and final["bit_exact"] and all(
            value if isinstance(value, bool) else True for value in lifecycle.values()
        )
        report["cases"][name] = {
            "equivalent": equivalent, "comparisons": comparisons,
            "final_H": final, "terminal_G_retained": terminal_retained,
            "lifecycle": lifecycle, "wall_seconds": result["wall_seconds"],
            "baseline_wall_seconds": next(
                item["wall_seconds"] for item in json.loads(
                    (OUTPUT / "baseline_report.json").read_text(encoding="utf-8")
                )["cases"].values() if item["baseline_path"] == str(baseline_path)
            ),
            "peak_allocated_bytes": result["peak_allocated_bytes"],
            "peak_reserved_bytes": result["peak_reserved_bytes"],
            "telemetry": result["telemetry"],
        }

    if args.mode == "capture-baseline":
        path = OUTPUT / "baseline_report.json"
    else:
        report["equivalent"] = all(c["equivalent"] for c in report["cases"].values())
        path = OUTPUT / "report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "equivalent": report.get("equivalent"),
        "cases": {
            name: {
                "equivalent": value.get("equivalent"),
                "wall_seconds": value["wall_seconds"],
                "previews": value.get("previews") or value.get("lifecycle", {}).get("four_previews"),
            }
            for name, value in report["cases"].items()
        },
        "report": str(path),
    }, indent=2))
    if args.mode == "compare-production" and not report["equivalent"]:
        raise AssertionError("Phase-6j production differs from frozen baseline")


if __name__ == "__main__":
    main()
