"""Phase 27 exact bridge and astronaut production-path qualification."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
PACKAGE = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
sys.path[:0] = [str(COMFY_ROOT), str(ROOT / "experiments"), str(PACKAGE)]
spec = importlib.util.spec_from_file_location(
    "blueprint_diffusion", PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
module = importlib.util.module_from_spec(spec)
sys.modules["blueprint_diffusion"] = module
spec.loader.exec_module(module)

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_native_blueprint_local_state as phase20
import flux2_candidate3_performance_characterization as perf
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    TerminalResamplingProcedure,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "terminal_resampling_production_semantic_results"
CASES = (
    {
        "name": "BRIDGE_TRAIN",
        "seed": 20260901,
        "prompt": "A wide cinematic photograph of one single long red suspension bridge stretching continuously from the far left edge to the far right edge over calm water, one yellow passenger train centered on the bridge, one white lighthouse at the far left, one dark stone tower at the far right, continuous bridge deck and cables, coherent perspective, no duplicate bridges, trains, lighthouses, or towers",
        "reference": ROOT / "experiments" / "flux2_candidate3_blueprint_resampling_cadence_results" / "stages" / "A_TERMINAL_ONLY.pt",
        "reference_key": "final_H",
        "semantic": "S3: one continuous bridge/train system with coherent horizon and water",
    },
    {
        "name": "ASTRONAUT",
        "seed": 20260912,
        "prompt": "A full-body astronaut standing alone in the center of a wide desert, facing the camera, both arms visible, both legs visible, one continuous body, distant low mountains across one horizon, exactly one astronaut, no duplicate people or body parts.",
        "reference": ROOT / "experiments" / "flux2_candidate3_terminal_resampling_generalization_results" / "cases" / "C_ASTRONAUT" / "BLUEPRINT_RESAMPLED.pt",
        "reference_key": "assembled",
        "semantic": "S3: one centered continuous astronaut with coherent anatomy and ground contact",
    },
)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    perf.prepare_model_state(model)
    device = comfy.model_management.get_torch_device()
    results = []
    for ordinal, case in enumerate(CASES, 1):
        print(f"[{ordinal}/{len(CASES)}] {case['name']} start", flush=True)
        clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
        del clip
        comfy.model_management.soft_empty_cache()
        guider = phase23.setup_guider(model, positive, device)
        procedure = TerminalResamplingProcedure(seed=case["seed"])
        destination = torch.zeros((1, 128, 128, 256), device=device)
        started = time.perf_counter()
        with torch.inference_mode():
            actual = procedure.sample(
                guider,
                torch.tensor(QUALIFIED_SIGMAS, device=device),
                {"model_options": phase20.phase8i_options({}), "seed": case["seed"]},
                None, torch.zeros_like(destination), destination, None, False,
            ).cpu()
        elapsed = time.perf_counter() - started
        reference_bundle = torch.load(case["reference"], map_location="cpu", weights_only=True)
        expected = reference_bundle[case["reference_key"]]
        delta = actual.float() - expected.float()
        exact = bool(torch.equal(actual, expected))
        item = {
            "case": case["name"], "seed": case["seed"], "semantic_class": "S3",
            "semantic_provenance": case["semantic"], "production_hash": tensor_hash(actual),
            "reference_hash": tensor_hash(expected), "bit_exact": exact,
            "rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max()),
            "wall_seconds": elapsed, "telemetry": procedure.telemetry,
        }
        torch.save(actual, OUTPUT / f"{case['name']}_production.pt")
        (OUTPUT / f"{case['name']}.json").write_text(json.dumps(item, indent=2), encoding="utf-8")
        results.append(item)
        print(f"[{ordinal}/{len(CASES)}] {case['name']} done exact={exact} elapsed={elapsed:.3f}s", flush=True)
        model.cleanup()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if ordinal != len(CASES):
            model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
            perf.prepare_model_state(model)
            device = comfy.model_management.get_torch_device()
    report = {"phase": 27, "results": results, "all_bit_exact": all(x["bit_exact"] for x in results)}
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(OUTPUT / 'report.json'), "all_bit_exact": report["all_bit_exact"]}, indent=2))


if __name__ == "__main__":
    main()
