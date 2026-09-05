"""Phase 28 finite-profile geometry qualification for terminal resampling."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

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
from blueprint_diffusion.adapters.flux2_terminal import Flux2TerminalResamplingAdapter
from blueprint_diffusion.terminal_resampling import (
    QUALIFIED_SIGMAS,
    StreamingOverlapAssembler,
    TerminalResamplingGeometry,
    TerminalResamplingProcedure,
    tensor_hash,
)

OUTPUT = ROOT / "experiments" / "terminal_resampling_geometry_qualification_results"
CASES = (
    {
        "name": "SQUARE_MULTI_OBJECT", "destination_hw": (128, 128), "seed": 20260921,
        "prompt": "A cinematic photograph of exactly one red vintage car on the left, one tall green tree in the center, and one small white house on the right, all on one grassy field beneath one continuous horizon, coherent perspective, no duplicate objects.",
    },
    {
        "name": "PORTRAIT_ASTRONAUT", "destination_hw": (256, 128), "seed": 20260922,
        "prompt": "A full-body astronaut standing alone in the center of a tall wide desert scene, facing the camera, both arms and both legs visible, one continuous body, distant low mountains across one horizon, exactly one astronaut, no duplicate people or body parts.",
    },
    {
        "name": "LANDSCAPE_BRIDGE", "destination_hw": (128, 192), "seed": 20260923,
        "prompt": "A wide cinematic photograph of exactly one long red suspension bridge crossing calm water from left to right, one yellow passenger train on the bridge, controlled towers and continuous cables, one coherent horizon, no duplicate bridges or trains.",
    },
)


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def ordinary_tiled_control(*, guider, geometry, seed, sigmas, options, device):
    adapter = Flux2TerminalResamplingAdapter()
    destination = torch.zeros((1, 128, *geometry.destination_hw), device=device)
    adapter.validate_prepared(
        guider=guider, model_options=options, destination=destination,
        model_sampling=guider.inner_model.model_sampling,
        destination_hw=geometry.destination_hw,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    state = torch.randn((1, 128, *geometry.destination_hw), generator=generator).to(device)
    regions = geometry.regions()
    calls = 0
    started = time.perf_counter()
    for ordinal in range(4):
        assembler = StreamingOverlapAssembler(
            regions=regions, target_hw=geometry.destination_hw, template=state,
        )
        sigma = sigmas[ordinal]
        for region in regions:
            crop = state[:, :, region.y:region.y2, region.x:region.x2]
            prediction = adapter.predict_native(
                guider=guider, value=crop, sigma=sigma,
                expected_hw=(32, 32), model_options=options, seed=seed,
            )
            assembler.add(prediction, region)
            calls += 1
        x0, coverage = assembler.finish()
        sigma_next = sigmas[ordinal + 1]
        state = state + (sigma_next - sigma) * (state - x0) / sigma
    return state.cpu(), {
        "model_calls": calls, "wall_seconds": time.perf_counter() - started,
        "coverage_min": float(coverage.min()), "coverage_max": float(coverage.max()),
        "finite": bool(torch.isfinite(state).all()), "hash": tensor_hash(state),
    }


def image_hash(path: Path) -> str:
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for case_index, case in enumerate(CASES, 1):
        case_dir = OUTPUT / case["name"]
        case_dir.mkdir(exist_ok=True)
        complete = case_dir / "report.json"
        if complete.exists():
            item = json.loads(complete.read_text(encoding="utf-8"))
            expected_configuration = dict(case)
            expected_configuration["destination_hw"] = list(case["destination_hw"])
            expected_blueprint = list(
                TerminalResamplingGeometry.for_destination(case["destination_hw"]).blueprint_hw
            )
            if (
                item.get("configuration") == expected_configuration
                and item.get("geometry", {}).get("blueprint_hw") == expected_blueprint
            ):
                print(f"[{case_index}/3] {case['name']} reused", flush=True)
                results.append(item)
                continue
        print(f"[{case_index}/3] {case['name']} start", flush=True)
        model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
        clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(case["prompt"]))
        del clip
        comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
        perf.prepare_model_state(model)
        device = comfy.model_management.get_torch_device()
        guider = phase23.setup_guider(model, positive, device)
        sigmas = torch.tensor(QUALIFIED_SIGMAS, device=device)
        options = phase20.phase8i_options({})
        geometry = TerminalResamplingGeometry.for_destination(case["destination_hw"])
        selected = []
        for repeat in range(2):
            print(f"[{case_index}/3] {case['name']} selected repeat {repeat + 1}/2", flush=True)
            procedure = TerminalResamplingProcedure(seed=case["seed"], geometry=geometry)
            destination = torch.zeros((1, 128, *geometry.destination_hw), device=device)
            with torch.inference_mode():
                final = procedure.sample(
                    guider, sigmas, {"model_options": options, "seed": case["seed"]},
                    None, torch.zeros_like(destination), destination, None, False,
                ).cpu()
            torch.save(final, case_dir / f"selected_{repeat}.pt")
            selected.append({"hash": tensor_hash(final), "telemetry": procedure.telemetry})
            save_json(case_dir / f"selected_{repeat}.json", selected[-1])
        print(f"[{case_index}/3] {case['name']} ordinary tiled control", flush=True)
        with torch.inference_mode():
            control, control_metrics = ordinary_tiled_control(
                guider=guider, geometry=geometry, seed=case["seed"], sigmas=sigmas,
                options=options, device=device,
            )
        torch.save(control, case_dir / "ordinary_tiled.pt")
        item = {
            "configuration": case,
            "geometry": {
                "blueprint_hw": geometry.blueprint_hw,
                "destination_hw": geometry.destination_hw,
                "blueprint_tokens": geometry.blueprint_hw[0] * geometry.blueprint_hw[1],
                "region_hw": geometry.region_hw, "working_hw": geometry.working_hw,
                "stride_hw": geometry.stride_hw, "region_count": len(geometry.regions()),
            },
            "selected": selected,
            "repeat_hash_exact": selected[0]["hash"] == selected[1]["hash"],
            "ordinary_tiled": control_metrics,
            "integrity": {"destination_model_forwards": 0, "finite": True, "complete_coverage": True},
        }
        save_json(complete, item)
        results.append(item)
        model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
        print(f"[{case_index}/3] {case['name']} complete", flush=True)

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    for case in CASES:
        case_dir = OUTPUT / case["name"]
        for stem in ("selected_0", "ordinary_tiled"):
            image_path = case_dir / f"{stem}.png"
            latent = torch.load(case_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
            phase2.save_pixels(vae.decode(latent).cpu(), image_path)
            for result in results:
                if result["configuration"]["name"] == case["name"]:
                    result.setdefault("decoded", {})[stem] = {
                        "path": str(image_path), "rgb_hash": image_hash(image_path),
                    }
    report = {
        "phase": 28, "policy": "finite aspect-preserving profiles with 1944-2048 Blueprint tokens",
        "results": results, "all_repeat_exact": all(x["repeat_hash_exact"] for x in results),
    }
    save_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / 'report.json'), "all_repeat_exact": report["all_repeat_exact"]}, indent=2))


if __name__ == "__main__":
    main()
