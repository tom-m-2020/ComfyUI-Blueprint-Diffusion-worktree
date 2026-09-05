"""Phase 27 exact production terminal-resampling qualification."""
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
sys.path.insert(0, str(COMFY_ROOT)); sys.path.insert(0, str(ROOT / "experiments")); sys.path.insert(0, str(PACKAGE))

SPEC = importlib.util.spec_from_file_location(
    "blueprint_diffusion",
    PACKAGE / "__init__.py",
    submodule_search_locations=[str(PACKAGE)],
)
if "blueprint_diffusion" not in sys.modules:
    MODULE = importlib.util.module_from_spec(SPEC)
    sys.modules["blueprint_diffusion"] = MODULE
    SPEC.loader.exec_module(MODULE)

import comfy.model_management
import comfy.sampler_helpers
import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_blueprint_local_resampling_trajectory as phase23
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_blueprint_local_state as phase20
from blueprint_diffusion.terminal_resampling import QUALIFIED_SIGMAS, TerminalResamplingProcedure, tensor_hash

OUTPUT = ROOT / "experiments" / "terminal_resampling_production_qualification_results"
REPORT = OUTPUT / "report.json"
SEED = 20260911
PROMPT = (
    "A cinematic photograph of one red vintage car on the far left, one tall green tree in the center, "
    "and one small white house on the far right, all standing on the same grassy field under one continuous "
    "horizon, coherent perspective, exactly one car, one tree, and one house, no duplicate objects."
)
REFERENCE_ROOT = ROOT / "experiments" / "flux2_candidate3_terminal_resampling_generalization_results"


def difference(left, right):
    delta = left.float() - right.float()
    return {"bit_exact": bool(torch.equal(left, right)), "rms": float(delta.square().mean().sqrt()), "max_abs": float(delta.abs().max())}


def decode(vae, latent, path):
    phase2.save_pixels(vae.decode(latent).cpu(), path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference_blueprint = torch.load(
        REFERENCE_ROOT / "cases" / "B_MULTI_OBJECT" / "BLUEPRINT.pt",
        map_location="cpu", weights_only=True,
    )
    reference_local = torch.load(
        REFERENCE_ROOT / "cases" / "B_MULTI_OBJECT" / "BLUEPRINT_RESAMPLED.pt",
        map_location="cpu", weights_only=True,
    )
    reference_report = json.loads((REFERENCE_ROOT / "report.json").read_text(encoding="utf-8"))

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT)); del clip
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()
    import flux2_candidate3_performance_characterization as perf
    perf.prepare_model_state(model); device = comfy.model_management.get_torch_device()
    guider = phase23.setup_guider(model, positive, device)
    sigmas = torch.tensor(QUALIFIED_SIGMAS, device=device)
    captures = {}
    def capture(name, ordinal, value):
        captures.setdefault(name, []).append(value.detach().float().cpu())

    procedure = TerminalResamplingProcedure(seed=SEED, capture=capture)
    destination = torch.zeros((1, 128, 128, 256), device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        final_h = procedure.sample(
            guider, sigmas, {"model_options": phase20.phase8i_options({}), "seed": SEED},
            None, torch.zeros_like(destination), destination, None, False,
        )
    runtime = time.perf_counter() - started

    comparisons = {
        "initial_B": {"expected_hash": reference_report["results"]["B_MULTI_OBJECT"]["blueprint"]["initial_hash"], "actual_hash": tensor_hash(captures["initial_B"][0])},
        "x0_B": [difference(a, b) for a, b in zip(captures["x0_B"], reference_blueprint["x0_history"])],
        "terminal_x0_B": difference(captures["x0_B"][-1], reference_blueprint["terminal_x0"]),
        "mapped_terminal_B": difference(captures["mapped_terminal_B"][0], reference_blueprint["mapped_terminal"]),
        "final_H": difference(final_h.cpu(), reference_local["assembled"]),
    }
    comparisons["accepted_B"] = []
    state = captures["initial_B"][0]
    for ordinal, x0 in enumerate(reference_blueprint["x0_history"]):
        sigma = torch.tensor(QUALIFIED_SIGMAS[ordinal])
        sigma_next = torch.tensor(QUALIFIED_SIGMAS[ordinal + 1])
        state = state + (sigma_next - sigma) * (state - x0) / sigma
        comparisons["accepted_B"].append(difference(captures["accepted_B"][ordinal], state))
    comparisons["initial_B"]["bit_exact"] = comparisons["initial_B"]["expected_hash"] == comparisons["initial_B"]["actual_hash"]
    expected_noise = reference_report["results"]["B_MULTI_OBJECT"]["resampled"]["noise_hashes"]
    actual_noise = [item["noise_hash"] for item in procedure.telemetry["regions"]]
    expected_restricted_hashes = [tensor_hash(value) for value in reference_local["restricted"]]
    actual_restricted_hashes = [item["restricted_hash"] for item in procedure.telemetry["regions"]]
    comparisons["noise_hashes"] = {"bit_exact": actual_noise == expected_noise, "count": len(actual_noise)}
    comparisons["restricted_hashes"] = {"bit_exact": actual_restricted_hashes == expected_restricted_hashes, "count": len(actual_restricted_hashes)}

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    output_image = OUTPUT / "B_MULTI_OBJECT_PRODUCTION.png"
    decoded_hash = decode(vae, final_h.cpu(), output_image)
    expected_rgb = reference_report["decoded"]["B_MULTI_OBJECT_BLUEPRINT_RESAMPLED"]["sha256_rgb"]
    all_exact = (
        comparisons["initial_B"]["bit_exact"]
        and all(item["bit_exact"] for item in comparisons["x0_B"])
        and all(item["bit_exact"] for item in comparisons["accepted_B"])
        and comparisons["terminal_x0_B"]["bit_exact"]
        and comparisons["mapped_terminal_B"]["bit_exact"]
        and comparisons["noise_hashes"]["bit_exact"]
        and comparisons["restricted_hashes"]["bit_exact"]
        and comparisons["final_H"]["bit_exact"]
        and decoded_hash == expected_rgb
    )
    report = {
        "phase": 27,
        "case": "B_MULTI_OBJECT",
        "seed": SEED,
        "comparisons": comparisons,
        "decoded": {"actual_hash": decoded_hash, "expected_hash": expected_rgb, "bit_exact": decoded_hash == expected_rgb, "path": str(output_image)},
        "telemetry": procedure.telemetry,
        "runtime_seconds": runtime,
        "all_available_boundaries_bit_exact": all_exact,
        "earliest_mismatch": None if all_exact else next((name for name, value in comparisons.items() if isinstance(value, dict) and value.get("bit_exact") is False), "per_interval_or_decoded"),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "all_exact": all_exact, "decoded_hash": decoded_hash}, indent=2))
    model.cleanup(); comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()


if __name__ == "__main__":
    main()
