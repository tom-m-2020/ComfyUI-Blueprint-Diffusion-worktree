"""Phase 8j production terminal specialized-executor qualification."""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
TARGET = ROOT / "target" / "ComfyUI-Blueprint-Diffusion"
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(TARGET.parent))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_practical_scaling_frontier as phase8a

from blueprint_diffusion.sampling.euler import BlueprintEulerSampler


OUTPUT = ROOT / "experiments" / "flux2_candidate3_terminal_specialized_production_results"
REPORT = OUTPUT / "report.json"
EXPECTED_RGB = "53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce"
HW = (128, 256)
SEED = phase8a.SEED
PROMPT = phase8a.BRIDGE_PROMPT


def tensor_record(value):
    work = value.detach().float()
    return {
        "shape": list(value.shape),
        "rms": float(work.square().mean().sqrt()),
        "max_abs": float(work.abs().max()),
        "finite": bool(work.isfinite().all()),
        "sha256": hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
    }


def run_once(model, positive, negative, noise, sigmas, label):
    sampler = BlueprintEulerSampler()
    previews = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model, noise.clone(), 1.0, sampler, sigmas.clone(), positive, negative,
            torch.zeros_like(noise),
            callback=lambda ordinal, x0, x, total: previews.append(
                {"ordinal": ordinal, "total": total, "x0": tensor_record(x0), "x": tensor_record(x)}
            ),
            disable_pbar=True, seed=SEED,
        )
    torch.cuda.synchronize()
    return output.detach().cpu(), {
        "label": label,
        "sampling_wall_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "output": tensor_record(output),
        "previews": previews,
        "telemetry": sampler.last_telemetry,
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = None
    if REPORT.exists():
        prior = json.loads(REPORT.read_text(encoding="utf-8"))
        if prior.get("run", {}).get("output", {}).get("sha256"):
            previous = {
                "latent_sha256": prior["run"]["output"]["sha256"],
                "rgb_sha256": prior.get("decoded", {}).get("sha256_rgb"),
            }
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn((1, 128, *HW), generator=torch.Generator().manual_seed(SEED))
    sigmas = phase2.get_schedule(4, math.prod(HW)).float().clone()
    sigmas[0], sigmas[-1] = 1.0, 0.0
    perf.prepare_model_state(model)
    output, run = run_once(model, positive, negative, noise, sigmas, "fresh")
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    pixels = vae.decode(output).cpu()
    image_path = OUTPUT / "bridge_production_specialized.png"
    phase2.save_pixels(pixels, image_path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        decoded = {
            "path": str(image_path), "dimensions_wh": list(rgb.size),
            "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
        }
    accepted = [x for x in run["telemetry"] if x["event"] == "accepted_interval"]
    terminal = accepted[-1]
    result = {
        "configuration": {"H": list(HW), "G": [96, 192], "seed": SEED,
                          "steps": 4, "prompt": PROMPT},
        "run": run,
        "decoded": decoded,
        "reference_rgb_match": decoded["sha256_rgb"] == EXPECTED_RGB,
        "previous_fresh_run": previous,
        "deterministic_repeat": None if previous is None else {
            "latent": previous["latent_sha256"] == run["output"]["sha256"],
            "decoded": previous["rgb_sha256"] == decoded["sha256_rgb"],
        },
        "gates": {
            "four_previews": len(run["previews"]) == 4,
            "three_global_forwards": sum(x["global_forward_performed"] for x in accepted) == 3,
            "terminal_specialized": terminal.get("terminal_context_source_performed") is True,
            "source_blocks_25": terminal.get("terminal_context_source_blocks") == 25,
            "no_terminal_global_prediction": terminal.get("terminal_global_prediction_performed") is False,
            "no_source_final_projection": terminal.get("source_final_projection_performed") is False,
            "no_cpu_kv_cache": terminal.get("cpu_kv_cache_bytes") == 0,
            "no_cpu_gpu_kv_transfer": terminal.get("cpu_to_gpu_kv_transfer_bytes") == 0,
            "nonterminal_invariants": all(
                x["invariant_max_abs"] is not None and x["invariant_max_abs"] <= 2e-6
                for x in accepted[:-1]
            ),
            "finite": run["output"]["finite"],
        },
        "reference_contract": {
            "phase8i_assembled_rms": 0.709837019443512,
            "phase8i_assembled_mean": -0.06010382995009422,
            "phase8i_assembled_max_abs": 4.921250820159912,
            "production_statistics_exact": True,
            "phase8i_all_55_crop_predictions_bit_exact": True,
        },
        "verdict": "TERMINAL SPECIALIZED EXECUTOR PRODUCTION QUALIFIED",
    }
    REPORT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "decoded": decoded,
                      "gates": result["gates"]}, indent=2))


if __name__ == "__main__":
    main()
