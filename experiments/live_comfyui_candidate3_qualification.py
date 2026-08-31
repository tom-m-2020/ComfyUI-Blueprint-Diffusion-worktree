"""Live ComfyUI queue qualification for the fixed Candidate-3 production slice."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import websocket
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "live_comfyui_candidate3_results"
BASE = "http://127.0.0.1:8191"
WS = "ws://127.0.0.1:8191/ws"
MODEL = r"f2k4_realrebelai_Rebels_w4a8s\Flux2-Klein-4B-w4a8.safetensors"
WRONG_MODEL = r"zit_realrebelai_Rebels_w4a8s\Z-Image-Turbo-w4a8.safetensors"
CLIP = (
    r"f2k4_Comfy-Org_vae-text-encorder-for-flux-klein-4b_text_encoders"
    r"\qwen_3_4b_fp4_flux2.safetensors"
)
VAE = r"Comfy-Org_vae-text-encorder-for-flux-klein-9b_vae\flux2-vae.safetensors"
PROMPT = (
    "A cinematic wide-angle photograph of exactly one large full-body woman "
    "standing centered in the foreground, occupying most of the image height; "
    "exactly one red vintage car parked on the far left; exactly one tall green "
    "tree on the far right; asymmetric left-center-right composition, continuous "
    "dry ground plane, coherent perspective and scale, distant low hills, sunset "
    "light, no duplicate people, no duplicate cars, no duplicate trees"
)
SEED = 20260831


def request_json(path: str, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    return None if not raw else json.loads(raw)


def base_prompt(prefix: str) -> dict:
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODEL, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "flux2", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
        "4": {"class_type": "BasicGuider", "inputs": {"model": ["1", 0], "conditioning": ["3", 0]}},
        "5": {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}},
        "6": {"class_type": "Flux2Scheduler", "inputs": {"steps": 4, "width": 1024, "height": 512}},
        "7": {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": 1024, "height": 512, "batch_size": 1}},
        "8": {"class_type": "BlueprintCandidate3EulerSampler", "inputs": {}},
        "9": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["5", 0], "guider": ["4", 0], "sampler": ["8", 0], "sigmas": ["6", 0], "latent_image": ["7", 0]}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["10", 0]}},
        "12": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


def output_hash(history: dict) -> tuple[str | None, str | None]:
    images = history.get("outputs", {}).get("12", {}).get("images", [])
    if not images:
        return None, None
    item = images[0]
    query = urllib.parse.urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    with urllib.request.urlopen(BASE + "/view?" + query, timeout=120) as response:
        data = response.read()
    with Image.open(io.BytesIO(data)) as image:
        pixel_digest = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    return pixel_digest, item["filename"]


def queue(prompt: dict, *, interrupt_after_preview: bool = False) -> dict:
    client_id = uuid.uuid4().hex
    socket = websocket.create_connection(f"{WS}?clientId={client_id}", timeout=180)
    queued = request_json("/prompt", {"prompt": prompt, "client_id": client_id})
    prompt_id = queued["prompt_id"]
    events = []
    binary_previews = 0
    interrupted = False
    started = time.perf_counter()
    try:
        while True:
            message = socket.recv()
            if isinstance(message, bytes):
                binary_previews += 1
                if interrupt_after_preview and not interrupted:
                    request_json("/interrupt", {})
                    interrupted = True
                continue
            event = json.loads(message)
            data = event.get("data", {})
            if data.get("prompt_id") not in (None, prompt_id):
                continue
            events.append(event.get("type"))
            if event.get("type") in {"execution_success", "execution_error", "execution_interrupted"}:
                break
    finally:
        socket.close()
    history_all = request_json(f"/history/{prompt_id}")
    history = history_all.get(prompt_id, {})
    status = history.get("status", {})
    messages = status.get("messages", [])
    error = None
    for kind, payload in messages:
        if kind == "execution_error":
            error = {
                "node_id": payload.get("node_id"),
                "node_type": payload.get("node_type"),
                "exception_type": payload.get("exception_type"),
                "exception_message": payload.get("exception_message"),
            }
            break
    digest, filename = output_hash(history)
    return {
        "prompt_id": prompt_id,
        "seconds": time.perf_counter() - started,
        "events": events,
        "binary_previews": binary_previews,
        "status_str": status.get("status_str"),
        "completed": status.get("completed"),
        "error": error,
        "output_sha256": digest,
        "output_filename": filename,
        "interrupted": interrupted,
    }


def invalid_prompts() -> dict[str, tuple[dict, str]]:
    cases = {}

    prompt = base_prompt("BlueprintPhase4c_invalid_resolution")
    prompt["7"]["inputs"]["width"] = 768
    cases["wrong_resolution"] = (prompt, "requires latent grid")

    prompt = base_prompt("BlueprintPhase4c_invalid_cfg")
    prompt["4"] = {"class_type": "CFGGuider", "inputs": {"model": ["1", 0], "positive": ["3", 0], "negative": ["3", 0], "cfg": 2.0}}
    cases["cfg_not_one"] = (prompt, "requires CFG exactly 1.0")

    prompt = base_prompt("BlueprintPhase4c_invalid_model")
    prompt["1"]["inputs"]["unet_name"] = WRONG_MODEL
    cases["wrong_model_family"] = (prompt, "requires native ComfyUI Flux2")

    prompt = base_prompt("BlueprintPhase4c_invalid_mask")
    prompt["13"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": 1024, "height": 512}}
    prompt["14"] = {"class_type": "SetLatentNoiseMask", "inputs": {"samples": ["7", 0], "mask": ["13", 0]}}
    prompt["9"]["inputs"]["latent_image"] = ["14", 0]
    cases["mask_inpainting"] = (prompt, "does not support masks")

    prompt = base_prompt("BlueprintPhase4c_invalid_nonempty")
    prompt["13"] = {"class_type": "AddNoise", "inputs": {"model": ["1", 0], "noise": ["5", 0], "sigmas": ["6", 0], "latent_image": ["7", 0]}}
    prompt["9"]["inputs"]["latent_image"] = ["13", 0]
    cases["nonempty_latent"] = (prompt, "supports only empty-latent T2I")

    prompt = base_prompt("BlueprintPhase4c_invalid_partial")
    prompt["13"] = {"class_type": "Flux2Scheduler", "inputs": {"steps": 5, "width": 1024, "height": 512}}
    prompt["14"] = {"class_type": "SplitSigmas", "inputs": {"sigmas": ["13", 0], "step": 1}}
    prompt["9"]["inputs"]["sigmas"] = ["14", 1]
    cases["partial_denoise"] = (prompt, "partial denoise schedules are unsupported")

    prompt = base_prompt("BlueprintPhase4c_invalid_count")
    prompt["6"]["inputs"]["steps"] = 3
    cases["wrong_sigma_count"] = (prompt, "exactly four Euler intervals")

    prompt = base_prompt("BlueprintPhase4c_invalid_schedule")
    prompt["13"] = {"class_type": "FlipSigmas", "inputs": {"sigmas": ["6", 0]}}
    prompt["9"]["inputs"]["sigmas"] = ["13", 0]
    cases["non_euler_schedule"] = (prompt, "terminate at exactly zero")

    prompt = base_prompt("BlueprintPhase4c_invalid_spatial")
    prompt["13"] = {"class_type": "ConditioningSetArea", "inputs": {"conditioning": ["3", 0], "width": 512, "height": 512, "x": 0, "y": 0, "strength": 1.0}}
    prompt["4"]["inputs"]["conditioning"] = ["13", 0]
    cases["spatial_conditioning"] = (prompt, "conditioning keys")
    return cases


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    object_info = request_json("/object_info/BlueprintCandidate3EulerSampler")["BlueprintCandidate3EulerSampler"]

    valid1 = queue(base_prompt("BlueprintPhase4c_valid_a"))
    request_json("/free", {"unload_models": True, "free_memory": True})
    time.sleep(2.0)
    valid2 = queue(base_prompt("BlueprintPhase4c_valid_b"))

    invalid = {}
    for name, (prompt, expected) in invalid_prompts().items():
        result = queue(prompt)
        message = (result.get("error") or {}).get("exception_message") or ""
        result["expected_fragment"] = expected
        result["message_matched"] = expected in message
        invalid[name] = result

    request_json("/free", {"unload_models": True, "free_memory": True})
    time.sleep(2.0)
    cancelled = queue(base_prompt("BlueprintPhase4c_cancelled"), interrupt_after_preview=True)
    post_failure_valid = queue(base_prompt("BlueprintPhase4c_post_failure_valid"))

    checks = {
        "registered": object_info["name"] == "BlueprintCandidate3EulerSampler",
        "display_name": object_info["display_name"] == "Blueprint Candidate-3 Euler Sampler",
        "normal_sampler_output": object_info["output"] == ["SAMPLER"],
        "valid_runs_completed": valid1["completed"] and valid2["completed"],
        "four_previews_each": valid1["binary_previews"] == 4 and valid2["binary_previews"] == 4,
        "decoded_outputs_saved": bool(valid1["output_sha256"] and valid2["output_sha256"]),
        "seed_reproducible": valid1["output_sha256"] == valid2["output_sha256"],
        "all_invalid_failed": all(not item["completed"] for item in invalid.values()),
        "all_invalid_messages_useful": all(item["message_matched"] for item in invalid.values()),
        "cancel_interrupted": cancelled["interrupted"] and not cancelled["completed"],
        "post_failure_valid_completed": bool(post_failure_valid["completed"]),
    }
    report = {
        "server": BASE,
        "object_info": object_info,
        "valid_first": valid1,
        "valid_after_unload": valid2,
        "invalid": invalid,
        "cancelled": cancelled,
        "post_failure_valid": post_failure_valid,
        "checks": checks,
        "qualified": all(checks.values()),
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"checks": checks, "qualified": report["qualified"], "report": str(OUTPUT / "report.json")}, indent=2))
    if not report["qualified"]:
        raise AssertionError("Live qualification failed; inspect report.json")


if __name__ == "__main__":
    main()
