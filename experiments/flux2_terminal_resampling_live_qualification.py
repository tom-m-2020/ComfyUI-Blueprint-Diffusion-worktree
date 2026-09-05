"""Phase 27 live dedicated-node qualification and repeatability check."""
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
sys.path.insert(0, str(COMFY_ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
SPEC = importlib.util.spec_from_file_location("blueprint_diffusion", PACKAGE / "__init__.py", submodule_search_locations=[str(PACKAGE)])
MODULE = importlib.util.module_from_spec(SPEC); sys.modules["blueprint_diffusion"] = MODULE; SPEC.loader.exec_module(MODULE)

import comfy.model_management
import flux2_coarse_global_local_falsification as phase2
from comfy_extras.nodes_custom_sampler import Guider_Basic
from blueprint_diffusion.nodes import BlueprintTerminalResampling
from blueprint_diffusion.terminal_resampling import QUALIFIED_SIGMAS, tensor_hash

OUTPUT = ROOT / "experiments" / "terminal_resampling_live_qualification_results"
REPORT = OUTPUT / "report.json"
SEED = 20260911
PROMPT = (
    "A cinematic photograph of one red vintage car on the far left, one tall green tree in the center, "
    "and one small white house on the far right, all standing on the same grassy field under one continuous "
    "horizon, coherent perspective, exactly one car, one tree, and one house, no duplicate objects."
)


def rgb_hash(path):
    with Image.open(path) as image:
        return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip([str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2)
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT)); del clip
    destination = {"samples": torch.zeros((1, 128, 128, 256), dtype=torch.float32)}
    sigmas = torch.tensor(QUALIFIED_SIGMAS)
    runs = []
    node = BlueprintTerminalResampling()
    for ordinal in range(2):
        guider = Guider_Basic(model); guider.set_conds(positive)
        started = time.perf_counter()
        output, denoised = node.sample(guider, sigmas, SEED, destination)
        elapsed = time.perf_counter() - started
        telemetry = output["blueprint_terminal_resampling_telemetry"]
        runs.append({
            "ordinal": ordinal,
            "latent_hash": tensor_hash(output["samples"]),
            "denoised_hash": tensor_hash(denoised["samples"]),
            "wall_seconds": elapsed,
            "telemetry": telemetry,
            "finite": bool(torch.isfinite(output["samples"]).all()),
        })
        torch.save(output["samples"].cpu(), OUTPUT / f"run_{ordinal}_latent.pt")

    vae = phase2.comfy.sd.VAE(sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True))
    decoded = []
    for ordinal in range(2):
        latent = torch.load(OUTPUT / f"run_{ordinal}_latent.pt", map_location="cpu", weights_only=True)
        path = OUTPUT / f"run_{ordinal}.png"
        phase2.save_pixels(vae.decode(latent).cpu(), path)
        decoded.append({"path": str(path), "sha256_rgb": rgb_hash(path)})

    workflow = {
        "last_node_id": 10,
        "last_link_id": 9,
        "nodes": [
            {"id": 1, "type": "UNETLoader", "pos": [0, 0], "size": [340, 82], "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": [{"name": "MODEL", "type": "MODEL", "links": [1], "slot_index": 0}], "properties": {"Node name for S&R": "UNETLoader"}, "widgets_values": ["f2k4_realrebelai_Rebels_w4a8s\\Flux2-Klein-4B-w4a8.safetensors", "default"]},
            {"id": 2, "type": "CLIPLoader", "pos": [0, 120], "size": [340, 106], "flags": {}, "order": 1, "mode": 0, "inputs": [], "outputs": [{"name": "CLIP", "type": "CLIP", "links": [2], "slot_index": 0}], "properties": {"Node name for S&R": "CLIPLoader"}, "widgets_values": ["f2k4_Comfy-Org_vae-text-encorder-for-flux-klein-4b_text_encoders\\qwen_3_4b_fp4_flux2.safetensors", "flux2", "default"]},
            {"id": 3, "type": "CLIPTextEncode", "pos": [380, 120], "size": [420, 210], "flags": {}, "order": 2, "mode": 0, "inputs": [{"name": "clip", "type": "CLIP", "link": 2}], "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [3], "slot_index": 0}], "properties": {"Node name for S&R": "CLIPTextEncode"}, "widgets_values": [PROMPT]},
            {"id": 4, "type": "BasicGuider", "pos": [850, 80], "size": [240, 80], "flags": {}, "order": 3, "mode": 0, "inputs": [{"name": "model", "type": "MODEL", "link": 1}, {"name": "conditioning", "type": "CONDITIONING", "link": 3}], "outputs": [{"name": "GUIDER", "type": "GUIDER", "links": [4], "slot_index": 0}], "properties": {"Node name for S&R": "BasicGuider"}, "widgets_values": []},
            {"id": 5, "type": "ManualSigmas", "pos": [380, 390], "size": [420, 58], "flags": {}, "order": 4, "mode": 0, "inputs": [], "outputs": [{"name": "SIGMAS", "type": "SIGMAS", "links": [5], "slot_index": 0}], "properties": {"Node name for S&R": "ManualSigmas"}, "widgets_values": ["1.0, 0.9991771578788757, 0.9975355267524719, 0.9926428198814392, 0.0"]},
            {"id": 6, "type": "EmptyFlux2LatentImage", "pos": [380, 490], "size": [300, 106], "flags": {}, "order": 5, "mode": 0, "inputs": [], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [6], "slot_index": 0}], "properties": {"Node name for S&R": "EmptyFlux2LatentImage"}, "widgets_values": [4096, 2048, 1]},
            {"id": 7, "type": "BlueprintTerminalResampling", "pos": [900, 300], "size": [370, 180], "flags": {}, "order": 6, "mode": 0, "inputs": [{"name": "guider", "type": "GUIDER", "link": 4}, {"name": "sigmas", "type": "SIGMAS", "link": 5}, {"name": "destination", "type": "LATENT", "link": 6}], "outputs": [{"name": "output", "type": "LATENT", "links": [7], "slot_index": 0}, {"name": "denoised_output", "type": "LATENT", "links": [], "slot_index": 1}], "properties": {"Node name for S&R": "BlueprintTerminalResampling"}, "widgets_values": [SEED, "fixed"]},
            {"id": 8, "type": "VAELoader", "pos": [900, 40], "size": [340, 60], "flags": {}, "order": 7, "mode": 0, "inputs": [], "outputs": [{"name": "VAE", "type": "VAE", "links": [8], "slot_index": 0}], "properties": {"Node name for S&R": "VAELoader"}, "widgets_values": ["Comfy-Org_vae-text-encorder-for-flux-klein-9b_vae\\flux2-vae.safetensors"]},
            {"id": 9, "type": "VAEDecode", "pos": [1320, 280], "size": [240, 80], "flags": {}, "order": 8, "mode": 0, "inputs": [{"name": "samples", "type": "LATENT", "link": 7}, {"name": "vae", "type": "VAE", "link": 8}], "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [9], "slot_index": 0}], "properties": {"Node name for S&R": "VAEDecode"}, "widgets_values": []},
            {"id": 10, "type": "SaveImage", "pos": [1620, 260], "size": [320, 270], "flags": {}, "order": 9, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 9}], "outputs": [{"name": "images", "type": "IMAGE", "links": [], "slot_index": 0}], "properties": {"Node name for S&R": "SaveImage"}, "widgets_values": ["BlueprintTerminalResampling/Phase27"]},
        ],
        "links": [[1,1,0,4,0,"MODEL"],[2,2,0,3,0,"CLIP"],[3,3,0,4,1,"CONDITIONING"],[4,4,0,7,0,"GUIDER"],[5,5,0,7,1,"SIGMAS"],[6,6,0,7,3,"LATENT"],[7,7,0,9,0,"LATENT"],[8,8,0,9,1,"VAE"],[9,9,0,10,0,"IMAGE"]],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.75, "offset": [20, 20]}},
        "version": 0.4,
    }
    workflow_path = OUTPUT / "blueprint_terminal_resampling_workflow.json"
    workflow_path.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
    report = {
        "phase": 27,
        "node": "BlueprintTerminalResampling",
        "display_name": "Blueprint Terminal Resampling",
        "runs": runs,
        "decoded": decoded,
        "repeat_latent_exact": runs[0]["latent_hash"] == runs[1]["latent_hash"],
        "repeat_decoded_exact": decoded[0]["sha256_rgb"] == decoded[1]["sha256_rgb"],
        "normal_guider_sample_path": True,
        "preview_events_expected": 5,
        "normal_vae_decode_save": True,
        "workflow": str(workflow_path),
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "repeat_latent_exact": report["repeat_latent_exact"], "repeat_decoded_exact": report["repeat_decoded_exact"]}, indent=2))
    comfy.model_management.unload_all_models(); comfy.model_management.soft_empty_cache()


if __name__ == "__main__":
    main()
