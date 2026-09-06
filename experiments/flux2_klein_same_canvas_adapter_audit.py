"""Phase 40 released-adapter audit. Performs no diffusion inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "flux2_klein_same_canvas_adapter_audit_results"
CONTROL_ROOT = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
PHASE39 = ROOT / "experiments" / "flux2_blueprint_as_guide_results" / "report.json"
EXPECTED_CONTROL = "1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3"
EXPECTED_MAPPED = "8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05"


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    phase39 = json.loads(PHASE39.read_text(encoding="utf-8"))
    if phase39["verdict"] != "E — ARCHITECTURAL BOUNDARY":
        raise RuntimeError("Phase-39 boundary changed")
    if phase39["control"]["mapped_blueprint_hash"] != EXPECTED_MAPPED:
        raise RuntimeError("Phase-39 mapped Blueprint fingerprint changed")
    if phase39["control"]["qualified_control_hash"] != EXPECTED_CONTROL:
        raise RuntimeError("Phase-39 control fingerprint changed")

    blueprint = torch.load(CONTROL_ROOT / "blueprint.pt", map_location="cpu", weights_only=True)
    blueprint_record = json.loads((CONTROL_ROOT / "blueprint.json").read_text(encoding="utf-8"))
    control = torch.load(CONTROL_ROOT / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    control_record = json.loads((CONTROL_ROOT / "sigma_0.25.json").read_text(encoding="utf-8"))
    if tensor_hash(blueprint["mapped"]) != EXPECTED_MAPPED or blueprint_record["mapped_hash"] != EXPECTED_MAPPED:
        raise RuntimeError("Persisted mapped Blueprint fingerprint mismatch")
    if tensor_hash(control) != EXPECTED_CONTROL or control_record["final_hash"] != EXPECTED_CONTROL:
        raise RuntimeError("Persisted qualified control fingerprint mismatch")
    if not control_record["phase28_regression"]["bit_exact"]:
        raise RuntimeError("Persisted control is not qualified")

    candidates = [
        {
            "name": "BFL official FLUX.2 Klein 4B releases",
            "identifier": "black-forest-labs/FLUX.2-klein-4B and FLUX.2-klein-base-4B",
            "url": "https://github.com/black-forest-labs/flux2",
            "base_architecture": "FLUX.2 Klein 4B",
            "trained_checkpoint": True,
            "conditioning": "text plus one or more separately indexed reference images",
            "spatially_aligned_same_canvas": False,
            "injection": "reference image tokens in native joint transformer sequence",
            "preprocessing": "official VAE/reference-token path",
            "geometry": "each reference is its own image/canvas",
            "canonical_strength": "native edit contract; no same-canvas control strength",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "yes for ordinary T2I/edit references",
            "extra_cost": "reference encoding and extra attention keys/tokens",
            "classification": "separate-canvas; rejected by Phase-40 gate",
        },
        {
            "name": "RefControl FLUX.2 Klein 4B reference-depth LoRA",
            "identifier": "thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora",
            "url": "https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora",
            "base_architecture": "FLUX.2 Klein Base 4B LoRA",
            "trained_checkpoint": True,
            "conditioning": "image 1 depth control plus image 2 identity/reference through Klein edit inputs",
            "spatially_aligned_same_canvas": False,
            "injection": "LoRA-modified DiT consuming separate reference-image tokens",
            "preprocessing": "depth estimator/control image and reference image; trigger phrase refcontrol",
            "geometry": "separate images; control-to-output structural intent is learned but not a same-canvas latent authority field",
            "canonical_strength": "recommended LoRA weight 0.8–1.0, therefore still a user range",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "LoRA keys may load, but the contract requires Klein edit references rather than the qualified BasicGuider T2I path",
            "extra_cost": "extra reference-token attention; no separate ControlNet branch",
            "classification": "architecture-matched but separate-canvas and free-weight; rejected",
        },
        {
            "name": "RefControl FLUX.2 Klein 4B pose LoRA",
            "identifier": "xocialize/refcontrol-FLUX.2-klein-4B-pose-lora",
            "url": "https://huggingface.co/xocialize/refcontrol-FLUX.2-klein-4B-pose-lora",
            "base_architecture": "FLUX.2 Klein Base 4B LoRA",
            "trained_checkpoint": True,
            "conditioning": "image 1 OpenPose skeleton plus image 2 identity/reference",
            "spatially_aligned_same_canvas": False,
            "injection": "rank-32 LoRA on DiT; two controls use Klein edit/reference path",
            "preprocessing": "COCO-18 colored skeleton on black plus plain-background reference; prompt trigger",
            "geometry": "separate input images, not an aligned Blueprint latent plane",
            "canonical_strength": "repository catalog recommends LoRA weight 0.8–1.0",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "requires edit references and a Base-trained LoRA contract",
            "extra_cost": "extra reference-token attention; LoRA matmuls",
            "classification": "architecture-matched but separate-canvas and task-specific; rejected",
        },
        {
            "name": "FLUX.2-dev Fun ControlNet Union",
            "identifier": "alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
            "url": "https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union",
            "base_architecture": "FLUX.2-dev 32B, not Klein 4B",
            "trained_checkpoint": True,
            "conditioning": "Canny, HED, depth, pose, MLSD, scribble, gray, inpaint controls",
            "spatially_aligned_same_canvas": True,
            "injection": "ControlNet residuals on four double blocks",
            "preprocessing": "control-type-specific images/preprocessors",
            "geometry": "full-frame control image",
            "canonical_strength": "documented recommended range 0.65–0.80, not fixed",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "incompatible checkpoint architecture/profile",
            "extra_cost": "additional ControlNet execution and block residual storage",
            "classification": "same-canvas but FLUX.2-dev-only; rejected",
        },
        {
            "name": "ReyChiaro FLUX.2 Klein ControlNet",
            "identifier": "ReyChiaro/flux.2-klein-controlnet",
            "url": "https://github.com/ReyChiaro/flux.2-klein-controlnet",
            "base_architecture": "FLUX.2 Klein 9B",
            "trained_checkpoint": False,
            "conditioning": "proposed control tensor with extra condition channels",
            "spatially_aligned_same_canvas": True,
            "injection": "control embedding plus zero-initialized double/single-block residual projections",
            "preprocessing": "training/inference code present; no released qualified 4B checkpoint contract found",
            "geometry": "packed control aligned to generated image tokens",
            "canonical_strength": "conditioning_scale argument defaults to 1.0 in code, but no qualified 4B weights",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "Diffusers research code for 9B; not native Comfy Klein 4B",
            "extra_cost": "additional transformer/control branch and residuals",
            "classification": "code-only/incompatible 9B target; rejected",
        },
        {
            "name": "FLUX.1 ControlNet / IP-Adapter families",
            "identifier": "InstantX/FLUX.1-dev-Controlnet-Union, XLabs-AI FLUX.1 adapters, related releases",
            "url": "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union",
            "base_architecture": "FLUX.1 dev, not FLUX.2 Klein 4B",
            "trained_checkpoint": True,
            "conditioning": "control images or adapter tokens depending on release",
            "spatially_aligned_same_canvas": "ControlNet variants yes; IP/reference variants no",
            "injection": "FLUX.1-specific residual or attention adapters",
            "preprocessing": "release-specific control preprocessors",
            "geometry": "release-specific",
            "canonical_strength": "user-selected in common integrations",
            "blueprint_compatible_without_reinterpretation": False,
            "native_comfy_compatibility": "architecture-incompatible",
            "extra_cost": "adapter/control branch dependent",
            "classification": "incompatible family; rejected",
        },
    ]
    qualified = [candidate["identifier"] for candidate in candidates if candidate["classification"] == "qualified"]
    if qualified:
        raise RuntimeError(f"Static audit unexpectedly marks candidates qualified: {qualified}")

    report = {
        "phase": 40,
        "audit_date": "2026-09-06",
        "status": "complete_no_inference",
        "verdict": "C — ONLY INCOMPATIBLE / SEPARATE-CANVAS ADAPTERS FOUND",
        "scope": "Released official/community resources discovered in the bounded search; absence is not a proof that no private or future checkpoint exists.",
        "control": {
            "case": "SQUARE_MULTI_OBJECT",
            "mapped_blueprint_hash": EXPECTED_MAPPED,
            "qualified_control_hash": EXPECTED_CONTROL,
            "phase28_regression_bit_exact": True,
            "phase39_boundary_validated": True,
        },
        "qualification_gates": {
            "trained_checkpoint": True,
            "flux2_klein_4b_compatible": True,
            "same_canvas_spatial_alignment": True,
            "exact_preprocessing_recoverable": True,
            "known_injection": True,
            "no_blueprint_projection_invented": True,
            "no_arbitrary_guidance_metric": True,
            "not_reference_latents_reinterpretation": True,
            "canonical_or_inherent_coupling": True,
        },
        "candidates": candidates,
        "qualified_candidates": qualified,
        "conclusion": "Released architecture-matched 4B controls found use Klein's separately indexed edit/reference inputs and optional LoRA weighting. Released true same-canvas ControlNets target FLUX.2-dev or incompatible architectures. No candidate satisfies all gates.",
        "execution": {
            "diffusion_model_forwards": 0,
            "local_forwards": 0,
            "destination_sized_forwards": 0,
            "decoded_images": 0,
            "production_changed": False,
            "comfyui_core_changed": False,
        },
        "next_phase": "Separate interleaved global↔local resampling architecture design discriminator, with explicit state ownership and feedback equations before inference.",
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
