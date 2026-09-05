"""Phase 39 source/control audit. Intentionally performs no diffusion inference."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
OUTPUT = ROOT / "experiments" / "flux2_blueprint_as_guide_results"
CONTROL_ROOT = ROOT / "experiments" / "flux2_terminal_resampling_refinement_strength_results" / "SQUARE_MULTI_OBJECT"
EXPECTED_CONTROL = "1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3"
EXPECTED_MAPPED = "8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05"


def tensor_hash(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main() -> None:
    model_base = COMFY_ROOT / "comfy" / "model_base.py"
    flux_model = COMFY_ROOT / "comfy" / "ldm" / "flux" / "model.py"
    flux_layers = COMFY_ROOT / "comfy" / "ldm" / "flux" / "layers.py"
    samplers = COMFY_ROOT / "comfy" / "samplers.py"
    requirements = {
        model_base: ("reference_latents", "CONDList(latents)"),
        flux_model: ("ref_latents_method", "reference_image_num_tokens", 'patches_replace.get("dit", {})'),
        flux_layers: ('"attn1_patch"', '"attn1_output_patch"'),
        samplers: ("sampler_post_cfg_function", "model_function_wrapper"),
    }
    for path, markers in requirements.items():
        source = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker not in source:
                raise RuntimeError(f"Missing expected source marker {marker!r} in {path}")

    blueprint = torch.load(CONTROL_ROOT / "blueprint.pt", map_location="cpu", weights_only=True)
    blueprint_record = json.loads((CONTROL_ROOT / "blueprint.json").read_text(encoding="utf-8"))
    control = torch.load(CONTROL_ROOT / "sigma_0.25.pt", map_location="cpu", weights_only=True)
    control_record = json.loads((CONTROL_ROOT / "sigma_0.25.json").read_text(encoding="utf-8"))
    if tensor_hash(blueprint["mapped"]) != EXPECTED_MAPPED or blueprint_record["mapped_hash"] != EXPECTED_MAPPED:
        raise RuntimeError("Mapped Blueprint fingerprint mismatch")
    if tensor_hash(control) != EXPECTED_CONTROL or control_record["final_hash"] != EXPECTED_CONTROL:
        raise RuntimeError("Qualified control fingerprint mismatch")
    if not control_record["phase28_regression"]["bit_exact"]:
        raise RuntimeError("Control is not Phase-28/29 qualified")

    mechanisms = [
        {
            "name": "prediction_or_denoised_space_guidance", "model_mediated": False,
            "same_canvas": True, "fixed_contract": False, "requires_backend_change": False,
            "classification": "reject: needs a loss/likelihood definition and free guidance scale; Phase 30 boundary",
        },
        {
            "name": "latent_or_state_constraint", "model_mediated": False,
            "same_canvas": True, "fixed_contract": False, "requires_backend_change": False,
            "classification": "reject: hard replacement/project is already tested; soft constraint needs a free strength and collapses toward resampling/projection",
        },
        {
            "name": "transformer_feature_or_residual_guidance", "model_mediated": True,
            "same_canvas": "mechanically possible", "fixed_contract": False, "requires_backend_change": True,
            "classification": "reject: hooks are engineering sinks, not trained Klein Blueprint conditioning; mapping and amplitude are undefined",
        },
        {
            "name": "same_canvas_attention_or_kv_guidance", "model_mediated": True,
            "same_canvas": "possible only through custom token/KV execution", "fixed_contract": False,
            "requires_backend_change": True,
            "classification": "reject: no native Blueprint role/mask; token identity, routing, normalization and coordinate contract are untrained or underdetermined; prior external-KV family already failed normalized-W coherence",
        },
        {
            "name": "native_reference_latents", "model_mediated": True,
            "same_canvas": False, "fixed_contract": True, "requires_backend_change": False,
            "classification": "reject: trained reference-image context uses a distinct image index/canvas, not same-canvas Blueprint authority",
        },
        {
            "name": "controlnet_or_adapter", "model_mediated": True,
            "same_canvas": True, "fixed_contract": False, "requires_backend_change": True,
            "classification": "reject: base Klein 4B has residual sinks but no qualified trained same-canvas ControlNet/adapter",
        },
    ]
    report = {
        "phase": 39,
        "status": "stopped_before_inference",
        "verdict": "E — ARCHITECTURAL BOUNDARY",
        "primary_answer": "No fixed native same-canvas contract lets an independent Klein local state consume the Blueprint as an authoritative guide under the allowed interface.",
        "control": {
            "case": "SQUARE_MULTI_OBJECT", "mapped_blueprint_hash": EXPECTED_MAPPED,
            "qualified_control_hash": EXPECTED_CONTROL, "phase28_regression_bit_exact": True,
            "semantic_grade": "S3 (persisted qualified evidence)", "model_work_reused": True,
        },
        "mechanisms": mechanisms,
        "selection": {"executable_mechanism": None, "reason": "Every same-canvas candidate needs an empirical strength/metric or untrained model/backend intervention; the sole trained native image condition is separate-canvas reference context."},
        "source_audit": [{"path": str(path), "sha256": file_hash(path), "markers": markers} for path, markers in requirements.items()],
        "integrity": {
            "diffusion_model_forwards": 0, "local_forwards": 0,
            "destination_sized_forwards": 0, "decoded_images": 0,
            "production_changed": False, "comfyui_core_changed": False,
        },
        "next_discriminator": "One explicitly authorized learned/qualified same-canvas Klein control-adapter feasibility audit: determine whether any released adapter supplies spatial Blueprint residuals before or through the native blocks; do not run inference unless a checkpoint-trained interface exists.",
    }
    atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({"report": str(OUTPUT / "report.json"), "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
    main()
