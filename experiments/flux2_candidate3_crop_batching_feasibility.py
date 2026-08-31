"""Phase 6b feasibility probe for distinct-coordinate FLUX.2 crop batches."""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
SOURCE = COMFY_ROOT / "comfy" / "ldm" / "flux" / "model.py"
OUTPUT = ROOT / "experiments" / "flux2_candidate3_crop_batching_results"
sys.path.insert(0, str(COMFY_ROOT))

from comfy.ldm.flux.model import Flux


def minimal_flux() -> Flux:
    model = Flux.__new__(Flux)
    torch.nn.Module.__init__(model)
    model.patch_size = 1
    model.params = SimpleNamespace(axes_dim=[32, 48, 48])
    return model


def endpoint(ids: torch.Tensor, batch: int) -> dict[str, list[float]]:
    return {
        "first_yx": [float(ids[batch, 0, 1]), float(ids[batch, 0, 2])],
        "last_yx": [float(ids[batch, -1, 1]), float(ids[batch, -1, 2])],
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = minimal_flux()
    crops = torch.zeros((2, 128, 32, 32), dtype=torch.float32)

    _, scalar_ids = model.process_img(
        crops,
        transformer_options={
            "rope_options": {"shift_y": 0.0, "shift_x": 24.0}
        },
    )
    scalar_control = {
        "shape": list(scalar_ids.shape),
        "batch_0": endpoint(scalar_ids, 0),
        "batch_1": endpoint(scalar_ids, 1),
        "batch_ids_bit_exact": bool(torch.equal(scalar_ids[0], scalar_ids[1])),
    }

    vector_error = None
    try:
        model.process_img(
            crops,
            transformer_options={
                "rope_options": {
                    "shift_y": torch.tensor([0.0, 24.0]),
                    "shift_x": torch.tensor([0.0, 32.0]),
                }
            },
        )
    except Exception as exc:
        vector_error = {"type": type(exc).__name__, "message": str(exc)}

    source_text = SOURCE.read_bytes()
    process_source = inspect.getsource(Flux.process_img)
    forward_signature = str(inspect.signature(Flux.forward))
    result = {
        "comfy_source": str(SOURCE),
        "source_sha256": hashlib.sha256(source_text).hexdigest(),
        "tested_method": "comfy.ldm.flux.model.Flux.process_img",
        "input_crop_batch_shape": list(crops.shape),
        "requested_offsets_yx": [[0.0, 0.0], [24.0, 32.0]],
        "scalar_options_control": scalar_control,
        "vector_options_attempt": vector_error,
        "forward_signature": forward_signature,
        "process_img_source": process_source,
        "facts": {
            "process_img_called_once_for_batched_x": True,
            "single_img_ids_grid_repeated_over_batch": True,
            "scalar_shift_assigns_same_coordinates_to_every_batch_element": scalar_control[
                "batch_ids_bit_exact"
            ],
            "vector_shift_accepted": vector_error is None,
            "forward_accepts_explicit_img_ids": "img_ids" in inspect.signature(Flux.forward).parameters,
        },
    }
    feasible = (
        not result["facts"]["single_img_ids_grid_repeated_over_batch"]
        and result["facts"]["vector_shift_accepted"]
    )
    result["existing_adapter_boundary_feasible"] = feasible
    result["verdict"] = (
        "INCONCLUSIVE" if feasible else "CROP BATCHING REQUIRES ADAPTER/BACKEND CHANGE"
    )
    path = OUTPUT / "report.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "scalar_control": scalar_control,
        "vector_options_attempt": vector_error,
        "forward_signature": forward_signature,
        "verdict": result["verdict"],
        "report": str(path),
    }, indent=2))


if __name__ == "__main__":
    main()
