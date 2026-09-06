"""CPU-only G/W/H transfer discriminator for staged Blueprint refinement."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "blueprint_staged_refinement_geometry_results"


@dataclass(frozen=True)
class Geometry:
    g_hw: tuple[int, int]
    h_hw: tuple[int, int]
    footprint_hw: tuple[int, int]
    stride_hw: tuple[int, int]
    w_hw: tuple[int, int]


def starts(length: int, size: int, stride: int) -> tuple[int, ...]:
    if not 0 < stride <= size <= length:
        raise ValueError("Require 0 < stride <= footprint <= destination.")
    values = list(range(0, length - size + 1, stride))
    if values[-1] != length - size:
        values.append(length - size)
    return tuple(values)


def footprints(geometry: Geometry) -> tuple[tuple[int, int, int, int], ...]:
    ys = starts(geometry.h_hw[0], geometry.footprint_hw[0], geometry.stride_hw[0])
    xs = starts(geometry.h_hw[1], geometry.footprint_hw[1], geometry.stride_hw[1])
    return tuple((y, x, *geometry.footprint_hw) for y in ys for x in xs)


def full_transfer(g: torch.Tensor, geometry: Geometry, rect: tuple[int, int, int, int]) -> torch.Tensor:
    mapped = F.interpolate(g, size=geometry.h_hw, mode="bilinear", align_corners=False)
    y, x, height, width = rect
    footprint = mapped[:, :, y:y + height, x:x + width]
    return F.interpolate(footprint, size=geometry.w_hw, mode="bilinear", align_corners=False)


def _axis_coordinates(
    start: int, count: int, source: int, destination: int, *, device: torch.device
) -> torch.Tensor:
    output = torch.arange(start, start + count, dtype=torch.float64, device=device)
    return ((output + 0.5) * source / destination - 0.5).clamp(0, source - 1)


def bounded_transfer(
    g: torch.Tensor, geometry: Geometry, rect: tuple[int, int, int, int]
) -> tuple[torch.Tensor, tuple[int, int]]:
    y, x, height, width = rect
    gy = _axis_coordinates(y, height, geometry.g_hw[0], geometry.h_hw[0], device=g.device)
    gx = _axis_coordinates(x, width, geometry.g_hw[1], geometry.h_hw[1], device=g.device)
    y0, y1 = int(torch.floor(gy.min())), int(torch.ceil(gy.max()))
    x0, x1 = int(torch.floor(gx.min())), int(torch.ceil(gx.max()))
    source = g[:, :, y0:y1 + 1, x0:x1 + 1]
    if source.shape[-2] < 2 or source.shape[-1] < 2:
        raise ValueError("The bounded bilinear source must contain at least 2x2 cells.")
    ny = 2.0 * (gy - y0) / (source.shape[-2] - 1) - 1.0
    nx = 2.0 * (gx - x0) / (source.shape[-1] - 1) - 1.0
    grid_y, grid_x = torch.meshgrid(ny, nx, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)[None].to(dtype=g.dtype)
    footprint = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)
    working = F.interpolate(footprint, size=geometry.w_hw, mode="bilinear", align_corners=False)
    return working, (source.shape[-2], source.shape[-1])


def feather(height: int, width: int) -> torch.Tensor:
    def axis(length: int) -> torch.Tensor:
        coordinate = torch.arange(length, dtype=torch.float64)
        return torch.minimum(coordinate + 1, torch.tensor(length, dtype=torch.float64) - coordinate)
    return axis(height)[:, None] * axis(width)[None, :]


def evaluate(geometry: Geometry, seed: int) -> dict:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    g = torch.randn((1, 8, *geometry.g_hw), generator=generator, dtype=torch.float64)
    rects = footprints(geometry)
    coverage = torch.zeros((1, 1, *geometry.h_hw), dtype=torch.float64)
    weighted = torch.zeros((1, 8, *geometry.h_hw), dtype=torch.float64)
    max_error = 0.0
    max_source_elements = 0
    source_shapes = []
    for rect in rects:
        expected = full_transfer(g, geometry, rect)
        actual, source_hw = bounded_transfer(g, geometry, rect)
        max_error = max(max_error, float((expected - actual).abs().max()))
        max_source_elements = max(max_source_elements, math.prod(source_hw) * g.shape[1])
        source_shapes.append(source_hw)
        restricted = F.interpolate(actual, size=geometry.footprint_hw, mode="bilinear", align_corners=False)
        y, x, height, width = rect
        weight = feather(height, width)[None, None]
        weighted[:, :, y:y + height, x:x + width] += restricted * weight
        coverage[:, :, y:y + height, x:x + width] += weight
    assembled = weighted / coverage
    if rects != footprints(geometry):
        raise RuntimeError("Footprint planning is nondeterministic.")
    return {
        "geometry": geometry.__dict__,
        "region_count": len(rects),
        "first_rect": rects[0],
        "last_rect": rects[-1],
        "max_transfer_abs_error": max_error,
        "coverage_min": float(coverage.min()),
        "coverage_max": float(coverage.max()),
        "finite_assembly": bool(torch.isfinite(assembled).all()),
        "max_bounded_source_elements": max_source_elements,
        "full_mapped_anchor_elements": g.shape[1] * math.prod(geometry.h_hw),
        "unique_bounded_source_hw": sorted(set(source_shapes)),
    }


def main() -> None:
    cases = (
        Geometry((45, 45), (128, 128), (32, 32), (24, 24), (64, 64)),
        Geometry((64, 32), (256, 128), (48, 40), (32, 28), (64, 64)),
        Geometry((36, 54), (128, 192), (40, 48), (28, 32), (64, 64)),
        Geometry((37, 61), (131, 227), (41, 53), (29, 37), (64, 64)),
        Geometry((48, 80), (257, 513), (48, 64), (32, 40), (64, 64)),
    )
    results = [evaluate(case, 7001 + index) for index, case in enumerate(cases)]
    report = {
        "experiment": "blueprint_staged_refinement_geometry",
        "device": "cpu",
        "model_calls": 0,
        "destination_sized_model_calls": 0,
        "passed": all(
            item["max_transfer_abs_error"] <= 1e-5
            and item["coverage_min"] > 0
            and item["finite_assembly"]
            for item in results
        ),
        "cases": results,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("Staged-refinement geometry discriminator failed.")


if __name__ == "__main__":
    main()
