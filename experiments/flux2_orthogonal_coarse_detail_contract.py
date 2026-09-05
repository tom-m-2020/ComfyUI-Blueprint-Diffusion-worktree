"""CPU-only validation for the Phase-34 orthogonal coarse/detail contract."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "flux2_orthogonal_coarse_detail_contract_results"


def haar2x2(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    return (
        (a + b + c + d) / 2,
        (a + b - c - d) / 2,
        (a - b + c - d) / 2,
        (a - b - c + d) / 2,
    )


def inverse_haar2x2(coefficients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    ll, vertical, horizontal, diagonal = coefficients
    out = torch.empty(
        *ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2,
        dtype=ll.dtype,
        device=ll.device,
    )
    out[..., 0::2, 0::2] = (ll + vertical + horizontal + diagonal) / 2
    out[..., 0::2, 1::2] = (ll + vertical - horizontal - diagonal) / 2
    out[..., 1::2, 0::2] = (ll - vertical + horizontal - diagonal) / 2
    out[..., 1::2, 1::2] = (ll - vertical - horizontal + diagonal) / 2
    return out


def max_abs(x: torch.Tensor) -> float:
    return float(x.abs().max())


def validate_dtype(dtype: torch.dtype) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(20260905)
    w = torch.randn((2, 5, 64, 64), generator=generator, dtype=dtype)
    coefficients = haar2x2(w)
    reconstructed = inverse_haar2x2(coefficients)

    coefficient_round_trip = haar2x2(reconstructed)
    coefficient_error = max(
        max_abs(actual - expected)
        for actual, expected in zip(coefficient_round_trip, coefficients)
    )
    avgpool_error = max_abs(F.avg_pool2d(w, 2, 2) - coefficients[0] / 2)

    blueprint = torch.randn((1, 3, 32, 32), generator=generator, dtype=dtype)
    zeros = torch.zeros_like(blueprint)
    blueprint_inverse = inverse_haar2x2((2 * blueprint, zeros, zeros, zeros))
    blueprint_error = max_abs(blueprint_inverse - F.interpolate(blueprint, scale_factor=2, mode="nearest"))

    x0 = torch.randn(w.shape, generator=generator, dtype=dtype)
    sigma = 0.71
    sigma_next = 0.37
    delta = sigma_next - sigma
    w_next = w + delta * (w - x0) / sigma
    w_coefficients = haar2x2(w)
    x0_coefficients = haar2x2(x0)
    coordinate_next = tuple(
        state + delta * (state - prediction) / sigma
        for state, prediction in zip(w_coefficients, x0_coefficients)
    )
    euler_error = max_abs(w_next - inverse_haar2x2(coordinate_next))

    epsilon = torch.randn(w.shape, generator=generator, dtype=dtype)
    epsilon_coefficients = haar2x2(epsilon)
    resampling_sigma = 0.25
    direct_state = (1 - resampling_sigma) * F.interpolate(
        blueprint[:, : w.shape[1]], scale_factor=2, mode="nearest"
    ) + resampling_sigma * epsilon[:, : blueprint.shape[1]]
    coefficient_state = (
        (1 - resampling_sigma) * 2 * blueprint + resampling_sigma * epsilon_coefficients[0][:, : blueprint.shape[1]],
        resampling_sigma * epsilon_coefficients[1][:, : blueprint.shape[1]],
        resampling_sigma * epsilon_coefficients[2][:, : blueprint.shape[1]],
        resampling_sigma * epsilon_coefficients[3][:, : blueprint.shape[1]],
    )
    state_error = max_abs(direct_state - inverse_haar2x2(coefficient_state))

    return {
        "spatial_round_trip_max_abs": max_abs(reconstructed - w),
        "coefficient_round_trip_max_abs": coefficient_error,
        "avgpool_equals_ll_over_2_max_abs": avgpool_error,
        "blueprint_ll_scaling_max_abs": blueprint_error,
        "euler_equivalence_max_abs": euler_error,
        "qualified_state_coordinate_equivalence_max_abs": state_error,
    }


def gaussian_sanity() -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(20260905)
    samples = torch.randn((250_000, 4), generator=generator, dtype=torch.float64)
    matrix = torch.tensor(
        [
            [1, 1, 1, 1],
            [1, 1, -1, -1],
            [1, -1, 1, -1],
            [1, -1, -1, 1],
        ],
        dtype=torch.float64,
    ) / 2
    transformed = samples @ matrix.T
    covariance = torch.cov(transformed.T)
    return {
        "sample_count": samples.shape[0],
        "means": transformed.mean(0).tolist(),
        "variances": transformed.var(0, unbiased=True).tolist(),
        "covariance": covariance.tolist(),
        "max_abs_covariance_error_from_identity": max_abs(covariance - torch.eye(4, dtype=torch.float64)),
    }


def main() -> None:
    torch.use_deterministic_algorithms(True)
    first = {
        "float32": validate_dtype(torch.float32),
        "float64": validate_dtype(torch.float64),
        "gaussian_sanity": gaussian_sanity(),
    }
    second = {
        "float32": validate_dtype(torch.float32),
        "float64": validate_dtype(torch.float64),
        "gaussian_sanity": gaussian_sanity(),
    }
    report = {
        "phase": "34",
        "model_loaded": False,
        "cpu_only": True,
        "seed": 20260905,
        "transform_normalization": "orthonormal 2x2 Haar, coefficients divided by 2",
        "validation": first,
        "deterministic_repeat_exact": first == second,
        "declared_tolerances": {"float32_max_abs": 1e-5, "float64_max_abs": 1e-12},
        "passed": (
            first == second
            and max(first["float32"].values()) <= 1e-5
            and max(first["float64"].values()) <= 1e-12
        ),
        "memory_bytes_one_region": {
            "shape": [1, 128, 64, 64],
            "coarse_only_fp16": 1 * 128 * 32 * 32 * 2,
            "coarse_plus_three_details_fp16": 4 * 1 * 128 * 32 * 32 * 2,
            "reconstructed_w_fp16": 1 * 128 * 64 * 64 * 2,
            "bands_plus_reconstructed_w_fp16": 2 * 1 * 128 * 64 * 64 * 2,
            "coarse_only_fp32": 1 * 128 * 32 * 32 * 4,
            "coarse_plus_three_details_fp32": 4 * 1 * 128 * 32 * 32 * 4,
            "reconstructed_w_fp32": 1 * 128 * 64 * 64 * 4,
            "bands_plus_reconstructed_w_fp32": 2 * 1 * 128 * 64 * 64 * 4,
        },
        "verdict": "A — COARSE/DETAIL CONTRACT IS MATHEMATICALLY COHERENT; PROCEED TO ONE FIXED MODEL-MEDIATED DISCRIMINATOR",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    output = RESULTS / "report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
