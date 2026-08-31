"""Candidate-3 hard-global-anchor and terminal-release falsifier.

Experiment only. Runs dense, tiled-only, uncoupled dual, hard global-anchor,
and terminal-release trajectories. No production integration, external K/V,
alternate coupling, cache, or parameter sweep.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate2_all_crop_assembly_probe as phase2d


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3
VARIANTS = (
    ("A_DENSE", "dense"),
    ("B_TILED_ONLY", "tiled"),
    ("C_UNCOUPLED_DUAL", "uncoupled"),
    ("D_HARD_GLOBAL_ANCHOR", "hard_anchor"),
    ("E_TERMINAL_RELEASE", "terminal_release"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "flux2_candidate3_hard_global_anchor_results",
    )
    parser.add_argument("--prompt", default=phase2.PROMPT)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument(
        "--global-geometry",
        choices=("mean_16x32", "block_dct_24x48"),
        default="mean_16x32",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def mean_restrict(value: torch.Tensor) -> torch.Tensor:
    """D: nonoverlapping 2x2 area restriction."""
    if value.shape[-2] % 2 or value.shape[-1] % 2:
        raise ValueError(f"D requires even spatial dimensions, got {value.shape[-2:]}")
    return F.avg_pool2d(value, kernel_size=2, stride=2)


def mean_prolong(value: torch.Tensor) -> torch.Tensor:
    """U: 2x nearest-neighbor prolongation; D(U(x)) == x."""
    return F.interpolate(value, scale_factor=2, mode="nearest")


def dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    positions = torch.arange(size, device=device, dtype=torch.float64)
    frequencies = positions[:, None]
    matrix = torch.cos(math.pi / size * (positions[None, :] + 0.5) * frequencies)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix.to(dtype=dtype)


def _blocks_to_grid(blocks: torch.Tensor) -> torch.Tensor:
    batch, channels, blocks_y, blocks_x, local_h, local_w = blocks.shape
    return blocks.permute(0, 1, 2, 4, 3, 5).reshape(
        batch, channels, blocks_y * local_h, blocks_x * local_w
    )


def _grid_to_blocks(value: torch.Tensor, block_h: int, block_w: int) -> torch.Tensor:
    batch, channels, height, width = value.shape
    if height % block_h or width % block_w:
        raise ValueError(
            f"Grid {value.shape[-2:]} is not divisible by block {(block_h, block_w)}."
        )
    blocks_y = height // block_h
    blocks_x = width // block_w
    return value.reshape(batch, channels, blocks_y, block_h, blocks_x, block_w).permute(
        0, 1, 2, 4, 3, 5
    )


def block_dct_restrict(value: torch.Tensor) -> torch.Tensor:
    """D: local constant-preserving 4x4 -> 3x3 orthonormal DCT restriction."""
    work = value.float()
    blocks = _grid_to_blocks(work, 4, 4)
    q4 = dct_matrix(4, work.device, work.dtype)
    q3 = dct_matrix(3, work.device, work.dtype)
    coefficients = torch.matmul(torch.matmul(q4, blocks), q4.T)
    retained = coefficients[..., :3, :3]
    global_blocks = 0.75 * torch.matmul(torch.matmul(q3.T, retained), q3)
    return _blocks_to_grid(global_blocks).to(dtype=value.dtype)


def block_dct_prolong(value: torch.Tensor) -> torch.Tensor:
    """U: local 3x3 -> 4x4 zero-padded orthonormal DCT synthesis."""
    work = value.float()
    blocks = _grid_to_blocks(work, 3, 3)
    q3 = dct_matrix(3, work.device, work.dtype)
    q4 = dct_matrix(4, work.device, work.dtype)
    retained = torch.matmul(torch.matmul(q3, blocks), q3.T)
    coefficients = torch.zeros(
        (*retained.shape[:-2], 4, 4), dtype=work.dtype, device=work.device
    )
    coefficients[..., :3, :3] = retained
    high_blocks = (4.0 / 3.0) * torch.matmul(torch.matmul(q4.T, coefficients), q4)
    return _blocks_to_grid(high_blocks).to(dtype=value.dtype)


def geometry_operators(name: str):
    if name == "mean_16x32":
        return mean_restrict, mean_prolong, (16, 32)
    if name == "block_dct_24x48":
        return block_dct_restrict, block_dct_prolong, (24, 48)
    raise ValueError(name)


def tensor_difference(value: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    difference = value.float() - reference.float()
    return {
        "rms": float(difference.square().mean().sqrt()),
        "mean_abs": float(difference.abs().mean()),
        "max_abs": float(difference.abs().max()),
        "norm": float(torch.linalg.vector_norm(difference)),
    }


def cosine(value: torch.Tensor, reference: torch.Tensor) -> float | None:
    left = value.float().flatten()
    right = reference.float().flatten()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def adjacent_correlation(value: torch.Tensor) -> dict[str, float | None]:
    def correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
        left = left.float().flatten()
        right = right.float().flatten()
        left = left - left.mean()
        right = right - right.mean()
        denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        if float(denominator) == 0.0:
            return None
        return float(torch.dot(left, right) / denominator)

    return {
        "horizontal": correlation(value[..., :, :-1], value[..., :, 1:]),
        "vertical": correlation(value[..., :-1, :], value[..., 1:, :]),
    }


def clone_options(base: dict[str, Any], rope: dict[str, float]) -> dict[str, Any]:
    return phase2.clone_options(base, rope)


def model_call(model, value, sigma, base_options, rope, seed, role, crop=None):
    started = time.perf_counter()
    prediction = model(
        value,
        sigma.expand(value.shape[0]),
        model_options=clone_options(base_options, rope),
        seed=seed,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    record = {
        "role": role,
        "sigma": float(sigma),
        "input_latent_hw": list(value.shape[-2:]),
        "image_tokens": int(value.shape[-2] * value.shape[-1]),
        "rope_options": rope,
        "input_object_id": id(value),
        "input_storage_data_ptr": value.untyped_storage().data_ptr(),
        "input": phase2.stats(value),
        "prediction": phase2.stats(prediction),
        "seconds": time.perf_counter() - started,
    }
    if crop is not None:
        record["crop"] = crop.__dict__
    return prediction, record


def assemble_live(predictions, crops, target_hw):
    output = torch.zeros(
        (predictions[0].shape[0], predictions[0].shape[1], *target_hw),
        dtype=predictions[0].dtype,
        device=predictions[0].device,
    )
    coverage = torch.zeros(
        (predictions[0].shape[0], 1, *target_hw),
        dtype=torch.float32,
        device=predictions[0].device,
    )
    for prediction, crop in zip(predictions, crops):
        weight = phase2.crop_weight(crop, crops, output.device)[None, None]
        output[:, :, crop.y:crop.y2, crop.x:crop.x2] += prediction * weight
        coverage[:, :, crop.y:crop.y2, crop.x:crop.x2] += weight
    if float(coverage.min()) <= 0:
        raise RuntimeError(f"Incomplete assembly coverage: {phase2.stats(coverage)}")
    return output / coverage, coverage


@dataclass
class TrajectoryTrace:
    name: str
    variant: str
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    h_predictions: dict[int, torch.Tensor] = field(default_factory=dict)
    g_predictions: dict[int, torch.Tensor] = field(default_factory=dict)
    crop_predictions: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    h_before: dict[int, torch.Tensor] = field(default_factory=dict)
    h_proposals: dict[int, torch.Tensor] = field(default_factory=dict)
    h_after: dict[int, torch.Tensor] = field(default_factory=dict)
    g_before: dict[int, torch.Tensor] = field(default_factory=dict)
    g_proposals: dict[int, torch.Tensor] = field(default_factory=dict)
    g_after: dict[int, torch.Tensor] = field(default_factory=dict)
    projections: dict[int, torch.Tensor] = field(default_factory=dict)
    atomic_acceptances: int = 0
    seconds: float = 0.0
    peak_allocated_gib: float = 0.0


class Candidate3Sampler(phase2.comfy.samplers.Sampler):
    def __init__(
        self, variant, target_hw, global_hw, crops, trace, seed,
        restrict_fn, prolong_fn, operator_name,
    ) -> None:
        self.variant = variant
        self.target_hw = target_hw
        self.global_hw = global_hw
        self.crops = crops
        self.trace = trace
        self.seed = seed
        self.restrict = restrict_fn
        self.prolong = prolong_fn
        self.operator_name = operator_name
        self.global_rope = phase2.rope_for_global(*target_hw, *global_hw)

    def local_prediction(self, model, h, sigma, base_options, step):
        predictions = []
        calls = []
        accepted_object_id = id(h)
        accepted_storage = h.untyped_storage().data_ptr()
        for crop in self.crops:
            view = h[:, :, crop.y:crop.y2, crop.x:crop.x2]
            prediction, call = model_call(
                model,
                view,
                sigma,
                base_options,
                phase2.rope_for_crop(crop),
                self.seed,
                "local",
                crop,
            )
            if call["input_storage_data_ptr"] != accepted_storage:
                raise AssertionError(f"Crop {crop.index} did not read accepted H storage.")
            call["accepted_h_object_id"] = accepted_object_id
            call["accepted_h_storage_data_ptr"] = accepted_storage
            call["crop_read_only_before_atomic_acceptance"] = True
            predictions.append(prediction)
            calls.append(call)
        assembled, coverage = assemble_live(predictions, self.crops, self.target_hw)
        overlap = phase2d.overlap_disagreement(predictions, self.crops)
        self.trace.crop_predictions[step] = [item.detach().float().cpu() for item in predictions]
        return assembled, calls, coverage, overlap

    def dense_prediction(self, model, h, sigma, base_options):
        prediction, call = model_call(
            model, h, sigma, base_options, {}, self.seed, "dense"
        )
        return prediction, [call], torch.ones_like(h[:, :1]), None

    def global_prediction(self, model, g, sigma, base_options):
        return model_call(
            model,
            g,
            sigma,
            base_options,
            self.global_rope,
            self.seed,
            "low_density_global",
        )

    def sample(
        self,
        model,
        sigmas,
        extra_args,
        callback,
        noise,
        latent_image=None,
        denoise_mask=None,
        disable_pbar=False,
    ):
        if denoise_mask is not None:
            raise ValueError("This T2I falsifier does not use a mask.")
        h = noise
        g = self.restrict(h) if self.variant in {"uncoupled", "hard_anchor", "terminal_release"} else None
        total = len(sigmas) - 1
        previous_projection = None

        for step in range(total):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            self.trace.h_before[step] = h.detach().float().cpu()
            if g is not None:
                self.trace.g_before[step] = g.detach().float().cpu()

            if self.variant == "dense":
                h_prediction, local_calls, coverage, overlap = self.dense_prediction(
                    model, h, sigma, extra_args["model_options"]
                )
                global_call = None
            else:
                global_call = None
                if g is not None:
                    g_prediction, global_call = self.global_prediction(
                        model, g, sigma, extra_args["model_options"]
                    )
                    self.trace.g_predictions[step] = g_prediction.detach().float().cpu()
                h_prediction, local_calls, coverage, overlap = self.local_prediction(
                    model, h, sigma, extra_args["model_options"], step
                )

            self.trace.h_predictions[step] = h_prediction.detach().float().cpu()
            h_derivative = (h - h_prediction) / sigma
            h_proposal = h + h_derivative * (sigma_next - sigma)
            self.trace.h_proposals[step] = h_proposal.detach().float().cpu()

            calls = ([] if global_call is None else [global_call]) + local_calls
            record: dict[str, Any] = {
                "step": step,
                "sigma": float(sigma),
                "sigma_next": float(sigma_next),
                "evaluation_identity": f"{self.variant}:euler:{step}:sigma={float(sigma):.9g}",
                "accepted_h_object_id": id(h),
                "accepted_h_storage_data_ptr": h.untyped_storage().data_ptr(),
                "h_prediction": phase2.stats(h_prediction),
                "h_proposal": phase2.stats(h_proposal),
                "coverage": phase2.stats(coverage),
                "overlap_disagreement": overlap,
                "calls": calls,
                "model_forwards": len(calls),
                "executed_image_tokens": sum(call["image_tokens"] for call in calls),
                "crop_state_updates": 0,
                "assembled_h_predictions_consumed": 1,
                "atomic_pair_acceptances": 0,
            }

            if g is None:
                h_next = h_proposal
                record["high_resolution_acceptances"] = 1
            else:
                g_derivative = (g - g_prediction) / sigma
                g_proposal = g + g_derivative * (sigma_next - sigma)
                self.trace.g_proposals[step] = g_proposal.detach().float().cpu()
                coarse_h_proposal = self.restrict(h_proposal)
                consistency_delta = g_proposal - coarse_h_proposal
                projection = self.prolong(consistency_delta)
                coarse_global_update = g_proposal - g
                coarse_local_update = coarse_h_proposal - self.restrict(h)

                release_terminal = self.variant == "terminal_release" and float(sigma_next) == 0.0
                apply_projection = self.variant in {"hard_anchor", "terminal_release"} and not release_terminal
                if apply_projection:
                    h_next = h_proposal + projection
                    applied_projection = projection
                else:
                    h_next = h_proposal
                    applied_projection = torch.zeros_like(projection)
                g_next = g_proposal

                before = tensor_difference(coarse_h_proposal, g_proposal)
                after = tensor_difference(self.restrict(h_next), g_next)
                if apply_projection and after["max_abs"] > 1e-5:
                    raise AssertionError(f"D(H_next) != G_next at step {step}: {after}")
                if apply_projection and not torch.equal(
                    self.restrict(applied_projection), consistency_delta
                ):
                    difference = tensor_difference(self.restrict(applied_projection), consistency_delta)
                    if difference["max_abs"] > 1e-6:
                        raise AssertionError(f"D/U projection identity failed: {difference}")

                record.update(
                    {
                        "accepted_g_object_id": id(g),
                        "accepted_g_storage_data_ptr": g.untyped_storage().data_ptr(),
                        "global_prediction": phase2.stats(g_prediction),
                        "g_proposal": phase2.stats(g_proposal),
                        "consistency_before_projection": before,
                        "projection_available": phase2.stats(projection),
                        "projection_applied": phase2.stats(applied_projection),
                        "projection_policy": (
                            "terminal_release" if release_terminal else
                            "hard_projection" if apply_projection else
                            "uncoupled"
                        ),
                        "terminal_interval": float(sigma_next) == 0.0,
                        "projection_rms_to_h_proposal_rms": float(
                            applied_projection.float().square().mean().sqrt()
                            / h_proposal.float().square().mean().sqrt()
                        ),
                        "consistency_after_acceptance": after,
                        "coarse_global_update": phase2.stats(coarse_global_update),
                        "coarse_local_update": phase2.stats(coarse_local_update),
                        "global_vs_local_coarse_update_cosine": cosine(
                            coarse_global_update, coarse_local_update
                        ),
                        "projection_cosine_vs_previous_step": (
                            None if previous_projection is None else cosine(projection, previous_projection)
                        ),
                        "global_forwards": 1,
                        "local_forwards": 3,
                        "global_rope_options": self.global_rope,
                        "global_operator": self.operator_name,
                        "shared_sigma_for_global_and_local": all(
                            call["sigma"] == float(sigma) for call in calls
                        ),
                        "high_resolution_acceptances": 1,
                    }
                )
                if len(calls) != 4 or record["global_forwards"] != 1 or record["local_forwards"] != 3:
                    raise AssertionError(f"Candidate-3 forward lifecycle mismatch: {record}")
                self.trace.projections[step] = applied_projection.detach().float().cpu()
                previous_projection = projection.detach()
                g = g_next
                self.trace.g_after[step] = g.detach().float().cpu()

            h = h_next
            self.trace.h_after[step] = h.detach().float().cpu()
            self.trace.atomic_acceptances += 1
            record["atomic_pair_acceptances"] = 1 if g is not None else 0
            record["accepted_h_after"] = phase2.stats(h)
            if g is not None:
                record["accepted_g_after"] = phase2.stats(g)
            self.trace.evaluations.append(record)
            if callback is not None:
                callback(step, h_prediction, h, total)

        if self.trace.atomic_acceptances != total:
            raise AssertionError(
                f"Expected {total} accepted intervals, got {self.trace.atomic_acceptances}."
            )
        return h


def run_trajectory(model, noise, positive, negative, sigmas, sampler, seed):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = phase2.comfy.sample.sample_custom(
            model,
            noise.clone(),
            1.0,
            sampler,
            sigmas.clone(),
            positive,
            negative,
            torch.zeros_like(noise),
            disable_pbar=True,
            seed=seed,
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        sampler.trace.peak_allocated_gib = torch.cuda.max_memory_allocated() / GIB
    sampler.trace.seconds = time.perf_counter() - started
    return output.detach().float().cpu()


def save_projection_heatmap(value: torch.Tensor, path: Path) -> None:
    magnitude = value.float().square().mean(dim=1, keepdim=True).sqrt()
    phase2.save_heatmap(magnitude, path, (512, 1024))


def save_contact_sheet(paths: list[tuple[str, Path]], output: Path) -> None:
    images = [(label, Image.open(path).convert("RGB")) for label, path in paths]
    width = max(image.width for _, image in images)
    height = max(image.height for _, image in images)
    rows = math.ceil(len(images) / 2)
    sheet = Image.new("RGB", (width * 2, (height + 32) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(images):
        x = (index % 2) * width
        y = (index // 2) * (height + 32)
        sheet.paste(image, (x, y + 32))
        draw.text((x + 8, y + 8), label, fill="black")
    sheet.save(output)
    for _, image in images:
        image.close()


def dry_run() -> None:
    torch.manual_seed(3)
    h = torch.randn(1, 2, 8, 12)
    g = mean_restrict(h)
    identity = tensor_difference(mean_restrict(mean_prolong(g)), g)
    g_proposal = g + 0.1 * torch.randn_like(g)
    h_proposal = h + 0.1 * torch.randn_like(h)
    h_next = h_proposal + mean_prolong(g_proposal - mean_restrict(h_proposal))
    invariant = tensor_difference(mean_restrict(h_next), g_proposal)
    if identity["max_abs"] > 1e-7 or invariant["max_abs"] > 1e-6:
        raise AssertionError({"identity": identity, "invariant": invariant})
    dct_h = torch.randn(1, 2, 8, 12)
    dct_g = block_dct_restrict(dct_h)
    dct_identity = tensor_difference(block_dct_restrict(block_dct_prolong(dct_g)), dct_g)
    if dct_identity["max_abs"] > 2e-6:
        raise AssertionError({"block_dct_identity": dct_identity})
    print(json.dumps({
        "mean_D_U_identity": identity,
        "mean_accepted_invariant": invariant,
        "block_dct_D_U_identity": dct_identity,
    }, indent=2))


def main() -> None:
    args = parse_args()
    if args.dry_run:
        dry_run()
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    width, height = 1024, 512
    target_hw = (height // 16, width // 16)
    restrict_fn, prolong_fn, global_hw = geometry_operators(args.global_geometry)
    crop_hw = (512 // 16, 512 // 16)
    overlap_pixels = 128
    seed = args.seed
    prompt = args.prompt
    crops = phase2.crops_for_canvas(*target_hw, *crop_hw, overlap_pixels // 16)
    sigmas = phase2.get_schedule(4, math.prod(target_hw)).float().clone()
    global_rope = phase2.rope_for_global(*target_hw, *global_hw)

    synthetic_generator = torch.Generator().manual_seed(314159)
    synthetic_g = torch.randn((1, 3, *global_hw), generator=synthetic_generator)
    synthetic_right_inverse = tensor_difference(
        restrict_fn(prolong_fn(synthetic_g)), synthetic_g
    )
    if synthetic_right_inverse["max_abs"] > 2e-6:
        raise AssertionError(
            f"Synthetic D(U(G)) right inverse failed before model load: {synthetic_right_inverse}"
        )

    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()

    noise = torch.randn((1, 128, *target_hw), generator=torch.Generator().manual_seed(seed))
    mapped_global_noise = restrict_fn(noise)
    initialization = {
        "H_0": phase2.stats(noise),
        "G_0_equals_D_H_0": tensor_difference(mapped_global_noise, restrict_fn(noise)),
        "G_0": phase2.stats(mapped_global_noise),
        "variance_ratio_G_to_H": float(mapped_global_noise.float().var() / noise.float().var()),
        "adjacent_correlation_H": adjacent_correlation(noise),
        "adjacent_correlation_G": adjacent_correlation(mapped_global_noise),
        "operator": args.global_geometry,
        "predicted_variance_ratio": 0.5625 if args.global_geometry == "block_dct_24x48" else 0.25,
    }

    outputs: dict[str, torch.Tensor] = {}
    traces: dict[str, TrajectoryTrace] = {}
    for name, variant in VARIANTS:
        print(f"Running {name}...", flush=True)
        trace = TrajectoryTrace(name, variant)
        sampler = Candidate3Sampler(
            variant, target_hw, global_hw, crops, trace, seed,
            restrict_fn, prolong_fn, args.global_geometry,
        )
        outputs[name] = run_trajectory(model, noise, positive, negative, sigmas, sampler, seed)
        traces[name] = trace

    uncoupled_equivalence = []
    global_equivalence = []
    for step in range(4):
        h_difference = tensor_difference(
            traces["C_UNCOUPLED_DUAL"].h_after[step],
            traces["B_TILED_ONLY"].h_after[step],
        )
        uncoupled_equivalence.append({"step": step, **h_difference})
        if h_difference["max_abs"] > 1e-5:
            raise AssertionError(f"C.H != B.H at accepted step {step}: {h_difference}")
        g_difference = tensor_difference(
            traces["C_UNCOUPLED_DUAL"].g_after[step],
            traces["D_HARD_GLOBAL_ANCHOR"].g_after[step],
        )
        global_equivalence.append({"step": step, **g_difference})
        if g_difference["max_abs"] > 1e-5:
            raise AssertionError(f"C.G != D.G at accepted step {step}: {g_difference}")

    hard_vs_release_before = []
    hard_vs_release_after = []
    hard_vs_release_prediction = []
    hard_vs_release_proposal = []
    hard_vs_release_global = []
    for step in range(4):
        before = tensor_difference(
            traces["D_HARD_GLOBAL_ANCHOR"].h_before[step],
            traces["E_TERMINAL_RELEASE"].h_before[step],
        )
        prediction = tensor_difference(
            traces["D_HARD_GLOBAL_ANCHOR"].h_predictions[step],
            traces["E_TERMINAL_RELEASE"].h_predictions[step],
        )
        proposal = tensor_difference(
            traces["D_HARD_GLOBAL_ANCHOR"].h_proposals[step],
            traces["E_TERMINAL_RELEASE"].h_proposals[step],
        )
        after = tensor_difference(
            traces["D_HARD_GLOBAL_ANCHOR"].h_after[step],
            traces["E_TERMINAL_RELEASE"].h_after[step],
        )
        global_state = tensor_difference(
            traces["D_HARD_GLOBAL_ANCHOR"].g_after[step],
            traces["E_TERMINAL_RELEASE"].g_after[step],
        )
        hard_vs_release_before.append({"step": step, **before})
        hard_vs_release_prediction.append({"step": step, **prediction})
        hard_vs_release_proposal.append({"step": step, **proposal})
        hard_vs_release_after.append({"step": step, **after})
        hard_vs_release_global.append({"step": step, **global_state})
        if before["max_abs"] > 1e-5 or prediction["max_abs"] > 1e-5 or proposal["max_abs"] > 1e-5:
            raise AssertionError(
                f"D/E diverged before acceptance at step {step}: "
                f"before={before}, prediction={prediction}, proposal={proposal}"
            )
        if step < 3 and after["max_abs"] > 1e-5:
            raise AssertionError(f"D/E earlier accepted states differ at step {step}: {after}")
        if global_state["max_abs"] > 1e-5:
            raise AssertionError(f"D/E global accepted states differ at step {step}: {global_state}")
    if hard_vs_release_after[3]["max_abs"] == 0.0:
        raise AssertionError("D/E final accepted H unexpectedly remained identical.")

    per_step = []
    for step in range(4):
        dense_prediction = traces["A_DENSE"].h_predictions[step]
        tiled_prediction = traces["B_TILED_ONLY"].h_predictions[step]
        uncoupled_prediction = traces["C_UNCOUPLED_DUAL"].h_predictions[step]
        coupled_prediction = traces["D_HARD_GLOBAL_ANCHOR"].h_predictions[step]
        release_prediction = traces["E_TERMINAL_RELEASE"].h_predictions[step]
        per_step.append(
            {
                "step": step,
                "sigma": float(sigmas[step]),
                "tiled_prediction_vs_dense": phase2d.comparison(tiled_prediction, dense_prediction),
                "uncoupled_prediction_vs_dense": phase2d.comparison(uncoupled_prediction, dense_prediction),
                "coupled_prediction_vs_dense": phase2d.comparison(coupled_prediction, dense_prediction),
                "terminal_release_prediction_vs_dense": phase2d.comparison(release_prediction, dense_prediction),
                "coupled_h_state_vs_dense": phase2d.comparison(
                    traces["D_HARD_GLOBAL_ANCHOR"].h_after[step],
                    traces["A_DENSE"].h_after[step],
                ),
                "tiled_h_state_vs_dense": phase2d.comparison(
                    traces["B_TILED_ONLY"].h_after[step],
                    traces["A_DENSE"].h_after[step],
                ),
                "uncoupled_h_equals_tiled": uncoupled_equivalence[step],
                "uncoupled_g_equals_coupled_g": global_equivalence[step],
                "hard_anchor_coupling": traces["D_HARD_GLOBAL_ANCHOR"].evaluations[step],
                "terminal_release_coupling": traces["E_TERMINAL_RELEASE"].evaluations[step],
            }
        )

    final_paths = []
    for name, output in outputs.items():
        with torch.inference_mode():
            pixels = vae.decode(output).cpu()
        path = args.output_dir / f"{name}_FINAL.png"
        phase2.save_pixels(pixels, path)
        final_paths.append((name, path))
        for step in range(4):
            with torch.inference_mode():
                estimate = vae.decode(traces[name].h_predictions[step]).cpu()
                accepted = vae.decode(traces[name].h_after[step]).cpu()
            phase2.save_pixels(estimate, args.output_dir / f"{name}_STEP_{step:02d}_DENOISED.png")
            phase2.save_pixels(accepted, args.output_dir / f"{name}_STEP_{step:02d}_ACCEPTED_H.png")

    for name in ("C_UNCOUPLED_DUAL", "D_HARD_GLOBAL_ANCHOR", "E_TERMINAL_RELEASE"):
        for step in range(4):
            with torch.inference_mode():
                g_prediction = vae.decode(traces[name].g_predictions[step]).cpu()
                g_accepted = vae.decode(traces[name].g_after[step]).cpu()
            phase2.save_pixels(g_prediction, args.output_dir / f"{name}_STEP_{step:02d}_GLOBAL_DENOISED.png")
            phase2.save_pixels(g_accepted, args.output_dir / f"{name}_STEP_{step:02d}_ACCEPTED_G.png")

    for step in range(4):
        with torch.inference_mode():
            proposed = vae.decode(traces["D_HARD_GLOBAL_ANCHOR"].h_proposals[step]).cpu()
        phase2.save_pixels(proposed, args.output_dir / f"D_HARD_GLOBAL_ANCHOR_STEP_{step:02d}_PROPOSED_H.png")
        save_projection_heatmap(
            traces["D_HARD_GLOBAL_ANCHOR"].projections[step],
            args.output_dir / f"D_HARD_GLOBAL_ANCHOR_STEP_{step:02d}_PROJECTION_MAGNITUDE.png",
        )

    save_contact_sheet(final_paths, args.output_dir / "FINAL_COMPARISON.png")
    save_contact_sheet(
        [
            ("TERMINAL LOCAL PROPOSAL", args.output_dir / "D_HARD_GLOBAL_ANCHOR_STEP_03_PROPOSED_H.png"),
            ("HARD-PROJECTED FINAL", args.output_dir / "D_HARD_GLOBAL_ANCHOR_FINAL.png"),
            ("TERMINAL-RELEASE FINAL", args.output_dir / "E_TERMINAL_RELEASE_FINAL.png"),
            ("LOW-DENSITY G FINAL", args.output_dir / "D_HARD_GLOBAL_ANCHOR_STEP_03_ACCEPTED_G.png"),
        ],
        args.output_dir / "TERMINAL_RELEASE_COMPARISON.png",
    )

    report = {
        "configuration": {
            "model": str(phase2.MODEL_PATH),
            "text_encoder": str(phase2.TEXT_ENCODER_PATH),
            "vae": str(phase2.VAE_PATH),
            "prompt": prompt,
            "seed": seed,
            "cfg": 1.0,
            "sampler": "four-step zero-churn Euler; one atomic accepted interval",
            "sigmas": sigmas.tolist(),
            "target_image_hw": [height, width],
            "target_latent_hw": list(target_hw),
            "global_image_hw": [global_hw[0] * 16, global_hw[1] * 16],
            "global_latent_hw": list(global_hw),
            "crop_image_hw": [512, 512],
            "crop_latent_hw": list(crop_hw),
            "overlap_pixels": overlap_pixels,
            "crops": [crop.__dict__ for crop in crops],
            "global_geometry_operator": args.global_geometry,
            "D": (
                "blockwise constant-preserving orthonormal DCT 4x4 -> retained 3x3 -> spatial 3x3"
                if args.global_geometry == "block_dct_24x48" else
                "torch.avg_pool2d(kernel_size=2, stride=2)"
            ),
            "U": (
                "blockwise spatial 3x3 -> DCT 3x3 -> zero-pad 4x4 -> orthonormal synthesis"
                if args.global_geometry == "block_dct_24x48" else
                "torch.interpolate(scale_factor=2, mode='nearest')"
            ),
            "synthetic_right_inverse_before_model": synthetic_right_inverse,
            "global_coordinate_convention": {
                "description": f"{global_hw[0]}x{global_hw[1]} global grid spans the full 32x64 target coordinate endpoints",
                "rope_options": global_rope,
                "y_coordinate": f"y_global * {global_rope['scale_y']}",
                "x_coordinate": f"x_global * {global_rope['scale_x']}",
                "endpoint_mapping": {
                    f"global_y_0_to_{global_hw[0] - 1}": [0.0, 31.0],
                    f"global_x_0_to_{global_hw[1] - 1}": [0.0, 63.0],
                },
            },
            "coupling": "G_next=G*; H_next=H*+U(G*-D(H*))",
            "terminal_release": "when sigma_next==0: G_next=G*; H_next=H*",
            "forbidden_mechanisms_present": False,
        },
        "initialization": initialization,
        "integrity": {
            "uncoupled_H_equals_tiled_H_per_accepted_state": uncoupled_equivalence,
            "uncoupled_G_equals_coupled_G_per_accepted_state": global_equivalence,
            "D_U_identity_expected": True,
            "candidate3_forwards_per_evaluation": {"global": 1, "local": 3, "total": 4},
            "candidate3_atomic_pair_acceptances": {
                name: traces[name].atomic_acceptances
                for name in ("C_UNCOUPLED_DUAL", "D_HARD_GLOBAL_ANCHOR", "E_TERMINAL_RELEASE")
            },
            "crop_state_updates": 0,
            "hard_vs_terminal_release": {
                "H_before_each_evaluation": hard_vs_release_before,
                "H_prediction_each_evaluation": hard_vs_release_prediction,
                "H_proposal_each_evaluation": hard_vs_release_proposal,
                "H_after_each_acceptance": hard_vs_release_after,
                "G_after_each_acceptance": hard_vs_release_global,
                "only_final_H_acceptance_differs": (
                    all(item["max_abs"] == 0.0 for item in hard_vs_release_after[:3])
                    and hard_vs_release_after[3]["max_abs"] > 0.0
                ),
            },
        },
        "per_step": per_step,
        "trajectories": {
            name: {
                "variant": trace.variant,
                "seconds": trace.seconds,
                "peak_allocated_gib": trace.peak_allocated_gib,
                "atomic_acceptances": trace.atomic_acceptances,
                "final_H": phase2.stats(outputs[name]),
                "final_G": (
                    None if not trace.g_after else phase2.stats(trace.g_after[3])
                ),
                "evaluations": trace.evaluations,
            }
            for name, trace in traces.items()
        },
        "final_latent_comparisons": {
            "tiled_vs_dense": phase2d.comparison(outputs["B_TILED_ONLY"], outputs["A_DENSE"]),
            "uncoupled_vs_tiled": phase2d.comparison(outputs["C_UNCOUPLED_DUAL"], outputs["B_TILED_ONLY"]),
            "hard_anchor_vs_dense": phase2d.comparison(outputs["D_HARD_GLOBAL_ANCHOR"], outputs["A_DENSE"]),
            "hard_anchor_vs_tiled": phase2d.comparison(outputs["D_HARD_GLOBAL_ANCHOR"], outputs["B_TILED_ONLY"]),
            "terminal_release_vs_dense": phase2d.comparison(outputs["E_TERMINAL_RELEASE"], outputs["A_DENSE"]),
            "terminal_release_vs_hard_anchor": phase2d.comparison(outputs["E_TERMINAL_RELEASE"], outputs["D_HARD_GLOBAL_ANCHOR"]),
            "terminal_release_coarse_consistency": phase2d.comparison(
                restrict_fn(outputs["E_TERMINAL_RELEASE"]),
                traces["E_TERMINAL_RELEASE"].g_after[3],
            ),
        },
        "outputs": {name: str(path) for name, path in final_paths},
        "final_comparison": str(args.output_dir / "FINAL_COMPARISON.png"),
        "terminal_release_comparison": str(args.output_dir / "TERMINAL_RELEASE_COMPARISON.png"),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "report.json"),
                "initial_variance_ratio_G_to_H": initialization["variance_ratio_G_to_H"],
                "uncoupled_max_errors": uncoupled_equivalence,
                "projection_steps": [
                    {
                        "step": item["step"],
                        "before_rms": item["consistency_before_projection"]["rms"],
                        "applied_rms": item["projection_applied"]["rms"],
                        "ratio_to_h_proposal": item["projection_rms_to_h_proposal_rms"],
                        "after_max": item["consistency_after_acceptance"]["max_abs"],
                        "proposal_update_cosine": item["global_vs_local_coarse_update_cosine"],
                    }
                    for item in traces["D_HARD_GLOBAL_ANCHOR"].evaluations
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
