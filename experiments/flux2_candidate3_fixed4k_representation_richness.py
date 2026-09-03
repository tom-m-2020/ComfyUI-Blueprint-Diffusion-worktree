"""Phase 16: fixed-4096 orthogonal spatial-mode representation probe."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_fixed4k_source_statistics as phase15
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_fixed4k_representation_richness_results"
REPORT = OUTPUT / "report.json"
MODE_NAMES = ("DC_0_0", "VERTICAL_1_0", "HORIZONTAL_0_1", "DIAGONAL_1_1")


def dct_matrix(length, *, device, dtype):
    positions = torch.arange(length, device=device, dtype=dtype) + 0.5
    frequencies = torch.arange(length, device=device, dtype=dtype)[:, None]
    matrix = torch.cos(math.pi * frequencies * positions[None, :] / length)
    matrix[0] *= math.sqrt(1.0 / length)
    matrix[1:] *= math.sqrt(2.0 / length)
    return matrix


def hadamard_128(*, device, dtype):
    matrix = torch.ones((1, 1), device=device, dtype=dtype)
    while matrix.shape[0] < 128:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1),
             torch.cat((matrix, -matrix), dim=1)), dim=0,
        )
    return matrix / math.sqrt(128.0)


def pack_modes(value):
    if tuple(value.shape[-2:]) != phase14.H_HW or value.shape[1] != 128:
        raise AssertionError(f"Expected [B,128,128,256], got {tuple(value.shape)}")
    dtype = torch.float32
    data = value.detach().to(dtype)
    q4 = dct_matrix(4, device=data.device, dtype=dtype)
    q2 = dct_matrix(2, device=data.device, dtype=dtype)
    channel_basis = hadamard_128(device=data.device, dtype=dtype)
    blocks = data.reshape(data.shape[0], 128, 32, 4, 128, 2).permute(0, 2, 4, 1, 3, 5)
    coefficients = torch.einsum("yu,brxcuv,zv->brxcyz", q4, blocks, q2)
    modes = torch.stack(
        (coefficients[..., 0, 0], coefficients[..., 1, 0],
         coefficients[..., 0, 1], coefficients[..., 1, 1]), dim=-1,
    )
    transformed = torch.einsum("oc,brxcm->brxom", channel_basis, modes)
    packed_basis = torch.zeros_like(transformed[..., 0])
    for mode_index in range(4):
        channel_slice = slice(32 * mode_index, 32 * (mode_index + 1))
        packed_basis[..., channel_slice] = transformed[..., channel_slice, mode_index]
    packed = torch.einsum("oc,brxo->brxc", channel_basis, packed_basis)
    source = packed.permute(0, 3, 1, 2).to(value.dtype)
    metadata = {
        "spatial_transform": "orthonormal separable DCT-II over each 4x2 cell",
        "retained_modes": list(MODE_NAMES),
        "channel_transform": "fixed normalized Sylvester-Hadamard 128x128",
        "channel_allocation": {
            MODE_NAMES[index]: [32 * index, 32 * (index + 1) - 1]
            for index in range(4)
        },
        "coefficients_before_pack_per_position": 512,
        "coefficients_after_pack_per_position": 128,
        "retained_coefficient_fraction": 0.25,
        "discarded": (
            "All spatial modes other than (0,0),(1,0),(0,1),(1,1), plus "
            "96 of 128 orthogonal channel components for each retained mode."
        ),
        "spatial_basis_orthogonality_max_error": float(
            max((q4 @ q4.T - torch.eye(4, device=data.device)).abs().max(),
                (q2 @ q2.T - torch.eye(2, device=data.device)).abs().max())
        ),
        "channel_basis_orthogonality_max_error": float(
            (channel_basis @ channel_basis.T - torch.eye(128, device=data.device)).abs().max()
        ),
    }
    return source, metadata


def reconstruct_modes(source):
    data = source.detach().float()
    q4 = dct_matrix(4, device=data.device, dtype=data.dtype)
    q2 = dct_matrix(2, device=data.device, dtype=data.dtype)
    channel_basis = hadamard_128(device=data.device, dtype=data.dtype)
    packed = data.permute(0, 2, 3, 1)
    packed_basis = torch.einsum("oc,brxc->brxo", channel_basis, packed)
    modes = torch.zeros((*packed.shape, 4), device=data.device, dtype=data.dtype)
    for mode_index in range(4):
        channel_slice = slice(32 * mode_index, 32 * (mode_index + 1))
        component = torch.zeros_like(packed_basis)
        component[..., channel_slice] = packed_basis[..., channel_slice]
        modes[..., mode_index] = torch.einsum("oc,brxo->brxc", channel_basis, component)
    coefficients = torch.zeros((*packed.shape, 4, 2), device=data.device, dtype=data.dtype)
    coefficients[..., 0, 0] = modes[..., 0]
    coefficients[..., 1, 0] = modes[..., 1]
    coefficients[..., 0, 1] = modes[..., 2]
    coefficients[..., 1, 1] = modes[..., 3]
    blocks = torch.einsum("yu,brxcyz,zv->brxcuv", q4, coefficients, q2)
    return blocks.permute(0, 3, 1, 4, 2, 5).reshape(data.shape[0], 128, 128, 256)


def reconstruction_metrics(reference, reconstructed):
    reference = reference.detach().float()
    reconstructed = reconstructed.detach().float()
    delta = reconstructed - reference
    ref_low = F.avg_pool2d(reference, kernel_size=8, stride=8)
    rec_low = F.avg_pool2d(reconstructed, kernel_size=8, stride=8)
    ref_dy, rec_dy = reference[..., 1:, :] - reference[..., :-1, :], reconstructed[..., 1:, :] - reconstructed[..., :-1, :]
    ref_dx, rec_dx = reference[..., :, 1:] - reference[..., :, :-1], reconstructed[..., :, 1:] - reconstructed[..., :, :-1]
    edge_error = torch.cat(((rec_dy - ref_dy).reshape(-1), (rec_dx - ref_dx).reshape(-1)))
    return {
        "H_reconstruction_rms": float(delta.square().mean().sqrt()),
        "H_reconstruction_max_abs": float(delta.abs().max()),
        "low_frequency_8x8_mean_rms": float((rec_low - ref_low).square().mean().sqrt()),
        "edge_gradient_rms": float(edge_error.square().mean().sqrt()),
        "reference_variance": float(reference.var(unbiased=False)),
        "reconstruction_variance": float(reconstructed.var(unbiased=False)),
        "variance_ratio": float(reconstructed.var(unbiased=False) / reference.var(unbiased=False)),
        "rms_ratio": float(reconstructed.square().mean().sqrt() / reference.square().mean().sqrt()),
    }


class Phase16Sampler(phase14.Phase14Sampler):
    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 16 requires empty-latent T2I without masks.")
        validate_schedule(sigmas)
        sampling = model.inner_model.model_sampling
        h0 = sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model, sigmas))
        coordinator = BlueprintCoordinator()
        state = coordinator.initialize(h0, sigmas[0])
        for ordinal in range(3):
            state, _ = coordinator.evaluate(
                guider=model, state=state, sigma=sigmas[ordinal],
                sigma_next=sigmas[ordinal + 1], model_options=extra_args["model_options"],
                seed=phase14.SEED,
            )
        sigma = sigmas[3]
        if float(sigmas[4]) != 0.0:
            raise AssertionError("Phase 16 is terminal-only.")
        regions = phase9b.DestinationPlanner().plan(phase14.H_HW)
        if len(regions) != 55:
            raise AssertionError(len(regions))
        accepted_h_hash = phase14.tensor_hash(state.h)
        accepted_g_hash = phase14.tensor_hash(state.g)
        working = []
        for region in regions:
            view = state.h[:, :, region.y:region.y2, region.x:region.x2]
            value = phase9c.make_working(view, sigma, 3, region)
            if float((phase9b.restrict2(value).float() - view.float()).abs().max()) > 1e-6:
                raise AssertionError(region.index)
            working.append(value)
        working_hashes = [phase14.tensor_hash(value) for value in working]

        source, representation = pack_modes(state.h)
        reconstructed = reconstruct_modes(source)
        mean_source = phase14.restrict_4x2(state.h)
        mean_reconstructed = mean_source.repeat_interleave(4, -2).repeat_interleave(2, -1)
        reconstruction = {
            "C3_orthogonal_modes": reconstruction_metrics(state.h, reconstructed),
            "C0_area_mean": reconstruction_metrics(state.h, mean_reconstructed),
        }
        assembled, crop_predictions, telemetry = self._block_major(
            coordinator, model, source, working, sigma,
            extra_args["model_options"], regions, model.inner_model.diffusion_model,
            "fixed_H",
        )
        overlap = phase8d.overlap_metrics(
            [item.detach().float().cpu() for item in crop_predictions], regions
        )
        phase15_report = json.loads(phase15.REPORT.read_text(encoding="utf-8"))
        if phase14.tensor_hash(state.h) != accepted_h_hash or phase14.tensor_hash(state.g) != accepted_g_hash:
            raise RuntimeError("C3 mutated accepted H/G.")
        if [phase14.tensor_hash(value) for value in working] != working_hashes:
            raise RuntimeError("C3 mutated W.")
        self.outputs["C3_FIXED_4096_ORTHOGONAL_MODES"] = assembled.detach().float().cpu()
        self.result = {
            "configuration": {
                **phase15_report["configuration"],
                "only_variable": "fixed source representation at identical 4096 positions",
            },
            "sigmas": [float(value) for value in sigmas],
            "accepted_state": {"H_hash": accepted_h_hash, "G_hash": accepted_g_hash},
            "working_hashes": working_hashes,
            "source": {
                "shape": list(source.shape), "tokens": 4096,
                "summary": phase14.summary(source),
                "per_channel": phase15.per_channel_summary(source),
                "representation": representation,
            },
            "reconstruction": reconstruction,
            "provenance": phase14.provenance_summary(),
            "variant": {
                "name": "C3_FIXED_4096_ORTHOGONAL_MODES",
                "overlap": overlap, "assembled": phase14.summary(assembled),
                **telemetry,
            },
            "reused_C0": {
                "variant": phase15_report["variants"]["C0_FIXED_4096_MEAN"],
                "decoded": phase15_report["decoded"]["C0_FIXED_4096_MEAN"],
            },
            "integrity": {
                "same_accepted_H_as_phase15": accepted_h_hash == phase15_report["accepted_state"]["H_hash"],
                "same_accepted_G_as_phase15": accepted_g_hash == phase15_report["accepted_state"]["G_hash"],
                "same_W_as_phase15": working_hashes == phase15_report["working_hashes"],
                "source_tokens": 4096,
                "source_blocks": telemetry["source_blocks"],
                "context_consumptions": telemetry["context_consumptions"],
                "finite": bool(torch.isfinite(assembled).all()),
                "terminal_state_updates": 0,
                "no_production_changes": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def make_sheet(c0_path, c3_path, destination):
    images = [
        ("C0_FIXED_4096_MEAN", Image.open(c0_path).convert("RGB")),
        ("C3_FIXED_4096_ORTHOGONAL_MODES", Image.open(c3_path).convert("RGB")),
    ]
    panels = []
    for name, image in images:
        image.thumbnail((4096, 640))
        panel = Image.new("RGB", (4096, image.height + 44), "white")
        panel.paste(image, ((4096 - image.width) // 2, 44))
        ImageDraw.Draw(panel).text((12, 12), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (4096, sum(item.height for item in panels)), "white")
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y)); y += panel.height
    sheet.save(destination)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = phase2.comfy.sd.load_diffusion_model(str(phase2.MODEL_PATH), model_options={})
    clip = phase2.comfy.sd.load_clip(
        [str(phase2.TEXT_ENCODER_PATH)], clip_type=phase2.comfy.sd.CLIPType.FLUX2
    )
    positive = clip.encode_from_tokens_scheduled(clip.tokenize(phase14.PROMPT))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
    del clip
    phase2.comfy.model_management.unload_all_models()
    phase2.comfy.model_management.soft_empty_cache()
    noise = torch.randn(
        (1, 128, *phase14.H_HW), generator=torch.Generator().manual_seed(phase14.SEED)
    )
    sigmas = phase2.get_schedule(phase14.STEPS, math.prod(phase14.H_HW)).float().clone()
    sigmas[0] = 1.0
    sampler = Phase16Sampler()
    perf.prepare_model_state(model)
    with torch.inference_mode():
        phase2.comfy.sample.sample_custom(
            model, noise, 1.0, sampler, sigmas, positive, negative,
            torch.zeros_like(noise), callback=lambda *args: None,
            disable_pbar=True, seed=phase14.SEED,
        )
    vae = phase2.comfy.sd.VAE(
        sd=phase2.comfy.utils.load_torch_file(str(phase2.VAE_PATH), safe_load=True)
    )
    pixels = vae.decode(sampler.outputs["C3_FIXED_4096_ORTHOGONAL_MODES"]).cpu()
    image_path = OUTPUT / "C3_FIXED_4096_ORTHOGONAL_MODES.png"
    phase2.save_pixels(pixels, image_path)
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        decoded = {
            "path": str(image_path), "dimensions_wh": list(rgb.size),
            "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
        }
    c0_path = Path(sampler.result["reused_C0"]["decoded"]["path"])
    sheet = OUTPUT / "C0_C3_comparison.png"
    make_sheet(c0_path, image_path, sheet)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet)
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet), "decoded": decoded}, indent=2))


if __name__ == "__main__":
    main()
