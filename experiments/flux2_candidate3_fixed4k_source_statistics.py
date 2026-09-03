"""Phase 15: fixed-4096 source scalar-statistics discriminator."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(r"C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev")
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import flux2_coarse_global_local_falsification as phase2
import flux2_candidate3_fixed4k_large_destination as phase14
import flux2_candidate3_native_local_global_context as phase9c
import flux2_candidate3_native_local_magnification as phase9b
import flux2_candidate3_performance_characterization as perf
import flux2_candidate3_terminal_context as phase8d

from blueprint_diffusion.sampling.euler import BlueprintCoordinator, validate_schedule


OUTPUT = ROOT / "experiments" / "flux2_candidate3_fixed4k_source_statistics_results"
REPORT = OUTPUT / "report.json"
GAINS = (
    ("C0_FIXED_4096_MEAN", 1.0),
    ("C1_FIXED_4096_PHASE13_SCALE", 2.0),
    ("C2_FIXED_4096_VARIANCE_PRESERVING", math.sqrt(8.0)),
)


def per_channel_summary(value):
    data = value.detach().float()
    flattened = data.movedim(1, 0).reshape(data.shape[1], -1)
    return {
        "mean": flattened.mean(dim=1).cpu().tolist(),
        "rms": flattened.square().mean(dim=1).sqrt().cpu().tolist(),
        "variance": flattened.var(dim=1, unbiased=False).cpu().tolist(),
    }


class Phase15Sampler(phase14.Phase14Sampler):
    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None or latent_image is None or bool(torch.count_nonzero(latent_image)):
            raise ValueError("Phase 15 requires empty-latent T2I without masks.")
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
            raise AssertionError("Phase 15 is terminal-only.")
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
        source_mean = phase14.restrict_4x2(state.h)
        h_variance = float(state.h.detach().float().var(unbiased=False))
        base_options = extra_args["model_options"]
        diffusion = model.inner_model.diffusion_model
        variants = {}
        outputs = {}
        for name, gain in GAINS:
            source = source_mean * gain
            source_data = source.detach().float()
            source_variance = float(source_data.var(unbiased=False))
            gc.collect()
            phase2.comfy.model_management.soft_empty_cache()
            assembled, crop_predictions, telemetry = self._block_major(
                coordinator, model, source, working, sigma, base_options,
                regions, diffusion, "fixed_H",
            )
            overlap = phase8d.overlap_metrics(
                [item.detach().float().cpu() for item in crop_predictions], regions
            )
            variants[name] = {
                "gain": gain,
                "source": {
                    "summary": phase14.summary(source),
                    "variance": source_variance,
                    "variance_ratio_vs_H": source_variance / h_variance,
                    "rms_ratio_vs_H": (
                        float(source_data.square().mean().sqrt())
                        / float(state.h.detach().float().square().mean().sqrt())
                    ),
                    "per_channel": per_channel_summary(source),
                },
                "overlap": overlap,
                "assembled": phase14.summary(assembled),
                **telemetry,
            }
            outputs[name] = assembled.detach().float().cpu()
            self.outputs[name] = outputs[name]
            if phase14.tensor_hash(state.h) != accepted_h_hash or phase14.tensor_hash(state.g) != accepted_g_hash:
                raise RuntimeError(f"{name} mutated accepted H/G.")
            if [phase14.tensor_hash(value) for value in working] != working_hashes:
                raise RuntimeError(f"{name} mutated W.")

        for name, _gain in GAINS[1:]:
            variants[name]["assembled_vs_C0"] = phase14.difference(
                outputs[name], outputs["C0_FIXED_4096_MEAN"]
            )
        phase14_report = json.loads(phase14.REPORT.read_text(encoding="utf-8"))
        self.result = {
            "configuration": {
                **phase14_report["configuration"],
                "only_variable": "fixed scalar gain after deterministic 4x2 area mean",
                "gains": {name: gain for name, gain in GAINS},
            },
            "sigmas": [float(value) for value in sigmas],
            "accepted_state": {"H_hash": accepted_h_hash, "G_hash": accepted_g_hash},
            "working_hashes": working_hashes,
            "source_mean_hash": phase14.tensor_hash(source_mean),
            "provenance": phase14.provenance_summary(),
            "variants": variants,
            "reused_phase14_controls": {
                "A_LOCAL_ONLY": phase14_report["variants"]["A_LOCAL_ONLY"],
                "B_DESTINATION_SCALED_CONTEXT": phase14_report["variants"]["B_DESTINATION_SCALED_CONTEXT"],
                "C0_expected_phase14_hash": phase14_report["variants"]["C_FIXED_DIRECT_4096"]["assembled"]["sha256"],
            },
            "integrity": {
                "same_accepted_H_as_phase14": accepted_h_hash == phase14_report["accepted_state"]["H_hash"],
                "same_accepted_G_as_phase14": accepted_g_hash == phase14_report["accepted_state"]["G_hash"],
                "same_W_as_phase14": working_hashes == phase14_report["working_hashes"],
                "C0_reproduces_phase14": phase14.tensor_hash(outputs["C0_FIXED_4096_MEAN"]) == phase14_report["variants"]["C_FIXED_DIRECT_4096"]["assembled"]["sha256"],
                "all_finite": all(item["assembled"]["finite"] for item in variants.values()),
                "terminal_state_updates": 0,
                "no_production_changes": True,
            },
        }
        return sampling.inverse_noise_scaling(sigmas[-1], state.h)


def make_sheet(paths, destination):
    images = [(name, Image.open(path).convert("RGB")) for name, path in paths]
    width, target_height = 4096, 512
    panels = []
    for name, image in images:
        image.thumbnail((width, target_height))
        panel = Image.new("RGB", (width, image.height + 44), "white")
        panel.paste(image, ((width - image.width) // 2, 44))
        ImageDraw.Draw(panel).text((12, 12), name, fill="black")
        panels.append(panel)
    sheet = Image.new("RGB", (width, sum(panel.height for panel in panels)), "white")
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
    sampler = Phase15Sampler()
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
    decoded, paths = {}, []
    for name, latent in sampler.outputs.items():
        pixels = vae.decode(latent).cpu()
        path = OUTPUT / f"{name}.png"
        phase2.save_pixels(pixels, path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            decoded[name] = {
                "path": str(path), "dimensions_wh": list(rgb.size),
                "sha256_rgb": hashlib.sha256(rgb.tobytes()).hexdigest(),
            }
        paths.append((name, path))
    sheet = OUTPUT / "C0_C1_C2_comparison.png"
    make_sheet(paths, sheet)
    sampler.result["decoded"] = decoded
    sampler.result["comparison_sheet"] = str(sheet)
    REPORT.write_text(json.dumps(sampler.result, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "sheet": str(sheet), "decoded": decoded}, indent=2))


if __name__ == "__main__":
    main()
