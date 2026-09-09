"""Step-0 clean-source feature/KV context diagnostic for Local Edit."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

import local_edit_prediction_state_ownership as previous
import local_edit_step0_source_context as noisy
import flux2_coarse_global_local_falsification as phase2


OUTPUT = previous.clown.ROOT / "experiments" / "local_edit_step0_clean_source_context_results"
FROZEN_D1 = previous.clown.ROOT / "experiments" / "local_edit_step0_source_context_results"


class CleanSourceStepZeroSampler(noisy.StepZeroSampler):
    def __init__(self, source_latent: torch.Tensor, fixed_noise: torch.Tensor, seed: int) -> None:
        super().__init__(source_latent, fixed_noise, seed)
        self.noisy_probe = self.probe
        self.clean_probe = noisy.SourceContextProbe()
        self.clean_source = None

    def sample(self, model, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        if denoise_mask is not None:
            raise ValueError("Clean-source context diagnostic rejects noise masks")
        if latent_image is None:
            raise ValueError("Clean-source context requires Comfy's processed source latent")
        if len(sigmas) != 2 or float(sigmas[0]) != 1.0 or float(sigmas[1]) != 0.0:
            raise ValueError(f"Expected diagnostic sigmas [1,0], got {sigmas.tolist()}")
        sampling = model.inner_model.model_sampling
        sigma = sigmas[0]
        clean_source = latent_image
        self.clean_source = clean_source.detach().float().cpu()
        fixed_noise = noise
        accepted = sampling.noise_scaling(sigma, fixed_noise, clean_source, self.max_denoise(model, sigmas))
        noisy_source = sampling.noise_scaling(sigma, fixed_noise, clean_source, self.max_denoise(model, sigmas))
        base_options = extra_args["model_options"]

        self.outputs["D0_STATE_RESTORATION"] = self.call(
            model, accepted, sigma, base_options, noisy.options(self.noisy_probe, "ordinary"), "D0_CONTROL"
        ).detach().float().cpu()
        self.call(
            model, noisy_source, sigma, base_options, noisy.options(self.noisy_probe, "capture"), "D1_NOISY_CAPTURE"
        )
        self.outputs["D1_NOISY_SOURCE_CONTEXT"] = self.call(
            model, accepted, sigma, base_options, noisy.options(self.noisy_probe, "context"), "D1_CONTEXT"
        ).detach().float().cpu()
        self.noisy_probe.assert_complete()

        self.call(
            model, clean_source, sigma, base_options, noisy.options(self.clean_probe, "capture"), "D2_CLEAN_CAPTURE"
        )
        self.outputs["D2_CLEAN_SOURCE_CONTEXT"] = self.call(
            model, accepted, sigma, base_options, noisy.options(self.clean_probe, "context"), "D2_CONTEXT"
        ).detach().float().cpu()
        self.clean_probe.assert_complete()

        self.records["state"] = {
            "D0_D1_D2_target_input_difference": noisy.tensor_difference(accepted, accepted),
            "noisy_source_vs_fixed_noise": noisy.tensor_difference(noisy_source, fixed_noise),
            "clean_source_vs_sampler_latent_image": noisy.tensor_difference(clean_source, latent_image),
            "clean_source_vs_fixed_noise": noisy.tensor_difference(clean_source, fixed_noise),
            "clean_source_equal_fixed_noise": bool(torch.equal(clean_source, fixed_noise)),
            "sampler_updates": 0,
            "returned_state_vs_accepted": noisy.tensor_difference(accepted, accepted),
        }
        return accepted


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = previous.clown.comfy.sd.VAE(sd=previous.clown.comfy.utils.load_torch_file(
        str(previous.clown.model_paths.VAE_PATH), safe_load=True
    ))
    prepared = {}
    for case, config in previous.CASES.items():
        pixels = previous.clown.load_pixels(Path(config["source"]))
        with torch.no_grad():
            latent = vae.encode(pixels)
            source_decode = vae.decode(latent).detach().cpu()
        prepared[case] = {"latent": latent.detach().clone(), "source_decode": source_decode}
        previous.clown.metrics_base.save_pixels(source_decode, OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png")
    del vae
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()

    model = previous.clown.comfy.sd.load_diffusion_model(str(previous.clown.model_paths.MODEL_PATH), model_options={})
    clip = previous.clown.comfy.sd.load_clip(
        [str(previous.clown.model_paths.TEXT_ENCODER_PATH)], clip_type=previous.clown.comfy.sd.CLIPType.FLUX2
    )
    sigma = previous.clown.get_schedule(previous.clown.STEPS, noisy.GENERATED_TOKENS).float()[0]
    outputs = {}
    for case, config in previous.CASES.items():
        item = prepared[case]
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(config["prompt"]))
        negative = previous.clown.nodes.ConditioningZeroOut().zero_out(positive)[0]
        noise = phase2.comfy.sample.prepare_noise(item["latent"], previous.clown.SEED)
        sampler = CleanSourceStepZeroSampler(item["latent"], noise, previous.clown.SEED)
        with torch.inference_mode():
            returned = phase2.comfy.sample.sample_custom(
                model, noise.clone(), 1.0, sampler, torch.stack((sigma, torch.zeros_like(sigma))),
                positive, negative, item["latent"].clone(), disable_pbar=True, seed=previous.clown.SEED,
            ).detach().float().cpu()
        outputs[case] = {"sampler": sampler, "returned": returned}
    del clip
    model.cleanup()
    del model
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()
    gc.collect()

    vae = previous.clown.comfy.sd.VAE(sd=previous.clown.comfy.utils.load_torch_file(
        str(previous.clown.model_paths.VAE_PATH), safe_load=True
    ))
    report_cases = {}
    for case, values in outputs.items():
        sampler = values["sampler"]
        decoded = {}
        authoritative_decoded = {}
        for arm, prediction in sampler.outputs.items():
            pixels = vae.decode(prediction).detach().cpu()
            decoded[arm] = pixels
            previous.clown.metrics_base.save_pixels(pixels, OUTPUT / f"{case}_{arm}_RAW_X0.png")
            edit = noisy.editable_mask(prediction).expand_as(prediction)
            authoritative = torch.where(edit, prediction, sampler.clean_source)
            authoritative_pixels = vae.decode(authoritative).detach().cpu()
            authoritative_decoded[arm] = authoritative_pixels
            previous.clown.metrics_base.save_pixels(
                authoritative_pixels, OUTPUT / f"{case}_{arm}_AUTHORITATIVE_X0.png"
            )
        previous.clown.metrics_base.make_sheet([
            ("SOURCE RECONSTRUCTION", OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png"),
            ("D0 STATE RESTORATION", OUTPUT / f"{case}_D0_STATE_RESTORATION_RAW_X0.png"),
            ("D1 NOISY SOURCE CONTEXT", OUTPUT / f"{case}_D1_NOISY_SOURCE_CONTEXT_RAW_X0.png"),
            ("D2 CLEAN SOURCE CONTEXT", OUTPUT / f"{case}_D2_CLEAN_SOURCE_CONTEXT_RAW_X0.png"),
        ], OUTPUT / f"{case}_STEP0_COMPARISON.png")
        previous.clown.metrics_base.make_sheet([
            ("SOURCE RECONSTRUCTION", OUTPUT / f"{case}_SOURCE_RECONSTRUCTION.png"),
            ("D0 + SOURCE", OUTPUT / f"{case}_D0_STATE_RESTORATION_AUTHORITATIVE_X0.png"),
            ("D1 + SOURCE", OUTPUT / f"{case}_D1_NOISY_SOURCE_CONTEXT_AUTHORITATIVE_X0.png"),
            ("D2 + SOURCE", OUTPUT / f"{case}_D2_CLEAN_SOURCE_CONTEXT_AUTHORITATIVE_X0.png"),
        ], OUTPUT / f"{case}_STEP0_AUTHORITATIVE_COMPARISON.png")
        d0 = sampler.outputs["D0_STATE_RESTORATION"]
        d1 = sampler.outputs["D1_NOISY_SOURCE_CONTEXT"]
        d2 = sampler.outputs["D2_CLEAN_SOURCE_CONTEXT"]
        edit = noisy.editable_mask(d0)
        frozen_path = FROZEN_D1 / f"{case}_D1_SOURCE_CONTEXT_RAW_X0.png"
        frozen = previous.clown.load_pixels(frozen_path)
        report_cases[case] = {
            "prompt": previous.CASES[case]["prompt"],
            "editable_raw_x0_D2_vs_D0": noisy.tensor_difference(d2, d0, edit),
            "editable_raw_x0_D2_vs_D1": noisy.tensor_difference(d2, d1, edit),
            "locked_raw_x0_D2_vs_D0": noisy.tensor_difference(d2, d0, ~edit),
            "decoded": {arm: noisy.decoded_boundary_metrics(image, prepared[case]["source_decode"])
                        for arm, image in decoded.items()},
            "authoritative_decoded": {
                arm: noisy.decoded_boundary_metrics(image, prepared[case]["source_decode"])
                for arm, image in authoritative_decoded.items()
            },
            "D1_frozen_decoded_png_max_abs": float((decoded["D1_NOISY_SOURCE_CONTEXT"] - frozen).abs().max()),
            "calls": sampler.records,
        }
    del vae
    previous.clown.comfy.model_management.unload_all_models()
    previous.clown.comfy.model_management.soft_empty_cache()

    clean_probe = outputs["rigid_bridge"]["sampler"].clean_probe
    report = {
        "experiment": "Local Edit step-0 clean-source feature/KV context",
        "contract": {
            "target": "ordinary step-0 accepted input at sigma=1",
            "source_feature_input": "Comfy-processed clean source latent y",
            "source_feature_modulation": "fixed target sigma=1 timestep embedding",
            "classification": "experimental untrained feature extractor; not a same-sigma diffusion trajectory",
            "source_coordinates": "native full-canvas RoPE retained before source K capture",
            "query_policy": "only editable generated queries retain augmented attention output",
            "restored_outputs": "all text queries and locked generated queries restored to ordinary attention",
            "blocks_modified": "all 5 double and all 20 single attention blocks",
            "attention_q_by_k": [2560, 3584], "added_source_kv_tokens": 1024,
            "sampler_updates": 0,
        },
        "clean_source_capture_layout": clean_probe.capture_records,
        "clean_source_context_layout": clean_probe.context_records,
        "cases": report_cases,
        "diffusion_trajectory_executed": False,
        "production_changed": False,
    }
    previous.clown.metrics_base.atomic_json(OUTPUT / "report.json", report)
    print(json.dumps({
        "report": str(OUTPUT / "report.json"),
        "cases": {case: {
            "D2_vs_D0": value["editable_raw_x0_D2_vs_D0"],
            "D2_vs_D1": value["editable_raw_x0_D2_vs_D1"],
            "clean_vs_noise": value["calls"]["state"]["clean_source_vs_fixed_noise"],
        } for case, value in report_cases.items()},
    }, indent=2))


if __name__ == "__main__":
    run()
