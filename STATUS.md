# Status

## Completed

- Phase 1 research-only architecture and feasibility audit.
- Native ComfyUI sampling boundary and full latent/prediction lifecycle traced.
- FLUX.2 Klein, Z-Image Turbo, and Anima source architectures classified.
- T2I, reference/edit, inpainting, and outpainting global-context regimes
  separated.
- Ordinary tiling, Mixture of Diffusers, MultiDiffusion, ElasticDiffusion,
  DyPE, and SpotEdit compared.
- Three falsifiable candidate architectures ranked.

## Validation status

- Source/audit conclusions: complete for the checked revisions.
- FLUX.2 sparse mechanics: supported by prior local runtime evidence, with
  provenance recorded in the audit.
- Z-Image and Anima selective execution: not runtime-qualified.
- Blueprint semantic quality, compute reduction, wall-clock acceleration, and
  VRAM reduction: not yet qualified.
- No production code or nodes were added.

## Active blockers / unknowns

- Whether a low-density global prediction actually governs high-resolution
  local composition in T2I.
- A validated cross-resolution prediction-fusion rule for modern flow DiTs.
- Cache validity and block-level sparse propagation for Z-Image and Anima.
- Whether periodic global refresh is sufficient after early denoising.

## Next concrete milestone

Run one controlled FLUX.2 Klein wide-canvas experiment comparing dense,
tiled-only, low-resolution-global-only, and global-plus-local-residual
predictions at identical seed, prompt, sigma schedule, and sampler settings.
Instrument per-step token work and retain intermediate predictions. Keep all
code and outputs under `experiments/`.
