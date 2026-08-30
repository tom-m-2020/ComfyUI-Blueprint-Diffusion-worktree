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
- Phase 2 FLUX.2 coarse-global/local-fusion falsification harness, real
  composition-scale run, intermediate visualizations, telemetry, and report.
- Phase 2b fixed-frequency correction falsifier with two fixed Gaussian scales,
  real outputs, residual visualizations, telemetry, and focused report.

## Validation status

- Source/audit conclusions: complete for the checked revisions.
- FLUX.2 sparse mechanics: supported by prior local runtime evidence, with
  provenance recorded in the audit.
- Z-Image and Anima selective execution: not runtime-qualified.
- Candidate-1 semantic premise: one controlled composition-stress case found
  credible global-plan benefit over tiled-only.
- Straight scalar local-residual fusion: falsified as a sufficient mechanism in
  that case; it trades global structure for local sharpness.
- Fixed high-pass output fusion: partially restrains compositional replacement
  and recovers detail, but semantic objects and geometry remain in the
  high-pass correction; it is not a sufficient fusion contract.
- Compute, wall-clock acceleration, and general VRAM reduction: not qualified.
- No production code or nodes were added.

## Active blockers / unknowns

- Whether the observed reduced-global control generalizes across prompts,
  seeds, sizes, and global resolutions.
- A mechanism that prevents crop-local semantic invention rather than merely
  attenuating its low spatial frequencies.
- Cache validity and block-level sparse propagation for Z-Image and Anima.
- Whether periodic global refresh is sufficient after early denoising.

## Next concrete milestone

Run the smallest one-evaluation Candidate-2 test: determine whether local
high-resolution queries consuming compact whole-canvas global context suppress
the bridge/lighthouse semantic alternatives before output projection. Do not
build a generalized executor or production nodes.
