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
- Phase 2c one-evaluation Candidate-2 probe with same-sigma compact-global K/V
  injected into local generated-query attention across all 25 Klein blocks.
- Phase 2d one-evaluation all-crop assembly probe using one shared compact-global
  capture, unchanged Phase-2 overlap weights, per-crop/full metrics, and decoded
  diagnostics.

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
- Candidate-2 semantic premise: positive for one high-noise left-crop
  evaluation. Compact context removed an invented tower and moved bridge
  geometry toward dense while retaining local structure.
- Candidate-2 all-crop premise: strong positive first-sigma evidence. All three
  crops moved toward dense and aggregate pre-blend overlap RMS fell from 0.849
  to 0.312; trajectory persistence remains unknown.
- Compute, wall-clock acceleration, and general VRAM reduction: not qualified.
- No production code or nodes were added.

## Active blockers / unknowns

- Whether the observed reduced-global control generalizes across prompts,
  seeds, sizes, and global resolutions.
- Whether fresh same-sigma compact context remains authoritative through actual
  sampler updates and later steps where train identity/detail becomes visible.
- Cache validity and block-level sparse propagation for Z-Image and Anima.
- Whether periodic global refresh is sufficient after early denoising.

## Next concrete milestone

Run a minimal four-step Euler comparison of dense, tiled-only, and all-crop
compact-global-context assembly, refreshing global K/V from the current latent
at every evaluation. Do not add stale caching, block selection, production
nodes, or generalized infrastructure.
