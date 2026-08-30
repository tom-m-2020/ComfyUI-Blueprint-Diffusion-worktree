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
- Phase 2e four-step Euler trajectory comparing dense, tiled-only, and fresh
  per-evaluation compact-global-context tiled sampling with strict K/V lifetime
  and one-update assertions.
- Phase 2f no-update late-state diagnostic comparing local-only, 512-token
  compact K/V, and 2048-token dense full-canvas K/V through the identical
  25-block integration path.
- Phase 2g no-update late-state discriminator comparing uniform 512-, 1152-,
  and 2048-token global K/V through the identical integration path.
- Phase 2h zero-update equal-external-token probe comparing uniform 1152 K/V
  against 512 coarse plus 640 fixed center-region K/V and dense 2048 K/V.
- Phase 2i zero-update discriminator comparing uniform and failure-local 1152
  against 512 coarse plus 640 deterministically distributed nonlocal K/V.

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
- Candidate-2 trajectory: partial semantic pass. Context stayed closer to dense
  and more overlap-consistent at all four evaluations while retaining detail,
  but one extra lighthouse-like object emerged at later steps.
- Duplicate-lighthouse discriminator: dense K/V removed the duplicate while
  compact K/V retained it and had roughly twice the dense-crop RMS error. The
  primary observed failure is compact representation loss.
- Intermediate-density discriminator: a uniform 1152-token grid improved RMS
  modestly but retained the duplicate lighthouse and stone structure. It is not
  a sufficient semantic threshold before dense cost in this case.
- Equal-budget multiscale discriminator: concentrating 640 native-density
  tokens at the affected center region worsened RMS and enlarged the duplicate
  lighthouse/stone failure versus uniform 1152. Naive local-density
  concatenation is insufficient.
- Distributed-nonlocal discriminator: balanced nonlocal tokens reversed the
  failure-local amplification but neither removed the duplicate nor improved
  on uniform 1152. Simple spatial reallocation is substantially weakened.
- Compute, wall-clock acceleration, and general VRAM reduction: not qualified.
- No production code or nodes were added.

## Active blockers / unknowns

- Whether the observed reduced-global control generalizes across prompts,
  seeds, sizes, and global resolutions.
- Why dense source K/V crosses the object-uniqueness threshold when uniform,
  failure-local, and distributed-nonlocal 1152-token representations do not:
  source hidden interaction, pairwise coverage, and representation composition
  remain unresolved possibilities.
- Cache validity and block-level sparse propagation for Z-Image and Anima.
- Whether periodic global refresh is sufficient after early denoising.

## Next concrete milestone

Pause density/layout sweeps. Reassess the dense-versus-concatenated source
hidden-trajectory difference before defining another falsifier; do not advance
to trajectories, caching, sparse execution, or production infrastructure.
