# Findings

## 2026-08-29 — Phase 1 architecture and feasibility audit

Evidence and detailed reasoning are in `BLUEPRINT_ARCHITECTURE_AUDIT.md`.

### Confirmed current ComfyUI behavior

- Native sampling retains one full latent through guider/model prediction and
  sampler update. Model wrappers can coordinate an evaluation, but post-CFG or
  output masking does not reduce diffusion-model compute.
- Dense spatial cost begins at patch/input projection. Image QKV/projections,
  MLPs, normalization, modulation, residual arithmetic, and final projection
  scale with image-token count in addition to attention.
- Cache/refresh state cannot use raw model-call count as accepted-step identity;
  samplers such as Heun can perform multiple evaluations for one update.

### Confirmed model-source behavior

- FLUX.2 uses explicit generated/reference spatial IDs, double-stream then
  single-stream joint processing, and per-block `patches_replace["dit"]`.
- Z-Image uses a Lumina-derived single joint stream with context/noise refiners,
  token padding, 3-axis RoPE, and optional latent/SigLIP reference tokens. It
  requires a separate executor adapter rather than FLUX-shaped blocks.
- Anima uses image self-attention, separate text cross-attention, MLPs, 3-D
  RoPE, and an Anima-specific text adapter. It also requires a family adapter.
- Source classification supports a common sampler/global-local abstraction for
  all three families. Z-Image and Anima selected-token execution remain runtime
  unqualified.

### External research findings

- Mixture of Diffusers and MultiDiffusion preserve a shared canvas update and
  fuse crop predictions, but overlap/shared optimization is not whole-scene
  model context.
- ElasticDiffusion supplies evidence for low-resolution global plus local signal
  separation in Stable Diffusion U-Nets. Its conditional-minus-unconditional
  decomposition is not established for CFG-1 distilled joint DiTs.
- DyPE addresses dense high-resolution positional extrapolation and is
  complementary; it does not reduce target-canvas model compute or VRAM.

### Architectural inference and open hypothesis

- T2I and reference/edit can plausibly share lifecycle, geometry, fusion, and
  executor interfaces while using different global-context policies.
- Reference tokens are conditioning, not the same-canvas evolving trajectory.
  Successful editing therefore does not establish T2I global planning.
- The central unproven hypothesis is that a cheap low-density whole-canvas
  prediction can remain authoritative enough to prevent local high-resolution
  regions from inventing incompatible composition.
- Ranked experiments: (1) coarse global prediction plus local residuals;
  (2) low-density global K/V plus selected local queries; (3) coupled persistent
  low/high-resolution trajectories.

### Provenance

- ComfyUI source revision:
  `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`.
- Prior native FLUX.2 SpotEdit evidence is reused only for sparse executor,
  K/V-cache, coordinate, and lifecycle feasibility; it is not treated as a T2I
  algorithm or image-quality result.

## 2026-08-29 — FLUX.2 coarse-global/local-fusion falsification

Detailed implementation, parameters, images, and evidence are in
`experiments/FLUX2_COARSE_GLOBAL_LOCAL_REPORT.md` and
`experiments/flux2_coarse_global_local_results/report.json`.

### Runtime evidence

- A controlled 1024 x 512 FLUX.2 Klein 4B W4A8, CFG-1, four-step Euler run
  compared dense, three overlapping 512 x 512 local crops, a 512 x 256 reduced
  global prediction, and global-plus-local correction strengths 0.25, 0.5,
  0.75, and 1.0. All variants and intermediate estimates were finite; assembled
  coverage was exactly complete.
- Dense produced the requested one continuous bridge, one centered train, left
  lighthouse, and right tower. Tiled-only produced incompatible repeated bridge
  spans/towers, train segments, and a second lighthouse. Reduced global-only
  preserved the requested whole-scene arrangement but was blurred/ghosted.
- Scalar fusion showed a strict tradeoff. Alpha 0.25 preserved the global plan
  but added little fidelity. Alpha 0.5 and 0.75 increasingly imported the local
  crops' alternative bridge/object layout. Alpha 1 is algebraically tiled-only
  at each assembled evaluation.
- The local-minus-global correction norm was 0.64 times the mapped-global norm
  at the first evaluation and about 0.72–1.07 later. Correction maps covered
  global geometry and object placement, not merely high-frequency detail.
- Per-step approximate image-token work was 2048 dense, 3072 tiled-only, 512
  global-only, and 3584 global-plus-local. This full-coverage falsification run
  does not demonstrate compute savings.
- Peak CUDA allocation was about 2.45 GiB global-only, 2.50 GiB tiled/fused,
  and 2.60 GiB dense in this process. The dense wall time included first-run
  warmup, so timing is not a qualified speed comparison.

### Conclusion

- The reduced global prediction credibly controlled cross-region composition
  better than tiled-only in this stress case.
- Unfiltered scalar local residuals did not recover strong local fidelity
  without replacing that composition. The residual contains competing
  low-frequency/semantic predictions.
- Candidate 1 needs revision; this result neither validates the original fusion
  nor rejects the broader reduced-global hypothesis.

## 2026-08-30 — FLUX.2 fixed-frequency correction falsifier

Detailed parameters, outputs, and limitations are in
`experiments/FLUX2_FIXED_FREQUENCY_REPORT.md` and
`experiments/flux2_fixed_frequency_results/report.json`.

### Runtime evidence

- On the identical Phase-2 run, a fixed Gaussian high-pass correction at alpha
  0.75 restrained but did not remove crop-local semantic divergence. Sigma 1
  latent token preserved much more of the reduced-global bridge silhouette and
  recovered visible train, cable, water, and masonry detail; it still produced
  a duplicate lighthouse and alternative/ghost bridge lines. Sigma 2 admitted
  a clearly incompatible second bridge organization.
- The decoded high-pass diagnostics visibly contain bridge decks/cables,
  towers, and lighthouse silhouettes. Direct residual VAE decodes are
  out-of-distribution visualizations, but the same semantic leakage appears in
  the final denoised images.
- Sigma-1 low/high correction RMS ratio fell from 1.55 at the first evaluation
  to 0.55 at the last; sigma-2 fell from 0.88 to 0.38. Increasing high-pass norm
  share did not imply semantic purity.
- Versus global-only, final low-frequency MAE was 0.311 for unfiltered alpha
  0.75, 0.095 for high-pass sigma 1, and 0.161 for high-pass sigma 2. Their
  high-frequency gains were respectively 0.309, 0.197, and 0.270. These are
  descriptive image-space diagnostics, not perceptual metrics.
- Model work was unchanged: 512 image tokens/one forward for global-only and
  3,584 tokens/four forwards for each local variant per evaluation. Filtering
  does not establish compute or VRAM savings.

### Conclusion

- Fixed output-space frequency splitting partially works as a restraint, not
  as a semantic/detail separator. Candidate 1 survives only in revised form;
  explicit global context or feature-aware fusion is now the higher-value
  direction than further scalar/filter sweeps.

## 2026-08-30 — Candidate-2 one-evaluation compact-global-context probe

Detailed implementation and evidence are in
`experiments/FLUX2_CANDIDATE2_ONE_EVAL_REPORT.md` and
`experiments/flux2_candidate2_one_eval_results/report.json`.

### Runtime evidence

- At the first Phase-2 sigma, all 25 FLUX.2 Klein blocks were instrumented so
  1024 left-crop generated queries could attend to 512 same-current-latent
  compact-global generated K/V tokens. Local and global K used their own RoPE;
  ordinary text-query attention output was restored, and no cross-step cache or
  sampler update was used.
- The ordinary left crop independently invented a large dark tower at its right
  boundary. Compact global context removed it, retained the single left
  lighthouse, moved bridge slope/deck geometry toward the dense reference, and
  retained visible cable, deck, shoreline, and water structure.
- Context-versus-dense-crop RMS was 0.582 versus 0.867 for local-only; the
  low-frequency RMS difference was 0.396 versus 0.657. Context changed the
  local prediction substantially (0.617 RMS versus local-only) without a norm
  instability: prediction RMS was 1.0725 context and 1.0731 local-only.
- Per modified block, normal local attention was 1536 Q by 1536 K including
  text. Context attention was 1536 Q by 2048 K, ordered as Q `[text, local]`
  and K/V `[text, local, compact-global]`; only generated query attention
  output retained the augmented result.
- This unoptimized probe retained all layers' compact K/V concurrently and
  recomputed ordinary attention to restore text output. Its roughly 3.76 GiB
  peak versus 2.50 GiB local-only is not a production VRAM result or evidence
  of efficiency.

### Conclusion

- Candidate 2 has positive single-evaluation semantic evidence: compact global
  context changes local prediction inside the transformer and suppressed one
  concrete crop-local duplicate-object alternative. Generalization across
  crops, sigmas, and a trajectory remains untested.
