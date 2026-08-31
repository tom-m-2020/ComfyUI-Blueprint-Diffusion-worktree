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

## 2026-08-30 — Candidate-2 all-crop one-evaluation assembly probe

Detailed evidence is in `experiments/FLUX2_CANDIDATE2_ALL_CROP_REPORT.md`
and `experiments/flux2_candidate2_all_crop_results/report.json`.

### Runtime evidence

- One same-sigma compact-global K/V capture was shared unchanged across all
  three Phase-2 crops. Each crop used compact context in all 25 FLUX.2 blocks;
  assembly used the unchanged normalized overlap weights and no output-space
  global fusion.
- Every crop moved toward its dense reference. RMS errors changed from
  0.867/0.941/0.679 tiled-only to 0.582/0.600/0.565 with context. Corresponding
  low-frequency RMS errors changed from 0.657/0.711/0.464 to
  0.396/0.433/0.379.
- Full assembled RMS versus dense fell from 0.758 to 0.569 and low-frequency
  RMS from 0.547 to 0.386. Prediction RMS stayed stable and coverage was
  exactly one everywhere.
- Pre-blend aggregate overlap disagreement fell from 0.849 RMS tiled-only to
  0.312 with context. The 0–1 and 1–2 overlap pairs both improved materially,
  so normalized blending is not hiding the measured consistency gain.
- Visually, context removed crop-local dark supports/towers from the left and
  center crops, retained one left lighthouse and one right stone tower, and
  produced a continuous bridge deck/cable organization and water horizon much
  closer to dense. A distinct centered train was not interpretable in any
  first-sigma decoded x0, so train count/placement remains unqualified.

### Conclusion

- Candidate 2 has strong positive evidence at one evaluation across all three
  regions. The next unresolved semantic question is trajectory persistence,
  not whether compact global context can influence multiple local predictions.

## 2026-08-30 — Candidate-2 minimal four-step trajectory

Detailed evidence is in `experiments/FLUX2_CANDIDATE2_FOUR_STEP_REPORT.md`
and `experiments/flux2_candidate2_four_step_results/report.json`.

### Runtime evidence

- Dense, three-crop tiled-only, and fresh compact-global-context tiled variants
  completed identical four-step Euler schedules with four accepted updates.
  Every context evaluation captured 25 fresh same-current-latent/sigma K/V
  layers once, shared the identical objects across all three crops, cleared
  them after assembly, and consumed exactly one assembled prediction in one
  update. Weak-reference checks found no prior-evaluation K/V survival.
- Context prediction RMS versus the dense trajectory was lower than tiled-only
  at every evaluation: 0.569/0.601/0.643/0.652 versus
  0.758/0.793/0.840/0.864. Low-frequency error was also lower at every step.
- Pre-blend overlap RMS stayed lower throughout: context
  0.312/0.382/0.291/0.181 versus tiled 0.849/0.734/0.548/0.304. Every crop was
  closer to its dense reference region at every evaluation.
- The context final retained high-detail bridge cables/truss, one continuous
  long train, water, lighthouse, and masonry detail while suppressing tiled
  output's incompatible bridge spans, extra central support, and separated
  train segments. Compact context did not collapse local detail.
- A second small lighthouse-like object emerged near the center-right during
  later context evaluations and remained in the final. Candidate 2 therefore
  reduced but did not eliminate semantic duplication.
- Work remained 1/3/4 forwards and 2048/3072/3584 generated image tokens per
  evaluation for dense/tiled/context. The unoptimized all-layer K/V retention
  peaked near 3.77 GiB and is not an efficiency result.

### Conclusion

- Candidate 2 partially passes the trajectory semantic gate. Fresh compact
  context remains beneficial across the schedule and permits detail, but object
  uniqueness is not yet reliable.

## 2026-08-30 — Late-state dense-K/V versus compact-K/V diagnostic

Detailed evidence is in `experiments/FLUX2_DENSE_VS_COMPACT_KV_REPORT.md`
and `experiments/flux2_candidate2_dense_vs_compact_results/report.json`.

### Runtime evidence

- The Phase-2e accepted context latent before evaluation 2 was reproduced after
  exactly two Euler updates with matching GPU-side statistics at sigma
  0.8925944. The diagnostic performed zero updates and returned the accepted
  latent bit-exactly (`max_abs=0`).
- On affected center crop 1, compact and dense global context used the identical
  25-block external-K/V integration, local query/MLP/residual path, text output
  restoration, and final projection. Only global source/density and its
  corresponding full-canvas RoPE differed: 512 versus 2048 generated K/V and
  per-block `1536x2048` versus `1536x3584` Q-by-K attention.
- RMS versus the dense crop was 0.416 local-only, 0.372 compact K/V, and 0.180
  dense K/V. Low-frequency RMS was 0.228, 0.188, and 0.099 respectively.
  Prediction RMS remained stable.
- Compact K/V retained the extra small lighthouse beside a dark stone
  structure. Dense K/V removed the lighthouse and substantially improved train,
  bridge, cable, horizon, and water alignment without suppressing detail.
- Dense K/V still retained a reduced dark stone tower absent from the dense
  crop except as a tiny distant silhouette. External K/V is therefore not exact
  even when global information is dense.

### Conclusion

- The trajectory's duplicate-lighthouse failure is evidence of compact
  representation loss, not a general failure of the external-K/V mechanism.
  Global representation quality/density becomes the next research variable,
  while residual dense-context divergence remains an explicit limitation.

## 2026-08-30 — Intermediate global-density discriminator

Detailed evidence is in `experiments/FLUX2_INTERMEDIATE_DENSITY_REPORT.md`
and `experiments/flux2_candidate2_intermediate_density_results/report.json`.

### Runtime evidence

- The exact Phase-2f late accepted state and center crop were reproduced with
  matching GPU-side statistics. The three-way diagnostic performed zero
  sampler updates and changed only global grid density and its whole-canvas
  RoPE mapping through the same all-25-block external-K/V path.
- For 512/1152/2048 global tokens, local attention dimensions were respectively
  `1536x2048`, `1536x2688`, and `1536x3584`. Changed QK products were
  1.0x/1.3125x/1.75x relative to 512 tokens; these are not total-model FLOPs.
- RMS versus dense crop was 0.372/0.328/0.180 and low-frequency RMS was
  0.188/0.162/0.099. Prediction RMS remained stable.
- The 1152-token branch modestly improved train, bridge, and horizon agreement,
  but retained the same extra center-right lighthouse and dark stone structure
  as the 512-token branch. Dense K/V removed the lighthouse, though it still
  retained a reduced stone structure and was not identical to dense execution.

### Conclusion

- A uniform `24x48` intermediate grid is insufficient for the target semantic
  failure. It closes only a minority of the compact-to-dense numerical gap and
  does not reach the duplicate-removal threshold. This does not prove that all
  1152-token representations are insufficient.

## 2026-08-30 — Equal-budget multiscale global-context probe

Detailed evidence is in `experiments/FLUX2_EQUAL_BUDGET_MULTISCALE_REPORT.md`
and `experiments/flux2_candidate2_equal_budget_multiscale_results/report.json`.

### Runtime evidence

- The exact Phase-2f/2g state and center crop were reproduced. The diagnostic
  performed zero updates and held the local all-25-block integration fixed.
- Uniform and multiscale variants both supplied 1152 external global tokens
  and used `1536x2688` local Q-by-K attention. Multiscale concatenated 512
  unchanged coarse whole-canvas tokens with 640 native-density tokens from
  `y=8..27, x=24..55`; overlapping coordinates were not deduplicated.
- RMS versus dense was 0.328 uniform, 0.398 multiscale, and 0.180 dense.
  Low-frequency RMS was 0.162, 0.208, and 0.099 respectively. Multiscale was
  therefore 21.5% worse in total RMS and 27.9% worse in low-frequency RMS than
  the equal-token uniform control.
- Visually, multiscale enlarged the duplicate center-right lighthouse/stone
  complex and worsened train and nearby bridge alignment. Dense context removed
  the lighthouse. Prediction RMS remained stable, excluding gross collapse.
- Equal external-token and local-attention budgets do not imply equal total
  compute: multiscale used two source captures and duplicated text/forward
  overhead, whereas uniform and dense used one capture.

### Conclusion

- The missing evidence is not recovered by concentrating the spare token budget
  at the visible failure location using naive concatenated K/V. Local-region
  density can reinforce the incompatible local semantic alternative. Whether
  distributed nonlocal native-density evidence is sufficient remains unknown.

## 2026-08-30 — Distributed nonlocal evidence discriminator

Detailed evidence is in `experiments/FLUX2_DISTRIBUTED_NONLOCAL_REPORT.md`
and `experiments/flux2_candidate2_distributed_nonlocal_results/report.json`.

### Runtime evidence

- The exact late state reproduced with zero diagnostic updates. Uniform,
  failure-local, and nonlocal variants each supplied 1152 external generated
  K/V and used identical `1536x2688` local attention in all 25 blocks.
- The content-blind nonlocal selector excluded center crop `x=24..55`, then
  sampled 480 of 768 native positions from the left strip and 160 of 256 from
  the right using evenly spaced row-major ranks. With 512 unchanged coarse
  tokens, nonlocal contained 1152 tokens, 1150 unique RoPE positions, and two
  retained exact coordinate duplicates.
- RMS versus dense was 0.328 uniform, 0.398 failure-local, 0.330 nonlocal, and
  0.180 dense. Low-frequency RMS was 0.162, 0.208, 0.165, and 0.099.
  Prediction norms remained stable.
- Nonlocal avoided failure-local's enlarged lighthouse/stone/landmass failure,
  but retained the original duplicate lighthouse and stone structure at roughly
  uniform-control severity. It did not materially improve train, bridge, or
  horizon semantics over uniform 1152.
- Nonlocal capture is not an efficiency result: it used separate coarse, left,
  and right source forwards and selected K/V only afterward.

### Conclusion

- Simple distributed nonlocal spatial reallocation is not supported as the
  missing semantic mechanism at this token budget. It is preferable to
  failure-local concentration but does not cross the dense-context semantic
  threshold or beat uniform 1152.

## 2026-08-30 — Dense-source interaction ablation

Detailed evidence is in
`experiments/FLUX2_DENSE_SOURCE_INTERACTION_ABLATION_REPORT.md` and
`experiments/flux2_candidate2_dense_source_interaction_ablation_results/report.json`.

### Runtime evidence

- At the exact late state, both variants retained all 2048 native source image
  positions, ordinary source projections/MLPs/residuals, `2560x2560` source
  Q-by-K geometry, and identical `1536x3584` local external-K/V attention.
- The restricted source masked only image-query to image-key edges in all 25
  FLUX.2 blocks. Text queries retained all text/image keys and image queries
  retained all text keys. Per block, 4,194,304 image-image edges were blocked.
- Captured source K/V was identical at double block 0, before any ablated
  mixing could affect hidden states, then diverged from double block 1 onward.
  This confirms that source hidden-trajectory interaction was the changed path.
- RMS versus the ordinary dense crop increased from 0.180 with ordinary dense
  source K/V to 0.252 with restricted source K/V. Low-frequency RMS increased
  from 0.099 to 0.134; prediction RMS remained stable.
- The extra center-right lighthouse suppressed by ordinary dense source K/V
  reappeared under the restriction, alongside regressions in train, bridge, and
  horizon alignment.

### Conclusion

- The successful dense context depends on global image-token interaction while
  constructing source hidden features, not only on exposing 2048 native
  positions at the final external-K/V interface. The experiment does not show
  which blocks or fraction of global interaction is sufficient.

## 2026-08-30 — Source interaction range discriminator

Detailed evidence is in `experiments/FLUX2_SOURCE_INTERACTION_RANGE_REPORT.md`
and `experiments/flux2_candidate2_source_interaction_range_results/report.json`.

### Runtime evidence

- All variants retained 2048 source image tokens, native coordinates,
  `2560x2560` source attention tensors, and identical `1536x3584` local
  external-K/V attention. The diagnostic performed zero updates.
- The windowed source used eight fixed nonoverlapping `16x16` windows in all 25
  blocks. Image queries retained all text keys and 256 same-window image keys;
  text queries retained ordinary connectivity. This preserves 524,288 of
  4,194,304 dense image-image edges, or 12.5%.
- RMS versus the ordinary dense crop was 0.180 dense, 0.249 windowed, and 0.252
  with no image-image source interaction. Low-frequency RMS was 0.099, 0.132,
  and 0.134. Windowing improved only about 1.3% over no mixing.
- Block-0 K/V remained exactly equal across variants. Later windowed K/V stayed
  closer to ordinary than no-mixing K/V, confirming meaningful within-window
  interaction, but the duplicate lighthouse returned under both restrictions.
- Windowed source execution retained globally plausible decoded structure and
  modestly better source-level numerical agreement, yet did not preserve train,
  bridge, stone-structure, or horizon semantics at the local output.

### Conclusion

- Independent 16x16 source windows are insufficient for the observed semantic
  constraint. Candidate 2 needs cross-window information propagation or larger-
  range interaction; direct dense pairwise attention is not yet proven necessary.

## 2026-08-30 — Cross-window propagation discriminator

Detailed evidence is in `experiments/FLUX2_CROSS_WINDOW_PROPAGATION_REPORT.md`
and `experiments/flux2_candidate2_cross_window_propagation_results/report.json`.

### Runtime evidence

- Alternating source blocks used unshifted `16x16` windows at even execution
  ordinals and non-wrapping clipped windows shifted by `(8,8)` at odd ordinals.
  Unshifted/shifted blocks retained 12.5%/8.20% of dense image-image edges.
- Analytical dependency propagation reached the full canvas: some tokens could
  depend on all 2048 image positions after ordinal 3, and all tokens could do so
  by the final source block. Source/local tensor dimensions remained unchanged.
- RMS versus dense was 0.180 ordinary, 0.249 fixed, 0.261 alternating shifted,
  and 0.252 no-mixing. Low-frequency RMS was 0.099, 0.132, 0.138, and 0.134.
  Shifted propagation was worse than both restricted controls.
- Fixed and alternating captures matched through ordinal 1 and first diverged
  at ordinal 2, exactly after the first shifted attention result. The partition
  schedule—not input/capture geometry—caused the changed hidden trajectory.
- The duplicate lighthouse, dark stone structure, and train/bridge divergence
  remained under alternating windows despite final full-canvas reachability.

### Conclusion

- Transitive full-canvas graph reachability across depth is insufficient for the
  observed dense-source semantic constraint. This topology does not replace
  broad per-layer interaction; fully dense attention is still not proven to be
  the only sufficient mechanism.

## 2026-08-30 — Dense-block necessity discriminator

Detailed evidence is in `experiments/FLUX2_DENSE_BLOCK_NECESSITY_REPORT.md`
and `experiments/flux2_candidate2_dense_block_necessity_results/report.json`.

### Runtime evidence

- Early dense used source ordinals 0–4 (all five double blocks); late dense used
  ordinals 20–24 (single blocks 15–19). Each retained 20% of the all-dense
  image-image edge budget with unchanged source/local token dimensions.
- RMS versus dense was 0.180 all-dense, 0.227 early-dense, 0.247 late-dense,
  and 0.252 no-mixing. Low-frequency RMS was 0.099, 0.124, 0.127, and 0.134.
  Early dense recovers materially more numerical/geometric benefit than late.
- Early-dense K/V tracked all-dense exactly through capture ordinal 5 and first
  diverged at 6, after the first restricted source result. Late-dense tracked
  no-mixing through capture ordinal 20 and first diverged at 21, after its first
  dense source result.
- Neither partial range suppressed the duplicate lighthouse. Early dense
  improved train and bridge organization but retained the lighthouse/stone pair;
  late dense remained much closer to the restricted semantic result.

### Conclusion

- Five early dense blocks are more useful than five late dense blocks, but both
  are insufficient for object uniqueness. Early broad interaction alone is not
  preserved through a long restricted tail at the final K/V interface.

## 2026-08-30 — Sparse dense-refresh maintenance probe

Detailed evidence is in `experiments/FLUX2_SPARSE_DENSE_REFRESH_REPORT.md`
and `experiments/flux2_candidate2_sparse_dense_refresh_results/report.json`.

### Runtime evidence

- The fixed refresh schedule used dense source ordinals
  `0,1,2,3,4,10,15,20`: eight blocks and 32% of the all-dense image-image edge
  budget. All tensor geometry and the local external-K/V path remained fixed.
- RMS versus dense was 0.180 all-dense, 0.227 early-only, 0.232 periodic refresh,
  and 0.252 no-mixing. Low-frequency RMS was 0.099, 0.124, 0.126, and 0.134.
  Refresh was slightly worse than early-only despite three additional dense
  blocks.
- Refresh ordinal 10 immediately increased K/V divergence from all-dense.
  Refreshes 15 and 20 briefly reduced K divergence at the following capture,
  but the gains disappeared and reversed before the next refresh/final capture.
- The duplicate lighthouse, dark stone structure, and horizon failure remained
  at early-only severity; train and bridge organization did not materially
  improve.

### Conclusion

- Early global planning plus isolated dense refreshes at 10/15/20 is
  insufficient. Dense events can briefly correct feature divergence, but the
  no-image-image blocks do not maintain that correction through source depth.

## 2026-08-31 — Candidate-3 coupled-trajectory design contract

Detailed design is in
`experiments/CANDIDATE3_COUPLED_TRAJECTORY_DESIGN.md`.

### Confirmed boundary and architectural inference

- Candidate 3 requires two persistent accepted states at the same sigma: a
  low-density whole-canvas latent `G` and a high-resolution full-canvas latent
  `H`. A resized global prediction fused into one high-resolution prediction is
  still Candidate 1, not a coupled trajectory.
- The smallest exact accepted-state invariant for the Phase-2 2:1 spatial
  grids is `D(H_i) = G_i`, where `D` is fixed `2x2` area restriction and `U`
  is its nearest-neighbor right inverse.
- One synchronized Euler interval can independently propose `G*` from one
  normal 512-token global forward and `H*` from one assembly of the three
  1024-token crop forwards, then accept
  `G_next = G*` and `H_next = H* + U(G* - D(H*))` exactly once.
- This rule has causal `G -> H` state flow and no `H -> G` flow. It cannot be
  reduced to two independent trajectories because every accepted `H` is a
  direct function of the independently evolved `G` and must satisfy the
  cross-state invariant.
- Mapped initialization `G_0 = D(H_0)` preserves spatial identity and exact
  initial consistency but reduces the global noise variance (fourfold for
  ideal independent samples under nonoverlapping `2x2` area restriction).
  The global branch must therefore be qualified independently in the probe; a
  malformed global-only trajectory makes the coupling result inconclusive.
- Fixed spatial scale separation is not assumed to equal semantic/detail
  separation. Prior Phase-2 evidence predicts the key failure mode: duplicate
  objects may remain in the nullspace of `D` even when coarse state consistency
  is exact.

### Compute inference

- With complete three-crop coverage, Candidate 3 executes 3584 generated-token
  positions per evaluation versus 2048 for dense (`1.75x` token-linear work).
  Its approximate image-image attention matrix work is 81.25% of dense, but
  this does not establish total-FLOP, VRAM, or wall-clock savings.
- Candidate 3's distinct potential advantage is semantic: the 512-token branch
  performs an ordinary complete globally interacting trajectory instead of
  approximating dense high-resolution hidden interaction through compact
  external K/V. Efficiency would require later selective local coverage or
  evaluation reduction and remains unqualified.

### Selected falsifier

- Test one hard accepted-state global anchor over the existing four-step FLUX.2
  setup, with dense, tiled-only, and uncoupled-dual controls. Do not add a
  coupling-strength sweep or bidirectional consensus until this strongest,
  parameter-free rule establishes that state coupling controls composition
  without reducing `H` to an upscaled global result.
