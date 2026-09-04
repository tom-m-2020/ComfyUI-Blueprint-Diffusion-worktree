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

## 2026-08-31 — Candidate-3 hard-global-anchor runtime result

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_HARD_GLOBAL_ANCHOR_REPORT.md` and
`experiments/flux2_candidate3_hard_global_anchor_results/report.json`.

### Runtime evidence

- The mapped global initialization had 0.248667 of the high-resolution noise
  variance, as expected for nonoverlapping `2x2` area restriction. Despite that
  shift, the independently evolved `G` trajectory was finite and visibly valid:
  one coherent bridge, centered train, left lighthouse, right stone tower, and
  continuous horizon/water.
- The uncoupled control's accepted `H` was bit-exact with tiled-only at all four
  steps. Uncoupled and hard-anchor `G` were also bit-exact. Every Candidate-3
  evaluation performed exactly one 512-token global plus three 1024-token local
  forwards, followed by one atomic pair acceptance; no crop updated state.
- Hard acceptance enforced `D(H_next) = G_next` with maximum error
  `2.38419e-7` at every step under the documented endpoint-spanning global RoPE
  mapping (`scale_y=31/15`, `scale_x=63/31`).
- Projection RMS grew monotonically from 0.02597 to 0.34933, and from 2.68% to
  38.37% of proposed-H RMS. Global/local coarse update cosines remained positive
  at 0.74–0.86, while consecutive projection cosines rose to 0.77. The branches
  agree broadly in direction but retain an increasing, persistent state-scale
  mismatch rather than converging.
- Hard anchoring reduced final latent RMS versus dense from 0.8639 to 0.8089 and
  low-frequency RMS from 0.5016 to 0.4767. Late pre-blend overlap disagreement
  also fell relative to tiled-only.
- The final local proposal before terminal projection followed the global
  single-bridge plan and contained useful high-resolution detail. The terminal
  exact projection visibly doubled/smeared cables, deck, train, supports,
  lighthouse, and tower structure.

### Conclusion

- Persistent low-density state can causally organize later crop predictions in
  this case, so coarse state coupling has semantic value beyond independent
  trajectories. Exact hard coarse equality after every proposal overconstrains
  local detail, especially across the terminal Euler interval. This rejects the
  tested hard-acceptance rule, not Candidate 3 as a class.

## 2026-08-31 — Candidate-3 terminal-release lifecycle result

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_RELEASE_REPORT.md` and
`experiments/flux2_candidate3_terminal_release_results/report.json`.

### Runtime evidence

- Hard-anchor and terminal-release `H` states, local predictions, and Euler
  proposals were bit-exact through the terminal proposal. Their accepted `G`
  states were bit-exact throughout, and accepted `H` differed only at final
  acceptance. The lifecycle intervention is therefore isolated.
- The unchanged terminal local proposal preserved the prior coupled
  trajectory's single bridge, centered train, left lighthouse, right stone
  tower, and horizon/water continuity. Omitting the final projection did not
  restore tiled-only semantic duplication.
- Terminal release reduced final RMS versus dense from 0.808925 to 0.727491 and
  low-frequency RMS from 0.476675 to 0.381146. Terminal overlap disagreement
  remained exactly 0.260869 because D/E used the same crop predictions.
- The released terminal coarse mismatch had RMS 0.349332 and max 3.423378,
  exactly the correction applied by the hard control. Keeping that mismatch
  removed most cable, train, deck, support, lighthouse, and tower ghosting.

### Conclusion

- In this case, hard coarse consistency is useful as an intermediate-state
  organizer but harmful as a terminal latent constraint. The local final model
  proposal can remain authoritative without immediately abandoning the global
  scene plan. This is positive Candidate-3 evidence, not yet a general or
  production-qualified policy.

## 2026-08-31 — Candidate-3 terminal-release generalization result

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_RELEASE_GENERALIZATION_REPORT.md` and
`experiments/flux2_candidate3_terminal_release_generalization_results/report.json`.

### Runtime evidence

- With only prompt and seed changed, every lifecycle control passed: C.H was
  bit-exact with B.H, D/E were bit-exact through terminal proposal, all
  intermediate invariants held, and only final H acceptance differed.
- Terminal release reduced final RMS versus dense to 0.676254, from 0.755579
  for hard anchoring and 0.756690 for tiled-only. Low-frequency RMS fell to
  0.354860 from 0.449037 and 0.444719 respectively.
- E preserved one centered person, the dominant left red car, the dominant
  right tree, coherent perspective, and continuous ground while removing most
  D terminal ghosting. It did not restore tiled-only's second person or second
  competing large tree.
- E retained one small extra background car and two thin extra trees. These
  were already present in D's identical terminal proposal and survived D's
  projection, so terminal release did not cause the semantic extras.

### Conclusion

- Terminal release's detail-preserving lifecycle effect generalizes to a second
  non-architectural scene, but fixed intermediate coarse consistency does not
  guarantee exact secondary-object uniqueness. Candidate 3 receives partial
  generalization evidence and remains unqualified for production.

## 2026-08-31 — Candidate-3 higher-bandwidth geometry selection

Detailed analysis is in
`experiments/CANDIDATE3_HIGHER_BANDWIDTH_GEOMETRY_NOTE.md`.

### Mathematical result

- A `24x48` global state admits a clean local right-inverse pair by partitioning
  H into `4x4` blocks, retaining each block's lowest `3x3` orthonormal DCT
  coefficients, and reconstructing a spatial `3x3` G block with scale `3/4`.
  U performs the inverse `3x3` DCT, zero-pads to `4x4`, synthesizes with the
  `4x4` basis, and scales by `4/3`.
- `D(U(G)) = G` for every G in exact arithmetic. The construction is the direct
  higher-bandwidth generalization of current `2x2 mean / nearest`, which is the
  same block-DCT pair retaining only DC.
- The 1152-token state is 2.25x current G and 56.25% of dense. With unchanged
  local coverage, generated-token work rises 17.86% and approximate image-image
  attention matrix work rises 31.25% versus current Candidate 3.
- The operator preserves aspect ratio and equal per-axis sampling. Its main new
  assumptions are a separable rectangular passband, even-reflected local DCT
  boundaries, and possible `4x4` macroblock artifacts.

## 2026-08-31 — Candidate-3 higher-bandwidth runtime result

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_HIGHER_BANDWIDTH_REPORT.md` and
`experiments/flux2_candidate3_higher_bandwidth_results/report.json`.

### Runtime evidence

- Synthetic `D(U(G))` maximum error was `7.15e-7` before model load. Every
  nonterminal coupled acceptance remained within `9.54e-7` maximum error.
  D/E lifecycle isolation and C.H/B.H bit-exactness also passed.
- Mapped 24x48 G noise variance was 0.563112 of H, close to the predicted
  0.5625 and materially different from Phase 3c's approximately 0.25. The
  decoded global branch was independently coherent and unique.
- Terminal-release 24x48 removed the Phase-3c small second car and two thin
  extra trees while preserving one centered person, one left car, one right
  tree, perspective, ground continuity, and sharp local detail.
- Final RMS versus dense improved from 0.676254 at 16x32 to 0.549170 at 24x48;
  low-frequency RMS improved from 0.354860 to 0.245447. Terminal overlap RMS
  was 0.242493 versus tiled-only 0.261909.
- No decoded 4x4/3x3 block pattern was visible in sky, ground, horizon, tree
  boundaries, or large gradients. The hard terminal projection still ghosted
  details, so terminal release remains necessary.
- E runtime increased from 3.898 to 4.419 seconds and peak CUDA allocation from
  2.4987 to 2.5135 GiB versus the 16x32 run. These are run-specific telemetry,
  not broad performance claims.

### Conclusion

- The selected 24x48 block-DCT geometry resolves the known secondary
  duplication under the complete operator/noise contract and becomes the
  leading Candidate-3 research geometry. The result does not isolate bandwidth
  from initialization variance and does not authorize production.

## 2026-08-31 — Candidate-3 mapped-noise-variance discriminator

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_MAPPED_VARIANCE_REPORT.md` and
`experiments/flux2_candidate3_mapped_variance_results/report.json`.

### Runtime evidence

- The scaled 16x32 pair achieved synthetic right-inverse maximum error
  `2.38e-7` and measured G/H noise variance 0.563771, closely matching 24x48
  block-DCT's 0.563112. All lifecycle and nonterminal invariants passed.
- Variance matching improved final RMS versus dense from 0.676254 to 0.645287
  and low-frequency RMS from 0.354860 to 0.313597, showing that mapped-state
  scale/statistics contribute to Phase 3d's gain.
- The scaled 16x32 result retained one thin extra tree and a small ambiguous
  vehicle-like horizon remnant. It improved but did not match 24x48's exact
  person/car/tree uniqueness. Terminal overlap also worsened from 0.248414 to
  0.288335, whereas 24x48 achieved 0.242493.
- All three global branches decoded valid unique scene plans. Residual extras
  emerge in coupled H rather than from an invalid global trajectory.

### Conclusion

- Both mapped-state statistics and spatial/operator bandwidth contribute. The
  matched-variance control leaves strong evidence for additional 24x48
  bandwidth or block-DCT content because only that branch fully removes
  secondary duplication and improves overlap. Token density versus DCT content
  remains intentionally unresolved.

## 2026-08-31 — Candidate-3 production boundary audit

Detailed design is in `CANDIDATE3_PRODUCTION_ARCHITECTURE.md`.

- Candidate 3 must own sampler-interval acceptance, not merely model
  prediction. ComfyUI's ordinary Euler loop invokes its callback before the
  Euler assignment and then unconditionally assigns one `x`; neither a model
  patch nor that callback can publish an atomic persistent `(G,H)` pair.
- ComfyUI's existing guider lifecycle is reusable. `CFGGuider.inner_sample`
  prepares conditioning/model options and invokes the selected `Sampler.sample`;
  `SamplerCustomAdvanced` already accepts an arbitrary `SAMPLER`, guider,
  noise, sigmas, and latent. A custom Candidate-3 `SAMPLER` can therefore own
  the two-state Euler loop without forking ComfyUI or bypassing its guider.
- A `SAMPLER_SAMPLE` wrapper could replace the sampler body from a patched
  model, but it would make an ordinary sampler selection misleading and hide
  the fail-closed Euler restriction. The cleaner first UI boundary is an
  explicit Blueprint Euler sampler node, not `MODEL -> patched MODEL`.
- Generic production ownership separates interval coordination, immutable
  accepted state, D/U geometry, crop planning/assembly, terminal policy, and a
  model-family adapter. Only the adapter translates generic canvas/region
  coordinates into FLUX.2 RoPE/model options.
- The fixed 24x48 block-DCT contract executes 4224 generated image tokens per
  accepted interval (1152 global plus 3 x 1024 local). This remains a semantic
  architecture selection; compute, acceleration, and VRAM advantages are not
  yet qualified.

## 2026-08-31 — Candidate-3 first production slice equivalence

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_PRODUCTION_EQUIVALENCE_REPORT.md` and
`experiments/flux2_candidate3_production_equivalence_results/report.json`.

- The explicit production `SAMPLER` path is bit-exact with the existing
  Phase-3 experiment path at initial H/G, every global and assembled-local
  denoised prediction, every Euler proposal, every accepted G/H, and final H.
  All compared RMS and maximum-absolute differences are zero.
- The production lifecycle records four atomic accepted intervals. Each has
  exactly one 1152-token global and three 1024-token local guider predictions;
  terminal release occurs only on interval 3.
- Startup block-DCT right-inverse error was `7.15e-7`. Nonterminal accepted
  invariant errors were at most `7.15e-7`, below the fixed `2e-6` tolerance.
- Normal production telemetry contains only scalar and shape summaries. The
  equivalence harness alone injects a capture sink that detaches full tensors
  to CPU for boundary comparison.
- Sequential wall times are not a performance comparison because the research
  path ran first and paid cold-run costs. This result qualifies control/tensor
  equivalence only, not acceleration or VRAM reduction.

## 2026-08-31 — Candidate-3 live ComfyUI workflow qualification

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_LIVE_WORKFLOW_REPORT.md` and
`experiments/live_comfyui_candidate3_results/report.json`.

- The production package registers from ComfyUI's ordinary `custom_nodes`
  directory as `Blueprint Candidate-3 Euler Sampler`, outputs a normal
  `SAMPLER`, and executes through `BasicGuider`, `SamplerCustomAdvanced`, and
  the normal VAE decode/save path.
- Three valid queue executions, including one after `/free` model/cache unload
  and one after invalid runs plus cancellation, each emitted four websocket
  previews and produced the same decoded RGB pixel hash.
- Cancellation after the first interval produced `execution_interrupted` and
  no final output. The next valid execution completed identically, providing
  live evidence that per-invocation `(G,H)` state does not poison later runs.
- Wrong resolution, CFG, model family, masks, nonempty input, partial denoise,
  sigma count, schedule orientation/termination, and spatial conditioning all
  failed with targeted errors and zero preview frames.
- Live partial-denoise testing showed that ComfyUI's inherited `max_denoise`
  heuristic is not a sufficiently strict fixed-slice guard. Requiring
  `sigma[0] == 1.0` is the explicit supported-schedule boundary; this changes
  validation only, not the qualified algorithm.

## 2026-08-31 — Candidate-3 arbitrary target geometry

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_ARBITRARY_GEOMETRY_REPORT.md` and
`experiments/live_comfyui_candidate3_geometry_results/report.json`.

- The block-DCT 4-to-3 operator generalizes without changing its mathematics:
  any positive H axes divisible by four map to G axes equal to three quarters
  of H. Right-inverse and constant preservation remain within `2e-6` across
  tested rectangular and non-crop-aligned block grids.
- Ordinary local model calls impose the separate practical minimum of 32 latent
  positions per axis. Smaller grids are rejected instead of padded or assigned
  out-of-canvas coordinates.
- Fixed 32x32 crops at stride 24 with an end-aligned final start reproduce the
  original 32x64 three-crop plan and give deterministic normalized full
  coverage for portrait, square, wide, and two-dimensional crop grids.
- FLUX.2 global RoPE uses endpoint-preserving per-axis scales
  `(H-1)/(G-1)`; local crops use their absolute latent-grid Y/X offsets.
- Real four-step ComfyUI runs succeeded at 512x512, 512x1024, and 1280x512,
  each with four previews, normal decode/save, and all runtime coarse-state
  invariants intact. Visual inspection found no gross seam, coordinate, or
  macroblock failure. This does not establish broad semantic generalization.

## 2026-08-31 — Candidate-3 performance characterization

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_PERFORMANCE_CHARACTERIZATION.md` and
`experiments/flux2_candidate3_performance_results/report.json`.

- Under matched native FLUX.2 Klein W4A8 conditions, current Blueprint warm
  sampling is slower than dense at every tested geometry: 1.77x at 512x512,
  2.25x at 1024x512, 1.97x at 1280x512, and 1.90x at 1024x2048.
- Blueprint has no peak allocator-memory advantage at 512x512. At 1024x2048,
  it reduces peak allocated memory by 331 MiB (10.0%) and peak reserved memory
  by 744 MiB (17.5%) relative to dense. The memory advantage grows over the
  tested target sizes but remains a PyTorch allocator result, not physical-free-
  VRAM evidence.
- The 1024x2048 case completed for both variants on the 12 GiB RTX 3060. Dense
  warm sampling took 10.233 s; Blueprint took 19.475 s.
- At 1024x2048, 15 crops execute 15360 local spatial tokens for 8192 unique H
  tokens (1.875x redundancy). Local forwards take 14.908 s, 76.6% of Blueprint
  model time; the global branch takes 4.440 s. Model calls collectively account
  for over 99% of warm sampling time.
- DCT, overlap assembly, Euler arithmetic, coupling, invariant validation, and
  other coordinator work are not current performance bottlenecks. Their total
  is about 0.65% at 1024x2048.
- Token counts and QxK dimensions are recorded as work descriptors, not FLOP
  estimates. The measured conclusion is that sequential local execution lowers
  peak activation memory at larger geometry while increasing total runtime.

## 2026-08-31 — Native FLUX.2 crop batching coordinate boundary

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_CROP_BATCHING_FEASIBILITY.md` and
`experiments/flux2_candidate3_crop_batching_results/report.json`.

- Native `Flux.process_img` constructs one image-position grid from scalar
  `rope_options` and repeats that grid over the entire input batch. Distinct
  absolute crop offsets are not representable per batch element through the
  current transformer-options adapter.
- A real call to the exact native method with `[2,128,32,32]` input and scalar
  `(0,24)` shifts produced bit-identical `[0,24]` to `[31,55]` coordinates for
  both batch elements. It would silently mis-coordinate the second crop.
- Vector shifts fail in native `torch.linspace` because its start/end must be
  scalars. `Flux.forward` exposes no explicit `img_ids` input.
- Correct crop batching therefore requires an adapter/backend change that can
  construct or accept per-batch image IDs. Spatial crop concatenation is not an
  equivalent fallback because it changes attention and geometry.
- The correctness gate failed before model benchmarking; no batch-size-2/4
  performance or memory claim is made.

## 2026-08-31 — Runtime per-batch FLUX.2 coordinate override discriminator

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_RUNTIME_COORDINATE_BATCH_AUDIT.md` and
`experiments/flux2_candidate3_runtime_coordinate_batch_results/report.json`.

- `Flux.process_img` can be narrowly overridden on one loaded model instance to
  construct `[B,T,axes]` image IDs with distinct absolute offsets. For crops
  `(0,0)` and `(24,24)`, patched IDs matched both sequential native grids
  bit-exactly, and the original method was restored.
- Exact IDs are insufficient for end-to-end equivalence on the qualified W4A8
  runtime. Batched predictions differed from sequential by `0.4189/0.4297`
  maximum and `0.0296/0.0356` RMS for the two crops.
- An ordinary control batching the same crop twice with the same scalar offset
  showed the same `0.4189` maximum and `0.0296` RMS drift. The failure lies in
  batch-size-sensitive guider/model/backend execution, not distinct coordinate
  construction.
- `Flux2Adapter` alone cannot deliver the required sequential prediction
  ensemble. Correcting B=2 execution requires a deeper backend investigation.

## 2026-09-01 — B=1 versus duplicated B=2 divergence localization

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_B1_B2_DIVERGENCE_LOCALIZATION.md` and
`experiments/flux2_candidate3_b1_b2_divergence_results/report.json`.

- Sampler input, patchified image tokens, absolute image IDs, and the input to
  `Flux.img_in` are bit-exact between B=1 and each duplicated B=2 element.
- The first differing tensor is `Flux.img_in` output: bfloat16
  `[B,1024,3072]`, with max absolute difference `0.00390625` and RMS
  `3.4613e-6` for each B=2 element versus B=1.
- `img_in` is an unquantized `comfy.ops.Linear` in this checkpoint: its
  `[3072,128]` weight is plain bfloat16, `quant_format` and `layout_type` are
  absent, and it has no input or pre-quant scale. The first divergence is
  therefore batch-shape-sensitive embedding linear execution, not W4A8
  scaling or a quantized operator.
- Text projection and RoPE remain exact. The timestep embedding also diverges
  from an exact input, independently confirming that batch-shape-sensitive
  linear execution exists before attention.
- The initial embedding difference amplifies to `0.418861` max and `0.029550`
  RMS at the final prediction. No claim is made about the lower-level CUDA GEMM
  cause without a separate backend/kernel investigation.

## 2026-09-01 — Batched Candidate-3 trajectory semantic/performance probe

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_BATCHED_TRAJECTORY_SEMANTIC_QUALIFICATION.md` and
`experiments/flux2_candidate3_batched_trajectory_results/report.json`.

- Experiment-only B=2 scheduling used exact per-element coordinates and
  reduced three local calls to two while leaving all Candidate-3 state and
  coupling semantics unchanged.
- In bridge/train and person/car/tree, final batched decodes retained the same
  object count, placement, composition, continuity, and visible detail as the
  sequential controls. No new seam or coordinate displacement was visible.
- Assembled local predictions differed by roughly `0.027-0.056` RMS across
  steps, but hard nonterminal coupling reduced accepted-H drift to at most
  `0.0049` RMS before terminal release. Accepted G remained bit-exact and all
  nonterminal D(H)=G checks remained below `7.2e-7` max error.
- The useful warm timing was 4.48 s sequential versus 4.35 s batched, only
  1.03x. Peak allocated/reserved memory increased by approximately 155/297 MiB.
  The larger first-scene apparent speedup included first-use setup and is not a
  steady-state claim.
- Therefore B=2 is semantically promising in these two cases but does not have
  a meaningful measured speed benefit at the 1024x512 three-crop geometry.

## 2026-09-01 — Selective overlap-token execution fails the correctness gate

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_SELECTIVE_OVERLAP_AUDIT.md` and
`experiments/flux2_candidate3_selective_overlap_results/report.json`.

- Prior SpotEdit mechanics transfer mechanically: an experiment-local FLUX.2
  executor skipped `img_in`, double/single token-local projections, MLPs,
  residual work, and final projection for 256 of 1,024 generated crop tokens,
  while 768 active queries attended to full 1,536-token K/V.
- Same-H, same-sigma, same-coordinate overlap K/V remapped from crop A is exact
  for crop B at double block 0. After one block the overlap hidden trajectory is
  crop-context-dependent: block-1 overlap K/V differs by key RMS `0.716` and
  value RMS `2.65`, and active hidden first diverges at double block 1.
- Active final prediction error is RMS `0.7050`, max `6.7662`; low-pass RMS is
  `0.4846`. The decoded result loses major bridge structure and introduces a
  rectangular discontinuity. This is a semantic failure, not tolerable drift.
- The invalid path measured only 4.0% faster at the core while increasing peak
  allocated/reserved memory by `0.459/0.281` GiB. Full text execution, K/V
  storage/reassembly, and full-context attention remain.
- Exact crop-B context would require propagating skipped overlap hidden through
  the same prior attention/MLP/residual stack, essentially restoring the
  repeated work. Cross-crop K/V reuse therefore fails under the fixed current
  Candidate-3 crop contract.

## 2026-09-01 — Candidate-3 local overlap remains necessary

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_OVERLAP_NECESSITY.md` and
`experiments/flux2_candidate3_overlap_necessity_results/report.json`.

- Deterministic end alignment makes nominal overlap nonuniform. At 32x64,
  stride 28 still produces three crops with actual adjacent overlaps 4 and 28;
  at 64x128 it still produces 3x5 crops with final-axis overlaps 28 and 20. It
  therefore executes exactly the same local tokens as stride 24 at both tested
  geometries.
- Stride 32 reduced crops from 3 to 2 and 15 to 8, reducing warm sampling wall
  time from 4.202 to 3.214 seconds and 20.050 to 12.788 seconds. Peak allocated
  memory changed by less than 4 MiB and peak reserved memory was unchanged.
- Persistent global coupling preserved broad bridge and person/car/tree scene
  organization without overlap, but did not ensure boundary-local semantics.
  The centered person, positioned on a crop boundary, acquired a visible
  duplicate/offset shoulder-torso structure.
- Final adjacent-boundary strip RMS rose from 0.6379 to 0.7913 (bridge) and
  0.8382 to 0.9971 (stress) for stride 32 versus current. This metric is not
  overlap RMS; it compares immediately adjacent assembled strips where no
  common crop prediction exists.
- Global coupling can replace some broad compositional function of overlap,
  but not its tested cross-boundary local reconstruction function. Zero
  overlap fails the quality gate; stride 28 supplies no measured work benefit.

## 2026-09-01 — Local-window runtime crossover is geometry-dependent

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_LOCAL_WINDOW_SCALE.md` and
`experiments/flux2_candidate3_local_window_scale_results/report.json`.

- At H=64x128, 48x48/stride36 reduces calls from 15 to 8 but increases local
  tokens from 15,360 to 18,432. It is 6.9% slower in sampling wall time. Fewer
  calls do not offset its larger token and attention work.
- At H=64x128, 64x64/stride48 uses three crops and 12,288 local tokens. It cuts
  local CUDA time 21.7% and warm sampling wall time 16.6% while retaining the
  person/car/tree scene without visible duplication or seams.
- At H=48x96, 48x48/stride36 uses three crops and 6,912 tokens versus current's
  eight crops and 8,192 tokens. It cuts local CUDA time 25.4% and wall time
  20.2% with no visible semantic regression.
- Ordinary terminal overlap disagreement improves for both larger windows,
  but final adjacent-strip RMS is 14.3% above current for 64x64 at H=64x128
  and 6.2% above current for 48x48 at H=48x96. This is a measurable boundary
  tradeoff even though no seam was visible in the decoded controls.
- Larger local windows can help at favorable target shapes, but no fixed window
  universally dominates. Deterministic end alignment and nonlinear per-forward
  attention cost must be included in any geometry policy.

## 2026-09-01 — Two exact local-window policies are production-qualified

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_GEOMETRY_WINDOW_PRODUCTION_QUALIFICATION.md` and
`experiments/flux2_candidate3_geometry_window_qualification_results/report.json`.

- Across four H=64x128 scenes, 64x64/stride48 preserved person/car/tree,
  bridge/train, centered-subject, and overlap-subject semantics while reducing
  mean warm sampling wall time 17.2% and local CUDA time 22.2%.
- Across three H=48x96 scenes, 48x48/stride36 preserved person/car/tree,
  centered-robot, and full-width-train semantics while reducing mean wall time
  18.6% and local CUDA time 24.0%.
- Every baseline/candidate run and deterministic recomputation passed the same
  Candidate-3 lifecycle/invariants; all final latent repeats were bit-exact.
- Candidate overlap disagreement improved in every scene. Adjacent-strip RMS
  sometimes rose, but no corresponding seam, missing part, displacement,
  duplication, horizon break, or detail loss was visible. This confirms why
  adjacent-strip RMS is diagnostic rather than a standalone quality gate.
- A fresh live ComfyUI process qualified both production mappings with four
  previews, normal VAE/save output, and identical repeated decoded hashes.

## 2026-09-01 — Terminal global prediction is causally dead for returned H

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_GLOBAL_REFRESH_CADENCE.md` and
`experiments/flux2_candidate3_global_refresh_cadence_results/report.json`.

- Reusing interval 2's global denoised estimate only at terminal interval 3
  produces final H exactly equal to the four-fresh-forward baseline in three
  H=64x128 scenes and one H=48x96 scene. Final RMS, max error, and low-frequency
  RMS are all zero.
- This exactness follows from terminal release: interval-3 G does not project
  into terminal H. All applied nonterminal projections and accepted states are
  already identical before the skipped call.
- The policy saves one of four global forwards, 24.8% mean global CUDA time,
  and 6.6% mean sampling wall time. Peak allocator memory is unchanged.
- Final G is not equivalent: its RMS is 3.6-4.5% below baseline when advanced
  with the stale terminal estimate. The result qualifies removal/redefinition
  of causally dead terminal G work, not faithful stale-G tracking.
- Skipping any nonterminal refresh changes accepted H. Early-2, alternating,
  and early-only policies show increasing final/low-frequency divergence and
  projection demand; alternating/early-only visibly alter bridge towers and
  cable geometry.

## 2026-09-01 — Terminal global-forward elimination is production-equivalent

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_GLOBAL_ELIMINATION.md` and
`experiments/flux2_candidate3_terminal_global_elimination_results/report.json`.

- The production coordinator now omits the terminal global model call and
  proposal. It retains the last nonterminal `G` only as explicitly
  unsynchronized diagnostic state; terminal returned `H` is authoritative.
- Across the Phase-6i four-scene matrix, initial states, all local predictions
  and proposals, all accepted H states, nonterminal global predictions and
  proposals, nonterminal G states, and final H were bit-exact with the frozen
  four-global-forward production baseline.
- Runtime telemetry confirms fresh global calls only at ordinals 0/1/2, fresh
  local sets and previews at all four ordinals, no terminal `x0_G`/`G*`, and
  unchanged nonterminal coarse invariants.
- Warm sampling wall time fell 6.06% in aggregate (per-case 3.36–7.49%). Peak
  allocated/reserved memory did not materially fall because larger
  nonterminal/local forwards continue to determine the peak.
- A fresh live ComfyUI process qualified normal BasicGuider → Blueprint sampler
  → SamplerCustomAdvanced → VAE Decode → Save Image execution at H=64x128 and
  H=48x96, with four first-run previews and exact repeated decoded hashes.

## 2026-09-01 — Candidate-3 hard coupling is stable through 20 Euler steps

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_STEP_COUNT_GENERALIZATION.md` and
`experiments/flux2_candidate3_step_count_generalization_results/report.json`.

- Four, eight, twelve, and twenty-step full-denoise CONST-flow schedules were
  tested across three H=64x128 semantic controls and one H=48x96 control using
  the unchanged production coordinator operations.
- All 16 Blueprint trajectories were finite, used fresh global predictions on
  every nonterminal interval, omitted only the terminal global call, executed
  exactly three local crops per interval, and kept nonterminal `D(H)=G` error
  at or below `9.54e-7`.
- Mean projection/H* ratio decreased from 6.59% at four steps to 2.27% at
  twenty steps. The worst interval ratio decreased from 12.91% to 9.84%; more
  coupling applications did not produce growing correction demand.
- Person/car/tree, bridge/train, and boundary-crossing astronaut compositions
  retained object uniqueness, geometry, continuous horizons/ground, and clean
  crop boundaries through 20 steps. No progressive blur, ghosting, anatomy
  split, duplication, or seam was observed relative to each schedule's dense
  semantic reference.
- Sampling wall time scales with forward count, while peak allocated/reserved
  CUDA memory remained constant across step counts for each geometry.

## 2026-09-01 — Variable-step production is reference-exact and live-qualified

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_VARIABLE_STEP_PRODUCTION_QUALIFICATION.md` and
`experiments/flux2_candidate3_variable_step_production_results/report.json`.

- Production schedule validation now accepts one or more full-denoise,
  strictly decreasing CONST-flow Euler intervals from sigma exactly 1 to exact
  zero. Model, conditioning, sampler-family, geometry, and lifecycle limits are
  unchanged.
- Production is bit-exact with the Phase-7a coordinator path at every captured
  boundary for 1/2/4/8/20 steps. The prior four-step frozen four-scene
  regression also remains bit-exact.
- Measured counts are exactly `steps - 1` global forwards, three local forwards
  per interval at the two production-qualified geometries, and one preview per
  interval. Only the terminal interval omits global execution.
- Fresh live ComfyUI workflows at H=64x128 and H=48x96 passed at 4/8/20 steps
  through BasicGuider, SamplerCustomAdvanced, VAE decode, and Save Image. Every
  repeated queue produced an identical decoded RGB hash.
- Clearing detached telemetry at invocation start prevents failed/cancelled
  sampler reuse from presenting a prior successful run's telemetry; accepted
  coordinator state remains invocation-local and atomic.

## 2026-09-02 — Phase 8a finds a semantic frontier before the execution frontier

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_PRACTICAL_SCALING_FRONTIER.md` and
`experiments/flux2_candidate3_practical_scaling_frontier_results/report.json`.

- Fresh dense and Blueprint executions both completed through 2048x4096;
  neither reached OOM. Blueprint reduced peak CUDA allocated/reserved memory
  at every matched point.
- Blueprint remained coherent at 1536x2048, began degrading near 2048x2048,
  fragmented clearly at 1536x3072, and produced severe repeated structures at
  2048x3072 and 2048x4096. A boundary-subject control produced repeated
  astronauts. Matched dense outputs remained substantially more coherent.
- Dynamic block-DCT geometry keeps G at 9/16 of H token count, not a fixed
  budget. G grew from 72x96 = 6,912 to 96x192 = 18,432 tokens, exceeding the
  full 12,288-token H grid at 1536x2048 and the previously qualified
  8,192-token H grid.
- Evidence does not assign the failure to local crops. Because G is the
  authoritative nonterminal trajectory, the next supported hypothesis is that
  growing G exceeded the model's useful spatial/token regime.
- Immediate repeated execution at 2048x4096 failed for both methods with
  `Cannot set version_counter for inference tensor`. Fresh successes classify
  this separately as a warm-reuse/backend issue, not OOM or semantic frontier.

## 2026-09-02 — Phase 8b rejects simple bounded DCT density as the large-canvas fix

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_BOUNDED_GLOBAL_DENSITY.md` and
`experiments/flux2_candidate3_bounded_global_density_results/report.json`.

- At fixed H=128x256, current 4→3 G has 18,432 tokens, bounded 8→4 has
  8,192, and bounded 8→3 has 4,608. All exact DCT pairs preserve constants and
  pass `D(U(G))=G` with maximum float32 error at or below 1.19e-6.
- Natural G0/H0 variance ratios are 0.561945, 0.249471, and 0.140167. Phase 8b
  intentionally does not separate density/bandwidth from mapped variance.
- Current-G decoded global x0 remains one continuous bridge through all three
  nonterminal evaluations. Bounded 8→4 already contains disconnected bridge
  alternatives in interval-0 global x0; bounded 8→3 loses most bridge geometry
  there. All final H outputs remain severely fragmented.
- The current-control final decoded hash exactly matches the Phase-8a
  fresh-process 2048x4096 Blueprint output, confirming harness continuity.
- Bounding reduced total global CUDA time from 28.89 s to 8.25 s and 4.16 s,
  but peak allocated/reserved memory remained 17.37/18.38 GiB because another
  component set the peak. Identical local work took about 74.5 s in all cases.
- Projection demand decreased with coarser G and every invariant passed. The
  failure is semantic information loss, not coupling instability. The current
  final fragmentation enters downstream of, or despite, a coherent global x0.

## 2026-09-02 — Phase 8c localizes large-canvas failure to terminal local authority

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_AUTHORITY.md` and
`experiments/flux2_candidate3_terminal_authority_results/report.json`.

- The terminal schedule interval is 0.99264282→0. Direct Euler produces
  terminal H_star numerically equal to assembled terminal x0_H (RMS 1.83e-8,
  max 4.77e-7). Production therefore returns the local assembled denoised
  estimate at terminal release.
- Assembled local x0_H is visibly fragmented from interval 0, while current
  global x0_G remains a coherent bridge through all nonterminal evaluations.
  Tiny preterminal Euler changes keep accepted H visually near noise; the
  fragmented local estimate becomes final during the dominant terminal move.
- From one bit-identical preterminal state and one shared set of terminal local
  predictions, a fresh terminal global hard projection restores one continuous
  bridge, deck, train, towers, horizon, and water. The production control
  exactly reproduces the Phase-8a/8b hash.
- Projecting toward retained synchronized G3 without a terminal global call
  produces a noise-like final. High-sigma G3 is not a valid sigma-zero coarse
  target; fresh terminal global evolution is required in this schedule.
- Fresh hard projection costs one 9.70 s global CUDA call and has projection
  RMS 0.8673, 102.5% of H_star RMS. It restores composition but causes visible
  cable/deck softness and ghosting, so it is causal evidence rather than a
  qualified production policy.

## 2026-09-02 — Phase 8d terminal current-G context repairs local fragmentation

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_CONTEXT.md` and
`experiments/flux2_candidate3_terminal_context_results/report.json`.

- From one shared H3/G3, all 55 terminal crops using fresh current-G generated
  K/V in all 25 blocks assemble into one coherent bridge scene without an
  output-space projection. Ordinary terminal crops reproduce the fragmented
  Phase-8a–8c control exactly.
- Fragmentation reduction occurs in the context-conditioned crop predictions
  before assembly. Pairwise overlap RMS falls from 1.000208 to 0.281238; the
  final has one bridge/deck/train system with controlled towers and coherent
  water/horizon, retaining fine detail with only minor residual fragments.
- Diagnostic projection RMS to terminal G_star falls from 0.867312 (102.5% of
  H_star RMS) to 0.371625 (52.4%). Context does not make H coarse-exact, but it
  removes the catastrophic semantic failure without applying that projection.
- Current G contributes 18,432 K/V tokens, yielding 1,536-by-19,968 local
  attention. Retaining all 25 blocks' K/V on GPU OOMs on the 12 GB device.
  Experiment-only CPU storage succeeds with a 5.27 GiB cache but transfers
  290.06 GiB across all crops.
- Context terminal crops take 98.34 s CUDA versus 19.05 s ordinary terminal
  crops; fresh capture costs 11.90 s. The mechanism is semantically successful
  but not an efficiency or production qualification.

## 2026-09-02 — Phase 8e requires context through all tested local depth

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_CONTEXT_DEPTH_LOCALIZATION.md` and
`experiments/flux2_candidate3_context_depth_localization_results/report.json`.

- One shared 25-block current-G source cache was consumed by fixed local block
  sets. Only double 0–4 plus single 0–19 reproduces the Phase-8d coherent
  bridge hash and passes the semantic gate.
- Single-only is the strongest partial result: overlap RMS 0.5393 and required
  projection/H* 65.9%, but multiple floating bridge alternatives remain.
  Full context reaches 0.2812 and 52.4% respectively.
- Double-only, early double 0–4 plus single 0–4, and late single 10–19 fail
  materially, with overlap RMS 0.9023, 0.7221, and 0.8247. The early ten-block
  failure does not justify a narrower prefix test.
- Full, single-only, and early/late ten transfer 290.06, 232.05, and 116.03
  GiB respectively. Terminal-local CUDA is 95.89, 80.18, 49.85, and 50.75 s.
  Contiguous depth omission saves work but loses the qualified semantics.
- Deep single-stream consumption carries most of the context benefit, but
  double-stream context still materially improves full-context correctness.
  Global context must be maintained rather than consumed once as an early plan.

## 2026-09-02 — Phase 8f regular post-interaction K/V decimation fails

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_POSTINTERACTION_KV_DENSITY.md` and
`experiments/flux2_candidate3_postinteraction_kv_density_results/report.json`.

- One unchanged, fully interacting 96×192 current-G source forward supplied
  every variant. Selection occurred only after each source block produced
  RoPE-positioned K/V; retained tokens kept their original full-source
  coordinates and every local block still consumed context.
- All 18,432 positions exactly reproduce the qualified Phase-8d/8e bridge.
  Fixed 2×2 decimation to 4,608 positions is catastrophic: many independent
  bridge systems return, overlap RMS rises from 0.2812 to 0.8027, and required
  projection/H-star rises from 52.4% to 94.1%.
- Fixed 4×4 decimation to 1,152 positions is worse (overlap 0.9727,
  projection/H-star 101.4%). Dense source interaction alone does not make the
  final K/V field safely reducible by uniform spatial subsampling.
- The 2×2 consumer cuts transfer by 75% and terminal-local CUDA from 96.17 s
  to 43.34 s; 4×4 cuts transfer by 93.75% and CUDA to 27.47 s. These savings
  are not qualified because semantics fail.
- The minimum qualified consumer density remains the full 96×192 field. This
  rejects regular decimation, not every possible information-preserving K/V
  compression or selection mechanism.

## 2026-09-02 — Phase 8g arithmetic K/V aggregation also fails

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_POSTINTERACTION_KV_AGGREGATION.md` and
`experiments/flux2_candidate3_postinteraction_kv_aggregation_results/report.json`.

- At every block, all 18,432 fully interacting source positions contribute
  exactly once to 4,608 nonoverlapping 2×2 arithmetic means. Generated K is
  pooled before RoPE, V is pooled directly, and K receives RoPE at the exact
  geometric-center mean of the original four-axis full-source IDs.
- Full and 2×2-selection controls exactly reproduce their Phase-8f hashes and
  metrics. The aggregation comparison therefore uses the same accepted state,
  full source hidden trajectory, crops, and all-25-block local consumption.
- Mean pooling remains catastrophically fragmented and is numerically worse
  than selection: overlap RMS 0.9090 versus 0.8027, projection/H-star 100.9%
  versus 94.1%, and assembled RMS versus full 0.7964 versus 0.7004.
- Pooling retains the 75% transfer reduction and takes 42.05 s terminal-local
  CUDA, but those resource reductions are not semantically qualified.
- Simple independent arithmetic means of K and V do not preserve the spatial/
  directional information required by local attention at this token budget.
  This result does not prove that every full-context token is irreducible.

## 2026-09-02 — Phase 8h native-call block-major streaming OOMs at block 0

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_BLOCK_MAJOR_CONTEXT_STREAMING.md` and
`experiments/flux2_candidate3_block_major_context_streaming_results/report.json`.

- Existing attention patches can coordinate a full source call and 55 local
  calls by block, eliminating the all-block CPU cache and host K/V transfer in
  principle. The first source block publishes exact 18,432-position K/V with a
  measured 216 MiB residency.
- Suspending complete native local forwards is not viable on the 12 GiB device.
  Only 46 of 55 local attention calls reach double block 0 before OOM; peak
  allocated memory reaches 16.30 GiB, 15.31 GiB above the streaming baseline.
- The dominant residency is retained native call-frame state—Q/K/V,
  modulation, ordinary/augmented attention, residual and MLP intermediates—not
  the one-block global K/V. No clean all-crop barrier completes, so persistent
  hidden and temporary memory cannot be isolated independently.
- Exact semantic equivalence is not established because no assembled output is
  produced. A true block-major implementation requires a specialized FLUX
  executor that explicitly owns block-to-block crop hidden states and releases
  per-crop temporaries between calls.

## 2026-09-02 — Phase 8i explicit block execution qualifies exact streaming

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_SPECIALIZED_EXECUTOR.md` and
`experiments/flux2_candidate3_specialized_executor_results/report.json`.

- An experiment-local executor reuses the unmodified native FLUX.2 input
  modules, five double blocks, twenty single blocks, RoPE/attention functions,
  and final layer, but explicitly owns block-to-block hidden state instead of
  suspending complete forwards.
- Ordinary one-crop execution is bit-exact at `img_in`, `txt_in`, every block,
  final generated tokens, and returned x0. With full current-G context, source
  K/V, every local block output, and final crop x0 are also bit-exact.
- All 55 context-conditioned crop predictions are bit-exact against a
  same-process CPU-offloaded reference. CPU-versus-GPU overlap assembly is the
  first difference (RMS 2.02e-8, max 4.77e-7); assembled RMS and overlap RMS
  reproduce Phase 8d at 0.709837 and 0.281238.
- Persistent 55-crop hidden state is 495 MiB in double blocks and 660 MiB in
  single blocks. One full source-block K/V is 216 MiB. Maximum observed CUDA
  allocation/reservation is 3.32/6.34 GiB, versus Phase 8h's 16.30-GiB OOM
  from suspended call frames.
- The executor uses no all-block CPU K/V cache and no CPU-to-GPU K/V transfer,
  eliminating the reference's 5.27-GiB cache and 311.4-GB aggregate transfer.
  Measured terminal source-plus-local CUDA is 65.72 s versus about 106.39 s
  for the Phase-8h CPU-offloaded reference.
- The full-context semantic result is preserved. The historical decoded hash
  is not reproduced because this run fell back to tiled VAE decoding, but the
  55 model predictions are exact and the decoded scene is visually identical.

## 2026-09-03 — Phase 8j production boundary for specialized execution

The production design is documented in
`PRODUCTION_SPECIALIZED_EXECUTOR_ARCHITECTURE.md`.

- The generic coordinator can retain ownership of planning, assembly, Euler
  proposals, D/U policy, validation, preview, and atomic `BlueprintState`
  commit if the adapter exposes one model-neutral bulk-region prediction
  operation. FLUX block semantics do not need to enter sampling code.
- The narrow implementation shape is a private `Flux2BlockExecutor` used by
  `Flux2Adapter`. A generic block/session abstraction is not justified because
  no other model family has qualified FLUX-shaped block execution.
- Normal ComfyUI conditioning preparation can be retained with a scoped
  `DIFFUSION_MODEL` preparation wrapper: ordinary CFG-1/calc-cond/BaseModel
  preparation reaches the native Flux `_forward` boundary, the wrapper records
  exactly one call descriptor and returns a discarded sentinel output, and the
  explicit executor consumes that descriptor. Reading and reconstructing
  `guider.conds` directly would bypass condition semantics and is rejected.
- Terminal source hidden state must traverse all 25 blocks, but terminal source
  `final_layer`, x0_G, G-star, and synchronized final G are unnecessary. The
  current terminal policy retains preterminal G explicitly as unsynchronized,
  and no later interval consumes a terminal global prediction.
- Initial production dispatch should be terminal-only and exact-geometry-only
  for H=128×256/G=96×192. Other geometries retain ordinary behavior; failure
  of specialized qualification at that exact geometry must not silently fall
  back to the known-fragmented terminal path.

## 2026-09-03 — Phase 8j terminal specialized executor is production-qualified

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_TERMINAL_SPECIALIZED_PRODUCTION_QUALIFICATION.md`
and its machine-readable report.

- Production now exposes a model-neutral ordered `RegionPredictionSet` while
  keeping FLUX block semantics private to `Flux2Adapter` and
  `Flux2BlockExecutor`. Ordinary evaluations remain an ordered loop over the
  prior `predict_region()` call.
- The exact terminal H=128×256/G=96×192 case prepares one source and 55 crop
  invocations through scoped normal ComfyUI CFG-1 conditioning, then advances
  native hidden states block-major through all 25 blocks. Source final
  projection is omitted and retained terminal G remains unsynchronized.
- Two fresh-process runs produced identical latent and decoded hashes, four
  previews, exactly three nonterminal global predictions, and finite output.
  Final latent RMS/mean/max exactly match Phase 8i's qualified result.
- Specialized CUDA time is 64.43 s and whole four-step sampling is 152.45 s.
  Whole-run peak allocation/reservation is 4.80/6.68 GiB. Current source K/V
  is 216 MiB; all-layer CPU cache and CPU-to-GPU K/V transfer are zero.
- The live VAE selected tiled fallback after ordinary decode OOM, so decoded
  hash differs from historical untiled decoding. Repeated tiled decoding is
  deterministic; transformer/crop/assembled-latent evidence remains the
  authoritative gate.

## 2026-09-03 — Phase 9 normalized local working-canvas falsifier

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NORMALIZED_LOCAL_WORKING_CANVAS.md`.

- At H=64×128, an experimental 8→3 block-DCT fixes the complete-canvas G
  model geometry at 24×48 (1,152 tokens). Three unchanged 64×64 destination
  regions can each be restricted to a fixed 32×32 working canvas while
  preserving endpoint-mapped absolute coordinates.
- This satisfies the desired execution-budget form: G tokens are fixed at
  1,152 and each local forward at 1,024. Relative to fixed-G direct 64×64
  calls, local token executions fall 75%, measured local CUDA falls 73.8%,
  sampling wall time falls 67.0%, and peak allocation falls 2.834→2.530 GiB.
- Simple bilinear state/prediction transport is not semantically adequate.
  Normalized assembled local x0 is visibly smeared from interval 0; the final
  retains a coarse single bridge/horizon but loses useful local structure and
  sharpness while showing train/structure ghosting.
- Lower terminal overlap RMS (0.12835 versus 0.14701 fixed-G direct) is a
  low-pass side effect, not a quality improvement. D(H)=G invariants pass, so
  the failure is local-fidelity transport rather than lifecycle instability.

## 2026-09-03 — Phase 9b native local magnification is partially supported

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NATIVE_LOCAL_MAGNIFICATION.md`.

- With H=64×128, fixed G=24×48 and 15 destination 32×32 regions, a 64×64
  local working state can preserve the destination crop exactly under 2×2 mean
  restriction while adding sigma-scaled zero-cell-mean degrees of freedom.
  Across all calls, coarse-state max error is 4.77e-7.
- Naïve bilinear magnification is not coarse-consistent under that restriction
  (max error 2.448) and produces strong blur/ghosting. State construction is a
  demonstrated cause of failure, not merely x0 restriction or coordinate scale.
- Sigma-consistent magnification uses the same coordinate mapping and x0
  restriction but recovers materially sharper truss, cable, support and water
  detail while retaining one principal whole-canvas bridge/horizon. It still
  produces repeated train/bridge elements and residual ghosting, so it is not
  semantically qualified.
- The invariant is achieved structurally: global work stays 1,152 tokens and
  each magnified local call stays 4,096 tokens as destination H grows. This is
  not an efficiency win at the tested H: 245,760 local token executions and
  61.25 s wall versus 61,440 and 16.84 s for direct 32×32 regions.

## 2026-09-03 — Phase 9c native-local global context is insufficient

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NATIVE_LOCAL_GLOBAL_CONTEXT.md`.

- From the exact Phase-9b C preterminal state, full accepted-H generated K/V
  consumed by all 25 local blocks reduces terminal overlap RMS from 0.214053
  to 0.134323, proving that same-sigma whole-canvas context materially changes
  and aligns magnified local predictions.
- The decoded full-H result still contains repeated bridge/train alternatives.
  Therefore the failure is not explained solely by the fixed 24×48 G being an
  insufficient context representation.
- Fixed 24×48 G context is weaker: overlap RMS is 0.200175 and the decoded
  semantic failure remains close to local-only. It is not qualified for a
  complete trajectory.
- Full-H and fixed-G context increase terminal source-plus-local CUDA from
  14.912 s to 41.820 s and 24.537 s respectively in the CPU-offloaded
  diagnostic. Context improves compatibility but does not meet the semantic or
  efficiency gate.

## 2026-09-03 — Phase 9d native working coordinates do not resolve repetition

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NATIVE_WORKING_COORDINATE.md`.

- Holding the accepted Phase-9b C state and W tensors fixed, changing local
  RoPE from compressed destination spacing to an ordinary unit-spaced 64×64
  local frame materially changes predictions (assembled RMS 0.316471) but does
  not remove repeated bridge/train alternatives.
- Native local coordinates worsen overlap RMS from 0.214053 to 0.234165.
  Whole-H context remapped per crop as `2*(H_coordinate-crop_origin)` recovers
  some agreement (0.191253) but still does not yield one bridge system.
- Because even the correctly frame-mapped full-H context upper bound fails the
  semantic gate, compressed positional geometry is weakened as the primary
  explanation. The next justified discriminator is prediction
  restriction/transport, not increased G density or more context.

## 2026-09-03 — Phase 9e transport variants collapse under CONST-flow algebra

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_MAGNIFIED_PREDICTION_TRANSPORT.md`.

- With linear 2×2 mean D and the Phase-9b invariant `D(W)=H_crop`, restricting
  x0, restricting FLUX velocity, restricting the exact denoising delta, and
  same-sigma re-noise/restriction all reduce algebraically to `D(x0_W)`.
- Runtime confirms the identity: assembled pairwise RMS is only
  `3.65e-8`–`5.20e-8`, max absolute is at most `7.15e-7`, and all assembled
  norms and overlap disagreement are effectively identical.
- No transport changes repeated bridge/train semantics. A distinct cross-scale
  result requires changing the state/evolution contract, not rearranging the
  terminal CONST-flow equation.

## 2026-09-03 — Phase 10 independent persistent W trajectories fragment

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_PERSISTENT_NATIVE_LOCAL_TRAJECTORY.md`.

- Fifteen native `64×64` W states initialized once from the same H0 and then
  independently Euler-updated do not improve Phase-9 semantics. They produce
  multiple bridge scenes, repeated towers, incompatible horizons, and severe
  regional ghosting.
- Persistent prediction overlap RMS is identical to the reconstructed control
  at interval 0, then remains `0.8173, 0.8429, 0.8031`; reconstructed W improves
  to `0.6534, 0.4913, 0.2594`.
- Maximum regional `D(W)-crop(H)` RMS grows monotonically from `0.0526` after
  the first update to `0.5234` at terminal acceptance. Existing H/G invariants
  continue to pass, localizing the divergence to independent W ownership.
- Each W has exact hash/id lineage, four Euler updates, and zero regeneration
  after initialization. Persistence alone is therefore rejected as the missing
  mechanism; any future persistent-W design requires explicit global or
  cross-region coupling.

## 2026-09-03 — Phase 10b coarse synchronization collapses persistent W toward A

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_PERSISTENT_COARSE_SYNC.md`.

- Post-step `W_next=W_star+U(crop(H_next)-D(W_star))` holds W/H coarse error to
  `4.77e-7` while preserving the W null-space component within `5.87e-8` RMS.
- Synchronization prevents Phase-10 B's catastrophic divergence: terminal
  prediction overlap RMS improves from 0.803052 to 0.356545. The reconstructed
  control remains better at 0.259399.
- C returns to A's principal-bridge semantic class but retains repeated
  train/support alternatives and shows no clear detail advantage over A.
  Persistence therefore contributes no qualified benefit once coarse state is
  synchronized.
- The terminal W_star and synchronized-W diagnostics yield the same returned H
  by construction; both were decoded, and synchronization magnitude reaches
  17.2% mean / 19.8% maximum of W_star RMS at terminal acceptance.

## 2026-09-03 — Phase 11 full-H context validates shared-context normalized W

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NORMALIZED_W_SHARED_CONTEXT.md`.

- Fresh same-sigma context from the complete accepted H, consumed by all 25
  local FLUX.2 blocks, produces one dominant continuous bridge system instead
  of repeated regional bridge/train alternatives. The improvement is visible
  from interval 0 and persists through terminal release.
- The identical mechanism fed by fixed 24×48 G only modestly lowers terminal
  overlap RMS (`0.259399` to `0.250149`) and retains semantic repetitions.
  Full-H context has terminal overlap RMS `0.266542`, showing that overlap RMS
  is not a sufficient proxy for whole-scene semantic coherence.
- B and C use fresh accepted-state/same-sigma sources at every interval with
  independent correct RoPE frames; no stale context is reused. All 25 blocks
  consume context for all 15 crops.
- The oracle is expensive: full-H source plus context local CUDA is about
  `166.33 s`, versus `59.32 s` local-only, with about `140.63 GiB` aggregate
  diagnostic CPU-to-GPU K/V traffic. It is a semantic upper bound, not an
  efficient implementation.
- The remaining Phase-11 bottleneck is fixed-budget global representation
  sufficiency, not the basic normalized-W scale transformation or absence of a
  workable inside-transformer sharing mechanism.

## 2026-09-03 — Phase 12 post-interaction compression remains insufficient at 1,152 positions

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_POSTINTERACTION_COMPRESSED_ORACLE.md`.

- Full accepted-H source execution followed by per-block adaptive-area pooling
  of already-positioned generated K and generated V from 8,192 to 1,152
  consumer positions does not reproduce the full-H oracle's coherent bridge.
  Repeated bridge/train/support alternatives remain.
- The compressed oracle and fixed-G variants expose the same 1,152 K/V count,
  cache size, and transfer volume. The former nevertheless pays for ordinary
  full-H source interaction, isolating consumer representation capacity from
  pre-interaction source construction.
- D's terminal overlap RMS is `0.259909`, versus `0.250149` for fixed G and
  `0.266542` for the semantically coherent full oracle. This reinforces that
  overlap RMS is secondary and cannot substitute for scene-level inspection.
- Under this fixed linear aggregation, pre-interaction compression is not the
  sole bottleneck. A 1,152-position context interface is below the observed
  semantic threshold; this does not rule out learned/nonlinear compression.

## 2026-09-03 — Phase 12b 2,048-token context partially improves coherence

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_POSTINTERACTION_2X2_CAPACITY.md`.

- Nonoverlapping 2×2 aggregation of full-H, post-interaction generated K/V to
  `32×64 = 2,048` consumer positions visibly reduces fragmentation relative to
  the 1,152-position post-interaction control, but repeated train/truss/support
  alternatives remain and the result is clearly below the 8,192-token oracle.
- E improves overlap RMS over D at every interval and reduces assembled-x0 RMS
  versus the oracle from `0.517928` to `0.495723` at terminal evaluation. These
  numerical gains are consistent with, but do not substitute for, the partial
  semantic improvement.
- All 8,192 source positions contribute exactly once. The complete 2,048-entry
  coordinate-quad provenance is recorded and stable across four fresh same-sigma
  probes; all 25 blocks consume context for all 15 crops.
- The 2,048-token context costs `117.57 s` wall, `0.586 GiB` cache per interval,
  and `35.156 GiB` aggregate diagnostic transfer, versus `110.44 s`, `0.330 GiB`,
  and `19.775 GiB` for the 1,152 post-interaction control.

## 2026-09-03 — Phase 12c finds a semantic threshold by 4,096 positions

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_POSTINTERACTION_2X1_CAPACITY.md`.

- An anisotropic `32×128 = 4,096` post-interaction context, formed by exact
  nonoverlapping vertical 2×1 averaging of full-H positioned K and matching V,
  resolves the bridge/train scene into the same one-dominant-system semantic
  class as the 8,192-token oracle. The 2,048-token condition remains visibly
  below that threshold.
- D reduces assembled terminal-x0 RMS versus the oracle from `0.495723` at
  2,048 tokens to `0.407488`, and reduces interval-0 RMS from `0.461038` to
  `0.349765`. Semantic inspection, not these metrics alone, determines the pass.
- All 8,192 source positions contribute exactly once to 4,096 entries, with no
  omissions/duplicates and stable complete coordinate provenance across four
  fresh same-sigma interval probes. All 25 blocks and 15 crops consume context.
- This is information-sufficiency evidence only. D still requires dense full-H
  source interaction, `1.172 GiB` context storage per interval, `70.312 GiB`
  aggregate diagnostic transfer, and `133.62 s` sampling wall time.

## 2026-09-03 — Phase 13 direct 4K interaction preserves oracle-class semantics

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_DIRECT4K_SOURCE.md`.

- A transient `32×128` source formed by exact vertical-pair means of accepted
  `64×128` H, with rows positioned at full-H centers `2r+0.5`, performs ordinary
  all-25-block global interaction and supplies context in the same
  one-dominant-bridge semantic class as post-interaction 4K and full-H 8K.
- Direct 4K assembled x0 is closer to the full-H oracle than post-interaction
  4K at intervals 1–3 (`0.327505/0.406851/0.332610` versus
  `0.491778/0.500892/0.407488`) and comparable at interval 0.
- Direct K/V is not a numerical surrogate for pooled full-H K/V. Mean K cosine
  falls from `0.9649` to `0.9131` and mean V cosine from `0.9061` to `0.7595`
  over the trajectory, yet semantic organization survives.
- Pair averaging yields direct/H variance ratios near 0.50 (`0.5013` initially,
  `0.5316` terminal). No variance control was needed to pass this semantic gate.
- Lower token count did not accelerate the native source: direct `32×128` took
  `23.826 s` total versus `12.785 s` for full-H source execution. Information
  sufficiency is established; compute and production suitability are not.

## 2026-09-03 — Phase 14 natural fixed-4K restriction does not scale semantically

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_FIXED4K_LARGE_DESTINATION.md`.

- At `H=128×256`, a destination-independent `32×128 = 4,096` source formed by
  exact nonoverlapping `4×2` area means and full-H center coordinates remains
  severely fragmented after ordinary all-25-block global interaction. The
  Phase-13 direct-4K semantic result does not survive this stronger `8:1`
  restriction.
- Natural area averaging produces a fixed-source/H variance ratio of
  `0.124651` and RMS ratio `0.353060`, versus Phase 13's successful pair-mean
  variance ratios near `0.50–0.53`. This identifies source construction and
  mapped-state statistics as the next causal boundary; it does not prove a
  universal 4K capacity failure.
- The `18,432`-token accepted-G context also fails to reproduce its historical
  single-bridge result after consumers change from direct `32×32` crops to
  reconstructed native `64×64` W. Destination-scaled context qualification
  therefore cannot be transferred across local consumer geometry without a
  new control.
- Under the same block-major executor, fixed 4K is measurably cheaper than
  18,432 tokens: source CUDA `1.083` versus `9.677 s`, terminal wall `105.234`
  versus `187.844 s`, current-block K/V `48` versus `216 MiB`, and peak allocated
  about `3.58` versus `5.23 GiB`. Semantic failure and backend efficiency are
  separate conclusions.

## 2026-09-03 — Phase 15 scalar variance restoration does not recover coherence

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_FIXED4K_SOURCE_STATISTICS.md`.

- Fixed a-priori gains `1`, `2`, and `sqrt(8)` produce accepted-H variance
  ratios `0.124651`, `0.498606`, and `0.997211` without changing the 4K source
  mapping, coordinates, executor, or consumers. All three remain in the same
  severely fragmented bridge/train semantic class.
- The input and `img_in` RMS scale directly with gain, but double-block-0 K RMS
  is effectively identical (`1.184487/1.184486/1.184487`) and V RMS differs by
  less than 0.01%. Native normalization/modulation largely erases simple source
  amplitude before useful transformer context is exposed.
- Terminal overlap RMS improves slightly from `0.784053` to `0.767068`, but
  this does not correspond to whole-scene improvement. Scalar statistics are
  not a sufficient proxy for information retained by the 4×2 restriction.
- Source CUDA, local CUDA, 48-MiB current-block K/V, and peak memory remain
  materially unchanged across gains, preserving Phase 14's work comparison.

## 2026-09-04 — Phase 16 fixed-4K orthogonal mode packing remains fragmented

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_FIXED4K_REPRESENTATION_RICHNESS.md`.

- A fixed 4×2 orthonormal DCT retaining DC, first vertical, first horizontal,
  and first diagonal modes, packed through disjoint 32-dimensional subspaces of
  a normalized 128-channel Hadamard basis, does not recover one bridge system.
  It remains in the Phase-15 fragmented semantic class.
- The fixed 128-channel model interface can hold only 128 of the 512 retained-
  mode/channel coefficients per spatial position. C3 explicitly retains 25%
  and discards complementary channel components; deterministic packing cannot
  make this dimensional loss disappear.
- C3 modestly improves H reconstruction RMS (`0.930890` versus `0.931140`) and
  edge-gradient RMS (`1.340725` versus `1.374486`) but worsens low-frequency
  8×8-mean reconstruction (`0.107615` versus near zero) because DC is no longer
  retained for every latent-channel component.
- Overlap RMS falls from `0.784053` to `0.742034`, but visual bridge/train
  fragmentation remains. This again shows overlap compatibility is not a
  whole-scene semantic proxy.
- Fixed-4K source/local timing and 48-MiB block K/V remain unchanged. The
  failure is representational/semantic, not a loss of the measured 4K resource
  advantage.

## 2026-09-04 — Phase 17 stronger joint consumption helps but does not unify the scene

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_FIXED4K_CONSUMER_INTERFACE.md`.

- The same `4×2` area-mean 4K source and normalized W were placed in one native
  bidirectional FLUX stream (`512` text + `4096` source + `4096` W) for all 25
  blocks. Only W was final-projected. This materially promotes larger, more
  continuous bridge structures compared with frozen external source K/V, but
  several independent bridge/train systems remain.
- Source and W K/V are identical to their external-control inputs at double
  block 0; hidden state diverges immediately after joint attention. By single
  block 19, representative source/W hidden RMS differences reach
  `41.64/42.97`. The consumer-interface intervention is causally active rather
  than a coordinate or preparation change.
- Terminal overlap RMS improves from `0.784053` to `0.766685`, but the required
  one-dominant-scene semantic gate still fails. Both fixed-source information
  and interface behavior likely contribute.
- The oracle costs `142.178 s` terminal wall and about `4.93/9.82 GiB` peak
  allocated/reserved because it evolves 55 W-specific joint source states.
  This is semantic evidence, not an efficiency architecture.

## 2026-09-04 — Phase 18b finds no transient coherent state across joint depth

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_JOINT_DEPTH_LOCALIZATION.md`.

- Prefix-joint/external-tail checkpoints at S0, S9, and S19 all remain in the
  same S1 class as the Phase-17 full-joint endpoint. RMS versus full joint
  decreases from `0.223708` to `0.101851` to zero, but no S2/S3 semantic state
  appears.
- External-prefix/joint-tail checkpoints at D4 and S0 recover at most that same
  weak S1 class; switches after S9 or S14 remain S0. Late joint depth therefore
  cannot repair already-established fragmentation.
- Overlap RMS stays in the narrow `0.766685–0.780861` range and does not track
  the categorical semantic result.
- Early prefix tails are causally expensive because joint interaction has
  already produced 55 W-specific source states. The prior D0/D4 costs
  (`1491.96/1157.48 s`) were legitimate; Phase 18b intentionally excludes them.
- Arm artifacts are persisted transactionally and resumed only after exact
  accepted-state/source/W/schedule fingerprint validation.

## 2026-09-04 — Phase 19 bounded single-scale and hierarchical states remain S0

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_BOUNDED_GLOBAL_STATE.md`.

- At the fixed Phase-14 terminal state, an exact `64x64` single-scale source
  and a `16x32 + 32x112` explicit whole-canvas hierarchy both use exactly 4,096
  shared source tokens and all 25 globally interacting source blocks. Neither
  produces a dominant bridge/train scene; both remain S0 with the established
  `32x128` control.
- The hierarchy reserves 512 tokens for an `8x8`-cell coarse canvas and 3,584
  for a complete-canvas medium level. Joint source interaction over both scales
  changes fragment placement but does not consolidate bridge, tower, train,
  horizon, or water identity.
- Overlap RMS (`0.784053`, `0.788349`, `0.780410`) again fails to predict
  semantic class. The small hierarchical improvement is not a pass.
- Runtime is controlled across arms: about `102.5–104.0 s` terminal wall,
  `1.06–1.08 s` source CUDA, `101.1–102.6 s` local CUDA, and 48 MiB current-
  block K/V. This isolates representation rather than work budget.

## 2026-09-04 — Phase 20 separates bounded planning from local-state transfer

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NATIVE_BLUEPRINT_LOCAL_STATE.md`.

- One ordinary native-coordinate `32x64` / 2,048-token Blueprint call produces
  an S3 bridge/train plan from the complete accepted `H=128x256`: one bridge,
  one train, controlled towers, and coherent horizon/water.
- Mapping that `x0_B` to destination space and using the exact same-sigma rule
  `W=(1-sigma)U(b)+sigma*N` leaves the 55 independently refined W outputs in
  S0. The failure is therefore at Blueprint-to-local transfer/refinement, not
  bounded native-coordinate global planning.
- At terminal sigma `0.99264282`, the coherent denoised Blueprint term has
  coefficient only `0.00735718`. The construction is algebraically valid
  (`D(W)` error `4.77e-7`) but provides little input-state authority before the
  ordinary local prediction.
- B changes the latent materially versus control (`0.855303` RMS) while overlap
  worsens from `0.856211` to `0.885702`; neither numerical fact predicts the S0
  semantic result.

## 2026-09-04 — Phase 20b exact prediction anchoring restores global composition

Detailed evidence is in
`experiments/FLUX2_CANDIDATE3_NATIVE_BLUEPRINT_PREDICTION_ANCHOR.md`.

- Applying `x0 + U_B(blueprint_crop - D_B(x0))` after each ordinary W forward
  moves the result from S0 to S3 and enforces the mapped Blueprint crop to
  `5.36e-7` max absolute error.
- The intervention is large: correction RMS is `0.888769`, or `1.119x` the
  original local-x0 RMS on average. It restores one scene but visibly blurs and
  overconstrains local detail.
- Ordinary Phase-20 B is reproduced bit-exactly. Accepted H/G, W inputs,
  transformer execution, conditioning, and lifecycle are unchanged; only
  post-forward terminal x0 changes.
- The almost-zero anchored overlap RMS (`6.59e-8`) follows algebraically from
  all crops sharing the mapped Blueprint field and is not an independent image-
  quality result.
