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
- Phase 2j zero-update dense-source interaction ablation retaining all 2048
  native source positions while blocking image-to-image source attention.
- Phase 2k zero-update interaction-range discriminator comparing ordinary dense,
  fixed 16x16 windowed, and no-image-image source attention.
- Phase 2l zero-update cross-window propagation discriminator using alternating
  unshifted and clipped shifted-by-(8,8) 16x16 source windows.
- Phase 2m zero-update dense-depth discriminator comparing all-dense, first-five
  dense, last-five dense, and no-image-image source interaction.
- Phase 2n zero-update maintenance probe comparing early-only against early
  dense plus fixed refreshes at ordinals 10, 15, and 20.
- Phase 3 Candidate-3 research/design pass defining persistent low/high states,
  a synchronized Euler lifecycle, three ranked state-coupling rules, compute
  bounds, failure criteria, and one selected FLUX.2 falsification experiment.
- Phase 3 four-step hard-global-anchor falsifier with dense, tiled-only,
  uncoupled-dual, and coupled trajectories; real outputs, state/projection
  telemetry, lifecycle assertions, and focused report.
- Phase 3b terminal-release discriminator adding one fixed lifecycle variant,
  with D/E bit-exact pre-acceptance controls, final metrics, decoded comparison,
  and focused report.
- Phase 3c fixed-lifecycle generalization qualification using a new asymmetric
  person/car/tree prompt and seed, with all five controls and real outputs.
- Candidate-3 higher-bandwidth geometry design selecting one exact-right-inverse
  `24x48` block-DCT operator pair; no model or sampler execution.
- Phase 3d 24x48 block-DCT higher-bandwidth discriminator with synthetic and
  accepted-state invariant checks, all five trajectories, decoded artifact
  inspection, performance telemetry, and focused report.
- Phase 3e mapped-noise-variance discriminator comparing original 16x32,
  variance-matched 16x32, and 24x48 block-DCT under one fixed terminal-release
  lifecycle, with a dense metric reference and focused report.

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
- Dense-source interaction discriminator: blocking source image-image mixing
  restored the duplicate lighthouse and increased dense-crop RMS by 40% despite
  unchanged 2048-token source and local-interface geometry. Global source
  interaction is required for the observed dense-context success.
- Source-range discriminator: fixed 16x16 windows retained 12.5% of dense
  image-image connectivity but restored the duplicate at near no-mixing severity.
  Within-window interaction alone is insufficient.
- Cross-window propagation discriminator: alternating windows achieved full
  analytical canvas reachability across depth but retained the duplicate and
  performed worse than fixed windows and no mixing. Reachability alone is
  insufficient.
- Dense-depth discriminator: the first five dense double blocks recover much
  more numerical/geometric benefit than the last five single blocks, but neither
  removes the duplicate. A 20% dense-depth budget is insufficient in both ranges.
- Sparse-refresh discriminator: three later dense refresh blocks briefly reduce
  some K/V divergence but gains decay through restricted blocks; final error is
  worse than early-only and the duplicate remains.
- Candidate-3 design contract: complete. The first test is a hard accepted-state
  global anchor with exact `D(H) = G` synchronization, not prediction-residual
  fusion.
- Candidate-3 hard-anchor result: composition is controlled but local detail is
  overconstrained. The global branch is valid; C.H equals B.H exactly; all
  lifecycle/invariant checks pass. Projection grows to 38.4% of proposed-H RMS
  and the terminal acceptance visibly ghosts fine structure.
- Candidate-3 terminal release: sufficient in the controlled case. It preserves
  the hard-coupled single-scene organization, removes most terminal ghosting,
  and improves final/low-frequency RMS versus both hard anchoring and tiled-only.
  D/E differ only at final acceptance.
- Candidate-3 generalization: partial. Terminal release again improves detail
  and numerical agreement without restoring the tiled-only second person or
  competing large tree, but a small duplicate car and two thin extra trees
  remain under both D and E.
- Candidate-3 24x48 result: the known extra car/trees are removed, numerical
  agreement improves materially, global decoding is valid, and no macroblock
  artifacts are visible. Runtime/global work increases as expected; terminal
  release remains necessary.
- Candidate-3 variance discriminator: both factors contribute. Matching 16x32
  variance improves error and suppresses some extras but leaves residual tree/
  vehicle-like structure and worsens overlap; only 24x48 fully resolves the
  known duplication.
- Candidate-3 production architecture design: complete. The selected boundary
  is an explicit custom Euler `SAMPLER` used with the existing ComfyUI guider
  and `SamplerCustomAdvanced`. State, interval coordination, block-DCT
  geometry, crop assembly, terminal release, and model-family coordinates have
  distinct fail-closed ownership.
- Candidate-3 first production slice: implemented under
  `target/ComfyUI-Blueprint-Diffusion/`. Ten focused unit tests pass. The real
  FLUX.2 regression is bit-exact with the research harness at every required
  state/prediction/proposal boundary and final H; no divergence exists.
- Candidate-3 live workflow: qualified as an ordinary custom node in ComfyUI
  0.33.0. Registration, `BasicGuider`/`SamplerCustomAdvanced`, 1024x512 empty
  latent, four interval previews, normal VAE decode, exact decoded-pixel seed
  reproducibility, unload/reload, cancellation recovery, and all requested
  fail-closed controls pass. A loadable workflow is saved under `experiments/`.
- Candidate-3 arbitrary geometry: implemented without lifecycle or algorithm
  changes. Latent axes at least 32 and divisible by four use dynamic block-DCT
  G geometry, deterministic 32x32 crop coverage, and absolute FLUX.2
  coordinates. Fifteen unit tests pass; real square, portrait, and >2:1 wide
  four-step workflows each completed with four previews and normal VAE output.
- Candidate-3 performance baseline: qualified on 512x512, 1024x512,
  1280x512, and 1024x2048 with matched dense controls, cold/warm runs, CUDA
  events, allocator peaks, per-interval work, and valid decodes. Current
  Blueprint is 1.77x-2.25x slower but saves 331 MiB peak allocated and 744 MiB
  peak reserved at 1024x2048. Local crop forwards are the dominant cost.
- Physical free VRAM, alternate GPUs/backends, and optimized performance remain
  unqualified.
- Candidate-3 crop batching feasibility: stopped at the correctness gate.
  Current native FLUX.2 repeats one scalar-offset image-ID grid over the batch;
  vector offsets fail and no explicit per-batch `img_ids` interface exists.
  No invalid batching benchmark or production change was made.
- Runtime coordinate follow-up: a scoped `process_img` override produced
  bit-exact per-batch absolute IDs, but native B=2 predictions remained
  materially different even for duplicated inputs with identical scalar
  coordinates. Adapter-only batching is not qualified; no production change or
  performance claim was made.
- B=1/B=2 localization: the first mismatch is now isolated to the unquantized
  bfloat16 `Flux.img_in` output, after exact sampler input, patchification, IDs,
  and embedding input. It amplifies through the model to the known prediction
  drift. Production remains unchanged and crop batching remains unqualified.
- Batched-trajectory probe: experiment-only B=2 scheduling retained visible
  bridge/train and person/car/tree semantics and all nonterminal coupling
  invariants, but its warm wall-time gain was only 3% while peak allocated and
  reserved memory increased. No production batching change is justified.
- Selective-overlap probe: 25% of crop-B generated-token-local work was truly
  skipped with full K/V context, but crop-A overlap features became invalid for
  crop B after block 0. Active prediction RMS reached 0.705 with severe decoded
  geometry loss; the invalid path was only 4% faster and used more peak memory.
  This optimization direction fails under the fixed crop contract.

## Active blockers / unknowns

- Whether the observed reduced-global control generalizes across prompts,
  seeds, sizes, and global resolutions.
- Whether any cheaper source mechanism can maintain globally organized hidden
  states through later depth. Hard no-image-image transitions discard both an
  early plan and isolated refresh corrections in the tested policies.
- Whether a normal 512-token trajectory initialized by restriction of the
  full-canvas noise remains a valid/coherent global branch despite the mapped-
  noise variance shift.
- Whether exact accepted-state coarse anchoring suppresses semantic alternatives
  or merely confines them to the high-resolution restriction nullspace. The
  first run shows semantic control, but exact terminal projection damages
  detail.
- Cache validity and block-level sparse propagation for Z-Image and Anima.
- Whether periodic global refresh is sufficient after early denoising.

## Next concrete milestone

Candidate 2 semantic research remains paused. The fail-closed Candidate-3
production slice is implemented, research-equivalent, and live-workflow
qualified across the tested target geometries. Local crop model work remains
the measured optimization target, but ordinary crop batching is blocked by
batch-size-sensitive unquantized embedding execution and has no meaningful
measured warm speed benefit at the three-crop geometry. Cross-crop selective
K/V reuse also fails because overlap hidden evolution is crop-context-specific.
Do not promote either mechanism or broadly change sampler/model scope
automatically.
