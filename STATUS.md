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
- Overlap-necessity probe: complete. Stride 28 preserved the two tested scenes
  but retained the same 3/15 crop counts because deterministic end alignment
  increased the final overlaps. Stride 32 cut crops to 2/8 and warm sampling
  time by 23.5%/36.2%, but produced a visible boundary-local duplicate torso
  structure in the 1024x2048 person case and raised boundary-strip RMS in both
  cases. All lifecycle/invariant checks passed. Keep production stride 24.
- Local-window scale probe: complete. At 1024x2048, 48x48 is 6.9% slower while
  64x64 is 16.6% faster; at 768x1536, 48x48 is 20.2% faster. Larger-window
  decodes preserve person/car/tree semantics with no visible seams, but final
  adjacent-strip RMS rises for the two faster configurations. Results are
  geometry-dependent and no production policy was changed.
- Geometry-dependent window qualification: complete and productionized for two
  exact shapes. H=64x128 now uses 64x64/stride48 and H=48x96 uses
  48x48/stride36; all other shapes retain 32x32/stride24. Seven semantic stress
  pairs, bit-exact repeats, 17 unit tests, and fresh live ComfyUI runs pass.
  Mean warm wall gains are 17.2% and 18.6%, respectively.
- Global-refresh cadence discriminator: complete, experiment-only. Fresh global
  estimates at 0/1/2 with stale reuse only at terminal interval 3 are exactly
  output-equivalent across four scenes and save 6.6% mean wall time. Skipping
  a nonterminal refresh changes accepted H, increases later projection demand,
  and can alter bridge geometry. Production was not changed.
- Terminal global-forward elimination: productionized and qualified. The
  terminal interval now performs fresh local work only and retains prior G as
  explicitly unsynchronized diagnostic state. All requested real-model tensor
  boundaries are bit-exact with the frozen four-global baseline across four
  scenes; 18 tests and fresh live H64/H48 workflows pass. Aggregate warm wall
  time improved 6.06%, with no material peak-memory change.
- Step-count generalization: experiment complete; production unchanged. The
  unchanged Candidate-3 lifecycle remains numerically and semantically stable
  at 4/8/12/20 Euler steps across four required scenes. Projection demand does
  not accumulate with longer schedules, all invariants pass, and only the
  terminal global call is omitted. A separate production validation/API task
  is now justified.
- Variable-step Euler production support: implemented and qualified. Full-
  denoise strictly decreasing schedules now accept at least one interval.
  Production is boundary-by-boundary bit-exact with Phase 7a at 1/2/4/8/20
  steps and with the frozen four-step baseline; live H64/H48 workflows pass at
  4/8/20 steps with exact repeated decoded hashes.

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

Phase 6f additionally shows that removing overlap is a real performance lever
but is not quality-safe. Persistent global coupling preserves broad composition
without overlap yet does not prevent crop-boundary object reconstruction
failures. The current overlap policy remains qualified; no production change
is pending.

Phase 6g finds a promising but shape-specific local-window crossover. The next
qualification target is 64x64/stride48 specifically at H=64x128, with more
boundary placements and a compatible long-geometry semantic prompt before any
production geometry policy is introduced.

Phase 6h completes that qualification. The two exact mappings are now the
production policy; no generic largest-fitting-window rule is implied or
qualified.

Phase 6j completes that narrow production optimization. The causally dead
terminal global forward is eliminated; no nonterminal cadence reduction is
qualified or pending from this evidence.

Phase 7a qualifies ordinary longer full-denoise CONST-flow Euler schedules up
to 20 steps experimentally. The next production task may generalize schedule
cardinality only; it must retain strict descent, sigma 1 → exact zero, current
terminal behavior, and all existing fail-closed boundaries.

Phase 7b completes that production generalization. Variable step cardinality
is now supported under the same full-denoise CONST-flow contract; partial
denoise, alternate samplers, CFG>1, masks/editing, and other model families
remain unsupported and fail closed.

Phase 8a is complete with no production changes. Fresh dense and Blueprint
runs both execute through 2048x4096, and Blueprint lowers peak CUDA allocator
memory, but Candidate-3 semantic coherence fails as scale increases.
Degradation begins around 2048x2048/1536x3072 and becomes severe by
2048x3072. The largest coherent Blueprint result in this ladder is 1536x2048;
matched dense results remain substantially more coherent above it. The next
milestone is one bounded/adaptive global-state-density discriminator at a
known failing canvas, not another resolution sweep. The repeated-run
inference-tensor error remains a separate backend warm-reuse issue.

Phase 8b is complete with no production changes. At fixed 2048x4096, neither
8→4 nor 8→3 bounded block-DCT G restores coherence. Both greatly reduce
global runtime, but 8→4 fragments the bridge in its first global x0 and 8→3
removes most bridge geometry; final H remains fragmented. Current 4→3 global
x0 is coherent through all nonterminal intervals, so excessive G density is
not the main supported explanation. The next milestone is one fixed-geometry
lifecycle-localization probe tracing assembled local H*, coupled accepted H,
and terminal release at this same canvas.

Phase 8c is complete with no production changes. It causally localizes the
2048x4096 failure: terminal H_star equals fragmented assembled local x0_H, and
production releases it during the dominant sigma-to-zero interval. One fresh
terminal global prediction plus hard projection restores a coherent bridge;
retained high-sigma G3 without a fresh prediction fails completely. Exact hard
projection is not production-qualified because its correction exceeds H_star
RMS and visibly damages fine structure. Any next terminal-policy work requires
a separate design/qualification task rather than directly reverting the
production lifecycle.

Phase 8d is complete with no production changes. Fresh current-G generated
K/V consumed by all terminal local crops repairs fragmentation before assembly
and yields one usable bridge scene with no hard output projection. Required
projection RMS falls by 57.2%, and overlap disagreement falls by 71.9%.
However, the native GPU-resident 18,432-token all-block cache OOMs; the
completed CPU-offloaded diagnostic requires a 5.27 GiB host cache, 290 GiB of
aggregate transfers, and 98.34 s of terminal context-local CUDA time. The
semantic mechanism is established, but no production implementation or
efficiency contract is qualified.

Phase 8e is complete with no production changes. Among the five fixed local
consumption policies, only all 25 blocks retain Phase-8d semantics. Single-only
is a strong but unqualified partial result with floating bridge alternatives;
double-only and both ten-block placements fragment materially. The minimum
qualified set is therefore double 0–4 plus single 0–19. No narrower early
prefix or block-count sweep is justified. Context-through-depth remains the
semantic contract, while its 290 GiB transfer and 95.89 s terminal-local cost
remain unqualified for production.

Phase 8f is complete with no production changes. The full 96×192 current-G
source was executed once and held constant; only post-interaction K/V rows
exposed to all 25 local blocks changed. Uniform 2×2 decimation to 4,608 rows
and 4×4 decimation to 1,152 rows both catastrophically restore independent
bridge systems. Full 18,432-row consumption remains the only qualified density.
The large measured transfer/runtime reductions of the decimated paths are not
usable. No intermediate density or broader sweep is justified by this result.

Phase 8g is complete with no production changes. Exact full and 2×2-selection
controls reproduce Phase 8f. A 4,608-token consumer field formed from
nonoverlapping arithmetic means of all four pre-RoPE K vectors and all four V
vectors, then positioned at exact four-axis geometric cell centers, remains
catastrophically fragmented and is numerically worse than selection. Simple
fixed spatial aggregation is not qualified, and no pooling/density sweep is
pending from this evidence.

Phase 8h is complete with no production changes. An experiment-local native
forward barrier removes the CPU K/V cache/transfer at the first source block,
but suspending all 55 full crop calls OOMs before double block 0 completes.
Only 46 local attention calls reach the barrier; measured peak allocation is
16.30 GiB versus a 1.00 GiB streaming baseline, while the block K/V itself is
216 MiB. No output or semantic-equivalence claim is available. Exact
block-major streaming now requires a separately authorized specialized FLUX
executor; crop-hidden CPU offload was not tested.

Phase 8i is complete with no production or ComfyUI-core changes. The
experiment-local specialized FLUX.2 executor passes ordinary and full-context
one-crop native-equivalence gates bit-exactly, then produces all 55 crop
predictions bit-exactly against a same-process CPU-offloaded reference. It
completes the exact block-major terminal case at 3.32 GiB peak allocated and
6.34 GiB peak reserved, with no CPU K/V cache or host transfer. Terminal
source-plus-local CUDA falls to 65.72 s from roughly 106.39 s. The mechanism is
qualified as an experiment; no production integration or broader adapter
contract is authorized.

Phase 8j is complete as a documentation-only architecture task. The selected
production boundary is an adapter-private `Flux2BlockExecutor` reached through
a model-neutral bulk-region prediction call. `BlueprintCoordinator` remains
the sole owner of assembly, Euler proposals, coupling/terminal policy,
validation, preview, and atomic state commit. The first eligible integration
is terminal-only at exact H=128×256/G=96×192; all nonterminal and other-geometry
execution remains ordinary. The design includes scoped native conditioning
preparation, exact Klein-4B qualification, structural block-barrier memory
ownership, transactional cleanup, and mandatory live/cancellation/OOM gates.
Production remains unchanged pending a separately authorized implementation
task.

Phase 8j implementation is complete. The terminal-only native Klein-4B
specialized executor is production-integrated only for exact H=128×256,
G=96×192, 55-crop execution. All nonterminal intervals and other geometries
retain the ordinary adapter path. The focused suite passes 22/22 tests; two
fresh native 2048×4096 runs are deterministic, emit four previews, preserve
all nonterminal invariants, decode/save through ComfyUI's tiled VAE fallback,
and reproduce the qualified Phase-8i final latent statistics. The specialized
path uses no host K/V cache or transfer and measures 64.43 s CUDA, with
4.80/6.68 GiB whole-run peak allocated/reserved memory. No broader model,
conditioning, geometry, or interval support is qualified.

Phase 9 is complete with no production changes. At 1024×2048 output, the
experiment fixed global model work at G=24×48 and local work at 32×32 per
destination region. The resource invariant and large measured CUDA/wall
reductions are real, but bilinear 64→32 noisy-state restriction plus 32→64 x0
prolongation produces immediate blur and final train/structure ghosting. It
preserves only a coarse bridge scene and fails the local-fidelity gate. This
exact normalized-working-canvas rule is stopped; production remains unchanged.

Phase 9b is complete with no production changes. The inverse 32→64 local
magnification regime separates naïve interpolation from sigma-consistent
working-state construction. Naïve magnification fails. Exact coarse-consistent
sigma-noise restores useful destination-visible local detail and a coherent
principal bridge, but duplicate train/bridge elements and ghosting remain.
The native-working-canvas hypothesis receives partial evidence only; neither
magnified variant is production-qualified.

Phase 9c is complete with no production or ComfyUI-core changes. The exact
Phase-9b C preterminal state was evaluated once at terminal sigma with ordinary
magnified crops, full accepted-H context, and fixed 24×48 G context. Full-H
context materially lowers overlap disagreement but does not remove repeated
bridge/train alternatives; fixed-G context is weaker and also fails. No
context variant is qualified for a four-step trajectory. The next justified
discriminator is scale transport/positional semantics/x0 restriction, not a G
density increase.

Phase 9d is complete with no production or ComfyUI-core changes. Identical
sigma-consistent W tensors were evaluated with compressed destination
coordinates, native unit-spaced local coordinates, and native coordinates plus
per-crop frame-mapped full-H context. Native coordinates alone worsen overlap
agreement and retain repetition; remapped full-H context partially restores
agreement but still fails bridge/train uniqueness. No terminal variant passes,
so no full trajectory is authorized. Prediction restriction/transport is the
next unresolved boundary.

Phase 9e is complete with no production or ComfyUI-core changes. Under the
actual CONST-flow equations, x0 mean, velocity mean, exact delta transport, and
same-sigma re-noise/restriction are the same linear operation because
`D(W)=H_crop`. Runtime differences are below `7.15e-7` max absolute and decoded
semantics are unchanged. No terminal mapping variant qualifies, so one-shot
magnification is stopped. A future architecture probe should maintain
persistent native-scale per-region working states across the trajectory.

Phase 10 is complete with no production or ComfyUI-core changes. The
reconstructed-W control and fifteen independently persistent native W
trajectories share initialization and interval-0 predictions. Persistent W
then diverges sharply: overlap disagreement remains above 0.80 and W/H drift
grows to 0.5234 maximum regional RMS, producing multiple incompatible bridge
scenes. Variant C was not run because B cleanly falsifies independent
persistence rather than yielding an ambiguous useful result. Training-free
fixed-working geometry remains unqualified without a new explicit
global/cross-region W coupling design.

Phase 10b is complete with no production or ComfyUI-core changes. Exact
accepted-H coarse synchronization preserves persistent W null-space detail and
prevents Phase-10 B's multi-scene divergence, but returns to the reconstructed
control's principal-bridge/repeated-support semantic class without a clear
detail benefit. C's terminal overlap RMS is 0.3565 versus A's 0.2594 and B's
0.8031. This is decision gate 2: persistent W is unnecessary complexity under
the tested coarse synchronization contract.

Phase 11 is complete with no production or ComfyUI-core changes. Three full
four-step reconstructed-W trajectories compared local-only execution, fresh
fixed-24×48-G context, and a fresh full-accepted-H context oracle. Full-H
context yields one dominant coherent bridge scene; fixed G retains repeated
bridge/train alternatives. All state/sigma provenance, all-25-block context
consumption, and no-stale-context checks pass. Normalized reconstructed W is
therefore still viable, but fixed-budget global representation sufficiency is
the unresolved research boundary; no production change is qualified.

Phase 12 is complete with no production or ComfyUI-core changes. A fourth
trajectory evolved full accepted H through ordinary global FLUX.2 interaction,
then compressed each block's already-positioned generated K/V to the same
24×48/1,152-position budget as fixed G. It remains semantically fragmented and
does not approach the coherent 8,192-position full-H oracle. Fresh/same-sigma,
all-block, finite, and no-stale-context checks pass. The fixed 1,152-position
consumer capacity is insufficient under the tested area-mean representation;
only a bounded larger post-interaction budget is justified next.

Phase 12b is complete with no production or ComfyUI-core changes. The sole new
condition aggregates each block's full-H generated K/V through exact
nonoverlapping 2×2 groups to `32×64 = 2,048` consumer positions. It improves
the scene relative to the 1,152-position post-interaction control but retains
clear repeated bridge/train/support alternatives and remains below the full-H
oracle. All provenance, block-shape, fresh-context, finite-state, and no-stale
checks pass. Decision 2 authorizes one 4,096-token discriminator only.

Phase 12c is complete with no production or ComfyUI-core changes. The final
brute-force capacity arm uses deterministic anisotropic vertical 2×1
post-interaction aggregation from full `64×128` H to `32×128 = 4,096` context
entries. It preserves every source position exactly once and passes all fresh
same-sigma, block-consumption, immutability, finite-state, and lifecycle checks.
The decoded result is in the oracle's one-dominant-bridge semantic class and
materially closes the numerical gap from 2,048. Information sufficiency is
established for this scene; cheap construction, generalization, compute, and
production suitability remain unqualified.

Phase 13 is complete with no production or ComfyUI-core changes. A fresh
ephemeral `32×128` source is reconstructed from each accepted H using exact
vertical-pair means and full-canvas center coordinates, then globally interacts
through all 25 FLUX blocks. Its local consumers produce an oracle-class coherent
bridge scene without first executing the 8,192-token source. Complete source
mapping, same-state/sigma, per-block K/V divergence, immutability, finite-state,
and lifecycle checks pass. The direct source is semantically viable but slower
than the full-H source on the current backend, so destination-independent
geometry and real execution efficiency remain research work.

Phase 14 is complete with no production or ComfyUI-core changes. At the fixed
`2048×4096` bridge terminal state, a natural `4×2` accepted-H area restriction
to a destination-independent `32×128 = 4,096` source is faster and lower-memory
than the `18,432`-token source under the same block-major executor, but its
decoded normalized-W result remains severely fragmented. The 18,432-token
control also fails after changing to normalized-W consumers, so its historical
direct-crop semantic qualification does not transfer. The next justified probe
is source-input restriction/statistics at the same 4K budget, not a token-count
increase or production change.

Phase 15 is complete with no production or ComfyUI-core changes. A fixed-gain
discriminator restored the 4K source to Phase-13-like variance (`gain=2`) and
near-native variance (`gain=sqrt(8)`) while preserving all Phase-14 geometry and
execution. Neither gain changes the fragmented bridge/train semantic class.
Early K/V telemetry shows transformer normalization largely removes the input
amplitude difference. Simple scalar-statistics correction is stopped; any next
probe must address representation/information loss at the same 4,096 positions.

Phase 16 is complete with no production or ComfyUI-core changes. One fixed
orthogonal representation retained four 4×2 DCT spatial modes and packed them
into the trained 128-channel source interface through deterministic Hadamard
subspaces. It changes reconstruction and overlap metrics but remains in the
same fragmented bridge/train semantic class. Simple deterministic local-mode
packing is stopped. The unresolved boundary is now 4K independent-source
capacity versus the normalized-W context-consumer interface.

Phase 17 is complete with no production or ComfyUI-core changes. A full native
bidirectional source/W oracle at the same 4,096 source positions materially
increases the scale and continuity of bridge systems relative to external K/V,
but still yields several independent whole-scene alternatives. This is the
partial decision gate: both source capacity/representation and consumer
interface likely contribute. Do not resume density or packing sweeps; the next
justified discriminator is where agreement emerges or fails across joint depth.

Phase 18b is complete with no production or ComfyUI-core changes. The bounded
matrix evaluated one full-joint reference, prefix checkpoints S0/S9/S19, and
external-prefix checkpoints D4/S0/S9/S14. No arm reaches S2/S3 or exposes a
coherent scene later destroyed by depth. D4/S0 late-start tails reproduce only
Phase 17's weak S1 class, while later starts regress to S0. Joint-interface
depth scheduling is closed; resumable per-arm artifacts and decoded sheets are
complete. Next research should return to representation/global-state
architecture rather than another block schedule sweep.

Phase 19 is complete with no production or ComfyUI-core changes. Three
resumable terminal arms compared the existing 32x128 fixed-4K control, a fixed
64x64 single-scale whole-canvas source, and an equal-budget explicit
16x32-plus-32x112 hierarchy. Every source is shared across all 55 normalized-W
consumers and remains exactly 4,096 tokens. All outputs remain S0 despite
complete coverage and valid invariants. Gate 1 applies: no trajectory or
production follow-up is authorized. Reassess the global representation/state
architecture rather than another grid allocation or depth/interface sweep.

Phase 20 is complete with no production or ComfyUI-core changes. A fixed
native-coordinate `32x64` Blueprint prediction is semantically coherent (S3),
but exact same-sigma initialization of 55 normalized local W states from that
prediction remains fragmented (S0). Gate 2 applies: bounded global planning is
viable in this discriminator and Blueprint-to-local state transfer/refinement
is the active bottleneck. No trajectory is authorized. One narrow terminal
prediction-anchor discriminator is the next justified experiment.

Phase 20b is complete with no production or ComfyUI-core changes. The single
exact post-forward anchor restores an S3 whole-scene bridge result from the S0
ordinary local assembly while preserving accepted-state and model-input
immutability. Detail is visibly overconstrained and the correction is scene-
scale (`1.119x` local-x0 RMS). A minimal trajectory using the same exact coarse
prediction contract is authorized; production remains unchanged.

Phase 20c is complete with no production or ComfyUI-core changes. Four
resumable intervals advance a persistent bounded Blueprint and destination H
with ordinary Euler updates and atomic pair commits. Blueprint, anchored
assembly, and terminal H remain S3; ordinary local assemblies remain S0.
Detail loss is classified REGENERATING rather than accumulating: fresh local
forwards recreate detail before every exact anchor, but that detail cannot enter
accepted H. The semantic gate passes. Stop before anchor optimization; the next
phase should isolate local-detail/null-space release under retained Blueprint
authority.

Phase 21 is complete with no production or ComfyUI-core changes. The one fixed
destination `2x2` mean/nearest coarse decomposition passed exact coarse and
null-space invariants through all four persistent intervals and retained far
more edge energy than Phase 20c. It did not pass the semantic gate: the terminal
S2 result retains faint repeated towers, ghost cables, and incompatible bridge
geometry. Do not sweep spatial filters/scales. The next justified discriminator
is model-mediated refinement/resampling that preserves Blueprint authority.

Phase 22 is complete with no production or ComfyUI-core changes. At the one
predeclared late sigma `0.25`, 55 ordinary native local refinements initialized
from the coherent Blueprint assemble to S3, retain one bridge/train scene, and
increase gradient RMS by about 40% over the mapped Blueprint. The matched
ordinary-local-anchor control remains S0. The semantic gate passes; stop before
sigma tuning. A minimal persistent/multiresolution trajectory is the next
authorized research question.

Phase 23 is complete with no production or ComfyUI-core changes. The unchanged
Phase-22 local resampling operation was evaluated against all four exact
persistent Phase-20c Blueprint predictions. Blueprint and refined assembly are
S3 at every point, detail gain is repeatable, and no independent bridge/train
alternatives emerge. The temporal semantic gate passes. Stop before cadence or
sigma optimization; the next research decision is terminal-only versus periodic
versus fully interleaved multiresolution sampling.

Phase 24 is complete with no production or ComfyUI-core changes. Terminal-only
and the fixed ordinal-1-plus-terminal persistent-residual arm both remain S3,
but periodic persistence gives no clear perceptual detail advantage and worsens
Blueprint distance, low-frequency discrepancy, and overlap. Terminal-only is
selected. Do not run a fully interleaved arm; the next architecture work should
use a completed bounded Blueprint trajectory followed by one terminal native-
local resampling pass.

Phase 25 is complete with no production or ComfyUI-core changes. The exact
terminal-only architecture is S3 for the reused bridge case and for fixed new
multi-object and astronaut cases; both new ordinary-local controls are S0.
All fixed provenance, coverage, finiteness, resumability, and immutability
checks pass. Semantic generalization is qualified for these three scene classes
at the existing geometry. A Phase-26 production-architecture design task is
authorized; production implementation and geometry generalization are not.

Phase 26 architecture design is complete with no production or ComfyUI-core
changes. The selected first slice uses a dedicated staged node backed
internally by normal `guider.sample()` preparation, an immutable Blueprint-only
run state, four bounded Euler intervals, and one streamed 55-region terminal
resampling pass. It explicitly excludes Candidate-3 DCT/coupling/H state and
the specialized K/V executor. The exact Phase-25 geometry, schedule, model,
conditioning, sigma, and batch profile remain the only supported contract.
The architecture is ready for a separately authorized implementation task.

Phase 27 is complete. `Blueprint Terminal Resampling` is registered as a
separate dedicated node and the legacy Candidate-3 sampler remains unchanged.
Focused tests pass 14/14 and the full custom-node suite passes 36/36. The
persisted Phase-25 multi-object regression and astronaut regression are
bit-exact; the freshly decoded bridge, multi-object, and astronaut outputs are
all S3. Two real BasicGuider executions are deterministic in latent and decoded
RGB, use 4 Blueprint plus 55 local calls and zero destination-sized calls, and
show flat post-region CUDA allocation. No ComfyUI-core file was modified.

Phase 28 is complete. The production node now accepts exactly four enumerated
destination latent profiles: `128x256`, `128x192`, `128x128`, and `256x128`.
Their aspect-preserving Blueprint shapes are capped at 2,048 tokens and every
local call remains `64x64`. The three new semantic cases are S3 versus S0
tiled-local controls, repeat hashes are exact, coverage is complete, and the
Phase-27 reference remains bit-exact. Arbitrary geometry remains fail-closed.

Phase 29 is complete with no production or ComfyUI-core changes. The persisted
Phase-28 Blueprint trajectories were reused and 15 terminal-only sigma branches
were completed at `0.10/0.15/0.25/0.35/0.50`. Every `0.25` branch is bit-exact
to Phase 28 and every result remains S3, but no tested sigma produces a
material cross-case fidelity improvement. Runtime and VRAM are sigma-invariant;
higher sigma increases Blueprint deviation and overlap disagreement. Scalar
terminal-sigma tuning is closed. The next authorized research question is one
fixed Blueprint-guided local-refinement mechanism at the qualified `0.25`
initialization, not another scalar sweep.

Phase 30 is complete at a fail-closed architectural boundary with no model
inference and no production or ComfyUI-core changes. All three persisted
Phase-29 `sigma=0.25` controls validate against their Blueprint fingerprints
and Phase-28 bit-exact regression records. The proposed soft prediction-space
Blueprint attractor cannot be uniquely defined without an arbitrary guidance
scale/metric; the native reference-latent route would instead change
conditioning and add excluded image-context attention. Outcome E is recorded.
The next research task should isolate the Blueprint-to-working-canvas
lift/initialization representation, unless a separate native-reference
conditioning experiment is explicitly authorized.

Phase 31 is complete with no production or ComfyUI-core changes. The persisted
Phase-29 nearest controls remain bit-exact to Phase 28. New fixed bilinear-lift
branches and independent repeats completed for all three cases; all outputs
remain S3 and deterministic, with flat region-barrier allocation. Bilinear
changes the result materially but slightly lowers gradient/detail energy and
does not visibly improve the car/tree/house, astronaut, or bridge structures.
Simple interpolation is closed as the missing detail mechanism. The next
research task should test one richer Blueprint-derived working-state
construction, not another lift kernel.

Phase 32 is complete at a fail-closed pre-inference boundary. All Phase-29
controls, Blueprint fingerprints, region order, and 120 deterministic noise
hashes validate. The proposed restriction-null noise is numerically in the
`avgpool2` nullspace, but its rank-deficient correlated covariance cannot
preserve the qualified i.i.d. sigma-0.25 noise law. No model calls or production
changes were made. The next research task must define a model-mediated or
explicit multi-state coarse/detail stochastic contract rather than another
handcrafted transform or strength sweep.

Phase 33 is complete with no production or ComfyUI-core changes. Exact
one-pass controls were reused, and one additional same-noise sigma-0.25 pass
plus an independent repeat completed for all three cases with per-region atomic
persistence. H2 remains S3 and deterministic, but does not show credible local
structural detail improvement; it chiefly raises texture/alias energy while
roughly doubling Blueprint deviation and increasing low-frequency/overlap
error. A third pass is not authorized. The next research task must use a
principled model-mediated or explicit multi-state coarse/detail contract.

Phase 34 is complete as a CPU-only mathematical/design qualification. The
orthonormal Haar transform, full-rank Gaussian-coordinate law,
`C=2*Blueprint` amplitude mapping, and W/Haar Euler equivalence are validated.
The contract is coherent only with a four-band accepted/output state; the
existing single coarse destination would discard the detail again. One fixed
terminal Phase-35 discriminator is specified using exact Blueprint ownership
of the coarse prediction and unchanged predicted detail bands. No model,
decoded image, production code, or ComfyUI-core path was executed or changed.

Phase 35 is complete with no production or ComfyUI-core changes. One square
terminal case executed 25 native local calls plus a deterministic repeat and
persisted all four Haar bands until a doubled-resolution latent was
reconstructed. Exact Blueprint coarse ownership, band coverage, detail
immutability, reconstruction, flat region-barrier memory, and repeat hashes
all pass. The decode remains one coherent scene but its untouched local detail
bands produce strong lattice/ghost artifacts rather than useful structural
detail. The result is partial/ambiguous and no tuning is authorized.

Phase 36 is complete at a fail-closed schedule boundary. The persisted square
sigma-0.25 S3 control and Blueprint hashes validate, but the selected
terminal-resampling contract defines no unique multi-step local schedule. Its
Blueprint vector has no late points near `0.25`, while the live sampling API
offers nine divergent scheduler families. No transformer forward, local call,
decode, accepted-state mutation, production edit, or ComfyUI-core edit was
performed. A separate schedule-provenance/design audit is required before any
short local trajectory inference.

Phase 37 is complete as a source/config audit with zero diffusion forwards.
Official BFL and Diffusers provenance establishes a shifted flow schedule,
Euler integration, and a partial-denoise index-truncation policy, but not a
unique short trajectory beginning at exact `sigma=0.25`: the full-schedule
step count remains caller-selected and changes the shifted points themselves.
Outcome is `B — CANONICAL FULL/PARTIAL SCHEDULE EXISTS, BUT LOCAL STEP COUNT
REMAINS FREE`. Phase 38 remains blocked unless one fixed empirical local
schedule policy is explicitly authorized. Production and ComfyUI core remain
unchanged.

Phase 38 is complete with one square-scene, four-interval empirical terminal
local schedule and no production/Core changes. The Phase-29 control is
bit-exact; all 25 candidate regions match its initial noise/W fingerprints,
execute four persistent-W Euler intervals, and assemble only once. The 100-call
candidate and independent repeat are bit-exact and remain S3, with no object
duplication or local recomposition. It adds texture/gradient energy but no
credible structural detail, worsens overlap disagreement, and costs about 4x
the local model time at unchanged peak VRAM. Outcome: `B — CHANGES OUTPUT BUT
DOES NOT SOLVE SOFTNESS`. No portrait/landscape generalization or schedule/depth
sweep is authorized.

Phase 39 is complete as a fail-closed mechanism audit. The qualified square
Blueprint/control fingerprints validate, but no executable B arm passed the
fixed-contract gate. Native `reference_latents` is trained separate-canvas
image context, not same-canvas Blueprint authority; prediction/state guidance
needs a free policy or collapses to prior resampling/projection; hidden,
attention, KV, and control hooks are untrained engineering boundaries without
a fixed mapping or strength. Outcome: `E — ARCHITECTURAL BOUNDARY`. Diffusion,
local, destination-sized, and decode call counts are all zero; production and
ComfyUI core remain unchanged.
