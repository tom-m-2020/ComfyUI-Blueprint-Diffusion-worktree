# Decisions

## 2026-08-29 — Separate generic sampling architecture from model-family execution

Blueprint Diffusion will treat the persistent full-canvas sampling coordinator,
geometry, global-context policy, prediction assembly, and lifecycle
instrumentation as generic boundaries. Internal crop coordinates, positional
encoding, token-stream ordering, block execution, selected-query propagation,
and K/V layout belong to explicit model-family adapters.

Reason: FLUX.2, Z-Image, and Anima can all participate in one full-canvas
trajectory, but their transformer streams and intervention boundaries differ
materially. Encoding FLUX double/single block assumptions into a generic API
would make nominal cross-model support misleading.

This decision does not select a final Blueprint algorithm. Three candidate
architectures remain experimental, ranked in `BLUEPRINT_ARCHITECTURE_AUDIT.md`.

## 2026-08-29 — Test prediction-level global/local fusion before sparse executors

The next experiment will test reduced-resolution global prediction plus local
residual correction using ordinary model calls before implementing a
model-specific selected-token executor.

Reason: this is the smallest experiment that can falsify the central semantic
hypothesis. Sparse execution would optimize a global-context mechanism whose
ability to preserve T2I composition is not yet established.

## 2026-08-29 — Do not use unfiltered scalar local residual fusion

The Phase 2 experiment found that the reduced global prediction carried a
coherent whole-scene plan, but `local - mapped_global` also carried incompatible
low-frequency bridge geometry and duplicated objects. A scalar correction
strength only moved between blurred global structure and sharp tiled structure;
it did not isolate local fidelity.

Therefore the unfiltered scalar residual is not an acceptable Candidate-1
fusion contract. Candidate 1 remains open in revised form because the global
branch itself showed useful compositional control. The next experiment will
test the smallest fixed low/high-frequency separation before any production or
sparse-execution work.

## 2026-08-30 — Do not adopt fixed output-space frequency splitting as the fusion contract

The Phase 2b experiment found that a sigma-1 high-pass correction can restrain
global-plan replacement and recover useful detail, but bridge geometry, tower
silhouettes, and duplicated lighthouse structure remain in the high-pass
component and final image. A broader sigma-2 split admits still more of the
alternative composition.

Therefore a fixed spatial-frequency split is not an acceptable final
Candidate-1 fusion contract, and further scalar/filter-scale sweeps are not the
next project milestone. Candidate 1 remains viable only in revised form; the
next experiment should test whether explicit compact global context prevents
semantic divergence during local model execution. This decision adds no
production implementation or generalized infrastructure.

## 2026-08-30 — Prioritize Candidate 2 for the next semantic experiment

The Phase 2c one-evaluation probe found that same-sigma compact-global K/V
inside all FLUX.2 transformer blocks removed a crop-local invented tower and
moved bridge geometry substantially closer to the dense reference without
destroying local structure. This is stronger semantic evidence than either
unfiltered or fixed-frequency output fusion produced.

Candidate 2 is therefore the leading mechanism for the next experiment. This
is not a final architecture selection and does not authorize a generalized
sparse executor, cache lifecycle, production node, or cross-model API. The next
test remains one evaluation and extends only to all three existing crops.

## 2026-08-30 — Advance Candidate 2 to a minimal trajectory test

The Phase 2d probe shared one fresh compact-global K/V capture across all three
crops. Every crop moved toward dense, assembled low-frequency and total error
fell, and pre-blend overlap disagreement dropped by about 63%. Visually, the
context assembly suppressed the left/center crop-local towers and recovered a
single coherent bridge organization with one lighthouse and one right tower.

Candidate 2 has therefore passed the single-evaluation semantic gate and should
advance to a minimal four-step trajectory with fresh global context at every
evaluation. This is not a production architecture decision: stale caching,
block selection, sparse execution, VRAM optimization, generalized lifecycle,
and cross-model APIs remain out of scope until trajectory behavior is known.

## 2026-08-30 — Keep Candidate 2 active but do not optimize it yet

The Phase 2e trajectory showed persistent numerical and visible benefit from
fresh compact-global context across all four Euler evaluations, with detailed
late-stage bridge and train structure. It also produced a surviving duplicate
lighthouse-like object. Candidate 2 therefore remains the leading architecture
but has only partially passed the trajectory semantic gate.

Do not proceed to stale caching, block selection, sparse execution, or
production integration. The next decision must distinguish compact-context
information loss from a limitation of the external-K/V integration mechanism,
using dense full-canvas K/V only as a late-evaluation diagnostic reference.

## 2026-08-30 — Treat global representation quality as Candidate 2's next bottleneck

The Phase 2f controlled diagnostic changed only external global K/V density and
source. Dense full-canvas K/V removed the duplicate lighthouse retained by the
512-token compact branch and cut dense-crop RMS from 0.372 to 0.180, while using
the same local execution and integration policy.

Candidate 2 therefore remains active, with global representation quality/token
density as the next research bottleneck. This does not authorize dense global
execution as an efficient architecture, and dense K/V's residual stone-tower
divergence means the integration mechanism is not exact. Test one intermediate
density before considering multiscale design, trajectory changes, caching,
sparse execution, or production infrastructure.

## 2026-08-30 — Do not pursue a uniform-density sweep

The fixed 1152-token `24x48` branch improved numerical agreement over the
512-token branch but retained the target duplicate lighthouse and stone
structure. The useful semantic threshold was therefore not reached at 56.25%
of dense global token density, while changed local attention-context work had
already risen to 75% of the dense-context value.

Do not continue a uniform-resolution parameter sweep. Candidate 2 remains
active because dense global context removes the duplicate through the same
integration path, but the next diagnostic should test whether a fixed
multiscale/nonuniform representation can preserve the missing localized
evidence at approximately the failed intermediate token budget. This does not
authorize adaptive selection, caching, sparse execution, or production code.

## 2026-08-30 — Reject naive failure-region multiscale concatenation

At the same 1152-token external-context budget and identical local attention
dimensions, concatenating 512 coarse whole-canvas K/V with 640 native-density
tokens from the affected center region increased error and reinforced the
duplicate lighthouse/stone alternative relative to uniform 1152.

Do not increase density in that region or advance this construction toward a
trajectory. Candidate 2 remains active because dense distributed context still
removes the duplicate, but the localization hypothesis is weakened. The next
probe may distinguish distributed nonlocal evidence from failure-local density;
it must remain fixed, one-state, zero-update, and experimental. This decision
does not authorize adaptive selection or generalized multiscale infrastructure.

## 2026-08-30 — Stop simple 1152-token spatial-reallocation probes

Distributed native-density samples outside the active crop eliminated the
failure-local amplification but did not suppress the original duplicate or
improve over uniform 1152. Along with the failed uniform-intermediate and
failure-local probes, this substantially weakens simple token placement as the
explanation for dense context's semantic advantage.

Do not proceed automatically to another density/layout sweep. Candidate 2 is
not rejected because dense K/V still removes the duplicate through the same
local integration path, but the next investigation must first reassess what
dense source execution contributes beyond external token count and placement.
No trajectory, caching, adaptive selection, sparse executor, or production work
is authorized by this result.

## 2026-08-30 — Treat source global interaction as a Candidate-2 requirement

With all 2048 native source positions retained, blocking image-to-image source
attention in all 25 blocks restored the duplicate lighthouse and materially
degraded the local prediction. Block-0 captured K/V remained identical and
later blocks diverged, isolating the accumulated source hidden interaction.

Candidate 2 must therefore account for global interaction during context
construction; a final-interface bag of independently formed spatial K/V is not
a sufficient architectural contract. This is not a decision to run ordinary
dense source forwards in production, nor evidence that every block or global
edge is necessary. Do not proceed to production, caching, or a generalized
sparse executor until a cheaper sufficient interaction pattern is established.

## 2026-08-30 — Reject isolated 16x16 windows as the source-context contract

Fixed 16x16 windows preserved meaningful within-window hidden interaction and
12.5% of dense image-image connectivity, but the duplicate lighthouse returned
at approximately the no-mixing severity and numerical gains were marginal.

Do not adopt independent fixed windows or automatically sweep window sizes.
Candidate 2 continues to require cross-window information flow, though this
result does not prove direct fully dense attention is necessary: gradual or
structured cross-window propagation remains untested. No production executor,
trajectory, caching, or generalized infrastructure is authorized.

## 2026-08-30 — Do not equate cross-depth reachability with sufficient context

Alternating shifted windows made all source tokens transitively reachable from
the full canvas, yet the duplicate lighthouse remained and numerical behavior
was worse than fixed windows and no mixing. Graph connectivity by itself is not
a sufficient Candidate-2 source-context contract.

Do not continue with another window topology automatically. Candidate 2 remains
supported only when its context source performs broad dense interaction in the
tested case; whether a different cheap mechanism can preserve that information
is unresolved. No trajectory, caching, sparse executor, or production
infrastructure is authorized.

## 2026-08-30 — Do not treat five dense source blocks as sufficient

The first five dense double blocks recover substantially more numerical and
geometric benefit than the final five dense single blocks, but neither range
suppresses the duplicate lighthouse at a 20% dense-edge budget.

Record early source depth as the stronger locus of broad-interaction value, but
do not adopt an early-five policy or automatically sweep block counts. Candidate
2 still lacks a cheap sufficient source-context contract. No trajectory,
caching, sparse executor, or production implementation is authorized.

## 2026-08-30 — Reject isolated dense refresh as a maintenance contract

Adding dense source blocks at ordinals 10, 15, and 20 to the early-five policy
increased the dense edge budget from 20% to 32% but slightly worsened final
error and left the duplicate lighthouse unchanged. Later refreshes briefly
reduced some K/V divergence, yet restricted blocks erased those gains.

Do not adopt this early-plus-refresh policy or sweep refresh schedules. Candidate
2's cheap source-context premise is materially weakened under hard no-image-
image restriction. No trajectory, caching, sparse executor, or production work
is authorized.

## 2026-08-31 — Test Candidate 3 with a hard accepted-state global anchor first

Candidate 2 semantic work is paused. The first Candidate-3 experiment will use
two persistent same-sigma states and independently compute an ordinary
low-density global Euler proposal plus an assembled high-resolution local Euler
proposal. It will then keep the global proposal authoritative and project the
accepted high-resolution state to satisfy `D(H_next) = G_next` exactly.

This parameter-free rule is selected ahead of soft anchoring and bidirectional
consensus because it gives the strongest causal discriminator without allowing
crop-local low-frequency alternatives to rewrite the global trajectory. The
experiment must include an uncoupled dual control and must qualify the mapped-
noise global branch independently. This is a decision about one falsification
probe, not adoption of Candidate 3, a production sampler, or a claim of compute
efficiency.

## 2026-08-31 — Reject exact hard projection as an all-step acceptance rule

The hard global anchor passed every lifecycle and numerical integrity check and
caused later local proposals to follow the valid low-density trajectory's
single-bridge organization. It therefore supplies positive evidence for
persistent two-state coupling.

Do not adopt exact `D(H_next) = G_next` projection after every proposal as the
Candidate-3 contract. Projection magnitude grew through the trajectory and
reached 38.4% of proposed-H RMS at the terminal interval, where it visibly
ghosted/doubled otherwise useful local cable, deck, train, support, lighthouse,
and tower detail. This decision rejects the tested all-step hard rule, not
Candidate 3. No soft coupling, consensus, production sampler, or generalized
infrastructure is authorized by this result.

## 2026-08-31 — Retain hard coupling only for nonterminal accepted states

The terminal-release control was identical to hard anchoring through every
state and proposal that could influence a later model evaluation. Accepting the
terminal local proposal without projection preserved the global single-scene
organization, removed most projection ghosting, and materially improved both
absolute and low-frequency error versus dense.

For Candidate-3 research, treat hard coarse equality as an intermediate-state
organization policy, not a required terminal-state invariant. The decoded
terminal `H` may intentionally differ from final `G` after both have taken the
same Euler interval. This is a bounded research decision from one controlled
case; it does not authorize production integration, soft anchoring, schedule
sweeps, or generalized lifecycle infrastructure.

## 2026-08-31 — Keep terminal release, but do not productionize Candidate 3

The fixed terminal-release lifecycle again outperformed all-step hard
projection numerically and visually under a different prompt and seed. It did
not cause tiled-only's duplicate person/large-tree scene to return, confirming
that local terminal authority is compatible with meaningful retained global
organization in a second case.

Retain terminal release as the preferred Candidate-3 research lifecycle, but do
not treat Candidate 3 as production-qualified. A small second car and two extra
thin trees survived both hard and released terminal states, showing that exact
intermediate `D(H)=G` does not guarantee object uniqueness. Do not proceed
automatically to soft anchoring, production integration, or another policy
sweep.

## 2026-08-31 — Select 24x48 block-DCT geometry for one bandwidth probe

For the next Candidate-3 runtime experiment, use a `24x48` global state formed
by independent `4x4 H -> 3x3 G` constant-preserving orthonormal DCT restriction,
with zero-padded DCT synthesis as U. This pair preserves aspect ratio and
satisfies `D(U(G))=G` in exact arithmetic without a global resize pseudoinverse.

This selection authorizes only one controlled higher-bandwidth experiment with
the terminal-release lifecycle fixed. It does not adopt DCT coupling for
production and does not authorize a geometry/operator sweep. Specifically do
not substitute ordinary bilinear resizing, an anisotropic `16x64`/`32x32`
state, or a global pseudoinverse without new evidence.

## 2026-08-31 — Retain 24x48 block-DCT as the leading Candidate-3 geometry

Under the fixed Phase-3c scene and lifecycle, 24x48 block-DCT terminal release
removed the small second car and two thin extra trees that survived 16x32
mean/nearest coupling. It also improved final and low-frequency error versus
dense, preserved local detail, and showed no visible macroblock artifacts.

Retain this geometry for further Candidate-3 research, but do not productionize
it. The experiment changed both spatial bandwidth and the mapped-noise variance
(0.563 versus about 0.25), so it does not establish density as the sole cause.
It also raises runtime and global token work. No geometry sweep, alternate
operator, soft coupling, or production optimization is authorized by this
decision.

## 2026-08-31 — Do not attribute the 24x48 gain to variance alone

At matched initial G variance, scaled 16x32 improves over original 16x32 but
retains secondary tree/vehicle-like structure and worsens terminal overlap.
The 24x48 block-DCT branch alone removes all known extras and achieves the best
numerical and overlap agreement.

Retain 24x48 block-DCT as the leading Candidate-3 research geometry. Record
mapped-state scale as a real contributing factor, not the primary complete
explanation. This does not distinguish token density from DCT retained content
and does not authorize another scale/operator sweep or production work.

## 2026-08-31 — Select Candidate 3 and an explicit custom Euler sampler boundary

Select Candidate 3 for the initial Blueprint Diffusion implementation with the
qualified 24x48 block-DCT global state, hard nonterminal accepted-state
coupling, and terminal release.

Implement lifecycle ownership as an explicit custom ComfyUI `SAMPLER` used
through the existing guider and `SamplerCustomAdvanced`. Do not implement the
algorithm as a stateless model patch: two persistent states and atomic pair
acceptance belong at the sampler-interval boundary. Do not use a hidden
`SAMPLER_SAMPLE` model wrapper for the first slice because it obscures the
Euler-only restriction and makes ordinary sampler selection misleading.

Separate the generic coordinator/state/policy/operator/planner interfaces from
the FLUX.2 coordinate adapter. This is an interface boundary, not a promise of
Z-Image or Anima support; those families remain fail-closed until qualified.
The first production slice is deliberately restricted to the exact native
FLUX.2 Klein, CFG-1 T2I, fixed geometry/crops, zero-churn Euler contract.

## 2026-08-31 — Accept the first Candidate-3 production slice as equivalent

The production sampler is bit-exact with the qualified research harness at all
required prediction, proposal, accepted-state, initialization, and final-output
boundaries. Retain the explicit custom `SAMPLER`, immutable pair-state,
block-DCT geometry, fixed crop assembly, terminal policy, and FLUX.2 adapter as
the initial production structure.

This accepts only equivalence to the fixed research contract. It does not
authorize broader geometry, CFG, samplers, editing, model families, caching,
sparse execution, or performance claims.

## 2026-08-31 — Qualify the fixed Candidate-3 slice for ordinary ComfyUI use

Accept `BlueprintCandidate3EulerSampler` as live-qualified for the exact native
FLUX.2 Klein 4B, CFG-1, 1024x512 empty-latent, four-interval contract through
`SamplerCustomAdvanced`. Retain strict rejection of every unqualified path.

Make full denoise explicit by requiring the first sigma to equal 1.0. The live
partial-denoise control demonstrated that ComfyUI's broader `max_denoise`
classification cannot define this fixed production boundary. This decision is
a validation constraint and does not alter Candidate-3 coupling or authorize
additional schedules.

## 2026-08-31 — Generalize Candidate-3 geometry without changing semantics

Support target latent grids with both axes at least 32 and divisible by four.
Instantiate the existing 4-to-3 block-DCT geometry from each run's accepted H,
so G is exactly three quarters of H on each axis and the same right-inverse
coupling contract applies blockwise.

Use deterministic 32x32 crops with stride 24 and append the end-aligned final
start on each axis. This preserves the qualified 32x64 crop plan exactly while
providing full normalized coverage for other aspect ratios. Keep local FLUX.2
coordinates as absolute crop offsets and map the global grid to both full-canvas
coordinate endpoints independently by axis.

Reject axes below 32 rather than padding local model inputs or inventing
out-of-canvas coordinate semantics. This phase does not change Candidate-3
lifecycle, terminal release, CFG/model/sampler boundaries, or authorize
optimization.

## 2026-08-31 — Target local crop model work in Phase 6b

The Phase-6a baseline establishes that Candidate-3 currently trades runtime for
peak allocator memory. At 1024x2048 it is 1.90x slower than dense while reducing
peak allocated/reserved memory by 331/744 MiB. Local crop forwards consume
14.908 seconds and 76.6% of measured Blueprint model time; DCT, assembly,
coupling, Euler arithmetic, and coordinator overhead together remain below 1%.

Select local crop model work—the number of local forwards and repeated
local/overlap token execution—as the one Phase-6b cost center. This does not yet
select crop geometry, batching, selective execution, caching, or another
optimization mechanism. Do not prioritize global cadence or coupling kernels
without contrary measurements.

## 2026-08-31 — Do not batch Blueprint crops through scalar FLUX.2 RoPE options

Reject naive local-crop batching at the current adapter boundary. Native FLUX.2
constructs one scalar-offset image-ID grid and repeats it across batch, so a
batched call would assign one crop's absolute coordinates to every element.
Vector offsets are rejected by the native grid constructor and the public
forward does not accept explicit per-batch image IDs.

Do not publish incorrect crop batching or benchmark it as an optimization.
Correct batching requires a separately authorized adapter/backend coordinate
change and a new sequential-versus-batched prediction equivalence gate.

## 2026-08-31 — Do not promote the runtime coordinate override to production

The experiment-only `process_img` override proves that distinct per-batch crop
IDs can be constructed narrowly and exactly without modifying transformer
blocks. It does not make crop predictions equivalent: the qualified native
W4A8 path changes materially at batch size two even for duplicated input and
identical scalar coordinates.

Do not add this override to `Flux2Adapter` or benchmark it as Candidate-3
optimization. The remaining blocker is below coordinate preparation and
requires a separately scoped guider/model/backend investigation. Accepting the
observed drift would change the qualified prediction ensemble.

## 2026-09-01 — Keep crop batching blocked after embedding-path localization

The duplicated-input control localizes the first B=1/B=2 mismatch to the
unquantized bfloat16 `Flux.img_in` linear output, before transformer attention.
The mismatch is small there but grows materially by final projection. Do not
relax the sequential-equivalence gate or modify Blueprint semantics. A future
attempt must explicitly qualify batch-shape-invariant embedding/backend linear
execution before crop batching can be reconsidered.

## 2026-09-01 — Do not productionize B=2 crop scheduling at three-crop geometry

The experiment-only trajectory retained the two qualified scenes' visible
semantics, so the known numerical B=1/B=2 drift is not an immediate semantic
failure in these cases. However, the useful warm run improved sampling wall
time by only 3% and increased peak allocated/reserved CUDA memory. Keep the
scoped coordinate override and batched scheduling out of production; the
measured benefit does not justify the numerical and memory tradeoff.

## 2026-09-01 — Reject cross-crop overlap K/V reuse under the fixed crop contract

Do not add the experiment-local sparse executor or cross-crop K/V cache to
Candidate-3 production. Although it genuinely omits repeated generated-token
work, overlap features from one crop encode that crop's surrounding context and
become invalid for the next crop after one transformer block. The resulting
active prediction and decoded geometry fail badly, while measured time improves
only 4% and peak memory increases. Exact replacement would require recomputing
the skipped tokens' hidden evolution, defeating the optimization premise.

## 2026-09-01 — Retain stride 24 overlap after the Phase-6f discriminator

Do not change Candidate-3 production crop stride from 24 based on this probe.
Stride 32 materially lowers local calls and runtime, but a person crossing a
tile boundary develops duplicate/offset body structure and both tested scenes
show higher adjacent-boundary discontinuity. Stride 28 remains visually close
to current but deterministic end alignment leaves the same crop count and
executed-token work at both qualified geometries. Candidate-3's persistent
global trajectory does not eliminate the tested need for local overlap.

## 2026-09-01 — Do not adopt one fixed larger local window after Phase 6g

Larger windows have a measured, geometry-dependent crossover rather than a
universal advantage. Keep the production 32x32/stride24 policy unchanged until
a shape-aware policy is separately qualified. The next eligible configuration
is 64x64/stride48 at H=64x128: it improved warm sampling time 16.6% with
preserved visible semantics, but its higher adjacent-boundary metric requires
additional boundary-placement and long-structure validation first.

## 2026-09-01 — Select larger windows for exactly two qualified H geometries

Use 64x64/stride48 only for H=64x128 and 48x48/stride36 only for H=48x96.
Keep 32x32/stride24 for every other target. The two mappings passed multiple
semantic boundary/long-structure scenes, bit-exact recomputation, Candidate-3
lifecycle checks, material warm runtime gates, unit tests, and live ComfyUI.
Do not infer a largest-fitting or interpolated policy from these two points.

## 2026-09-01 — Advance terminal-only global refresh omission, not stale cadence

The interval-3 global model call is causally dead for returned H under terminal
release, and stale reuse there is exactly output-equivalent while saving 6.6%
mean wall time. Advance this narrow lifecycle optimization for a separate
production/API qualification. Do not skip intervals 0-2: stale estimates used
on nonterminal intervals alter accepted H and can visibly reorganize long
structures. Phase 6i makes no production change.

## 2026-09-01 — Omit terminal global execution and retain G as unsynchronized diagnostic state

Production no longer performs a global prediction or constructs `G*` when
`sigma_next == 0`. Terminal release accepts the fresh local `H*` directly.
Keep the last accepted nonterminal `G` tensor in the frozen state rather than
making `G` optional, but label it `retained_preterminal_unsynchronized` and do
not assert terminal `D(H)=G`. This is the smallest honest API contract: it
preserves nonterminal state ownership and avoids a broad optional-state
refactor while eliminating one of four global forwards. Frozen-baseline tensor
regression, unit tests, warm timing, and live ComfyUI workflows all qualify the
change. This decision does not authorize skipping any nonterminal global call.

## 2026-09-01 — Advance variable Euler step counts to a separate production task

The unchanged Candidate-3 lifecycle passed 4/8/12/20-step full-denoise
CONST-flow schedules across four semantic controls. Repeated hard projections
did not accumulate: mean and worst projection/H* ratios declined with schedule
length, all nonterminal invariants held, and decoded composition/boundaries
remained stable. Production remains unchanged in Phase 7a. A following task may
generalize only schedule validation and loop cardinality while retaining sigma
1 start, exact-zero termination, strict descent, fresh global work on every
nonterminal interval, and the terminal global omission. This evidence does not
authorize alternate samplers, partial denoise, or changed coupling.

## 2026-09-01 — Support variable full-denoise Euler cardinality in production

Accept any one-or-more-interval sigma schedule that begins at exactly 1, ends
at exact zero, is finite, remains positive before the terminal value, and is
strictly decreasing. Keep CONST-flow/full-denoise enforcement and every prior
model, CFG, conditioning, geometry, crop, DCT, coupling, cadence, and terminal
policy unchanged. The loop, preview total, telemetry ordinals, and terminal
selection derive only from `len(sigmas)-1`. This generalization is supported by
Phase-7a semantic evidence, bit-exact production/reference regression through
20 steps, the unchanged frozen four-step baseline, and fresh live ComfyUI
qualification. It does not imply support for partial denoise or other samplers.

## 2026-09-02 — Stop the scaling ladder and test bounded global-state density

Do not extend Phase 8a beyond 2048x4096 or treat successful execution as a
usable-resolution qualification. The uncapped `G=(3/4 H_y, 3/4 H_x)`
trajectory remains coherent at 1536x2048 but fragments globally at larger
canvases while matched dense generation remains coherent. No evidence yet
assigns this failure to local crops. Test the narrower causal hypothesis that
the authoritative global trajectory exceeded FLUX.2 Klein's useful
spatial/token regime by comparing bounded/adaptive G density at one known
failing geometry. Do not promote a production geometry change until that
discriminator succeeds.

## 2026-09-02 — Do not advance simple bounded block-DCT global density

Reject 8→4 and 8→3 per-block DCT restriction as the next Candidate-3
production direction. Although they reduce global CUDA time substantially,
they do not restore the 2048x4096 scene: 8→4 introduces disconnected bridge
alternatives in the first global prediction and 8→3 removes essential bridge
geometry. The current 4→3 global prediction remains coherent even though final
H fragments. Before changing coordinates or coupling, localize the first
fragmented H boundary under current geometry by decoding assembled local H*,
hard-coupled accepted H, and terminal release in one fixed-canvas probe. This
decision does not authorize another DCT ratio or variance sweep.

## 2026-09-02 — Treat terminal local-only authority as the large-scale failure mechanism

At 2048x4096, production terminal release publishes the fragmented assembled
local x0_H because the terminal Euler H_star is numerically equal to x0_H. A
fresh terminal global proposal followed by exact coarse projection restores a
single coherent bridge from the identical preterminal state; retained G3 does
not, because it remains a high-sigma state. Record terminal local-only
authority as the causal failure mechanism for this schedule. Do not restore
hard terminal projection in production automatically: its projection RMS is
larger than H_star RMS and it introduces fine-detail softness/ghosting. A new
terminal coupling rule, if pursued, requires a separate research design and
qualification task.

## 2026-09-02 — Advance terminal internal global context as a semantic mechanism only

At the fixed failing 2048x4096 state, fresh current-G K/V inside all 25 FLUX.2
blocks makes terminal crop predictions mutually compatible and restores a
single scene without hard output projection. Record internal terminal global
context as the first mechanism that both preserves detail and repairs the
large-canvas failure before assembly. Do not productionize the tested path:
18,432-token all-block K/V OOMs when retained on the 12 GB GPU, while CPU
offload incurs a 5.27 GiB host cache, roughly 290 GiB aggregate transfers, and
large runtime cost. Candidate-3 remains the state/lifecycle architecture; this
does not reopen Candidate 2 as a separate trajectory or authorize a new
compression/density sweep.

## 2026-09-02 — Retain all-depth local context as the current semantic contract

Do not advance double-only, single-only, early-ten, or late-ten local context
consumption. Only all 25 FLUX.2 blocks reproduce the qualified Phase-8d scene;
single-only retains a main bridge but fails the floating-fragment gate, and the
other subsets regress further. Treat context persistence through local depth as
the current semantic requirement. This does not authorize production work or
another contiguous block-count sweep: the tested all-depth CPU-offloaded path
remains far too transfer- and runtime-heavy.

## 2026-09-02 — Reject fixed regular post-interaction K/V decimation

Do not advance one-per-2×2 or one-per-4×4 spatial sampling of full-source
current-G K/V. Even after ordinary dense 96×192 source interaction, 4,608
consumer positions catastrophically fragment the terminal bridge scene; 1,152
is worse. Retain all 18,432 positions as the current qualified semantic
contract. This is a rejection of these fixed regular layouts, not evidence
that every token is irreducible or authorization for learned compression,
saliency, quantization, streaming redesign, production changes, or a density
sweep.

## 2026-09-02 — Reject fixed 2×2 arithmetic K/V aggregation

Do not advance arithmetic mean pooling of each 2×2 post-interaction current-G
K/V cell. Pooling every pre-RoPE K and V contributor exactly once at a
geometric-center RoPE coordinate does not improve the failed 4,608-token scene;
it is numerically and semantically at least as fragmented as regular selection.
This decision does not authorize another pooling function, density/window
sweep, learned compression, token merging, quantization, or production change.

## 2026-09-02 — Do not implement block-major streaming with suspended native forwards

Reject the experiment's thread/barrier suspension of 55 monolithic FLUX.2
calls. Although it eliminates the all-block CPU cache and repeated host K/V
transfer at the first block, retained native call-frame intermediates drive
peak allocation to 16.30 GiB and OOM before block 0 completes. Any exact
block-major continuation requires a specialized executor with explicit
block-to-block hidden ownership and per-crop temporary release. This decision
does not authorize that executor, CPU hidden offload, production integration,
or any semantic compression.

## 2026-09-02 — Qualify explicit FLUX block execution experimentally

The Phase-8i specialized executor is the first mechanically qualified way to
execute the exact all-depth, 18,432-token current-G context contract without
the Phase-8h suspended-call-frame OOM. It retains only explicit block outputs
for each crop and one source block K/V, reproducing all 55 crop predictions
bit-exactly while eliminating the all-block CPU cache and host transfer. Record
this as experiment-level feasibility only. Do not modify production, generalize
the executor, add backend support, or infer support for other models/CFG/
conditioning without a separately authorized architecture and qualification
task.

## 2026-09-03 — Select terminal-only adapter-private FLUX executor architecture

For the first production integration, place explicit double/single block
orchestration in a private `Flux2BlockExecutor` owned per call by
`Flux2Adapter`. Extend the adapter boundary only with a model-neutral ordered
bulk-region prediction operation. Keep accepted state, region planning,
assembly, Euler proposals, D/U coupling, terminal release, validation, preview,
and atomic commit in `BlueprintCoordinator`/`BlueprintEulerSampler`.

Activate specialized execution only for the qualified terminal H=128×256,
G=96×192 case. Leave nonterminal intervals and every other geometry on the
existing ordinary path. At the qualified geometry, unsupported model profile,
conditioning, wrappers, coordinates, or runtime state must fail closed rather
than fall back to the semantically failed local-only terminal evaluation.

Use normal ComfyUI CFG-1 conditioning preparation through a scoped
diffusion-model wrapper and validate exactly one prepared native call per
source/crop. Do not rebuild conditions from `guider.conds`, introduce a generic
FLUX-shaped execution session, or transfer whole-interval/state ownership to an
adapter. The source executes all 25 blocks to provide K/V but does not run an
unused terminal final projection or claim synchronized final G. This decision
authorizes architecture selection only; production code remains unchanged.

## 2026-09-03 — Integrate only the qualified terminal specialized path

Production now dispatches the private native FLUX block executor only for the
exact terminal H=128×256/G=96×192, 32/24, 55-crop Klein-4B case. Keep every
nonterminal evaluation and every other geometry on the previous ordinary
prediction path. Unsupported profiles or wrapper/conditioning behavior at the
qualified geometry fail closed; do not silently fall back to the known
fragmented local-only terminal result.

The adapter returns only ordered region predictions and compact telemetry.
Assembly, Euler arithmetic, terminal release, preview, and atomic state commit
remain coordinator-owned. This decision does not qualify other FLUX variants,
models, CFG modes, conditioning schemes, geometries, or context execution on
nonterminal intervals.

## 2026-09-03 — Reject naive bilinear normalized local-state transport

Do not productionize the Phase-9 rule that bilinearly restricts a 64×64
destination latent region to a 32×32 local working canvas and bilinearly
prolongs its denoised estimate before assembly. Although it establishes fixed
global/per-local token budgets and materially reduces runtime, it discards the
local fidelity Blueprint is intended to recover. This rejects that transport
rule, not the broader objective of destination-independent working geometry.

## 2026-09-03 — Do not productionize Phase-9b local magnification

Reject naïve bilinear 32→64 working-state magnification. Retain
sigma-consistent, exactly coarse-preserving magnification only as a research
result: it proves that native working resolution can contribute useful detail,
but its repeated semantic structures, ghosting, and 3.64× measured wall cost
do not satisfy production quality or efficiency gates. This does not reject
the broader fixed-working-geometry architecture.

## 2026-09-03 — Do not advance Phase-9c fixed-G context to a trajectory

Do not run or productionize the fixed 24×48 G context variant. At the exact
Phase-9b C terminal state it neither removes repeated bridge/train semantics
nor materially approaches the full-H context's overlap agreement. Full-H
context itself remains semantically fragmented despite a large numerical
agreement improvement, so increasing G density is not the next justified
experiment. Local scale transport, positional semantics, or prediction
restriction must be isolated first.

## 2026-09-03 — Do not advance Phase-9d native working coordinates

Do not run a full trajectory or productionize the native-unit-coordinate
construction. With W, accepted state, x0 restriction, and assembly held fixed,
native coordinates do not remove repeated bridge/train alternatives, and even
per-crop full-H context expressed in the same magnified frame remains
semantically insufficient. This result directs the next discriminator to
prediction restriction/transport; it does not authorize a coordinate-scale
sweep, denser G, or additional context mechanisms.

## 2026-09-03 — Stop terminal one-shot magnification transport variants

Do not advance velocity, delta, or same-sigma re-noise transport to a full
trajectory or production. Under linear 2×2 mean restriction and exact
`D(W)=H_crop`, they are algebraically identical to averaging x0 and reproduce
the same semantic failure. Further terminal mapping variants are not justified.
The next architecture, if pursued, should test persistent per-region
native-scale working states rather than reconstructing W from H only at an
evaluation boundary.

## 2026-09-03 — Reject independent persistent native-scale W trajectories

Do not advance or productionize Phase-10 variant B. Persistent W states retain
detail but diverge into multiple incompatible regional scenes, with terminal
prediction overlap RMS 0.8031 and maximum W/H coarse drift 0.5234. Variant C's
conditional gate was not met because B is not semantically useful or merely
ambiguous. Any later persistent-W architecture requires a separately designed
and authorized global/cross-region coupling contract; do not return to
terminal resize/transport variants.

## 2026-09-03 — Do not advance coarse-synchronized persistent W

Phase-10b C preserves the W null-space component and exact destination-scale
agreement, but it only returns the output to the reconstructed control's
semantic class and does not demonstrate superior detail or consistency. Treat
persistent W as unnecessary complexity under this synchronization contract.
Do not productionize it or add inside-transformer context automatically; any
new persistent-W coupling mechanism requires a separate architectural task.

## 2026-09-03 — Retain normalized reconstructed W; isolate global representation

Phase 11 demonstrates that complete accepted-H context inside every local
FLUX.2 block can suppress reconstructed-W bridge/train repetitions, while the
same mechanism sourced from fixed 24×48 G cannot. Do not reject normalized
native working canvases as a training-free architecture on the basis of the
earlier local-only failures. Also do not productionize the full-H oracle or
increase G density automatically: its token, transfer, and runtime costs violate
the fixed-budget objective. The next authorized research direction, if pursued,
is a bounded global-representation sufficiency discriminator that preserves a
fixed global model-token budget and measures against the full-H semantic oracle.

## 2026-09-03 — Treat 1,152-position context capacity as insufficient

Phase 12 shows that moving compression after ordinary full-H global interaction
does not recover the Phase-11 oracle at a 24×48/1,152-position consumer budget.
Do not attribute the failure solely to pre-interaction G construction, and do
not productionize the post-interaction pooling path. If research continues,
test exactly one larger, clean post-interaction capacity (`32×64 = 2,048` via
nonoverlapping 2×2 aggregation) before considering any production mechanism or
broader capacity sweep.

## 2026-09-03 — Authorize one 4,096-token post-interaction discriminator

Phase 12b's 2,048-position context moves semantic coherence toward the full-H
oracle but does not suppress all competing bridge/train/support structures.
Do not advance 2,048 tokens to production or cheap-context construction yet.
Authorize exactly one 4,096-token post-interaction capacity test using one fixed
spatial aggregation rule. If that also remains below the oracle, stop simple
spatial K/V-density scaling rather than extending a capacity sweep.

## 2026-09-03 — Stop density sweeps; pursue cheap ~4K global-information construction

Phase 12c finds that one anisotropic 4,096-token post-interaction representation
is semantically oracle-like for the bridge/train discriminator, while 2,048 is
not. Record a useful scene-specific information threshold between those budgets.
Do not production-qualify the representation, infer orientation-independent
generalization, or test 5K/6K spatial densities. Any next research must ask how
to construct approximately equivalent ~4K globally informed context materially
more cheaply than the dense full-H source trajectory; the brute-force source is
only an oracle mechanism.

## 2026-09-03 — Advance direct globally interacting ~4K source architecture

Phase 13 shows that useful 4K context does not require 8K full-H interaction
before compression. Retain an ephemeral, same-sigma, whole-canvas, directly
interacting ~4K source as the next normalized-working-canvas architecture.
Do not make it persistent, productionize it, or claim efficiency: the tested
`32×128` geometry is tied to this destination/aspect and runs slower than the
8K source on the current native backend. Next isolate destination-independent
source geometry and backend execution behavior without reopening density or
post-interaction compression sweeps.

## 2026-09-03 — Do not advance the natural 4×2 fixed-4K large-canvas source

Phase 14's `H=128×256 -> S=32×128` arithmetic area restriction does not
preserve the Phase-13 one-bridge semantic result, despite executing a complete
globally interacting 4K source. Do not productionize it or respond by growing
the source budget. Its mapped variance ratio (`0.124651`) is materially outside
the successful Phase-13 regime, so the only authorized next discriminator is
one principled source-state construction comparison at the same 4,096 spatial
positions. Also do not treat accepted-G context as a positive normalized-W
control until that combined consumer geometry is separately qualified.

## 2026-09-03 — Reject scalar correction of the fixed-4K source

Phase 15 shows that neither matching Phase-13's mapped variance with gain `2`
nor preserving native variance with gain `sqrt(8)` changes the Phase-14
fragmentation class. Do not add source-gain controls, runtime-statistics
normalization, or a gain sweep. If fixed-4K research continues, test an
information-richer source representation at the same 4,096 spatial positions;
do not increase token density yet.

## 2026-09-04 — Reject simple orthogonal local-mode packing at fixed 4K

Phase 16's four-mode DCT/Hadamard source remains visibly fragmented despite
retaining directional information absent from the area mean. Do not add another
hand-designed mode allocation, channel rotation, or packing sweep. A trained
128-channel source token cannot losslessly carry four modes for all 128 latent
channels, and the tested fixed projection does not cross the semantic gate.
The next architectural discriminator must separate insufficient 4K source
interaction capacity from failure of the normalized-W consumer interface.

## 2026-09-04 — Treat fixed-source capacity and consumer interface as joint bottlenecks

Phase 17's bidirectional native source/W interaction materially improves scene
organization using the unchanged 4K source, so the frozen external-K/V
interface is not sufficient and cannot be exonerated. It still fails to produce
one dominant bridge system, so stronger consumption alone is also insufficient.
Do not productionize the joint oracle, repeat source representation searches,
or resume token-density sweeps. If research continues, localize semantic
agreement through joint transformer depth before selecting a new architecture.

## 2026-09-04 — Close joint-interface depth scheduling after Phase 18b

The bounded prefix/tail discriminator finds no S2/S3 state at any tested
transition. Prefix S0 and S9 retain only the full-joint endpoint's weak S1
organization; external-prefix D4/S0 can recover that same weak class, and later
joint starts cannot repair fragmentation. Do not run another joint
block-count/schedule sweep or optimize around D4/S0. The missing mechanism is
not a transient coherent solution being overwritten by later depth. Reassess
the information represented by the fixed global source/state before selecting
another architecture.

## 2026-09-04 — Reject the tested bounded 4K whole-canvas states

Phase 19's fixed 64x64 single-scale source and explicit 512-token coarse plus
3,584-token medium hierarchy both remain in the fragmented S0 class at the
2048x4096 terminal state. Do not advance either to a trajectory or production,
and do not promote the hierarchy based on its slightly lower overlap RMS. Also
do not respond with another fixed-grid allocation, deterministic packing,
density sweep, or joint-depth schedule. The next architecture decision must
reconsider how a bounded state represents scene-level relational identity.

## 2026-09-04 — Preserve native-coordinate Blueprint planning; reject the tested state transfer

Phase 20 shows that a bounded 2,048-token native-coordinate Blueprint model
call can form an S3 whole-scene bridge plan, so fixed-budget global planning is
not rejected. Do not return to external-K/V density, hierarchy, packing, or
depth sweeps. Reject the tested same-sigma local-state initialization as the
transfer contract: all 55 local refinements remain S0 despite exact algebra and
complete coverage. Before any trajectory, test exactly one prediction-space
coarse anchor using the coherent Blueprint x0; do not add a strength sweep or
production path.

## 2026-09-04 — Advance exact Blueprint prediction authority to one trajectory

Phase 20b reaches S3 by imposing the coherent bounded Blueprint prediction on
the coarse component of each ordinary local x0. This establishes viability of
bounded planning plus prediction-space transfer, but not acceptable detail:
the correction exceeds local-x0 RMS on average and visibly blurs structure.
Authorize one minimal trajectory experiment with the identical exact coarse
contract. Do not tune anchor strength or filters, and do not productionize
before trajectory stability and local-detail retention are established.

## 2026-09-04 — Accept the persistent Blueprint/anchor semantic contract

Phase 20c confirms that the exact Phase-20b anchor preserves an S3 scene across
the complete four-interval persistent Blueprint/H Euler trajectory. Treat
trajectory-level semantic authority as established for this case. Do not
introduce strength, filter, frequency, or context sweeps and do not productionize
the current blurred result. The next discriminator must address how regenerated
local detail or null-space information can reach accepted H without weakening
the demonstrated Blueprint composition.

## 2026-09-04 — Reject fixed spatial coarse/null release as sufficient

Phase 21 proves that a destination `2x2` mean/nearest decomposition can preserve
its null component numerically while retaining exact Blueprint coarse authority,
but the preserved component includes competing scene geometry. Do not tune the
coarse scale, filter, or anchor weight: the requested single-pair discriminator
failed its perceptual gate. Preserve the bounded Blueprint planning result and
move next to model-mediated refinement/resampling rather than another external
frequency decomposition.

## 2026-09-04 — Advance Blueprint-initialized model-mediated refinement

Phase 22 shows that one ordinary native FLUX denoise at fixed late sigma 0.25,
initialized from the coherent mapped Blueprint prediction, preserves an S3
whole scene and regenerates modest detail; the matched independent-local anchor
returns S0. Retain this as the supported transfer mechanism for the next narrow
trajectory discriminator. Do not tune sigma, add output-space residuals, or
productionize from this single-state result.

## 2026-09-05 — Accept temporal viability of Blueprint-initialized resampling

Phase 23 keeps all four Blueprint and refined estimates in the S3 scene basin
while producing a stable measurable detail increase. Treat Blueprint
initialization as established for these bounded evaluations; do not revisit it
with sigma or guidance sweeps. The next architectural discriminator should
compare local-refinement cadence—terminal-only, periodic, or fully interleaved—
without production changes until one lifecycle is qualified.

## 2026-09-05 — Select terminal-only Blueprint-local resampling

Phase 24 finds no clear quality benefit from persisting the ordinal-1 local
residual into a second terminal refinement. Both arms are S3, but periodic
persistence costs an additional 55 local forwards and increases semantic/coarse
distance diagnostics. Select the simpler staged lifecycle: finish the bounded
Blueprint trajectory, then run one terminal fixed-`0.25` native-local resampling
pass. Do not authorize fully interleaved sampling from this evidence.

## 2026-09-05 — Advance terminal-only resampling to production architecture design

Phase 25 preserves S3 semantics for the bridge, ordered car/tree/house, and
single-astronaut cases under the exact frozen Phase-24 mechanism, while both
new ordinary-local controls fragment to S0. Treat terminal-only Blueprint
resampling as semantically generalized for these tested scene classes and
authorize a separate Phase-26 production-architecture design task. Do not
implement production automatically or infer untested geometry generalization.

## 2026-09-05 — Select a dedicated staged node for terminal resampling

Do not force the selected mechanism into the Candidate-3 public `SAMPLER` or
its persistent `(G,H)` coordinator. The production design uses a dedicated
node that truthfully owns the bounded Blueprint trajectory and separate fixed-
sigma terminal refinement, while internally entering through `guider.sample()`
to preserve native ComfyUI conditioning/model lifecycle. Use a Blueprint-only
accepted state and stream one batch-one W prediction at a time into the
existing normalized overlap arithmetic. Keep the legacy Candidate-3 node
unchanged during first-slice implementation; the new path must not invoke DCT
coupling, persistent H/W, global K/V, or `Flux2BlockExecutor`.

## 2026-09-05 — Qualify the exact terminal-resampling first slice

Accept the dedicated `Blueprint Terminal Resampling` node for the exact
Phase-25 contract only. Keep its geometry, four-value interval schedule,
BasicGuider CFG-1 profile, native Klein 4B validation, and terminal sigma fixed
and fail closed. Preserve the legacy Candidate-3 node independently. Do not
generalize geometry, cadence, model family, conditioning, sigma, batching, or
sampler behavior until separately qualified.

## 2026-09-05 — Qualify a finite bounded geometry profile set

Extend terminal resampling only to the explicit aspect-preserving profiles
`B/H = 32x64/128x256`, `36x54/128x192`, `45x45/128x128`, and
`64x32/256x128`. This keeps Blueprint work approximately fixed at 1,944–2,048
tokens instead of scaling with destination area. Preserve Phase-27 integer-area
initialization where divisible; use deterministic adaptive area restriction and
analytic per-cell variance preservation for the two noninteger mappings.
Reject destination-proportional, fixed-long-axis expansion, or aspect-distorting
policies. All other destination geometries continue to fail closed.

## 2026-09-05 — Stop scalar terminal-sigma tuning

Keep the production terminal sigma at its qualified `0.25` value. Phase 29 is
a research result only and does not authorize a production parameter or default
change. Although every tested `0.10`–`0.50` branch remains S3, no value
materially improves fidelity across all three geometry/scene controls. Treat
the remaining softness as a mechanism-level limitation of mapped-Blueprint
initialization plus one local denoising call. The next discriminator may test
one fixed Blueprint-guided local-refinement mechanism at `0.25`; do not run
more sigma points or a strength/filter sweep.

## 2026-09-05 — Do not invent a parameter-free Blueprint attractor

Classify Phase 30 as `E — ARCHITECTURAL BOUNDARY` and stop before model
inference. A soft denoised-prediction correction requires a structural metric
and scalar guidance/likelihood scale that the qualified CONST-flow contract
does not supply. Do not disguise `lambda=1` as parameter-free, reuse hard
projection, or introduce transformer hooks. Native FLUX reference latents are
a distinct conditioning/image-context architecture and require separate
authorization. Return next to one fixed Blueprint-to-working-canvas
lift/initialization representation discriminator.

## 2026-09-05 — Retain nearest lift; stop interpolation sweeps

Classify Phase 31 as `B — LIFT CHANGES OUTPUT BUT DOES NOT SOLVE SOFTNESS`.
Do not productionize bilinear lifting: it preserves S3 but is slightly smoother
in all three controls and naturally breaks exact `avgpool2(lift(b))=b`
consistency. Keep the qualified nearest lift in production. If local detail is
pursued further, test one richer Blueprint-derived working-state construction
at fixed sigma/geometry with explicit provenance; do not try bicubic, area, or
an interpolation-strength sweep.

## 2026-09-05 — Reject nullspace-only noise as sigma-equivalent

Classify Phase 32 as `D — NO PARAMETER-FREE SIGMA-CONSISTENT CONSTRUCTION
EXISTS` and stop before inference. Although `(I-UD)epsilon` has exact coarse
restriction zero, it is rank-deficient correlated Gaussian noise, not the
qualified i.i.d. noise consumed by CONST `noise_scaling`. Do not introduce the
variance-only `2/sqrt(3)` scale, double-count the existing null component, or
tune a detail coefficient. Any continuation must explicitly introduce and
qualify a model-mediated or multi-state coarse/detail contract.

## 2026-09-05 — Stop repeated ordinary terminal resampling

Classify Phase 33 as `B — SECOND PASS RETAINS S3 BUT DOES NOT IMPROVE DETAIL`.
The second pass is semantically stable but adds mostly texture/alias energy,
not meaningful object/anatomy/structure detail, while increasing Blueprint,
low-frequency, and overlap error. Keep the single-pass production baseline.
Do not test a third pass, another sigma, or fresh second-pass noise. Any further
detail work must define a principled model-mediated or multi-state
coarse/detail construction.

## 2026-09-05 — Qualify the Haar coarse/detail contract for one terminal discriminator

Classify Phase 34 as `A — COARSE/DETAIL CONTRACT IS MATHEMATICALLY COHERENT;
PROCEED TO ONE FIXED MODEL-MEDIATED DISCRIMINATOR`. Use normalized `2x2` Haar
coordinates with Blueprint LL amplitude `C=2b`. Retain independent noisy LL
and three independent Gaussian detail coordinates; never describe a
fixed-coarse/detail-only reconstruction as ordinary i.i.d. noise.

Any continuation must assemble and retain four global band fields and
reconstruct a `2x` spatial latent. The current single coarse contribution
cannot preserve the additional degrees of freedom. Phase 35 is limited to one
terminal square-scene test with exact coarse prediction replacement and
unchanged model-predicted Haar detail; no basis, sigma, strength, schedule, or
coupling sweep is authorized.

## 2026-09-05 — Do not promote untouched local Haar detail

Classify Phase 35 as `B — partial/ambiguous`: exact Blueprint ownership of the
Haar coarse coordinate preserves the intended single scene, but untouched
FLUX-predicted local detail bands disagree across overlaps and decode as
crosshatch/ghost energy rather than credible higher-resolution structure. Do
not tune detail scaling, coarse blending, wavelets, sigma, or overlap to force
a pass. Retain Phase 34's mathematical result while rejecting this fixed
independent-local-detail assembly as an image mechanism. Any continuation must
address shared/persistent detail compatibility explicitly.

## 2026-09-06 — Stop Phase 36 before inference: no unique local schedule

Classify Phase 36 as `D — NO UNIQUE SHORT LOCAL SCHEDULE`. Do not infer a
multi-step local path from the qualified `[0.25,0]` terminal evaluation or
splice `0.25` into the near-one Blueprint schedule. ComfyUI exposes multiple
materially different scheduler families, and neither family nor interval count
is selected by existing evidence. Any future short-trajectory discriminator
must first establish schedule provenance or openly authorize one predeclared
empirical schedule. Keep this decision separate from the untested interleaved
global/local architecture.

## 2026-09-06 — Require an explicit empirical policy for any Phase-38 short trajectory

Classify Phase 37 as `B — CANONICAL FULL/PARTIAL SCHEDULE EXISTS, BUT LOCAL
STEP COUNT REMAINS FREE`. Adopt the official shifted flow/Euler family as
relevant provenance, while rejecting the claim that it uniquely extends the
qualified `[0.25,0]` Stage-2 path. Diffusers' Klein partial-denoise rule only
truncates a full schedule after the caller chooses `num_inference_steps`, and
numeric strength is not numeric sigma. Do not publish a Phase-38 schedule from
this audit. A later experiment must explicitly authorize one fixed empirical
interval count and exact discretization, and must label that choice as a new
research contract rather than model-native Klein policy.

## 2026-09-06 — Stop terminal-local depth experiments after Phase 38

Accept Phase 38's single schedule as an explicitly empirical discriminator,
not official Klein partial denoising. Classify it as `B — CHANGES OUTPUT BUT
DOES NOT SOLVE SOFTNESS`: four persistent-W Euler intervals retain S3 and exact
determinism but add mainly line/texture energy, not credible car/tree/house
structure, at fourfold local model work. Do not sweep interval count, start
sigma, or schedule spacing, and do not generalize this arm to portrait or
landscape. Any continuation must change the Blueprint-to-working authority or
representation contract rather than adding more terminal-local depth.

## 2026-09-06 — Do not treat generic Klein hooks or references as same-canvas Blueprint guidance

Classify Phase 39 as `E — ARCHITECTURAL BOUNDARY`. Do not use
`reference_latents` as the candidate: its trained semantics identify a separate
reference canvas. Do not infer a same-canvas guide from `post_input`, attention,
block, control, or sampler hooks; these expose tensor intervention points but
do not define a trained Blueprint role, structural metric, or fixed coupling.
Do not repeat prediction-space guidance or state projection. The only next
discriminator authorized by this result is a source/checkpoint audit for a
trained architecture-matched Klein 4B same-canvas control adapter; run no model
unless its input normalization and canonical coupling are established.

## 2026-09-06 — Close the released-adapter Blueprint-as-guide branch

Accept Phase 40's verdict `C — ONLY INCOMPATIBLE / SEPARATE-CANVAS ADAPTERS
FOUND`. Do not reinterpret Klein RefControl/reference inputs as an aligned
Blueprint plane, transplant FLUX.2-dev/FLUX.1 ControlNets into Klein 4B, or
treat untrained Klein ControlNet code as checkpoint evidence. No Phase-41
inference is authorized. The next major phase is a bounded architecture design
discriminator for separate interleaved global↔local resampling, beginning with
explicit state ownership and feedback equations rather than model execution.

## 2026-09-06 — Begin staged refinement as a separate research branch

Keep Candidate-3 and `Blueprint Terminal Resampling` frozen and coexisting.
Treat the new MrFlow-like work as a separate experimental lineage, but do not
create a sibling production node merely by renaming Terminal Resampling: the
first denoised-handoff algorithm overlaps its established contract.  Select
halo-aware crop-before-upscale as the spatial-transfer policy because it is
numerically equivalent to complete-map-then-crop under the declared bilinear
contract and avoids a second destination-sized mapped anchor.  Do not expose
noisy handoff until its sigma, covariance, and trajectory meaning are proven.
Do not implement interleaving, learned upscaling, guide, ControlNet, or backend
optimization in this branch.  Require credible semantic local-detail gain, not
texture energy alone, before proposing the smallest sibling-node architecture.

## 2026-09-06 — Stop fixed persistent coarse-guide tuning

Reject the tested fixed-guide branch for production. Both declared policies
retain S3 and measurably reduce cross-region disagreement, but neither improves
credible detail; the fixed terminal Blueprint acts as a soft locking target.
Do not sweep strength, release shape, or sigma, and do not modify Terminal
Resampling. If continued, the next genuinely different discriminator is
recurrent/interleaved G/W latent refresh with explicit accepted-state ownership,
not another post-prediction guide variant.
## 2026-09-06 — Reject fresh-W recurrent reconstruction, not recurrence generally

Do not advance either fresh-W policy or sweep its cadence, interleaving stop, or
coupling strength. G-authoritative and bidirectional exchange both lose S3 by
freshly constructing near-one-sigma W regions from clean G predictions and
regional noise. This decision does not reject persistent-H or other recurrent
multiresolution sampling.
Keep Candidate-3, Terminal Resampling, and persistent-guidance artifacts frozen,
and make no production change from this experiment.

## 2026-09-06 — Stop persistent-H shared-provenance refinement without G-to-H authority

Reject both tested persistent-H arms. Exact shared stochastic provenance and one
retained normalized H trajectory do not prevent bounded local calls from
creating repeated prompt-complete scenes. H-to-G feedback changes G but cannot
change H when no G-to-H/W path is present, so its identical H result is a causal
contract consequence rather than evidence that feedback was skipped. Do not
sweep cadence or policy for this discriminator. This does not reject recurrent
architectures with a separately justified evolving-G influence on W/H.

## 2026-09-06 — Accept a narrow configurable Terminal-Resampling sibling

Add `Blueprint Configurable Prototype` as a sibling and retain
`BlueprintTerminalResampling` unchanged as the frozen oracle. Generalize G/H/F/W
geometry, destination stride, bounded working geometry, deterministic seed, and
late refinement sigma only inside the empirically validated envelope. Reuse the
halo-aware bounded transfer and streaming normalized assembly; never issue an
H-sized model forward. Preserve the exact frozen path by direct oracle delegation
when all Phase-25 values match.

Do not yet expose alternate interpolation/restriction, multistep local schedules,
noisy handoff, non-integer footprint/working scaling, masks, other sampling
contracts, or other model families. The next sibling feature, in a separate task,
should be the smallest independently qualified local-schedule extension; it must
not reopen stopped recurrent, persistent-H, or guidance sweeps.

## 2026-09-07 — Do not promote configurable local schedule depth

Reject the tested two- and three-interval local W schedules. They preserve S3
and bounded residency but do not improve credible local structure over the
bit-exact one-step control; their additional gradient energy accompanies worse
overlap disagreement and approximately proportional model work. Keep
`Blueprint Configurable Prototype` at one local interval. Apply the requested
stop rule: do not follow this result with further local sigma/depth sweeps or
resume high-sigma, interleaved, persistent-H, guide, ControlNet, or noisy-handoff
work in this branch.

## 2026-09-07 — Publish a compatible pixel-based terminal-refinement wrapper

Use `Blueprint Diffusion (Terminal Refine)` as the recommended public name and
`BlueprintDiffusion` as its registration key. Keep
`BlueprintConfigurablePrototype` unchanged because its class key and inputs may
already be serialized in workflows. The public wrapper converts strict
16-divisible pixel inputs into the same qualified latent-grid procedure; it does
not fork sampling behavior.

Derive H from the destination latent and derive stride as `footprint-overlap`.
Expose auto/manual mode, G pixels, footprint pixels, overlap pixels, W pixels,
refinement sigma, and seed. Auto must remain an explicit live-qualified profile
table; do not infer arbitrary aspect-ratio geometry. Keep interpolation,
restriction, handoff, G schedule, local step count, region order, and assembly
fixed. The node is ready for ordinary user testing only inside this documented
Klein 4B contract.

## 2026-09-07 — Phase 41 selects sequential accepted-delta exchange for design only

Keep `Blueprint Diffusion (Terminal Refine)`, Terminal Resampling, Candidate-3,
and specialized execution frozen. Make no production change. Do not resume
fresh-W reconstruction, fixed prediction guidance, or Candidate-3 hard D/U
projection.

Select exactly one possible future discriminator: scheduler-midpoint operator
splitting with persistent H. The bounded G model owns `sigma_i -> sigma_mid`;
its accepted delta is mapped additively into H before any local call. Bounded W
models then own `sigma_mid -> sigma_next`; their normalized assembled accepted H
delta is restricted additively into the next G. Accept G/H/sigma tuples only at
the two explicit atomic barriers. This is the minimum design in which current
global model information changes the local model input and accepted local change
alters the next global model input without re-noising or a scalar coupling
strength.

Phase 41 authorizes documentation only. Do not run diffusion inference until a
later task explicitly authorizes the one fixed square discriminator and its
predeclared gates. If that run fails S3 or credible-detail improvement, stop the
branch rather than tuning midpoints, cadence, transfer strength, or schedules.

## 2026-09-07 — Reject and stop Phase-41 accepted-delta exchange

Reject the fixed scheduler-midpoint sequential accepted-delta mechanism. Its
implementation satisfies shared provenance, deterministic retry, transfer
identity, bounded model geometry, atomic pair acceptance, and non-dominant
increment requirements, but its `0.522903` terminal overlap RMS exceeds the
`0.190627` gate and the decoded result contains pervasive patch/lattice-like
cross-region structure. It does not improve the credible composition/detail
frontier over Terminal Refine.

Apply the requested stop rule. Do not test another midpoint, cadence,
interleaving start/stop, transfer scaling, A/R operator, sigma, or local-step
schedule as a continuation of this architecture. Keep `Blueprint Diffusion
(Terminal Refine)`, frozen Terminal Resampling, Candidate-3, and specialized
execution unchanged. Any further architecture requires a separate design task.

## 2026-09-07 — Permit only an experiment-first native Z-Image-Turbo Terminal Refine port

Do not generalize or modify the frozen Klein `Blueprint Diffusion (Terminal
Refine)` node. A Z-Image port must be a later sibling/model adapter and must
preserve the same terminal-denoised, one-late-refinement algorithm.

Narrow the first target to native latent-space Z-Image-Turbo with shift 3,
8-step simple schedule, CFG 1, no reference/control/edit inputs, and ordinary
native-origin RoPE for every bounded call. Do not reuse Klein sigmas or silently
add destination-registered RoPE. Reuse geometry/transfer/assembly semantics but
do not prematurely refactor production's hard-coded 128-channel utilities.

Authorize no inference in this audit. A later task may implement exactly the
single fixed experiment described in
`experiments/ZIMAGE_TERMINAL_REFINE_PORT_AUDIT.md`. Failure of that discriminator
closes the direct port; it does not authorize a RoPE, sigma, CFG, geometry, or
sampler sweep.

## 2026-09-07 — Reject and stop the direct native-coordinate Z-Image Terminal Refine port

Reject the exact Phase-44 port. It satisfies fixed sampling, halo transfer,
coverage, determinism, normalized assembly, zero-origin positional, bounded
model-call, flat region-residency, and S3 composition requirements, but fails
the mandatory credible-detail improvement over plain Blueprint mapping. Keep
the Klein `Blueprint Diffusion (Terminal Refine)` frozen and make no production
or registration change.

Apply the declared stop rule: do not tune sigma, overlap, footprint, W size,
CFG, sampler, positional coordinates, guide strength, or local-step count as a
continuation. Those interventions require a separate architecture design task.

## 2026-09-08 — Permit one projection/channelwise Clown Guide discriminator, not plain guidance

Do not reopen ordinary Clown `epsilon` guidance: under Klein/Z-Image CONST it
is mechanically equivalent to the already-rejected persistent coarse x0 mix.
Treat `epsilon_projection_cw` as a narrowly distinct sampler-level hypothesis
because its channel-adaptive derivative projection is not a scalar x0 pull.

Authorize design only, not implementation or inference. A later task may run
exactly one Klein Phase-29/38 arm using the existing initial W, four-interval
schedule, mapped guide, full mask, and constant base weight 0.25, changing only
the post-model operator. Do not import RES4LYF, add model conditioning, or sweep
weights/schedules. Failure closes this Clown Guide branch.

## 2026-09-08 — Reject `epsilon_projection_cw` and close Clown Guide branch

Reject the only authorized projection/channelwise discriminator. It preserves
coarse S3 composition and satisfies causal isolation, determinism, bounded
execution, coverage, and residency checks, but fails the mandatory credible-
detail gate and exceeds the non-dominant increment criterion. The nonlinear
projection does not advance the qualified composition/detail frontier.

Apply the Phase-46 stop rule. Do not test another weight, projection mode,
channelwise toggle, cutoff, sigma schedule, sampler, or geometry as a
continuation. Keep all production nodes and registrations unchanged.

## 2026-09-08 — Reject only the literal interpolation-only alternating prototype

Reject the exact direct-state-resize `G,W,G,W` experiment because it loses S3,
duplicates complete prompt interpretations, destroys horizon continuity, and
raises terminal overlap disagreement to `0.6462023`. No stabilization or
corrective state construction is authorized as part of this result.

Scope the rejection narrowly. It applies to the tested literal interpolation-
only trajectory, not to all recurrent/interleaved architectures. Production
nodes and registrations remain unchanged.

## 2026-09-08 — Start Local Edit with a hard CONST source trajectory

For the first independent Local Edit discriminator, select a source-authoritative
hard spatial derivative/state constraint under native Klein CONST sampling.
Use one explicit user convention (`M=1` editable), derive a measurement-only
transition band on the editable side, and keep the locked complement immutable.
Do not feather the lock.

The future prototype must be an optional Local Edit policy hosted by a minimal
custom `SAMPLER`, because accepted state—not only denoised prediction—must be
authoritative. Reject an incoming latent `noise_mask` to prevent ComfyUI's
inpaint wrapper and Local Edit from owning the same state twice. Do not add
pseudoimplicit, Clown channelwise projection, a transition correction,
Blueprint state, sparse execution, K/V reuse, or compute optimization to the
first discriminator. This decision authorizes design only; implementation and
diffusion inference require an explicit continuation.

## 2026-09-09 — Reject hard-source-only; retain the lock for boundary research

Reject the exact hard-source sampler as a complete Klein Local Edit policy. It
passes source-trajectory exactness, edit freedom, and determinism, but fails the
strict decoded-source and geometric boundary-continuity gates. This rejection
does not weaken or replace the hard source invariant.

The next discriminator may act only on a transition band derived wholly inside
the editable mask, or test another explicitly editable-side boundary correction.
Do not feather or move the locked source, and do not reopen pseudoimplicit,
projection sweeps, Blueprint coupling, sparse execution, or production sampler
work as part of this result. This decision does not reject a separately
reproduced real-user Clown epsilon/projection workflow.

## 2026-09-09 — Stop scalar editable-side transition guidance

Reject the fixed two-column linear `d_model` to `d_source` interpolation. It
preserves the immutable lock and edit freedom but materially worsens the
boundary, producing a wide dark barrier without resolving the restarted bridge.
Do not run the optional falloff arm or tune scalar strength, width, sigma window,
schedule, seed, or projection mode.

Preserve the exact hard-source contract for experiments that claim immutable
source authority. At this point the next Local Edit research branch was to
audit a model-visible or prediction-refresh boundary mechanism; the later
real-user Clown reproduction below supersedes that sequencing decision.
The VAE locality finding is diagnostic only and does not authorize changing the
Local Edit algorithm or source lock.

## 2026-09-09 — Adopt the reproduced Clown epsilon projection as a behavioral reference

The fixed two-column scalar editable-side transition remains falsified. Do not
generalize that result to epsilon guidance as a family.

The uploaded workflow's serialized `SetLatentNoiseMask` node is bypassed. Its
actual behavior is full-canvas Klein Euler plus RES4LYF `epsilon_projection`,
with no latent noise mask. An experiment-only direct-RES4LYF harness reproduces
the uploaded output within one 8-bit code value per channel. Adopt that arm as
the current behavioral reference for geometric continuity.

This is not a production algorithm decision and does not relax the exact-source
contract. The reproduced behavior allows the entire accepted canvas to evolve
and has measurable source drift. Do not proceed to model-visible injection,
K/V injection, sparse execution, or productionization until a later discriminator
addresses how to combine the observed global continuity with authoritative
source preservation.

## 2026-09-09 — Do not continue the late sampler-state restoration family

Final-only and final-two accepted-state restoration preserve the reproduced
Clown D bridge geometry and make the source latent exact, but both worsen
decoded source-region error. Therefore do not run the conditional source-mask
inset/overlap follow-up or broaden this into a restoration schedule sweep.

This does not reinstate the early hard-lock geometry rejection: late restoration
does not create an independent bridge. Post-decode compositing remains a
separate possible contract, but was not activated here because terminal-only
restoration did not visibly break the bridge join.

## 2026-09-09 — Retain inward post-decode compositing as the source-preservation candidate

Stop sampler-state restoration for source preservation in this sequence. The
zero-diffusion bridge discriminator shows that a post-decode composite can keep
an exact original-source interior and frozen D's generated exterior geometry.

Prefer an inward transition derived by shrinking the source footprint, never by
expanding source outward. The tested 24-pixel raised-cosine strip reduces the
hard composite's sharp boundary jump without ghosting the bridge, although a
subtle tonal band remains. Do not productionize or sweep widths on this single
case; retain it only as the next candidate requiring broader validation.

## 2026-09-09 — Do not qualify fixed 24-pixel compositing as the default

The fixed policy preserves exact interior/exterior ownership and is
structurally safe on the eligible bridge, tree, and desert cases, but it fails
photometric robustness: smooth content exposes a full-height tonal band even
when nominal-boundary gradient metrics improve.

Keep the compositor as an exactness/reference contract, not the default Local
Edit mechanism. Do not widen or tune it per image. Continue to reject any case
whose epsilon A is already geometrically inconsistent before evaluating
compositing. Any next branch must address photometric reconciliation explicitly
without weakening ownership or concealing structural failure.

## 2026-09-09 — Stop scalar and affine photometric reconciliation

Reject the fixed automatic luminance-affine transition correction. It leaves
the desert band visible and materially degrades the previously safe organic
case with a brightness halo, despite preserving exact ownership and geometry.

Do not tune affine coefficients, clipping, neighborhoods, per-channel variants,
or strength. The next eligible compositing mechanism is a fixed gradient-domain
or multiband discriminator under the same exact source-interior and generated-
exterior contract. This is an experiment direction, not authorization to
productionize.

## 2026-09-09 — Stop post-decode frequency-domain blending

Reject the fixed multiband policy. Broader 64/128-pixel low-frequency supports
redistribute the desert mismatch but leave its visible tonal band and worsen
several strip residuals. Do not tune pyramid depth, Gaussian radii, supports,
or weights. The next eligible compositor experiment is one fixed gradient-
domain policy under the same exact-interior and byte-exact-exterior contract.

## 2026-09-09 — Close post-decode compositor research

Reject the final fixed bounded gradient-domain policy. Although exact source
interior and generated exterior ownership remain correct, the Poisson solve
creates severe tonal ramps, a colored tree halo, and bridge-boundary damage.
Do not tune its width, gradient rule, or boundary handling.

The tested direct RGB, affine, multiband, and gradient-domain post-decode
families have not met the cross-case perceptual gate. Close this compositor
sequence and return to the diffusion-stage epsilon/projection algorithm. Do
not productionize any experimental compositor.

## 2026-09-09 — Stop locked-only prediction/state projection

Do not continue tuning hard source derivative or accepted-state projection.
The exact combined contract makes locked prediction and state authoritative,
but is identical to state-only restoration in the editable trajectory and
does not prevent independent bridge, tree/lighting, or desert photometric
solutions.

Any next Local Edit discriminator must act where source information can alter
the editable model prediction itself. A narrowly scoped model-level source-
context/KV mechanism is eligible; pixel compositing, feathering, and new
sampler-family exploration remain closed. This is not authorization to
productionize.

## 2026-09-09 — Do not extend step-0 same-sigma source K/V into a trajectory

The all-block dense K/V intervention is active, but fails the rigid-bridge
expected-direction gate. It changes editable raw x0 strongly while worsening
both bridge seam gradients and reorganizing independent tower/cable structure.
Do not run a trajectory or sweep strengths, blocks, or context ranges.

At sigma 1, native CONST source noising exactly removes the clean source latent;
therefore this test is coordinate-preserving source-region token reweighting,
not clean-source conditioning. A future source-appearance mechanism would need
a separately justified contract. This result does not authorize production.

## 2026-09-09 — Stop untrained clean-source K/V injection

Do not extend the fixed sigma-one clean-source feature extractor into a
trajectory or tune source timestep, strength, block subsets, or context range.
It is strongly active and improves aggregate low-frequency boundary metrics,
but fails the primary rigid-bridge direction gate: editable towers and cables
remain an independently composed bridge system.

This closes the tested untrained clean-source K/V mechanism. It does not reject
a genuinely trained source-conditioning interface, and it does not authorize
production changes.

## 2026-09-09 — Preserve source co-evolution as the defining epsilon mechanism

Treat full-canvas source-region co-evolution followed by later dense model
coupling as the demonstrated causal mechanism behind successful Clown epsilon
outpainting. Do not describe it as direct editable-region epsilon guidance.

The editable-only exact-correction ablation is ordinary T2I because the Clown
guide-weight mask gives the correction no editable support. Any future epsilon
variant intended to retain its geometric benefit must preserve a way for the
source-attracted trajectory to influence subsequent editable predictions. This
finding does not authorize tuning or production implementation.

## 2026-09-09 — Classify source co-evolution as localized causal communication

Classify the measured mechanism as SOURCE CO-EVOLUTION LOCALIZED: the first
accepted source displacement exactly reproduces the next editable epsilon
prediction, and fixed distant-source/low-frequency components reproduce
substantial directionally aligned portions. Do not reduce this to boundary-only
negotiation.

A single pulse is not selected as an algorithm. Its influence persists but
redirects relative to full epsilon when later corrections are omitted. The
result authorizes no production implementation, strength/support/frequency
tuning, or final-quality claim.

## 2026-09-09 — Classify the low-frequency carrier as scene-spatial

Classify the fixed interval-0 discriminator as SCENE-SPATIAL SOURCE SIGNAL.
Correct source-crop arrangement is required for the strong epsilon-like
editable response; equal-energy and equal-channel-statistic rearrangements
substantially weaken or redirect it.

Do not reduce the carrier to latent energy or channel statistics alone. The
observed response is not a clean transformed relocation,
so do not claim a coordinate-transfer mechanism. This result authorizes no
block tracing, algorithm optimization, or production implementation.
