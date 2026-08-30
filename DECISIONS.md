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
