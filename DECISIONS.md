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
