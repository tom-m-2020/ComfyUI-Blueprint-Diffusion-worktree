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
