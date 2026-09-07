# Phase 42 — scheduler-midpoint accepted-delta exchange

## Verdict

`REJECT_AND_STOP`. The fixed Phase-41 mechanism retains the coarse S3 object
layout but fails the declared overlap and credible-detail gates. No alternate
midpoint, cadence, transfer, strength, sigma, or local-step experiment is
authorized by this result.

## Fixed run

The live run used seed `20260921`, the existing S3 prompt, CONST sigmas
`[1.0, 0.9991771578788757, 0.9975355267524719, 0.9926428198814392, 0.0]`,
and `G=45x45`, `H=128x128`, `F=32x32`, stride `16x16`, `W=64x64` with 49
row-major regions. Scheduler midpoints were computed by bisecting in the
inverse-shift coordinate with `mu=2.291179894115571`.

H noise was generated once. G noise was its variance-normalized area
restriction. The candidate and uncoupled persistent-H control had identical
initial H and G hashes. There were no regional noise streams or subsequent
noise-scaling calls. At every interval, all W inputs were lifted crops of one
immutable H midpoint, normalized assembly produced one H proposal, and exactly
one proposed `(G,H)` pair was committed.

## Causal and numerical evidence

- The primary and independent repeat are bit-exact, final SHA-256
  `fb95dc8b3564936c8aa1b90dfc61fc4d014408e17ead022782833afd25889eff`.
- Initial H and G match the uncoupled persistent-H arm exactly. At interval 0,
  divergence occurs only after `A(delta_G)`.
- All `H_mid-H_i = A(delta_G)` and `G_next-G_mid = R(delta_H)` errors remain
  below float32 RMS `1e-7` and maximum `1e-6` tolerances.
- Per-interval `RMS(A(delta_G))/RMS(H_i)` is `0.000447`, `0.000873`,
  `0.002446`, `0.089227`. The corresponding H-to-G ratios are `0.000252`,
  `0.000481`, `0.001300`, `0.413649`. All increments are finite and below 1.
- Candidate H RMS relative to the uncoupled H arm is `1.000204`, `1.000613`,
  `1.001807`, `1.355382`; all remain below the declared 2x bound.
- Work is exactly four `45x45` G calls plus 196 `64x64` W calls and zero
  `128x128` H calls. Interval-barrier allocation is exactly flat at
  `2,508,399,104` bytes. Peak CUDA allocated/reserved is
  `3,028,564,992` / `3,514,826,752` bytes.
- Per-region allocated-memory range is `3,110,400` bytes rather than exact
  zero, so the literal flat-region-residency gate also fails. No completed W
  tensors are retained; interval barriers are flat, local live count is one,
  and the small range reflects allocator/model-call settling rather than growth
  with completed-region index.

## Semantic result

The candidate visibly retains one car, one tree, and one house in a single
coarse scene. It does not deliver a clean structural improvement over Terminal
Refine: the sky and ground contain pervasive patch/lattice-like local
interpretations. Final overlap RMS is `0.5229029794`, 2.743 times the fixed
`0.190627` ceiling (Terminal control `0.1732971654`). Therefore both the
cross-footprint and credible-detail gates fail even though the latent is
deterministic and the transfers are numerically well behaved.

Artifacts are in `flux2_phase42_accepted_delta_exchange_results/`, including
`report.json`, per-run telemetry, latents, decoded PNGs, and `COMPARISON.jpg`.

