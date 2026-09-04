# Phase 20 — native-coordinate Blueprint-to-local-state discriminator

## Result

The bounded native-coordinate Blueprint model call produces a coherent whole-
scene plan, but the tested same-sigma Blueprint-to-local state construction does
not transmit that plan through 55 independent normalized local refinements.
Gate 2 applies.

**Verdict: `GLOBAL PLAN WORKS — LOCAL STATE TRANSFER FAILS`.**

This is a terminal, zero-update research result. Production and ComfyUI core
were not changed. The optional prediction-anchor arm was not run because the
state-initialization arm remained clearly S0 rather than reaching S2 with only
a detail deficit.

## Fixed experiment

- Accepted state: Phase-19 bridge/train state, `H=128x256`, `G=96x192`, seed
  `20260901`, terminal sigma `0.9926428198814392`.
- Destination execution: 55 ordered `32x32` regions at stride 24, each mapped
  to a reconstructed `64x64` native local working canvas.
- Blueprint: one fixed `32x64` latent canvas, 2,048 generated tokens, preserving
  the destination's 2:1 aspect ratio.
- Blueprint model coordinates: ordinary native `y=0..31`, `x=0..63`. Expanded
  destination-space RoPE was deliberately not used.
- Destination mapping: bilinear, `align_corners=False`, through normalized
  `[0,1]` canvas coordinates.
- Accepted-state updates: zero. Accepted H/G hashes and all control-W hashes
  remained unchanged.

The measured preflight estimate was eight minutes, below the 30-minute stop
boundary. Both arms persisted atomically and the run is resumable by exact
configuration hash.

## Algebra

The Blueprint state is built from the complete accepted H once:

```text
B_coarse = area_mean_4x4(H)                         # 128x256 -> 32x64
B_sigma  = B_coarse + sigma * sqrt(15/16) * N_B
```

The added independent component analytically restores the noise variance lost
by averaging 16 positions without scaling the coarse signal. Measured RMS was
`0.994493` for `B_sigma` (`0.248441` coarse plus `0.963084` added noise).

After the ordinary Blueprint prediction `x0_B`, it is mapped to destination
space. For region `r`, with mapped crop `b_r`:

```text
W_r = (1 - sigma) * nearest2(b_r) + sigma * N_r
D(W_r) = (1 - sigma) * b_r + sigma * D(N_r)
D = mean_pool_2x2
```

Equivalently, the local stochastic null component is
`sigma * (N_r - nearest2(D(N_r)))`, whose restriction is zero. The invariant
held to max absolute error `4.77e-7`; the null component error was `2.38e-7`.
No heuristic gain was used.

At this terminal sigma, however, `1-sigma = 0.00735718`. Thus a mathematically
same-sigma working state gives the coherent denoised Blueprint plan very little
direct input-state authority; the ordinary local model is still asked to infer
nearly everything from region-specific noise.

## Numerical and runtime results

| Arm | Blueprint forwards/tokens | W forwards | Overlap RMS | Assembled RMS | CUDA source | CUDA local | Wall (arm) | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A normalized-W control | 0 / 0 | 55 | 0.856211 | 0.744668 | 0 | 58.860 s | 59.334 s | 1.632 / 2.021 GiB |
| B Blueprint state initialization | 1 / 2,048 | 55 | 0.885702 | 0.668791 | 0.556 s | 59.119 s | 60.708 s | 1.799 / 2.096 GiB |

B differs materially from A (`RMS=0.855303`, max absolute `5.721803`), but this
numerical change does not cross the semantic gate. The average restricted local
prediction error versus the mapped Blueprint crop is `0.931751` RMS. Coverage
is complete and normalized (`0.99999994..1.00000012`), and all tensors are
finite.

The fresh Blueprint `x0` itself has RMS `0.835571` and is finite. Its source
cost is small in this controlled call, but Phase 20 is a semantic discriminator,
not an end-to-end performance qualification.

## Semantic result and failure localization

- **Blueprint x0: S3.** One continuous bridge dominates the canvas, with one
  centered train, controlled towers, coherent horizon/water, and no repeated
  independent bridge systems.
- **A: S0.** The canvas is a field of independent bridges, trains, towers, and
  incompatible water/horizon patches.
- **B: S0.** Fragment placement changes and some forms soften, but local calls
  still invent independent bridge/train alternatives. No dominant scene is
  transferred from the Blueprint.

Therefore fixed native-coordinate whole-canvas planning is not the failure in
this case. The first visible failure is after mapping the coherent Blueprint
prediction into same-sigma local W states and applying independent ordinary
local refinement. The tested state construction preserves its exact algebra,
so this specifically rejects that transfer rule, not bounded Blueprint
planning in general.

One next experiment is justified: a single terminal **prediction-anchor** test
using the already computed coherent `x0_B`, ordinary H-derived W input, and an
exact coarse correction of local `x0` toward the mapped Blueprint crop. This
would isolate prediction-space authority from noisy-state initialization. It
must remain zero-update and use no strength sweep.

## Artifacts

- Machine report: `flux2_candidate3_native_blueprint_local_state_results/report.json`
- Per-arm resumable artifacts: `flux2_candidate3_native_blueprint_local_state_results/arms/`
- Blueprint decode: `flux2_candidate3_native_blueprint_local_state_results/BLUEPRINT_X0.png`
- A/B sheet: `flux2_candidate3_native_blueprint_local_state_results/A_B_COMPARISON.png`
