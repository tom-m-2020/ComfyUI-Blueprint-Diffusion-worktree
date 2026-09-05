# Phase 29 — Terminal-resampling refinement-strength discriminator

## Result

Phase 29 reused the exact Phase-28 Blueprint trajectories and varied only the
model-native terminal local resampling sigma over `0.10`, `0.15`, `0.25`,
`0.35`, and `0.50`. All fifteen completed branches are finite, fully covered,
and persisted independently. The `0.25` control is bit-exact to the Phase-28
selected latent for every case.

**Decision: SCALAR TERMINAL SIGMA INSUFFICIENT.**

All tested outputs remain S3, including `0.50`, but no lower or higher sigma
materially and consistently improves visible local fidelity over `0.25` across
the three scene classes. Higher sigma increases measured gradient energy, but
also increases deviation from the coherent Blueprint and pre-blend overlap
disagreement. The visible softness is therefore not primarily explained by the
choice of scalar terminal sigma within the tested range.

## Fixed controls and resume validation

- Model, conditioning, prompt, seed, Blueprint initialization and four-step
  trajectory, destination/Blueprint geometry, terminal mapping, region order,
  native `64x64` local coordinates, deterministic region noise, nearest lift,
  `2x2` arithmetic restriction, and normalized overlap assembly are unchanged.
- Each case has one persisted Blueprint artifact. Its initial, per-interval
  prediction, accepted-state, terminal-prediction, and mapped-prediction hashes
  are shared by all five sigma branches.
- Each completed branch has its own latent, JSON telemetry, and decoded PNG.
  Resume validation checks configuration and Blueprint/mapping fingerprints.
- The `0.25` final latent comparison is `RMS=0`, `max_abs=0` for all cases.
- No branch executed a destination-sized model forward. Blueprint work was
  reused (four calls per case); terminal local calls are exactly the qualified
  region count: 25 square, 55 portrait, and 40 landscape.

## Semantic review

| Case | 0.10 | 0.15 | 0.25 | 0.35 | 0.50 | Visual finding |
|---|---:|---:|---:|---:|---:|---|
| square car/tree/house | S3 | S3 | S3 | S3 | S3 | One car, one central tree, and one house remain in the same ordered scene. No crop-local alternatives. |
| portrait astronaut | S3 | S3 | S3 | S3 | S3 | One continuous astronaut remains grounded in one scene. No duplicate body system appears. Softness is not materially relieved at lower sigma. |
| landscape bridge | S3 | S3 | S3 | S3 | S3 | One continuous bridge/train interpretation and coherent water/horizon remain. No competing bridge system appears. |

The frozen ordinary tiled-local references remain S0 and were not recomputed.
No semantic duplication/recomposition boundary was reached by `sigma=0.50`.
Numerically stronger local freedom therefore did not translate into a reliable
cross-case fidelity improvement.

## Measurements

`BP RMS` is final latent RMS from the mapped terminal Blueprint; `LF RMS` is
the existing low-frequency discrepancy; `overlap` is pre-blend disagreement.

| Case | sigma | overlap | gradient RMS | BP RMS | LF RMS | local CUDA s | wall s | peak alloc / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| square | 0.10 | 0.07035 | 0.20043 | 0.05125 | 0.02300 | 24.58 | 25.14 | 2.838 / 3.316 |
| square | 0.15 | 0.10317 | 0.20901 | 0.07751 | 0.03703 | 24.65 | 25.13 | 2.838 / 3.316 |
| square | 0.25 | 0.17345 | 0.23882 | 0.13250 | 0.06465 | 24.59 | 25.13 | 2.837 / 3.316 |
| square | 0.35 | 0.23721 | 0.27218 | 0.18928 | 0.10055 | 24.77 | 25.22 | 2.838 / 3.316 |
| square | 0.50 | 0.31728 | 0.31312 | 0.26949 | 0.16180 | 24.84 | 25.31 | 2.838 / 3.316 |
| portrait | 0.10 | 0.06727 | 0.14559 | 0.05025 | 0.02924 | 55.08 | 56.10 | 2.859 / 3.275 |
| portrait | 0.15 | 0.09884 | 0.15524 | 0.07445 | 0.04529 | 55.22 | 56.24 | 2.859 / 3.275 |
| portrait | 0.25 | 0.17225 | 0.18823 | 0.12441 | 0.07378 | 54.65 | 55.68 | 2.859 / 3.275 |
| portrait | 0.35 | 0.24768 | 0.22901 | 0.18021 | 0.11093 | 55.30 | 56.31 | 2.859 / 3.275 |
| portrait | 0.50 | 0.33732 | 0.28622 | 0.26243 | 0.17442 | 55.25 | 56.25 | 2.859 / 3.275 |
| landscape | 0.10 | 0.06742 | 0.16330 | 0.05000 | 0.02495 | 40.53 | 41.26 | 2.848 / 3.412 |
| landscape | 0.15 | 0.09728 | 0.17208 | 0.07402 | 0.03881 | 40.20 | 40.95 | 2.848 / 3.412 |
| landscape | 0.25 | 0.16277 | 0.20241 | 0.12333 | 0.06427 | 40.36 | 41.13 | 2.848 / 3.412 |
| landscape | 0.35 | 0.22089 | 0.23495 | 0.17464 | 0.09820 | 40.18 | 40.91 | 2.848 / 3.412 |
| landscape | 0.50 | 0.30008 | 0.28480 | 0.25120 | 0.15422 | 41.08 | 41.83 | 2.848 / 3.412 |

Relative to `0.25`, final-latent RMS for `0.10 / 0.15 / 0.35 / 0.50` is:

- square: `0.08591 / 0.05950 / 0.07009 / 0.16939`;
- portrait: `0.08015 / 0.05624 / 0.07197 / 0.17766`;
- landscape: `0.07894 / 0.05480 / 0.06530 / 0.16113`.

Runtime and peak memory are effectively sigma-invariant. The scalar changes
state content, not model geometry, call count, or attention work. Post-region
barrier telemetry remains bounded rather than growing with region ordinal.

## Interpretation and next discriminator

Measured facts: lowering sigma pulls the result closer to the mapped Blueprint
and reduces overlap disagreement and gradient energy. Raising sigma does the
opposite. Visual judgment: neither direction yields a material, repeatable
detail improvement across square, portrait, and landscape while retaining the
already-qualified scene organization; all variants remain S3.

Architectural inference: the remaining softness is more plausibly caused by
the mapped low-resolution Blueprint anchor, nearest lift, and single ordinary
denoising call than by selecting the wrong scalar sigma.

Stop scalar sigma tuning. The next recommended task is **Phase 30 — fixed
Blueprint-guided local refinement discriminator**: retain the qualified
`sigma=0.25` initialization and test exactly one mechanism that maintains
Blueprint authority inside a freer model-mediated local refinement, against
the current initialization-only control. It must not be a sigma, strength, or
filter sweep.

## Artifacts

- Harness: `experiments/flux2_terminal_resampling_refinement_strength.py`
- Machine report: `experiments/flux2_terminal_resampling_refinement_strength_results/report.json`
- Comparison sheet: `experiments/flux2_terminal_resampling_refinement_strength_results/PHASE29_SIGMA_COMPARISON.jpg`
- Per-case Blueprint, per-sigma latent/telemetry, and decoded images are under
  `experiments/flux2_terminal_resampling_refinement_strength_results/`.
