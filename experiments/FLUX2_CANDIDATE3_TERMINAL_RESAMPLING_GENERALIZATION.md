# Phase 25 — Terminal-only Blueprint resampling generalization

## Result

The frozen Phase-24 terminal-only architecture generalized across both new
semantic classes. The multi-object and articulated-subject outputs are S3,
while their matched ordinary-local controls are S0. Together with the exact
reused bridge control, all three fixed cases pass.

This qualifies semantic generalization only for the tested prompt classes at
the existing `H=128x256` geometry. It does not qualify other geometries,
resampling sigmas, models, conditioning modes, or a production implementation.

## Frozen contract

- FLUX.2 Klein 4B, CFG 1, four-step CONST-flow Euler.
- Destination `H=128x256`; bounded native-coordinate Blueprint `32x64`.
- Destination regions `32x32`, stride 24, 55 end-aligned regions.
- Local working canvas `64x64` with native local coordinates.
- Terminal resampling sigma `0.25`.
- `W_0.25 = 0.75 * nearest2(Blueprint terminal crop) + 0.25 * epsilon_region`.
- The same deterministic region noise is used by each matched ordinary and
  Blueprint-resampled comparison.
- Ordinary FLUX local forward, exact `restrict2`, and the existing normalized
  overlap assembler. No post-anchor, D/U projection, K/V context, residual
  persistence, filtering, or additional refinement.

Seeds were declared before inference: `20260911` for the multi-object case and
`20260912` for the astronaut case. The bridge result is an exact Phase-24
artifact reuse and performs no new model work.

## Semantic evidence

| Case | Blueprint | Ordinary local | Blueprint-resampled | Result |
|---|---:|---:|---:|---|
| Bridge/train | S3 | S0 | S3 | Reused Phase-24 control; one bridge/train system |
| Car/tree/house | S3 | S0 | S3 | Exactly one car/tree/house in correct left/center/right order |
| Astronaut | S3 | S0 | S3 | One continuous centered body, two arms/legs, coherent ground contact |

The multi-object ordinary control contains repeated rows of independently
invented cars, trees, and houses. Terminal Blueprint initialization collapses
these into the Blueprint's one ordered scene with a continuous field and
horizon.

The astronaut ordinary control contains many independent astronauts and
repeated horizon strips. Terminal Blueprint initialization preserves one
centered full-body astronaut without duplicated people or body systems.

Softness from the fixed late refinement was not treated as semantic failure,
as required. No parameter was changed after inspecting the first new case.

## Numerical and runtime evidence

| Case/arm | Calls | Overlap RMS | Gradient RMS | RMS vs Blueprint | Low-frequency RMS | CUDA | Wall | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Multi ordinary | 220 | 0.893222 | 0.493040 | 0.919907 | 0.800899 | 220.587 s | 220.859 s | 3.160 / 3.429 GiB |
| Multi resampled | 55 | 0.162660 | 0.185873 | 0.122425 | 0.069409 | 55.495 s | 55.903 s | 3.310 / 3.615 GiB |
| Astronaut ordinary | 220 | 0.834318 | 0.532170 | 0.901844 | 0.758013 | 221.569 s | 221.704 s | 3.160 / 3.429 GiB |
| Astronaut resampled | 55 | 0.139880 | 0.175711 | 0.121437 | 0.082846 | 54.941 s | 55.332 s | 3.310 / 3.615 GiB |

The Blueprint trajectories require four model calls: 2.650 s CUDA / 2.775 s
wall for the multi-object case and 2.126 s CUDA / 2.206 s wall for the
astronaut case. The bridge terminal refinement reuses the Phase-24 telemetry:
55 calls, 54.869 s CUDA, overlap RMS `0.164391`, gradient RMS `0.187551`, and
RMS `0.119778` versus its terminal Blueprint.

The lower overlap RMS of selected outputs is compatible with the visual result
but was not used as the semantic decision criterion.

## Integrity and provenance

- Complete coverage is `[0.99999994, 1.00000012]` in both new cases.
- All Blueprint, ordinary, and selected tensors are finite.
- Working inputs remain immutable and accepted-state update count is zero for
  terminal diagnostics.
- Matched ordinary/selected region-noise hash lists are identical.
- Blueprint model calls receive no H feedback.
- Artifacts are persisted per case and stage with configuration hashes, making
  completed inference resumable without repeating model calls.
- No production or ComfyUI-core files changed.

Decoded RGB hashes:

| Case | Blueprint | Ordinary local | Blueprint-resampled |
|---|---|---|---|
| Multi-object | `3398b14d...81c62` | `7431b747...99db` | `7b156a83...9bf6` |
| Astronaut | `0f8e4750...368a` | `835034d9...e959` | `3b7b43b9...fd8b` |

Full hashes, tensor hashes, per-region noise hashes, prompts, timing, and memory
telemetry are in the machine-readable report.

## Artifacts

- Harness: `experiments/flux2_candidate3_terminal_resampling_generalization.py`
- Telemetry: `experiments/flux2_candidate3_terminal_resampling_generalization_results/report.json`
- Multi-object sheet: `experiments/flux2_candidate3_terminal_resampling_generalization_results/B_MULTI_OBJECT_COMPARISON.png`
- Astronaut sheet: `experiments/flux2_candidate3_terminal_resampling_generalization_results/C_ASTRONAUT_COMPARISON.png`
- Summary: `experiments/flux2_candidate3_terminal_resampling_generalization_results/SUMMARY.png`

## Decision

The exact staged mechanism preserves one coherent interpretation for a long
structure, a discrete spatial composition, and a large articulated subject.
This authorizes a separate Phase-26 production-architecture design task, not
production implementation.

**TERMINAL RESAMPLING GENERALIZES**
