# Phase 15 — fixed-4096 source-statistics discriminator

Date: 2026-09-03

## Question

Was Phase 14's large-canvas failure primarily caused by the natural `4×2`
area-mean source collapsing accepted-H variance to approximately one eighth?

This terminal-only experiment changes exactly one value: a fixed scalar applied
after the deterministic 8-contributor mean. Geometry remains `H=128×256`,
source `32×128 = 4,096`, 55 normalized `W=64×64` consumers, and all 25 source
and consumer blocks use the same block-major executor. Production and ComfyUI
core are unchanged.

## Variants

| Variant | Definition | A-priori target variance ratio |
|---|---|---:|
| C0 mean | `S = mean(cell)` | 0.125 |
| C1 Phase-13 scale | `S = 2 mean(cell)` | 0.5 |
| C2 variance preserving | `S = sqrt(8) mean(cell)` | 1.0 |

Phase-14 A local-only and B destination-scaled evidence is reused rather than
rerun. C0 reproduces the Phase-14 C assembled latent and decoded RGB hashes
exactly.

## Integrity

- Accepted H/G and all 55 W hashes match Phase 14 exactly.
- The source always has 4,096 positions with the same complete `4×2`
  provenance, coordinates, and zero omissions/duplicates.
- Every arm executes 25 source blocks and 1,375 consumer-block calls.
- Accepted H/G and W remain immutable; no terminal update occurs.
- Coverage is complete and every result is finite.

## Source and early-model statistics

| Variant | Gain | Variance ratio | RMS ratio | input image-token RMS | after img_in RMS |
|---|---:|---:|---:|---:|---:|
| C0 | 1.0000 | 0.124651 | 0.353060 | 0.351376 | 0.185482 |
| C1 | 2.0000 | 0.498606 | 0.706121 | 0.702753 | 0.370965 |
| C2 | 2.8284 | 0.997211 | 0.998605 | 0.993846 | 0.524627 |

The scalar survives input projection exactly as an amplitude change. It does
not survive the first transformer attention boundary in the same way:

| Variant | double-0 hidden RMS | double-0 K RMS | double-0 V RMS |
|---|---:|---:|---:|
| C0 | 2.078963 | 1.184487 | 7.383815 |
| C1 | 2.013299 | 1.184486 | 7.383898 |
| C2 | 1.956393 | 1.184487 | 7.383458 |

Double-0 K is invariant to displayed precision and V changes by less than
`0.01%`. Native normalization/modulation therefore largely removes the simple
input-amplitude distinction before the first useful context boundary.

Later source hidden/K/V remains close in scale rather than tracking the input
gain:

| Variant | single-9 hidden/K/V RMS | single-19 hidden/K/V RMS |
|---|---|---|
| C0 | 64.9004 / 1.55785 / 0.37069 | 94.8470 / 1.49413 / 0.91589 |
| C1 | 65.0843 / 1.55811 / 0.39727 | 94.5117 / 1.49277 / 0.93170 |
| C2 | 65.1551 / 1.55822 / 0.42133 | 94.1027 / 1.49076 / 0.94468 |

Full per-channel input summaries and representative block tensors are recorded
in `report.json`.

## Numerical and semantic result

| Variant | Terminal overlap RMS | Assembled RMS | Assembled RMS vs C0 |
|---|---:|---:|---:|
| C0 | 0.784053 | 0.694729 | 0 |
| C1 | 0.774983 | 0.691374 | 0.145132 |
| C2 | 0.767068 | 0.689794 | 0.240878 |

The modest monotonic overlap reduction does not correspond to a semantic
class change. The comparison panel shows all three arms retain many independent
bridge spans, towers/supports, train-like alternatives, and discontinuous
horizon/water structure. Neither C1 nor C2 approaches one dominant coherent
bridge system.

Thus Phase 14 was not primarily a scalar variance-collapse failure. Matching
the Phase-13 variance regime and even restoring near-native variance do not
restore the information discarded by the `4×2` area restriction.

## Cost check

| Variant | Source CUDA | Local CUDA | Wall | Peak allocated/reserved | block K/V |
|---|---:|---:|---:|---:|---:|
| C0 | 1.063 s | 101.740 s | 104.211 s | 3.79 / 5.89 GiB | 48 MiB |
| C1 | 1.085 s | 102.562 s | 105.038 s | 3.83 / 5.93 GiB | 48 MiB |
| C2 | 1.071 s | 102.294 s | 104.710 s | 3.83 / 5.93 GiB | 48 MiB |

Scalar rescaling does not materially change the established fixed-4K work or
memory behavior. Performance remains secondary because no arm passes the
semantic gate.

## Decision

**Neither scalar changes the fragmentation class. Simple scalar-statistics
correction is rejected.** The next discriminator must preserve richer
information at the same 4,096 spatial positions; increasing global density is
not justified.
