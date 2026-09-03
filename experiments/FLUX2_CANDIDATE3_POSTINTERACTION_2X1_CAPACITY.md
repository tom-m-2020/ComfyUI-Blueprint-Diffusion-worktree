# Phase 12c — final brute-force post-interaction spatial capacity

Date: 2026-09-03

## Objective

Determine whether one deterministic 4,096-token post-interaction spatial
context can recover the whole-scene coherence of the 8,192-token full-H oracle.
This is an information-sufficiency experiment, not an efficiency or production
qualification.

No production or ComfyUI-core code changed.

## Fixed controls

All five trajectories use the exact Phase-12b setup: native FLUX.2 Klein 4B,
seed `20260901`, CFG 1, four-step CONST-flow Euler, `H=64×128`, fixed
`G=24×48`, 15 `32×32` destination regions at stride 24, reconstructed
sigma-consistent `64×64` W at every evaluation, native W coordinates, unchanged
W-to-H restriction and overlap assembly, and unchanged Candidate-3 coupling and
terminal release.

Context is regenerated from the current accepted source at the consuming sigma.
All 25 FLUX blocks expose context through the same generated-query integration
path while preserving ordinary text-query behavior.

| Variant | Source/context construction | Tokens/block |
|---|---|---:|
| A_FIXED_G_1152 | accepted 24×48 G | 1,152 |
| B_POST_INTERACTION_1152 | full H, Phase-12 area aggregation | 1,152 |
| C_POST_INTERACTION_2048 | full H, nonoverlapping 2×2 | 2,048 |
| D_POST_INTERACTION_4096 | full H, nonoverlapping vertical 2×1 | 4,096 |
| E_FULL_H_8192_ORACLE | full H, uncompressed | 8,192 |

## New 4,096-token condition

D advances ordinary accepted `64×128` H through the same globally interacting
source blocks as the oracle. At each block it obtains generated K after native
full-H RoPE and generated V at the same Phase-12 boundary. Tensor shape before
aggregation is `[1,24,8192,128]`.

The spatial field is reshaped to `64×128`; each pair of vertically adjacent
positions is arithmetic-mean aggregated, yielding `32×128` and K/V shape
`[1,24,4096,128]`. Pooling already-positioned K preserves the two contributing
full-H RoPE phases without assigning compact-G coordinates.

This construction is explicitly anisotropic: it retains full horizontal
sampling density because the test scene contains long horizontal bridge/train
geometry. It tests one deterministic 4,096-token representation, not all such
representations.

The machine report records all 4,096 source-coordinate pairs. All 8,192 source
positions appear exactly once, with zero omissions and zero duplicates. Mapping
SHA-256 is
`f66dcbe8809711d51ba198ab3ec9d651427c809d40c1047eef2b873768b4377e` across all
four fresh interval probes. The first pair is `(0,0),(1,0)` and the last is
`(62,127),(63,127)` under the live four-axis coordinate convention.

## Integrity

- Source positions: 8,192; D consumer entries: exactly 4,096.
- Every D interval records 25 source/context block boundaries and 375 context
  consumptions (`25×15`).
- Source and consumer tensor shapes are recorded at every block.
- Fresh accepted-state/same-sigma provenance, no stale reuse, accepted-state
  immutability, finite predictions/state, complete coverage, and Candidate-3
  coupling/terminal lifecycle checks pass.
- All non-context controls, local execution, assembly, and acceptance are
  unchanged.

## Results

### Overlap RMS per interval

| Variant | i0 | i1 | i2 | terminal i3 |
|---|---:|---:|---:|---:|
| A 1,152 fixed G | 0.846419 | 0.631362 | 0.469160 | 0.250149 |
| B 1,152 post | 0.897635 | 0.657390 | 0.498455 | 0.259909 |
| C 2,048 post | 0.862110 | 0.645429 | 0.484400 | 0.254994 |
| D 4,096 post | 0.829927 | 0.626119 | 0.459728 | 0.246554 |
| E 8,192 oracle | 0.824706 | 0.636335 | 0.471722 | 0.266542 |

D has the lowest terminal overlap RMS, but overlap remains a secondary
compatibility diagnostic rather than the semantic gate.

### Assembled x0 RMS versus oracle

| Variant | i0 | i1 | i2 | i3 |
|---|---:|---:|---:|---:|
| A | 0.514189 | 0.513629 | 0.535160 | 0.440736 |
| B | 0.508043 | 0.656207 | 0.633278 | 0.517928 |
| C | 0.461038 | 0.618936 | 0.605536 | 0.495723 |
| D | 0.349765 | 0.491778 | 0.500892 | 0.407488 |

Relative to C, D closes the oracle gap by about 24.1% at interval 0 and 17.8%
at terminal evaluation. The Phase-11/12 harness does not define a separate
low-frequency metric, so none is introduced post hoc.

### Runtime and context residency

| Variant | Tokens | Source CUDA | Local CUDA | Wall | Peak alloc/reserved | Cache/interval | Transfer/run |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1,152 | 1.922 s | 95.748 s | 99.809 s | 2.990/3.436 GiB | 0.330 GiB | 19.775 GiB |
| B | 1,152 | 11.157 s | 95.970 s | 109.203 s | 3.665/4.373 GiB | 0.330 GiB | 19.775 GiB |
| C | 2,048 | 11.695 s | 103.305 s | 117.096 s | 3.665/4.373 GiB | 0.586 GiB | 35.156 GiB |
| D | 4,096 | 12.166 s | 119.354 s | 133.621 s | 3.665/4.373 GiB | 1.172 GiB | 70.312 GiB |
| E | 8,192 | 12.946 s | 151.066 s | 166.132 s | 3.665/4.373 GiB | 2.344 GiB | 140.625 GiB |

D retains the oracle's full-H source cost and is not an efficient Blueprint
mechanism. Its consumer-side storage and transfer are exactly half the oracle's.

## Semantic inspection

The directly comparable panel is
`flux2_candidate3_postinteraction_2x1_capacity_results/FINAL_COMPARISON.png`.

- A/B show competing bridge spans, repeated truss/support alternatives, and a
  less stable train/deck band.
- C partially reduces these failures but retains obvious competing structures.
- D resolves to one dominant continuous horizontal bridge/train system. The end
  towers, deck, horizon, and water are coherent; floating or independently
  competing bridge systems are substantially suppressed. No new compression-
  specific scene-scale structure is apparent.
- E remains the cleanest/smoothest oracle, but D is in the same whole-scene
  semantic class rather than the fragmented A/B/C class.

## Decision

**A useful semantic threshold exists between 2,048 and 4,096 tokens for this
anisotropic post-interaction representation.** D is visually essentially
oracle-like at the requested scene-organization level and materially closes the
assembled numerical gap.

This does not qualify compute efficiency, generalization, or production use.
The source is still dense H-scale execution and D still moves 70.312 GiB of K/V
in this diagnostic. Do not test 5K/6K densities or productionize D. The next
research question is how to construct approximately equivalent ~4K globally
informed context substantially more cheaply than full-H source execution.
