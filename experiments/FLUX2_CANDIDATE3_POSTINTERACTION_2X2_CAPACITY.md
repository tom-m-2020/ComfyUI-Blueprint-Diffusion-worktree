# Phase 12b — intermediate post-interaction context capacity

Date: 2026-09-03

## Question

Does increasing post-interaction full-H context from 1,152 to 2,048 consumer
positions cross the semantic threshold established by the 8,192-position
full-H oracle?

This is a research-only capacity discriminator. Production and ComfyUI core are
unchanged.

## Fixed trajectory

The experiment exactly retains Phase 12: native FLUX.2 Klein 4B, seed
`20260901`, CFG 1, four-step CONST-flow Euler, `H=64×128`, fixed `G=24×48`, 15
destination crops (`32×32`, stride 24), reconstructed sigma-consistent `64×64`
native-coordinate W at every evaluation, unchanged W restriction and overlap
assembly, and unchanged Candidate-3 coupling and terminal release.

Every source is freshly constructed from the corresponding accepted state at
the current sigma. All 25 transformer blocks consume context. No context is
reused across accepted intervals.

## Variants

| Variant | Source | Consumer positions/block |
|---|---|---:|
| B_FIXED_G_CONTEXT | ordinary accepted 24×48 G | 1,152 |
| D_POST_INTERACTION_COMPRESSED | full accepted H, Phase-12 adaptive-area aggregation | 1,152 |
| E_POST_INTERACTION_2X2_2048 | full accepted H, nonoverlapping 2×2 aggregation | 2,048 |
| C_FULL_H_CONTEXT_ORACLE | full accepted H, no compression | 8,192 |

## E construction and positional provenance

At each source block, E obtains the exact generated K after native full-H RoPE
and the matching generated V from ordinary globally interacting `64×128` H.
Each tensor has shape `[1,24,8192,128]`. It is reshaped spatially and every
nonoverlapping `2×2` group is arithmetic-mean reduced to consumer K/V shape
`[1,24,2048,128]` (`32×64`).

K is pooled after native source RoPE, so each contribution retains its original
full-H positional phase. No fixed-G coordinate is assigned. The machine report
contains all 2,048 source-coordinate quads. Every source position occurs exactly
once. The provenance mapping hash is
`d9b591954980ee6b924bfd49e68f5906684ac89e6fba51e0a000425e4852aea6` and is
identical across the four fresh interval probes. The first quad covers
`(y,x)=(0,0),(0,1),(1,0),(1,1)` and the last covers
`(62,126),(62,127),(63,126),(63,127)` under the live four-axis IDs.

All 25 per-block records preserve these source/consumer shapes and boundaries.

## Integrity

- Four fresh E probes were created, one per accepted interval.
- Each captures 25 source blocks and records 375 local consumptions
  (`25 blocks × 15 crops`).
- Fresh accepted-state/same-sigma provenance, finite output, no stale context,
  immutable accepted inputs, and ordinary Candidate-3 invariants pass.
- Context source, local W execution, crop order, W transport, and assembly are
  identical except for the stated consumer representation.

## Results

### Overlap RMS per interval

| Variant | i0 | i1 | i2 | terminal i3 |
|---|---:|---:|---:|---:|
| B_FIXED_G_CONTEXT | 0.846419 | 0.631362 | 0.469160 | 0.250149 |
| D_POST_INTERACTION_COMPRESSED | 0.897635 | 0.657390 | 0.498455 | 0.259909 |
| E_POST_INTERACTION_2X2_2048 | 0.862110 | 0.645429 | 0.484400 | 0.254994 |
| C_FULL_H_CONTEXT_ORACLE | 0.824706 | 0.636335 | 0.471722 | 0.266542 |

E improves over D at every interval. This is secondary evidence only; C remains
semantically best despite having the highest terminal overlap RMS of the four.

### Assembled x0 RMS versus full-H oracle

| Variant | i0 | i1 | i2 | i3 |
|---|---:|---:|---:|---:|
| B_FIXED_G_CONTEXT | 0.514189 | 0.513629 | 0.535160 | 0.440736 |
| D_POST_INTERACTION_COMPRESSED | 0.508043 | 0.656207 | 0.633278 | 0.517928 |
| E_POST_INTERACTION_2X2_2048 | 0.461038 | 0.618936 | 0.605536 | 0.495723 |

E is consistently closer than D, but not close enough to reproduce C. B's
smaller terminal RMS again demonstrates that whole-scene semantics cannot be
ranked from latent RMS alone.

### Runtime and memory

| Variant | Tokens | Source CUDA | Local CUDA | Sampling wall | Peak allocated | Peak reserved | Cache/interval | Aggregate transfer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 1,152 | 1.862 s | 95.694 s | 99.694 s | 2.990 GiB | 3.436 GiB | 0.330 GiB | 19.775 GiB |
| D | 1,152 | 11.225 s | 97.141 s | 110.439 s | 3.665 GiB | 4.373 GiB | 0.330 GiB | 19.775 GiB |
| E | 2,048 | 11.697 s | 103.774 s | 117.566 s | 3.665 GiB | 4.373 GiB | 0.586 GiB | 35.156 GiB |
| C | 8,192 | 13.038 s | 151.578 s | 166.730 s | 3.665 GiB | 4.373 GiB | 2.344 GiB | 140.625 GiB |

The common full-H source determines D/E/C peak memory. Consumer cardinality
drives cache, transfer, and augmented local-attention runtime.

## Semantic result

The final comparison is
`flux2_candidate3_postinteraction_2x2_capacity_results/FINAL_COMPARISON.png`.

- E materially cleans the field relative to D: competing fine bridge fragments
  and incoherent water/sky structures are reduced, and the principal bridge
  reads more consistently.
- E still contains repeated train/truss/support interpretations along the deck
  and remains visibly below C's single dominant continuous bridge system.
- Towers, horizon, and water move toward C but do not meet the oracle semantic
  gate.

## Decision

**2,048 positions partially improve coherence but remain clearly below the
oracle.** This selects Decision 2. The context-capacity threshold is not shown to
lie between 1,152 and 2,048.

Exactly one further post-interaction discriminator is authorized: a 4,096-token
consumer obtained through a single fixed spatial aggregation rule, compared
against E and C. No production work, pooling-family sweep, or adaptive selection
is justified.
