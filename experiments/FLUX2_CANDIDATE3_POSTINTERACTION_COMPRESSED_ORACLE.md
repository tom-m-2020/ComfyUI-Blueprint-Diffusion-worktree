# Phase 12 — post-interaction compressed oracle context

Date: 2026-09-03

## Decision being tested

Phase 11 selected Gate 2: normalized local working canvases remain viable, but
the fixed-budget global representation construction is the bottleneck. Phase 12
asks whether that bottleneck is specifically compression *before* global
transformer interaction, or whether a 1,152-position consumer context is itself
too small under a simple information-preserving spatial aggregation rule.

No production or ComfyUI-core code changed.

## Controlled setup

All variants use the exact Phase-11 bridge/train trajectory: FLUX.2 Klein 4B,
seed `20260901`, CFG 1, four-step CONST-flow Euler, `H=64×128`, `G=24×48`, 15
destination regions of `32×32` at stride 24, reconstructed sigma-consistent
`64×64` native-coordinate W at every evaluation, unchanged W-to-H restriction,
overlap assembly, Candidate-3 coupling, and terminal release.

| Variant | Source execution | Consumer K/V positions/block |
|---|---|---:|
| A_LOCAL_ONLY | none | 0 |
| B_FIXED_G_CONTEXT | ordinary accepted 24×48 G | 1,152 |
| D_POST_INTERACTION_COMPRESSED_H_CONTEXT | ordinary accepted 64×128 H, then per-block compression | 1,152 |
| C_FULL_H_CONTEXT_ORACLE | ordinary accepted 64×128 H | 8,192 |

Every context source is freshly evaluated from the current accepted state at
the exact local-evaluation sigma. Context is consumed by all 25 blocks for all
15 local W calls. No context object survives an accepted interval.

## Post-interaction compression

D executes the same full `64×128` accepted-H source and ordinary globally
interacting FLUX.2 blocks as C. At each block:

1. obtain the generated source K after its native full-H RoPE and generated V;
2. reshape their 8,192 spatial positions to `64×128`;
3. apply deterministic adaptive-area arithmetic averaging to `24×48`;
4. expose exactly 1,152 pooled K/V positions to local W attention.

Pooling positioned K deliberately retains the native full-H positional phase of
every contributing source vector; it does not renumber the result onto compact-G
coordinates. Averaged full-H IDs are recorded only as spatial-support telemetry.
The first/last pooled support centers are `(y=1,x=1)` and `(y=62,x=126)` under
the live four-axis convention. PyTorch adaptive pooling defines the exact
nonuniform area supports needed for `64→24` and `128→48`.

All 25 records report source K/V shape `[1,24,8192,128]` and consumer shape
`[1,24,1152,128]`. Source computation and hidden evolution remain full-size;
only the local consumer interface is compressed.

## Integrity

- A/B/D/C share identical initial H/G and fixed non-context configuration.
- D has 25 compression records per interval, 25 captured blocks, and 375 local
  consumptions (`25×15`).
- Fresh-probe identity, accepted-state/same-sigma provenance, finite output, and
  no-stale-context checks pass for every context variant.
- All variants perform three ordinary Candidate-3 global forwards and 60 local
  W forwards. B/D/C perform four additional context-source forwards.

## Results

### Overlap RMS by interval

| Variant | i0 | i1 | i2 | terminal i3 |
|---|---:|---:|---:|---:|
| A_LOCAL_ONLY | 0.913870 | 0.653408 | 0.491292 | 0.259399 |
| B_FIXED_G_CONTEXT | 0.846419 | 0.631362 | 0.469160 | 0.250149 |
| D_POST_INTERACTION_COMPRESSED_H_CONTEXT | 0.897635 | 0.657390 | 0.498455 | 0.259909 |
| C_FULL_H_CONTEXT_ORACLE | 0.824706 | 0.636335 | 0.471722 | 0.266542 |

D is not numerically close to C. Its assembled-x0 RMS versus C is `0.508043,
0.656207, 0.633278, 0.517928` across the four evaluations. D versus B is
`0.324631, 0.484326, 0.370165, 0.280190`. As in Phase 11, overlap is treated as
a compatibility diagnostic rather than the semantic decision criterion.

### Runtime and memory

| Variant | Context tokens | Source CUDA | Local CUDA | Sampling wall | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| A_LOCAL_ONLY | 0 | 0 | 59.114 s | 61.445 s | 2.874 GiB | 3.260 GiB |
| B_FIXED_G_CONTEXT | 1,152 | 1.754 s | 96.314 s | 100.218 s | 2.990 GiB | 3.436 GiB |
| D_POST_INTERACTION_COMPRESSED_H_CONTEXT | 1,152 | 11.323 s | 96.560 s | 110.048 s | 3.665 GiB | 4.373 GiB |
| C_FULL_H_CONTEXT_ORACLE | 8,192 | 13.085 s | 151.353 s | 166.642 s | 3.665 GiB | 4.373 GiB |

D and B have the same per-block cache (`0.330 GiB`) and aggregate diagnostic
CPU-to-GPU K/V transfer (`19.775 GiB`). C uses `2.344 GiB` per interval and
`140.625 GiB` aggregate transfer. D retains C's full-source peak because source
execution is intentionally not compressed.

## Semantic result

The final comparison is
`flux2_candidate3_postinteraction_compressed_oracle_results/FINAL_COMPARISON.png`.

- A and B reproduce their Phase-11 outputs and repeated bridge/train/support
  alternatives.
- D remains in the same fragmented semantic class. It changes texture and local
  structures, but does not collapse the repeated red bridge/train band into one
  continuous interpretation. It is visually much closer in outcome class to
  A/B than to C.
- C again yields one dominant continuous bridge system, controlled end towers,
  and coherent water/horizon.

Thus globally interacting the source *before* compression is not sufficient at
the 1,152-position consumer budget with this fixed area-mean rule.

## Decision

**The fixed context capacity itself is insufficient at 1,152 positions under
the tested post-interaction aggregation.** This falsifies the claim that
pre-interaction compression alone explains B's failure. It does not prove that
every possible 1,152-token learned or nonlinear representation must fail.

The only justified next discriminator is one clean, bounded larger
post-interaction budget: nonoverlapping 2×2 aggregation from full interacted
`64×128` H K/V to `32×64 = 2,048` consumer positions, compared against this
1,152 control and the 8,192 oracle. No production work is justified.
