# FLUX.2 Candidate-3 post-interaction K/V aggregation discriminator

## Verdict

**SIMPLE SPATIAL AGGREGATION DOES NOT COMPRESS GLOBAL CONTEXT.**

At the exact Phase-8f terminal state, arithmetic-mean aggregation of every
nonoverlapping 2×2 cell of the fully interacting 96×192 source K/V field does
not restore the qualified scene at a 4,608-token consumer budget. The pooled
result remains catastrophically fragmented and is numerically worse than the
already-failed top-left decimation control.

Aggregation therefore does not qualify. This result is scoped to the single
predeclared arithmetic-mean construction and does not establish that 18,432
tokens are intrinsically irreducible.

## Fixed state and controls

```text
output:                 2048×4096
H / G:                  128×256 / 96×192
seed:                   20260901
CFG / schedule:         1 / qualified four-step CONST-flow Euler
terminal crops:         55, 32×32, stride 24
source:                 one full 18,432-token current-G forward
source interaction:     ordinary dense, unchanged through all 25 blocks
local context depth:    double 0–4 and single 0–19
```

Production intervals 0–2 run once. Initial H/G, accepted H3/G3, terminal crop
inputs, crop rectangles, local coordinates, overlap assembly, and terminal
release are common to all variants.

The controls reproduce Phase 8f exactly:

| Control | Phase-8f decoded RGB hash | Phase-8g hash | Result |
|---|---|---|---|
| Full 18,432 | `53cad700…f259ce` | `53cad700…f259ce` | exact |
| 2×2 decimation | `001bb9ab…d2ce18` | `001bb9ab…d2ce18` | exact |

Their overlap, projection, and assembled-difference metrics also reproduce
exactly. This establishes strict accepted-state and source-path comparability.

## Aggregation boundary

For every double and single source block, the attention hook sees ordinary
Q/K/V projections from the fully evolved 96×192 source hidden state. The
generated source portion is then handled as follows:

```text
raw generated K [18,432] -> reshape 96×192 -> mean each 2×2 cell
generated V     [18,432] -> reshape 96×192 -> mean each 2×2 cell
source IDs      [18,432] -> reshape 96×192 -> mean each 2×2 cell
pooled raw K      [4,608] -> source pe_embedder(center IDs) -> apply RoPE
pooled V          [4,608] -> unchanged after mean
```

Instrumentation verifies in all 25 blocks:

- full source generated tokens: 18,432;
- pooled consumer tokens: 4,608;
- exactly four contributors per pooled token;
- all 18,432 positions contribute exactly once;
- K is pooled before RoPE;
- V is pooled at the ordinary generated-V boundary;
- all 25 local blocks consume the resulting context.

FLUX.2 Klein uses four positional axes in the live checkout. Floating-point
half-grid coordinates are natively accepted, so no representative-coordinate
fallback is used. The geometric-center IDs are:

```text
first: [0.0,   0.66842103,   0.66753924, 0.0]
last:  [0.0, 126.33157349, 254.33245850, 0.0]
```

They use the same endpoint-scaled full-canvas convention as the 96×192 source.

## Results

| Variant | Consumer tokens | Overlap RMS | Projection RMS | Projection/H-star | Assembled RMS vs full | Terminal-local CUDA |
|---|---:|---:|---:|---:|---:|---:|
| Full | 18,432 | 0.2812 | 0.3716 | 52.35% | 0 | 99.95 s |
| 2×2 decimation | 4,608 | 0.8027 | 0.7620 | 94.13% | 0.7004 | 45.43 s |
| 2×2 mean aggregation | 4,608 | 0.9090 | 0.8385 | 100.94% | 0.7964 | 42.05 s |

Per-crop RMS against full context is retained in `report.json`. Mean pooling
does not produce a hidden partial success masked by assembly: individual crops
and their overlaps remain far from the full-context reference.

All 4,608-token variants expose identical attention cardinality:

```text
local Q×K per block/crop:              1,536 × 6,144
total attention elements over 55×25:  12.976 billion
external-context portion:              9.732 billion
```

These are attention-matrix dimensions, not FLOP or speedup claims.

## Storage, transfer, and memory

```text
full source K/V cache:                5.27 GiB
additional pooled K/V cache:          1.32 GiB
full consumer transfer:             290.04 GiB
decimated consumer transfer:         72.51 GiB
pooled consumer transfer:            72.51 GiB
transfer reduction for pooling:          75%
whole experiment before decode:      288.74 s
```

The shared source capture takes 12.74 s CUDA in this experiment, including the
experiment-only construction of the pooled cache. That side computation does
not alter source hidden states or the returned source attention path.

Phase-local peak CUDA allocated/reserved memory is 1.28/1.41 GiB for both
4,608-token consumers, versus 1.59/6.67 GiB for full consumption. These values
are measured after shared model/source setup and are not physical-free-VRAM
measurements. The memory and timing reductions are not qualified because the
semantic result fails.

## Semantic review

- **Full:** one dominant bridge/deck/train system with controlled towers and a
  coherent horizon/water plane, matching Phase 8d–8f.
- **Decimation:** many independent bridge spans, cable fans, towers, and
  floating fragments, matching Phase 8f's catastrophic failure.
- **Mean pooling:** the same failure class persists. Multiple disconnected
  bridges occupy the sky and water, tower count is uncontrolled, and no single
  train/deck system governs the scene. Pooling does not visibly restore the
  full-context organization.

Fragmentation is present in the pooled assembled terminal x0_H before terminal
release. The result is comparable to, and by the recorded metrics worse than,
simple selection. It therefore does not satisfy either the qualification gate
or the “substantial improvement but still fragmented” interpretation.

## Interpretation

The failure of selection was not repaired by preserving the arithmetic mean of
all four vectors. Current-G's globally informed K/V field contains spatially
and directionally structured information that is not preserved by independently
averaging K and V inside fixed 2×2 cells and assigning a center RoPE coordinate.

No additional pooling function, density, window size, learned compressor,
quantizer, or token-merging heuristic is justified by this discriminator.

## Artifacts

- `flux2_candidate3_postinteraction_kv_aggregation.py`
- `flux2_candidate3_postinteraction_kv_aggregation_results/report.json`
- `POSTINTERACTION_KV_AGGREGATION_FINAL_COMPARISON.png`
- `POSTINTERACTION_KV_AGGREGATION_TERMINAL_X0_COMPARISON.png`

**SIMPLE SPATIAL AGGREGATION DOES NOT COMPRESS GLOBAL CONTEXT**
