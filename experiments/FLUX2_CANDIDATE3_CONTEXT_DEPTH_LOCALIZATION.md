# Phase 8e — terminal global-context consumption localization

## Result

The Phase-8d semantic result requires global context to persist through the
complete tested local transformer depth. Only all 5 double-stream plus all 20
single-stream blocks produce the qualified single bridge scene. Single-stream
context alone retains much of the main bridge but leaves multiple independent
floating bridge fragments. Double-only and both fixed ten-block placements
fail materially.

No narrower early prefix was run because the predeclared early ten-block
variant did not retain the full-context semantic result.

## Fixed controls

- 2048x4096; H=128x256; G=96x192; production 4→3 DCT.
- Phase-8d bridge/train prompt, seed 20260901, CFG 1, four-step schedule.
- Production intervals 0–2 run once; one terminal H3/G3 and sigma.
- One ordinary, globally interacting current-G source forward captured all 25
  blocks once. Every local variant used the same K/V tensor objects.
- Identical 55 crops, crop order, inputs, coordinates, overlap weights,
  terminal release, and no output-space projection.
- The 18,432-token context, RoPE, token ordering, and CPU-offloaded storage are
  identical to Phase 8d. Only local consumption is gated.

The full-context final RGB hash is
`53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce`,
exactly reproducing Phase 8d.

## Predeclared block sets

| Variant | Consuming blocks | Count |
|---|---|---:|
| A full | double 0–4; single 0–19 | 25 |
| B double only | double 0–4 | 5 |
| C single only | single 0–19 | 20 |
| D early ten | double 0–4; single 0–4 | 10 |
| E late ten | single 10–19 | 10 |

Disabled blocks run their ordinary local attention. They do not append global
K/V, transfer cache tensors, or execute an augmented attention call.

## Semantic observations

### A — full context

One dominant suspension bridge, continuous deck/cables, one yellow train,
controlled large towers, coherent water/horizon, and retained local detail.
Minor small red remnants remain, matching Phase 8d.

### B — double only

Catastrophic fragmentation returns: many independent decks, towers, floating
spans, repeated train-like marks, and incompatible horizons. Five early double
blocks do not establish a scene that ordinary single blocks maintain.

### C — single only

This is the strongest partial result. It preserves one main bridge/deck/train
system and broadly coherent water, but several independent red bridge fragments
float across the sky and lower water. It is materially better than double-only
and either ten-block variant but does not meet the no-floating-fragments gate.

### D — early ten

A recognizable main bridge begins to form, but many large alternative spans
and broken cable systems remain across the sky and water. Early context does
not survive the ordinary remaining single-stream depth.

### E — late ten

The image remains close to the fragmented local-only failure, with many
independent structures. Late correction without earlier maintained context is
insufficient.

Thus deep single-stream consumption is more important than double-only
consumption, but double context still materially improves the full result over
single-only. The minimum qualified set in this bounded experiment is all 25
blocks.

## Quantitative comparison

| Variant | Overlap RMS | Projection RMS | Projection/H* | Assembled RMS vs full |
|---|---:|---:|---:|---:|
| Full 25 | 0.281238 | 0.371625 | 52.35% | 0 |
| Double 5 | 0.902332 | 0.838373 | 99.67% | 0.805749 |
| Single 20 | 0.539251 | 0.468209 | 65.85% | 0.354130 |
| Early 10 | 0.722103 | 0.720927 | 93.28% | 0.658394 |
| Late 10 | 0.824729 | 0.737828 | 96.89% | 0.693111 |

Every partial variant requires materially more scene-scale diagnostic
projection than full context. Single-only closes much of the numerical gap,
but its visible floating alternatives prevent qualification.

## Consumption work, transfer, and runtime

Every enabled block/crop performs augmented attention with Q×K =
1,536×19,968. The table reports only augmented attention elements, not total
model FLOPs.

| Variant | Augmented Q×K elements | K/V transfer | Reduction vs full | Terminal-local CUDA | CUDA reduction |
|---|---:|---:|---:|---:|---:|
| Full 25 | 42.172 B | 290.06 GiB | 0% | 95.89 s | 0% |
| Double 5 | 8.434 B | 58.01 GiB | 80% | 34.01 s | 64.5% |
| Single 20 | 33.738 B | 232.05 GiB | 20% | 80.18 s | 16.4% |
| Early 10 | 16.869 B | 116.03 GiB | 60% | 49.85 s | 48.0% |
| Late 10 | 16.869 B | 116.03 GiB | 60% | 50.75 s | 47.1% |

The shared source capture remains 5.27 GiB on CPU and costs 11.71 s CUDA.
Whole experiment wall time, including one shared trajectory/source, five local
variants, decoding, and bookkeeping before decode completion, is 410.42 s.

Per-variant peak CUDA allocated/reserved after the shared source capture:

| Variant | Peak allocated | Peak reserved |
|---|---:|---:|
| Full | 1.59 GiB | 6.67 GiB |
| Double | 1.51 GiB | 1.62 GiB |
| Single | 1.59 GiB | 1.75 GiB |
| Early | 1.59 GiB | 1.73 GiB |
| Late | 1.59 GiB | 1.85 GiB |

These allocator peaks are phase-local after CPU cache construction and model
offload/reuse; they must not be read as whole-generation residency. Transfer
and CUDA-time reductions are the meaningful controlled comparisons here.

## Interpretation

Global context is not consumed once to create a durable early local plan.
Ordinary downstream blocks redirect the crop hidden trajectories back toward
independent local alternatives. The all-single result shows that deep context
maintenance carries most of the benefit; the remaining difference from full
context shows that double-stream consumption also contributes.

This is consistent with prior Candidate-2 source-depth evidence that early
global organization was not preserved through later restricted execution, but
Phase 8e establishes the distinct local-consumption boundary directly.

The result does not justify another block-count sweep or production change.
The present semantic contract is context throughout local depth; making that
efficient requires a different mechanism than simply omitting contiguous
consumer ranges.

## Verdict

GLOBAL CONTEXT MUST PERSIST THROUGH LOCAL DEPTH
