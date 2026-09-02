# FLUX.2 Candidate-3 post-interaction K/V density discriminator

## Verdict

**LOCAL CONSUMPTION REQUIRES HIGH-DENSITY GLOBAL K/V.**

At the fixed Phase-8d/8e terminal state, deterministic 2×2 spatial
decimation of the already globally informed 96×192 source K/V is catastrophic.
The 4,608-token consumer context does not retain a dominant bridge system: the
55 local predictions again assemble into many independent bridge alternatives.
The 1,152-token 4×4 control is worse. Only all 18,432 post-interaction positions
reproduce the qualified Phase-8d scene.

This is not a reduced-source experiment. One ordinary full 96×192 current-G
forward performs unchanged dense global interaction and captures all 25
blocks. Selection happens only afterward, at the local consumer interface.

## Fixed state and causal isolation

```text
output:                  2048×4096
H / G:                   128×256 / 96×192
seed:                    20260901
CFG / schedule:          1 / qualified four-step CONST-flow Euler
terminal crops:          55, 32×32, stride 24
source capture:          one full 18,432-token current-G forward
local context depth:     double 0–4 and single 0–19 for every variant
sampler updates tested:  zero beyond the shared accepted H3/G3
```

Initial H/G, all three nonterminal accepted states, H3/G3, terminal sigma,
crop rectangles, terminal crop inputs, local RoPE, overlap weights, and local
execution order match Phase 8e. The full-source K/V object identities remain
unchanged across variants. All variants use the same full source output and
fresh terminal G-star diagnostic.

The full control exactly reproduces the Phase-8d/8e decoded RGB hash:

```text
53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce
```

## Post-interaction selection

The source capture applies source RoPE before storing generated K. V is stored
at the same ordinary capture boundary. For decimation, the experiment retains
the top-left row-major position from each fixed spatial cell:

```text
A: every 1×1 cell -> 96×192 -> 18,432 positions
B: every 2×2 cell -> 48×96  ->  4,608 positions
C: every 4×4 cell -> 24×48  ->  1,152 positions
```

The selected K tensors are already positioned with their original 96×192
absolute source RoPE. They are not renumbered to 48×96 or 24×48. No reduced
G latent, reduced-resolution source model, pooling, interpolation, learned
compression, K/V quantization, block selection, or source-attention change is
present.

## Quantitative results

| Variant | Retained K/V per block | Fraction | Local Q×K | External-context Q×K across 55×25 | Overlap RMS | Projection RMS | Projection/H-star | Assembled RMS vs full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full | 18,432 | 100% | 1,536×19,968 | 38.928B | 0.2812 | 0.3716 | 52.35% | 0 |
| 2×2 decimation | 4,608 | 25% | 1,536×6,144 | 9.732B | 0.8027 | 0.7620 | 94.13% | 0.7004 |
| 4×4 decimation | 1,152 | 6.25% | 1,536×2,688 | 2.433B | 0.9727 | 0.8621 | 101.42% | 0.8339 |

“External-context Q×K” counts local/text query rows against only the appended
current-G keys. It is not a FLOP or wall-time claim. The total attention matrix
counts, including ordinary text/local keys, are 42.172B, 12.976B, and 5.677B.

The 2×2 result is not a close numerical approximation whose perceptual status
is ambiguous. Relative to full context, its assembled prediction changes by
0.700 RMS and overlap disagreement rises 2.85×. Its diagnostic coarse
correction rises from 52.4% to 94.1% of H-star RMS. The 4×4 result approaches
the Phase-8c scene-scale projection requirement.

## Runtime, transfer, and memory

The shared ordinary full-source capture is unchanged for all variants:

```text
source CUDA:             11.75 s
full CPU source cache:    5.27 GiB
source peak alloc/res:    3.68 / 6.67 GiB
```

| Variant | Selected consumer K/V footprint | CPU→GPU K/V consumed | Transfer reduction | Terminal-local CUDA | Terminal-local wall | Phase-local peak alloc/res |
|---|---:|---:|---:|---:|---:|---:|
| Full | 5.27 GiB | 290.04 GiB | 0% | 96.17 s | 96.17 s | 1.59 / 6.67 GiB |
| 2×2 | 1.32 GiB | 72.51 GiB | 75% | 43.34 s | 43.34 s | 1.28 / 1.41 GiB |
| 4×4 | 0.33 GiB | 18.13 GiB | 93.75% | 27.47 s | 27.47 s | 1.21 / 1.38 GiB |

The selected-footprint values describe the consumer subset; the experiment
retains the full 5.27 GiB capture for provenance. Peak allocator measurements
are phase-local after shared model/source setup and must not be interpreted as
physical free VRAM. Whole experiment wall time before six VAE decodes was
266.33 s.

The reduced variants demonstrate that consumer cardinality materially affects
runtime and transfers. They do not qualify those savings because they destroy
the established semantic result.

## Semantic review

- **Full:** one dominant bridge/deck/train system, controlled major towers,
  coherent horizon and water; minor red fragments remain as in Phase 8d.
- **2×2:** numerous independent bridge decks, cable fans, and towers fill the
  sky and water. Train/deck continuity and major-tower control are lost. This
  is catastrophic fragmentation rather than a modest detail regression.
- **4×4:** similarly fragmented with still weaker cross-crop agreement and no
  coherent dominant bridge system.

Fragmentation is already present in the assembled terminal x0_H for both
decimated variants; terminal release merely publishes it.

Because the 2×2 variant fails catastrophically, the conditional additional
intermediate density was not run. The fixed 4×4 control was part of the
required comparison, not an added sweep.

## Interpretation

Dense source interaction is necessary but not sufficient for this scene:
local consumers also require substantially denser spatial access than the
tested 4,608-position field. A single regularly sampled representative per
2×2 source cell cannot carry the distributed semantic constraints expressed by
the full post-interaction field, despite every retained vector having been
constructed by the same globally dense source trajectory.

The minimum qualified consumer density in this discriminator is therefore the
unchanged 96×192 field, 18,432 positions per block. This does not prove every
position is individually necessary, nor does it rule out a different
information-preserving compression. It does reject fixed regular 2×2 and 4×4
post-interaction decimation as production directions.

## Artifacts

- `flux2_candidate3_postinteraction_kv_density.py`
- `flux2_candidate3_postinteraction_kv_density_results/report.json`
- `POSTINTERACTION_KV_DENSITY_FINAL_COMPARISON.png`
- `POSTINTERACTION_KV_DENSITY_TERMINAL_X0_COMPARISON.png`

**LOCAL CONSUMPTION REQUIRES HIGH-DENSITY GLOBAL K/V**
