# Phase 14 — destination-independent fixed-4K source at 2048×4096

Date: 2026-09-03

## Questions

1. Can a `2048×4096` destination remain globally coherent using a fixed
   `32×128 = 4,096`-token whole-canvas source?
2. Under the same block-major executor, is that source cheaper than the
   destination-scaled `96×192 = 18,432`-token source?

This is a terminal, zero-update research discriminator. Production and ComfyUI
core are unchanged.

## Controlled state and execution

The bridge/train case uses seed `20260901`, CFG 1, the qualified four-step
CONST-flow schedule, accepted preterminal `H=128×256` and `G=96×192`, 55
destination regions (`32×32`, stride 24), and reconstructed sigma-consistent
native-coordinate `W=64×64` consumers. A, B, and C share identical accepted-H,
accepted-G, and all 55 W hashes. No terminal state is accepted.

| Variant | Context source | Source tokens | Source execution |
|---|---|---:|---|
| A local-only | none | 0 | ordinary sequential W calls |
| B destination-scaled | accepted G `96×192` | 18,432 | 25-block block-major |
| C fixed direct | `4×2` area mean of accepted H to `32×128` | 4,096 | same 25-block block-major |

B and C use the same explicit native FLUX.2 block executor and the same local
generated-query context integration. Each advances one source block, exposes
only that block's full generated K/V to every W crop, releases it, and moves to
the next block. Neither uses an all-layer host cache, host-to-device K/V
transfer, or source final projection.

## Fixed-source mapping and integrity

For C:

```text
S[r,c] = mean(H[4r:4r+4, 2c:2c+2])
y = 4r + 1.5
x = 2c + 0.5
```

Live RoPE options are `scale_y=4`, `shift_y=1.5`, `scale_x=2`,
`shift_x=0.5`. All 32,768 H positions contribute once to 4,096 source
positions, eight contributors per position, with zero omissions or duplicates.
The machine report records complete source-coordinate provenance.

All variants use 55 regions and complete positive coverage. B/C execute 25
source blocks and 1,375 local context consumptions. Accepted H/G and W are
immutable; all outputs are finite.

## State statistics

Natural `4×2` area averaging changes the noisy-state distribution materially:

| Tensor | RMS | standard deviation | mean |
|---|---:|---:|---:|
| accepted H | 0.995232 | 0.995232 | -0.000434 |
| fixed source | 0.351377 | 0.351377 | -0.000434 |

The fixed-source/H variance ratio is `0.124651` and RMS ratio `0.353060`, far
below Phase 13's successful pair-mean variance ratios near `0.50–0.53`. No gain
or variance correction was applied, as required.

## Semantic result

The comparison panel is
`flux2_candidate3_fixed4k_large_destination_results/A_B_C_comparison.png`.

- A reproduces catastrophic local fragmentation: many independent bridge,
  tower, and train alternatives with discontinuous horizon/water.
- B reduces overlap disagreement and produces a lower-contrast result, but it
  does **not** reproduce the historical single-bridge positive result after the
  consumer was changed from direct `32×32` crops to normalized `64×64` W.
  Separated truss/bridge fragments remain. Thus the old destination-scaled
  qualification does not transfer automatically to this combined consumer.
- C is sharper than B but remains clearly fragmented, with many independent
  bridge spans, towers, supports, and train-like fragments. It does not pass the
  one-scene semantic gate.

This means Phase 14 does not support destination-independent fixed-4K context
under the natural `4×2` restriction. It also reveals that B is not a valid
positive semantic control for normalized W; the previous 18,432-token success
used direct `32×32` consumers. This control failure is reported rather than
silently treating B as qualified.

## Numerical diagnostics

| Variant | Terminal overlap RMS | Assembled RMS |
|---|---:|---:|
| A local-only | 0.856211 | 0.744668 |
| B scaled context | 0.616649 | 0.576828 |
| C fixed 4K | 0.784053 | 0.694729 |

C differs from B by RMS `0.552490` (max absolute `4.70294`) and from A by RMS
`0.580216` (max absolute `5.21671`). Lower B overlap RMS does not imply semantic
success; visual whole-scene organization is the primary gate.

## Measured cost under the same executor

| Metric | B 18,432 | C 4,096 | C relative to B |
|---|---:|---:|---:|
| source CUDA | 9.677 s | 1.083 s | 11.2% |
| local CUDA | 177.909 s | 103.882 s | 58.4% |
| final projection CUDA | 0.209 s | 0.204 s | 98.0% |
| specialized CUDA total | 187.794 s | 105.169 s | 56.0% |
| terminal wall | 187.844 s | 105.234 s | 56.0% |
| peak allocated | 5.23 GiB | 3.58 GiB | 68.5% |
| peak reserved | 7.43 GiB | 5.65 GiB | 76.0% |
| current-block source K/V | 216 MiB | 48 MiB | 22.2% |
| source hidden state | 111 MiB | 27 MiB | 24.3% |

Local generated-token work is identical at `55 × 4,096 = 225,280` tokens.
The local CUDA reduction comes from lower attention source-key cardinality, not
reduced local-token MLP/projection work. These are measured runtime/memory
results, not token-count-derived speed claims.

## Answers and next discriminator

1. **No.** The fixed 4,096-token source does not keep this 2048×4096 bridge
   scene coherent under natural `4×2` area restriction; it returns to the
   fragmented semantic class.
2. **Yes.** Under the identical block-major executor, the 4K source is genuinely
   cheaper: source CUDA falls by 88.8%, total specialized/wall time by about
   44.0%, current-block K/V by 77.8%, and peak allocation by about 31.5%.

The result does not show that fixed-budget Blueprint is impossible. The source
variance ratio (`0.12465`) is dramatically outside Phase 13's successful
regime. The one justified next experiment is a constant-4K source-input
restriction discriminator using one analytically defined variance-preserving
or richer coarse representation—without growing the token budget.

**FIXED DIRECT 4096 RETURNS TO FRAGMENTED SCENE ORGANIZATION**
