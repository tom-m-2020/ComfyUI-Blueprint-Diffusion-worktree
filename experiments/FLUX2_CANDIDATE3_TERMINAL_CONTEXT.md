# Phase 8d — terminal local/global context discriminator

## Result

Fresh current-G context inside all 25 FLUX.2 blocks solves the catastrophic
terminal local fragmentation at the fixed 2048x4096 state. Context-aware
terminal x0_H itself becomes one usable bridge scene before assembly and is
released directly with no output-space projection.

The result is semantic evidence for terminal local/global coupling, not an
efficient implementation. Current G contributes 18,432 K/V tokens per block;
the native GPU-resident Candidate-2 cache path OOMed during source capture on
the 12 GB device. The completed discriminator retained the exact K/V tensors
on CPU and transferred one block's unchanged K/V back for each consuming local
block.

## Fixed configuration and controls

- Output 2048x4096; H=128x256; production 4→3 DCT G=96x192.
- FLUX.2 Klein, bridge/train prompt, seed 20260901, CFG 1.
- Production four-step schedule and intervals 0–2 executed once.
- Production 55 crops, 32x32, stride 24, unchanged coordinates and weights.
- A and B share bit-identical H0/G0, all nonterminal accepted states, H3/G3,
  terminal sigma, crop inputs, crop rectangles, and crop order.
- B uses one fresh same-sigma G3 source forward. The same captured K/V objects
  are consumed by all 55 crops in all 25 blocks.
- Neither A nor B applies output-space projection.

A's final RGB hash is
`e4f624ea0160a5193bea0266942367e7c781896f20fb90dee0a4372bc17c2fbc`,
exactly reproducing Phases 8a–8c.

## Context integration

The previously qualified Candidate-2 path is unchanged semantically:

1. run the ordinary globally interacting G3 model at terminal sigma;
2. capture generated K/V after RoPE at all 5 double-stream and 20
   single-stream blocks;
3. preserve ordinary text/local queries and local K/V;
4. append positioned current-G generated K/V;
5. restore ordinary text-query attention output, so only generated local
   queries consume the added context.

Each local block has 1,536 total queries and 19,968 keys/values:

```text
512 text + 1024 local generated + 18432 current-G generated K/V
```

The GPU-resident prior implementation OOMed while accumulating the 25-block
source cache. The experiment-only storage adaptation retained 5,662,310,400
bytes (5.27 GiB) of identical K/V on CPU, then transferred one block at a time.
Across 55 crops this moved 311,427,072,000 bytes (290.06 GiB). This changes
storage/transfer scheduling only, not K/V values, positions, or integration.

The optional dense-H context diagnostic was not run. Compact current-G already
answers the primary causal question positively; dense H would require a
32,768-token all-block cache and 55 still-larger attention calls and would not
change this verdict.

## Semantic localization

### A — production terminal local-only

The 55 ordinary crop predictions disagree and assemble into many independent
bridge systems, broken decks, floating spans, repeated towers, and repeated
train/lighthouse-like structures. Pairwise overlap RMS is 1.000208.

### B — fresh current-G context

The context changes every crop materially (per-crop RMS difference from A
ranges 0.6913–1.0909, mean 0.9121). Before assembly, neighboring predictions
become far more mutually compatible: overlap RMS falls to 0.281238, a 71.9%
reduction.

The assembled terminal x0_H and directly released final H show one dominant
suspension bridge system, a continuous deck and cable system, one continuous
yellow train, controlled large-tower count, and coherent horizon/water. Fine
cable and deck detail remains. A few small floating red fragments and imperfect
far-right continuation remain, but the catastrophic field of independently
invented bridges is gone. No hard projection is applied.

The first demonstrated reduction in local semantic fragmentation is therefore
the set of context-conditioned crop predictions themselves, before overlap
assembly or sampler acceptance.

## Projection diagnostic

The same fresh global prediction supplies terminal G_star for measurement
only:

| Variant | Required projection RMS | Projection/H_star | D(H)-G_star RMS |
|---|---:|---:|---:|
| A local-only | 0.867312 | 102.51% | 0.867312 |
| B current-G context | 0.371625 | 52.35% | 0.371625 |

Context reduces required projection RMS by 57.2% and the relative projection
ratio by 49.0 percentage points. It does not make H numerically coarse-exact,
but it converts the output from globally unusable to a coherent scene without
applying the remaining diagnostic projection.

A and B terminal/final latent RMS difference is 0.849363 with max absolute
7.0493, confirming that context changes scene-scale local predictions rather
than only smoothing seams.

## Runtime and memory

| Work | CUDA time |
|---|---:|
| Shared nonterminal global | 28.91 s |
| Shared nonterminal local | 56.61 s |
| Fresh terminal G capture | 11.90 s |
| A terminal ordinary crops | 19.05 s |
| B terminal context crops | 98.34 s |

Shared wall time containing both A and B terminal paths was 217.02 s. A
production-shaped context path would retain the shared nonterminal work, one
11.90 s G capture, and 98.34 s context-local terminal work; the present CPU
transfer implementation is much slower than production A and is not an
optimization candidate.

Baseline CUDA allocation/reservation was 2.32/2.38 GiB. Peak was 3.73 GiB
allocated and 6.70 GiB reserved. Relative to the Phase-8c pre-terminal-global
peak, the CPU-offloaded context path adds roughly 0.84 GiB allocated and
0.47 GiB reserved, plus the 5.27 GiB CPU K/V cache. The initial all-GPU cache
attempt OOMed before any context crop and is recorded as a feasibility failure,
not a semantic result.

## Outputs

- `TERMINAL_X0_H_COMPARISON.png`
- `FINAL_OUTPUT_COMPARISON.png`
- per-variant decoded PNGs and machine-readable `report.json`

## Interpretation

Candidate-3's terminal local failure is not intrinsic to crop size or overlap
alone. The same local queries, inputs, coordinates, and assembly become
globally coherent when they consume fresh whole-canvas hidden context inside
the transformer. This avoids the scene-scale output reconstruction required by
Phase 8c hard projection and preserves substantially better local detail.

The demonstrated mechanism is currently expensive and memory-transfer heavy.
It does not revive Candidate 2 as the global architecture: Candidate-3 still
owns G/H state and lifecycle. It establishes only that terminal Candidate-3
local execution needs access to fresh global semantic context at this scale.

## Verdict

TERMINAL GLOBAL CONTEXT SOLVES LOCAL FRAGMENTATION
