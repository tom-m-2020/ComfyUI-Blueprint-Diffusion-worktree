# Phase 16 — fixed-4096 representation-richness discriminator

Date: 2026-09-04

## Question

Can a richer deterministic restriction of every `4×2` accepted-H cell retain
enough whole-scene information for the fixed `32×128 = 4,096` source to recover
one coherent bridge/train scene?

The experiment reuses Phase 14/15 exactly: accepted `H=128×256`, accepted G,
terminal sigma, conditioning, 55 reconstructed native-coordinate `W=64×64`
consumers, full-canvas source coordinates, all 25 globally interacting source
blocks, block-major context consumption, assembly, and zero-update lifecycle.
Production and ComfyUI core are unchanged.

## Fixed representation

Each channel's eight samples in a `4×2` cell are transformed by a separable
orthonormal DCT-II. C3 retains four spatial modes:

```text
(0,0) DC
(1,0) first vertical low-frequency mode
(0,1) first horizontal low-frequency mode
(1,1) first diagonal mode
```

Four modes across 128 latent channels produce 512 coefficients per spatial
position, but FLUX accepts only 128 channels. C3 therefore uses a fixed
normalized `128×128` Sylvester-Hadamard channel basis. Its transformed channel
space is partitioned into four disjoint 32-dimensional subspaces, one per mode,
then inverse-transformed into the existing 128-channel source interface.

This is an explicit orthogonal projection, not lossless packing:

- 32 channel components are retained for each of four modes;
- 96 channel components per retained mode are discarded;
- all other four DCT spatial modes are discarded;
- 128 of the candidate 512 coefficients remain (`25%`).

No parameters are learned or tuned. DCT and Hadamard orthogonality max errors
are `2.38e-7` and `5.96e-8`. Spatial-token count and complete `4×2` contributor
provenance remain unchanged.

## Reconstruction diagnostic

The inverse uses the adjoint channel projection, restores the four retained DCT
modes, zero-fills missing coefficients, and applies inverse DCT.

| Diagnostic | C0 area mean | C3 orthogonal modes |
|---|---:|---:|
| H reconstruction RMS | 0.931140 | 0.930890 |
| low-frequency 8×8-mean RMS | 0.00000004 | 0.107615 |
| edge-gradient RMS | 1.374486 | 1.340725 |
| reconstruction/H variance ratio | 0.124651 | 0.125121 |
| reconstruction/H RMS ratio | 0.353060 | 0.353725 |

C3 slightly improves total and edge reconstruction but damages C0's exact
per-channel block-DC/large-scale mean preservation because only one quarter of
the orthogonal channel components carry DC. It is richer in directional modes,
not uniformly richer for every latent channel.

## Source-model telemetry

The packed source itself has RMS `0.995716` (near accepted-H RMS), unlike C0's
`0.351376`, because retained orthonormal coefficients are not divided by the
cell size.

| Boundary | C0 hidden/input | C3 hidden/input | C0 K/V | C3 K/V |
|---|---:|---:|---:|---:|
| source input RMS | 0.351376 | 0.995721 | — | — |
| after img_in RMS | 0.185482 | 0.525554 | — | — |
| double 0 | 2.078963 | 1.900361 | 1.184487 / 7.383815 | 1.184619 / 7.415339 |
| single 9 | 64.900436 | 64.932266 | 1.557853 / 0.370688 | 1.557624 / 0.415770 |
| single 19 | 94.847000 | 94.553230 | 1.494132 / 0.915891 | 1.485474 / 0.976381 |

As in Phase 15, normalization makes early K scale nearly invariant, although
mode content changes later V values.

## Semantic result

The direct comparison is
`flux2_candidate3_fixed4k_representation_richness_results/C0_C3_comparison.png`.

C3 changes which bridge fragments dominate and reduces overlap disagreement,
but it remains in C0's fragmented semantic class:

- many independent bridge spans and suspension alternatives;
- repeated train-like streaks and supports;
- uncontrolled tower count;
- discontinuous decks/cables;
- multiple horizon/water bands.

It does not approach one dominant whole-canvas bridge/train system.

| Metric | C0 | C3 |
|---|---:|---:|
| terminal overlap RMS | 0.784053 | 0.742034 |
| assembled prediction RMS | 0.694729 | 0.699411 |

The lower overlap RMS is a secondary compatibility change, not semantic
success.

## Runtime and integrity

C3 source CUDA is `1.054 s`, local CUDA `102.168 s`, and terminal wall time
`104.611 s`. Peak allocated/reserved memory is `3.79/5.84 GiB`; current-block
source K/V remains `48 MiB`. The fixed-4K work and memory contract is unchanged.

Accepted H/G and all W hashes match Phase 15. The source has exactly 4,096
positions, 25 source blocks, and 1,375 context consumptions. Output is finite,
coverage complete, and no state update occurs.

## Decision

**C3 remains in the same fragmented semantic class. Simple deterministic local-
mode packing is insufficient.** It also illustrates a hard interface problem:
packing additional spatial modes into a fixed trained 128-channel latent input
necessarily discards channel components or changes their learned semantics.

The next question is no longer scalar source statistics or another simple mode
packing. It is whether 4,096 independent source positions are intrinsically
insufficient with normalized-W consumers, or whether the local context-
consumer interface is the dominant failure.
