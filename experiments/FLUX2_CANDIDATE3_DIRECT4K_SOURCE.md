# Phase 13 — direct-4K globally interacting source

Date: 2026-09-03

## Question

Does useful 4K global context require full-H interaction before compression,
or can global interaction occur directly at 4K?

This is an architecture discriminator. It changes neither production nor
ComfyUI core and does not qualify performance or generalization.

## Fixed experiment

All variants preserve the Phase-12c FLUX.2 Klein 4B bridge/train trajectory:
seed `20260901`, CFG 1, four-step CONST-flow Euler, accepted `H=64×128`, fixed
Candidate-3 `G=24×48`, 15 destination crops (`32×32`, stride 24), reconstructed
sigma-consistent native-coordinate `64×64` W, unchanged W restriction/assembly,
hard nonterminal coupling, and terminal release. Context is fresh from the
current accepted state and sigma and is consumed by all 25 blocks and 15 crops.

| Variant | Source interaction | Consumer context |
|---|---|---:|
| A_POST_INTERACTION_4096 | full 8,192-token H, then vertical 2×1 K/V aggregation | 4,096 |
| B_DIRECT_INTERACTION_4096 | vertical 2×1 accepted-H restriction, then ordinary 4,096-token interaction | 4,096 |
| C_FULL_H_8192_ORACLE | ordinary full 8,192-token H | 8,192 |

For B only, a separate full-H/post-interaction reference is captured from the
same accepted H and sigma before the candidate source call. It exists solely to
measure per-block K/V divergence. Local consumers use only direct-source K/V.
Its time is reported separately and is not part of candidate source time.

## Direct source construction

The ephemeral direct source is:

```text
S[b,c,r,x] = mean(H[b,c,2r:2r+2,x])
```

It has shape `1×128×32×128`. This exact linear restriction partitions all 8,192
H positions into 4,096 pairs with zero omissions or duplicates. It is rebuilt
once from current accepted H at each sigma and is never accepted or persistent.

Direct source row `r` represents full-H rows `(2r,2r+1)` at `y=2r+0.5`; x
remains native `0..127`. Live RoPE options are `scale_y=2.0` and `shift_y=0.5`.
The machine report contains all 4,096 direct coordinates and contributors.

Pair averaging changes variance materially. Direct/H variance ratios are
`0.501314, 0.501755, 0.505403, 0.531643`. No variance normalization is added:
a scalar correction would rescale both signal and noise without an independently
observable decomposition, while the unnormalized arm passes the semantic gate.

The direct source executes ordinary dense image-image interaction through all
25 FLUX blocks and exposes every resulting generated K/V position through the
unchanged local generated-query mechanism.

## Integrity

- A and C reproduce their Phase-12c hashes exactly (`356396cd...102b4` and
  `a62b006a...0cf7e`).
- B uses 4,096 source/consumer positions and is rebuilt four times.
- Every H position contributes exactly once.
- Every interval captures 25 blocks and 375 context consumptions.
- The K/V diagnostic reference shares B's accepted-H hash and sigma.
- Fresh-source, no-stale-context, state immutability, finite output, complete
  coverage, and Candidate-3 lifecycle checks pass.

## Semantic and numerical result

The panel is `flux2_candidate3_direct4k_source_results/FINAL_COMPARISON.png`.

B remains in the coherent A/C semantic class: one dominant continuous
bridge/train/deck system, controlled end towers/supports, and continuous
horizon/water. It does not revert to earlier compact-context fragmentation.

### Overlap RMS

| Variant | i0 | i1 | i2 | terminal i3 |
|---|---:|---:|---:|---:|
| A post-interaction 4K | 0.829927 | 0.626119 | 0.459728 | 0.246554 |
| B direct 4K | 0.832193 | 0.612545 | 0.465343 | 0.251294 |
| C full-H oracle | 0.824706 | 0.636335 | 0.471722 | 0.266542 |

### Assembled-x0 RMS

| Comparison | i0 | i1 | i2 | i3 |
|---|---:|---:|---:|---:|
| B versus A | 0.290845 | 0.356682 | 0.270959 | 0.211372 |
| B versus C | 0.369932 | 0.327505 | 0.406851 | 0.332610 |
| A versus C | 0.349765 | 0.491778 | 0.500892 | 0.407488 |

B is closer to the oracle than A at intervals 1–3 and comparable at interval
0. Semantic whole-scene organization, rather than numerical identity, is the
primary gate.

## Source representation divergence

Direct interaction does not numerically reproduce pooled post-interaction K/V:

| Interval | K RMS | K cosine | K norm ratio | V RMS | V cosine | V norm ratio |
|---|---:|---:|---:|---:|---:|---:|
| i0 | 0.3903 | 0.9649 | 1.0327 | 0.4785 | 0.9061 | 1.0810 |
| i1 | 0.4021 | 0.9629 | 1.0367 | 0.5971 | 0.8904 | 1.0712 |
| i2 | 0.4460 | 0.9541 | 1.0414 | 0.4680 | 0.8617 | 1.0638 |
| i3 | 0.6066 | 0.9131 | 1.0599 | 0.4825 | 0.7595 | 1.0878 |

The fields diverge increasingly at late sigma, especially V, but direct context
retains enough information for the same semantic scene organization. Full
per-block metrics are in `report.json`.

## Cost

| Variant | Source CUDA | Local CUDA | Wall | Peak alloc/reserved | Cache/interval | Transfer/run |
|---|---:|---:|---:|---:|---:|---:|
| A post 4K | 12.785 s | 120.501 s | 135.599 s | 3.665/4.373 GiB | 1.172 GiB | 70.312 GiB |
| B direct 4K | 23.826 s | 121.330 s | 161.082 s | 3.665/4.373 GiB | 1.172 GiB | 70.312 GiB |
| C full 8K | 13.066 s | 153.845 s | 169.065 s | 3.665/4.373 GiB | 2.344 GiB | 140.625 GiB |

B wall time includes `13.234 s` of extra full-H diagnostic capture. Removing
that instrumentation gives roughly `147.85 s`, still slower than A. The direct
`32×128` source itself measured `5.90–6.08 s` per interval versus about `3.2 s`
for full-H source execution. Current backend shape behavior therefore prevents
an acceleration claim. Peak memory is also confounded by the diagnostic full-H
capture.

## Answer and decision

**Useful 4K global context does not require full-H interaction before
compression in this discriminator. Global interaction can occur directly at
4K.**

This is a major positive architectural result for normalized local working
canvases. It is scene- and anisotropy-specific, not a universal 4K threshold.
Do not productionize it. Next research should make direct-source geometry
destination-independent and diagnose the unfavorable current backend timing
before asserting practical savings.
