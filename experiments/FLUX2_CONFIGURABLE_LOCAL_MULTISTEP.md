# Configurable Blueprint local multi-step qualification

## Verdict

**REJECT — short multi-step W refinement does not improve the credible-detail
frontier.** Do not add a local-step control to `Blueprint Configurable
Prototype`.

## Fixed discriminator

The experiment held seed, prompt, terminal G, `G=45x45`, `H=128x128`,
`F=32x32`, stride `16x16`, `W=64x64`, regional noise, transfer, assembly, and
sigma start `0.25` constant. Scheduler-derived pre-shift subdivisions produced:

```text
A: [0.25, 0]
B: [0.25, 0.14082231971176423, 0]
C: [0.25, 0.1801615626963453, 0.09801717049814734, 0]
```

Each region's W was constructed once. The same accepted W trajectory then
continued through the arm's intervals without reconstruction or re-noising.

## Isolation and regression

Across all 49 regions, B and C match A exactly for regional noise hash, initial
W hash, first W-state hash, and first pre-update model-prediction hash. Thus the
first divergence is the accepted target after the shared sigma-0.25 prediction.

The one-step arm reproduces the qualified configurable control bit-exactly:

`09ee4f8be0f88bb577a0a142ea9c5a298845019be552b09b380fd57c3acb75b4`

## Measurements

| Arm | W calls | Wall time | Overlap RMS | Gradient RMS | Laplacian RMS | Low-frequency RMS vs A |
|---|---:|---:|---:|---:|---:|---:|
| A, one interval | 49 | 50.13 s | 0.173297 | 0.221675 | 0.477588 | 0 |
| B, two intervals | 98 | 97.74 s | 0.184967 | 0.229360 | 0.505645 | 0.010753 |
| C, three intervals | 147 | 148.58 s | 0.189986 | 0.232334 | 0.517234 | 0.015176 |

All arms have complete normalized coverage, zero destination-sized model calls,
fixed `64x64` W geometry, and zero completed-region allocation range. CUDA
reserved memory is identical at `3,512,729,600` bytes. Peak allocated memory is
`3,025,436,160` bytes for A and `3,028,057,600` bytes for B/C, a roughly 2.6 MB
difference independent of interval count after the second call is introduced.

## Semantic review

All arms retain S3: exactly one car, one central tree, and one house on a shared
field under one continuous horizon, without miniature-scene repetition. B and C
raise gradient and Laplacian energy, but visual inspection does not find clearer
wheels or car contour, organized foliage, improved house geometry, or credible
ground structure. The extra energy reads mainly as denser repeated fine edging.
Overlap disagreement worsens monotonically.

The stop rule therefore applies. No further local-depth or sigma sweep is
justified, and no production control is promoted.
