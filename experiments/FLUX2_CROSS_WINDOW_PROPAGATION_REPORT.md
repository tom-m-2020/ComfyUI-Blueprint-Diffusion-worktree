# FLUX.2 cross-window propagation discriminator

## Verdict

**CROSS-WINDOW PROPAGATION INSUFFICIENT.**

Alternating unshifted and shifted 16-by-16 source windows made every image token
analytically reachable from the entire canvas across transformer depth, but did
not preserve the successful dense-source semantic result. The duplicate
lighthouse remained, and numerical agreement was worse than fixed windows and
the no-image-image control.

This shows that transitive graph reachability alone is not a sufficient source-
context contract. It does not prove that every source block needs direct dense
all-to-all attention; it falsifies this one gradual propagation topology.

## Exact state and controls

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  y=0..31, x=24..55
diagnostic updates:    0
```

The Phase-2k state reproduced exactly and the diagnostic returned the accepted
latent unchanged (`max_abs=0`). All four variants retained 2048 source image
tokens, native full-canvas coordinates, identical conditioning, source input,
local crop execution, 2048-token external K/V, all-25-block local integration,
and final projection.

## Alternating partition

The actual source execution order was treated as one sequence:

```text
ordinals 0..4:   double blocks 0..4
ordinals 5..24:  single blocks 0..19
```

Even ordinals used the Phase-2k unshifted `16 x 16` partition. Odd ordinals
used a non-wrapping partition shifted by `(8,8)`:

```text
window bin = floor((coordinate + shift) / 16)
```

Canvas edges were clipped rather than wrapped. Every token therefore belonged
to exactly one valid window, without introducing artificial left-right or
top-bottom adjacency.

Per unshifted block:

```text
8 windows x 256 tokens
allowed image-image edges: 524,288
fraction of dense edges:    12.5%
```

Per shifted block:

```text
15 nonempty clipped windows
window sizes:               64..256 tokens
allowed image-image edges:  344,064
fraction of dense edges:    8.203125%
```

Text queries retained all text and image keys; image queries retained all text
keys. Only image-image connectivity followed the current partition. Source
attention tensors remained `2560 x 2560`, so this is a connectivity diagnostic,
not a measured sparse-kernel speedup.

Local external attention remained `1536 x 3584` in every variant.

## Analytical cross-depth reachability

Reachability was propagated exactly through the sequence of partition graphs:

| After ordinal | Partition | Minimum reachable tokens | Maximum reachable tokens | Tokens with full-canvas reachability |
|---:|---|---:|---:|---:|
| 0 | unshifted | 256 | 256 | 0 |
| 1 | shifted | 256 | 1024 | 0 |
| 2 | unshifted | 1024 | 1536 | 0 |
| 3 | shifted | 1024 | 2048 | 512 |
| 4 | unshifted | 1536 | 2048 | 1024 |
| 5 | shifted | 1536 | 2048 | 1536 |
| 24 | unshifted | 2048 | 2048 | 2048 |

Thus every final source token can transitively depend on every original image
position under this schedule. The semantic failure cannot be attributed merely
to a disconnected source-attention graph.

## Quantitative result

| Source interaction | Source tokens | Source Q x K | Local Q x K | RMS vs dense | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary dense | 2048 | 2560 x 2560 | 1536 x 3584 | 0.179977 | 0.098817 | 0.846498 |
| Fixed 16x16 | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.248911 | 0.132296 | 0.835039 |
| Alternating shifted | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.260950 | 0.137666 | 0.834740 |
| No image-image | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.252297 | 0.133993 | 0.825402 |

Alternating shifted windows were 4.8% worse in RMS than fixed windows and 3.4%
worse than no image-image mixing. They were 45.0% worse than ordinary dense
source context. Prediction scale remained finite and stable.

The shifted full-source prediction was slightly closer to ordinary dense than
the no-mixing source (`0.238` versus `0.243` RMS), but worse than the fixed-
window source (`0.231`). None of that source-level plausibility translated into
the required local semantic constraint.

## K/V control

The first two captures from fixed and alternating variants matched exactly:

```text
ordinal 0 / block 0: equal before any restricted result
ordinal 1 / block 1: equal because block 0 used the same unshifted partition
```

They first diverged at ordinal 2 capture, after ordinal 1 used shifted windows:

```text
ordinal 2 positioned-K RMS difference: 0.09
ordinal 2 V RMS difference:            0.32
```

This verifies that the new variable was the alternating partition after the
first differing source-attention operation. Per-block K/V differences from
ordinary dense and between fixed/shifted variants are recorded for all blocks.

## Decoded semantic observations

- **Duplicate lighthouse:** suppressed only by ordinary dense source; present
  under fixed, shifted, and no-image-image source paths.
- **Dark stone structure:** remains prominent under all restricted variants,
  paired with the false lighthouse.
- **Train:** shifted windows do not recover ordinary dense placement or carriage
  segmentation.
- **Bridge:** deck continuity remains, but train/cable geometry does not improve
  over fixed windows.
- **Horizon/water:** the false lighthouse/tower pair remains despite final
  full-canvas analytical reachability.

## Artifacts

- `flux2_candidate2_cross_window_propagation.py`
- `flux2_candidate2_cross_window_propagation_results/report.json`
- decoded local predictions and dense-surrounding composites
- prediction-error maps
- fixed, shifted, and no-mixing full-source estimates

## Interpretation

Gradual communication is not automatically equivalent to dense global
interaction. Although alternating windows make every token transitively
reachable, information is repeatedly compressed through local attention,
residual, and MLP updates under partitions that never expose broad evidence in
one operation. The dense source's advantage may depend on direct long-range
comparison, higher per-layer bandwidth, or a different structured propagation
mechanism—not connectivity alone.

No further window topology is proposed or run here. Candidate 2 still has
positive dense-context evidence, but this experiment does not establish a cheap
sufficient source-interaction contract.

**CROSS-WINDOW PROPAGATION INSUFFICIENT**
