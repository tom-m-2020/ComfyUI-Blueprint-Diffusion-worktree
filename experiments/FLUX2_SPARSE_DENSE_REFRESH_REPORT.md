# FLUX.2 sparse dense-refresh maintenance probe

## Verdict

**PERIODIC DENSE REFRESH INSUFFICIENT.**

Adding dense source refreshes at ordinals 10, 15, and 20 after the five early
dense blocks did not preserve the all-dense semantic constraint. The duplicate
lighthouse remained, and final total and low-frequency error were slightly
worse than early-only despite using 60% more dense blocks.

Individual later refreshes briefly moved some K/V closer to all-dense, but that
improvement did not survive the intervening no-image-image blocks. This fixed
schedule does not provide a credible cheap maintenance contract.

## Exact state and controls

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  y=0..31, x=24..55
diagnostic updates:    0
```

The Phase-2m state reproduced exactly and the diagnostic returned the accepted
latent unchanged (`max_abs=0`). Every variant retained all 2048 source image
tokens, native coordinates, identical source input and conditioning, the same
center crop, 2048 external K/V, all-25-block local integration, and final
projection. Restricted blocks reused the exact Phase-2j no-image-image mask.

## Fixed schedule and edge budget

```text
A all dense:
  ordinals 0..24

B early-only:
  ordinals 0,1,2,3,4

C early + periodic refresh:
  ordinals 0,1,2,3,4,10,15,20

D no image-image:
  no dense ordinals
```

Each dense block exposes all `2048 x 2048 = 4,194,304` image-image edges.

| Variant | Dense blocks | Total dense image-image edges | Fraction of all-dense budget |
|---|---:|---:|---:|
| All dense | 25 | 104,857,600 | 100% |
| Early-only | 5 | 20,971,520 | 20% |
| Early + refresh | 8 | 33,554,432 | 32% |
| No image-image | 0 | 0 | 0% |

Source attention tensor dimensions remained `2560 x 2560`; local external
attention remained `1536 x 3584`. These are masked dense operations and do not
establish a realized sparse-kernel speedup.

## Quantitative result

| Source schedule | RMS vs dense crop | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|
| All dense | 0.179977 | 0.098817 | 0.846498 |
| Early-only | 0.227273 | 0.124224 | 0.828532 |
| Early + refresh | 0.231562 | 0.125796 | 0.831404 |
| No image-image | 0.252297 | 0.133993 | 0.825402 |

Relative to early-only, periodic refresh increased RMS by 1.89% and low-
frequency RMS by 1.27%. It remains better than no mixing because it retains the
same useful early dense blocks, not because the refreshes improve the final
result.

The full-source prediction follows the same ordering: RMS from all-dense source
is 0.207 early-only, 0.213 refresh, and 0.243 no-mixing.

## Refresh-by-refresh K/V behavior

K/V distance was recorded from all-dense and early-only at every block capture.
A refresh affects the following capture because K/V is captured before the
current block's attention result.

### Refresh ordinal 10

The refresh and early-only trajectories are identical entering ordinal 10. At
capture 11, the refresh makes both K and V slightly farther from all-dense:

```text
K RMS change versus early-only baseline: -0.0045
V RMS change versus early-only baseline: -0.0025
```

Negative values mean divergence increased. The regression remains at capture
15 before the next refresh.

### Refresh ordinal 15

Prior refresh history means this trajectory already differs from early-only at
capture 15. The ordinal-15 dense block briefly reduces divergence at capture 16:

```text
K RMS reduction versus early-only: 0.0149
V RMS reduction versus early-only: 0.0007
```

By capture 20, before the next refresh, those gains are gone and slightly
reversed (`-0.0023 K`, `-0.0038 V`).

### Refresh ordinal 20

The final refresh again produces a short-lived improvement at capture 21:

```text
K RMS reduction versus early-only: 0.0198
V RMS reduction versus early-only: 0.0039
```

By final capture 24, it has again reversed (`-0.0023 K`, `-0.0054 V`).

Thus dense refreshes can perturb a restricted trajectory toward all-dense
locally, but the tested blocked tail does not maintain the corrected hidden
state. The report records complete per-block K/V distances and refresh effects.

## Decoded semantic observations

- **Duplicate lighthouse:** suppressed only by all-dense context. It remains at
  early-only severity under periodic refresh.
- **Dark stone structure:** remains prominent beside the duplicate.
- **Train:** refresh produces no material final improvement over early-only.
- **Bridge/cables:** organization remains comparable to early-only rather than
  moving toward all-dense.
- **Horizon/water:** the false lighthouse/tower pair remains after all three
  refreshes.

## Artifacts

- `flux2_candidate2_sparse_dense_refresh.py`
- `flux2_candidate2_sparse_dense_refresh_results/report.json`
- decoded local predictions and dense-surrounding composites
- prediction-error maps
- early-only, refresh, and no-mixing full-source estimates

## Interpretation

An early plan plus three isolated refresh blocks is not self-maintaining under
the no-image-image transition used here. The issue is not simply absence of any
later dense event: refreshes briefly improve some features, but intervening
restricted blocks dissipate or redirect the benefit before final K/V capture.

This weakens Candidate 2 as an efficiency architecture under hard source-
interaction blocking. It does not prove that continuous all-dense attention is
the only solution, but no schedule sweep is justified or performed.

**PERIODIC DENSE REFRESH INSUFFICIENT**
