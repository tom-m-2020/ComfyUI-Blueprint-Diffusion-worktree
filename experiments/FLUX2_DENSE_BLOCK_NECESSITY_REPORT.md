# FLUX.2 dense-block necessity discriminator

## Verdict

**PARTIAL DENSE DEPTH INSUFFICIENT.**

Five early dense source blocks recover materially more numerical and geometric
benefit than five late dense blocks, but neither subset preserves the successful
all-dense object-uniqueness constraint. The duplicate lighthouse remains in
both partial-depth variants.

This establishes an early-depth asymmetry without showing that the first five
blocks are sufficient. Repeated broad interaction across more than the tested
20% depth budget remains necessary in this case.

## Exact diagnostic boundary

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  y=0..31, x=24..55
diagnostic updates:    0
```

The Phase-2l state reproduced exactly and the diagnostic returned the accepted
latent unchanged (`max_abs=0`). Every variant retained all 2048 source image
tokens, native full-canvas coordinates, identical source input/conditioning,
the same center crop, a 2048-token external K/V interface, all-25-block local
integration, and final projection.

## Fixed block ranges

The actual source execution order is:

```text
ordinals 0..4:   double blocks 0..4
ordinals 5..24:  single blocks 0..19
```

Variants used:

```text
A all dense:
  ordinals 0..24

B early dense:
  ordinals 0..4 = double blocks 0..4
  ordinals 5..24 use no image-image interaction

C late dense:
  ordinals 0..19 use no image-image interaction
  ordinals 20..24 = single blocks 15..19

D no image-image:
  no dense source blocks
```

Each dense block permits all `2048 x 2048 = 4,194,304` image-image edges.

| Variant | Dense blocks | Total dense image-image edges | Fraction of all-dense edge budget |
|---|---:|---:|---:|
| All dense | 25 | 104,857,600 | 100% |
| Early dense | 5 | 20,971,520 | 20% |
| Late dense | 5 | 20,971,520 | 20% |
| No image-image | 0 | 0 | 0% |

Text query/key connectivity and image-query access to all text keys remained
ordinary in restricted blocks. The mask is a causal ablation, not a sparse-
kernel performance result.

Source attention tensor dimensions remained `2560 x 2560`; local external
attention remained `1536 x 3584`.

## Quantitative result

| Source dense depth | RMS vs dense crop | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|
| All 25 blocks | 0.179977 | 0.098817 | 0.846498 |
| Early ordinals 0–4 | 0.227273 | 0.124224 | 0.828532 |
| Late ordinals 20–24 | 0.247148 | 0.126885 | 0.835192 |
| No image-image | 0.252297 | 0.133993 | 0.825402 |

Relative to no image-image mixing:

- early dense improves RMS by 9.9% and low-frequency RMS by 7.3%;
- late dense improves RMS by 2.0% and low-frequency RMS by 5.3%.

Early dense closes about 34.6% of the total-RMS gap between no mixing and all
dense; late dense closes only about 7.1%. Prediction scale remains stable.

The full-source estimates show the same ordering: RMS from all-dense source is
0.207 early, 0.230 late, and 0.243 no-mixing.

## Causal K/V checks

The early-dense hidden trajectory tracks ordinary all-dense K/V through capture
ordinal 5. This includes the capture entering single block 0, because dense
double block 4 has already executed but restriction has not yet affected a
result. It first diverges at capture ordinal 6:

```text
ordinal 5 early-vs-all-dense K/V RMS: 0 / 0
ordinal 6 early-vs-all-dense K/V RMS: 0.27 / 0.10
```

The late-dense hidden trajectory tracks the no-image-image control through
capture ordinal 20, before the first late dense attention result. It first
diverges at capture ordinal 21:

```text
ordinal 20 late-vs-restricted K/V RMS: 0 / 0
ordinal 21 late-vs-restricted K/V RMS: 0.16 / 0.13
```

Later dense blocks begin changing the restricted trajectory and reduce some K
distance to ordinary dense, but final K/V remains substantially restricted-like.
The report contains per-block K/V differences from all-dense and the two causal
tracking comparisons.

## Decoded semantic observations

- **Duplicate lighthouse:** suppressed only by all-dense source context. It
  remains in early-dense, late-dense, and no-mixing variants.
- **Dark stone structure:** remains prominent beside the duplicate in both
  partial variants.
- **Train:** early dense materially improves carriage organization relative to
  no mixing; late dense provides only modest recovery. Neither matches all dense.
- **Bridge:** early dense improves deck/cable organization but does not restore
  the full semantic constraint. Late dense remains restricted-like.
- **Horizon/water:** the false lighthouse/tower pair remains under both partial
  depth policies.

## Artifacts

- `flux2_candidate2_dense_block_necessity.py`
- `flux2_candidate2_dense_block_necessity_results/report.json`
- decoded local predictions and dense-surrounding composites
- prediction-error maps
- early, late, and no-mixing full-source estimates

## Interpretation

Broad interaction in the five double blocks is more valuable than the same
edge budget in the final five single blocks, supporting an early source-depth
role in organizing geometry. However, that early plan is not preserved well
enough through twenty subsequently restricted blocks to maintain object
uniqueness at the external K/V interface.

The result does not prove that every one of 25 blocks is necessary. It shows
that neither tested five-block contiguous subset is sufficient, and it does not
authorize a block-count or placement sweep.

**PARTIAL DENSE DEPTH INSUFFICIENT**
