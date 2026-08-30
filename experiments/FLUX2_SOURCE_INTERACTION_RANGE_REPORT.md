# FLUX.2 source interaction range discriminator

## Verdict

**WINDOWED SOURCE INTERACTION INSUFFICIENT.**

Fixed 16-by-16 source windows retained 12.5% of ordinary dense image-image
connectivity and improved numerical behavior only marginally over the no-image-
image control. The duplicate lighthouse returned at approximately the same
severity, together with the dark stone structure and associated train/bridge
regressions.

The result indicates that independent 256-token spatial neighborhoods do not
propagate enough cross-canvas information to preserve the successful dense-
source semantic constraint. It does not establish the exact range or topology
that would be sufficient.

## Reproduced boundary

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  y=0..31, x=24..55
diagnostic updates:    0
```

The Phase-2j state reproduced exactly and the diagnostic returned the accepted
latent unchanged (`max_abs=0`). All three variants retained identical source
tokens, native coordinates, conditioning, local execution, 2048-token external
K/V, all-25-block local integration, and final projection.

## Windowed source construction

The native `32 x 64` source image-token grid was partitioned into eight fixed,
nonoverlapping `16 x 16` windows:

```text
source image tokens:              2048
windows:                          2 x 4 = 8
image tokens per window:          256
allowed image-image edges:        8 * 256 * 256 = 524,288
ordinary dense image-image edges: 2048 * 2048 = 4,194,304
retained connectivity:            12.5%
```

In every five double-stream and twenty single-stream source blocks:

```text
text query -> text key:          allowed
text query -> every image key:   allowed
image query -> every text key:   allowed
image query -> same-window image key: allowed
image query -> other-window image key: blocked
```

No shifted windows, dilation, summaries, landmarks, or cross-window tokens were
used. Source Q-by-K tensor dimensions remained `2560 x 2560`; this mask is a
connectivity ablation, not evidence of realized sparse-kernel acceleration.

The local external attention remained `1536 x 3584` for all variants.

## Quantitative result

| Source interaction | Source image tokens | Source Q x K | Local Q x K | RMS vs dense | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary dense | 2048 | 2560 x 2560 | 1536 x 3584 | 0.179977 | 0.098817 | 0.846498 |
| 16x16 windowed | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.248911 | 0.132296 | 0.835039 |
| No image-image | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.252297 | 0.133993 | 0.825402 |

Windowed source interaction reduced RMS by only 1.34% and low-frequency RMS by
1.27% relative to no image-image interaction. It remained 38.3% worse in RMS
and 33.9% worse in low-frequency RMS than ordinary dense-source context.

Prediction norms remained stable. The windowed full-source prediction was
somewhat closer to ordinary dense than the no-mixing source prediction
(`0.231` versus `0.243` RMS), but that source-level improvement did not preserve
the target local semantic behavior.

## Per-block K/V control

Both ablations captured exactly the ordinary block-0 K/V before any restricted
attention could alter the hidden trajectory:

```text
double block 0 K RMS difference: 0
double block 0 V RMS difference: 0
```

At double block 1, windowed K/V diverged less from ordinary than no-mixing K/V:

```text
windowed:       K RMS 0.19, V RMS 0.51
no image-image: K RMS 0.37, V RMS 1.03
```

The report records K/V divergence for all 25 blocks. This confirms that the
window mask preserves meaningful within-window hidden interaction, even though
that interaction is insufficient for the lighthouse constraint.

## Decoded semantic observations

- **Duplicate lighthouse:** suppressed by ordinary dense source; returns under
  both windowed and no-image-image source interaction.
- **Dark stone structure:** remains prominent under both restricted variants,
  forming the same false object pair with the lighthouse.
- **Train:** windowed carriage extent and segmentation regress toward the
  no-image-image control rather than the ordinary dense result.
- **Bridge:** deck continuity survives, but train/cable alignment remains far
  from ordinary dense.
- **Horizon/water:** the false lighthouse/tower pair reappears on the horizon in
  both restricted variants.
- **Source image:** windowed source execution stays globally interpretable and
  numerically improves over no mixing, but source-image plausibility does not
  predict successful local K/V semantics.

## Artifacts

- `flux2_candidate2_source_interaction_range.py`
- `flux2_candidate2_source_interaction_range_results/report.json`
- decoded local predictions and dense-surrounding composites
- local error maps
- windowed and no-image-image full-source estimates

## Interpretation

Phase 2k rules out one materially reduced interaction topology: isolated 16x16
windows across all source blocks. Retaining local image mixing over meaningful
256-token neighborhoods is not enough; information must propagate across
window boundaries or over larger spatial distances for this semantic constraint.

No window-size sweep is justified or performed here. The experiment does not
separate a need for direct long-range pairwise attention from a need for gradual
cross-window propagation across blocks. That distinction requires a different,
explicitly authorized falsifier rather than another density/layout sweep.

**WINDOWED SOURCE INTERACTION INSUFFICIENT**
