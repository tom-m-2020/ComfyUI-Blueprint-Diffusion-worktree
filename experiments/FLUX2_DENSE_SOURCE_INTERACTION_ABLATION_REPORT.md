# FLUX.2 dense-source interaction ablation

## Verdict

**GLOBAL SOURCE INTERACTION REQUIRED.**

At the exact late Phase-2 state, retaining all 2048 native full-canvas source
positions but blocking image-query to image-key attention during source feature
construction caused the duplicate lighthouse to reappear and materially
degraded agreement with ordinary dense execution. The local crop still consumed
2048 external generated K/V through the unchanged all-25-block integration.

This establishes that final-interface token density alone does not explain the
successful dense-source result. Globally interacting source hidden trajectories
carry information needed for the observed lighthouse suppression.

The conclusion is scoped to this strong ablation. It does not prove that every
source block or every image-image edge is necessary.

## Reproduced boundary

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  y=0..31, x=24..55
diagnostic updates:    0
```

The Phase-2i accepted-state statistics reproduced exactly and the diagnostic
returned the latent unchanged (`max_abs=0`). Both variants used identical
conditioning, source input, native coordinates, source token count, local crop,
local RoPE, local projections/MLPs/residuals, all 25 external-K/V blocks, text
handling at the local integration point, and final projection.

## Source-attention intervention

Both source forwards retained the ordinary sequence:

```text
512 text tokens + 2048 image tokens = 2560 tokens
source Q x K: 2560 x 2560
```

Variant A used ordinary dense all-token attention.

For Variant B, a boolean attention mask was added in all five double-stream and
all twenty single-stream blocks:

```text
text query -> text key:   allowed, unchanged
text query -> image key:  allowed, unchanged
image query -> text key:  allowed, unchanged
image query -> image key: blocked
```

Per source block, 4,194,304 image-image edges were blocked. The remaining
2,359,296 text-involving edges were allowed. Input projections,
normalization/modulation, RoPE, timestep/text conditioning, attention output
projection, residual paths, MLPs, and final projection still executed.

“Text unchanged” refers to attention connectivity and query/key construction.
Text hidden values can diverge in later blocks because they continue attending
image values whose hidden trajectories were changed by the ablation.

The boolean mask does not prove compute savings: the backend may still execute
dense masked attention. This is a causal interaction diagnostic only.

## Capture and local integration control

K/V were captured at the same pre-attention point in each source block. The
block-0 ordinary and restricted captures were exactly identical:

```text
double block 0 positioned-K RMS difference: 0
double block 0 V RMS difference:            0
```

This is expected because no restricted source attention has yet influenced the
hidden trajectory. Captures diverged immediately afterward—for example:

```text
double block 1 positioned-K RMS difference: 0.365
double block 1 V RMS difference:            1.034
```

The report records K and V differences for all 25 blocks. Both local calls used:

```text
local queries:                 1536 = 512 text + 1024 local image
local K/V:                     3584 = 512 text + 1024 local + 2048 external
local external attention QxK: 1536 x 3584
```

Thus source hidden interaction—not external token count, coordinates, K/V
dimensionality, or local integration—was the isolated variable.

## Quantitative result

| Source | Source image tokens | Source Q x K | Local Q x K | RMS vs dense crop | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary dense | 2048 | 2560 x 2560 | 1536 x 3584 | 0.179977 | 0.098817 | 0.846498 |
| Restricted dense | 2048 | 2560 x 2560 masked | 1536 x 3584 | 0.252297 | 0.133993 | 0.825402 |

Removing source image-image mixing increased local RMS by 40.2% and
low-frequency RMS by 35.6%. Prediction scale remained finite and broadly
stable. The restricted full-canvas source prediction itself differed from the
ordinary source by 0.243 RMS.

## Decoded semantic observations

- **Duplicate lighthouse:** ordinary dense-source K/V suppresses the extra
  white lighthouse; restricted-source K/V makes it reappear beside the stone
  structure.
- **Stone structure:** the reduced dark tower remains under ordinary dense
  context; it becomes part of a clearer false lighthouse/tower pair under the
  restriction.
- **Train:** restricted source changes carriage extent and segmentation and
  moves away from the ordinary dense crop.
- **Bridge:** the deck remains continuous, but train and cable geometry regress.
- **Horizon/water:** restricted K/V restores the false center-right object pair
  on the horizon.
- **Restricted source output:** it remains globally interpretable, but its
  bridge/train geometry changes. Superficial whole-image coherence is therefore
  insufficient to preserve the successful external-K/V semantics.

## Artifacts

- `flux2_candidate2_dense_source_interaction_ablation.py`
- `flux2_candidate2_dense_source_interaction_ablation_results/report.json`
- decoded local predictions and dense-surrounding composites
- local prediction-error maps
- ordinary and restricted full-canvas source estimates

## Interpretation

Phase 2j changes the Candidate-2 bottleneck. Dense K/V is successful not merely
because it presents 2048 position-indexed vectors at the local interface; how
those vectors were constructed through cross-canvas interaction matters.
Independent coarse, regional, or strip captures cannot be assumed composable by
simple K/V concatenation, even when their final token layout is spatially broad.

This does not yet specify the cheapest sufficient interaction. No block subset,
window size, or interaction schedule was tested. Those remain separate
questions, and this result does not authorize a sweep, trajectory, caching,
sparse executor, or production implementation.

**GLOBAL SOURCE INTERACTION REQUIRED**
