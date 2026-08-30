# FLUX.2 equal-budget multiscale global-context probe

## Verdict

**MULTISCALE BUDGET INSUFFICIENT.**

The fixed equal-token multiscale representation did not recover the semantic
evidence missing from the uniform 1152-token global grid. It made the duplicate
lighthouse and adjacent stone structure more prominent, while increasing both
total and low-frequency prediction error relative to the uniform control.

This falsifies the tested construction—coarse whole-canvas K/V concatenated
with native-density K/V from the affected center region. It does not establish
that every multiscale representation is ineffective.

## Exact controlled state

The Phase-2f/2g late accepted state was reproduced after exactly two accepted
Euler updates:

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
local center crop:     y=0..31, x=24..55
diagnostic updates:    0
```

All GPU-side state statistics matched exactly and the diagnostic returned the
accepted latent unchanged (`max_abs=0`). The same local crop, sigma, text,
conditioning, local RoPE, local hidden/MLP path, attention integration, all 25
FLUX.2 blocks, and final projection were used for every variant.

## Multiscale construction

```text
coarse branch:
  16 x 32 bilinearly reduced whole canvas = 512 tokens
  RoPE spans full y=0..31, x=0..63 coordinates

regional branch:
  native-density accepted-latent crop
  y=8..27, x=24..55 = 20 x 32 = 640 tokens
  RoPE preserves those absolute full-canvas coordinates

combined external context:
  512 coarse + 640 regional = 1152 tokens
```

Coarse and regional K/V were concatenated in every block. Spatially nearby
coarse and regional evidence was deliberately retained twice; no coordinate
deduplication was performed. The regional rectangle covers the affected crop's
train/bridge/horizon band and the duplicate lighthouse/stone area.

## Quantitative comparison

| Variant | Coarse | Regional | Total global K/V | Local Q x K | Relative changed attention work | RMS vs dense | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform 24x48 | 1152 | 0 | 1152 | 1536 x 2688 | 1.000x | 0.327636 | 0.162362 | 0.819669 |
| Multiscale | 512 | 640 | 1152 | 1536 x 2688 | 1.000x | 0.398137 | 0.207734 | 0.809548 |
| Dense 32x64 | 2048 | 0 | 2048 | 1536 x 3584 | 1.333x | 0.179977 | 0.098817 | 0.846498 |

At identical external-token count and local attention dimensions, multiscale
increased RMS by 21.5% and low-frequency RMS by 27.9% relative to uniform 1152.
Prediction scale remained finite and broadly stable.

The attention-work ratio is only Q-by-K products for the changed local
attention context. It is not total-model FLOPs. Multiscale additionally needs
two global capture forwards rather than one. Although their generated-token
sum is 1152, duplicated text processing and separate-forward overhead mean
this probe is equal in external K/V budget, not equal in total execution cost.

## Decoded semantic observations

- **Duplicate lighthouse:** present in the uniform control; larger and more
  conspicuous with multiscale; removed only by dense context.
- **Dark stone structure:** multiscale expands it into a stronger stone/landmass
  complex instead of suppressing it. Dense context reduces it substantially
  but remains imperfect relative to ordinary dense execution.
- **Train:** multiscale compresses/resegments the visible train compared with
  uniform and moves farther from the dense reference.
- **Bridge:** the deck remains continuous, but train, cable, and nearby bridge
  alignment deteriorate under multiscale context.
- **Horizon/water:** the enlarged false regional object complex disrupts the
  horizon more strongly than the uniform control.
- **Detail:** all variants remain detailed; failure is semantic/geometric, not
  simple smoothing or prediction collapse.

The dense-surrounding composites place only the tested center-crop prediction
inside unchanged dense surroundings. They are interpretability diagnostics,
not assembled generation results.

## Artifacts

- `flux2_candidate2_equal_budget_multiscale.py`
- `flux2_candidate2_equal_budget_multiscale_results/report.json`
- decoded crop predictions and dense-surrounding composites
- prediction-error magnitude maps
- `MULTISCALE_POSITION_MAP.png`, showing coarse and regional coordinates

The JSON report records per-block captures and context calls, positional-ID
extents, source and local tensor statistics, token layouts, and manual decoded
observations.

## Interpretation

The result argues against the hypothesis that the missing object-count evidence
is recoverable merely by concentrating an equal token budget inside the visibly
affected crop. Repeating local-area evidence can reinforce the crop-local
alternative rather than restore the broader scene constraint. The dense result
therefore should not be interpreted as proof that only the failure vicinity
needs density.

The next investigation should not increase this region's density. The
highest-information next probe is a **global-distribution control** at the same
1152-token budget: keep 512 coarse whole-canvas tokens and allocate the other
640 as fixed sparse native-density samples distributed across the rest of the
canvas, excluding the local center crop. That distinguishes whether dense K/V
works because it supplies distributed nonlocal relationships rather than
high-density evidence at the failure location. It remains a one-state,
zero-update diagnostic.

**MULTISCALE BUDGET INSUFFICIENT**
