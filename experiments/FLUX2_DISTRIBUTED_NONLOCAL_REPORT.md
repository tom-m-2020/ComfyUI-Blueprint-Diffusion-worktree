# FLUX.2 distributed nonlocal evidence discriminator

## Verdict

**DISTRIBUTED NONLOCAL EVIDENCE NOT SUPPORTED.**

At the exact Phase-2f/2g/2h late state, a deterministic 640-token native-density
sample distributed outside the active center crop avoided the severe
failure-local amplification, but did not suppress the known duplicate
lighthouse or dark stone structure relative to uniform 1152. It was also
slightly worse numerically than the uniform control.

This substantially weakens simple spatial reallocation at a 1152-token budget.
The result does not prove that nonlocal information is irrelevant; it shows
that this fixed coarse-plus-distributed-native representation remains below the
observed semantic threshold.

## Controlled state

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
active center crop:    y=0..31, x=24..55
diagnostic updates:    0
modified blocks:       all 5 double + all 20 single
```

The Phase-2h accepted-state statistics reproduced exactly. The diagnostic
returned the accepted latent unchanged (`max_abs=0`). Conditioning, local
queries, local hidden/MLP/residual execution, text handling, RoPE convention,
attention integration, block coverage, and final projection were fixed.

## Deterministic nonlocal selection

Excluding the active crop leaves two native-density strips:

```text
left:   y=0..31, x=0..23   = 768 available positions
right:  y=0..31, x=56..63  = 256 available positions
```

The selector retained the same 5/8 fraction from each strip:

```text
left:   480 tokens
right:  160 tokens
total:  640 tokens
```

Within each strip, selected row-major index `i` was
`floor(i * available_count / selected_count)`. This fixed content-blind rule
covers both available regions proportionally and does not use objects,
attention, errors, saliency, or the known failure position.

The 640 selected native-density K/V tokens were concatenated after the unchanged
512-token coarse whole-canvas K/V. Original full-canvas coordinates were
preserved. Two selected native positions exactly duplicate coarse RoPE
coordinates and were retained, producing 1150 unique positions from 1152 K/V
tokens. The duplicates are `(y=0,x=0)` and `(y=31,x=0)`. Uniform,
failure-local, and dense variants had no exact coordinate duplicates under the
same accounting.

## Quantitative result

| Variant | Coarse | Added | External K/V | Unique positions | Duplicates | Local Q x K | Changed attention work | RMS vs dense | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Uniform 24x48 | 1152 | 0 | 1152 | 1152 | 0 | 1536 x 2688 | 1.000x | 0.327636 | 0.162362 | 0.819669 |
| Failure-local | 512 | 640 | 1152 | 1152 | 0 | 1536 x 2688 | 1.000x | 0.398137 | 0.207734 | 0.809548 |
| Nonlocal | 512 | 640 | 1152 | 1150 | 2 | 1536 x 2688 | 1.000x | 0.330331 | 0.164595 | 0.819376 |
| Dense | 2048 | 0 | 2048 | 2048 | 0 | 1536 x 3584 | 1.333x | 0.179977 | 0.098817 | 0.846498 |

Nonlocal was 17.0% lower in RMS and 20.8% lower in low-frequency RMS than
failure-local, confirming that concentrating tokens around the failure was
actively harmful. However, nonlocal was 0.8% worse in RMS and 1.4% worse in
low-frequency RMS than uniform 1152. Prediction norms were stable.

The changed-attention figure is Q-by-K products for the local context call only,
not total-model FLOPs. The nonlocal diagnostic captured full 768- and 256-token
strips before selecting 480 and 160 K/V entries, in addition to the coarse
capture. It is therefore a representation discriminator, not an efficient
execution result; its total capture work exceeds the uniform control.

## Decoded observations

- **Duplicate lighthouse:** amplified by failure-local; still present at about
  uniform-control severity with nonlocal; removed only by dense context.
- **Dark stone structure:** failure-local enlarges it; nonlocal restores roughly
  the uniform form but does not suppress it.
- **Train:** nonlocal remains continuous and is broadly comparable to uniform,
  but does not move materially closer to dense.
- **Bridge deck/cables:** nonlocal avoids failure-local deterioration, yet shows
  no clear semantic advantage over uniform.
- **Horizon/water:** nonlocal removes the enlarged failure-local landmass but
  retains the original false object pair.
- **New failures:** no clear new nonlocal-specific object was introduced in the
  inspected center crop. This does not qualify unobserved regions because only
  the affected crop prediction was evaluated.

## Artifacts

- `flux2_candidate2_distributed_nonlocal.py`
- `flux2_candidate2_distributed_nonlocal_results/report.json`
- decoded predictions, dense-surrounding composites, and error maps
- `NONLOCAL_POSITION_MAP.png`

The machine-readable report includes all selected indices and coordinates,
exact position accounting, all 25 per-variant context records, source capture
records, tensor statistics, and manual decoded observations.

## Interpretation

The requested discriminator is negative. Distributed nonlocal evidence is much
better than repeating failure-local evidence, but at this budget it behaves like
another approximation to uniform 1152 and does not cross the object-uniqueness
threshold. This weakens the idea that a simple fixed spatial reallocation of
1152 K/V tokens can substitute for the successful dense context.

Per the experiment boundary, no further density or layout sweep is proposed or
run here. The architectural question should be reassessed before another probe:
dense K/V may benefit from globally dense hidden-state interaction during its
source forward, from full pairwise coverage, or from information that simple
concatenation of independently captured coarse/strip trajectories does not
preserve.

**DISTRIBUTED NONLOCAL EVIDENCE NOT SUPPORTED**
