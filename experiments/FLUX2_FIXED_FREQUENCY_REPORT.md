# FLUX.2 fixed-frequency local-correction falsifier

## Question and verdict

Can a fixed spatial high-pass filter remove the crop-local alternative
composition from `local - mapped_global_crop` while retaining useful local
fidelity?

**FREQUENCY SPLIT PARTIALLY WORKS.** A Gaussian low-pass with sigma 1 latent
token (7 x 7 kernel) greatly restrained the compositional replacement seen in
the unfiltered control and recovered visible train, cable, water, and masonry
detail. It did not isolate detail from semantics: a duplicate lighthouse and
alternative bridge structure remained, and those structures are directly
visible in the decoded high-pass diagnostic. Sigma 2 (13 x 13) admitted a much
stronger alternative bridge layout. Fixed spatial frequency is therefore a
useful correction constraint, but not a sufficient Candidate-1 fusion rule.

## Controlled setup

This experiment imports the Phase-2 harness and changes only correction
filtering. It uses the same:

- FLUX.2 Klein 4B W4A8 model, text encoder, and VAE;
- composition-stress prompt and seed 20260829;
- 1024 x 512 target, 512 x 256 global branch, and three overlapping 512 x 512
  crops at latent x offsets 0, 24, and 32;
- CFG 1, four-step Euler schedule, zero churn, conditioning, initial noise,
  coordinate handling, overlap weights, and model-prediction definitions.

For each crop, the experiment computes

```text
r      = local - mapped_global_crop
r_low  = depthwise_gaussian_low_pass(r)
r_high = r - r_low
```

with reflection padding. Crop corrections are overlap-assembled before the
ordinary single Euler update. Four variants were run:

```text
A  mapped_global
B  mapped_global + 0.75 * r
C1 mapped_global + 0.75 * r_high, sigma=1 token, kernel=7
C2 mapped_global + 0.75 * r_high, sigma=2 tokens, kernel=13
```

The implementation is `flux2_fixed_frequency_falsifier.py`; complete machine
telemetry is in `flux2_fixed_frequency_results/report.json`.

The recorded run command was:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_fixed_frequency_falsifier.py
```

## Execution integrity and work

The filter identity was checked numerically: `r_low + r_high` reconstructed
`r` with zero maximum absolute error in the deterministic CPU probe. All real
run outputs were finite and assembled coverage was exactly one over the full
canvas.

Per sampler evaluation was unchanged from Phase 2:

| Variant | Model forwards | Executed image tokens |
|---|---:|---:|
| Global-only | 1 x 512-token global | 512 |
| Unfiltered control | 1 x 512 + 3 x 1024-token crops | 3,584 |
| Either high-pass variant | 1 x 512 + 3 x 1024-token crops | 3,584 |

Filtering adds tensor operations but no model forward or image-token work.
Peak CUDA allocation was approximately 2.45 GiB for global-only and 2.50 GiB
for all local variants. Recorded wall times were 8.73 s for the first/warm-up
global run, 3.90 s unfiltered, 3.99 s sigma 1, and 3.93 s sigma 2; these are
diagnostic only, not a speed qualification.

## Correction statistics

The ratio is the assembled low-pass RMS divided by assembled high-pass RMS.

| Accepted step | Sigma | sigma-1 low/high | sigma-1 low/high fractions | sigma-2 low/high | sigma-2 low/high fractions |
|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 1.55 | 0.61 / 0.39 | 0.88 | 0.47 / 0.53 |
| 1 | 0.9614 | 0.92 | 0.48 / 0.52 | 0.58 | 0.37 / 0.63 |
| 2 | 0.8926 | 0.68 | 0.40 / 0.60 | 0.48 | 0.32 / 0.68 |
| 3 | 0.7348 | 0.55 | 0.36 / 0.64 | 0.38 | 0.28 / 0.72 |

At the first evaluation, sigma-1 low/high RMS values were 0.53/0.34 and
sigma-2 values were 0.42/0.47. The high-pass fraction grows later for both
filters. Norm partition alone is not semantic partition: visible large-scale
objects survive in `r_high`.

Final-image diagnostics relative to global-only show the restraint/detail
tradeoff:

| Variant | Low-frequency MAE vs global-only | High-frequency RMS | Gain over global-only |
|---|---:|---:|---:|
| Global-only | 0.000 | 0.240 | 0.000 |
| Unfiltered 0.75 | 0.311 | 0.550 | 0.309 |
| High-pass sigma 1 | 0.095 | 0.438 | 0.197 |
| High-pass sigma 2 | 0.161 | 0.510 | 0.270 |

These image-space statistics are descriptive, not perceptual metrics.

## Visual observations

- **Global-only:** retains one long, shallow bridge plan, one centered train,
  and the dominant right tower, but is visibly blurred/ghosted.
- **Unfiltered control:** replaces the plan with a sharper crop-local
  alternative: multiple bridge peaks/supports and duplicated lighthouse-like
  structures return.
- **High-pass sigma 1:** remains substantially closer to the global bridge
  silhouette while making the train, suspension cables, water, and right
  masonry tower visibly sharper. This is credible local-fidelity recovery.
  However, it resolves an additional lighthouse on the left and retains
  ghost/alternative bridge lines. Cross-tile semantic duplication is reduced,
  not eliminated.
- **High-pass sigma 2:** gains more sharpness but clearly introduces a second
  bridge organization, including a strong left support and incompatible cable
  arcs. It is compositionally worse than sigma 1.
- Direct VAE decodes of residual tensors are out-of-distribution diagnostics,
  not literal denoised images. Even with that limitation, both high-pass
  diagnostics visibly encode bridge decks/cables, towers, and lighthouse
  silhouettes. The semantic leakage is also corroborated by the final images.

Selected early, middle, and late estimate, low/high residual, magnitude, and
`mapped_global + alpha*r_low` images are saved beside the finals under
`flux2_fixed_frequency_results/`.

## Interpretation

Question 1—can filtering protect global composition? **Partly.** Sigma 1
reduces low-frequency drift by about 69% versus the unfiltered control
(`0.095` versus `0.311` low-frequency MAE) and visibly preserves much more of
the global bridge plan.

Question 2—does high-pass contain only local fidelity? **No.** The high-pass
branch still carries semantic geometry and duplicated objects. A fixed spatial
filter cannot reliably distinguish a fine cable or train edge from the edge of
an independently invented tower, lighthouse, or bridge span.

Candidate 1 therefore survives only in the broad sense that a reduced global
prediction can remain compositionally influential while local computation adds
some detail. It needs a revision beyond fixed output-space frequency splitting.
No production or generalized infrastructure is justified by this result.

## Highest-value next experiment

Test Candidate 2 at one denoising evaluation only: compare ordinary crop
predictions against the same local high-resolution queries given compact
whole-canvas global context, and measure whether the correction's semantic
bridge/lighthouse alternatives shrink before output projection. This directly
tests whether explicit global context prevents local invention; another scalar,
filter-scale, or timestep sweep would not resolve the demonstrated semantic
leakage.

**FREQUENCY SPLIT PARTIALLY WORKS**
