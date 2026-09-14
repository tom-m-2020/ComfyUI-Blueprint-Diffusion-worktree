# Phase 1b — Natural-source generalization of inference-only FSS

Date: 2026-09-14  
Verdict: **WEAK-FSS CASE-DEPENDENT**

## Scope and controls

This experiment retains the Phase 1 stock FLUX.2 Klein 4B mechanism exactly:
the source latent supplies phase to the selected endpoint noise, which is passed
through native CONST initialization and an unchanged Euler trajectory. No
production source, node, model feature, sigma, mask, or guidance mechanism was
changed.

`DRIFT_PHASE_1B_PREREGISTRATION.json` was written and validated before the
output directory existed. It fixed four sources, crop geometry, prompts, hashes,
seed, schedule, and arms:

- A: ordinary Gaussian;
- weak FSS `r=1`;
- primary weak FSS `r=2`;
- strong reference FSS `r=8`.

All cases use the Phase 1 W4A8 Klein checkpoint, Qwen3 encoder, FLUX.2 VAE,
512x512 latent geometry, CFG 1.0, seed `2026091401`, native 8-step schedule
trimmed at fixed `start_index=2`, and six Euler evaluations. Only endpoint noise
differs within a case.

The sources are fixed existing repository artifacts: a portrait crop of the
woman/car/tree image; centered full-body astronaut; centered bridge/train crop;
and the Phase 1 flat illustration. Their input hashes and exact crops are in the
preregistration.

## Diagnostics

Every model evaluation records the same latent trajectory diagnostics as Phase
1: state and x0 summaries, RMS from Gaussian control, source-relative latent
RMS, Fourier-phase deviation, 4x pooled deviation, and gradient-difference RMS.

Final decoded diagnostics add:

- source-relative 4x coarse RGB RMS;
- symmetric Canny edge Chamfer distance in pixels;
- Canny edge-overlap F1 within two pixels;
- source-relative RGB RMS as the primary appearance-change measure;
- per-channel RGB histogram Jensen-Shannon distance as a secondary appearance
  statistic.

OpenCV is installed but no face-landmark model is available. No package or model
was installed. A face box was not mislabeled as landmarks; portrait identity
was assessed visually and is explicitly the limiting case.

## Fixed acceptance rule

Weak FSS counts as useful only where it preserves geometry materially better
than Gaussian and changes appearance materially more than `r=8`. The primary
comparison is `r=2`; `r=1` is only a nearby weak-cutoff robustness check, not a
per-image selection opportunity.

## Results

| case | arm | coarse RGB RMS | edge Chamfer px | edge F1 | RGB appearance change |
|---|---|---:|---:|---:|---:|
| portrait | Gaussian | 0.555 | 7.04 | 0.600 | 0.559 |
| portrait | r=2 | 0.183 | 6.35 | 0.760 | 0.193 |
| portrait | r=8 | 0.148 | 7.39 | 0.844 | 0.156 |
| astronaut | Gaussian | 0.488 | 7.19 | 0.712 | 0.498 |
| astronaut | r=2 | 0.273 | 4.01 | 0.803 | 0.290 |
| astronaut | r=8 | 0.194 | 2.26 | 0.831 | 0.212 |
| bridge | Gaussian | 0.344 | 7.66 | 0.588 | 0.361 |
| bridge | r=2 | 0.160 | 3.50 | 0.742 | 0.180 |
| bridge | r=8 | 0.134 | 3.05 | 0.845 | 0.154 |
| illustration | Gaussian | 0.183 | 21.46 | 0.195 | 0.192 |
| illustration | r=2 | 0.139 | 4.13 | 0.570 | 0.156 |
| illustration | r=8 | 0.125 | 3.22 | 0.730 | 0.138 |

Minor rounding above is for readability; full precision is in `telemetry.json`.

### Portrait — fail

Gaussian replaces the subject with a seated older statue and changes pose and
framing. Both weak arms preserve the standing portrait layout much better, and
`r=2` improves edge F1 from `0.600` to `0.760`. However, weak FSS changes the
woman's apparent age and facial identity. Strong `r=8` is closest to the source
identity but performs little bronze transformation. Because identity/face
geometry is the defining stressor, this case fails despite favorable coarse and
edge metrics.

### Full-body astronaut — pass

Gaussian follows the steampunk/jungle prompt strongly but alters suit silhouette
and detailed body organization. `r=1` and `r=2` retain stance, limb proportions,
helmet, torso panel, and broad suit layout while adding brass details and a more
changed environment. At `r=2`, edge Chamfer improves `7.19 -> 4.01` and edge F1
`0.712 -> 0.803`; RGB change `0.290` remains materially above strong `r=8` at
`0.212`. This passes the preregistered criterion.

### Bridge — pass

Gaussian invents a different bridge type and skyline. Weak FSS retains the
single suspension-tower crop, deck slope, cable system, and train placement
while producing a night/crystalline reinterpretation. `r=2` more than halves
edge Chamfer (`7.66 -> 3.50`) and improves F1 (`0.588 -> 0.742`), while RGB
appearance change remains above `r=8` (`0.180` versus `0.154`). This passes.

### Stylized illustration — pass, consistent with Phase 1

Weak FSS keeps the two-span layout and endpoint structures and adds realistic
lighting/material cues. Gaussian replaces the composition, while `r=8` traces
the source. `r=2` improves edge Chamfer `21.46 -> 4.13` and F1 `0.195 -> 0.570`,
with more RGB change than `r=8` (`0.156` versus `0.138`). This passes but is not
independent natural-image evidence.

## Interpretation

Weak FSS generalizes across coarse articulated-body and straight-contour scene
structure in this small fixed set. It also consistently occupies a measurable
middle ground between Gaussian freedom and strong-FSS tracing. It does not
generalize to the finest identity-sensitive requirement: phase-biased stock
Klein can preserve portrait layout while still changing the face's identity and
age.

Therefore the result is not `WEAK-FSS GENERALIZES`. It is also not
`WEAK-FSS DOES NOT GENERALIZE`, because the astronaut and bridge satisfy both
sides of the preregistered criterion with coherent visual evidence and aligned
non-learned metrics.

Artifacts:

- `experiments/DRIFT_PHASE_1B_PREREGISTRATION.json`
- `experiments/drift_phase_preserving_natural_generalization.py`
- `experiments/drift_phase_1b_natural_results/telemetry.json`
- `experiments/drift_phase_1b_natural_results/assessment.json`
- four comparison sheets, final decodes, source crops, and selected x0 previews
  in `experiments/drift_phase_1b_natural_results/`.

## Stop boundary

Do not productionize FSS from this result. Do not tune a radius per image, add a
face detector/identity loss, or proceed to ILVR. Any next experiment requires a
new explicit task and should target the observed identity-sensitive failure
without changing the established stock-Klein result into a hidden optimization
sweep.
