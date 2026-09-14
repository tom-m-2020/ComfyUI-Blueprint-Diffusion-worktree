# Phase 2 - ILVR-style drift restraint on stock FLUX.2 Klein 4B

## Verdict

**ILVR-STYLE CORRECTION PARTIAL**

The fixed ILVR-like arm materially restrains coarse drift in all three cases and
does so with a small state correction by RMS/energy. It preserves astronaut and
bridge geometry while permitting visible material and lighting changes.
However, the preregistered decisive portrait still loses facial identity,
apparent age, arm pose, and framing. Scalar edge/coarse improvements therefore
do not qualify it as a general identity-preserving method.

No production node or Blueprint architecture changed. No FSS+ILVR arm, learned
loss, mask, landmark model, ControlNet, PiD, or FBSDiff was used.

## Primary-source rule and Klein interpretation

ILVR defines `phi_N` as a linear downsample/upsample low-pass operation. After
an unconditional DDPM proposal `x'_(t-1)`, it samples the reference at the same
noise level, `y_(t-1) ~ q(y_(t-1) | y)`, and applies:

```text
x_(t-1) = phi_N(y_(t-1)) + x'_(t-1) - phi_N(x'_(t-1))
```

This experiment does not claim DDPM equivalence. The controlled CONST/flow
interpretation is:

```text
proposal = x_sigma + (x_sigma - denoised) / sigma * (sigma_next - sigma)
source_at_next = native_CONST_noise_scaling(sigma_next, gaussian, source)
accepted = proposal + alpha * (phi(source_at_next) - phi(proposal))
```

The correction is applied only when `sigma_next > 0`, after the accepted Euler
interval and before the next model evaluation. There is no terminal correction
at sigma zero. The ordinary model forward, Euler derivative, sigma schedule,
and native CONST initialization are untouched.

`phi` is fixed before outputs: area downsample from the `32 x 32` latent grid to
`8 x 8`, followed by nearest upsample. This blockwise operator is linear and
idempotent, so the full replacement (`alpha=1`) exactly replaces the projected
coarse component. It controls 1/16 (6.25%) of spatial degrees of freedom per
channel, although its smooth correction is represented over all latent cells.

The source path uses the same Gaussian endpoint as the ordinary arm. Numerical
telemetry verifies native CONST equals
`sigma_next * gaussian + (1-sigma_next) * source` at each correction interval.
This is the smallest deterministic matching-sigma interpretation; it is not the
paper's stochastic DDPM transition law.

## Preregistered controls

- Model: stock `Flux2-Klein-4B-w4a8.safetensors`
- VAE/text encoder: unchanged from Phase 1b
- Resolution: `512 x 512`
- CFG: `1.0`
- Sampler: Euler, native CONST
- Schedule: 8-step Klein schedule starting at index 2 (six evaluations)
- Seed: `2026091401` for every case
- Arms: Gaussian, Phase 1b FSS `r=2`, Gaussian endpoint plus ILVR `N=4`
- Cases: portrait bronze (decisive), astronaut steampunk, bridge crystal night

The preregistration is
`experiments/DRIFT_PHASE_2_PREREGISTRATION.json`.

## Numerical preflight

For every source, the fixed projection passed:

- projection idempotence RMS error: `0`;
- projection linearity RMS error: approximately `3.0e-8`;
- full-replacement projected mismatch RMS error: approximately `3.2e-8`;
- source-at-matching-sigma formula RMS error: `0` at every corrected interval.

All nine trajectories completed finite. Five nonterminal intervals were
corrected per ILVR arm; the terminal interval to sigma zero was recorded and
left uncorrected.

## Final results

| Case / arm | coarse RGB RMS | edge Chamfer px | edge F1 | appearance RGB RMS | latent coarse RMS | phase rad | gradient diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Portrait Gaussian | 0.5550 | 7.043 | 0.600 | 0.5589 | 0.6340 | 1.759 | 1.436 |
| Portrait FSS r=2 | 0.1832 | 6.350 | 0.760 | 0.1927 | 0.4073 | 1.642 | 1.263 |
| Portrait ILVR | **0.1237** | **3.832** | 0.703 | 0.1371 | **0.2192** | 1.696 | 1.389 |
| Astronaut Gaussian | 0.4881 | 7.189 | 0.712 | 0.4979 | 0.6596 | 1.719 | 1.583 |
| Astronaut FSS r=2 | 0.2731 | 4.013 | **0.803** | 0.2896 | 0.4250 | **1.634** | **1.511** |
| Astronaut ILVR | **0.1317** | **3.416** | 0.785 | 0.1637 | **0.2180** | 1.667 | 1.589 |
| Bridge Gaussian | 0.3444 | 7.659 | 0.588 | 0.3607 | 0.5771 | 1.740 | 1.474 |
| Bridge FSS r=2 | 0.1598 | 3.502 | 0.742 | 0.1802 | 0.4191 | **1.635** | **1.342** |
| Bridge ILVR | **0.0912** | **2.492** | **0.759** | 0.1168 | **0.2069** | 1.676 | 1.493 |

Lower is better except edge F1. Appearance RMS is descriptive: the prompts
require a visible change, so lower is not intrinsically better.

## Correction magnitude / overwrite accounting

| Case | mean correction/proposal RMS | maximum | mean correction/proposal energy | projected mismatch removed |
|---|---:|---:|---:|---:|
| Portrait | 5.12% | 10.95% | 0.361% | 100% |
| Astronaut | 5.19% | 10.45% | 0.350% | 100% |
| Bridge | 4.65% | 9.87% | 0.296% | 100% |

Thus each correction fully replaces the declared 8x8 projected component, but
the actual change is modest relative to the full proposal: roughly 5% by RMS
and 0.3-0.4% by energy on average. The successful coarse metrics do not arise
from repeatedly replacing most of the state. They do arise from hard ownership
of the selected coarse subspace at every nonterminal interval.

## Visual discriminator

- Portrait: fail. ILVR retains more source-like location/background and changes
  the subject into a statue, but the output is an older person with different
  facial geometry, crossed arms, and altered body framing. This overrides its
  favorable coarse and Chamfer scores.
- Astronaut: pass for the requested scope. ILVR closely retains stance,
  silhouette, and proportions while changing the suit material and adding the
  stylized environment. It is more source-like in lighting than Gaussian.
- Bridge: pass for the requested scope. ILVR retains tower locations, deck,
  cables, train, and perspective while changing material/lighting. Its change
  is visibly weaker than Gaussian but still material.

The lower appearance RMS shows a real restraint/change tradeoff. The setting is
not classified as overconstrained because all outputs visibly transform, and
the correction energy is small. It is not promising because the decisive face
identity condition fails. A stronger arm is not needed: full projected
replacement already establishes the limitation, and strengthening it would
expand the constrained subspace rather than resolve the discriminator cleanly.

## Artifacts

- Focused harness: `experiments/drift_ilvr_klein.py`
- Machine-readable telemetry: `experiments/drift_phase_2_ilvr_results/telemetry.json`
- Source/final PNGs, comparison sheets, and selected x0 previews:
  `experiments/drift_phase_2_ilvr_results/`

No combined FSS+ILVR experiment is authorized by this result.
