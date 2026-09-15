# Phase 5 - Sampler-latent FBSDiff adaptation for Klein

## Verdict

**SAMPLER-LATENT FBSDIFF FALSIFIED**

The one preregistered sampler-latent adaptation completed successfully, but it
did not preserve the decisive portrait's identity, age, pose, or framing. It
also remained substantially less source-structural than FSS `r=2` in all three
cases. This falsifies this fixed stock-Klein interpretation, not published
FBSDiff on its DDIM/Stable Diffusion contract.

## Reference-trajectory resolution before inference

Published FBSDiff pairs an inverted/reconstructed source trajectory with a
target trajectory at the same DDIM timesteps. Stock Klein has no qualified
equivalent of that inversion/reconstruction path. The smallest controlled
analogue used here was therefore:

1. encode the same clean source used by ordinary img2img;
2. construct the native Klein CONST endpoint with the ordinary Gaussian tensor,
   same seed, and same starting sigma as the target;
3. evolve an independent reference branch with empty conditioning and CFG 1;
4. store every accepted reference state and pair it with the target state at
   the identical sigma ordinal.

Empty conditioning follows FBSDiff's inversion/reconstruction intent without
inventing a source-description prompt. It is explicitly not DDIM inversion.
The final empty-reference states remained `1.176/1.228/1.226` latent RMS from
the clean portrait/astronaut/bridge source latents, respectively. Consequently,
the reference branch is synchronized in sigma but is not a faithful source
reconstruction. This is the principal formulation limitation of the result.

## Preregistered band and calibration

The stated objective was contour preservation with appearance freedom, so the
single selected policy was FBSDiff's high-FBS band. The paper's default
high-pass threshold is 5 on a 64x64 latent. Rather than silently copying an
absolute integer onto Klein's 32x32 latent, the experiment preserved its
normalized single-axis frequency:

```text
tau = 5 / (64 - 1)
Klein continuous threshold = tau * (32 - 1) = 2.460317
reference mask = (u + v) / 31 > tau
               = integer u + v >= 3
```

The mask substitutes 1018 of 1024 spatial coefficients per channel
(`99.4141%`). The distinction matters: `u+v>=3` is derived from the normalized
`5/63` boundary, not a reuse of Phase 4's literal threshold 3.

Calibration was fixed to evaluations 0, 1, and 2, followed by release for
evaluations 3-5. Three of six model evaluations is 50%, the nearest unswept
integer span to the published early approximately 45-55% period.

## Exact operation

At each active target evaluation `i`, before the untouched model forward:

```text
reference state x_ref(sigma_i)       target state x_tgt(sigma_i)
              |                                  |
       channel-wise 2D DCT                 channel-wise 2D DCT
              |                                  |
       high-band coefficients  ----------> replace target high band
                                                   |
                                                  IDCT
                                                   |
                                      ordinary Klein model evaluation
                                                   |
                                      ordinary accepted Euler interval
```

Both inputs are `[1,128,32,32]` sampler states at the same sigma. No hidden
feature, model weight, denoised estimate, Euler equation, sigma value, or
terminal sigma-zero state is changed. The target endpoint is ordinary Gaussian;
FSS is a separate control only.

The DCT preflight passed with `3.00e-7` orthonormal-identity RMS error and
`2.43e-6` roundtrip RMS error.

## Results

Lower is better except edge F1.

| Case / arm | coarse RGB RMS | Chamfer px | edge F1 | appearance RGB RMS | latent coarse RMS | phase rad | gradient diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Portrait Gaussian | 0.5550 | 7.043 | 0.600 | 0.5589 | 0.634 | 1.759 | 1.436 |
| Portrait FSS r=2 | **0.1832** | **6.350** | **0.760** | 0.1927 | **0.407** | **1.642** | **1.263** |
| Portrait sampler-latent FBS | 0.4898 | 10.353 | 0.516 | 0.4938 | 0.623 | 1.748 | 1.428 |
| Astronaut Gaussian | 0.4881 | 7.189 | 0.712 | 0.4979 | 0.660 | 1.719 | 1.583 |
| Astronaut FSS r=2 | **0.2731** | **4.013** | **0.803** | 0.2896 | **0.425** | **1.634** | **1.511** |
| Astronaut sampler-latent FBS | 0.4423 | 4.370 | 0.747 | 0.4526 | 0.610 | 1.758 | 1.606 |
| Bridge Gaussian | 0.3444 | 7.659 | 0.588 | 0.3607 | 0.577 | 1.740 | 1.474 |
| Bridge FSS r=2 | **0.1598** | **3.502** | **0.742** | 0.1802 | **0.419** | **1.635** | **1.342** |
| Bridge sampler-latent FBS | 0.3019 | 5.889 | 0.598 | 0.3155 | 0.557 | 1.744 | 1.484 |

The sampler-latent arm retains substantial appearance change, so it is not
overconstrained. It provides modest decoded coarse or edge gains over Gaussian
for astronaut and bridge, but it is inferior to FSS on every reported
structural measure in those cases. On portrait it makes Chamfer and F1 worse
than Gaussian.

## Substitution accounting

| Case | coefficient fraction | reference high RMS | target high RMS | substituted RMS | substituted/target RMS | energy ratio |
|---|---:|---:|---:|---:|---:|---:|
| Portrait | 99.414% | 0.9281 | 0.9299 | 0.0196 | 2.13% | 0.070% |
| Astronaut | 99.414% | 0.9300 | 0.9311 | 0.0244 | 2.66% | 0.109% |
| Bridge | 99.414% | 0.9329 | 0.9318 | 0.0201 | 2.18% | 0.073% |

Nearly all coordinate positions are selected, yet the actual changed energy is
small because paired states share their initial endpoint and diverge only under
conditioning. Coefficient count must therefore not be interpreted as state
overwrite magnitude.

## Visual discriminator

- Portrait: substantial bronze-like appearance change occurs, but the arm
  produces an older different person, crossed arms, different pose and body
  framing, and shifted face geometry. It is no identity improvement over either
  Gaussian or FSS.
- Astronaut: the arm changes an adult astronaut into a child-like figure and
  alters body proportions, face, and suit silhouette. FSS retains the adult
  source geometry more closely.
- Bridge: the output remains a bridge at night, but tower locations, cable
  geometry, deck contour, and perspective diverge from the source. FSS retains
  those contours more closely.

Because the portrait is decisive and fails qualitatively as well as on edge
metrics, scalar improvements elsewhere cannot qualify the method.

## Scope and stopping boundary

This result rejects exactly one policy: normalized high-FBS `5/63`, evaluations
0-2, and the empty-conditioned native CONST/Euler reference analogue. It does
not reject published FBSDiff, whose inverted/reconstructed DDIM reference is
unavailable here. No alternate band, threshold, calibration span, reference
conditioning, inversion method, or hybrid was tested. No production code or
model internal was modified.

Artifacts:

- `experiments/DRIFT_PHASE_5_PREREGISTRATION.json`
- `experiments/drift_sampler_latent_fbsdiff_klein.py`
- `experiments/drift_phase_5_sampler_latent_fbsdiff_results/telemetry.json`
- final images, comparison sheets, and selected x0 previews under
  `experiments/drift_phase_5_sampler_latent_fbsdiff_results/`
