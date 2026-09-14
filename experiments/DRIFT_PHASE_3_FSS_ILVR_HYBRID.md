# Phase 3 - Minimal FSS + ILVR hybrid falsifier

## Verdict

**FSS+ILVR STILL INSUFFICIENT**

The fixed hybrid exhibits quantitative complementarity: relative to standalone
ILVR it inherits better Fourier phase, gradient, and edge-overlap behavior from
FSS, while relative to standalone FSS it gains substantially tighter coarse
state restraint. Its ILVR corrections are also smaller than Gaussian+ILVR in
all three cases. Nevertheless, it fails the decisive preregistered requirement:
the portrait does not show a qualitative identity improvement over both
standalone arms and its bronze-statue transformation is weaker than ILVR.

No production code changed. No parameter was tuned after outputs and no extra
arm was run.

## Fixed contract

All Phase 2 runtime inputs are unchanged: stock FLUX.2 Klein 4B W4A8, FLUX.2
text encoder and VAE, 512x512 source img2img, CFG 1.0, Euler, the same six-call
suffix of the eight-step Klein schedule, identical prompts, and seed
`2026091401`.

Exactly four arms were compared:

1. Gaussian control.
2. FSS `r=2`.
3. ILVR `N=4, alpha=1` using the Gaussian endpoint.
4. FSS `r=2` plus ILVR `N=4, alpha=1`.

For the hybrid, one validated FSS tensor is used in both locations:

```text
initial_state = native_CONST_noise_scaling(sigma_start, fss_r2, source)

proposal = ordinary unchanged Euler interval
source_at_next = native_CONST_noise_scaling(sigma_next, fss_r2, source)
accepted = proposal + phi(source_at_next) - phi(proposal)
```

The correction follows every accepted nonterminal interval and precedes the
next model evaluation. Sigma zero is not corrected. Telemetry hashes confirm
that the hybrid endpoint and matching-sigma source endpoint are the same tensor.

## Final diagnostics

| Case / arm | coarse RGB RMS | Chamfer px | edge F1 | appearance RGB RMS | latent coarse RMS | phase rad | gradient diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| Portrait Gaussian | 0.5550 | 7.043 | 0.600 | 0.5589 | 0.6340 | 1.759 | 1.436 |
| Portrait FSS | 0.1832 | 6.350 | 0.760 | 0.1927 | 0.4073 | 1.642 | **1.263** |
| Portrait ILVR | 0.1237 | **3.832** | 0.703 | 0.1371 | 0.2192 | 1.696 | 1.389 |
| Portrait hybrid | **0.1070** | 4.846 | **0.790** | 0.1213 | **0.2030** | **1.618** | 1.282 |
| Astronaut Gaussian | 0.4881 | 7.189 | 0.712 | 0.4979 | 0.6596 | 1.719 | 1.583 |
| Astronaut FSS | 0.2731 | 4.013 | 0.803 | 0.2896 | 0.4250 | 1.634 | **1.511** |
| Astronaut ILVR | 0.1317 | **3.416** | 0.785 | 0.1637 | 0.2180 | 1.667 | 1.589 |
| Astronaut hybrid | **0.1242** | 3.654 | **0.806** | 0.1564 | **0.2036** | **1.612** | 1.518 |
| Bridge Gaussian | 0.3444 | 7.659 | 0.588 | 0.3607 | 0.5771 | 1.740 | 1.474 |
| Bridge FSS | 0.1598 | 3.502 | 0.742 | 0.1802 | 0.4191 | 1.635 | **1.342** |
| Bridge ILVR | 0.0912 | **2.492** | 0.759 | 0.1168 | 0.2069 | 1.676 | 1.493 |
| Bridge hybrid | **0.0903** | 2.834 | **0.797** | 0.1159 | **0.1959** | **1.618** | 1.389 |

Lower is better except edge F1. Appearance RMS describes change from the source;
it is not a preservation score because each prompt requires transformation.

## Correction and overwrite accounting

| Case | Gaussian+ILVR mean correction RMS | Hybrid mean correction RMS | Hybrid/Gaussian ratio | Gaussian+ILVR relative RMS | Hybrid relative RMS | Hybrid mean energy ratio |
|---|---:|---:|---:|---:|---:|---:|
| Portrait | 0.0423 | 0.0339 | 0.803 | 5.12% | 4.03% | 0.244% |
| Astronaut | 0.0443 | 0.0356 | 0.803 | 5.19% | 4.10% | 0.243% |
| Bridge | 0.0394 | 0.0351 | 0.893 | 4.65% | 4.10% | 0.245% |

The answer to the explicit correction-size question is yes. Hybrid corrections
are 19.7% smaller by absolute RMS for portrait and astronaut, and 10.7% smaller
for bridge. Relative-to-proposal correction is also smaller in every case. This
supports the narrow inference that the phase-biased endpoint begins/evolves
closer to the selected source coarse manifold. It does not establish that the
full latent is on a learned structural manifold.

Every ILVR arm still replaces 100% of the declared 8x8 projected mismatch at
five nonterminal intervals. The small energy percentages describe the actual
full-state delta, not partial ownership of that projected subspace.

## Critical comparisons

### Hybrid versus FSS

Repeated coarse correction substantially improves decoded and latent coarse
RMS in every case. Portrait coarse RGB falls `0.1832 -> 0.1070`; astronaut
`0.2731 -> 0.1242`; bridge `0.1598 -> 0.0903`. Edge F1 also improves. Thus ILVR
adds measurable coarse/contour restraint to the FSS trajectory.

### Hybrid versus ILVR

FSS initialization improves final Fourier phase and gradient difference in all
three cases and edge F1 from `0.703 -> 0.790`, `0.785 -> 0.806`, and
`0.759 -> 0.797`. Chamfer is slightly worse than ILVR in all three. The phase
bias therefore adds a repeatable metric-level contour/frequency effect, not a
uniform improvement across structural metrics.

### Hybrid versus source

Astronaut and bridge retain visible material/lighting changes. Portrait retains
nonzero color/texture change (`appearance RGB RMS 0.1213`) but looks more like
an aged recolored person than the requested detailed bronze statue. Appearance
freedom is nonzero but weaker than either standalone arm.

## Decisive portrait assessment

The hybrid restores source-like placement and uncrossed-arm silhouette better
than standalone ILVR. It is not a qualitative identity improvement over both
standalones: the face remains the older, different identity already produced by
FSS, with altered facial proportions and age. Its improved edge F1, phase, and
coarse RMS cannot substitute for this failed visual criterion. The requested
bronze material conversion is also visibly less complete than ILVR's statue.

Accordingly, the mechanisms have complementary scalar effects but the fixed
combination does not solve the failure that motivated the hybrid test. It is
not classified as redundant, because each mechanism contributes distinct
measured changes; it is not classified as broadly overconstrained, because the
nonportrait cases retain material transformation.

## Artifacts

- Preregistration: `experiments/DRIFT_PHASE_3_PREREGISTRATION.json`
- Focused harness: `experiments/drift_fss_ilvr_hybrid_klein.py`
- Machine-readable telemetry: `experiments/drift_phase_3_fss_ilvr_results/telemetry.json`
- Final PNGs, comparison sheets, and selected x0 previews:
  `experiments/drift_phase_3_fss_ilvr_results/`

Stop here. This result authorizes no hybrid tuning or further method.
