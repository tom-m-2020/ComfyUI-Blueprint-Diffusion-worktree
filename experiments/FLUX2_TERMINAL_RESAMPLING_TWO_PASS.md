# Phase 33 — Fixed two-pass terminal resampling discriminator

## Result

**B — SECOND PASS RETAINS S3 BUT DOES NOT IMPROVE DETAIL**

Exactly one additional model-mediated terminal pass preserves the single-scene
S3 interpretation in all three qualified cases. It does not, however, recover
credible local structural detail. The second pass primarily amplifies fine
texture/alias energy while leaving the car, astronaut, and bridge structures
similarly soft.

Production and ComfyUI core remain unchanged. Blueprint trajectories and the
one-pass controls were reused from Phase 29 rather than recomputed.

## Fixed execution contract

For every region, pass B uses the completed one-pass destination `H1` directly:

```text
crop(H1)
 -> nearest2
 -> model_sampling.noise_scaling(0.25, same_region_noise, anchor)
 -> ordinary native 64x64 FLUX prediction
 -> avgpool2
 -> unchanged normalized overlap assembly
 -> H2
```

The exact pass-1 region seed/noise convention is reused. There is no fresh
stochasticity, Blueprint return, projection, guidance, K/V context, or third
pass.

## Integrity and resumability

- All three A controls are bit-exact to the Phase-28/29 selected latents.
- Terminal Blueprint/mapped hashes, prompt, conditioning, geometry, ordering,
  and overlap weights validate before B.
- H1 hashes before and after pass B are identical.
- Each second-pass restricted prediction and telemetry record is atomically
  persisted before the next region. Primary and repeat passes resume by region
  only after H1/noise/tensor fingerprints validate.
- Primary versus independent repeat is bit-exact for H2 and every restricted
  region in all three cases.
- Coverage is complete and positive; all tensors are finite.
- Local calls are exactly 25 square, 55 portrait, and 40 landscape per pass;
  destination-sized model forwards are zero.
- Post-region CUDA allocation is flat (range `0` bytes) in every primary pass.

## Measurements

| Case | A/B grade | RMS H2-H1 | BP RMS H1 -> H2 | LF RMS H1 -> H2 | gradient H1 -> H2 | overlap H1 -> H2 | B CUDA / wall | peak alloc / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| square multi-object | S3 / S3 | 0.10989 | 0.13250 -> 0.24025 | 0.06465 -> 0.11292 | 0.23882 -> 0.33838 | 0.17345 -> 0.17736 | 24.40 / 25.68 s | 2.823 / 3.299 |
| portrait astronaut | S3 / S3 | 0.10087 | 0.12441 -> 0.22239 | 0.07378 -> 0.12582 | 0.18823 -> 0.28255 | 0.17225 -> 0.17706 | 53.89 / 56.44 s | 2.831 / 3.314 |
| landscape bridge | S3 / S3 | 0.10123 | 0.12333 -> 0.22205 | 0.06427 -> 0.11144 | 0.20241 -> 0.29632 | 0.16277 -> 0.16758 | 39.41 / 41.30 s | 2.827 / 3.311 |

Second-pass latent hashes:

- square: `96f1958b4d0bfccb3a44241375b54356efc980beb03b325beaac27b84a0f48d3`;
- portrait: `16ee8f65b1673c319733d7058872e647b9da08c6f39094d00bc8f98925992289`;
- landscape: `7fa99876a64333742da0579d15eda6ff715b5b6f1ed4a8d230795aebadde409e`.

## Semantic and detail review

- **Square:** one car, tree, and house remain. The second pass does not
  materially clarify wheels, windows, foliage, roof edges, or grass structure;
  stronger microtexture is predominantly alias-like.
- **Astronaut:** one coherent body remains. Helmet, limb boundaries, suit
  panels, shadow, and ground structure remain soft despite greater texture
  energy.
- **Bridge:** one continuous bridge remains. Cables, deck, supports, secondary
  tower, and water structure do not become materially clearer; line texture is
  stronger without corresponding geometric recovery.

Gradient RMS rises by roughly 42–50%, but this is not accepted as detail. The
visual comparison shows no consistent improvement in meaningful native-scale
structure. At the same time, Blueprint RMS nearly doubles, low-frequency error
rises materially, and overlap disagreement increases slightly.

## Interpretation

An already Blueprint-organized H1 is a semantically stable anchor for one more
local pass, so the second pass does not immediately destroy Blueprint authority.
Nevertheless, repeated ordinary resampling is not the missing fidelity
mechanism: it spends another complete local pass mostly amplifying texture and
deviation rather than resolving structure.

Stop at two passes. The next phase should investigate a **model-mediated or
explicit multi-state construction** with a principled coarse/detail contract,
not another pass-count, sigma, noise, lift, or spatial-transform sweep.

## Artifacts

- Harness: `experiments/flux2_terminal_resampling_two_pass.py`
- Machine report: `experiments/flux2_terminal_resampling_two_pass_results/report.json`
- Comparison: `experiments/flux2_terminal_resampling_two_pass_results/PHASE33_COMPARISON.jpg`
- Detail review: `experiments/flux2_terminal_resampling_two_pass_results/DETAIL_REVIEW.jpg`
- Per-region primary/repeat tensors and telemetry are stored under each case in
  `experiments/flux2_terminal_resampling_two_pass_results/`.
