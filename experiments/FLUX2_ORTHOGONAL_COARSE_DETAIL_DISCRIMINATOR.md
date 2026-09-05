# Phase 35 — Fixed model-mediated orthogonal coarse/detail discriminator

## Result

**B — partial/ambiguous.** Exact Blueprint coarse ownership works and retains
the single S3 composition, but the untouched FLUX-predicted Haar detail bands
are strongly artifacted and do not add credible higher-resolution structure.

The output still contains exactly one red car on the left, one tree in the
center, and one white house on the right under one continuous horizon.
However, the doubled-resolution decode shows dense regular crosshatch/ghost
texture and broadened edge echoes. Wheels, glazing, foliage, roof/window
structure, and grass do not become meaningfully better resolved.

This is a fixed one-scene discriminator. No parameter, transform, sigma, or
coupling variant was tested. Production and ComfyUI core are unchanged.

## Fixed provenance

- Case: Phase-29 `SQUARE_MULTI_OBJECT`
- Seed: `20260921`
- Prompt, conditioning, model, sigma `0.25`, native `64x64` calls, region
  ordering, deterministic noise, and overlap weights are unchanged.
- Mapped Blueprint shape/hash:
  `[1,128,128,128]` /
  `8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05`
- Qualified one-pass control hash:
  `1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3`
- Regions: 25 row-major `32x32` regions, unchanged coverage and weights.
- Local model calls: 25 primary plus 25 independent repeat; no
  destination-sized model forward.

## Tensor path

For each `[1,128,64,64]` native prediction, the orthonormal Haar transform
produces four `[1,128,32,32]` bands. Model coarse is retained diagnostically,
then replaced exactly by `2*b`; the three model detail tensors are passed to
assembly without arithmetic modification.

All four assembled fields have shape `[1,128,128,128]`. Final inverse Haar
produces `[1,128,256,256]`, decoded as `4096x4096` pixels.

All four coverage fields are finite and identical, with minimum `1.0` and
maximum `4.0` before normalization. Coarse overlap disagreement is exactly
zero because overlapping regions use crops of the same mapped Blueprint.

## Numerical integrity

| Check | Result |
|---|---:|
| maximum per-region Haar round-trip error | `9.5367e-7` |
| all detail hashes unchanged by coarse replacement | true |
| assembled `C` vs `2*Blueprint` RMS / max | `4.2410e-8` / `9.5367e-7` |
| `avgpool2(reconstructed)` vs Blueprint RMS / max | `2.8378e-8` / `4.7684e-7` |
| mean / max region model-C vs forced-C RMS | `0.28413` / `0.33064` |
| reconstructed vs nearest Blueprint RMS / max | `0.18169` / `1.19609` |
| primary/repeat latent difference | exactly zero |

Reconstructed latent hash:
`e6ffacfa728906134a04d11738b5407e7b98c11a601fede1a74ad3ab770f3b15`.
Decoded RGB hash:
`98ac0f643eb4fe48d4eebb05c02483ac1be7d3e4b8cc8481231d509f464287a5`.

## Band telemetry

| Band | RMS | norm | overlap RMS |
|---|---:|---:|---:|
| forced `C` | `1.57019` | `2273.79` | `0.00000` |
| `F1` | `0.21263` | `307.91` | `0.31421` |
| `F2` | `0.21191` | `306.86` | `0.31557` |
| `F3` | `0.20476` | `296.51` | `0.30218` |

The three independently predicted detail fields disagree substantially in
overlaps despite exact shared coarse authority. Their energy is real and
survives reconstruction, but its decoded manifestation is regular alias/ghost
texture rather than compatible structural detail.

Reconstructed latent RMS is `0.80584`, norm `2333.17`, and all values are
finite.

## Memory and execution

- One region's four fp32 bands: 2 MiB.
- Four assembled fp32 fields: 32 MiB.
- Reconstructed fp32 latent: 32 MiB.
- Primary local CUDA: 24.46 s; model-pass wall time: 26.11 s.
- Primary peak CUDA allocated/reserved: 2.82/3.30 GiB.
- Post-region allocated memory was exactly constant at every primary barrier.
- The repeat was bit-exact at every prediction hash and final tensor. Its
  higher reserved watermark reflects execution after tiled VAE decode, not
  growing region-state residency.

## Semantic/detail judgment

The coarse coordinate succeeds at its narrow task: the decoded output remains
in the Blueprint scene basin with the correct object count and arrangement.
There is no independent alternative car/tree/house scene.

The detail gate fails. Compared with the qualified one-pass output, retained
bands do not resolve meaningful object parts. They create a conspicuous
high-frequency lattice, repeated edge ghosts around all three objects, and
additional blur-like spreading. Higher spatial sampling and nonzero detail
energy therefore do not constitute useful fidelity.

The result is not classified C because the whole-scene structure does not
clearly fail. It is classified B because exact coarse authority is proven but
the model-predicted local detail is incompatible/artifacted and empirically
inconclusive as a useful higher-resolution representation.

## Decision

Do not tune the bands, add scaling, change wavelets, or repeat the discriminator
until it passes. The experiment falsifies the fixed assumption that untouched
independent local FLUX Haar detail can simply be assembled beneath an exact
Blueprint coarse field. Any next architecture must address cross-region/model
compatibility of detail state rather than alter the already-correct Haar
algebra.

## Artifacts

- Harness: `experiments/flux2_orthogonal_coarse_detail_discriminator.py`
- Machine report:
  `experiments/flux2_orthogonal_coarse_detail_discriminator_results/report.json`
- Reconstructed latent and four assembled bands are persisted as `.pt` files.
- `C`, `F1`, `F2`, and `F3` energy maps are persisted separately.
- Final decode:
  `experiments/flux2_orthogonal_coarse_detail_discriminator_results/RECONSTRUCTED_FINAL.png`
- Comparison:
  `experiments/flux2_orthogonal_coarse_detail_discriminator_results/PHASE35_COMPARISON.jpg`
- Each primary/repeat region has an atomic tensor and JSON artifact.
