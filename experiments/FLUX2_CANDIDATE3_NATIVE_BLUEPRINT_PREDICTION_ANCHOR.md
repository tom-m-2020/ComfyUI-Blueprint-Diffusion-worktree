# Phase 20b — terminal Blueprint prediction anchor

## Result

One exact output-space coarse anchor changes the Phase-20 fragmented normalized-
W assembly from S0 to S3. Bounded native-coordinate planning and global-to-local
prediction transfer are therefore viable in this one-state discriminator.
The resulting image is visibly overconstrained and blurred, so detail retention
is not qualified.

No production or ComfyUI-core code changed. No accepted state was updated. No
anchor strength, filter, geometry, or model-input variant was tested.

## Exact intervention

For every ordinary `64x64` local prediction `x`, define:

```text
D_B(x)[c,y,x] = 1/4 * sum(dy,dx in {0,1}) x[c,2y+dy,2x+dx]
U_B(z)[c,2y+dy,2x+dx] = z[c,y,x]

x0_corrected = x0_local + U_B(blueprint_crop - D_B(x0_local))
```

`D_B` is the same nonoverlapping `2x2` arithmetic restriction used to return
the normalized W prediction to its `32x32` destination region. `U_B` is exact
`2x` nearest-neighbor prolongation. Consequently:

```text
D_B(U_B(z)) = z
D_B(x0_corrected) = blueprint_crop
```

The measured maximum absolute post-correction error is `5.36e-7`, below the
declared `1e-5` tolerance. Before correction, the crop-mean coarse discrepancy
is `0.931751` RMS (worst crop `1.350992`).

Only the terminal local x0 tensor after its ordinary model forward changes.
Accepted H/G, W input tensors, sigma, conditioning, coordinates, transformer
execution, noise, region order, overlap weights, and sampler lifecycle are
identical to Phase 20 B. The recomputed ordinary B assembly is bit-exact with
the saved Phase-20 artifact.

## Measurements

| Metric | B ordinary | C anchored |
|---|---:|---:|
| Semantic class | S0 | S3 |
| Overlap RMS | 0.885702 | 6.59e-8 |
| Coverage | 0.99999994..1.00000012 | 0.99999994..1.00000012 |

- Correction RMS: `0.888769`.
- Mean correction/local-x0 RMS ratio: `1.119254` (maximum `1.702614`).
- C versus B assembled RMS/max absolute: `0.888769 / 6.232982`.
- Ordinary local model work: `56.752 s` CUDA and wall time.
- Peak allocated/reserved CUDA: `1.632 / 2.072 GiB`.
- All results are finite; accepted H/G and all W inputs are hash-immutable.

The near-zero overlap is expected from the exact anchor: every restricted crop
is forced to the corresponding crop of the same mapped Blueprint field. It is
not by itself evidence of perceptual quality.

## Semantic interpretation

- B remains a field of independent bridge/train alternatives (S0).
- The bounded `32x64` Blueprint remains a single coherent bridge scene (S3).
- C restores one dominant bridge, one train, consistent endpoints, horizon,
  and water (S3).
- C also suppresses local structure strongly: cables, deck, train, water, and
  towers are visibly soft/ghosted. The correction is larger than the original
  local prediction RMS on average, confirming that this is a scene-scale
  authority intervention rather than a minor seam adjustment.

The causal conclusion is narrow: simple prediction-space Blueprint authority
can transfer the coherent bounded global plan where same-sigma state
initialization could not. The remaining research problem is retaining useful
local null-space/detail while preserving that authority.

Exactly one minimal trajectory follow-up is authorized. It should apply this
same exact prediction-space coarse contract across the existing schedule and
measure whether detail survives/evolves; it must not introduce a strength or
filter sweep.

## Artifacts

- Machine report: `flux2_candidate3_native_blueprint_prediction_anchor_results/report.json`
- Atomic arm artifact: `flux2_candidate3_native_blueprint_prediction_anchor_results/C_BLUEPRINT_PREDICTION_ANCHOR.pt`
- Comparison: `flux2_candidate3_native_blueprint_prediction_anchor_results/B_BLUEPRINT_C_COMPARISON.png`
- Anchored decode: `flux2_candidate3_native_blueprint_prediction_anchor_results/C_BLUEPRINT_PREDICTION_ANCHOR.png`
