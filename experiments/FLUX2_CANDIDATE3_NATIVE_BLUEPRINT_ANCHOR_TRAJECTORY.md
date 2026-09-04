# Phase 20c — minimal Blueprint prediction-anchor trajectory

## Result

The exact Phase-20b prediction anchor passes the complete existing four-
interval Euler trajectory. The persistent bounded Blueprint remains S3 at every
evaluation, each anchored assembly is S3, and terminal accepted H is one
dominant bridge/train scene. Ordinary local assemblies remain S0 throughout.

Detail behavior is **REGENERATING**: each fresh ordinary local evaluation
produces sharp local bridge detail again, but in incompatible independent
scenes; exact anchoring then removes that destination-visible detail and
returns the coherent, blurred Blueprint plan.

No production or ComfyUI-core code changed. No strength, filter, frequency,
context, representation, or resampling variant was run.

## Exact trajectory

The accepted state is the pair `(B_i, H_i)`, where `B` is a persistent bounded
`32x64` native-coordinate Blueprint and `H` is the `128x256` destination.
`B_0` uses the Phase-20 analytic same-noise construction once; it is never
reconstructed from H afterward.

For every interval:

```text
x0_B = model(B_i, sigma_i)                         # native B coordinates

W_r = ordinary normalized W constructed from crop(H_i, r)
x0_W_r = model(W_r, sigma_i)
x0_W_r' = x0_W_r + U_B(mapped(x0_B)_r - D_B(x0_W_r))
x0_H = overlap_assemble(D_B(x0_W_r'))

B_next = B_i + (sigma_next-sigma_i)*(B_i-x0_B)/sigma_i
H_next = H_i + (sigma_next-sigma_i)*(H_i-x0_H)/sigma_i
```

`D_B` is nonoverlapping `2x2` mean and `U_B` is `2x` nearest, so
`D_B(U_B(z))=z`. B and H are committed atomically only after all 56 model calls,
right-inverse checks, finiteness checks, and input-hash checks pass.

Each completed interval is fsynced as an independent JSON/PT artifact. Resume
requires the exact configuration fingerprint and accepted B/H lineage.

## Semantic scores

| Interval | Blueprint x0 | Ordinary local assembly | Anchored assembly | Accepted H |
|---:|---:|---:|---:|---|
| 0 | S3 | S0 | S3 | sigma 0.999177; decode not meaningful |
| 1 | S3 | S0 | S3 | sigma 0.997536; decode not meaningful |
| 2 | S3 | S0 | S3 | sigma 0.992643; decode not meaningful |
| 3 | S3 | S0 | S3 | S3 terminal |

The final image has one continuous bridge, one train, controlled endpoint
towers, and coherent horizon/water. It is strongly blurred, as expected from
exact Blueprint authority, but blur is explicitly not this phase's failure
condition.

## Measurements

| i | sigma -> next | correction RMS | correction/x0 mean-max | coarse before RMS | coarse after max | overlap ordinary/anchored | anchored-vs-ordinary RMS | B/local CUDA | wall | peak alloc/reserved |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 -> 0.999177 | 1.078492 | 1.1687 / 1.7753 | 1.115877 | 9.54e-7 | 0.956372 / 7.28e-8 | 1.078492 | 1.162 / 60.635 s | 62.993 s | 1.604 / 2.039 GiB |
| 1 | 0.999177 -> 0.997536 | 1.066230 | 1.1631 / 1.7664 | 1.101638 | 7.15e-7 | 0.944957 / 7.63e-8 | 1.066230 | 0.561 / 59.862 s | 61.524 s | 1.818 / 2.195 GiB |
| 2 | 0.997536 -> 0.992643 | 1.039481 | 1.1536 / 1.7272 | 1.072394 | 9.54e-7 | 0.923817 / 7.39e-8 | 1.039481 | 0.556 / 60.624 s | 62.330 s | 1.817 / 2.199 GiB |
| 3 | 0.992643 -> 0 | 0.958928 | 1.1146 / 1.5570 | 0.991291 | 7.15e-7 | 0.854390 / 6.84e-8 | 0.958928 | 0.596 / 59.982 s | 61.755 s | 1.817 / 2.188 GiB |

Totals are four Blueprint forwards and 220 local forwards. All B/H inputs and
all W tensors remain hash-immutable during their evaluations; all outputs are
finite and coverage is complete.

The near-zero anchored overlap is an algebraic result, not an independent
quality metric. Since restriction follows exact anchoring, assembled x0_H is
the mapped Blueprint prediction. Local W computation therefore regenerates
detail, but no local null-space information reaches accepted H under this exact
contract.

## Decision

The semantic gate passes. Exact bounded Blueprint authority maintains one
scene through an actual persistent Blueprint/H trajectory. Stop here: do not
optimize the anchor automatically. The next phase should isolate a principled
way for local-detail/null-space information to affect destination H while
retaining the demonstrated Blueprint authority.

## Artifacts

- `flux2_candidate3_native_blueprint_anchor_trajectory_results/persistent_blueprint_report.json`
- `flux2_candidate3_native_blueprint_anchor_trajectory_results/persistent_blueprint_intervals/`
- `flux2_candidate3_native_blueprint_anchor_trajectory_results/PERSISTENT_BLUEPRINT_TRAJECTORY_COMPARISON.png`
- `flux2_candidate3_native_blueprint_anchor_trajectory_results/PERSISTENT_TERMINAL_COMPARISON.jpg`
