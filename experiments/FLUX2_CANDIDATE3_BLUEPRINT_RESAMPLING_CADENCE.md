# Phase 24 — Terminal versus periodic Blueprint-local resampling cadence

## Verdict

**TERMINAL-ONLY RESAMPLING SUFFICIENT**

Both cadence arms remain S3, but carrying the midpoint refinement residual into
the terminal local seed provides no clear perceptual detail advantage and moves
the result farther from the authoritative terminal Blueprint. The additional
midpoint local pass is therefore not justified by this bounded discriminator.

No fully interleaved arm was run. No production or ComfyUI-core code changed.

## Fixed comparison

The four exact Phase-20c Blueprint states/predictions were reused from persisted
artifacts. Blueprint forwards were not recomputed.

Common terminal local construction:

```text
W_0.25 = 0.75 * nearest2(clean_anchor_crop) + 0.25 * epsilon_region
```

The same Phase-22 per-region noise tensors, native `64x64` local coordinates,
55-region ordering, conditioning, model path, prediction restriction, and
overlap assembly were used.

### A — terminal only

The exact Phase-23 ordinal-3 terminal refinement was reused with
`clean_anchor = B_final`.

### B — periodic persistent

Ordinal 1 was declared as the midpoint before inference. Its exact Phase-23
refinement was reused and persisted:

```text
R_mid = H_refined_mid - B_mid
H_seed_final = B_final + R_mid
clean_anchor = H_seed_final
```

This is dimensionally valid and obeys the same CONST-flow resampling contract:
`H_seed_final` is the terminal denoised anchor, then mixed with the fixed
`sigma=0.25` noise. It adds no gain, projection, filter, or alternate transfer.

Only B's terminal batch was newly executed: 55 ordinary `64x64` local forwards.
A and midpoint stages were copied into resumable Phase-24 artifacts first.

## Results

| Metric | A terminal only | B periodic persistent |
|---|---:|---:|
| Semantic class | S3 | S3 |
| Gradient RMS | 0.187551 | 0.282925 |
| RMS vs terminal Blueprint | 0.119778 | 0.215070 |
| Low-frequency RMS vs Blueprint | 0.065425 | 0.113498 |
| Overlap RMS | 0.164391 | 0.172020 |
| Terminal CUDA | reused 54.869 s | 54.600 s |
| Terminal wall | reused 56.026 s | 56.125 s |
| Peak allocated/reserved | reused 3.19/3.59 GiB | 3.04/3.43 GiB |

Final A/B latent RMS is `0.103246`. The retained midpoint residual has RMS
`0.122346`; after terminal refinement its observable difference from A is
`0.103246` RMS.

The higher B gradient energy is not a quality win by itself. Direct inspection
shows no clear improvement in cables, train structure, tower definition, or
water detail. B also increases low-frequency departure from the Blueprint and
overlap disagreement. It remains coherent, but persistence supplies no clear
meaningful advantage.

## Integrity and decision

- Exact Phase-20c/23 Blueprint hashes were reused.
- Blueprint forwards executed now: zero; full-H forwards: zero.
- Newly executed local forwards: 55, all at `64x64`.
- B uses the exact same terminal noise hashes as A/Phase 23.
- No accepted Blueprint or H state was mutated or committed.
- Coverage is complete and all outputs are finite.
- A, midpoint, and B terminal stages are atomically persisted and resumable.

Select the simpler staged architecture:

```text
bounded Blueprint trajectory
-> one terminal native-local resampling pass
-> final destination H
```

Do not proceed to fully interleaved multiresolution sampling from this result.

Artifacts:

- `flux2_candidate3_blueprint_resampling_cadence.py`
- `flux2_candidate3_blueprint_resampling_cadence_results/report.json`
- `flux2_candidate3_blueprint_resampling_cadence_results/stages/`
- `flux2_candidate3_blueprint_resampling_cadence_results/A_B_COMPARISON.png`
- `flux2_candidate3_blueprint_resampling_cadence_results/A_B_DETAIL_REVIEW.jpg`
