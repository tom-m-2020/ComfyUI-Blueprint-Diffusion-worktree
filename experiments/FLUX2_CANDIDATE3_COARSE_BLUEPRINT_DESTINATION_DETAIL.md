# Phase 21 — Exact coarse-Blueprint / destination-detail decomposition

## Result

**FAIL.** The fixed decomposition preserves its mathematical null space exactly
and produces a substantially sharper result than the Phase-20c hard anchor, but
the retained component contains incompatible scene geometry. The terminal image
has one dominant bridge (S2), yet faint repeated towers, bridge fragments, and
ghost cables survive across the sky and water. It therefore does not satisfy the
required detail-preserving semantic gate.

No production or ComfyUI-core files changed.

## Fixed operator

The experiment selected the smallest deterministic spatial reduction that has
an exact right inverse while leaving a nonempty destination detail subspace:

```text
C(x)[c,y,x] = (1/4) sum_{dy,dx in {0,1}} x[c,2y+dy,2x+dx]
P(z)[c,2y+dy,2x+dx] = z[c,y,x]
```

Thus `C : 128x256 -> 64x128`, `P` is 2x nearest replication, and
`C(P(z)) = z` exactly in real arithmetic. The corrected prediction was:

```text
x0_H_C = L + P(C(B_H) - C(L))
N_L = L - P(C(L))
```

This pair was fixed before model execution. No strength, filter, frequency,
context, representation, or resampling variant was tested.

## Execution

The run reused the Phase-20c persistent `32x64` native-coordinate Blueprint/H
trajectory, four CONST-flow Euler intervals, prompt, seed, 55 normalized W
regions, model calls, mapping, and assembly. Each interval performed one
Blueprint and 55 ordinary W forwards. Ordinary local predictions were
restricted and assembled first; the correction was applied only in destination
prediction space. Blueprint and H proposals were committed atomically.

All four interval artifacts were persisted before the next interval. The run
used 4 Blueprint and 220 local model calls. Total model-run wall time was
244.41 s; Blueprint CUDA was 2.95 s and local CUDA was 236.65 s. Peak CUDA
allocation/reservation was 1.82/2.28 GiB. VAE diagnostics used the normal tiled
fallback after full-frame decode exceeded available memory.

## Numerical evidence

| Interval | coarse max abs | retained-null max abs | null RMS | ordinary overlap RMS | corrected vs hard RMS | corrected gradient RMS | hard gradient RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 7.15e-7 | 7.75e-7 | 0.433656 | 0.956372 | 0.443400 | 0.640398 | 0.123295 |
| 1 | 4.77e-7 | 8.34e-7 | 0.432636 | 0.945148 | 0.442621 | 0.639545 | 0.125528 |
| 2 | 4.77e-7 | 7.15e-7 | 0.434034 | 0.924782 | 0.444703 | 0.643143 | 0.130564 |
| 3 | 4.77e-7 | 5.96e-7 | 0.430671 | 0.855107 | 0.442094 | 0.640282 | 0.134212 |

`C(x0_H_C)=C(B_H)` and null/detail preservation both hold below the declared
`1e-5` tolerance in every interval. The terminal accepted H equals the corrected
terminal prediction under Euler-to-zero. Its gradient RMS is about 4.77x the
Phase-20c hard-anchor result, establishing that substantial local structure was
retained numerically and visibly.

The ordinary overlap value is reported as a diagnostic of the uncorrected crop
ensemble; correction was applied after assembly, so no misleading zero-overlap
claim is made for the destination-space result.

## Semantic review

- Blueprint x0: S3 in all intervals; one bridge, centered train, stable horizon.
- Ordinary local assembly: S0 in all intervals; many independent bridge scenes.
- Coarse-corrected prediction/accepted H: S2 in all intervals; one dominant
  bridge and broadly continuous horizon/water, but persistent faint alternative
  towers, doubled cables, and structural ghosts.
- Compared with Phase 20c, the terminal result is materially sharper, but the
  recovered structure is partly incompatible geometry rather than harmless
  texture or fine detail.

The exact algebra therefore succeeds while the semantic decomposition fails:
the spatial 2x2 null space is not a semantic-detail subspace for this model and
scene. The next justified direction is model-mediated refinement/resampling
under Blueprint authority, not another spatial filter, scale, or anchor sweep.

## Artifacts

- `flux2_candidate3_coarse_blueprint_destination_detail.py`
- `flux2_candidate3_coarse_blueprint_destination_detail_results/report.json`
- `flux2_candidate3_coarse_blueprint_destination_detail_results/intervals/`
- `flux2_candidate3_coarse_blueprint_destination_detail_results/TRAJECTORY_COMPARISON.png`
- `flux2_candidate3_coarse_blueprint_destination_detail_results/TERMINAL_REVIEW.jpg`

