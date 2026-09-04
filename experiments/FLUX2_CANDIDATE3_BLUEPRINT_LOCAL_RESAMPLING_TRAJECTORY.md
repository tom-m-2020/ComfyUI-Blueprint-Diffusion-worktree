# Phase 23 — Persistent Blueprint to local-resampling trajectory

## Result

**PASS.** Fixed-late-sigma local resampling remains in the coherent Blueprint
scene basin at all four existing persistent Blueprint evaluations. Every mapped
Blueprint and every refined assembly is S3. No independent bridge, tower, train,
or lighthouse alternative appears, and the measurable detail increase is
repeatable rather than progressively destructive.

No production or ComfyUI-core code changed. No Phase-21 null decomposition,
Candidate-3 D/U projection, external K/V, filtering, anchor strength, or
prediction mixing was used.

## Trajectory contract

Phase 23 reuses the four exact, persisted Phase-20c Blueprint states and model
predictions. This avoids recomputing unrelated Blueprint work while preserving
the accepted `G_i` lineage and the complete existing four-interval schedule.

At each bounded evaluation `i`:

```text
persisted accepted G_i -> persisted ordinary x0_B,i
x0_B,i -> bilinear map 32x64 to destination 128x256
mapped crop 32x32 -> nearest2 native W anchor 64x64
W_0.25 = 0.75 * anchor + 0.25 * epsilon_region
ordinary FLUX x0_W -> 2x2 mean restriction -> normalized assembly
```

The local resampling state is deliberately an independent fixed-sigma
refinement of the denoised Blueprint estimate. It is not claimed to be the
accepted Blueprint state at `sigma_i`, so literal `sigma=0.25` remains a valid
CONST-flow reconstruction:

```text
x_sigma = (1-sigma) * x0 + sigma * epsilon
```

The Phase-22 region-noise convention is unchanged:
`seed = 20260901 + 22,000,003 + 1,009*region.index`. The same 55 tensors and
hashes are reused at every Blueprint evaluation, isolating temporal Blueprint
changes from noise changes.

The assembled outputs are diagnostic H estimates only. No accepted H or G is
updated by this zero-update experiment.

## Measurements

| i | Blueprint sigma -> next | Blueprint class | Refined class | RMS refined vs B | low-frequency RMS | gradient B/refined | overlap RMS | local CUDA | wall | peak alloc/reserved |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 -> 0.999177 | S3 | S3 | 0.123023 | 0.074190 | 0.123295 / 0.175752 | 0.161195 | 55.048 s | 56.415 s | 3.04 / 3.44 GiB |
| 1 | 0.999177 -> 0.997536 | S3 | S3 | 0.122346 | 0.072733 | 0.125528 / 0.177744 | 0.161623 | 54.450 s | 55.581 s | 3.19 / 3.60 GiB |
| 2 | 0.997536 -> 0.992643 | S3 | S3 | 0.121369 | 0.069615 | 0.130564 / 0.182754 | 0.163008 | 54.812 s | 55.952 s | 3.19 / 3.59 GiB |
| 3 | 0.992643 -> 0 | S3 | S3 | 0.119777 | 0.065425 | 0.134212 / 0.187551 | 0.164391 | 54.869 s | 56.026 s | 3.19 / 3.59 GiB |

New work was 220 ordinary local calls. The four exact Blueprint predictions
were reused, corresponding to one already-completed Blueprint forward per
evaluation. New local CUDA totaled 219.18 s and model-run wall time totaled
223.97 s. Peak allocation/reservation was 3.19/3.60 GiB.

Gradient RMS is 40–43% above each current Blueprint estimate. Meanwhile,
Blueprint-to-refined RMS and low-frequency discrepancy both fall slightly over
the sequence. This is classified as **stable regeneration**: the detail gain is
present at every evaluation without accumulating scene drift.

## Semantic review

At every interval both A and B retain:

- one dominant continuous suspension bridge;
- one centered train;
- a controlled left bridge tower and right stone endpoint;
- coherent horizon and water;
- no field of local bridge scenes or duplicated trains.

The refined images remain somewhat soft, so Phase 23 does not establish final
detail quality or an optimal refinement cadence. It establishes the narrower
claim that ordinary native FLUX refinement initialized from successive coherent
Blueprint estimates does not fall back into the S0 tiled semantic basin.

The requested pass gate is satisfied. Stop here. The next decision should
compare when this already-qualified local refinement occurs—terminal-only,
periodic, or fully interleaved multiresolution sampling—without reopening
Blueprint initialization or tuning the fixed resampling sigma.

## Integrity and artifacts

- Exact Phase-20c configuration and accepted-Blueprint lineage recorded.
- Every completed interval was atomically persisted and is resumable.
- Blueprint artifacts and all W inputs remained hash-immutable.
- Same deterministic region noise at every interval.
- Complete positive coverage and finite predictions throughout.
- Accepted-state updates: zero.

Artifacts:

- `flux2_candidate3_blueprint_local_resampling_trajectory.py`
- `flux2_candidate3_blueprint_local_resampling_trajectory_results/report.json`
- `flux2_candidate3_blueprint_local_resampling_trajectory_results/intervals/`
- `flux2_candidate3_blueprint_local_resampling_trajectory_results/BLUEPRINT_RESAMPLING_TRAJECTORY.png`

