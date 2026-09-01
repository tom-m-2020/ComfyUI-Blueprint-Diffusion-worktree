# Phase 7a — Candidate-3 step-count generalization

Date: 2026-09-01

## Scope and method

This experiment tested 4, 8, 12, and 20 full-denoise FLUX.2 Klein CONST-flow
Euler schedules without modifying production. Every schedule began at sigma
exactly 1, ended at exactly 0, and was positive and strictly decreasing before
the terminal value.

The experiment invoked the current production coordinator and its current
geometry-specific crop planner directly so only production's four-interval
front-door validation was bypassed. The executed algorithm was unchanged:

- every nonterminal interval used a fresh global prediction, all fresh local
  crops, independent Euler proposals, hard block-DCT coupling, and atomic G/H
  acceptance;
- the terminal interval omitted global execution, accepted fresh local `H*`,
  and retained prior G as unsynchronized diagnostic state.

The four fixed cases were H=64x128 person/car/tree, bridge/train, and centered
boundary-crossing astronaut, plus H=48x96 person/car/tree. Ordinary dense
FLUX.2 was run at every matching step count as a semantic reference. The full
machine-readable telemetry is in
`flux2_candidate3_step_count_generalization_results/report.json`; the harness
is `flux2_candidate3_step_count_generalization.py`.

## Lifecycle and correctness

All 16 Blueprint trajectories passed:

- finite final output and complete normalized crop coverage;
- one atomic accepted state per interval;
- exactly `steps - 1` global forwards;
- exactly `3 * steps` local forwards at both qualified geometries;
- a global forward on every nonterminal interval and only the final interval
  omitting it;
- terminal release only on the final interval;
- every nonterminal `D(H)=G` error at or below `9.54e-7`.

| Steps | Global forwards | Local forwards | Total Blueprint forwards |
| ---: | ---: | ---: | ---: |
| 4 | 3 | 12 | 15 |
| 8 | 7 | 24 | 31 |
| 12 | 11 | 36 | 47 |
| 20 | 19 | 60 | 79 |

Each interval records global and assembled-local prediction norms, overlap
disagreement, H* norm, projection RMS/ratio, invariant error, CUDA timing, and
terminal/global-execution status in the JSON report.

## Repeated projection behavior

Projection demand does not grow systematically with step count. Smaller Euler
increments produce smaller average corrections even though there are more of
them:

| Steps | Mean projection/H* ratio across scenes | Worst interval ratio | Mean terminal overlap RMS |
| ---: | ---: | ---: | ---: |
| 4 | 6.59% | 12.91% | 0.158 |
| 8 | 4.38% | 12.23% | 0.097 |
| 12 | 3.36% | 11.97% | 0.066 |
| 20 | 2.27% | 9.84% | 0.040 |

Within each trajectory, projection magnitude generally rises toward the last
nonterminal interval as sigma falls. That is also true at four steps. Across
trajectory lengths, however, mean ratio declines monotonically and the worst
ratio does not increase. There is no evidence of cumulative projection
instability or progressively stronger overconstraint through 20 steps.

Final latent RMS increases moderately with step count in all scenes, as does
the corresponding dense trajectory's development; outputs remain finite and
visually well formed. This is not accompanied by blur, projection ghosting, or
boundary failure.

## Semantic inspection

The report directory contains every decoded dense/Blueprint output and one
two-row contact sheet per scene.

- **H64 person/car/tree:** Blueprint retains exactly one person, one left red
  car, and one right tree at 4/8/12/20 steps. Ground and horizon stay
  continuous; later schedules add detail without anatomy splitting or seams.
- **H64 bridge/train:** one continuous bridge, centered yellow train, left
  lighthouse, and right dark stone structure persist. Deck, cables, water, and
  horizon remain continuous through 20 steps. Greater stone articulation at
  later steps is one connected structure, not a new crop-local alternative.
- **H64 centered astronaut:** one astronaut continues to cross the central
  crop boundary cleanly, with one rover left and one antenna right. No doubled
  torso/limbs, displacement, or ground seam appears.
- **H48 person/car/tree:** object uniqueness and left/center/right placement
  persist through 20 steps. Local texture and subject detail improve without
  visible overlap boundaries.

Dense references also remain stable and develop more detail at longer
schedules. Blueprint is not numerically compared to dense because each is a
different trajectory; dense images serve only to confirm that the longer
schedule itself produces an ordinary coherent reference.

## Timing and memory

Warm sampling-only wall times were:

| Case | Variant | 4 | 8 | 12 | 20 |
| --- | --- | ---: | ---: | ---: | ---: |
| H64 person/car/tree | Dense | 10.51 s | 21.14 s | 31.92 s | 53.22 s |
|  | Blueprint | 15.39 s | 32.05 s | 48.93 s | 82.72 s |
| H64 bridge/train | Dense | 10.70 s | 21.39 s | 32.08 s | 53.51 s |
|  | Blueprint | 15.62 s | 32.42 s | 49.22 s | 82.79 s |
| H64 astronaut | Dense | 10.71 s | 21.44 s | 32.15 s | 53.58 s |
|  | Blueprint | 15.62 s | 32.40 s | 49.22 s | 82.96 s |
| H48 person/car/tree | Dense | 4.68 s | 9.35 s | 14.03 s | 23.41 s |
|  | Blueprint | 8.20 s | 17.09 s | 26.06 s | 43.92 s |

Runtime scales approximately with the explicitly increasing forward count.
Peak CUDA allocation/reservation does not grow with step count: Blueprint
remained at 2.885/4.156 GiB for H64 and 2.668/3.402 GiB for H48 in this process.
Longer schedules increase total work and wall time, not the per-forward peak.

## Conclusion

The tested Candidate-3 lifecycle is not intrinsically tied to four Euler
intervals. Repeated exact coarse coupling remains numerically bounded and
semantically stable through 20 steps across the required scenes and both
qualified production geometries.

A separate production task should generalize `validate_schedule()` and loop
cardinality while preserving the existing full-denoise CONST-flow,
terminal-only global omission, and fail-closed model/CFG/conditioning rules.

```text
STEP-COUNT GENERALIZATION QUALIFIES
```
