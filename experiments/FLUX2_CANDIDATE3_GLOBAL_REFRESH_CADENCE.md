# Phase 6i — Candidate-3 global-refresh cadence discriminator

## Verdict

**REDUCED GLOBAL REFRESH CADENCE QUALIFIES FOR FURTHER TESTING**

`B_EARLY_3` qualifies. It computes fresh global estimates at intervals 0, 1,
and 2, then causally reuses interval 2's estimate at terminal interval 3.
Across all four scenes its returned final H is exactly equal to baseline:
final latent RMS, max-absolute error, and 4x4 low-frequency RMS are all zero.

No production change was made in Phase 6i.

## Exact reuse definition

The experiment caches the most recent global denoised/model estimate `x0_G`.
On a skipped interval it computes:

```text
G* = G_current
   + (sigma_next - sigma_current)
   * (G_current - stale_x0_G) / sigma_current
```

This is a well-defined, causal stale-estimator experiment. It does not reuse an
accepted `G_next`, freeze G, or alter H execution. G remains persistent and is
updated and atomically accepted on every interval. Every local crop prediction
is fresh at the current accepted H and sigma.

Telemetry records the cached estimate's source ordinal and age. Interval 0 is
always fresh.

## Matrix

| Policy | Fresh global ordinals | Reused estimate ordinals |
|---|---|---|
| A BASELINE | 0,1,2,3 | none |
| B EARLY 3 | 0,1,2 | 3 uses x0 from 2 |
| C EARLY 2 | 0,1 | 2 and 3 use x0 from 1 |
| D ALTERNATING | 0,2 | 1 uses 0; 3 uses 2 |
| E EARLY ONLY | 0 | 1,2,3 use x0 from 0 |

Scenes were person/car/tree, bridge/train, and a centered boundary-crossing
astronaut at H=64x128, plus person/car/tree at H=48x96 as the second-geometry
generalization pass. The current production window policy was used:
64x64/stride48 and 48x48/stride36, respectively.

## Lifecycle explanation for B

The terminal-release rule is:

```text
G_next = G*
H_next = H*
```

There is no terminal projection from G into H. Therefore interval 3's global
model prediction cannot affect returned H. B's intervals 0-2, including all
nonterminal projections and accepted H/G states, are bit-identical to A. At
interval 3, B changes only the final accepted G proposal and the diagnostic
projection that is computed but intentionally not applied. The local terminal
proposal and returned H remain exact.

This is not evidence that a stale global estimate faithfully represents the
terminal G state. B's final accepted-G RMS is 3.6-4.5% below baseline across the
four cases. It is evidence that terminal global work is causally dead for the
current returned-H lifecycle.

## Numerical results

| Case | Policy | Global forwards | Global CUDA | Local CUDA | Wall | Final RMS | Low RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| H64 person | A | 4 | 4.594 s | 11.923 s | 16.647 s | 0 | 0 |
| H64 person | B | 3 | 3.467 s | 11.998 s | 15.588 s | 0 | 0 |
| H64 bridge | A | 4 | 4.662 s | 12.053 s | 16.845 s | 0 | 0 |
| H64 bridge | B | 3 | 3.493 s | 12.092 s | 15.709 s | 0 | 0 |
| H64 astronaut | A | 4 | 4.675 s | 12.144 s | 16.947 s | 0 | 0 |
| H64 astronaut | B | 3 | 3.513 s | 12.170 s | 15.810 s | 0 | 0 |
| H48 person | A | 4 | 2.409 s | 6.358 s | 8.850 s | 0 | 0 |
| H48 person | B | 3 | 1.811 s | 6.362 s | 8.254 s | 0 | 0 |

Across all four cases B reduces mean global CUDA time 24.8% and mean sampling
wall time 6.6%. Per-case wall reduction is 6.4-6.8%. Peak allocated changes by
less than 0.4 MiB on average and peak reserved remains 5,104 MiB on average;
this is a runtime optimization, not a meaningful VRAM reduction.

## Projection behavior

B's applied nonterminal projection RMS values are exactly baseline because its
first three global estimates are fresh. Its terminal diagnostic projection
changes slightly but is not applied:

- H64 person: 0.31032 -> 0.31072;
- H64 bridge: 0.28500 -> 0.28993;
- H64 astronaut: 0.31916 -> 0.31959;
- H48 person: 0.24707 -> 0.25875.

Skipping a nonterminal refresh is materially different:

- C final latent RMS is 0.132-0.193 and low-frequency RMS 0.054-0.083.
  Interval-2 accepted H diverges, and terminal projection demand rises.
- D final RMS is 0.340-0.415 and low RMS 0.164-0.211. Its stale interval-1
  estimate immediately raises the second projection.
- E final RMS is 0.458-0.501 and low RMS 0.240-0.278. Projection demand grows
  repeatedly as its global estimate ages.

Thus stale G estimates lag the changing accepted trajectory when they are used
on nonterminal intervals. The hard projection enforces coarse equality but
must make larger corrections, and later local execution starts from the altered
accepted H.

## Semantic inspection

A and B decoded images are identical because their final H tensors are exact.
They therefore have identical object counts, anatomy, bridge/train continuity,
horizon/ground, overlap behavior, and local detail.

C remains broadly recognizable but is no longer numerically equivalent and
changes fine/coarse structure. D and E visibly reorganize the bridge tower and
cable geometry; repeated stale use creates additional/shifted suspension
structure and changes global proportions. These policies do not pass this
qualification.

Twenty decoded outputs and complete telemetry are under
`experiments/flux2_candidate3_global_refresh_cadence_results/`.

## Recommendation

Advance only the narrow policy “omit the terminal global forward under the
qualified terminal-release Euler lifecycle.” Do not generalize this to skipping
any nonterminal refresh, arbitrary samplers, or a periodic cadence. A future
production task should prove that omitting the terminal G proposal entirely—or
defining final G without a model call—preserves coordinator/API expectations;
Phase 6i deliberately retained atomic G acceptance using a stale estimate.

Production remains unchanged.
