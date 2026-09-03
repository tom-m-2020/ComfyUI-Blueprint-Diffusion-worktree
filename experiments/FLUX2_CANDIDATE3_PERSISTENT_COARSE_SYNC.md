# Phase 10b — persistent native-scale W with accepted-state coarse synchronization

## Verdict

Accepted-state coarse synchronization prevents the catastrophic independent-W
divergence, preserves W's null-space component exactly to numerical tolerance,
and returns the trajectory to the reconstructed control's semantic class. It
does not produce materially better local detail or scene organization than
reconstructing W.

**PERSISTENCE COLLAPSES TOWARD THE RECONSTRUCTED CONTROL**

This is decision gate 2. Production and ComfyUI core are unchanged.

## Fixed controls

All variants use the exact Phase-10 configuration:

- native FLUX.2 Klein 4B, bridge/train prompt, seed `20260901`, CFG 1;
- four-step CONST-flow Euler schedule;
- destination `H=64x128`, fixed `G=24x48`;
- fifteen `32x32` destination regions at stride 24;
- `64x64` native-coordinate W states;
- identical initial H/G/noise, initial sigma-consistent W lift, global calls,
  local model execution, restriction, overlap assembly, Candidate-3 H/G
  coupling, and terminal H release;
- no external K/V or other context.

A reconstructs W from accepted H for every model evaluation. B initializes W
once and independently Euler-updates it. C also initializes W once and forms
the same ordinary W Euler proposal, then synchronizes only its coarse component
after Candidate-3 H has been accepted:

```text
correction = crop(H_next) - D(W_star)
W_next = W_star + U(correction)
```

Here D is nonoverlapping 2x2 mean and U is 2x nearest. Therefore
`D(U(correction))=correction` exactly up to float rounding.

## State and lifecycle integrity

For B and C, fifteen W states are constructed exactly once, evaluation
reconstruction count is zero, each W receives exactly four Euler proposals,
and each accepted W hash is the next interval's parent hash. All crop calls
read immutable accepted W; H/G/W candidates validate before their single
logical publication; and all nonterminal Candidate-3 H/G invariants pass.

C's post-synchronization `D(W_next)-crop(H_next)` maximum is at most
`4.768e-7` at every interval.

The terminal H result does not depend on terminal W handling. Both unmodified
terminal W_star and synchronized terminal W are saved and decoded for all
regions. The synchronized diagnostic satisfies the same coarse invariant; the
unmodified diagnostic retains its measured pre-sync drift.

## Prediction compatibility

| Interval | A reconstructed | B independent | C coarse-sync |
|---:|---:|---:|---:|
| 0 | 0.913870 | 0.913870 | 0.913870 |
| 1 | 0.653408 | 0.817340 | 0.658321 |
| 2 | 0.491292 | 0.842901 | 0.540675 |
| 3 | 0.259399 | 0.803052 | 0.356545 |

C tracks A much more closely than B. It remains measurably less compatible
than A, especially at the terminal prediction.

## Synchronization magnitude and null-space preservation

| Interval | Pre-sync drift RMS mean/max | Correction/W_star RMS mean/max | Post-sync max abs | Detail RMS before/after |
|---:|---:|---:|---:|---:|
| 0 | 0.04795 / 0.05260 | 0.03754 / 0.04122 | 4.77e-7 | 0.83556 / 0.83556 |
| 1 | 0.06040 / 0.06556 | 0.04997 / 0.05448 | 4.77e-7 | 0.78700 / 0.78700 |
| 2 | 0.13140 / 0.13839 | 0.12055 / 0.12884 | 4.77e-7 | 0.69302 / 0.69302 |
| 3 | 0.19386 / 0.22557 | 0.17177 / 0.19778 | 4.77e-7 | 0.63648 / 0.63648 |

The synchronization correction becomes substantial late in the trajectory.
Nevertheless, it changes only the lifted coarse component: maximum null-space
detail RMS change is `5.87e-8`, with maximum absolute change `7.15e-7`.
Thus any absence of useful persistent benefit cannot be attributed to the
reconciliation directly erasing the stored W detail component.

## Pairwise trajectory differences

A and C share bit-exact accepted H through state 1. Their accepted-H RMS
differences then grow to `0.029674`, `0.095666`, and `0.411596` for states
2–4. Their assembled x0_H RMS differences are `0`, `0.388837`, `0.449020`, and
`0.411596` across intervals 0–3.

B and C remain distinct despite shared initial W: final accepted-H RMS is
`0.720092`. Coarse synchronization therefore causally controls the independent
trajectory rather than merely producing the same B result.

## Semantic inspection

- A retains one principal horizontal bridge but includes the known repeated
  train/support alternatives and terminal artifacts.
- B breaks into several independent bridge scenes, repeated towers, and
  incompatible horizon/water regions.
- C removes B's catastrophic multi-scene layout and returns to a principal
  bridge and horizon similar to A. It still exhibits repeated train/support
  alternatives and does not show a clearly useful sharpness/detail advantage
  over A. Some residual regional artifacts remain stronger than in A, matching
  its higher terminal overlap RMS.

C therefore does not meet gate 1. It also does not meet gate 3: exact coarse
agreement is sufficient to prevent B's independent whole-scene alternatives.
The result is gate 2—once coarse synchronization is enforced, persistence adds
state and lifecycle complexity without demonstrated image benefit over
reconstruction.

## Work and memory

| Variant | Global/local forwards | Local tokens | Global CUDA | Local CUDA | Wall | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|
| A | 3 / 60 | 245,760 | 1.608 s | 61.846 s | 65.252 s | 2.903 / 3.291 GiB |
| B | 3 / 60 | 245,760 | 0.968 s | 61.981 s | 65.853 s | 2.939 / 3.299 GiB |
| C | 3 / 60 | 245,760 | 1.006 s | 59.657 s | 63.826 s | 2.949 / 3.301 GiB |

The small timing variation is not treated as an optimization result. Token and
forward work are identical.

All accepted-H trajectories, representative W trajectories, every restricted
D(W) view, both terminal-W diagnostics, lineage, pairwise tensors, and raw
telemetry are under `flux2_candidate3_persistent_coarse_sync_results/`.

## Decision

Do not advance persistent coarse-synchronized W to production or a broader
trajectory benchmark. Under this fixed training-free construction it does not
demonstrate useful information beyond the simpler reconstructed-W control.
