# Local Edit Source-Stitch Trajectory

## Result

**Classification: SIMPLIFIED SOURCE-STITCH MECHANISM INSUFFICIENT.**

Stitching an exact source-consistent latent after interval 0 causally changes
the next editable prediction. Repeating the stitch keeps source coordinates on
the exact CONST trajectory, but does not make the editable bridge continue the
immutable source bridge. It reproduces the characteristic hard-lock failure:
independently organized spans and cables restart at the source boundaries.

The branch stops here without mask, strength, width, or schedule tuning.

## Fixed experiment

- Case: existing `rigid_bridge`, 1024x512 canvas.
- Mask: latent columns `16:48` are source; both exterior sides are editable.
- Model: FLUX.2 Klein 4B under native `CONST` sampling.
- Schedule: fixed eight-step Flux2 schedule ending at zero.
- Sampler: deterministic Euler, CFG 1, seed 0.
- Model execution: ordinary dense full-canvas prediction at every interval.
- No Clown Guide, epsilon/projection, latent noise mask, blur, attention patch,
  K/V injection, compositing, or production code.

The fixed source trajectory was

```text
y_sigma = (1 - sigma) * y + sigma * z_L
```

where `z_L` is the single fixed initial noise tensor shared by all arms and
reused at every interval.

## Arms

- `A_ORDINARY`: ordinary proposal accepted at all intervals.
- `B_SINGLE_STITCH`: ordinary interval-0 proposal, then exact source-coordinate
  replacement with `y_sigma_next`; all later proposals accepted ordinarily.
- `C_REPEATED_STITCH`: the same source-coordinate replacement after every
  Euler proposal.

Editable proposal coordinates were never changed by stitching.

## Mechanical invariants

All invariants passed:

- all arms used the same fixed noise and identical first model input;
- all arms produced bit-exact first raw x0;
- B stitched only accepted interval 0;
- C stitched every accepted interval;
- every enabled stitch had zero source-coordinate RMS and max error against
  `y_sigma_next`;
- editable proposal-to-accepted max difference was exactly zero in every arm;
- no incoming noise mask or model/attention patch was present.

Every interval persisted input state, raw model x0, Euler proposal,
source-consistent next state, accepted state, and the following raw x0 where one
exists.

## Causal next-prediction result

B and C are identical through the model evaluation immediately following the
shared interval-0 stitch. At that evaluation:

| Measurement versus Ordinary | B_SINGLE_STITCH | C_REPEATED_STITCH |
|---|---:|---:|
| Editable raw-x0 RMS | 0.440 | 0.440 |
| Source raw-x0 RMS | 0.470 | 0.470 |
| Editable accepted-state RMS | 0.010 | 0.010 |

Thus the stitch does exactly what the causal hypothesis predicts at the narrow
mechanical level: a source-only accepted-state change enters the next dense
model call and changes the editable raw prediction without direct editable
state modification.

This numerical response is not by itself evidence of compatible geometry.

## Persistence

Single-stitch editable raw-x0 RMS versus Ordinary remains about `0.51–0.56`
through subsequent evaluations. Repeated stitching increases the difference to
about `0.57–0.67`. Repeated stitching maintains zero source-state trajectory
error at every acceptance; Single Stitch drifts after interval 0 and reaches
`1.22` source RMS error at the terminal acceptance, close to Ordinary's `1.26`.

One stitch therefore changes the later trajectory but does not preserve source
authority. Repeated stitching is required for exact source ownership, but exact
ownership does not imply compatible editable composition.

## Rigid-bridge semantic result

The final decoded outputs answer the primary gate:

- Ordinary produces its own coherent full-canvas bridge composition but does
  not preserve the supplied source.
- Single Stitch remains visually close to an unconstrained coherent generation;
  its one-time source influence is not retained as source authority.
- Repeated Stitch preserves the center source trajectory but generates separate
  left/right bridge organizations. Cable slopes, tower scale, and bridge spans
  restart at both vertical boundaries, with conspicuous discontinuities.

The interval-1 raw-x0 decode already proves that source stitching can reorganize
editable geometry. The final repeated-stitch decode proves that the induced
change is not the required continuation of the same bridge.

## Decision

The causal chain

```text
source-only accepted-state stitch
-> next full-canvas prediction changes in editable coordinates
```

is confirmed. The stronger algorithmic hypothesis

```text
repeat exact source stitching
-> geometrically compatible outpaint
```

is falsified for the fixed rigid-bridge discriminator. Exact source authority
prevents co-evolution of the source side needed for a single negotiated scene;
the editable regions independently compose around the immutable center state.

No production mechanism is retained from this branch.

Artifacts:

- `experiments/local_edit_source_stitch_trajectory.py`
- `experiments/test_local_edit_source_stitch_trajectory.py`
- `experiments/local_edit_source_stitch_trajectory_results/report.json`
- `experiments/local_edit_source_stitch_trajectory_results/runtime_tensors.pt`
- `experiments/local_edit_source_stitch_trajectory_results/RIGID_BRIDGE_STITCH_FINAL_COMPARISON.png`
- `experiments/local_edit_source_stitch_trajectory_results/RIGID_BRIDGE_STITCH_RAW_X0_TRAJECTORY.png`
