# Local Edit FLUX.2 Klein Epsilon Mechanism Discriminator

Date: 2026-09-09

## Verdict

The successful Clown epsilon mechanism is **indirect editable guidance through
source-region co-evolution**.

In the exact reproduced no-noise-mask workflow, both plain epsilon and
`epsilon_projection` change only source-region derivative coordinates. Their
editable correction is exactly zero at every interval. The changed source state
is accepted without restoration, then participates in the next dense
full-canvas Klein call; editable raw x0 first diverges at interval 1. That later
model-mediated coupling is what the rejected hard-lock mechanisms remove.

Arm D, which retains C's exact correction only on editable coordinates, is
bit-exact to ordinary T2I at every accepted interval in all three cases and
loses the epsilon bridge behavior. No second ablation is needed.

## Scope and controls

The fixed 8-step Flux2Scheduler schedule, deterministic RES4LYF linear/Euler,
CFG 1, model, prompts, seed, source geometry, direct Clown guide mask, weight 1,
cutoff 1, and constant full-window guidance were preserved. There is no native
noise mask, hard restoration, K/V injection, compositing, or Blueprint path.

- A: ordinary full-canvas T2I, no guide.
- B: exact Clown plain `epsilon`.
- C: exact Clown non-channelwise `epsilon_projection`.
- D: C's exact post-projection correction, retained only at editable
  coordinates; source derivative remains ordinary.

## Exact mechanism and ordering

For the primary Euler row, RES4LYF receives Comfy/Klein's raw denoised estimate
`data_row` and sampler derivative `eps_row`. It constructs
`eps_substep_guide` with `RK.get_guide_epsilon(...)`. Plain epsilon performs:

```text
eps_plain = eps_row + lgw_mask * (eps_substep_guide - eps_row)
```

Non-channelwise projection first constructs the full-tensor candidate:

```text
eps_lerp = eps_row
         + mask * (eps_substep_guide - eps_row)
         + (1-mask) * (0 - eps_row)

eps_sum = collinear(eps_row, eps_lerp)
        + orthogonal(eps_lerp, eps_row)

eps_projection = eps_row + lgw_mask * (eps_sum - eps_row)
```

The collinear/orthogonal coefficients are computed over the flattened batch
tensor, so projection is globally coupled. The final `lgw_mask`, however, is
the source-region guide-weight mask. Consequently the accepted correction has
zero editable support even though its source correction depends on a
whole-tensor projection calculation.

Observed lifecycle:

```text
accepted x_sigma
-> Klein/CFG raw x0 and raw CONST derivative
-> exact Clown guide epsilon
-> plain or globally projected source correction
-> guided derivative
-> Euler proposal
-> proposal accepted unchanged
-> next dense full-canvas model call
```

No post-proposal projection exists in these arms, so the entire source-region
Euler increment survives into accepted state.

## Causal evidence

At rigid-bridge interval 0, A/B/C raw model x0 are identical because guidance
is post-prediction. The first proposal differs immediately:

| Arm | Correction RMS | Accepted-state RMS vs A |
|---|---:|---:|
| B plain epsilon | 0.815194 | 0.014144 |
| C epsilon projection | 1.194355 | 0.020723 |
| D editable-only | 0 | 0 |

For B, `abs(sigma_next-sigma) * correction_RMS = 0.014144`, exactly accounting
for the accepted-state divergence. At interval 1, after the changed source has
been visible to a new full-canvas model call, editable-inclusive raw-x0 RMS
versus A becomes `0.495167` for B and `0.641312` for C. D remains exactly zero
through all eight intervals.

Rigid-bridge A-relative progression:

| Interval | B raw-x0 RMS | C raw-x0 RMS | B accepted RMS | C accepted RMS |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.014144 | 0.020723 |
| 1 | 0.495167 | 0.641312 | 0.032380 | 0.047769 |
| 2 | 0.851778 | 0.895307 | 0.060562 | 0.082419 |
| 4 | 0.934838 | 1.001884 | 0.167995 | 0.209087 |
| 7 | 1.055040 | 1.104064 | 1.143133 | 1.199649 |

Thus geometry is already causally separated at the interval-0 accepted state,
and visibly differentiated in raw x0 on interval 1. The decoded interval-0 raw
x0 sheets are identical, as required by the hook location.

## What projection changes

Projection changes source-correction magnitude and direction, not its final
spatial support. On the bridge, whole-canvas correction RMS at interval 0 rises
from `0.815194` plain to `1.194355` projected. Since the initial raw prediction
is shared, the plain/projected correction cosine is `0.997060` at interval 0:
initially mostly a magnitude change. On later, already-diverged trajectories,
the measured correction cosine ranges approximately `0.75-0.89`; projection
then materially changes direction as well as magnitude.

Projection is globally coupled in how it computes its candidate, but its final
correction remains spatially source-only. It does not directly guide editable
pixels.

## Visual mechanism

- A and D are identical and form a clean but newly composed bridge unrelated to
  preservation of the supplied center source.
- B lets the center source content evolve toward the guide and produces broad
  single-scene continuity, but retains more duplication and larger seams.
- C changes the co-evolving source trajectory more strongly/directionally and
  reproduces the established coherent one-bridge reference most closely.

Exact source-state ownership fails because it prevents this negotiation: every
new call sees an immutable source trajectory while the editable solution forms
independently. Epsilon instead permits the source region to move under
attraction, and native dense attention then propagates that negotiated geometry
into editable predictions on subsequent calls. The cost is source drift; the
mechanism does not solve exact preservation by itself.

## Artifacts and stop boundary

Per-interval tensors contain accepted state, raw x0, raw derivative, guide
quantity, pre-projection correction, projected correction, final derivative,
Euler proposal, and accepted next state. JSON statistics split source,
editable, two-column boundary strips, and whole canvas. Raw-x0 and accepted-state
decodes are saved at every bridge interval.

Artifacts are under `experiments/local_edit_epsilon_mechanism_results/`. The
causal question is established; no tuning, second ablation, or production change
was performed.
