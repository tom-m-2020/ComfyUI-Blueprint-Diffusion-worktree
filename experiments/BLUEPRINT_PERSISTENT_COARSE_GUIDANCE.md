# Persistent coarse guidance during local W refinement

## Question

Can a completed, mapped terminal Blueprint remain an explicit coarse authority
during a short local W trajectory, permitting more native local development
than one-step Terminal Resampling without losing its S3 scene composition?

This is experiment-only.  Candidate-3, `BlueprintTerminalResampling`, and
ComfyUI core remain unchanged.

## RES4LYF / Clown Guide audit

RES4LYF commit `0d753fada0cd5ae1dd69372caee9c3e7012a5dcb` was inspected.
`ClownGuide_Beta` builds a guide configuration; the actual intervention is in
`beta/rk_guide_func_beta.py`.  For a linear/flow RK method, its epsilon guide is

```text
epsilon_guide = (x - guide) / sigma
```

and ordinary `epsilon` mode linearly moves the sampler epsilon toward that
guide epsilon using the scheduled guide weight.  Projection, channelwise,
pseudoimplicit, style, attention-injection, sync, and flow-guide variants are
separate mechanisms and are not reproduced here.

For ComfyUI Klein/CONST, with model denoised prediction `m`:

```text
v_model = (x - m) / sigma
v_guide = (x - guide) / sigma
v_used  = v_model + lambda * (v_guide - v_model)
        = (x - ((1-lambda)*m + lambda*guide)) / sigma
```

Therefore the smallest clean reproduction is a **sampler-level velocity /
post-model denoised-prediction intervention**:

```text
m_guided = (1-lambda) * m + lambda * guide
x_next   = x + (sigma_next-sigma) * (x-m_guided)/sigma
```

It is not:

- initial-state modification: every arm starts from the same exact
  `noise_scaling(0.25, epsilon_region, guide_region)` W;
- model/conditioning guidance: the model sees the same W, sigma, text
  conditioning, coordinates, and model options and produces the same `m` at a
  given accepted input;
- hidden/attention guidance: no transformer feature or K/V is changed;
- a second model call: the guide is a fixed tensor and adds no inference.

The intervention occurs after each ordinary model prediction and before the
Euler update.  It changes later accepted W states, so later model predictions
naturally diverge; telemetry records pre-guide model predictions, guide deltas,
weights, and accepted-state hashes.

Primary reference:
https://github.com/ClownsharkBatwing/RES4LYF/blob/0d753fada0cd5ae1dd69372caee9c3e7012a5dcb/beta/rk_guide_func_beta.py

## Fixed discriminator

- Reuse the qualified Phase-29 square multi-object terminal Blueprint,
  `G=45x45`, `H=128x128`, 25 overlapping `32x32` footprints, and W=`64x64`.
- Reuse every Phase-29 region seed/noise and exact initial W hash.
- Reuse Phase-38's declared four-interval W schedule:
  `[0.25, 0.1986604057521303, 0.14082231971176423,
  0.07516852136507746, 0]`.
- Reuse the frozen Phase-29 one-step Terminal result and Phase-38 unguided
  four-interval result without recomputation.
- Run exactly two new arms:
  - constant: `lambda=[0.25,0.25,0.25,0.25]`;
  - release: `lambda=[0.5,1/3,1/6,0]`.

Both schedules have total discrete weight `1.0`; only temporal placement
differs.  The release arm reaches zero on the terminal interval.  These are
explicit empirical discriminator policies, not trained or canonical Klein
strengths.

For each region, `guide_r` is the same mapped terminal `x0_G` crop used for
initialization, lifted once to W with the qualified nearest transform.  It is
immutable and spatially registered for all four intervals.  Restriction and
normalized overlap assembly occur once after the W trajectory.

## Gates

All arms must retain exact initial W/noise provenance, four ordinary calls per
region, no destination-sized model call, positive complete coverage, finite
state, deterministic planning, immutable guide tensors, and flat post-region
allocation.  A guided arm succeeds only if visual review finds credible car,
tree, house, or ground structure beyond both frozen Terminal and unguided
four-step controls while retaining S3 composition and avoiding cross-region
disagreement.  Gradient energy alone is not detail.

If neither fixed guide policy improves that frontier, stop guide-strength and
guide-schedule tuning and advance only to a separately authorized recurrent
G/W refresh discriminator.

