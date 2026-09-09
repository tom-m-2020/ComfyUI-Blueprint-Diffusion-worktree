# Local Edit hard-source Klein discriminator

## Verdict

**Reject hard-source-only as a complete Local Edit policy.** The explicit
sampler passes S1, S4, and S5, but fails S2 and S3. Exact locked-latent authority
does not make ordinary FLUX.2 Klein T2I continue boundary geometry seamlessly.
The hard lock remains the invariant; this result authorizes only an
editable-side transition/boundary correction as the next experiment.

No production node, registration, Blueprint component, Candidate-3, Terminal
Resampling sampler, or existing production sampler was modified.

## Fixed setup

- Native ComfyUI FLUX.2 Klein 4B CONST sampling, batch one.
- One 768x512 source placed on the left of a 1024x512 canvas; the right 256
  pixels are editable.
- One binary latent mask with `1=editable`, `0=locked`.
- Four deterministic Euler intervals with strictly decreasing sigmas
  `[1.0, 0.96143687, 0.89259440, 0.73475969, 0.0]`.
- Ordinary text conditioning, full-canvas dense evaluation, no reference
  conditioning, ControlNet, inpaint model patch, sparse execution, K/V reuse,
  SpotEdit, Blueprint state, or RES4LYF runtime dependency.
- A two-latent-column transition band was derived entirely inside the editable
  mask and recorded, but never activated.

The experiment-only implementation is
`experiments/local_edit_hard_source_klein.py`; algebra/lifecycle tests are in
`experiments/test_local_edit_hard_source_klein.py`.

## Pre-inference tests

Ten tests passed before inference. They establish the exact source derivative,
Euler landing on `y_sigma_next`, restoration idempotence, editable-coordinate
bit identity with ordinary Euler, independence from locked `d_model`, the
`1=editable` convention, incoming-noise-mask rejection, fixed `z_L` reuse,
terminal equality with `y`, and transition-band containment.

## Controls

| Arm | Contract | Result |
| --- | --- | --- |
| A | Ordinary full-canvas T2I, no enforcement | Coherent new scene, but source replaced; locked decoded MAE `0.210761`. |
| B | Native ComfyUI binary noise mask | Same decoded result and metrics as C; terminal latent differs from C by at most `4.768e-7`. |
| C | Explicit derivative replacement plus accepted-state restoration | Exact locked trajectory, but visible boundary restart/seam. |

A frozen Clown epsilon arm was not added because reproducing the prior workflow
would introduce unrelated RES4LYF lifecycle and mask variables.

## Mechanical results

At every interval, both locked input-state error against `y_sigma` and accepted
locked-state error against `y_sigma_next` were exactly zero at recorded
precision. Terminal locked error against clean `y` was zero. Editable derivative
change was zero. The same locked-noise SHA-256 appeared at every interval.

| Step | sigma to sigma_next | editable d RMS | locked raw d RMS | replacement d RMS | replacement increment RMS | transition latent RMS |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000000 to 0.961437 | 1.221145 | 1.205664 | 1.286405 | 1.105521 | 0.973338 |
| 1 | 0.961437 to 0.892594 | 1.030512 | 1.070656 | 1.286405 | 0.968594 | 0.927542 |
| 2 | 0.892594 to 0.734760 | 1.135582 | 1.127933 | 1.286405 | 0.740817 | 0.833440 |
| 3 | 0.734760 to 0.000000 | 1.180676 | 1.158494 | 1.286405 | 0.643264 | 0.947624 |

The ordinary and hard-source arms have the same first raw prediction hash. Their
first intended divergence is therefore the locked derivative replacement after
that prediction. The independent C repeat is bit-exact.

## Gates

- **S1 exact source trajectory — PASS.** All recorded locked errors and the
  terminal locked error are zero.
- **S2 source fidelity — FAIL.** Identity and composition remain recognizable,
  but decoded locked MAE is `0.045182` versus the `0.011302` VAE
  self-reconstruction baseline; locked boundary-strip MAE is `0.098183`, and
  strict color/quality preservation is not met. The decoder mechanism causing
  the difference was not isolated here.
- **S3 geometric boundary continuity — FAIL.** The editable side restarts a
  smaller bridge with disconnected scale/position and an abrupt sky/water
  change across a hard vertical seam. The mismatch is clearly identifiable by
  accepted preview step 2 and fully visible in the final image.
- **S4 edit freedom — PASS.** Editable-region change RMS is `0.746809`, with
  substantial generated content.
- **S5 determinism — PASS.** The independent repeat is bit-exact.

## Interpretation and stop condition

This run answers the discriminator question negatively: even exact
source-authoritative CONST state at both required boundaries is insufficient
for seamless adjacent outpainting with ordinary Klein T2I. Control B also shows
that the explicit hard-source contract is not a meaningfully different baseline
from native binary noise masking for this deterministic Euler case.

Per the declared stop rule, the source lock is not weakened. The next eligible
experiment is a policy confined to the already-derived editable-side transition
band (or another explicitly editable-side correction) that makes generated
structure bend toward the immutable source.

Raw metrics, hashes, interval telemetry, tensors, decoded arms, and accepted-step
previews are under `experiments/local_edit_hard_source_klein_results/`.
