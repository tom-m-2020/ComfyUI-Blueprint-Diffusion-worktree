# Local Edit Step-0 Clean-Source Context Diagnostic

Date: 2026-09-09

## Verdict

**Fail the rigid-bridge expected-direction gate.** Clean-source K/V is strongly
active, but it reorganizes independent bridge spans and towers instead of making
the editable prediction continue the same locked bridge. Stop untrained
clean-source K/V injection; do not run a trajectory or sweep timestep, strength,
blocks, or context range.

## Fixed diagnostic contract

Only the first FLUX.2 Klein evaluation at `sigma=1` was executed for
`rigid_bridge`, `organic_tree`, and `photometric_desert`. No Euler proposal or
accepted-state update occurred.

- D0: ordinary step-0 state-restoration control, without appended K/V.
- D1: frozen same-sigma noisy-source K/V mechanism.
- D2: a separate source forward receives Comfy's processed clean source latent
  `y`, while retaining the fixed target `sigma=1` timestep/modulation contract.
- Target D0/D1/D2 input state is identical.
- Native full-canvas RoPE is applied before source-region K capture.
- All 5 double and 20 single attention blocks append 1,024 source K/V tokens to
  the ordinary 512-text + 2,048-image sequence (`2560 x 3584` Q/K geometry).
- Only the 1,024 editable image-query outputs retain augmented attention.
  Per-block ordinary outputs are restored for text and locked image queries.
- Source context is read-only. Returned state equals the accepted input and the
  sampler records zero updates.

D2 is an untrained experimental feature extractor. Clean `y` under target
`sigma=1` modulation is not claimed to be a native trained conditioning or
same-sigma diffusion contract.

## Integrity evidence

Runtime checks establish that the source branch receives clean `y` exactly,
while clean `y` differs materially from `z_L` (RMS `1.303917`, `1.323584`, and
`1.302931`). The D1 source branch equals fixed noise at `sigma=1`. Three unit
tests verify distinct clean/noisy source inputs, separate K/V probe storage, and
the unchanged 1,024 editable / 1,024 locked query partition.

Restoring ordinary attention output for locked queries is a per-block rule.
Later blocks can still observe the changed editable hidden states through native
self-attention, so final locked raw-x0 equality to D0 is neither asserted nor
required. This does not alter accepted-state ownership.

## Raw prediction activity

| Case | Editable D2-D0 RMS | Editable D2-D1 RMS |
|---|---:|---:|
| rigid bridge | 0.643779 | 0.703549 |
| organic tree | 0.483762 | 0.525320 |
| photometric desert | 0.293480 | 0.301975 |

D2 is therefore strongly active in every case.

## Authoritative-boundary diagnostics

For visual diagnosis only, each raw x0 was also decoded after replacing locked
latent coordinates with clean source `y`. This does not feed the model, change
the sampler, or create a trajectory.

| Case | Arm | Left LF RMS | Right LF RMS | Left seam RMS | Right seam RMS |
|---|---|---:|---:|---:|---:|
| bridge | D0 | 0.163058 | 0.183047 | 0.064442 | 0.067925 |
| bridge | D2 | 0.142580 | 0.153094 | 0.047919 | 0.056172 |
| tree | D0 | 0.244510 | 0.224841 | 0.027582 | 0.108309 |
| tree | D2 | 0.231381 | 0.211586 | 0.030366 | 0.138143 |
| desert | D0 | 0.223761 | 0.242308 | 0.025305 | 0.024973 |
| desert | D2 | 0.210188 | 0.228649 | 0.034113 | 0.033629 |

Scalar low-frequency mismatch improves on both sides of every case. That is not
sufficient for the geometric pass gate. Tree and desert seam-gradient metrics
worsen on both sides.

## Visual assessment

- Rigid bridge: D2 preserves a broadly aligned deck height, but creates a
  doubled/compound suspension system on the left and a differently scaled large
  span on the right. Towers and cable envelopes remain independently organized;
  this is not continuation of the source bridge's cable system, scale, and
  perspective.
- Organic tree: the ground tone is somewhat closer, but the right tree boundary
  becomes less clean and the seam metric worsens. This is not reliable secondary
  support.
- Photometric desert: sky/sand low-frequency mismatch decreases modestly, while
  both boundary gradients worsen. It does not establish a safe compatibility
  mechanism.

The bridge result is an expected-direction failure despite favorable aggregate
photometric metrics. It directly fails the requested primary gate.

## Decision boundary

Do not continue this untrained clean-source K/V family with a short trajectory,
alternate source timestep, strength, subset of blocks, or context restriction.
The result does not falsify trained source-conditioning mechanisms, but it gives
no authorization to infer that arbitrary clean features under sigma-one
modulation form a valid Klein conditioning contract.

Artifacts, comparison sheets, and per-block K/V provenance are under
`experiments/local_edit_step0_clean_source_context_results/`. No production code
or registration changed.
