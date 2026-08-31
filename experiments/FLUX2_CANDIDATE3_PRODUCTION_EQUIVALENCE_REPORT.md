# FLUX.2 Candidate-3 production equivalence report

## Scope

This regression compares the first fail-closed production slice directly with
the Phase-3 experiment implementation. Both paths use the Phase-3c prompt and
seed, native FLUX.2 Klein 4B, the same four-step schedule, 32 x 64 accepted `H`,
24 x 48 block-DCT `G`, three 32 x 32 crops, normalized overlap, hard
nonterminal coupling, and terminal release.

The production path was invoked as a normal ComfyUI `SAMPLER` through
`comfy.sample.sample_custom`, the same guider preparation used by
`SamplerCustomAdvanced`.

## Fixed inputs

```text
seed: 20260831
sigmas:
  1.0
  0.9614368677139282
  0.8925943970680237
  0.7347596883773804
  0.0
H: [1, 128, 32, 64]
G: [1, 128, 24, 48]
crops: x=0,24,32; y=0; each 32x32
CFG: 1.0
```

## Equivalence result

Every required boundary was bit-exact (`RMS=0`, `max_abs=0`):

- initial `H`;
- initial `G=D(H)`;
- every `x0_G`;
- every assembled `x0_H`;
- every `G*`;
- every `H*`;
- every accepted `G`;
- every accepted `H`;
- final `H`.

There is no first divergence. No floating-point tolerance was needed for
research-versus-production equivalence.

## Lifecycle and invariant evidence

- Four accepted intervals were committed.
- Each interval performed exactly four guider predictions: one 1152-token
  global prediction and three crop predictions totaling 3072 local tokens.
- Terminal flags were exactly `[false, false, false, true]`.
- Startup synthetic `max_abs(D(U(G))-G)` was `7.15e-7`, below the documented
  `2e-6` tolerance.
- Nonterminal accepted-state invariant maximum errors were `7.15e-7`,
  `5.96e-7`, and `4.77e-7`.
- The terminal interval deliberately did not enforce `D(H_final)=G_final`.
- Projection RMS by interval was `0.033695`, `0.033817`, `0.070745`, and
  `0.329385` (the last value is measured but not applied).

The production coordinator stores only scalar/shape summaries during normal
execution. Full CPU tensors in this regression came from its explicitly
injected test capture and are not retained by the node-created sampler.

Recorded wall times were 10.60 seconds for the research path and 4.31 seconds
for the production path in sequential execution. The first path paid cold-run
costs, so these numbers are not a performance comparison.

## Fail-closed boundary

The implementation rejects non-FLUX.2 models/backends, CFG other than 1,
non-CONST sampling, non-four-interval or invalid schedules, non-32x64/batch-one
latents, nonempty edit latents, masks, nested latents, conditioning hooks,
spatial/control/reference conditions, and any failed coverage, mutation,
finiteness, right-inverse, or nonterminal coupling check.

## Verdict

The first production slice reproduces the qualified research contract without
numerical drift.

```text
PRODUCTION SLICE EQUIVALENT
```
