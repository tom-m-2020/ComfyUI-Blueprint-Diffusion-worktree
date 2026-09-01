# Phase 6j — production elimination of the terminal global forward

Date: 2026-09-01

## Production contract

The coordinator now recognizes the terminal Euler interval before model
execution. Intervals 0–2 retain the qualified production path: one fresh
global prediction, fresh local predictions, Euler proposals, hard coarse
coupling, and one atomic `(G,H)` acceptance. Interval 3 performs only the
fresh local predictions, constructs `H*`, and accepts it directly.

`BlueprintState.g` remains non-optional. On terminal acceptance it retains the
last accepted nonterminal `G` tensor. This is diagnostic state only and is
explicitly reported as `retained_preterminal_unsynchronized`; it is not
claimed to correspond to terminal `H`. This avoided a broad state/API change
while removing the unused model call.

Terminal telemetry now records:

```text
global_forward_performed: false
terminal_global_unused: true
terminal_release: true
global_synchronized: false
global_state_status: retained_preterminal_unsynchronized
executed_global_tokens: 0
projection_rms: null
```

No terminal `x0_G` or `G*` capture is emitted. The reported model-prediction
count is reduced by one only on that interval.

## Frozen-baseline regression

The current four-global-forward package was captured before the production
edit. The modified package was then run against the same real FLUX.2 Klein
W4A8 model, prompts, seeds, conditioning, geometry-specific local-window
policy, and four-step schedule. Evidence is in
`flux2_candidate3_terminal_global_elimination_results/report.json`; the
executable harness is
`flux2_candidate3_terminal_global_elimination_regression.py`.

Across all four required cases, every compared tensor was bit-exact:

- initial `H` and `G`;
- interval 0–2 `x0_G`, `G*`, and accepted `G`;
- all four assembled `x0_H`, `H*`, and accepted `H`;
- returned final `H`.

The terminal retained `G` is bit-exact with interval 2 accepted `G`. Terminal
`x0_G` and `G*` are absent. Each run emitted four previews, performed exactly
three global forwards and four sets of local forwards, and retained every
nonterminal `D(H)=G` invariant within the existing tolerance.

| Case | Four-global warm wall | Optimized warm wall | Change |
| --- | ---: | ---: | ---: |
| H64 person/car/tree | 16.451 s | 15.336 s | -6.78% |
| H64 bridge/train | 16.613 s | 16.054 s | -3.36% |
| H64 boundary astronaut | 16.665 s | 15.418 s | -7.49% |
| H48 person/car/tree | 8.679 s | 8.061 s | -7.12% |

The aggregate wall-time reduction is 6.06%. The bridge measurement is noisier
than the other three, but the combined result is consistent with Phase 6i's
6.6% mean discriminator result. Peak allocator memory remains effectively an
unchanged property of the larger nonterminal/local forwards (2.88–2.89 GiB
allocated and 3.42 GiB reserved at H64; 2.67/3.00 GiB at H48).

## Atomicity and failure behavior

All model calls still read immutable accepted state. The coordinator constructs
and validates the full local assembly and terminal `H*` before publishing a
new frozen state, so cancellation or a local-call failure cannot publish a
partial terminal acceptance. Schedule, model family, CFG, geometry,
conditioning, mask/edit, and flow-sampler validation paths were not broadened.
The focused suite passes 18 tests, including a terminal coordinator test that
fails if `predict_global` is invoked.

## Live ComfyUI qualification

A fresh ComfyUI process loaded the package through its existing custom-node
junction. Normal graphs used `BasicGuider`, `Blueprint Candidate-3 Euler
Sampler`, `SamplerCustomAdvanced`, `VAEDecode`, and `SaveImage` at both
H=64×128 and H=48×96. Each first queue execution emitted four previews and
completed normal VAE decode/save. Repeated queues returned the same decoded
RGB hashes exactly (the second executions were valid ComfyUI cache hits):

- H64: `0318b3fed040af1f4f336d58391c92774f123399137543b7fae372b1aaacf19d`
- H48: `595cc0b233fabef3c763ccfb2d6930f0526a66506a8aec9541f1a9a29c554e29`

Machine-readable live evidence is in
`live_comfyui_candidate3_terminal_global_elimination_results/report.json`.

## Verdict

The terminal global prediction and proposal are causally unused by returned
`H` and can be removed without numerical or lifecycle drift. Nonterminal
behavior is unchanged, and terminal diagnostic `G` is now honestly labeled.

```text
TERMINAL GLOBAL FORWARD ELIMINATED
```
