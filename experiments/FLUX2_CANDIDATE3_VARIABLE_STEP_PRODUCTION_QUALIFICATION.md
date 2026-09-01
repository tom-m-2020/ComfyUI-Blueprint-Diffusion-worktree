# Phase 7b — production variable-step Euler support

Date: 2026-09-01

## Production change

`BlueprintEulerSampler` now accepts any one-or-more-interval schedule satisfying
the already-qualified full-denoise CONST-flow contract:

```text
one-dimensional sigma tensor with at least two values
sigma[0] == 1.0
sigma[-1] == 0.0
all values finite
all preterminal values > 0
strictly decreasing
```

The coordinator, geometry, crop/window selection, DCT operators, model adapter,
coupling, global cadence, and terminal-release policy were not changed. The
existing production loop already derived `total` from `len(sigmas) - 1`; only
the four-step validation restriction was removed.

The terminal contract remains cardinality-independent:

- ordinals `0 .. N-2`: fresh global prediction, all fresh local predictions,
  Euler proposals, hard `D(H)=G` coupling, atomic acceptance;
- ordinal `N-1`: no global prediction, fresh local predictions, terminal H*
  acceptance, prior G retained as explicitly unsynchronized diagnostic state.

Detached telemetry is cleared at the beginning of every sampler invocation.
A validation error, cancellation, or model-call failure therefore cannot expose
an earlier run's telemetry as though it belonged to the failed run. Coordinator
state remains invocation-local and is published only after a complete atomic
interval acceptance.

The node description now says “variable-step” and “full-denoise” rather than
implying a four-step-only contract.

## Focused and frozen regression

The focused unit suite passes 18 tests. Schedule tests cover 1, 2, 4, 8, and 20
steps plus empty, scalar, non-1D, nonfinite, nonzero-terminal, nonpositive, and
nonstrict schedules. Existing mask/edit, CFG/model, geometry, state
immutability, DCT, crop coverage, coupling, and terminal tests continue to pass.

The frozen Phase-6j four-step production regression was rerun across all four
scenes. Initial H/G, every prediction/proposal/accepted state, and final H
remain bit-exact (`RMS=0`, `max_abs=0`).

The new machine-readable boundary regression is in
`flux2_candidate3_variable_step_production_results/report.json`. It compares
production through `sample_custom` against the Phase-7a direct-coordinator
lifecycle at H=48x96 for 1/2/4/8/20 steps. Every available tensor was bit-exact:

```text
initial_H
initial_G
every nonterminal x0_G and G_star
every assembled_x0_H and H_star
every accepted G and H
final H
```

| Steps | Previews | Global forwards | Local forwards | Boundary result |
| ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0 | 3 | bit-exact |
| 2 | 2 | 1 | 6 | bit-exact |
| 4 | 4 | 3 | 12 | bit-exact |
| 8 | 8 | 7 | 24 | bit-exact |
| 20 | 20 | 19 | 60 | bit-exact |

Only the final interval reported `global_forward_performed=false` and
`terminal_release=true`. Every nonterminal invariant remained within the
qualified DCT tolerance.

## Fresh live ComfyUI qualification

A fresh ComfyUI process loaded the ordinary custom node and ran:

```text
BasicGuider
→ Blueprint Candidate-3 Euler Sampler
→ SamplerCustomAdvanced
→ VAE Decode
→ Save Image
```

Both currently qualified production geometries were run at 4, 8, and 20 steps.
Each first execution emitted exactly one preview per interval. Repeated queues
completed as valid cache hits and produced identical decoded RGB hashes:

| Geometry | Steps | Decoded RGB SHA-256 |
| --- | ---: | --- |
| H64x128 | 4 | `0318b3fed040af1f4f336d58391c92774f123399137543b7fae372b1aaacf19d` |
| H64x128 | 8 | `0422d683733a014cbded3106d33b4d69e6fe6cd310f856ceb74d4b88864675f6` |
| H64x128 | 20 | `2704439af60b1e35166882d114fd9ffc78f43c204c079a9ce8c08b3c49721183` |
| H48x96 | 4 | `595cc0b233fabef3c763ccfb2d6930f0526a66506a8aec9541f1a9a29c554e29` |
| H48x96 | 8 | `7006d3d2527fa267c4ec510e7370c74eca66d43650f2e1c3231d4fa649f2eca4` |
| H48x96 | 20 | `a92bf617b925b6c14cf81b60a7fd733264f04193155c16ff512570193f6cd1d4` |

Live evidence is in
`live_comfyui_candidate3_variable_step_results/report.json`.

## Verdict

```text
VARIABLE-STEP PRODUCTION SUPPORT QUALIFIED
```
