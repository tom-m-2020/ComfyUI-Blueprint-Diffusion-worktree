# Phase 36 — Short terminal local trajectory discriminator

## Result

**D — NO UNIQUE SHORT LOCAL SCHEDULE**

The experiment stopped before diffusion inference. The qualified terminal
contract specifies exactly one local evaluation at `sigma=0.25`, algebraically
equivalent to the Euler interval `[0.25,0]`. It does not identify a scheduler
family, a higher starting sigma, an interval count, or intermediate sigma
locations. The fixed Blueprint trajectory also cannot supply a late tail: its
last positive point is `0.9926428198814392`, and it contains no positive value
at or below `0.25`.

Selecting a multi-step schedule would therefore introduce several independent
empirical choices explicitly prohibited by Phase 36. No FLUX transformer call,
local prediction, decode, or comparison image was produced.

## Control validation

The persisted square control was validated without recomputation:

- case: `SQUARE_MULTI_OBJECT`;
- seed: `20260921`;
- mapped Blueprint hash:
  `8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05`;
- qualified sigma-0.25 output hash:
  `1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3`;
- Phase-28 regression: bit-exact;
- persisted semantic class: S3.

Thus the A control remains exact. It was not rerun because no B schedule passed
the pre-inference gate.

## Schedule audit

The qualified schedules have different ownership:

```text
Stage 1 Blueprint:
  [1.0, 0.9991771579, 0.9975355268, 0.9926428199, 0.0]
  source: fixed ManualSigmas / production QUALIFIED_SIGMAS

Stage 2 local:
  [0.25, 0.0]
  source: fixed terminal-resampling contract
```

Stage 2 has no named scheduler. A one-step Euler update from `0.25` to zero
returns the denoised prediction, so the existing production path does not need
to choose one.

The live ComfyUI model sampling exposes nine scheduler families. Even with an
arbitrarily fixed four-interval count, representative outputs differ sharply:

| Scheduler | Four-interval sigma vector |
|---|---|
| simple | `1, .957654, .882881, .715325, 0` |
| sgm_uniform | `1, .957695, .883037, .715938, 0` |
| karras | `1, .185323, .020031, .000753, 0` |
| exponential | `1, .090991, .008279, .000753, 0` |
| beta | `1, .972502, .882922, .616551, 0` |
| normal | `1, .937864, .790694, .005651, 0` |
| linear_quadratic | `1, .987500, .975000, .725000, 0` |
| kl_optimal | `1, .577685, .268488, .000753, 0` |

`ddim_uniform` is also available and starts below the model maximum. Three-
interval schedules were recorded as well and introduce a different set of
points. None is selected by Phase 25–29 evidence.

The requested approximate form—start above `0.25`, include one or more
intermediates, and end at zero—still requires all of these choices:

1. scheduler family;
2. full schedule step count;
3. local start point;
4. truncation/splice rule around `0.25`;
5. number of retained local intervals.

The live noise-scaling equation determines `W_sigma` after a sigma is chosen;
it does not choose the sigma path. Likewise, Euler determines an update between
two supplied sigmas but does not generate those sigmas.

## Why no candidate was inferred from the Blueprint schedule

Using the last three Blueprint intervals would start at approximately `0.9975`
and provide only another near-one point before a direct jump to zero. That is
not a late refinement schedule around the qualified `0.25` state and would
materially replace its initialization regime. Inserting `0.25` into that vector
would be a new hand-selected splice. Increasing an underlying scheduler's step
count until it happens to cross `0.25` would make both family and step count
new parameters.

Therefore “reuse the qualified full-denoise schedule” does not yield one
well-defined B arm.

## Execution accounting

| Item | Result |
|---|---:|
| A model calls in this phase | 0; exact persisted evidence reused |
| B calls per region | 0 |
| B total local calls | 0 |
| destination-sized model forwards | 0 |
| Blueprint recomputations | 0 |
| decoded images | 0 |
| accepted-state mutations | 0 |

Runtime, CUDA, VRAM, gradients, overlap, low-frequency discrepancy, and B
semantic grade are not applicable because B did not pass preflight. Creating
placeholder comparison/detail images would misleadingly imply an executed
candidate, so those artifacts are intentionally absent.

## Architectural interpretation

Phase 36 does not falsify or qualify a short local trajectory. It establishes
that the currently qualified terminal-resampling architecture contains no
parameter-free multi-step extension. A valid future experiment needs a
separate, explicit design decision selecting one scheduler rule from model or
training provenance. That decision must be justified before inference and
must not be presented as inherited from the existing `0.25` terminal contract.

The next recommendation is a schedule-design audit, not a model sweep: trace
the Klein training/official inference timestep distribution and determine
whether it canonically defines a short image-to-image refinement schedule from
a fixed denoising strength. If it does not, any future short-trajectory test
must openly predeclare one empirical schedule as a new research policy.

Phase 36 concerns terminal-local multi-step refinement only. It is **not** the
separate interleaved global-to-local/local-to-global resampling architecture;
no local-to-Blueprint feedback was designed or executed.

## Artifacts

- Preflight harness:
  `experiments/flux2_terminal_resampling_short_local_trajectory.py`
- Machine report:
  `experiments/flux2_terminal_resampling_short_local_trajectory_results/report.json`

The harness validates persisted control fingerprints and enumerates live
scheduler outputs without invoking the diffusion transformer.
