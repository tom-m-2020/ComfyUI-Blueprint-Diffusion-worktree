# Phase 32 — Exact restriction-nullspace working-state discriminator

## Decision

**D — NO PARAMETER-FREE SIGMA-CONSISTENT CONSTRUCTION EXISTS**

The experiment stopped before model inference. The proposed restriction-null
noise is a valid algebraic nullspace tensor, but it cannot replace or augment
the qualified i.i.d. Gaussian noise while preserving the same sigma-0.25 CONST
state distribution without an additional, unqualified construction choice.

Production and ComfyUI core are unchanged. No Blueprint trajectory or control
model branch was recomputed.

## Validated controls

The persisted Phase-29 artifacts were validated for all three cases:

- terminal Blueprint and mapped-Blueprint hashes match;
- sigma is exactly `0.25`;
- the production control remains bit-exact to Phase 28;
- region count/order and every original noise hash match regenerated noise;
- control semantic grade remains S3.

The validated region counts are 25 square, 55 portrait, and 40 landscape.

## Operator and exact nullspace check

Define

```text
D = avg_pool2d(kernel=2, stride=2)
U = nearest2
P = U D
n = epsilon - P epsilon = (I-P) epsilon
```

`D(n)` is zero within float32 arithmetic for every channel and region:

| Case | maximum `max_abs D(n)` | maximum `RMS D(n)` | mean `Var(n)/Var(epsilon)` |
|---|---:|---:|---:|
| square multi-object | `2.3842e-7` | `2.5193e-8` | `0.749917` |
| portrait astronaut | `2.3842e-7` | `2.5131e-8` | `0.750056` |
| landscape bridge | `2.9802e-7` | `2.5157e-8` | `0.750014` |

Every region record contains the Blueprint-crop, nearest-anchor, original-noise,
and null-noise hashes plus restriction RMS/max and variance diagnostics.
Working-state and prediction hashes are explicitly null because fail-closed
execution stopped before constructing an unqualified state.

## Why sigma consistency cannot be preserved

For one scalar channel in a `2x2` block, let `epsilon ~ N(0,I_4)`. The
nearest/mean projector is

```text
P = (1/4) 1 1^T
```

and therefore

```text
Cov((I-P)epsilon) = I_4 - (1/4) 1 1^T.
```

This covariance has:

- rank `3`, not `4`;
- diagonal variance `3/4`;
- off-diagonal covariance `-1/4`.

The qualified CONST construction is

```text
W = 0.75 * anchor + 0.25 * epsilon,
epsilon ~ N(0,I).
```

Replacing `epsilon` with the null residual changes both its variance and joint
law. Multiplying by `2/sqrt(3)` restores each element's marginal variance, but
the covariance remains singular and the off-diagonal covariance becomes
`-1/3`; it is not i.i.d. Gaussian noise. That scaling is also explicitly a
renormalization forbidden by the task.

The remaining alternatives do not define the requested discriminator:

- retaining `epsilon` exactly reproduces control A because its null component
  is already present;
- adding another copy of the null component double-counts it and requires a
  detail-strength coefficient;
- replacing only the coarse component requires defining a new correlated noise
  law or independent coarse/detail coupling.

Thus no coefficient is uniquely implied by sigma `0.25` or by
`model_sampling.noise_scaling`. The flow equation specifies how a valid noise
sample enters the state; it does not make a rank-deficient nullspace sample a
valid replacement for `N(0,I)`.

## Execution and telemetry

- New model calls: `0`
- Destination-sized forwards: `0`
- CUDA/model timing: not applicable
- New peak VRAM: not applicable
- Accepted source mutation: none
- Production changes: none
- ComfyUI-core changes: none

## Interpretation

The restriction nullspace is mathematically real, but treating it as the sole
sigma noise is not distribution-preserving. Phase 32 therefore cannot test its
semantic/detail effect without adding a prohibited renormalization or a new
multi-component state model.

The next phase should investigate a **model-mediated or explicit multi-state
coarse/detail construction**, with its stochastic contract declared directly,
rather than another handcrafted spatial transform or strength sweep.

## Artifacts

- Validator: `experiments/flux2_terminal_resampling_restriction_nullspace.py`
- Machine report: `experiments/flux2_terminal_resampling_restriction_nullspace_results/report.json`
