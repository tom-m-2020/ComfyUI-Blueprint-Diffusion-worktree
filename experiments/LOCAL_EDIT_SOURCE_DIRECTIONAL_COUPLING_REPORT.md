# Local Edit Source Directional Coupling

## Result

**Classification: BIDIRECTIONAL CO-EVOLUTION.**

The source branch cannot evolve independently while acting only as a one-way
context provider under this discriminator. Removing direct editable-token K/V
from source generated queries across all 25 Klein blocks materially destroys
the established TRUE_LOW editable response. Removing the opposite
source-to-editable direction also materially destroys it.

This is one zero-sampler-update model evaluation at the archived rigid-bridge
state. It does not define a production architecture.

## Fixed contract

- Input: archived accepted interval-0 `TRUE_LOW` state.
- Case, prompt, conditioning, sigma, partition, model, and native conversion:
  unchanged from source block localization.
- Model: native Klein 4B, double blocks `0..4`, then single blocks `0..19`.
- Geometry: 512 text tokens followed by 2,048 row-major generated tokens;
  latent columns `16:48` are source and both outer regions are editable.
- Sampler updates: zero.

`TRUE_LOW / FULL` reproduced the archived raw-x0 tensor bit-exactly. The archived
input hash remained unchanged after all calls.

## Directional intervention

Each native attention call retained its ordinary joint text/image softmax. A
full additive query-by-key mask used `-finfo(dtype).max` only on the forbidden
directed edge rectangle:

- `NO_EDITABLE_TO_SOURCE`: source query rows by editable K/V columns;
- `NO_SOURCE_TO_EDITABLE`: editable query rows by source K/V columns.

All text-query rows and text-key columns remained zero in the mask. The opposite
generated direction, source-to-source, editable-to-editable, Q/K/V, RoPE,
modulation, residual, and MLP behavior were unchanged. Each arm blocked exactly
`1,048,576` edges per block (`1,024 x 1,024`). This changes normalization inside
the joint attention itself; it does not mask an already-computed output or
separately normalize subsets.

Text intentionally evolves normally. Thus the experiment removes the direct
generated-token edge, not every possible indirect route through text state.

## Quantitative result

Primary metrics are over editable raw x0.

| Arm | Projection onto full epsilon | Cosine to full epsilon | Projection onto TRUE_LOW | Cosine to TRUE_LOW | Editable RMS from TRUE_LOW |
|---|---:|---:|---:|---:|---:|
| TRUE_LOW / FULL | 0.834 | 0.912 | 1.000 | 1.000 | 0.000 |
| NO_EDITABLE_TO_SOURCE | 0.548 | 0.471 | 0.509 | 0.403 | 0.711 |
| NO_SOURCE_TO_EDITABLE | 0.452 | 0.407 | 0.450 | 0.379 | 0.694 |

Source raw-x0 RMS from TRUE_LOW is `0.785` when editable-to-source edges are
removed. This is direct evidence that source evolution materially depends on
editable generated-token feedback. Source RMS for the opposite ablation is
`0.403`, because source queries retain ordinary access but later coupled
evolution still changes.

Neither ablation retains most of TRUE_LOW. Source-to-editable removal destroys
slightly more aligned response, but editable-to-source removal is itself far too
destructive for the one-way-provider classification.

## Decoded evidence

The fixed sheet contains Ordinary, full epsilon, TRUE_LOW, and both directional
arms. Both ablations substantially reorganize bridge deck, cables, tower/span
relationships, and train alignment. `NO_EDITABLE_TO_SOURCE` creates a different
central tower/span organization. `NO_SOURCE_TO_EDITABLE` changes the left span,
cable fan, deck origin, and right-tower relationship. These are structural, not
merely photometric, changes.

## Mechanical invariants

All required invariants passed:

- all executions used the identical archived state;
- TRUE_LOW reproduced bit-exactly and the input remained immutable;
- all 5 double and 20 single blocks were intercepted exactly once per arm;
- every block masked exactly the intended directional edge set;
- text rows/columns and unrelated generated interactions remained allowed;
- Q, K, V, and position tensors were returned by identity;
- normalization used one native joint softmax over remaining keys;
- no sampler update, state replacement, prediction blend, cache, sparse
  execution, or production change occurred.

## Interpretation and limits

The scene-spatial source signal is not context produced by an autonomous source
branch. Editable tokens provide causally necessary feedback into source-query
evolution, and source tokens provide causally necessary context back to editable
queries. The next-call response depends on recurrent bidirectional interaction
through transformer depth.

This does not isolate every indirect text-mediated route, identify required
blocks/heads, qualify a trajectory, or establish a cache/separation policy. No
production implementation is authorized.

Artifacts:

- `experiments/local_edit_source_directional_coupling.py`
- `experiments/test_local_edit_source_directional_coupling.py`
- `experiments/local_edit_source_directional_coupling_results/report.json`
- `experiments/local_edit_source_directional_coupling_results/runtime_tensors.pt`
- `experiments/local_edit_source_directional_coupling_results/RIGID_BRIDGE_DIRECTIONAL_COUPLING_COMPARISON.png`
