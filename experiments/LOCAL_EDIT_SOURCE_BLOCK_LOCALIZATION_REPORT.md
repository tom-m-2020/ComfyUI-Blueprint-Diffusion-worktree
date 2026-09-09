# Local Edit Source Block Localization

## Result

**Classification: DISTRIBUTED SOURCE-CONTEXT TRANSFER.**

The correctly arranged interval-0 low-frequency source signal is transferred into editable hidden state cumulatively across Klein's depth. No tested boundary cleanly separates an early source-dependent phase from a later source-independent phase. Exact single-block source-K/V substitutions confirm that generated-image attention is causal, with the largest individual effects in the double-stream stage, but they do not account for the full depth-distributed state-crossover curve.

This is a one-evaluation causal diagnostic, not a sampling algorithm or an execution-skipping result.

## Scope and fixed inputs

- Case: `rigid_bridge` only.
- Model: native FLUX.2 Klein 4B (`double_blocks[0..4]`, then `single_blocks[0..19]`).
- State: the accepted interval-0 Ordinary state plus either archived `TRUE_LOW` or archived `TRANSLATED_LOW` source-only pulse.
- Evaluation: the immediately following ordinary dense model call at the archived sigma.
- Sampler updates during intervention calls: zero.
- Generated token order: row-major latent grid `32x64`; center columns `16:48` are source (1,024 tokens), and the two exterior regions are editable (1,024 tokens).
- Text, prompt, conditioning, sigma, weights, generated position IDs/RoPE, and all non-pulse inputs were shared.

The Ordinary and full-epsilon trajectories were rerun only to reconstruct their accepted interval-0 inputs. The four required next-call baselines reproduced the archived tensors bit-exactly: `ORDINARY`, `FULL_EPSILON`, `TRUE_LOW`, and `TRANSLATED_LOW`.

## Intervention definitions

### Source hidden-state crossover

Each crossover follows `TRUE_LOW` through the named boundary. At that boundary only source generated-token hidden states are replaced by the corresponding `TRANSLATED_LOW` states. Editable generated tokens remain bit-exact to `TRUE_LOW`. In the double stream, replacement addresses the source subset of the 2,048-token image stream. In the single stream, it addresses the same spatial subset inside the generated slice `[512:2560]` of the joint text/image stream.

At `before_double_0`, replacement occurs before the first native block. At all `after_*` boundaries, the named native block executes first and replacement occurs on its output. `after_single_19` is therefore immediately before the ordinary tokenwise final projection.

### Source-K/V attention substitution

For one named block, TRUE_LOW editable queries retain their ordinary Q. Text and editable generated K/V remain TRUE_LOW, while only source generated K/V are substituted from the paired TRANSLATED_LOW execution. The block then uses one ordinary native joint softmax. No separately normalized attention subset was constructed.

The normal TRUE_LOW attention output was computed in the same call. After substituted attention, text-query and source-query outputs were restored bit-exactly, leaving only editable generated-query attention changed. Residual input, MLP, modulation, and positional data remained ordinary TRUE_LOW.

## Mechanical invariants

All arms passed:

- the model input before double block 0 matched its intended baseline exactly;
- only source generated tokens were replaced at state crossovers;
- editable tokens at crossover were bit-exact to TRUE_LOW;
- source tokens at crossover were bit-exact to TRANSLATED_LOW;
- attention substitutions used one joint softmax;
- text-query and source-query attention outputs were restored exactly;
- no sampler update occurred before raw-x0 capture;
- final projection and model-output conversion were native and unpatched.

The test module also proves the fixed native block sets and source-only token replacement.

## Baseline response

All projections below are measured over editable raw x0 relative to `delta_E = FULL_EPSILON - ORDINARY`.

| Arm | Projection onto delta_E | Cosine to delta_E | Projection onto TRUE_LOW |
|---|---:|---:|---:|
| ORDINARY | 0.000 | degenerate | 0.000 |
| TRANSLATED_LOW | 0.120 | 0.321 | 0.151 |
| TRUE_LOW | 0.834 | 0.912 | 1.000 |
| FULL_EPSILON | 1.000 | 1.000 | 1.000 vs delta_E |

## Experiment 1: block-boundary source-state crossover

| Crossover boundary | Projection onto delta_E | Cosine to delta_E | Projection onto TRUE_LOW | Cosine to TRUE_LOW |
|---|---:|---:|---:|---:|
| before double 0 | 0.120 | 0.321 | 0.151 | 0.368 |
| after double 4 | 0.212 | 0.442 | 0.268 | 0.512 |
| after single 4 | 0.449 | 0.645 | 0.561 | 0.737 |
| after single 9 | 0.589 | 0.778 | 0.728 | 0.880 |
| after single 14 | 0.774 | 0.881 | 0.942 | 0.980 |
| after single 19 | 0.834 | 0.912 | 1.000 | 1.000 |

The curve is progressive. The five double blocks alone retain only 0.212 projection, substantially below TRUE_LOW's 0.834. Correct source state remains causally important through the single-stream stage: replacing it after single 4 or single 9 still removes substantial aligned response. Even after single 14, replacement produces a measurable residual loss, although most TRUE_LOW response has already accumulated.

Internal activation differences are diagnostic rather than causal evidence. TRUE versus TRANSLATED editable hidden-state RMS is zero before double 0, then `1.13` after double 4, `4.54` after single 4, `4.56` after single 9, `9.48` after single 14, and `34.57` after single 19. This confirms that source arrangement has already affected editable state in the double stage and continues to propagate.

## Experiment 2: exact source-K/V substitution

| Substituted block | Projection onto delta_E | Cosine to delta_E | Projection onto TRUE_LOW | Editable attention-output RMS change |
|---|---:|---:|---:|---:|
| double 0 | 0.689 | 0.828 | 0.845 | 0.051 |
| double 4 | 0.637 | 0.814 | 0.782 | 0.544 |
| single 4 | 0.790 | 0.894 | 0.960 | 0.028 |
| single 9 | 0.764 | 0.884 | 0.927 | 0.031 |
| single 14 | 0.811 | 0.898 | 0.982 | 0.055 |
| single 19 | 0.831 | 0.910 | 0.997 | 0.059 |

Every tested source-K/V substitution changes editable attention output and final editable raw x0. The largest single-block loss relative to TRUE_LOW occurs at double block 4, followed by double block 0. Individual single-block substitutions are smaller, but remain nonzero through single 19. This supports causal source-to-editable attention communication; it does not imply that one block or cached K/V is sufficient.

The state crossover is deliberately stronger than a one-block K/V substitution: it changes source residual state entering all later attention and MLP computation. The difference between their curves therefore leaves later source-state propagation, repeated attention, text-state evolution, and non-attention transformations entangled. It does not justify the classification `SOURCE-STATE / NON-ATTENTION DEPENDENCE`, because exact K/V interventions themselves have qualified causal effects.

## Spatial and decoded evidence

The result JSON retains the fixed coarse editable response map for every state and attention intervention. The decoded sheet shows the same ordering as the tensor evidence:

- replacing source state before double 0 visually returns toward the spatially incorrect/Ordinary bridge organization;
- retaining TRUE_LOW until after single 19 preserves its reorganized deck and cable trajectory;
- double-block K/V substitutions visibly weaken/change the TRUE_LOW organization without simply relocating the bridge.

These are step prediction decodes only. They establish where the source signal affects bridge geometry, not final-image quality.

## Why no binary refinement ran

The predeclared boundaries do not show a sharp transition. Sensitivity falls progressively across both stages, and all fixed boundaries before final projection retain some causal dependence. Under the stopping rule, this is the distributed outcome; adding midpoint probes would overstate a transition that the fixed evidence does not contain.

## Causal conclusion

The scene-spatial low-frequency source pulse begins communicating into editable state during the double-stream stage, with source-image attention causally contributing. The useful response is not completed there. Correct source state remains necessary across a broad portion of the single-stream stack, where repeated dense computation progressively accumulates the editable response.

Therefore the narrow classification is **DISTRIBUTED SOURCE-CONTEXT TRANSFER**, not early transfer and not an editable-state-dominated handoff after an identified early boundary.

## Explicit limits and unresolved questions

- This experiment does not isolate attention from source residual/MLP and text-state pathways across the complete stack.
- A one-block K/V substitution does not test sufficiency of source K/V caching.
- Becoming less sensitive late in this one pulse diagnostic does not authorize skipping later source tokens or computation.
- No trajectory, sparse execution, cache, guidance change, parameter tuning, or production implementation was tested.

Artifacts:

- `experiments/local_edit_source_block_localization.py`
- `experiments/test_local_edit_source_block_localization.py`
- `experiments/local_edit_source_block_localization_results/report.json`
- `experiments/local_edit_source_block_localization_results/runtime_tensors.pt`
- `experiments/local_edit_source_block_localization_results/RIGID_BRIDGE_BLOCK_LOCALIZATION_COMPARISON.png`
