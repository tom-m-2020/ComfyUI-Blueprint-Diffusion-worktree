# Local Edit Step-0 Native Source-Context Diagnostic

Date: 2026-09-09

## Question

Does explicit same-sigma source-region K/V in native FLUX.2 attention change the editable raw model `x0` at the first evaluation toward a geometrically or photometrically compatible continuation?

## Scope

Only the first evaluation was run for the qualified `rigid_bridge`, `organic_tree`, and `photometric_desert` cases. There was no Euler update, accepted-state change, diffusion trajectory, sparse execution, selector, pixel compositing, new sampler algorithm, affine correction, feathering, or schedule tuning.

The control D0 is B's step-0 model evaluation with exact source-state ownership as the declared sampler contract. At the first call, before any acceptance, its state is the ordinary CONST initial state. D1 uses the identical state and conditioning but augments native attention.

## Mechanism

For D1, a separate read-only native Klein forward captures source-region generated-token K/V at the same sigma and full-canvas RoPE coordinates. The D1 target forward retains ordinary text and full generated-token self-context, then appends those source-region K/V tokens. Only editable generated-query attention outputs are retained from the augmented call. Ordinary attention outputs are restored exactly for all text queries and locked generated queries.

The intervention covers all five double and all twenty single attention blocks:

- ordinary sequence: 512 text + 2,048 generated tokens;
- editable generated queries: 1,024;
- locked generated queries: 1,024;
- appended source K/V: 1,024;
- augmented attention geometry: 2,560 queries by 3,584 keys;
- source K is captured after native RoPE, preserving its original full-canvas coordinates.

There is no cache persistence. Source K/V is read-only and no source value becomes sampler state.

## Important CONST limitation at step 0

The live Flux2 schedule begins at `sigma=1`. Native CONST noising is

```text
source_state(sigma) = (1-sigma)*y + sigma*z_L
```

so at this diagnostic sigma, `source_state(1)=z_L` exactly. Runtime telemetry confirms source state versus fixed noise RMS/max error `0`. Thus D1 does not expose clean source appearance at step 0. It duplicates and reweights the same-coordinate locked-region token stream from an ordinary native forward. This is the mechanically faithful same-sigma result, not trained clean-source conditioning.

## Integrity checks

Three focused tests pass: exact 1,024/1,024 editable/locked token masks; restoration of text and locked-query attention outputs; and the CONST sigma-one identity. Runtime checks show:

- D0/D1 model input state is identical;
- no accepted-state update occurs;
- returned state is bit-exact to the input accepted state;
- source capture and target use the same sigma, seed, prompt, and full-canvas coordinates.

## Raw prediction differences

| Case | Editable D1-D0 RMS | Mean absolute | Maximum absolute |
|---|---:|---:|---:|
| rigid bridge | 0.278225 | 0.159431 | 3.437500 |
| organic tree | 0.207007 | 0.129811 | 2.097656 |
| photometric desert | 0.077836 | 0.057157 | 0.578125 |

The intervention is active and materially changes editable raw `x0`; this is not a no-op.

## Boundary measurements

| Case | Arm | Left LF discrepancy | Right LF discrepancy | Left seam gradient | Right seam gradient |
|---|---:|---:|---:|---:|---:|
| bridge | D0 | 0.199975 | 0.212276 | 0.035484 | 0.051063 |
| bridge | D1 | 0.200685 | 0.207569 | 0.040790 | 0.059159 |
| tree | D0 | 0.248424 | 0.288322 | 0.019206 | 0.106609 |
| tree | D1 | 0.250743 | 0.291645 | 0.019276 | 0.111994 |
| desert | D0 | 0.211362 | 0.231226 | 0.009024 | 0.011363 |
| desert | D1 | 0.212193 | 0.228550 | 0.010165 | 0.012264 |

Bridge right low-frequency discrepancy improves only about 2.2%, while left worsens and both seam gradients worsen. Tree discrepancy and seams worsen. Desert movement is small and mixed, with both seam gradients worsening.

## Visual result

- Bridge: D1 strongly changes bridge/tower organization, but strengthens and relocates independent tower/cable structure rather than aligning deck/cables to the two locked source boundaries. It is not an expected-direction continuation improvement.
- Tree: D1 shifts the tree and suppresses the already faint woman, without improving ground/illumination compatibility at the source edges.
- Desert: D1 is nearly the same global solution; the small change does not establish improved sky/sand continuity.

These observations are limited to decoded first-evaluation `x0` estimates, not final images.

## Verdict

**Fail the requested pass condition.** D1 materially changes the editable prediction, but not in the expected source-compatible direction in the rigid bridge case. Stop before implementing a trajectory. Do not sweep strengths, blocks, or context ranges.

The result also identifies why this exact first-sigma contract is weak: the same-sigma source trajectory contains no clean source contribution at `sigma=1`. Any future model-level experiment would need a separately justified mechanism for making source appearance available; that would be a different contract and is not authorized by this result.

Artifacts and machine-readable block/context telemetry are in `experiments/local_edit_step0_source_context_results/`. No production code changed.
