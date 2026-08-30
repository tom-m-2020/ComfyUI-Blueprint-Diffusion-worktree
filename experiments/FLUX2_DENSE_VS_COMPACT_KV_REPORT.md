# FLUX.2 late-evaluation dense-K/V versus compact-K/V diagnostic

## Verdict

**COMPACT INFORMATION LOSS.**

At the exact late Candidate-2 trajectory state where the duplicate lighthouse
first becomes clearly interpretable, replacing 512-token compact-global K/V
with 2048-token dense full-canvas K/V removed that lighthouse and moved the
local prediction much closer to the corresponding dense crop. Both variants
used the identical local execution and external-K/V integration path.

Dense K/V did not make the local prediction identical to dense: it retained a
dark stone structure that the dense crop contains only as a small distant
silhouette. The external-K/V mechanism is therefore imperfect, but the specific
duplicate-lighthouse failure discriminated here is primarily attributable to
compact representation quality/density.

## Reproduced state and controlled boundary

The experiment deterministically reproduced the Phase-2e global-context tiled
trajectory through exactly two accepted Euler updates, then stopped before
evaluation 2:

```text
step:  2
sigma: 0.8925943970680237
accepted latent: [1,128,32,64]
latent RMS: 0.9205285906791687
```

All recorded accepted-state statistics matched the prior Phase-2e report
exactly at its GPU-side accepted-state boundary. The float32 accepted latent
hash is recorded in `report.json`. The diagnostic returned this latent
unchanged (`max_abs=0`) and performed zero sampler updates.

The affected center crop was crop 1:

```text
latent x: 24..55
pixel x:  384..895
shape:    32x32 latent / 512x512 pixels
```

It contains the center-right duplicate lighthouse/stone structure, the train,
bridge deck, and the wide overlap with crop 2.

An initial reproduction check stopped before any diagnostic forward because a
CPU-reduced vector norm differed from the prior GPU-reduced norm by `0.00061`
while RMS differed only `5.96e-8` and extrema were exact. The corrected check
uses the reproduction sampler's GPU-side accepted-state record—the same
boundary as Phase 2e—and matched exactly.

## Exact variants

```text
A LOCAL-ONLY
  ordinary crop execution

B LOCAL + COMPACT GLOBAL K/V
  global input [1,128,16,32]
  512 generated global K/V tokens

C LOCAL + DENSE FULL-CANVAS K/V
  global input [1,128,32,64]
  2048 generated global K/V tokens
```

The dense full-canvas capture output also supplies the reference crop. No
learned compressor, output-space fusion, sampler update, stale context, block
selection, or sparse execution was used.

Implementation and outputs:

- `flux2_candidate2_dense_vs_compact_kv.py`
- `flux2_candidate2_dense_vs_compact_results/report.json`
- decoded local variants, dense reference, surrounding-context diagnostics,
  and error maps in the same directory.

The recorded command was:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate2_dense_vs_compact_kv.py
```

## Integration-path control

Compact and dense context use the same `OneEvaluationContextProbe` class and
the same methods for:

- retaining the ordinary local/text Q path;
- applying source-specific RoPE to global K before concatenation;
- ordering K/V as `[text, local-generated, external-global-generated]`;
- recomputing/restoring ordinary text-query attention output;
- modifying generated-query attention only;
- executing all five double and twenty single blocks;
- leaving local MLPs, projections, residuals, normalization, and final
  projection unchanged.

Only the global source input, its trained full-canvas coordinates, and generated
token density differ. Per block:

| Quantity | Compact K/V | Dense K/V |
|---|---:|---:|
| Local generated Q | 1024 | 1024 |
| Text tokens | 512 | 512 |
| External generated K/V | 512 | 2048 |
| Total Q | 1536 | 1536 |
| Total K/V | 2048 | 3584 |
| Q x K | `1536 x 2048` | `1536 x 3584` |

All 25 block records confirm these dimensions. Compact global IDs span the
whole target extent through the qualified scaled RoPE mapping; dense IDs use
the ordinary `32 x 64` full-canvas grid.

## Numerical results

| Variant versus dense crop | RMS | Mean abs | Low-frequency RMS | Relative residual/dense norm | Prediction RMS |
|---|---:|---:|---:|---:|---:|
| Local-only | 0.416 | 0.255 | 0.228 | 0.489 | 0.848 |
| Compact global K/V | 0.372 | 0.237 | 0.188 | 0.438 | 0.819 |
| Dense global K/V | 0.180 | 0.089 | 0.099 | 0.212 | 0.846 |

Dense K/V reduced RMS error by approximately 52% and low-frequency RMS by
approximately 47% relative to compact K/V. Its prediction norm closely matches
the dense crop (`0.846` versus `0.851` RMS), with no instability.

The context interventions were both substantial rather than inert:

| Change versus local-only | RMS | Low-frequency RMS |
|---|---:|---:|
| Compact K/V | 0.321 | 0.158 |
| Dense K/V | 0.371 | 0.192 |

Dense-versus-compact context predictions differ by `0.317` RMS and `0.145`
low-frequency RMS.

## Visual observations

- **Local-only:** independently predicts a small white lighthouse directly
  beside a dark stone tower/island in the center-right water. Train cars and
  bridge cables/deck differ from the dense crop.
- **Compact K/V:** improves train continuity, deck/cable alignment, and horizon,
  but retains both the small white lighthouse and dark stone structure. This is
  the semantic alternative that later survives the compact-context trajectory.
- **Dense K/V:** removes the small lighthouse. Train placement, bridge slope,
  cable spacing, deck, horizon, and water align substantially more closely with
  dense. It retains a reduced dark stone tower, so the result is not exact.
- **Dense reference crop:** contains no lighthouse and only a tiny distant
  island/silhouette near that position. The intended left lighthouse and right
  stone tower lie outside this crop.
- All three local variants retain sharp train windows, bridge truss/cables, and
  water texture. The dense-context improvement is not caused by smoothing away
  local detail.

The surrounding-context images insert each crop into the same dense prediction
only for interpretation. Their hard boundary is not fusion or a sampler result.

## Interpretation

The decisive comparison follows the task's compactness discriminator:

```text
dense global K/V removes duplicate lighthouse
compact global K/V retains duplicate lighthouse
```

Therefore the current 512-token global representation loses or fails to express
enough late-stage object-count/spatial evidence for this crop. Candidate 2's
external-context mechanism remains viable, and the next research problem is
global representation quality/density rather than deeper local/global hidden
coupling as the first response.

The residual dense-K/V stone tower is an important limitation: appending even
complete global K/V does not force exact equality with dense, because local
hidden/MLP trajectories remain independent. This result does not establish that
increasing density alone will solve every semantic divergence.

## Single next experiment

Repeat this same no-update late-state diagnostic with one fixed intermediate
whole-canvas grid, `24 x 48` latent tokens (384 x 768 pixels, 1152 generated
tokens), between compact 512 and dense 2048. Use the identical integration path
and affected crop. This smallest density test can show whether the lighthouse
failure disappears before dense cost, without beginning a resolution sweep or
trajectory run.

**COMPACT INFORMATION LOSS**
