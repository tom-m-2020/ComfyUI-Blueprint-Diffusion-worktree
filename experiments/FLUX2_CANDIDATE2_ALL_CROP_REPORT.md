# FLUX.2 Candidate-2 all-crop compact-global-context assembly probe

## Verdict

**ALL-CROP GLOBAL CONTEXT WORKS** at the tested first/high-noise evaluation.

One fresh same-sigma compact-global K/V set made all three local predictions
closer to their dense reference regions. When assembled with the unchanged
Phase-2 overlap weights, global-context tiled prediction had substantially
better whole-canvas agreement, bridge organization, object uniqueness, and
pre-blend overlap consistency than tiled-only.

This is strong single-evaluation semantic evidence for Candidate 2. It is not
a sampling-trajectory, efficiency, memory, or production qualification.

## Exact controlled setup

The probe reused without change:

- FLUX.2 Klein 4B W4A8, Qwen conditioning, prompt, seed 20260829, and CFG 1;
- the first four-step Euler sigma, `1.0`, with no sampler update;
- 1024 x 512 target (`32 x 64`, 2048 generated tokens);
- 512 x 256 compact global (`16 x 32`, 512 generated tokens) and full-canvas
  RoPE scaling (`scale_y=2.0666667`, `scale_x=2.0322581`);
- three 512 x 512 crops (`32 x 32`, 1024 generated tokens) at latent x offsets
  `0`, `24`, and `32`;
- 128-pixel/8-token nominal overlap and the exact normalized Phase-2 ramp
  weights, producing coverage exactly one everywhere.

One native ComfyUI conditioning/model lifecycle made eight calls:

```text
1 dense reference
3 ordinary local controls
1 compact-global capture
3 context-conditioned local calls
```

The 25 captured compact-global generated K/V pairs were retained once. Runtime
identity assertions showed the same objects and capture count across all three
context calls; no per-crop recapture occurred. There was no output-space global
prediction or residual fusion.

Implementation and raw evidence:

- `flux2_candidate2_all_crop_assembly_probe.py`
- `flux2_candidate2_all_crop_results/report.json`
- decoded full-canvas/per-crop predictions and error maps in the same output
  directory.

The recorded command was:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate2_all_crop_assembly_probe.py
```

## Transformer intervention

The intervention is unchanged from Phase 2c and affects all five double-stream
and all twenty single-stream blocks. At each block:

```text
Q:   [512 text, 1024 local generated]
K/V: [512 text, 1024 local generated, 512 compact-global generated]
```

Local and compact-global K receive their own RoPE before concatenation. The
augmented attention dimension is `1536 Q x 2048 K`. Ordinary text-query
attention output is recomputed and restored, so only local generated-query
attention output consumes global K/V. Local projections, MLPs, residuals,
normalization, and final projection remain ordinary dense local computation.

The probe retains all global layers simultaneously and recomputes ordinary
attention to restore text output. It is deliberately not an optimized executor
and makes no compute/VRAM claim.

## Full-canvas metrics

| Assembled comparison against dense | RMS | Mean abs | Low-frequency RMS | Relative residual/dense norm |
|---|---:|---:|---:|---:|
| Tiled-only | 0.758 | 0.486 | 0.547 | 0.730 |
| Global-context tiled | 0.569 | 0.358 | 0.386 | 0.548 |

Global context reduced assembled RMS error by about 25%, mean absolute error by
about 26%, and low-frequency RMS error by about 29%.

Prediction norms remained stable: dense RMS was `1.039`, tiled-only `1.044`,
and global-context tiled `1.070`. All tensors were finite. Final coverage had
min/max/mean `1.0` for both assemblies.

## Per-crop metrics

| Crop / target region | Tiled RMS vs dense | Context RMS vs dense | Tiled low-frequency RMS | Context low-frequency RMS | Context change vs local-only RMS |
|---|---:|---:|---:|---:|---:|
| 0 / left lighthouse | 0.867 | 0.582 | 0.657 | 0.396 | 0.617 |
| 1 / center bridge/train | 0.941 | 0.600 | 0.711 | 0.433 | 0.718 |
| 2 / right tower | 0.679 | 0.565 | 0.464 | 0.379 | 0.400 |

All three crops moved toward dense. Crop 2 improved less than crops 0 and 1,
but did not regress. Per-crop prediction RMS values remained in the same broad
range (`1.04–1.08`) without instability.

## Pre-blend overlap disagreement

| Crop pair / absolute overlap | Tiled RMS | Context RMS | Tiled mean abs | Context mean abs |
|---|---:|---:|---:|---:|
| 0–1 / x 24–31 | 1.076 | 0.378 | 0.689 | 0.238 |
| 1–2 / x 32–55 | 0.758 | 0.287 | 0.463 | 0.171 |
| Aggregate | 0.849 | 0.312 | 0.520 | 0.187 |

The shared global context reduced aggregate pre-blend overlap RMS disagreement
by about 63% and mean absolute disagreement by about 64%. This improvement
exists before overlap weights hide disagreement.

## Visual observations

- **Dense reference:** one continuous bridge with a high left suspension span,
  a central valley, a rising right span, exactly one white lighthouse on the
  left, and one dark stone tower on the right. The water horizon is continuous.
- **Tiled-only:** crop 0 invents a dark tower/support at its right edge; crop 1
  invents a large central suspension support and conflicting steep cable peak;
  crop 2 supplies the intended right tower. Assembly retains a dark central
  mass, incompatible bridge slopes, discontinuous cable systems, and visible
  horizon/structural disagreement through overlap regions.
- **Global-context tiled:** the crop-0 and crop-1 invented towers disappear.
  The assembly retains one left lighthouse and one right stone tower. It forms
  one continuous deck and a left-high/center-low/right-rising cable arrangement
  much closer to dense. Water and horizon structure are substantially more
  continuous, with no comparable central duplicate mass.
- **Local detail:** suspension lines, deck structure, lighthouse/tower edges,
  shoreline, and water texture remain visible. No obvious broad detail collapse
  accompanies the semantic improvement.
- **Train:** a distinct centered train is not yet interpretable in the dense,
  tiled, or context decoded x0 at this first sigma. This probe therefore cannot
  establish train placement/count control; that observation requires later
  denoising state.

The context result is not pixel-identical to dense. Its central/right cable
peaks remain softer and differ in exact geometry. Candidate 2 nevertheless
meets this stage's criterion because all regions improve, duplicate structures
are suppressed, and overlap disagreement falls materially.

## Interpretation and limits

The result isolates the intended mechanism: same-current-state compact global
context inside transformer attention makes independently executed crops more
mutually compatible without output-space global fusion. It strengthens Phase
2c from one crop to all three regions.

Still unknown:

- whether the improvement persists through actual Euler updates and later
  sigmas where train identity/detail emerges;
- whether all 25 blocks are necessary;
- how to stream or recompute global features without the probe's all-layer K/V
  retention and duplicate text-attention work;
- whether this generalizes across prompts, seeds, aspect ratios, and models.

Candidate 2 is now justified as the first architecture to test over a minimal
trajectory, but not as a production architecture.

## Single next experiment

Run the same three-crop compact-global-context assembly through the existing
four-step Euler trajectory, refreshing compact-global K/V from the current
whole-canvas latent at every evaluation and making exactly one assembled sampler
update per step. Compare dense, tiled-only, and global-context tiled for whether
the first-sigma coherence survives and whether the centered train emerges
without duplicated objects. Do not add stale caching, block selection, or
production infrastructure.

**ALL-CROP GLOBAL CONTEXT WORKS**
