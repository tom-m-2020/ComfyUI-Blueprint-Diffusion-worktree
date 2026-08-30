# FLUX.2 Candidate-2 minimal four-step trajectory

## Verdict

**CANDIDATE 2 TRAJECTORY PARTIALLY WORKS.**

Fresh compact-global context kept the tiled trajectory materially closer to
dense than tiled-only at every Euler evaluation, reduced pre-blend cross-crop
disagreement at every step, suppressed the major crop-local bridge/support
alternatives, and retained detailed bridge, train, tower, lighthouse, and water
structure in the final image. It did not fully suppress semantic duplication:
a second small lighthouse-like object emerged near the center-right during
later denoising and survived the final output.

This passes the numerical and detail-retention portions of the trajectory gate,
but only partially passes the object-uniqueness criterion.

## Controlled setup

All three trajectories reused the Phase 2/2d configuration:

- FLUX.2 Klein 4B W4A8 with the same Qwen text conditioning and VAE;
- prompt, seed 20260829, CFG 1, and four-step zero-churn Euler schedule;
- 1024 x 512 target (`32 x 64`, 2048 generated tokens);
- 512 x 256 compact global (`16 x 32`, 512 generated tokens);
- three 512 x 512 crops (`32 x 32`, 1024 generated tokens) at latent x offsets
  `0`, `24`, and `32`;
- 128-pixel overlap and the exact normalized Phase-2 ramp weights;
- full-canvas crop offsets and compact-global full-extent RoPE mapping.

The variants were:

```text
A dense:                 1 full-canvas forward/evaluation
B tiled-only:            3 local forwards -> one assembly -> one update
C global-context tiled:  1 fresh compact capture + 3 context local forwards
                          -> one assembly -> one update
```

There is no output-space global prediction/residual fusion. No crop updates the
latent independently.

Implementation and raw outputs:

- `flux2_candidate2_four_step_trajectory.py`
- `flux2_candidate2_four_step_results/report.json`
- three final images and all twelve per-step denoised estimates in the output
  directory.

The recorded command was:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate2_four_step_trajectory.py
```

## Lifecycle verification

At each context evaluation, the probe:

1. started from the current accepted full-canvas latent;
2. downsampled that exact tensor to the compact-global input;
3. captured 25 layers of generated K/V at the current sigma;
4. gave the same K/V tensor objects to all three crop calls;
5. assembled one full-canvas prediction;
6. cleared the global K/V owner before the Euler update;
7. consumed that assembled prediction in exactly one update.

Weak-reference checks at the beginning of evaluations 1–3 confirmed that no
prior-evaluation K/V tensor survived. Every evaluation recorded exactly 25
capture records, 75 context-consumption block records, the same object identity
for all three crops, and `previous_capture_survived=false`.

Each trajectory performed exactly four accepted updates. Model forwards per
evaluation were exactly `1 / 3 / 4` for dense/tiled/context. Executed generated
image-token work was respectively `2048 / 3072 / 3584`. Context attention in
all five double and twenty single blocks remained `1536 Q x 2048 K`, including
512 text tokens, 1024 local generated tokens, and 512 compact-global K/V tokens.

Coverage min/max/mean was exactly `1.0` for every tiled and context evaluation.
All predictions, accepted latents, and final samples were finite.

An initial run exposed a device-only harness error: the Phase-2d diagnostic
assembler allocated CPU tensors while live sampler predictions were CUDA. It
stopped before the first tiled update. A trajectory-local assembler now uses
the prediction device with the identical geometry and weight formula; the
complete run then passed all lifecycle assertions.

## Per-evaluation results

The reference at each row is the dense trajectory's prediction at the same
schedule evaluation. Because trajectories diverge after step 0, these metrics
include both current-latent trajectory divergence and prediction divergence;
that is the intended end-to-end trajectory comparison.

| Step | Sigma | Tiled RMS vs dense | Context RMS vs dense | Tiled low-freq RMS | Context low-freq RMS | Tiled overlap RMS | Context overlap RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.0000 | 0.758 | 0.569 | 0.547 | 0.386 | 0.849 | 0.312 |
| 1 | 0.9614 | 0.793 | 0.601 | 0.474 | 0.303 | 0.734 | 0.382 |
| 2 | 0.8926 | 0.840 | 0.643 | 0.488 | 0.327 | 0.548 | 0.291 |
| 3 | 0.7348 | 0.864 | 0.652 | 0.502 | 0.329 | 0.304 | 0.181 |

Context is closer to dense on total and low-frequency prediction error at all
four evaluations. It also reduces overlap disagreement by approximately
41–63% at every evaluation.

## Per-crop RMS against dense crop

| Step | Crop 0 tiled/context | Crop 1 tiled/context | Crop 2 tiled/context |
|---:|---:|---:|---:|
| 0 | 0.867 / 0.582 | 0.941 / 0.600 | 0.679 / 0.565 |
| 1 | 0.773 / 0.630 | 0.914 / 0.609 | 0.836 / 0.594 |
| 2 | 0.800 / 0.655 | 0.932 / 0.632 | 0.896 / 0.633 |
| 3 | 0.818 / 0.671 | 0.921 / 0.624 | 0.920 / 0.637 |

Every crop remains numerically closer to its corresponding dense region at
every evaluation. The context benefit persists rather than being limited to
the first-sigma plan.

Final-latent RMS versus dense was `0.864` tiled-only and `0.652` context;
low-frequency RMS was `0.502` and `0.329`. Final latent prediction RMS stayed
finite and comparable: dense `0.942`, tiled `0.930`, and context `0.895`.

Peak allocation was approximately 2.61 GiB dense, 2.50 GiB tiled-only, and
3.77 GiB context. The context probe retains all layers' compact K/V and
recomputes ordinary text attention, so this is not an optimized or favorable
VRAM result. Recorded wall times are also not comparable because dense was the
first/warm-up trajectory.

## Semantic trajectory inspection

### Dense

Dense resolves one continuous symmetric suspension bridge, one centered yellow
train, one white lighthouse on the left, and one dark stone tower on the right.
The deck, cables, horizon, and water remain continuous.

### Tiled-only

- Multiple incompatible suspension spans and a dominant extra central support
  emerge.
- The yellow train appears as separated/duplicated train segments on different
  bridge pieces.
- A second lighthouse appears toward the right-center in addition to the left
  lighthouse.
- The bridge deck, cable slopes, and support placement conflict across crop
  regions despite normalized blending.
- Local texture is sharp, but it belongs to incompatible crop-local scenes.

### Global-context tiled

- The final bridge is one continuous deck with a coherent left-to-right cable
  system and only the intended major suspension towers.
- One long yellow train develops across the center-left/center deck. It remains
  detailed and substantially more continuous than tiled-only's separated train
  segments.
- The intended left lighthouse and right stone tower remain correctly placed.
- Water, horizon, bridge truss, vertical cables, masonry, and train windows
  develop clear late-stage detail. Compact context is not simply smoothing the
  local outputs.
- However, a small second lighthouse-like object emerges near the center-right
  at step 1, becomes clearly lighthouse-shaped by steps 2–3, and survives the
  final image. Semantic duplication is reduced but not eliminated.
- Exact bridge geometry still differs from dense: context produces a broader
  asymmetric span and a longer train.

## Interpretation

Fresh compact-global context remains semantically useful throughout a real
shared-latent Euler trajectory. It is not merely a first-sigma visualization
effect, and it does not prevent later local fidelity from developing.

The surviving extra lighthouse prevents a full pass. The next uncertainty is
whether this failure is caused by the compact representation losing decisive
object-count information, or by the external-K/V attention mechanism being
insufficient even with complete global information. Optimizing K/V lifetime,
selecting blocks, or building sparse execution before answering that question
would be premature.

## Single next experiment

At the late evaluation where the second lighthouse becomes clear, compare the
same three local crops using (a) the current 512-token compact-global K/V and
(b) generated K/V from an ordinary dense full-canvas forward of the identical
current latent and sigma. Assemble once without updating. If dense global K/V
removes the duplicate, compactness/token density is the limiting factor; if it
does not, the external-context integration policy is insufficient. This is a
diagnostic reference only, not an efficient candidate.

**CANDIDATE 2 TRAJECTORY PARTIALLY WORKS**
