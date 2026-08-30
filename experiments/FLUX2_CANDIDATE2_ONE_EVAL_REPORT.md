# FLUX.2 Candidate-2 one-evaluation compact-global-context probe

## Verdict

**COMPACT GLOBAL CONTEXT WORKS** for the tested first/high-noise evaluation and
left crop.

Giving local generated queries direct access to same-sigma compact whole-canvas
K/V changed what the transformer predicted before final projection. The local
control invented a large dark tower inside the left crop and flattened the
bridge organization. Compact context removed that tower, retained the single
left lighthouse, and moved bridge slope/deck geometry visibly and numerically
toward the ordinary dense reference while preserving local cable and water
structure.

This is positive semantic evidence, not trajectory, efficiency, or production
qualification. Only one crop, seed, prompt, sigma, and evaluation were tested.

## Controlled setup

The probe reuses Phase 2/2b exactly where applicable:

- FLUX.2 Klein 4B W4A8, Qwen conditioning, and FLUX.2 VAE;
- the bridge/train/lighthouse/tower composition-stress prompt;
- seed 20260829, CFG 1, and the first four-step Euler sigma (`1.0`);
- 1024 x 512 target (`32 x 64`, 2048 generated tokens);
- 512 x 256 compact global (`16 x 32`, 512 generated tokens);
- the left 512 x 512 crop (`32 x 32`, 1024 generated tokens), with full-canvas
  offsets `(x=0,y=0)`.

There was no sampler update. One native ComfyUI conditioning/model lifecycle
made four calls at the identical current latent and sigma:

```text
C dense full canvas
A ordinary local crop
  compact-global capture
B local crop with compact-global generated K/V
```

The recorded run command was:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate2_one_eval_probe.py
```

Implementation and raw evidence:

- `flux2_candidate2_one_eval_probe.py`
- `flux2_candidate2_one_eval_results/report.json`
- decoded crop and surrounding-global-context diagnostics in the same output
  directory.

## Exact transformer intervention

The compact branch first runs normally. At every attention layer, the probe
captures only its **generated** K/V after K has received the compact branch's
own RoPE. Text K/V are not duplicated.

The subsequent local call retains its ordinary text and local generated Q/K/V.
After applying local RoPE, attention uses:

```text
Q ordering:   [text, local-generated]
K/V ordering: [text, local-generated, compact-global-generated]
```

Thus local generated Q attend to both their native local/text context and 512
compact whole-canvas generated K/V tokens. Ordinary attention output for text
queries is separately computed and restored; only generated-query attention
output consumes the appended global context. Downstream projections, MLPs,
residuals, normalization, and final projection execute normally for all local
tokens.

The intervention covers **all five double-stream and all twenty single-stream
blocks**. This probe therefore does not establish that one block, early blocks,
or a smaller subset is sufficient.

For experimental clarity, all 25 layers' global K/V were retained between the
immediately adjacent global and local calls. This is same-evaluation feature
retention, not cross-step caching. It is deliberately not memory optimized.
The ordinary local attention is recomputed inside each block to restore text
query output, so the probe is not a compute-saving implementation.

## Positions, token counts, and attention dimensions

ComfyUI revision: `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`.

All branches used 512 text tokens. Recorded generated IDs were:

| Branch | Generated ID shape | First ID | Last ID | Spatial extent |
|---|---:|---|---|---|
| Dense | `[1,2048,4]` | `[0,0,0,0]` | `[0,31,63,0]` | y 0–31, x 0–63 |
| Local | `[1,1024,4]` | `[0,0,0,0]` | `[0,31,31,0]` | y 0–31, x 0–31 |
| Compact global | `[1,512,4]` | `[0,0,0,0]` | `[0,31,63,0]` | y 0–31, x 0–63 |

The compact branch obtains the full target extent using RoPE scaling
`scale_y=2.0666667`, `scale_x=2.0322581`; the left local crop uses zero x/y
shift at native density. Text IDs span `[0,0,0,0]` to `[0,0,0,511]`.

At every modified block:

| Quantity | Tokens |
|---|---:|
| Local generated Q | 1024 |
| Compact global generated K/V | 512 |
| Total Q including text | 1536 |
| Ordinary local K/V including text | 1536 |
| Augmented K/V including text | 2048 |
| Augmented Q x K | `1536 x 2048` |

The local PE tensor was `[1,1,1536,64,2,2]`; compact K was rotated with the
compact branch's corresponding PE before retention. Head count and head
dimension were recorded per block in the machine report.

## Numerical results

All predictions were finite. The intervention was large enough to be semantic,
not a negligible perturbation:

| Comparison | Mean abs | Max abs | RMS | Low-frequency RMS |
|---|---:|---:|---:|---:|
| Context vs local-only | 0.369 | 7.555 | 0.617 | 0.484 |
| Local-only vs dense crop | 0.547 | 10.508 | 0.867 | 0.657 |
| Context vs dense crop | 0.367 | 8.078 | 0.582 | 0.396 |

Compact context reduced total RMS error to the dense crop by about 33% and
low-frequency RMS error by about 40% relative to local-only. Relative residual
to the dense prediction norm fell from `0.825` to `0.553`. These tensor metrics
do not alone establish semantic correctness; the decoded diagnostics agree
with their direction.

Prediction RMS remained stable: `1.0731` local-only, `1.0725` with compact
context, and `1.0390` for full dense. There was no norm explosion or obvious
loss of local high-frequency structure.

Peak allocation is not a candidate VRAM result. It was approximately 2.50 GiB
for local-only and 3.76 GiB for local-with-context because this simple probe
retains all layers' compact K/V concurrently. A streaming or family-specific
executor would have a different memory lifecycle and was intentionally not
built here.

## Visual inspection

- **Local-only:** one lighthouse appears on the left, but a large dark
  tower/support is independently invented at the local right boundary. The
  bridge deck is comparatively flat and its cable organization conflicts with
  the dense/global plan.
- **Compact context:** the invented dark tower disappears. The bridge rises on
  the left and descends toward the right like the dense crop; the single left
  lighthouse and water horizon remain. Fine suspension lines, deck structure,
  shoreline, and water texture remain visible.
- **Dense reference crop:** has the clearest left bridge peak/slope and one
  lighthouse. Compact context does not reproduce it exactly, but follows the
  same broad organization substantially better than local-only.
- **Right tower and centered train:** the chosen left crop does not contain the
  intended far-right tower or most of the centered train. No claim is made for
  those regions. The local-only crop's extra dark tower is nevertheless a
  direct instance of the unwanted crop-local semantic invention.
- The surrounding-global composites are placement diagnostics only. Their hard
  crop boundary is not a proposed fusion method or sampler result.

## Interpretation and limits

This probe answers the narrow objective positively: same-current-state compact
global context can change the internal local transformer prediction toward
whole-canvas scene organization, rather than attempting to repair a divergent
prediction afterward. Unlike the Phase 2b frequency split, the improvement
occurs before output projection and suppresses a concrete duplicate-object
alternative.

It does not establish:

- behavior for the center/right crops, later sigmas, or a persistent sampling
  trajectory;
- which blocks need context, whether context strength requires control, or
  whether compact K/V remain authoritative under all prompts;
- compute savings, acceptable VRAM, cache lifetime, or production feasibility;
- equivalence to dense generation.

Candidate 2 receives positive evidence and becomes the leading semantic
mechanism to test. No generalized sparse executor, cache system, node, sampler,
or cross-model abstraction is justified yet.

## Single next experiment

At the same first sigma, apply the identical fresh compact-global K/V to all
three existing crops and assemble their predictions with the existing overlap
weights. Compare that one assembled prediction against tiled-only and dense.
This is the smallest test of whether the observed suppression generalizes
across lighthouse, train, and right-tower regions and actually reduces
cross-region duplication before adding a sampling trajectory.

**COMPACT GLOBAL CONTEXT WORKS**
