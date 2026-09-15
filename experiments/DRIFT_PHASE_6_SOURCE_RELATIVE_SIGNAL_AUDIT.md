# Phase 6 - Source-relative drift-signal audit

## Verdict

**NATIVE FEATURES NOT SUFFICIENT FOR DRIFT MEASUREMENT**

The fixed native correspondence audit detects gross portrait drift in the
Gaussian arm, but no tested representation distinguishes the still-failed
portrait FSS `r=2` arm from all four substantially successful astronaut/bridge
FSS and ILVR arms. The signals remain dominated by general appearance and
feature ambiguity rather than fine object-part identity.

No sampler state was changed and no production code was modified.

## Saved-data boundary

Phase 2 preserved decoded x0 previews at evaluations 0, 2, and 5, but did not
persist exact sampler latents or hidden tensors. Phase 6 therefore did not rerun
any diffusion trajectory. It re-encoded the saved 512x512 x0 previews through
the same fixed VAE, then used measurement-only Klein forwards at each saved
evaluation sigma to expose native spatial representations. Forward outputs were
discarded.

This introduces a documented decode/re-encode approximation. The audit measures
features of the saved visible x0 estimates, not bit-exact original model inputs.

## Native representation audit

Four preregistered spatial maps were compared without a layer sweep:

1. VAE latent `[1,128,32,32]`. This is the smallest native spatial code and
   requires no denoiser forward. Its channels are learned for reconstruction,
   not correspondence.
2. Klein `post_input/pre_double_block_0`, `[1,1024,3072]`, reshaped to
   `[1,3072,32,32]`. Each token still maps exactly to one latent coordinate,
   but this bias-free linear projection cannot create a new spatial receptive
   field or explicit correspondence semantics.
3. Output of `double_block_0`, same spatial grid. It includes the first global
   image/text attention and is already context- and prompt-dependent.
4. Output of `single_block_7`, generated-image slice on the same grid. This is
   the fixed midpoint of the 5-double-plus-20-single Klein stack. Repeated
   global mixing preserves positional indexing but not locality or object-part
   identity.

Source and current images used identical transformation conditioning and sigma
within every pair. This avoids making a source prompt an uncontrolled variable.

## Correspondence rule

For every source token, the audit selected the current token with maximum
cosine similarity inside a fixed radius-4 window (at most 9x9). It recorded:

- mean and p90 Euclidean displacement in latent-token units;
- source-gradient-weighted mean displacement;
- best cosine and the best-minus-second-best match margin;
- same-position cosine distance;
- displacement in top-quartile source-gradient cells as a structure-bearing
  source-region proxy;
- displacement in bottom-quartile gradient cells as a background proxy.

The gradient regions are diagnostic proxies, not semantic segmentation. No
face model, landmark model, CLIP, DINO, optical flow, or external learned model
was used.

## Final-evaluation results

Mean displacement in 32x32 token units at evaluation 5:

| Known outcome | VAE latent | post-input | double 0 | single 7 |
|---|---:|---:|---:|---:|
| Portrait Gaussian - fail | **3.200** | **3.264** | **2.983** | **3.149** |
| Portrait FSS r=2 - fail | 2.348 | 2.438 | 2.120 | 1.997 |
| Astronaut FSS r=2 - success | 2.050 | 2.101 | 1.809 | 1.673 |
| Astronaut ILVR - success | 2.234 | 2.301 | 2.080 | 2.071 |
| Bridge FSS r=2 - success | 2.155 | 2.227 | 1.980 | 1.792 |
| Bridge ILVR - success | 2.420 | 2.473 | 2.181 | 2.039 |

Portrait Gaussian is the largest-drift case at every representation. Portrait
FSS is not separable: bridge ILVR has larger VAE, post-input, and double-block
displacement, while astronaut and bridge ILVR have larger midpoint displacement.
The same overlap occurs for gradient-weighted and structure-proxy displacement.

Final same-position cosine distance tells the same story. Portrait Gaussian is
high (`0.808/0.726/0.324/0.133`), but portrait FSS
(`0.517/0.421/0.193/0.095`) overlaps successful astronaut/bridge ILVR arms.
This is a broad content-change measure, not a reliable fine-identity detector.

## Confidence and spatial behavior

The transformer representations become more self-similar with depth, but their
correspondences become less decisive. At evaluation 5, mean best cosine for
single block 7 is `0.898-0.933`, while mean best-minus-second-best margins are
only `0.0041-0.0092`. High cosine with near-tied candidates means the selected
displacement is unstable among many similar globally mixed tokens.

Structure-rich versus background proxies are inconsistent across scene types.
At midpoint, portrait Gaussian shows larger structure-region displacement than
background (`3.549` versus `2.917`), but portrait FSS shows the opposite
(`1.364` versus `2.184`). The successful astronaut and bridge cases exhibit
both directions as well. The proxy therefore cannot isolate object-part drift.

## Trajectory evolution

Gross separation is already visible at evaluation 0: portrait Gaussian VAE
displacement is `3.307`, while portrait FSS is `2.607`. Across evaluations
0/2/5, the VAE/post-input measures generally decrease modestly for FSS/ILVR as
the saved x0 estimates stabilize. They do not develop a new late separation
between failed portrait FSS and successful geometry cases.

Deeper features reduce absolute displacement and same-position cosine distance,
but this reflects contextual homogenization rather than more trustworthy
correspondence: match margins collapse by roughly an order of magnitude from
VAE/post-input to transformer features.

## Interpretation and stopping boundary

Native features provide a partial gross-drift indicator, especially the VAE
latent and its post-input projection. They do not satisfy the preregistered
requirement for a useful fine-geometry signal because the decisive portrait FSS
failure is not ranked above all known geometry successes.

Arbitrary alternate layers, search radii, feature normalizations, or semantic
region heuristics were not tuned after observing results. No constraint is
implemented. A future phase would need explicit authorization to investigate a
different correspondence representation or an external trained vision model.

Artifacts:

- `experiments/DRIFT_PHASE_6_PREREGISTRATION.json`
- `experiments/drift_native_spatial_signal_audit.py`
- `experiments/drift_phase_6_native_signal_results/telemetry.json`
