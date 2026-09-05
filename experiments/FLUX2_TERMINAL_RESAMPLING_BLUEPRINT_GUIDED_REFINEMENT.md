# Phase 30 — Fixed Blueprint-guided local refinement discriminator

## Decision

**E — ARCHITECTURAL BOUNDARY**

The guided arm was stopped before inference. Under the qualified native
FLUX.2 Klein T2I interface and the explicit Phase-30 exclusions, there is no
well-defined, parameter-free denoised-prediction guidance rule that keeps the
Blueprint available during refinement without becoming one of the already
excluded mechanisms.

No production or ComfyUI-core file was modified. No Phase-29 model branch was
recomputed.

## Reused control validation

The Phase-29 `sigma=0.25` artifacts were fingerprinted before reuse. For all
three cases:

- the saved terminal branch sigma is exactly `0.25`;
- its mapped-Blueprint hash matches the persisted Blueprint record;
- its Phase-28 comparison is recorded as bit-exact (`RMS=0`, `max_abs=0`);
- its latent is finite;
- the persisted Blueprint, latent, decoded image, region seeds, noise hashes,
  working-state hashes, and restricted-prediction hashes remain available;
- the control semantic grade remains S3.

| Control | Geometry | Regions | Final latent hash | Grade |
|---|---|---:|---|---:|
| square multi-object | `H=128x128`, `B=45x45` | 25 | `1b61a401451c…` | S3 |
| portrait astronaut | `H=256x128`, `B=64x32` | 55 | `0a70d2c8ae1e…` | S3 |
| landscape bridge | `H=128x192`, `B=36x54` | 40 | `c4e07a2dc0b1…` | S3 |

This satisfies the control gate without rerunning inference. Because the
experimental rule is undefined under the contract, no B telemetry or semantic
claim is manufactured.

## Why the requested rule is underdetermined

Let the ordinary local prediction be `x0_free`, the lifted Blueprint reference
be `x0_bp`, and a structural guidance direction be `g(x0_free, x0_bp)`. A soft
prediction-space attractor necessarily has the form

```text
x0_guided = x0_free + lambda * g(x0_free, x0_bp)
```

or an equivalent proximal/gradient update. The CONST-flow relationship fixes
how `x0`, noise, and the sigma-state relate; it does not determine `lambda`, a
structural likelihood variance, or even a unique structural metric. Setting
`lambda=1` would still be an arbitrary strength choice in latent units, not a
model-native consequence. Backpropagating a loss through the denoiser does not
remove this missing scale.

The available alternatives were rejected before inference:

1. **Scalar blend.** Explicitly prohibited and requires `alpha`.
2. **Exact/coarse output projection.** This is hard Blueprint authority already
   investigated in Phases 20b/20c/21, not soft in-process guidance.
3. **Gradient/score attractor.** Requires both a chosen structural loss and an
   unqualified guidance/likelihood scale. FLUX CONST flow supplies neither.
4. **Native `reference_latents`.** Current native FLUX code can append reference
   image tokens to the transformer. That changes prepared conditioning and
   supplies image attention context/K/V, while Phase 30 explicitly prohibits
   global K/V context and requires the same conditioning. It is also not the
   requested denoised-prediction-space rule.
5. **Transformer hooks or patches.** These are model surgery outside the
   qualified ordinary-call path and would add a second confound.

The native BasicGuider path otherwise forwards the local state, timestep,
prepared text conditioning, and model options. It exposes no trained
same-canvas Blueprint-attractor input for the qualified T2I call.

## Measured facts versus inference

Measured facts:

- all three Phase-29 controls and fingerprints validate;
- native FLUX has an optional reference-token route, but it appends reference
  image tokens into transformer attention;
- the production terminal adapter rejects reference/edit conditioning and
  transformer/context patches for the qualified path;
- no model inference, CUDA allocation, or new model call occurred in Phase 30.

Architectural inference:

- a soft attractor cannot be derived uniquely from the current flow contract;
- choosing one would silently introduce the very guidance-strength degree of
  freedom the task forbids;
- the clean next step is to decide and qualify a specific trained/native image
  guidance interface as a separate architecture, or move upstream to the
  Blueprint-to-working-canvas lift/representation as prescribed for boundary
  outcome E.

## Artifacts

- Validator: `experiments/flux2_terminal_resampling_blueprint_guided_refinement.py`
- Machine record: `experiments/flux2_terminal_resampling_blueprint_guided_refinement_results/report.json`
- Control/boundary sheet: `experiments/flux2_terminal_resampling_blueprint_guided_refinement_results/PHASE30_COMPARISON.jpg`
- Detail/boundary sheet: `experiments/flux2_terminal_resampling_blueprint_guided_refinement_results/DETAIL_REVIEW.jpg`

## Next recommendation

Do not tune a guidance strength. Return upstream to one narrowly specified
**Blueprint-to-working-canvas lift/initialization representation**
discriminator. If native reference-image conditioning is to be considered
instead, it must be authorized as its own experiment because it changes the
conditioning and transformer-context contract.
