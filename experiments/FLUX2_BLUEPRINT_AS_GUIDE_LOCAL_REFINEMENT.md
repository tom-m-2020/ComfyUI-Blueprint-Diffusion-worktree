# Phase 39 — Blueprint-as-Guide local refinement discriminator

## Executive result

**E — ARCHITECTURAL BOUNDARY**

The audit found no defensible fixed same-canvas Blueprint-guidance mechanism
for an independently initialized native FLUX.2 Klein 4B local state under the
allowed interface. The only trained native image-conditioning path is
`reference_latents`, whose tokens deliberately represent a separate indexed
image/canvas. Same-canvas alternatives are either already-tested
resampling/projection, require an arbitrary guidance metric/strength, or use
untrained transformer/backend intervention.

Stage 2 therefore selected no executable mechanism. No diffusion inference,
decode, or CUDA model work was performed.

## Control provenance

The persisted square Phase-29 control was validated without recomputation:

- mapped Blueprint hash:
  `8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05`
- final control hash:
  `1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3`
- Phase-28 regression: bit-exact
- semantic grade: S3

This establishes A, but there is no B arm because no candidate passed the
fixed-contract gate.

## Native execution boundary

An ordinary qualified local call contains only:

```text
generated local state x_t [1,128,64,64]
sigma/timestep
prepared text conditioning
native local positional IDs
model options
```

Klein projects generated tokens through `img_in`, jointly attends with text,
and returns one model prediction. `BasicGuider` and CONST provide no second
same-canvas structural tensor or structural likelihood.

ComfyUI exposes generic intervention points:

- sampler CFG/post-CFG functions;
- an apply-model wrapper;
- `post_input` and attention patches;
- whole-block replacement;
- block residual `control` inputs;
- `reference_latents`.

These hooks prove mechanical accessibility, not that the Klein checkpoint was
trained to interpret a Blueprint at those boundaries.

## Candidate mechanism audit

| Candidate | Blueprint tensor | Independent local tensor | Interaction | Model-mediated detail? | Same-canvas? | Fixed contract? | Cost / disposition |
|---|---|---|---|---|---|---|---|
| Prediction/denoised guidance | lifted/mapped Blueprint x0 | independently generated `x0_free` | post-model prediction correction or loss gradient | Only the free branch | Yes | **No** | At least one local call, often backward work; requires a structural metric and free scale. Phase 30 already established this boundary. |
| Latent/state constraint | mapped Blueprint state | independent `W_t` | replace/project/blend W before or between calls | Model sees constrained state | Yes | **No** | Hard replacement is resampling/projection already tested; soft constraint requires strength/schedule. |
| Transformer feature/residual guidance | projected Blueprint hidden features | local hidden stream | `post_input`, block residual, or block replacement | Yes | Mechanically | **No** | One or more local calls plus Blueprint feature work; mapping, normalization, depth, and amplitude are untrained/underdetermined. Requires custom model options/backend behavior. |
| Same-canvas attention/K/V | Blueprint token K/V or hidden tokens | local query/state tokens | attention augmentation/joint token execution | Yes | Only by custom construction | **No** | Increases key cardinality and needs custom block execution. There is no trained Blueprint token role, routing mask, or fixed positional/normalization contract. Prior normalized-W external-K/V work already failed coherence. |
| Native `reference_latents` | Blueprint supplied as reference latent tokens | ordinary generated local state | concatenated image-reference tokens in native joint attention | Yes | **No** | Native but wrong relation | Extra tokens and attention work. Trained, but each reference carries a distinct image index and its own canvas coordinates; inappropriate for required same-canvas authority. |
| ControlNet/adapter residual | adapter-produced Blueprint residuals | local hidden stream | residual added after native blocks | Yes | Potentially | **No available qualified model** | Base Klein provides residual sinks, not a trained Klein 4B same-canvas control encoder. Handcrafted residuals are OOD. |

## Why reference conditioning is rejected

For native FLUX reference processing, generated tokens use image index zero.
With Klein's default `index` method, each reference advances by
`ref_index_scale=10`, retaining local y/x coordinates but acquiring a distinct
image-index RoPE coordinate. ComfyUI concatenates these tokens to generated
tokens before transformer interaction and returns predictions only for the
generated prefix.

This is a coherent trained **reference-image** contract. It is not the required
claim that Blueprint `(y,x)` and local `(y,x)` describe the same canvas or that
the reference is structurally authoritative. Reindexing it onto the generated
canvas would change the trained positional role and become a new unqualified
attention intervention.

## Why feature/KV intervention is not selected

The earliest accessible hidden boundary is `post_input`, after generated and
reference values have been independently projected. ComfyUI can also patch
attention Q/K/V or outputs and replace whole blocks. None supplies:

1. a trained Blueprint-role/type embedding;
2. a fixed mapping from Blueprint latent values to a compatible residual;
3. a fixed coupling amplitude;
4. a structural metric or likelihood variance;
5. an authoritative same-canvas routing rule.

Using `img_in(Blueprint)` as an additive generated residual does not solve the
problem. `img_in` is linear and bias-free for the qualified checkpoint, so an
additive scaled projection is equivalent to changing the generated latent
input itself. It collapses back toward state interpolation and still needs a
scale.

Appending Blueprint K/V without a scale is not parameter-free in the relevant
sense: token duplication/cardinality, positional identity, query eligibility,
and softmax competition define an implicit coupling policy. No native trained
contract picks those choices.

## Expected execution implications if these boundaries were crossed

- Prediction attractor: one ordinary forward plus optional backward/reference
  evaluation per region; no fixed memory claim without the chosen loss.
- State constraint: ordinary calls, but semantically resampling/projection.
- Feature residual: ordinary or explicit-block forward plus Blueprint feature
  construction; hooks can retain substantial intermediates if implemented
  naively.
- Attention/KV: local query attention grows by Blueprint key count; exact
  block-major execution would avoid all-layer caches but is backend-specific.
- Native references: one ordinary transformer call with extra reference tokens
  at every block; attention and memory grow with the combined sequence.
- Trained adapter: additional encoder/residual memory and compute determined by
  a checkpoint that is not currently present or qualified.

No empirical timing or VRAM measurement is reported because no mechanism
passed Stage 2.

## Architectural answer

Under the current released native Klein 4B T2I/edit contract, Blueprint cannot
be both a same-canvas authoritative plan and a distinct guide to an independent
local generative state without adding one of:

- a free empirical constraint/guidance policy;
- an untrained custom hidden/attention intervention;
- a specifically trained same-canvas control/adapter interface;
- a different global/local lifecycle.

The first two violate this discriminator's fixed-contract requirement. The
third is not available in the audited model. The fourth belongs to a different
architecture.

This does not prove that Blueprint-as-guide is impossible after training or
backend research. It establishes that the required semantic relation is absent
from the qualified ordinary native interface; engineering hooks alone do not
create it.

## Integrity

- Diffusion model forwards: 0
- Local forwards: 0
- Destination-sized forwards: 0
- Decoded images: 0
- Control recomputation: 0
- Production changes: none
- ComfyUI-core changes: none

Machine-readable evidence is in
`flux2_blueprint_as_guide_results/report.json`; the companion audit script
validates persisted control fingerprints and exact source markers/hashes.

## Exactly one next discriminator

Run a **trained same-canvas Klein control-adapter availability and contract
audit**. Its sole question should be whether any released, architecture-matched
FLUX.2 Klein 4B adapter/checkpoint was trained to consume a spatially aligned
latent/image plan and inject compatible residuals through the native blocks.
Do not run inference unless such a trained interface and its canonical input
normalization/strength are established. If none exists, close this guide family
and move to the separately designed interleaved global/local architecture.

## Verdict

**E — ARCHITECTURAL BOUNDARY**
