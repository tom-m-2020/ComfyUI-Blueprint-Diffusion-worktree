# Blueprint Diffusion Phase 1 architecture and feasibility audit

Date: 2026-08-29

## Scope, evidence, and confidence

This is a research and architecture audit. It does not qualify a production
Blueprint Diffusion implementation and does not establish image-quality,
wall-clock, or VRAM gains.

Evidence used:

- current native ComfyUI source at commit
  `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`;
- local model implementations and model detection for FLUX.2, Z-Image, and
  Anima in that checkout;
- local official reference checkouts: ElasticDiffusion
  `45a679327caabd0a26ceca6ef4709ca5ce9e9bd5`, MultiDiffusion
  `69bcdcef437dfdbf48c53624d6bf6f397b5f4894`, and Mixture of Diffusers
  `af42292d0a8cb414f6da2eeac79be4c60afbbe48`;
- the original [ElasticDiffusion paper](https://openaccess.thecvf.com/content/CVPR2024/html/Haji-Ali_ElasticDiffusion_Training-free_Arbitrary_Size_Image_Generation_through_Global-Local_Content_Separation_CVPR_2024_paper.html),
  [MultiDiffusion project/paper](https://multidiffusion.github.io/),
  [DyPE paper](https://arxiv.org/abs/2510.20766), and official model cards for
  [FLUX.2 Klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B),
  [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), and
  [Anima](https://huggingface.co/circlestone-labs/Anima);
- prior local SpotEdit work, carried over explicitly as FLUX.2 engineering
  evidence rather than as a Blueprint algorithm decision.

No new model forward was required for this audit. Z-Image and Anima are source
classified, not runtime-qualified for selective execution.

Terminology in this document follows the project definition: the full-canvas
state is the one persistent sample updated by the sampler; a global
representation is information available for whole-scene reasoning; local
execution is model work whose expensive token/pixel-dependent operations are
actually restricted, not a dense forward followed by an output mask.

## Executive conclusion

The smallest credible Blueprint architecture is not “tile and blend.” It is a
sampling-lifecycle coordinator that owns one full-canvas latent and, for every
accepted denoising step, constructs one full-canvas model prediction from:

1. a cheap, spatially registered whole-canvas prediction or context;
2. genuinely local or selected-token corrections at the same sigma and under
   the same conditioning contract; and
3. an explicit fusion rule before the ordinary guider/sampler update.

The sampler can remain native. Model execution cannot remain wholly generic:
crop execution is broadly expressible, but selected-query execution and hidden
state/K/V reuse require a family-specific transformer executor. FLUX.2's
`patches_replace["dit"]` boundary is unusually useful; Z-Image has a different
joint-stream block body and Anima has separate image self-attention and text
cross-attention, so neither should be forced through FLUX-shaped block APIs.

The strongest first candidate is a reduced-resolution global trajectory plus
overlapping local full-model corrections. It is the least invasive way to test
the core semantic hypothesis. It may reduce peak VRAM, but it reduces total
compute only when global tokens remain bounded and local coverage, overlap,
and refresh frequency stay substantially below dense full-canvas work.

The most promising eventual compute architecture is selected image queries
attending to compact or cached global context while running projections, MLPs,
residuals, and final projection only for selected tokens. It has higher upside
and much higher implementation and cache-validity risk.

T2I and reference/edit can share the same coordinator, geometry, persistence,
fusion, and sampler contracts. They should not share a single global-context
policy. T2I must create and evolve a scene plan from noise and text. Reference,
inpainting, and some editing tasks can use known source geometry as additional
global evidence, but reference tokens are not the same object as the evolving
same-canvas latent and do not prove that the generated canvas is globally
coherent.

## 1. Native ComfyUI sampling and synchronization boundary

The generic native path is:

```text
noise + optional latent image
  -> CFGGuider.outer_sample / prepare_sampling
  -> sampler evaluation at sigma
  -> KSamplerX0Inpaint (optional same-canvas mask mixing)
  -> CFGGuider.predict_noise
  -> sampling_function
  -> calc_cond_batch (areas, hooks, cond/uncond batching)
  -> BaseModel.apply_model
  -> model_sampling.calculate_input
  -> model-specific extra conditions and diffusion model
  -> model_sampling.calculate_denoised
  -> CFG combination / post-CFG hooks
  -> sampler rule (Euler, Heun, DPM, ...)
  -> next full-canvas latent
```

Source anchors are `comfy/samplers.py:208-356,609-643,1188-1270` and
`comfy/model_base.py:203-252`.

State that must be synchronized for one accepted step:

- the complete current latent/sample and its exact sigma;
- positive/negative or other guider branches and their conditions;
- full-canvas geometry, crop/token mappings, positional coordinates, and
  padding;
- any source/reference conditioning and its ordering;
- the assembled full-canvas denoised/model prediction consumed by the sampler;
- local overlap weights and coverage (every updated position must have a
  defined prediction);
- caches associated with exactly the current accepted trajectory state;
- sampler history for multistep or multi-evaluation samplers.

Raw model-call count is not a step clock. Heun and other samplers can evaluate
the model more than once before accepting an update. Cache promotion and global
refresh policy must bind to sampler evaluation identity and accepted-step
lifecycle, or initially fail closed to a simple Euler/flow configuration.

ComfyUI already exposes useful intervention points:

- `DIFFUSION_MODEL`/`APPLY_MODEL` wrappers for replacing or coordinating a
  model evaluation;
- `CALC_COND_BATCH` or `sampler_calc_cond_batch_function` when branch/area
  batching itself must change;
- `PREDICT_NOISE`, `SAMPLER_SAMPLE`, and outer-sample wrappers for lifecycle;
- model-specific `transformer_options` patches and FLUX
  `patches_replace["dit"]` for internal execution.

A post-CFG mask is too late to save model compute. An attention output patch is
too narrow to skip image projections, MLPs, normalizations, residual updates,
or the final projection.

## 2. Where dense spatial cost begins

Let `G` be full-canvas image tokens, `L` active/local image tokens, `T` text
tokens, `R` reference tokens, hidden width `D`, MLP expansion `m`, and layers
`B`.

Common costs are:

- patch/input projection: approximately linear in image tokens, `O(G D Cin)`;
- normalization, modulation, residual arithmetic: `O(G D)`;
- Q/K/V and output projections: `O(G D^2)`;
- image MLP: `O(G m D^2)`, often a major cost even with efficient attention;
- dense self/joint attention: nominally `O(G^2 D)` for image-image interaction,
  plus interactions with text/reference tokens;
- final projection/unpatchify: `O(G D Cout)` plus linear rearrangement;
- CFG may multiply model work by branch count, although distilled/configured
  models can run at CFG 1;
- guider arithmetic and sampler updates are full-canvas but only linear in
  latent size and normally much cheaper than transformer execution.

Thus full spatial cost starts at image patchification/input projection, not at
attention. Dense global interaction first appears in image self-attention or a
joint stream. Cross-attention to fixed text is globally conditioned but scales
linearly in image queries times text length.

Operations that are intrinsically token-local once their inputs are known are
input/final projections, normalization, modulation, MLPs, and residual
arithmetic. They can run on selected tokens only if every layer propagates only
those selected hidden states. Attention is globally coupled, but it admits
local/selected queries over global K/V. That reduces the attention matrix from
`G x G` to `L x G`; it does not remove the need to obtain valid global K/V.

## 3. Qualified model-family paths

### 3.1 FLUX.2 Klein

Path:

```text
[B,C,H,W] latent
  -> BaseModel flow input scaling
  -> Flux.process_img: pad, 2-D patchify, row-major tokens
  -> explicit [stream/index, y, x] IDs and RoPE
  -> img_in + text projection + timestep/guidance/vector modulation
  -> double-stream blocks (joint text/image attention; separate MLPs)
  -> concatenate text+image
  -> single-stream blocks (joint QKV+MLP projection)
  -> image slice -> final AdaLN/projection -> unpacked model output
  -> flow denoised prediction -> guider -> sampler update
```

Source anchors: `comfy/ldm/flux/model.py:63-145,147-312,314-340` and
`comfy/ldm/flux/layers.py:163-258,261-...`.

FLUX-specific facts:

- generated and reference images are tokenized with explicit independent 2-D
  coordinates and ordered stream/index IDs;
- double-stream blocks update text and image streams; single-stream blocks
  concatenate them;
- `patches_replace["dit"]` can replace each double/single block;
- reference latents are image tokens in the model sequence, not the persistent
  generated-canvas state;
- previous SpotEdit qualification proved that reduced generated queries,
  full-context K/V reassembly, sparse hidden propagation, and sparse final
  projection are mechanically possible in native FLUX.2 with a specialized
  executor. It also showed substantial persistent K/V memory and lifecycle
  complexity. This is engineering evidence, not evidence that the same
  selection policy produces coherent T2I.

Global interactions occur in every joint attention layer. The per-token QKV,
MLP, residual, normalization, and final projection costs remain dense in the
ordinary implementation. A local crop can preserve absolute coordinates via
offset IDs, but its hidden states cannot see omitted generated tokens unless
they are supplied as K/V or another context representation.

Classification: FLUX.2 is the initial reference implementation family for all
three candidate architectures below.

### 3.2 Z-Image Turbo

Path:

```text
[B,C,H,W] latent
  -> flow input scaling
  -> pad and patchify -> image token projection
  -> explicit 3-axis IDs/RoPE
  -> text context refinement
  -> optional reference-latent and SigLIP token preparation/refinement
  -> image noise refinement
  -> concatenate context/reference/image streams
  -> repeated JointTransformerBlock:
       RMSNorm/modulation -> joint attention -> residual
       RMSNorm/modulation -> feed-forward -> residual
  -> final modulation/projection -> unpatchify
  -> denoised prediction -> guider -> sampler update
```

Source anchors: `comfy/ldm/lumina/model.py:240-411,419-430,654-815,825-874`.
The official model describes Z-Image as an S3-DiT whose text, visual semantic,
and image VAE tokens are concatenated into one stream.

Important differences from FLUX:

- one joint stream and GQA-capable Lumina-derived attention, not FLUX
  double/single blocks;
- context and noise refiners precede the main joint layers;
- Z-Image token padding to a configured multiple affects token counts;
- reference/edit variants can include latent, text, and SigLIP-derived visual
  tokens, but the available Turbo checkpoint is primarily T2I;
- current ComfyUI exposes generic diffusion-model and patch hooks, but not the
  FLUX block-replacement contract around each Z-Image layer.

Classification: **2, compatible but requiring a model-family adapter** for
reduced-global plus crop correction. Selected-query/K/V execution is
**4, currently unknown pending a block-level runtime prototype**, because the
joint stream, refiner stages, padding, and cache ownership need qualification.

### 3.3 Anima

Path:

```text
[B,C,H,W] latent (temporally lifted to [B,C,1,H,W])
  -> optional padding-mask channel
  -> convolutional/patch embedding to [B,T,H',W',D]
  -> 3-D RoPE using full token-grid shape
  -> timestep AdaLN embedding
  -> repeated blocks:
       image self-attention over flattened spatial tokens
       text cross-attention
       MLP
       (each with normalization, modulation, residual)
  -> final layer -> unpatchify
  -> flow denoised prediction -> guider -> sampler update
```

Text embeddings are first adapted through an Anima-specific LLM adapter and
cached/preprocessed during inference when token IDs are present. Source anchors:
`comfy/ldm/anima/model.py:143-214`,
`comfy/ldm/cosmos/predict2.py:430-590,594-932`, and
`comfy/model_base.py:1477-1500`.

Unlike FLUX/Z-Image, text is fixed cross-attention context rather than part of
the image self-attention stream. This makes selected image queries over cached
image K/V conceptually cleaner, while text K/V can be recomputed or cached
independently. However, current blocks accept dense grid-shaped residuals,
Anima keeps fp32 residuals under fp16 inference for quality, and there is no
per-block replacement path equivalent to FLUX.

Classification: **2, compatible but requiring a model-family adapter** for
crop/global-prediction fusion. Selected-token execution is **4, unknown pending
runtime validation** of sparse residual propagation, 3-D RoPE slicing, and
image-attention K/V caches. No evidence requires a substantially different
sampler algorithm.

## 4. Scene planning versus detail: what is known

The exact semantic responsibility of model layers or branches is not proven by
source inspection. The defensible evidence is weaker and should be stated as
such:

- diffusion trajectories exhibit a coarse-to-fine spectral tendency; DyPE
  explicitly exploits frequency evolution by changing positional encoding over
  denoising;
- ElasticDiffusion demonstrated on Stable Diffusion U-Nets that a lower-density
  conditional-minus-unconditional direction can contribute global structure,
  while local unconditional patch predictions contribute local signal;
- reference/source images can strongly constrain geometry in editing, but
  model reference tokens are conditioning, not the same-canvas trajectory;
- dense self/joint attention is a mechanism for long-range interaction, not a
  proof that every full-resolution token must execute every per-token MLP.

Working hypothesis, not established fact:

- object count, placement, relative scale, pose, perspective, large geometry,
  and broad lighting/color require low-frequency whole-canvas information and
  long-range interaction;
- texture, edges, materials, small anatomy, and high-frequency structure can
  be refined predominantly from local neighborhoods if globally registered
  conditioning and an authoritative shared trajectory are available.

The boundary is not clean. Pose and anatomy span scales; repeated patterns can
cause local regions to invent duplicate subjects; lighting and materials
interact. Therefore Blueprint must test semantic outcomes rather than encode a
fixed “early=composition, late=detail” rule.

## 5. Candidate global representations

### Reduced-resolution latent or prediction

Maintain a low-density whole-canvas latent trajectory or derive a
whole-canvas prediction from a resized current latent. Preserve aspect ratio and
map it explicitly to the high-resolution canvas.

It is cheaper only if the global model sees `g << G` tokens. For a 2-D
downsample factor `s`, token count is roughly `G/s^2` and dense attention work
roughly `1/s^4`; projections/MLPs fall roughly `1/s^2`. Local correction cost
must still be counted.

Risks: downsampling noisy latents changes their distribution; a separately
sampled low-resolution trajectory may diverge from the high-resolution one;
upsampled prediction lacks high frequencies; model resolution priors and RoPE
may change semantics.

### Sparse or strided whole-canvas tokens

Select spatially distributed tokens from the current full latent while retaining
their absolute coordinates. This keeps samples of the actual full-canvas state
without resizing every location.

It is cheaper only if unselected tokens do not pass through input projections,
every block MLP/residual, and final projection. Sparse queries attending to
sparse global K/V cost approximately `g^2`; local queries attending to global
context cost `L*g` or `L*(g+local context)`.

Risks: striding aliases and removes unsampled content; a set of coordinates is
not automatically a coherent scene representation; pretrained attention may
not tolerate missing token density.

### Cached hidden states or K/V

Cache global per-layer K/V (and, where needed, hidden state) from a global
refresh, then execute selected local queries through subsequent evaluations.

It is cheaper only between refreshes. Selected image Q/projection/MLP cost is
linear in `L`; attention is `L x context`. Full refresh remains dense at the
cache's token density. Persistent cache is `O(B * context * D)` and can increase
VRAM substantially.

Risks: cached values become stale when sigma, latent, conditioning, CFG branch,
or local updates change. Reusing K/V from an earlier denoising state is not
mathematically equivalent to current global context. Transactional lifecycle
and branch separation are mandatory.

### Hierarchical/multiscale representation

Maintain coarse global tokens plus one or more finer regional levels, with
explicit parent/child coordinate maps and possibly local-to-global summaries.

It is cheaper when the sum of token work over levels and refreshes is far below
dense `G`, and when fine levels cover only selected regions. It can represent
large and small structure better than one low-resolution grid.

Risks: no audited model natively consumes this hierarchy. Cross-level fusion is
an algorithm, not plumbing, and may require training or intrusive model changes.

### Early-only or periodic/adaptive refresh

This is a scheduling policy over another representation, not a representation
by itself. It is cheaper in proportion to skipped refreshes.

Risks: composition can continue evolving beyond an assumed cutoff; local
updates invalidate global context. Periodic refresh is safer than a hard early
cutoff, while adaptive refresh needs a meaningful drift signal.

## 6. T2I versus reference/edit regimes

### Pure T2I

T2I begins from noise. The global branch must create, not merely preserve,
object count, layout, relationships, and scene geometry. A low-density global
trajectory is plausible because coarse structure is spatially low-frequency,
but it is unproven that local high-resolution calls will follow it rather than
independently instantiate subjects.

The global state likely needs to evolve at every early/high-noise step and be
refreshed periodically later. “Early only” is a hypothesis to test, not a safe
default. Local calls need either:

- an authoritative resized global prediction fused before one sampler update;
- global hidden/K/V context at the same sigma; or
- a spatial conditioning signal derived from the global plan.

Text alone is not a spatial plan. Absolute coordinates alone do not convey what
exists elsewhere.

### Reference/edit

Reference tokens may provide global semantic, appearance, and geometric
evidence. In FLUX.2 and Z-Image editing paths they are separate tokens with
their own spatial IDs/order. They can be visible to local generated queries,
and their K/V may be immutable within a run if their timestep treatment is
fixed.

They do not replace the generated canvas state:

- the output can change layout relative to the source;
- reference coordinates may be a separate frame/stream;
- image conditioning can supply appearance without same-canvas correspondence;
- the noisy generated latent still must stay mutually consistent across local
  updates.

Known source geometry plausibly permits more aggressive local execution for
low-strength edits, restoration, and masked changes. Large semantic edits still
need a generated global plan.

### Inpainting and outpainting

- Inpainting has a fixed same-canvas anchor outside the mask. The unchanged
  region can supply strong global context, but objects crossing the mask and
  altered lighting still require long-range reasoning. Native mask compositing
  preserves pixels; it does not make a crop aware of them.
- Outpainting has source geometry only on part of the target. The new area is a
  T2I problem constrained by boundary/source context; a global planner is
  likely more important as the expansion grows.

Recommended policy separation over common mechanics:

| Regime | Initial global evidence | Likely refresh policy |
|---|---|---|
| T2I | text + low-density view of current noise/latent | dense early, periodic/adaptive later |
| Reference/edit | generated state + reference tokens/source geometry | strength/change-dependent; can be less frequent for constrained edits |
| Inpainting | generated state + fixed same-canvas exterior + mask | focus local work, refresh when changes cross mask/context boundary |
| Outpainting | partial source + generated exterior | T2I-like in new area; refresh grows with expansion/semantic novelty |

## 7. Local execution options

### Ordinary spatial crop

Run the unchanged model on overlapping latent crops. This genuinely reduces
per-call work and is portable. It preserves sigma and one trajectory if crop
predictions are assembled before one sampler update. It preserves absolute
coordinates only when the model adapter supplies full-canvas coordinate
offsets/scale.

It does not preserve global semantic context by itself. Overlap gives only
neighbor communication and can still duplicate subjects or break long
structures. Total compute is approximately number of crops times crop-model
cost and can equal/exceed dense execution.

### Selected image queries with global K/V/context

At each layer, execute selected image-token queries while attending to global
or cached K/V. Preserve the full-canvas IDs for selected queries. Run selected
projections, MLPs, residuals, and final projection, then scatter the prediction
into the full canvas.

This is true selective computation. It can preserve long-range context but only
to the extent that global K/V is current and semantically sufficient. It is
model-family-specific and cache-heavy.

### Selected tokens through all per-token paths

Reducing attention queries alone is insufficient. A faithful sparse executor
must keep the residual stream sparse through input projection (where possible),
normalization, modulation, Q/projection, MLP, residual, and final projection.
FLUX SpotEdit demonstrates this mechanically for generated tokens. Z-Image and
Anima remain unqualified.

### Local prediction fused with resized global prediction

Compute a full-canvas coarse prediction, resize it, compute local fine
predictions, and fuse them into exactly one full-resolution prediction before
the sampler update. This preserves one trajectory and gives every position a
defined global direction.

Fusion is the critical algorithm: overwrite, weighted residual correction,
frequency split, or consistency optimization have different semantics.
ElasticDiffusion's CFG-direction decomposition is U-Net/CFG-specific and cannot
be assumed valid for CFG-1 distilled modern DiTs. The portable first test is a
prediction-residual or frequency-band fusion, not a direct port of conditional
minus unconditional guidance.

## 8. Comparison with existing methods

| Method | Global state | Local state / information crossing | Forwards per step and scaling | Main failure/assumption |
|---|---|---|---|---|
| Ordinary tiled diffusion | one canvas latent, sometimes only implicit through merged tiles | overlapping crops; information crosses through overlap and later shared latent updates | roughly number of tiles; peak model activation bounded by tile, total token work covers canvas plus overlap | no true whole-scene reasoning; duplicates, broken geometry, seams; architecture-portable |
| Mixture of Diffusers | one canvas latent and averaged prediction | region-specific crop predictions blended by masks/overlap | one U-Net call per region/tile (batchable); peak tile/batch, total grows with regions and overlap | shared update improves seams but regions do not see distant content; original code is SD U-Net/CFG based |
| MultiDiffusion | one shared canvas latent; per-crop denoised targets constrained by an optimization/least-squares fusion | crops interact through the shared solution and overlap | one reference-model call per crop per step; peak tile/batch, total grows with coverage | consensus is not global scene planning; still repeats or drifts without controls; demonstrated on pretrained SD |
| ElasticDiffusion | full-canvas latent plus low-resolution global CFG direction/reference trajectory | local unconditional patch predictions + upsampled global direction + optional reduced-resolution guidance | local patch calls plus low-resolution conditional/unconditional calls and optional resampling; bounded per-call memory but nontrivial total compute | decomposition depends on CFG and SD U-Net behavior; transfer to CFG-1/joint DiTs is unproven |
| DyPE | ordinary dense full-canvas latent | no local state; dynamic timestep-dependent positional encoding | ordinary dense model forward; no additional sampling cost but no compute/VRAM reduction versus dense target canvas | solves positional extrapolation/frequency alignment, not Blueprint selective execution; useful complementary coordinate policy |
| SpotEdit (engineering reference) | full generated latent plus source latent and per-layer full-context K/V cache | selected generated queries attend to fresh/cached text/reference/generated K/V; prediction scattered full-size | full/reset calls plus sparse calls; sparse work scales with selected queries but cache/storage remains global | selection is source-edit-specific; cache validity and VRAM are substantial; not evidence for T2I planning |

Peak memory and total compute are separate. Tiling reduces activation peak but
can increase time. Cached K/V can reduce executed compute while increasing
persistent VRAM. DyPE can improve dense high-resolution quality without saving
either.

## 9. Candidate architectures to test (ranked)

### Rank 1 — coarse global prediction + local residual corrections

**Global representation:** a low-resolution, aspect-ratio-preserving latent or
model prediction derived from the current full-canvas latent at the same sigma.

**Local representation:** overlapping full-model crops with full-canvas
coordinate offsets. The local correction is the difference between local
prediction and the correspondingly resized/cropped global prediction, blended
over the high-resolution prediction canvas.

**One step:** derive low-resolution state; run one global prediction; resize it;
run selected or covering local crops; fuse weighted corrections; apply ordinary
guider/sampler once to the full latent.

**Persistent:** only the full-resolution latent is authoritative. Optionally
persist a low-resolution trajectory in a separately tested variant; do not
silently mix the two definitions.

**Scaling:** global `g` tokens plus local coverage `sum(L_i)`; dense attention
inside each local crop. Cheaper than dense only if `g << G` and local coverage
is selective or sufficiently sub-dense. Full covering crops may save peak VRAM
but not total compute.

**VRAM:** global/crop activation bounded by their respective sizes; full latent
and prediction buffers remain linear in canvas size; no per-layer global K/V.

**Suitability:** best first T2I falsification; also suitable for edit/inpaint,
where source geometry can choose fewer local corrections.

**ComfyUI intervention:** sampler/diffusion-model wrapper coordinating multiple
`apply_model` calls and one returned full prediction; model adapter for absolute
crop coordinates.

**Model assumptions:** model tolerates resized noisy inputs and crop sizes;
predictions are meaningfully composable across resolutions. No CFG-direction
assumption.

**Highest risk:** low-resolution prediction does not impose a sufficiently
authoritative scene plan, so local crops invent incompatible objects.

**Smallest falsifier:** one controlled FLUX.2 Klein Euler step/run on a wide
canvas comparing dense baseline, tiled-only, global-only-upsampled, and
global+local-residual variants at identical seed/sigmas. Log token work and
measure cross-tile object/count/long-line continuity. If global+local is no
better than tiled-only or destabilizes the shared trajectory, reject or revise
the fusion before building nodes.

### Rank 2 — coarse global tokens + selected local queries over global K/V

**Global representation:** per-layer K/V generated from a low-density
whole-canvas token grid with absolute coordinates, refreshed at a declared
schedule.

**Local representation:** selected high-density tokens propagated through all
per-token paths; local queries attend to global K/V plus current local K/V.

**One step:** refresh or reuse global context; gather local tokens; execute every
layer sparsely; scatter local predictions; fill unselected prediction from the
coarse global field; update the one full latent.

**Persistent:** full latent, refresh metadata, per-layer global K/V, branch and
sigma identity, selected-token mappings.

**Scaling:** refresh `O(B*g*D^2 + B*g^2*D)`; local
`O(B*L*D^2 + B*L*(g+L)*D)`. Benefit depends on `g,L << G` and refresh rate.

**VRAM:** potentially lower activations but persistent K/V proportional to
layers and global context; may exceed dense peak on smaller canvases.

**Suitability:** potentially strong for T2I if global tokens retain composition;
stronger for reference/edit where reference/source K/V is meaningful and
partly immutable.

**ComfyUI intervention:** lifecycle wrapper plus model-family sparse executor;
FLUX block replacement initially, separate Z-Image/Anima adapters later.

**Model assumptions:** pretrained blocks tolerate cross-density attention and
stale/periodic global features.

**Highest risk:** low-density/stale K/V is not a valid representation of the
current high-resolution global scene.

**Smallest falsifier:** on one real FLUX.2 block stack, compare full prediction
against (a) same-step low-density-global K/V + all high-density queries and (b)
periodically stale K/V, measuring tensor error and image semantics separately.
Do not start with selection policy; first test whether the context abstraction
itself works.

### Rank 3 — persistent two-level latent trajectory

**Global representation:** an independently sampled low-resolution latent with
the same prompt/reference conditioning and aspect ratio.

**Local representation:** high-resolution full-canvas latent updated through
local crop predictions, constrained each step toward the upsampled global
denoised estimate or low-frequency bands.

**One step:** advance global latent at its sigma; compute local high-resolution
predictions; impose low-frequency/global consistency; advance the high-resolution
latent once.

**Persistent:** both latent trajectories, exact sigma alignment, and their
coordinate/frequency mapping.

**Scaling:** one small global forward plus local crop coverage per step. Similar
to Rank 1 but with added state and potentially stronger planning continuity.

**VRAM:** modest extra latent/prediction storage; activations bounded by global
and local call sizes.

**Suitability:** plausible for T2I; reference/edit can initialize/constrain the
global trajectory from source geometry.

**ComfyUI intervention:** sampler-sample lifecycle coordination; a normal model
can perform both resolutions, but dual sigma/state handling is outside a simple
stateless model patch.

**Model assumptions:** low- and high-resolution diffusion trajectories can be
coupled without distribution mismatch or overconstraint.

**Highest risk:** the two trajectories encode incompatible scene hypotheses,
and forcing agreement damages detail or causes temporal instability.

**Smallest falsifier:** compare independently denoised low/high trajectories at
matched accepted steps; quantify low-pass prediction agreement and visually
inspect whether coupling improves or worsens wide-scene composition. Reject if
alignment is not stable enough to define a simple correction.

The hierarchical/multilevel extension is deliberately not a fourth candidate.
It adds an unvalidated cross-level algorithm before the two-level hypothesis is
known to work.

## 10. Generic API boundary implied by the audit

A future generic Blueprint API should describe data and lifecycle, not FLUX
block names:

- full-canvas geometry and authoritative latent;
- accepted evaluation identity: sigma, branch, sampler evaluation, and step;
- global-context policy selected by generation regime;
- model-family geometry adapter: patchification, absolute IDs/RoPE, padding,
  reference ordering;
- executor capability: dense crop, full prediction, selected-query execution,
  cacheable context;
- explicit full-prediction assembler/fusion contract;
- cache preflight, promotion, invalidation, and cleanup;
- instrumentation for global/local token counts, forward count, attention
  dimensions, cache bytes, peak VRAM, and time.

The API must fail closed when an executor cannot preserve coordinates,
conditioning branches, sampler lifecycle, or complete prediction coverage.
“Supports attention patching” is not a sufficient capability declaration.

## 11. Required final verdict

### 1. Generic Blueprint execution requirements

One authoritative full-canvas latent trajectory; globally registered spatial
coordinates; same-sigma/conditioning local and global evaluations; genuinely
reduced expensive execution; an explicit global-to-local information path; one
complete fused prediction before each sampler update; sampler-aware cache and
refresh lifecycle; and separate measurement of semantic quality, executed
token/FLOP work, wall time, peak VRAM, and persistent cache memory.

### 2. Currently known FLUX-specific facts

FLUX.2 uses double then single joint text/image streams, explicit 3-axis IDs for
generated/reference token geometry, ordered reference tokens, global modulation
variants, and a native per-block `patches_replace["dit"]` boundary. Prior sparse
execution and K/V reassembly evidence applies to this block topology. These are
not generic Blueprint contracts.

### 3. Can T2I and reference/edit share an execution architecture?

Yes, plausibly: they can share the persistent full-canvas sampler coordinator,
geometry/mapping contracts, global/local executor interface, prediction fusion,
and lifecycle instrumentation. They cannot yet be assumed to share one global
representation or refresh/selection policy.

### 4. Where global-context policy likely differs

T2I needs an evolving planner that creates scene structure from noise and text,
probably strongest early but periodically validated later. Reference/edit can
use source/reference geometry and immutable conditioning more aggressively;
inpainting can anchor to the known exterior; outpainting becomes increasingly
T2I-like as new unconstrained area grows. Edit strength, mask topology, and
semantic novelty should influence refresh and local coverage.

### 5. Ranked candidates

1. Reduced-resolution global prediction plus local residual corrections.
2. Low-density global tokens/K/V plus selected high-density local queries.
3. A persistent coupled low-resolution and high-resolution trajectory.

### 6. Single next experiment with highest information value

Run the Rank-1 four-way controlled FLUX.2 Klein comparison: dense baseline,
tiled-only, upsampled low-resolution global-only, and global-plus-local residual
fusion, using the identical initial full latent, prompt, seed, Euler sigma
schedule, CFG-1 branch, and wide composition-stress prompt. Instrument per-step
global/local token work and retain intermediate denoised predictions. This one
experiment tests the central semantic premise—whether a cheap low-density
prediction can govern local high-detail work—without first paying the cost of a
model-specific sparse executor.
