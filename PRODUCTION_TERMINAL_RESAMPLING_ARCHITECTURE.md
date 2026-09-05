# Production architecture for terminal-resampling Blueprint Diffusion

Date: 2026-09-05

## 1. Executive decision

The first production integration should be a dedicated staged sampling node,
not a public `SAMPLER` for `SamplerCustomAdvanced` and not an extension of the
Candidate-3 `(G,H)` coordinator.

The node owns one truthful operation:

```text
bounded 32x64 Blueprint Euler trajectory (four intervals)
    -> terminal Blueprint x0
    -> map to 128x256 destination
    -> 55 independent native 64x64 late-noise refinements at sigma 0.25
    -> 2x2 restriction and normalized overlap assembly
    -> final destination LATENT
```

Internally, the node should still call the supplied ComfyUI `GUIDER` through
`guider.sample()` with a private procedure sampler. This preserves normal
ComfyUI model loading, conditioning preparation, wrappers, cleanup, and model
management. The procedure sampler is an implementation detail and is never
exposed as a user-selectable `SAMPLER`.

The initial production contract is the exact Phase-25 contract only. It does
not replace the old Candidate-3 node immediately. The old node may remain
available under its existing name during one compatibility release, explicitly
marked legacy/deprecated; the new path shares no accepted-state lifecycle with
it.

## 2. ComfyUI integration boundary

### Alternatives

| Boundary | Assessment | Decision |
|---|---|---|
| Public custom `SAMPLER` + `SamplerCustomAdvanced` | Can technically own arbitrary code, but its input/output latent and callback contract imply one trajectory at one geometry. Here Stage 1 samples `32x64`, Stage 2 evaluates fresh `64x64` states at a separate sigma, and output is `128x256`. It also makes an arbitrary sampler selector look meaningful when Euler and the terminal procedure are fixed. | Reject for the new path. |
| Dedicated all-in-one sampling node | Truthfully owns the staged algorithm, fixed policy, seed derivation, destination descriptor, previews, and final LATENT. It can invoke `guider.sample()` internally to retain ComfyUI preparation and cleanup. | Select. |
| Model patch + ordinary sampler | A model patch cannot truthfully introduce a second fixed-sigma, 55-call terminal stage or change output geometry without hiding lifecycle and state outside the sampler. | Reject. |
| Outer-sample/sampler wrapper | Wrappers are appropriate for intercepting an existing lifecycle, not replacing its state geometry and appending a second sampling procedure. Wrapper ordering would also broaden compatibility risk. | Reject. |

### Proposed node contract

Conceptual node name and display name:

```text
BlueprintTerminalResampling
Blueprint Terminal Resampling
```

Inputs:

```text
guider: GUIDER                 # BasicGuider only in first slice
sigmas: SIGMAS                 # exact qualified 5-value schedule
noise_seed: INT               # ordinary control-after-generate seed
destination: LATENT           # geometry/metadata descriptor only
```

Outputs:

```text
output: LATENT                # assembled terminal destination x0
denoised_output: LATENT       # same terminal x0 in the first slice
```

`destination["samples"]` must be a zero, non-nested, batch-one
`[1,128,128,256]` tensor with no mask. It declares output geometry and carries
normal latent metadata; it is never promoted to an accepted H trajectory.
Using a destination descriptor avoids hardcoding width/height across every
component while still failing closed to the only qualified geometry.

The node constructs a private `_TerminalResamplingProcedure` implementing the
internal `comfy.samplers.Sampler.sample` call shape, then invokes:

```text
guider.sample(procedure_noise, destination_samples, procedure,
              sigmas, denoise_mask=None, callback=..., seed=noise_seed)
```

The private procedure ignores the destination tensor as trajectory state after
validating it. This entry preserves `CFGGuider.outer_sample/inner_sample`,
`prepare_sampling`, `process_conds`, copied model options, wrappers, pre-run,
cleanup, device context, and model unloading semantics. It does not manually
reconstruct `guider.conds`.

## 3. Exact lifecycle

### Stage 0 — validate and initialize

Before model work:

1. Validate the complete first-slice profile and schedule.
2. Build the exact immutable geometry policy and 55 ordered regions.
3. Derive deterministic initialization/noise descriptors from `noise_seed`.
4. Construct the initial Blueprint only. There is no accepted destination H.

To reproduce Phase 25 exactly, initialization is:

```text
E_H ~ N(0,I), shape [1,128,128,256], CPU generator seed = user_seed
B_coarse = avg_pool2d(E_H, kernel=4, stride=4)
E_B ~ N(0,I), shape [1,128,32,64], CPU generator seed = user_seed+20_000_003
B_0 = B_coarse + sqrt(15/16) * E_B
```

`E_H` is an initialization construction buffer, not H state. It should be
generated and reduced on CPU, transferred only as `B_0`, and released before
Stage 1. A later implementation may prove an exactly RNG-equivalent streamed
reduction, but the first slice should prefer the frozen regression contract.

### Stage 1 — bounded Blueprint trajectory

The first slice accepts exactly the Phase-25 sigma vector:

```text
[1.0000000000, 0.9991771579, 0.9975355268, 0.9926428199, 0.0000000000]
```

The separate terminal local call at sigma `0.25` is not inserted into this
schedule and is not a fifth Blueprint Euler interval.

For each of exactly four intervals:

```text
x0_B = guider(B_i, sigma_i, native Blueprint coordinates)
B_next = B_i + (sigma_next-sigma_i) * (B_i-x0_B) / sigma_i
```

The accepted state is atomically replaced only after the prediction and finite,
shape, schedule, and immutability checks pass. The terminal prediction
`x0_B_terminal` is retained separately; Stage 2 consumes the denoised estimate,
not the accepted zero-sigma state by implication.

### Stage 2 — terminal local resampling

Map `x0_B_terminal` from `32x64` to `128x256` using the exact Phase-25
bilinear mapping (`align_corners=False`). For each ordered destination region:

```text
b_r = mapped_B[:, :, y:y+32, x:x+32]
epsilon_r = deterministic CPU region noise [1,128,64,64]
W_r = 0.75 * nearest2(b_r) + 0.25 * epsilon_r
x0_W_r = guider(W_r, sigma=0.25, native 64x64 local coordinates)
p_r = avg_pool2d(x0_W_r, kernel=2, stride=2)
accumulator[r] += weight_r * p_r
coverage[r] += weight_r
```

After all 55 regions pass:

```text
H_final = accumulator / coverage
```

Validate complete positive coverage, finiteness, cardinality, shape, and input
immutability, then publish one final destination LATENT. There is no destination
Euler update, D/U projection, terminal policy, persistent W, or post-anchor.

## 4. State ownership

```text
Dedicated node / run transaction
|
+-- immutable RunConfiguration
|     seed, schedule, qualified geometry, model/conditioning profile
|
+-- BlueprintRunState (accepted Stage-1 state only)
|     blueprint_latent B_i
|     sigma_i
|     ordinal
|     accepted_evaluation_id
|
+-- transient terminal operation
      terminal x0_B
      mapped destination Blueprint
      one region descriptor
      one epsilon_r and W_r
      one x0_W_r and restricted p_r
      destination accumulator + coverage
      compact detached telemetry
```

Suggested value type:

```python
@dataclass(frozen=True)
class BlueprintRunState:
    blueprint: Tensor
    sigma: float
    ordinal: int
    accepted_evaluation_id: str
```

It contains no `h`. Terminal objects never become accepted state. The model
adapter receives tensors and returns predictions; it does not own run state,
assembly, seed policy, or acceptance.

On any failure, no final LATENT is returned. A completed local region may have
contributed to a run-local accumulator, but that accumulator is not observable
outside the failing call.

## 5. Components and ownership

Use a small number of explicit value objects and pure functions rather than a
new execution framework.

### `TerminalResamplingGeometry`

Frozen value object containing:

```text
blueprint_hw=(32,64)
destination_hw=(128,256)
destination_region_hw=(32,32)
destination_stride=(24,24)
working_hw=(64,64)
blueprint_to_destination_scale=(4,4)
destination_to_working_scale=(2,2)
```

It validates relationships and stores no tensors. The representation is
general enough to make each coordinate system explicit, but
`QualifiedTerminalResamplingPolicy.select()` accepts only this exact value.

### Region planning and overlap assembly

Reuse the `Region` value and the existing deterministic end-aligned 32/24
planning rule after separating it from Candidate-3 geometry exceptions. Reuse
the existing overlap weights and ordering.

For bounded residency, give `OverlapAssembler` a streaming accumulator API (or
one small accumulator function) that performs the exact existing ordered
weighted additions. Do not require a `RegionPredictionSet` of 55 tensors.

### Mapping and working-canvas construction

Keep these as explicit pure functions in a terminal-resampling geometry module:

```text
map_blueprint_to_destination(x0_B, geometry)       # bilinear, align_corners=False
blueprint_region(mapped_B, region)                 # immutable 32x32 view
build_working_canvas(region_anchor, epsilon)        # nearest2 + CONST mixture
restrict_working_prediction(x0_W)                  # exact 2x2 arithmetic mean
```

`build_working_canvas` validates the algebraic same-sigma contract. These
operations are model-neutral tensor geometry, not FLUX adapter behavior.

### Noise policy

One `TerminalRegionNoise` policy owns stable region seed derivation and CPU
generation. For exact Phase-25 regression:

```text
region_seed = (user_seed + 22_000_003 + 1009 * region.index) mod 2^64
epsilon_r = torch.randn([1,128,64,64],
                        generator=CPUGenerator(region_seed))
```

Region identity is the planner's deterministic row-major index backed by the
full geometry tuple `(y,x,height,width)`. Before generation, validate that the
index/rectangle mapping equals the qualified plan. No Python `hash()`, global
CPU RNG, CUDA RNG, or completion order participates. Resume or retry therefore
reconstructs identical noise.

Preserve the literal research formula in the first slice. A future versioned
hash-based derivation may be cleaner, but changing it before the exact
production regression would prevent bit-exact comparison.

### `ModelFamilyAdapter`

The new-path adapter needs only:

```text
validate_run(...)
predict_blueprint(guider, B, sigma, model_options, seed)
predict_working(guider, W, sigma=0.25, model_options, seed)
```

Both prediction methods are ordinary guider calls using copied model options
and native local coordinates. They do not accept destination H, crop offsets,
global K/V, a block executor, assembly objects, or trajectory state.

The coordinator/procedure owns the sequence; the adapter owns strict native
FLUX.2 Klein qualification and the minimal model-family call contract.

## 6. Geometry and coordinate contract

The first slice supports exactly:

| Space | Latent grid | Model coordinates | Meaning |
|---|---:|---|---|
| Blueprint | `32x64` | Native unit-spaced `0..31, 0..63` | Complete scene in its own bounded model canvas |
| Destination | `128x256` | No model call | Output/assembly coordinate system |
| Region | `32x32` within destination | No model call | Destination coverage/assembly rectangle |
| Working | `64x64` | Native unit-spaced `0..63, 0..63` | Independent local refinement canvas |

The Blueprint is not presented to FLUX as sparse positions over the destination.
The full-canvas relationship is external:

```text
Blueprint native coordinate
<-> normalized [0,1] canvas coordinate
<-> destination coordinate
```

Similarly, a working canvas is not assigned destination absolute coordinates.
Every local prediction sees ordinary native `64x64` coordinates, matching
Phases 20–25. Changing this would be a new algorithm.

Mapping is `32x64 -> 128x256` bilinear with `align_corners=False`, followed by
an exact destination crop and nearest-neighbor `2x` lift. Prediction restriction
is nonoverlapping arithmetic `2x2` mean. No DCT or endpoint-scaled RoPE is used.

The abstraction can describe other sizes, but the policy rejects them. There
is no “largest fitting,” aspect-ratio, or arbitrary-scale inference.

## 7. Terminal sigma and flow/noise contract

ComfyUI's live `CONST` model sampling defines:

```text
noise_scaling(sigma, epsilon, x0)
    = sigma * noise_scale * epsilon + (1-sigma) * x0
calculate_input(sigma, state) = state
calculate_denoised(sigma, model_output, model_input)
    = model_input - sigma * model_output
```

For the qualified model `noise_scale == 1`. Therefore the Phase-25 expression

```text
W_0.25 = 0.75 * anchor + 0.25 * epsilon
```

is exactly `model_sampling.noise_scaling(0.25, epsilon, anchor, False)`, not an
approximate blend. Production should call `noise_scaling` and assert numerical
equality with the frozen expression in qualification. It stores `W_0.25` as
sampler/model state and passes it directly to the guider; `CONST.calculate_input`
does not apply another scale.

The guider returns the ordinary ComfyUI denoised estimate `x0_W`. No Euler step
from `0.25` to zero is separately necessary because for CONST Euler that
terminal proposal equals the denoised estimate. The first slice rejects a
model sampling object with a different parameterization or non-unit
`noise_scale`.

`0.25` is an internal qualified constant, not a node input.

## 8. GPU residency and streaming design

Correct first-slice residency is:

```text
Stage 1:
    one B state + one x0_B prediction + model working memory

Stage 2:
    terminal x0_B
    mapped destination Blueprint (or CPU-backed source with one GPU crop)
    destination accumulator + coverage
    one epsilon/W/x0_W/restricted prediction at a time
    model working memory
```

Do not retain 55 W states or 55 predictions. Process regions in the exact
planner order and accumulate immediately. Release region tensors after their
ordered weighted addition. Compact telemetry may retain only CPU scalars,
hashes, shapes, and timing.

For the first slice, keep mapped Blueprint, accumulator, and coverage on GPU.
They are modest relative to model execution and this preserves the exact
existing assembly arithmetic without 55 host transfers. Return the final
latent through ComfyUI's normal `intermediate_device()` handling at the node
boundary. CPU-backed accumulation is mechanically possible but unqualified;
make it a later measured memory policy, not a hidden configuration.

Do not call `torch.cuda.empty_cache()` per region. Cancellation cleanup drops
run-owned references; allocator reservation is not treated as live tensor
ownership.

### Local batching classification

All W calls use equal shape, sigma, conditioning, and native local coordinate
grids, so they are mechanically batchable without distinct absolute crop RoPE.
However, batch-dependent native/quantized execution differences, memory growth,
and actual speed remain unqualified. First production executes batch one and
55 ordered calls. Batching is **mechanically plausible but unqualified** and is
deferred.

## 9. Conditioning and adapter boundary

The dedicated node accepts a prepared ComfyUI `GUIDER`, initially only the
native `BasicGuider` one-positive-branch configuration. It enters
`guider.sample()` once, allowing ComfyUI to:

- prepare/load the model;
- process conditioning against the declared destination run;
- clone model options and attach sample sigmas;
- run model patcher pre-run/cleanup and wrapper chains;
- place tensors in the configured device context.

Inside the private sampler, every Blueprint and W prediction calls that same
prepared guider with copied per-call model options. The adapter must reject
area/mask/reference/control/hook/spatial conditions and unsupported wrappers
before the first prediction. It must not read or rebuild raw conditioning from
`guider.conds`.

Validate the native ComfyUI FLUX.2 Klein 4B profile explicitly (native
`model_base.Flux2`, native `comfy.ldm.flux.model.Flux`, CONST, 128 channels,
patch size one, 3072 width, 24x128 heads, 5 double/20 single blocks, 7680
context width, 512 prepared text tokens, `[32,32,32,32]` RoPE and theta 2000,
no guidance embedding/references/control/compiled replacement). Reuse the
strict checks from `Flux2Adapter`; do not invoke `Flux2BlockExecutor`.

The old specialized executor is completely absent from this path. It solved a
different destination-scaled K/V architecture and would add cost and semantics
that Phases 22–25 explicitly do not use.

## 10. Preview, cancellation, and restart behavior

### Preview

Expose five progress events:

1. four Stage-1 Blueprint `x0_B` previews at native Blueprint resolution;
2. one final assembled destination preview after all 55 regions validate.

Do not preview individual regions. The standard latent previewer accepts the
current callback tensor and can decode the smaller Blueprint tensors honestly;
the final callback replaces `x0_output["x0"]` with the destination tensor used
for the node's denoised output. If live qualification shows that varying latent
shapes break a frontend consumer, keep four progress updates without images
and retain only the final destination latent preview; do not upsample previews
and imply destination detail that does not exist.

### Cancellation and atomicity

Check ComfyUI interruption:

- before initialization and each Blueprint call;
- between Blueprint intervals;
- before terminal mapping;
- before and after every local model call;
- before final normalization/publication.

Stage-1 acceptance is run-local. Stage 2 may mutate only its private
accumulator. Cancellation, OOM, nonfinite data, shape drift, or model failure
raises and returns no LATENT, no final callback, and no partial success. A
`finally` block drops the current region tensors, accumulator, mapped Blueprint,
and telemetry scratch. No state/cache is stored on the node class, guider,
model patcher, or adapter.

The next queue execution reconstructs all noise from seed and geometry and is
deterministic. There is no transaction requirement for a nonexistent H
trajectory and no rollback journal.

### Resumability

Do not write checkpoints or experiment artifacts in production. ComfyUI node
execution is restart-from-seed after cancellation. Optional disk/GPU resume is
deferred until there is a ComfyUI-native cache ownership contract and measured
need; it must never alter region noise or accepted Blueprint identity.

## 11. First-slice compatibility matrix

| Capability | Classification | First-slice behavior |
|---|---|---|
| Native ComfyUI FLUX.2 Klein 4B exact profile | **SUPPORTED** | Strict validation |
| BasicGuider, CFG 1, one positive branch | **SUPPORTED** | Required |
| Batch 1 T2I, zero destination descriptor | **SUPPORTED** | Required |
| Destination `128x256`, Blueprint `32x64`, W `64x64`, 55 regions | **SUPPORTED** | Exact policy only |
| Four-step qualified CONST-flow Euler schedule | **SUPPORTED** | Exact five sigmas only |
| Terminal sigma `0.25` | **SUPPORTED** | Internal fixed constant |
| FLUX.2 Klein 9B | **MECHANICALLY PLAUSIBLE BUT UNQUALIFIED** | Reject |
| Other step counts or sigma sequences | **MECHANICALLY PLAUSIBLE BUT UNQUALIFIED** | Reject |
| Other destination geometry/aspect/scale, including lower-than-native or greater than 4x | **REQUIRES NEW ADAPTER/RESEARCH** | Reject |
| Local batch greater than one | **MECHANICALLY PLAUSIBLE BUT UNQUALIFIED** | Execute batch one |
| CFG greater than one or negative branch | **REQUIRES NEW ADAPTER/RESEARCH** | Reject |
| Reference/edit conditioning, ControlNet, spatial conditions | **REQUIRES NEW ADAPTER/RESEARCH** | Reject |
| Masks, inpainting, outpainting, nonempty/partial-denoise destination input | **REJECT FIRST SLICE** | Reject before model work |
| Alternate sampler/scheduler/flow parameterization | **REJECT FIRST SLICE** | Fixed procedure only |
| FLUX.1, Z-Image, Anima | **REQUIRES NEW ADAPTER/RESEARCH** | Reject |
| Nunchaku or compiled/replaced FLUX backend | **REQUIRES NEW ADAPTER/RESEARCH** | Reject |

“Mechanically plausible” is not support and does not enable fallback.

## 12. Migration from Candidate-3 production

| Existing component | New-path disposition | Reason |
|---|---|---|
| `BlueprintCandidate3EulerSampler` node | Retain temporarily as legacy; deprecate after workflow migration | Its public SAMPLER and persistent G/H semantics do not describe the selected architecture. |
| `BlueprintEulerSampler` | Not used; replace in new path with private staged procedure | New lifecycle changes geometry and performs a fixed-sigma terminal stage. |
| `BlueprintCoordinator` | Do not reuse implementation; retain only the concept of a generic orchestration owner | It assumes accepted G/H, D/U coupling, and per-step regions. |
| `BlueprintState(g,h,...)` | Replace for new path with `BlueprintRunState(blueprint,...)` | There is no accepted H trajectory. |
| `BlockDCTGeometry` | Obsolete in new path; leave for legacy node only | Selected Blueprint initialization/mapping is fixed pooling/noise plus image-space latent resizing, not 4-to-3 DCT. |
| `HardNonterminalTerminalRelease` | Obsolete in new path | No coupled G/H acceptance or terminal H release exists. |
| `Region` | Reuse | Correct explicit rectangle value. |
| `FixedCropPlanner` | Extract/reuse exact 32/24 end-aligned rule; do not inherit geometry-dependent Candidate-3 windows | Phase 25 requires 55 32x32 regions. |
| `OverlapAssembler` | Reuse weights/order; add narrow streaming accumulation capability | Assembly semantics qualify; retaining all predictions does not. |
| `ModelFamilyAdapter` protocol | Simplify for the new procedure or introduce a separate narrow protocol beside legacy | Old global/region API encodes G/H and absolute-coordinate crops. Avoid pretending both state models are one abstraction. |
| `Flux2Adapter` | Reuse strict native-profile/conditioning validation; add ordinary Blueprint/W calls in a new-path adapter surface | Both new calls are normal guider predictions with native coordinates. |
| `Flux2BlockExecutor` | Explicitly not used | No K/V injection or explicit block execution is part of the selected mechanism. |
| Candidate-3 telemetry | Do not reuse schema; define staged Blueprint/terminal telemetry | Old synchronized-G/H and projection fields would be false. |

Do not delete legacy files in the first implementation slice. Register the new
node independently, freeze old behavior, and document migration. A later
explicit removal task may delete legacy architecture after saved workflows are
handled.

## 13. Recommended implementation slices

1. **Pure geometry/noise slice.** Add `TerminalResamplingGeometry`, exact
   planner policy, Blueprint initialization, mapping, W construction,
   restriction, streaming accumulator, and deterministic seed functions with
   no Comfy imports beyond tensors.
2. **Blueprint-only lifecycle slice.** Add immutable `BlueprintRunState` and
   four-interval Euler coordinator with atomic replacement and compact
   telemetry.
3. **Narrow native adapter slice.** Reuse/extract strict Flux2 validation and
   ordinary native-coordinate `predict_blueprint`/`predict_working`; explicitly
   exclude specialized executor imports.
4. **Terminal streaming slice.** Add ordered region execution, same-sigma
   noise construction, immediate restrict/accumulate, cancellation, and final
   validation.
5. **Dedicated node slice.** Add the all-in-one node and private internal
   procedure sampler entering through `guider.sample()`, normal LATENT output,
   progress/previews, and fail-closed messages.
6. **Live qualification slice.** Run exact Phase-25 regression and workflow
   tests. Only after all gates pass should the old node be visibly marked
   legacy/deprecated; do not remove it automatically.

Each slice must leave existing Candidate-3 tests bit-exact until a separately
authorized removal.

## 14. Future implementation acceptance plan

### Mandatory unit and integration gates

1. Exact geometry relationships, row-major 55-region plan, end alignment,
   mapping mode, nearest2 lift, average restriction, weights, and complete
   normalized coverage.
2. Seed derivation tests for Blueprint initialization and every region,
   including independence from execution order and repeat/restart identity.
3. `CONST.noise_scaling(0.25, epsilon, anchor)` equality with
   `0.75*anchor+0.25*epsilon`; reject other parameterizations/noise scales.
4. BlueprintRunState immutability, exactly four atomic Euler acceptances,
   exact schedule, and no H field/state.
5. Streaming assembly equality with the existing list assembler in exact
   region order, including dtype/device behavior.
6. Cancellation before Stage 1, between intervals, at early/late local regions,
   and before publication; no final callback/LATENT and deterministic restart.
7. Injected local error/OOM/nonfinite/shape mismatch leaves no persistent node,
   adapter, accumulator, or RNG state.
8. Strict failures for every unsupported model, backend, guider, conditioning,
   geometry, mask, latent, schedule, batch, and wrapper case in the matrix.

### Mandatory real-model regression

1. One exact Phase-25 case reproduces every available boundary: initial B,
   all four Blueprint x0/accepted states, terminal mapped Blueprint, all 55
   W/noise hashes, all local x0/restricted predictions, coverage, assembled H,
   and decoded output. Prefer bit exact; locate the first boundary before
   accepting any narrow tolerance.
2. Live normal flow through BasicGuider, the dedicated node, VAE Decode, and
   Save Image; repeated queue executions have identical latent and decoded
   hashes.
3. Exactly four Blueprint model calls plus exactly 55 batch-one local model
   calls. No destination-size model forward occurs.
4. Telemetry and call interception prove zero execution of BlockDCT D/U,
   Candidate-3 H local trajectory, hard coupling/terminal policy,
   `Flux2BlockExecutor`, global K/V context, periodic refresh, or post-anchor.
5. Peak allocated/reserved memory and per-region barrier telemetry prove W and
   predictions do not accumulate with region ordinal.
6. Preview qualification proves four truthful Blueprint progress events and
   one final destination preview, or selects the documented progress-only
   Stage-1 fallback if mixed preview shapes are incompatible.

### Semantic regression set

After exact bridge parity, rerun the fixed car/tree/house and astronaut cases.
All three selected outputs must remain S3 under the Phase-25 rubric. These are
mandatory semantic acceptance gates, not optional characterization.

## 15. Explicitly deferred features

- arbitrary destination/Blueprint/working geometry or aspect ratio;
- any scale other than destination-to-Blueprint 4x per axis and W-to-region 2x;
- step-count or schedule generalization;
- configurable terminal sigma or resampling/noise strength;
- periodic or interleaved local refinement and persistent H/W/residual state;
- DCT coupling, prediction anchors, null-space/detail release, or filters;
- external global K/V or the specialized block executor;
- local batching, concurrency, cache, sparse execution, and CPU accumulation;
- masks, edits, references, ControlNet, CFG above one, negative branches;
- FLUX.2 9B, FLUX.1, Nunchaku, Z-Image, Anima, compiled/replaced backends;
- disk checkpoint/resume and automatic workflow migration;
- removal of the legacy Candidate-3 node.

## Readiness verdict

The lifecycle, ComfyUI boundary, state ownership, exact flow/noise algebra,
geometry, memory lifetime, strict support envelope, migration path, and future
acceptance gates are sufficiently specified for a narrow implementation task.
No further architectural discriminator is required before implementing the
first exact production slice.

**TERMINAL-RESAMPLING PRODUCTION ARCHITECTURE READY**
