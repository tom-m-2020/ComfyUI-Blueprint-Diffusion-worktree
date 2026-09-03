# Candidate-3 production specialized-executor architecture

Date: 2026-09-03

## Scope and selected direction

Phase 8i qualifies an exact FLUX.2 Klein block-major execution mechanism for
the failing 2048×4096 terminal evaluation. This design introduces that
mechanism without moving accepted-state or sampler policy into model-specific
code.

The selected direction is:

```text
ordinary Candidate-3 execution for every nonterminal interval
+
terminal-only full-current-G context at H=128×256
+
an adapter-private native FLUX.2 block executor
```

The first production slice must activate the specialized path only for the
exact qualified target/global geometry:

```text
H = 128×256
G = 96×192
crop = 32×32, stride 24, 55 ordered regions
```

Every other currently supported geometry retains its existing ordinary local
execution. This is not a general large-canvas threshold or a promise that the
transposed geometry is qualified. If the exact geometry selects specialized
execution but its stricter model/conditioning checks fail, the run fails
closed rather than silently returning to the semantically failed terminal
local-only path.

No new node or user-facing algorithm control is needed. The existing
`Blueprint Candidate-3 Euler Sampler` remains the owner and continues to be
used through `SamplerCustomAdvanced`.

## Interface alternatives

### A. Put coordinated block execution directly in `Flux2Adapter`

This keeps FLUX behavior out of the coordinator, but would turn the adapter
into a large class containing validation, ComfyUI input acquisition, native
block arithmetic, memory ownership, and telemetry. It is workable but obscures
the boundary between model-family policy and the explicit executor proven by
Phase 8i.

**Reject as the implementation shape.** Retain a small adapter entry point,
but place block semantics in a separate private component.

### B. Add an adapter-private `Flux2BlockExecutor`

`Flux2Adapter` chooses ordinary versus qualified coordinated local execution,
validates the exact native model/profile, and translates generic canvas/region
geometry into FLUX coordinates. `Flux2BlockExecutor` receives already scoped
FLUX call descriptors and returns ordered crop predictions plus compact
telemetry. It never sees or returns `BlueprintState`.

**Select.** This is the narrowest design that gives block-local memory an
explicit owner without putting FLUX block structure in generic sampling code.

### C. Add a generic execution-session abstraction

A generic session with prepare/advance/capture/finalize methods would largely
mirror FLUX double/single stream structure or become so abstract that its
lifetime guarantees are hard to inspect. No second model family has qualified
the same execution need.

**Reject for now.** Introduce no session framework solely for hypothetical
Z-Image or Anima implementations. Their future adapters may use completely
different private executors behind the same model-neutral bulk prediction
boundary.

### D. Give an adapter the whole Blueprint interval

An `evaluate_interval()` API would absorb Euler proposals, D/U coupling,
terminal policy, assembly, and state commit into the model family. It would
duplicate sampler logic and break the existing model-independent ownership
contract.

**Reject.** Model execution must not become the accepted-step owner.

## Selected production interface

Add one model-neutral bulk-local operation to the adapter boundary while
leaving `predict_global()` and `predict_region()` intact:

```python
@dataclass(frozen=True)
class RegionPredictionSet:
    predictions: tuple[Tensor, ...]
    telemetry: dict[str, object]

class ModelFamilyAdapter(Protocol):
    def predict_regions(
        self,
        *,
        guider,
        accepted_g: Tensor,
        accepted_h: Tensor,
        sigma: Tensor,
        canvas: tuple[int, int],
        regions: tuple[Region, ...],
        terminal: bool,
        model_options: dict[str, object],
        seed: int,
    ) -> RegionPredictionSet: ...
```

This interface is deliberately not a block/session API. It says only that a
model-family adapter may coordinate the evaluation of a fixed ordered region
set while reading the accepted global state. A future adapter can implement it
as the existing sequential `predict_region()` loop. It is not required to
have double blocks, K/V, RoPE, or joint text/image attention.

For `Flux2Adapter`:

- nonterminal calls and all non-specialized geometries execute the existing
  sequential `predict_region()` loop exactly;
- the exact qualified terminal geometry invokes a private
  `Flux2BlockExecutor`;
- the returned tuple must have one prediction per input region, in identical
  order and shape;
- no assembly, Euler arithmetic, projection, or state acceptance occurs in
  the adapter or executor.

`BlueprintCoordinator.evaluate()` continues to plan regions, snapshot accepted
G/H, invoke the adapter, verify input immutability, call `OverlapAssembler`,
construct `H_star`, apply terminal policy, validate, and atomically publish the
next `BlueprintState`.

The likely minimum production file change in the later implementation task is:

```text
adapters/base.py              RegionPredictionSet + bulk-local protocol
adapters/flux2.py             exact dispatch and fail-closed qualification
adapters/flux2_executor.py    private specialized native executor
sampling/euler.py             call bulk-local adapter operation
tests/...                     focused qualification and failure tests
```

`nodes.py`, `state.py`, geometry, region planning/assembly, and terminal policy
need no algorithmic change.

## Exact terminal accepted-step lifecycle

Given immutable accepted state `(G_i, H_i, sigma_i, ordinal_i)` with
`sigma_next == 0` at the exact qualified geometry:

1. The coordinator validates state, schedule, regions, adapter, model profile,
   and specialized execution eligibility before starting transformer work.
2. It retains its existing accepted-G/H immutability snapshots. The adapter
   creates a run-local executor; no object is published globally.
3. Through normal CFG-1 conditioning preparation, the adapter obtains one
   source descriptor from accepted `G_i` and one local descriptor for every
   immutable crop view of accepted `H_i`. Each descriptor has the same sigma,
   conditioning, seed-related options, and its qualified absolute coordinates.
4. The executor applies native FLUX input preparation to the source and all
   crops, producing source/crop image and text hidden state, modulation state,
   and RoPE from the original full-canvas IDs.
5. For double blocks 0–4 and then single blocks 0–19:
   - advance the source through the native block;
   - retain that block's full 18,432 generated source K/V on GPU;
   - sequentially advance crop 0 through crop 54 through the same local block;
   - preserve ordinary text-query attention and augment only generated-query
     attention with the full source generated K/V;
   - after each crop returns, retain only its next-block state;
   - after all crops consume the block, release source K/V;
   - check cancellation and finite/shape invariants at the clean barrier.
6. After single block 19, final-project each crop through native
   `final_layer`, convert the model output through the unchanged CONST
   denoised-estimate contract, and return the ordered `RegionPredictionSet`.
7. The coordinator verifies accepted G/H were not mutated and assembles one
   `x0_H` with the existing `OverlapAssembler`.
8. It constructs the ordinary Euler terminal proposal:

   ```text
   H_star = H_i + (0 - sigma_i) * (H_i - x0_H) / sigma_i
   ```

   Under the current CONST-flow contract this is numerically `x0_H`.
9. `HardNonterminalTerminalRelease.accept_terminal()` retains the prior G as
   `retained_preterminal_unsynchronized` and accepts `H_star` without D/U
   projection.
10. Only after all output, coverage, finiteness, shape, and input-immutability
    checks pass does the coordinator create one next `BlueprintState`. The
    sampler then emits its existing single preview for the interval.

No crop commits state, and source/crop preparation count is not an accepted
step clock.

## Terminal source output contract

The terminal source is required to evolve through all 25 transformer blocks
because each block's generated K/V is consumed by local queries. A terminal
global x0, `G_star`, synchronized final G, and global final projection are not
required:

- terminal release returns only locally assembled `H_star`;
- production already retains preterminal G and explicitly marks it
  unsynchronized;
- no later accepted interval consumes terminal G;
- Phase 8i qualified source block execution without applying the source
  `final_layer`.

Therefore the executor should complete source single block 19, release its K/V
after local consumption, and stop the source path before `final_layer`. This is
not reported as a terminal global prediction. Telemetry must distinguish:

```text
terminal_context_source_performed: true
terminal_context_source_blocks: 25
terminal_global_prediction_performed: false
source_final_projection_performed: false
global_state_status: retained_preterminal_unsynchronized
```

## Guider and conditioning preparation

The executor must not reconstruct conditioning by reading `guider.conds`
directly. The native path performs area/default handling, condition object
processing, hook preparation, condition concatenation, timestep replication,
transformer-option merging, control preparation, `BaseModel.apply_model`
casting, model-sampling input calculation, and model-specific extra-condition
handling before `Flux._forward`.

The minimum safe acquisition mechanism is a scoped ComfyUI
`DIFFUSION_MODEL` wrapper installed only in a copied per-call model-options
tree:

1. call the existing guider once for the source or crop;
2. allow normal CFG-1 `sampling_function` and `calc_cond_batch` preparation;
3. at the native `Flux.forward` wrapper boundary, capture the exact
   `_forward` call descriptor (`x`, processed timestep, context, vector,
   guidance, control/reference arguments, transformer options, and keyword
   conditions);
4. return a correctly shaped sentinel model output; discard the guider result;
5. require exactly one captured invocation with batch one and the expected
   input shape.

The specialized executor then uses native `process_img`, text IDs, native
input modules, modulation, and RoPE to perform the explicit block path. This
uses ComfyUI's normal condition preparation without suspending a transformer
forward and without monkey-patching a core method on disk.

The implementation must use ComfyUI's wrapper registration/merging helpers,
not mutate `guider.model_options` or the model patcher. Existing
`DIFFUSION_MODEL`, `APPLY_MODEL`, `CALC_COND_BATCH`, `PREDICT_NOISE`, context
window, model-function, control, reference, area, hook, or spatial condition
behavior is unqualified and must fail closed before descriptor capture. A
descriptor-count or shape mismatch is an error, not a fallback.

The sentinel call is input preparation, not a model prediction. Telemetry must
report preparation calls separately from global/local model forwards.

## Exact native FLUX.2 Klein qualification

Specialized dispatch is permitted only after explicit checks establish:

- `guider` is the qualified CFG-1/one-positive-branch configuration;
- `guider.inner_model` is exactly native ComfyUI `model_base.Flux2`;
- `diffusion_model` is exactly `comfy.ldm.flux.model.Flux`, not a similarly
  named class;
- CONST model sampling, batch one, 128 latent channels, patch size one;
- hidden width 3072, 24 heads of width 128;
- exactly five native `DoubleStreamBlock` and twenty native
  `SingleStreamBlock` instances;
- context input width 7680 and the live prepared text length 512;
- four-axis RoPE dimensions `[32, 32, 32, 32]` and theta 2000;
- required modules exist with their native types: `img_in`, `txt_in`,
  `time_in`, optional/vector modules as expected, `pe_embedder`, double/single
  modulation, all blocks, and `final_layer`;
- no guidance embedding, reference latents, ControlNet, attention mask,
  timestep-zero indexing, model compilation/replacement, unsupported wrapper,
  model-function override, multigpu path, or dynamic patch changes;
- source IDs span the complete H canvas endpoints and every crop ID equals its
  absolute planned full-canvas coordinates.

These checks qualify the exact native Klein 4B profile used by Phase 8i. They
do not accept Nunchaku, FLUX.1, other FLUX.2 sizes/backends, Z-Image, or Anima.
Do not qualify by class-name duck typing alone.

## Structural memory ownership

Create one run-local `Flux2BlockExecution` object inside the adapter call. Its
only persistent GPU tensor owners across block barriers are:

- source image/text hidden state;
- one image/text or combined hidden state per crop;
- source/crop positional embeddings and modulation state required by native
  blocks;
- immutable source/crop prepared inputs where still required;
- the current block's source K/V;
- immutable accepted G/H references owned by the coordinator.

Within each crop block call, Q/K/V, ordinary attention output, augmented
attention buffers, projections, MLP activations, modulation temporaries,
residual temporaries, and concatenation scratch are local variables only. The
returned next hidden state replaces the prior state. No hooks, closures,
telemetry entries, exceptions, or callbacks may retain live GPU tensors.

After each crop call, drop scratch references. After the block's final crop,
drop source K/V. After final projection and conversion, drop each crop hidden
state as soon as its prediction is published into the ordered result. Compact
telemetry stores integers, floats, shapes, dtypes, and detached CPU summaries
only.

The implementation should assert non-monotonic barrier residency during live
qualification, but it must not call `empty_cache()` per crop or treat allocator
reservation as a leak.

## Failure, cancellation, and atomicity

The coordinator's accepted `BlueprintState` remains the transaction boundary.
The adapter/executor returns no partial prediction set.

- Check ComfyUI interruption before source preparation, between blocks, and
  between crop calls. An interruption raises normally.
- Source failure, crop failure, CUDA OOM, nonfinite state, unexpected shape,
  unsupported patch, or coordinate mismatch aborts the bulk operation.
- Use `try/finally` to clear the current K/V owner, crop/source state lists,
  scoped wrapper state, and any pending ordinary-attention output.
- Never store an executor/session on the adapter, node, model patcher, or
  sampler after the call. There is no cross-run K/V or hidden cache.
- Preserve the original exception with contextual block/crop information via
  exception chaining. Do not convert cancellation into fallback execution.
- Because the coordinator has not assembled, proposed, accepted, or assigned a
  new state, failure leaves the prior immutable state authoritative.
- The next queued run constructs a new adapter-local executor and cannot see
  stale context from the failed run.

OOM handling may release run-owned references and re-raise. It must not retry
with CPU K/V offload, compressed context, ordinary terminal crops, or another
geometry because those paths have different qualified semantics.

## Preview and telemetry

The specialized operation returns the same ordered crop x0 objects expected by
the existing assembler. The coordinator therefore produces the same assembled
`x0_H`, and `BlueprintEulerSampler` continues to call:

```text
callback(ordinal, assembled_x0_H, accepted_H, total_intervals)
```

exactly once after atomic acceptance. No block/crop progress becomes a preview
or accepted-step event.

Required compact telemetry includes execution mode, source/context token
count, source blocks, local blocks, crop count, coordinate endpoints, peak
allocated/reserved CUDA, per-block barrier residency, source/local CUDA time,
and explicit zero CPU K/V cache/transfer. It must not retain diagnostic model
tensors.

## Qualification matrix for the implementation task

### Mandatory correctness and lifecycle gates

1. Unit tests: exact specialized dispatch geometry, ordinary fallback,
   ordered prediction cardinality/shapes, immutable state, terminal-only
   selection, and no partial result publication.
2. Gate-1 native test: one ordinary crop is bit-exact at input embeddings,
   every double/single block, final projection, and x0.
3. Gate-2 native test: one full-context crop has bit-exact source K/V, local
   block outputs, final projection, and x0.
4. Small-canvas live regression: existing qualified geometries do not select
   specialized execution and remain bit-exact with the frozen production
   baseline.
5. 2048×4096 bridge regression: 55 crop predictions and assembled x0 meet the
   Phase-8i tensor gate and the decoded output preserves the qualified single
   bridge system.
6. Two fresh queued 2048×4096 runs produce deterministic latent and decoded
   hashes under the same VAE decode mode.
7. Cancellation at preparation, a double block, a single block, and a late
   crop leaves no accepted partial state; a following valid run succeeds.
8. Injected source/crop shape failure plus a controlled OOM path verifies
   cleanup, exception context, atomicity, and restartability.
9. Live `BasicGuider -> Blueprint sampler -> SamplerCustomAdvanced -> VAE
   Decode -> Save Image` completion with one preview per interval.
10. Memory telemetry proves no all-layer K/V container, zero CPU K/V cache,
    zero CPU-to-GPU K/V transfer, bounded barrier residency, and no
    crop-ordinal growth.

The existing fail-closed matrix for masks, nonempty latents, CFG other than
one, alternate model families/backends, reference/ControlNet/spatial
conditioning, partial denoise, and non-CONST/non-Euler behavior remains
mandatory.

### Optional characterization after correctness

- cold/warm terminal timing beyond the mandatory evidence;
- allocator fragmentation across several repeated runs;
- secondary large-canvas prompts;
- detailed per-block CUDA breakdown.

None of these optional measurements can weaken a mandatory correctness,
semantic, cancellation, or memory-ownership gate.

## Recommendation

Proceed with a terminal-only specialized native FLUX.2 Klein executor, private
to `Flux2Adapter`, selected only for the exact qualified H=128×256 geometry.
Keep all nonterminal execution, accepted-state lifecycle, coupling, assembly,
and callbacks unchanged. Do not introduce a generic execution-session API or
give the adapter ownership of a whole Blueprint interval.

**PROCEED WITH TERMINAL-ONLY SPECIALIZED FLUX EXECUTOR**
