# Candidate 3 production architecture

## Decision scope

Candidate 3 is the initial Blueprint Diffusion production architecture. The
first implementation must reproduce the qualified two-state FLUX.2 contract;
it must not generalize the algorithm while doing so.

The selected ComfyUI boundary is a custom `SAMPLER` object used through the
existing guider and `SamplerCustomAdvanced`. A stateless `MODEL -> MODEL`
patch is not the state owner: ComfyUI's Euler callback runs before the native
Euler assignment, while Blueprint must replace that assignment with one atomic
`(G, H)` acceptance. A `SAMPLER_SAMPLE` wrapper could technically replace the
entire sampler, but would hide the required sampler restriction behind a model
patch and make ordinary sampler selection misleading.

The first UI therefore consists of a **Blueprint Candidate-3 Euler Sampler**
node which outputs `SAMPLER`. Users connect the normal `MODEL` to a normal
ComfyUI guider, and connect that guider, Blueprint sampler, noise, sigmas, and
latent to `SamplerCustomAdvanced`:

```text
MODEL -> Basic/CFG Guider -----------\
                                       SamplerCustomAdvanced -> LATENT
Blueprint Candidate-3 Euler -> SAMPLER /
```

No model patch is required for the first FLUX.2 slice. Full-canvas positional
options are supplied per global/local guider call by the FLUX.2 adapter. If a
future model family genuinely requires a cloned model patch, that patch may
carry adapter configuration, but it must not own accepted trajectory state.

## Fixed production contract

Let `H_i` be the accepted target-grid latent, `G_i` the accepted global-grid
latent, and `sigma_i` their shared noise level. For the qualified 32 x 64
target grid:

```text
H_i: [B, C, 32, 64]
G_i: [B, C, 24, 48]
```

The geometry generalizes only mechanically to target latent grids whose two
axes are divisible by four: each nonoverlapping 4 x 4 spatial block of every
batch/channel is transformed by the constant-preserving orthonormal 2-D DCT.
`D` retains the lowest 3 x 3 coefficients and packs those coefficients as one
3 x 3 block in `G`. `U` unpacks each 3 x 3 block, zero-pads the missing DCT
row/column to 4 x 4, and applies the inverse orthonormal DCT. In exact
arithmetic:

```text
D(U(G)) = G
```

Production startup must numerically test this identity for the active
device/dtype and reject the run if the configured tolerance is exceeded. The
operator is channel-agnostic and model-external; it does not decode latents.

The initial sampling state is constructed after the model's ordinary input
noise scaling:

```text
H_0 = ordinary sampler initial state
G_0 = D(H_0)
state_0 = (G_0, H_0, sigma_0, ordinal=0)
```

There is no independent global noise draw. The mapped global variance and
operator identity are logged because the qualified behavior includes that
initialization distribution.

## Responsibilities

### `BlueprintEulerSampler`

- Implements only deterministic, zero-churn, first-order Euler over the
  supplied sigma list.
- Obtains the normal model/guider prediction contract from ComfyUI rather than
  bypassing CFG or conditioning preparation.
- Creates the run-local coordinator and state after ordinary noise scaling.
- Calls exactly one global prediction and all planned local predictions at the
  same accepted state and sigma.
- Computes both Euler proposals, asks the terminal policy for the acceptance,
  validates it, and publishes one new state tuple.
- Calls ComfyUI's preview callback once per accepted interval with the assembled
  local denoised estimate and newly accepted `H`.
- Returns final `H`; `G` remains run-local diagnostic state.

### `BlueprintCoordinator`

- Contains model-independent orchestration only.
- Holds immutable run configuration: operator, planner, assembler, adapter,
  and terminal policy.
- Never mutates accepted tensors. It passes views of accepted `H` to local
  execution, collects predictions, and builds proposals in new tensors.
- Checks shared sigma, call counts, finite values, shapes, complete positive
  coverage, right-inverse error, and acceptance ordinal before commit.
- Commits with one assignment of a validated `BlueprintState`; exceptions
  abort the run with the previous tuple still intact.

### `BlockDCTGeometry`

- Defines global shape, `D`, `U`, startup identity qualification, and invariant
  tolerance.
- Owns no model and no trajectory state.
- Initially accepts only 4-D image latents with both spatial axes divisible by
  four. The qualified 32 x 64 grid maps to 24 x 48 (1152 tokens).

### `FixedCropPlanner` and `OverlapAssembler`

- The planner converts the accepted target grid into an ordered list of
  immutable crop rectangles. The first slice uses the qualified three 32 x 32
  rectangles and overlap geometry.
- Every crop is a view/read of the same accepted `H_i`. No crop may receive a
  proposal or update `H`.
- The assembler applies precomputed normalized overlap weights, asserts
  strictly positive full-canvas coverage, and returns exactly one full `H`
  prediction plus telemetry.
- Crop coordinates and weights are generic spatial data, not FLUX token IDs.

### `ModelFamilyAdapter`

- Validates that the model family and latent/token geometry are supported.
- Converts a generic full-canvas region descriptor into model-specific spatial
  options without changing accepted state.
- Executes one normal guider call and returns a prediction in the input
  latent's shape.
- Does not perform sampling, coupling, crop assembly, or state mutation.

The minimal interface is:

```python
class ModelFamilyAdapter(Protocol):
    family: str

    def validate_run(self, *, guider, high_shape, global_shape,
                     crops, sigmas, latent) -> None: ...

    def predict_global(self, *, guider, g, sigma, canvas,
                       model_options, seed) -> Tensor: ...

    def predict_region(self, *, guider, h_view, sigma, canvas, region,
                       model_options, seed) -> Tensor: ...

    def describe_work(self, *, global_shape, crops) -> WorkEstimate: ...
```

`canvas` and `region` use grid coordinates plus the complete canvas extent;
they do not expose RoPE keys in the generic API. The initial `Flux2Adapter`
maps each crop to the already-qualified absolute coordinate offsets. Its
24 x 48 global grid spans the 32 x 64 target extent with:

```text
scale_y = 31 / 23
scale_x = 63 / 47
```

and zero shifts under the qualified convention. The adapter records the
actual coordinate endpoints and refuses a backend that cannot accept the
required rectangular full-canvas coordinate mapping. Z-Image and Anima can
later implement this protocol with their own coordinate preparation; no
generic component assumes FLUX token ordering, RoPE option names, 128 latent
channels, or a 25-block transformer.

### `BlueprintState`

An immutable run-local value:

```text
BlueprintState(
    g, h, sigma, ordinal,
    accepted_evaluation_id,
)
```

It owns the only accepted tensor references. Proposals, predictions,
coverage, and projections are temporary evaluation data and are never placed
in the accepted state. Debug telemetry may retain detached CPU summaries, not
live GPU tensors by default.

### `HardNonterminalTerminalRelease`

- For `sigma_next > 0`, returns
  `G_next = G_star` and
  `H_next = H_star + U(G_star - D(H_star))`.
- For `sigma_next == 0`, returns `G_next = G_star`, `H_next = H_star`.
- Rejects negative, increasing, repeated, nonfinite, or ambiguous terminal
  sigmas. Terminal means numerically exact zero in the supplied schedule; no
  epsilon policy is introduced.
- Enforces `D(H_next) = G_next` only on nonterminal acceptances and records the
  intentional final mismatch.

## Exact accepted-step lifecycle

For interval `i`, with the immutable accepted tuple `(G_i, H_i, sigma_i)`:

1. Validate ordinal, `sigma_i == sigmas[i]`, and, for `i > 0`, the prior
   nonterminal invariant.
2. Create an evaluation ID from run ID, ordinal, `sigma_i`, and
   `sigma_(i+1)`. Do not derive it from model-call count.
3. Call the adapter once on `G_i` using the whole-canvas global coordinates.
4. For every planned region, take a view of the unchanged `H_i`, call the
   adapter with that region's absolute full-canvas coordinates, and retain the
   returned prediction. No state assignment occurs in this loop.
5. Normalize/blend all local predictions into one full-canvas `x0_H`; validate
   coverage. The global call supplies `x0_G`.
6. Form ordinary Euler proposals using the same sigma pair:

   ```text
   G_star = G_i + (sigma_next - sigma_i) * (G_i - x0_G) / sigma_i
   H_star = H_i + (sigma_next - sigma_i) * (H_i - x0_H) / sigma_i
   ```

7. Apply the fixed policy. Nonterminal acceptance projects `H_star`; terminal
   acceptance releases it unchanged.
8. Validate shapes, finiteness, right-inverse/projection identity, and the
   nonterminal `D(H_next) == G_next` invariant. Confirm accepted input storage
   was not mutated during any call.
9. Atomically assign one new `BlueprintState(G_next, H_next, sigma_next,
   ordinal+1, evaluation_id)`. Then and only then emit telemetry/preview.

The global and local branches may execute sequentially, but they consume the
same accepted tuple. A failed call or validation commits neither proposal.

```text
accepted BlueprintState_i
        | immutable G_i ----------------> global adapter call -> x0_G -> G*
        | immutable H_i -> crop views --> local adapter calls  -> blend -> H*
        |                                                        |
        +---------------- policy(D,U,sigma_next) <---------------+
                                 |
                         validate complete pair
                                 |
                    one atomic BlueprintState_(i+1)
```

## Fail-closed first-slice support

The first production slice supports only the behavior actually qualified:

- native ComfyUI FLUX.2 Klein execution accepted by `Flux2Adapter`;
- batch size one, 4-D image latent, 32 x 64 target grid;
- fixed three-crop 32 x 32 layout and qualified normalized overlaps;
- 24 x 48 block-DCT global state;
- T2I from the ordinary initial noise state;
- CFG 1 / one conditioning branch as used by the qualification;
- deterministic zero-churn Euler with a finite strictly decreasing schedule
  ending in exactly zero;
- no denoise mask, inpainting, outpainting, reference latent, nested latent,
  video/time axis, ControlNet or conditioning whose spatial tensors cannot be
  safely remapped for both global and crop calls.

Reject rather than approximate: other samplers, Euler ancestral/churn, partial
denoise or nonzero latent initialization, batches greater than one, unsupported
canvas/crop geometry, a model/backend without coordinate override, incompatible
conditioning, nonfinite tensors, incomplete coverage, or failed coupling
invariants. Z-Image and Anima remain unsupported until their adapters and
ordinary prediction semantics are separately qualified.

## Expected work and first production slice

Each accepted interval executes one 1152-token global call plus three
1024-token local calls: 4224 generated-token executions. Relative to the former
512-token global branch, generated-token work rises 17.86%; the approximate
image-image attention-matrix term rises 31.25%. This is not a total-FLOP,
wall-clock, or VRAM-saving claim. Calls remain sequential, no K/V cache exists,
and persistent GPU state is only `G` plus `H`.

The smallest implementation slice is:

1. `Blueprint Candidate-3 Euler Sampler` returning a ComfyUI `SAMPLER`;
2. immutable state and explicit coordinator implementing the fixed lifecycle;
3. fixed block-DCT geometry, fixed three-crop planner/assembler, and one native
   `Flux2Adapter`;
4. startup/runtime assertions and compact per-step telemetry;
5. one regression workflow through `SamplerCustomAdvanced` reproducing the
   Phase-3c scene and Phase-3d/3e invariants.

No production model node, external K/V, adaptive geometry, soft anchoring,
caching, sparse execution, alternate sampler, or generalized adapter registry
belongs in that slice.
