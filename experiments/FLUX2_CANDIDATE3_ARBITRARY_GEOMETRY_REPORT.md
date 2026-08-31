# Candidate-3 arbitrary target geometry qualification

Date: 2026-08-31

## Scope

Phase 5 generalizes only spatial geometry. It preserves the qualified
Candidate-3 algorithm, four-step CONST/Euler schedule, atomic `(G,H)` state,
hard nonterminal coupling, terminal release, CFG-1 T2I boundary, and native
FLUX.2 execution. No optimization, alternate sampler, editing, Nunchaku, or
additional model family was added.

## Geometry contract

For a target latent grid `H` of `(h,w)`:

- batch remains exactly one;
- both axes must be divisible by four;
- both axes must be at least 32 so every local call is an ordinary 32x32 crop;
- `D` independently partitions every 4x4 block, retains its lowest 3x3
  orthonormal-DCT coefficients, and represents those coefficients as a 3x3
  global block;
- `G` therefore has shape `(3h/4,3w/4)`;
- `U` applies the exact inverse 3x3 DCT representation, zero-pads the missing
  4x4 frequency row/column, and synthesizes each 4x4 high block;
- the existing `D(U(G)) = G` tolerance remains `2e-6`.

The operation is block-local and unchanged from the qualified 32x64 to 24x48
mapping. Only the number of blocks changes.

## Crop plan and assembly

Each axis uses crop length 32 and deterministic stride 24. Starts are generated
from zero, and the final start `axis_length - 32` is appended when the stride
does not land on it. Two-dimensional regions are the row-major Cartesian
product of the Y and X starts. This reproduces the original 32x64 plan exactly:

```text
(y,x) = (0,0), (0,24), (0,32)
```

The existing separable overlap ramps and normalized accumulation are retained.
Assembly now requires one prediction per planned region instead of exactly
three, validates that every region lies inside the target, and fails if any
pixel has zero/nonfinite coverage. Crop calls remain immutable views of the
accepted `H`; no crop commits state.

## FLUX.2 coordinates

Local calls retain ordinary 32x32 local coordinates and receive absolute
offsets:

```text
shift_y = crop.y
shift_x = crop.x
```

The global branch spans the full target coordinate endpoints:

```text
scale_y = (h - 1) / (global_h - 1)
scale_x = (w - 1) / (global_w - 1)
```

The adapter copies model/transformer options before adding `rope_options`.
Unit tests confirm the wide 32x80 case uses `scale_y=31/23`, `scale_x=79/59`,
and the rightmost crop uses `shift_x=48` without modifying caller options.

## Tests

Fifteen focused tests pass. Added coverage includes:

- DCT right-inverse and constant-field preservation at 32x32, 64x32, 32x80,
  and 36x68;
- rejection of nonpositive or non-divisible-by-four target geometry;
- deterministic plans and normalized full coverage for square, portrait,
  wide, and two-dimensional multi-crop grids;
- rejection when an axis is smaller than 32;
- exact FLUX.2 global endpoint scale and local crop offset options.

## Real ComfyUI qualification

The installed custom node ran the same Phase-3c prompt, seed, native model,
four-step scheduler, normal guider, `SamplerCustomAdvanced`, preview callback,
VAE decode, and SaveImage path for all three cases:

| Canvas | H | G | Global tokens | Crops | Local tokens/interval | Forwards/interval | Result |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 512x512 square | 32x32 | 24x24 | 576 | 1 | 1024 | 2 | success, 4 previews |
| 512x1024 portrait | 64x32 | 48x24 | 1152 | 3 | 3072 | 4 | success, 4 previews |
| 1280x512 wide | 32x80 | 24x60 | 1440 | 3 | 3072 | 4 | success, 4 previews |

The machine-readable results and decoded outputs are under
`live_comfyui_candidate3_geometry_results/`. Successful completion also means
the runtime nonterminal `D(H_next)=G_next` assertions remained within tolerance
for every accepted nonterminal interval.

Visual inspection found one coherent person/car/tree scene in all three aspect
ratios, continuous ground and horizon, no crop seams, no coordinate displacement,
and no obvious DCT macroblock pattern. This is a geometry/execution
qualification, not a broad semantic generalization claim.

## Result

Candidate-3 now supports practical arbitrary latent grids whose axes are at
least 32 and divisible by four, using the same fixed semantics and explicit
fail-closed boundary.
