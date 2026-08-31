# Candidate 3: coupled low/high trajectory design

## Scope and question

This note defines a falsifiable research contract for Candidate 3. It does not
define a production sampler or node.

Candidate 3 asks whether a complete, ordinarily interacting low-density FLUX.2
trajectory can keep a persistent high-resolution trajectory organized while
ordinary high-resolution crop forwards add regional fidelity. It deliberately
does not append compact global K/V to local attention (Candidate 2), and it does
not resize a global model prediction and add a local prediction residual
(Candidate 1). Coupling occurs between the two **accepted latent states** after
both branches have taken the same sampler interval.

The first probe should answer only:

```text
Does exact coarse-state anchoring improve the four-step tiled trajectory's
whole-scene organization, while the unconstrained high-frequency state adds
visible detail beyond the low-density trajectory?
```

## State and geometry contract

For the existing FLUX.2 composition-stress setup, let

```text
H_i : [B, 128, 32, 64]  persistent 1024 x 512 full-canvas latent state
G_i : [B, 128, 16, 32]  persistent  512 x 256 whole-canvas latent state
sigma_i                 shared accepted noise level
```

The spatial grids retain the same full-canvas extent and aspect ratio. A
position in `G` represents the corresponding `2 x 2` cell in `H`; it is not a
crop. The normal FLUX.2 positional mapping for `G` spans the whole target
canvas, and every `H` crop keeps its existing full-canvas coordinate offset.

Define two fixed, model-external spatial operators:

```text
D(H) = 2 x 2 area downsample of H
U(G) = 2 x nearest-neighbor expansion of G
```

They operate independently per batch and latent channel and satisfy `D(U(G)) =
G` up to floating-point error. The accepted-state consistency invariant is

```text
D(H_i) = G_i.
```

This is an intentionally coarse constraint. It does not assert that the VAE or
FLUX latent space has an ideal linear scale decomposition. Whether this
operator preserves useful semantics without suppressing detail is part of the
probe.

### Initialization

Generate the ordinary full-canvas initial noise `H_0` exactly as in Phase 2,
from seed `20260829`, and set

```text
G_0 = D(H_0).
```

This gives exact spatial correspondence and satisfies the invariant from the
first evaluation. For nonoverlapping area cells it reduces global noise
variance relative to an independently sampled unit Gaussian (by a factor of
four for ideal independent input samples), although it does not itself couple
adjacent cells. That distribution shift is not hidden: record the
per-channel/global mean, RMS, standard deviation, and adjacent spatial
correlation for `H_0` and `G_0` to verify the actual implementation.
The existing Phase-2 compact branch used a resized current latent and provides
a practical precedent, but not proof that this is an ideal low-resolution
prior.

The first probe is interpretable only if the independently evolved `G`
trajectory forms a coherent low-resolution scene. If it does not, the coupling
result is **inconclusive**, not evidence against persistent two-level
trajectories. A separately sampled `G_0` would restore the ordinary marginal
noise distribution but introduce an uncontrolled competing scene hypothesis;
it is therefore a worse first discriminator.

## One synchronized Euler transition

Both states use the same sigma schedule and conditioning. Let `M(x, sigma)` be
the ordinary ComfyUI denoised model output, and let

```text
Euler(x, x0, sigma_i, sigma_next)
    = x + (sigma_next - sigma_i) * (x - x0) / sigma_i.
```

At evaluation `i`:

1. Assert that `G_i` and `H_i` are the current accepted states and that
   `D(H_i)` agrees with `G_i` within a recorded tolerance.
2. Run one ordinary whole-canvas model forward on `G_i` at `sigma_i`, producing
   `x0_G`.
3. Run the existing three ordinary `32 x 32` high-resolution crop forwards on
   `H_i`, at that identical sigma and conditioning. Assemble one full `x0_H`
   with the existing normalized overlap weights. No crop updates state.
4. Form the independent Euler proposals:

   ```text
   G* = Euler(G_i, x0_G, sigma_i, sigma_next)
   H* = Euler(H_i, x0_H, sigma_i, sigma_next)
   ```

5. Apply the selected coupling exactly once:

   ```text
   G_(i+1) = G*
   H_(i+1) = H* + U(G* - D(H*))
   ```

6. Assert `D(H_(i+1)) = G_(i+1)`, publish both accepted states, and advance the
   shared evaluation ordinal exactly once.

The correction `U(G* - D(H*))` is a **state projection after two ordinary
Euler proposals**. It is not added to `x0_H`, is not fed through individual
crop updates, and is not an upsampled global prediction/local-prediction
residual. `G` has its own persistent model trajectory; changing `G_i` changes
its next model input even if `H_i` is held fixed.

One accepted sampler transition is therefore well-defined as the ordered pair

```text
(G_i, H_i) -> coupled_accept(G*, H*) -> (G_(i+1), H_(i+1)).
```

There is one accepted high-resolution canvas update per sigma interval, not
three crop updates. A ComfyUI production integration would need lifecycle
coordination outside a stateless model patch, but the experiment can use the
same explicit `Sampler` harness already used for Phase 2e.

### Information flow

- `G -> H`: the complete accepted coarse state of `H` is replaced by the
  independently evolved global state after every proposal.
- `H -> G`: none in the first experiment. This preserves `G` as the causal
  global anchor and prevents crop-local semantic alternatives from contaminating
  it.
- Within `H`: overlap blending and persistence carry local information between
  evaluations; only the component visible through `D` is constrained.

This differs from two independent trajectories because `H_(i+1)` is a direct
function of `G_(i+1)` and must satisfy the cross-state invariant at every
accepted step. In the uncoupled control, evolving `G` has exactly no effect on
`H`.

## Ranked coupling rules

### 1. Hard accepted-state global anchor — first experiment

```text
G_next = G*
H_next = H* + U(G* - D(H*))
```

Why first: it has no coupling-strength parameter, preserves one authoritative
normal global trajectory, and supplies the clearest causal test. It also makes
the strongest possible version of the hypothesis easy to reject.

Primary risk: the fixed downsample nullspace is not a semantic/detail split.
Candidate 1/2 evidence already shows that duplicate objects and geometry can
survive in nominally high-frequency components. Hard anchoring can therefore
produce a coarse-consistent image that still contains a duplicate, or it can
repeatedly erase useful local low-frequency structure and cause temporal
flicker/blockiness.

### 2. Soft proximal global anchor

```text
G_next = G*
H_next = H* + lambda * U(G* - D(H*)),  0 < lambda < 1
```

This trades exact synchronization for less destructive correction. It is
ranked second because `lambda` introduces a sweep temptation and weakens the
causal test. It is warranted only if the hard anchor controls semantics but
demonstrably suppresses local fidelity.

### 3. Bidirectional consensus

```text
C      = (w_G * G* + w_H * D(H*)) / (w_G + w_H)
G_next = C
H_next = H* + U(C - D(H*))
```

This enforces exact mutual consistency and allows stable high-resolution
evidence to update the global branch. It is ranked last for the first pass:
local low-frequency semantic alternatives can rewrite the supposedly global
plan, and the two weights add an unqualified policy. It becomes relevant only
if one-way anchoring is coherent but persistently conflicts with legitimate
high-resolution structure.

No rule based on fixed high-pass prediction fusion is retained. Phase 2b found
that semantic divergence remains in the high-pass correction, so frequency
labels alone do not justify accepting local semantic content.

## Expected work and memory

For the exact Phase-2 geometry, each accepted Candidate-3 evaluation performs:

```text
global:        1 forward x  512 generated tokens
local:         3 forwards x 1024 generated tokens
total:         4 forwards, 3584 generated-token executions
dense control: 1 forward  x 2048 generated tokens
```

Ignoring text tokens and constants, generated image-image attention matrices
scale approximately as

```text
512^2 + 3 * 1024^2 = 3,407,872
2048^2            = 4,194,304
ratio              = 0.8125 of dense
```

This is **not** a total-FLOP saving claim. Image input/output projections,
normalization, modulation, residual paths, and MLPs execute over 3584 generated
tokens, `1.75x` the dense control's generated-token count, and four calls add
launch/model overhead. The first all-crop probe tests semantics and activation
shape, not end-to-end acceleration. A later architecture would need selective
local coverage or fewer local evaluations to establish compute reduction.

Calls are sequential, so transformer activation peak is bounded by the larger
1024-token crop call rather than a 2048-token dense call. Persistent storage
adds `G`, one global denoised output/proposal while active, and the coarse
consistency delta; it does not add per-block K/V caches. Peak VRAM and
wall-clock must still be measured rather than inferred.

## Failure modes and decisive telemetry

The probe must record per evaluation:

- shared sigma, accepted ordinal, and exactly one global plus three local
  forwards;
- object identities proving every crop read the same accepted `H_i` and that
  both proposals were accepted together;
- `RMS(D(H_i) - G_i)` before evaluation, before projection, and after
  projection;
- RMS/norm of the state projection and its ratio to `H*`, split by step;
- global and assembled-local denoised/prediction norms;
- crop coverage and pre-blend overlap disagreement;
- global/local token counts, forwards, peak VRAM, and wall time;
- decoded `G_i`, `U(G_i)`, uncoupled `H_i`, proposed `H*`, projected
  `H_(i+1)`, and final outputs at every step.

Qualitative scoring remains composition-first: lighthouse and stone-tower
uniqueness, centered train count/placement, continuous bridge deck/cables,
horizon/water continuity, and only then local sharpness/detail.

The main failure interpretations are:

- **Global branch fails independently:** mapped-noise distribution or compact
  trajectory is invalid; result is inconclusive for coupling.
- **Invariant holds but duplicates survive:** the nullspace of `D` still carries
  semantic alternatives; hard coarse state consistency is insufficient.
- **Composition follows `G` but detail disappears:** coupling is semantically
  effective but overconstrained; only then test the soft rule.
- **Large alternating projection corrections:** local and global vector fields
  encode incompatible scene hypotheses; the coupled integrator is unstable.
- **Good four-step image but high work:** semantic architecture survives, while
  efficiency remains unqualified.

## Selected first real FLUX.2 experiment

Reuse the exact Phase-2e model, prompt, seed, CFG, four-step Euler sigmas,
1024 x 512 target, 512 x 256 global grid, three 512 x 512 crops at latent x
offsets `0`, `24`, and `32`, full-canvas RoPE, and normalized overlap weights.
Run only these controlled variants:

```text
A. DENSE
   ordinary full-canvas four-step trajectory

B. TILED-ONLY
   existing three-crop assembled four-step trajectory

C. UNCOUPLED DUAL CONTROL
   evolve G and H at matched sigmas, but do not couple them
   (H must match B within numerical tolerance)

D. HARD GLOBAL-ANCHOR COUPLED
   evolve both proposals and apply the rank-1 accepted-state projection
```

Do not add external K/V, prediction residual fusion, strength sweeps, alternate
initializers, or additional coupling rules. The uncoupled control proves that
mere execution of `G` cannot change `H`; any B/C mismatch is a lifecycle bug.

The rank-1 rule survives this first falsifier only if D preserves materially
more of G's whole-scene organization than B/C while showing useful local
structure absent from decoded/upsampled G. It fails if exact coarse agreement
holds but crop-local duplication remains, if state corrections destabilize the
trajectory, or if the result is only a smoothed/upscaled global image. General
quality and efficiency are explicitly outside this single-scene gate.
