# Blueprint staged refinement: semantic gate and first discriminator

## Scope and relation to existing nodes

This note begins a new staged-refinement research branch without changing or
reclassifying Candidate-3 or `Blueprint Terminal Resampling`.  No production
node is authorized by this note.

The requested first denoised-handoff pipeline is algorithmically close to the
frozen Terminal Resampling path: both finish a bounded whole-canvas trajectory,
map its terminal denoised estimate, construct late-sigma local states, run
independent bounded local model calls, restrict them, and assemble normalized
overlaps.  A new sibling node is justified only if a newly tested contract
differs materially (for example open destination/footprint geometry and a
bounded crop-before-upscale transfer), not by renaming the frozen node.

## Source semantics

### MrFlow

The official paper and implementation use this sequence:

1. finish low-resolution T2I sampling;
2. decode the low-resolution result to pixels;
3. apply a learned Real-ESRGAN pixel-space x2 super-resolution model;
4. pass that image through the model VAE encoder in an img2img pipeline;
5. inject scheduler-consistent low-strength noise;
6. refine at high resolution with a short direct-sigma schedule.

The released helper replaces the scheduler schedule with a linear sigma list
from `first_sigma` to zero.  The default FLUX.1 example uses 12 low-resolution
steps and one refinement interval from `0.12` to zero.  Applicable independent
concepts are the completed low-resolution layout, denoised handoff, explicit
late-sigma construction, and short high-resolution refinement.  The learned
pixel upscaler and pixel decode/encode transfer are deliberately excluded here;
therefore this is MrFlow-like, not a reproduction or claimed equivalent.

Primary sources:

- paper: https://arxiv.org/abs/2607.01642
- official repository: https://github.com/Xingyu-Zheng/MrFlow
- staged FLUX example and direct-sigma helper (current mirror inspected):
  https://github.com/Ardynai/mrflow/blob/main/flux1_mrflow.py and
  https://github.com/Ardynai/mrflow/blob/main/mrflow_utils.py

### Current ComfyUI and FLUX.2 Klein/CONST

Current local ComfyUI source at commit
`5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`
(`comfy/model_sampling.py`) defines CONST:

```text
noise_scaling(sigma, epsilon, x0)
  = sigma * noise_scale * epsilon + (1 - sigma) * x0
```

For the qualified Klein profile `noise_scale == 1`, denoised handoff is exactly

```text
x_sigma = sigma * epsilon + (1 - sigma) * x0
```

This equality is a consequence of the active model-sampling object, not a
generic diffusion formula.  The experiment and any later node must call
`model_sampling.noise_scaling(..., False)` and fail closed unless the prepared
model is native Klein/CONST with `noise_scale == 1`.  ComfyUI's ordinary
`KSAMPLER.sample()` applies this operation once to its input noise and latent
image at `sigmas[0]`, then applies `inverse_noise_scaling(sigmas[-1], output)`.
At a terminal zero sigma, CONST inverse scaling is the identity.

The model prediction is the denoised estimate
`x0 = x_t - sigma * model_output`; an Euler flow interval is

```text
v = (x_t - x0) / sigma
x_next = x_t + (sigma_next - sigma) * v
```

Normal `guider.sample()` remains the required outer integration boundary for a
future node so conditioning, wrappers, model preparation, cancellation, and
cleanup stay owned by ComfyUI.  A private staged procedure may own accepted G
and transient W state, as proven by Phase 27.

### RES4LYF comparison

RES4LYF exposes three distinct sampler modes: `standard`, `unsample`, and
`resample`.  Its current sampler constructs unsampling schedules by reversing
the chosen sigma sequence and uses zero-valued sigma padding as a control flag;
resampling also pads the schedule with zero endpoints.  Both modes disable the
ordinary ComfyUI initial-noise addition because the RES sampler owns its state
and trajectory semantics.  It returns separate output, denoised, and SDE-noise
latents and may carry sampler/guider/state metadata across chained nodes.

This is useful evidence that noisy trajectory state and denoised estimates are
different contracts.  It does not establish that an arbitrary resized noisy G
state is a valid W state: spatial resizing changes its noise covariance and a
G sigma does not by itself define the corresponding W trajectory.  No noisy
handoff option is exposed in this branch until a forward/inverse trajectory
contract, covariance law, and controlled test exist.

Source: https://github.com/ClownsharkBatwing/RES4LYF/blob/main/README.md and
`beta/samplers.py` at the same revision inspected on 2026-09-06.

## Geometry and coordinate contract

All geometry below is in latent cells:

- `G=(G_h,G_w)`: bounded whole-canvas Blueprint latent;
- `H=(H_h,H_w)`: complete destination latent and assembly domain;
- `F=(y,x,f_h,f_w)`: one destination footprint;
- `W=(W_h,W_w)`: bounded local model working canvas;
- `S=(s_h,s_w)`: deterministic footprint stride; overlap is `F-S` per axis.

Footprints use row-major, end-aligned starts: generate regular starts at the
stride and append `length-size` if needed.  This is deterministic, includes the
far edge, permits overlap, and must produce strictly positive assembly coverage.
Each final W prediction is resized/restricted to exactly its footprint and
combined by deterministic separable feather weights divided by accumulated
weights.

Destination cell center `(h_y,h_x)` maps to continuous G coordinates with
PyTorch bilinear `align_corners=False` semantics:

```text
g_y = (h_y + 0.5) * G_h/H_h - 0.5
g_x = (h_x + 0.5) * G_w/H_w - 0.5
```

Coordinates are border-clamped.  Crop-first transfer retains only the smallest
G rectangle containing both bilinear neighbors for every footprint cell,
samples the footprint from that rectangle, and then resizes the footprint to W.
It must numerically match:

```text
bilinear_resize(G, H) -> crop(F) -> bilinear_resize(F, W)
```

without materializing the complete mapped-H anchor.  The destination output
and normalized assembly buffers still scale with H; model working state and
the transfer source crop remain bounded by F/W and interpolation support.

## Selected first experiment

Run `blueprint_staged_refinement_geometry.py` before any new model inference.
It compares complete-map-then-crop against halo-aware crop-before-upscale for
square, portrait, wide, non-divisible, and larger arbitrary destination shapes.
It checks exact tile order, complete coverage, finite normalized assembly,
transfer error, and peak tensor-element accounting.

This is the minimum new discriminator because existing evidence already covers
the other first-variant axes:

- Phase 25/27: one-way denoised terminal handoff preserves S3 composition and
  normal guider lifecycle with zero destination-sized model calls;
- Phase 29: fixed starts `0.10, 0.15, 0.25, 0.35, 0.50` preserve S3 but do not
  produce a consistent credible detail improvement;
- Phase 38: a four-interval persistent-W refinement from `0.25` preserves S3
  but adds mostly texture/line energy, not resolved object structure.

Repeating those GPU calls under a new label would not answer a new question.

## Acceptance and falsification gates

Geometry passes only if crop-first and complete-map paths agree within
`1e-5`, planning is deterministic, coverage is positive everywhere, and the
largest transient transfer tensor is independent of destination area.

The staged-refinement semantic premise passes only if, for a small declared
set of refinement starts, the tiled result:

- retains the realized G object count, placement, perspective, large geometry,
  and cross-footprint continuity (S3);
- improves credible local structure over plain resized/decoded G, not merely
  gradient energy or alias texture;
- outperforms a matched tiled-only control semantically;
- uses no H-sized model call and has flat per-region post-barrier allocation.

Existing evidence passes composition but fails to establish the required
credible-detail gain.  Therefore this first branch remains **not accepted for
production**.  If later inference is authorized, it should change one causal
variable beyond the already-tested Terminal contract and compare against the
persisted controls.  Do not add interleaving, guide, ControlNet, learned
upscaling, or noisy handoff to rescue a failed result.
