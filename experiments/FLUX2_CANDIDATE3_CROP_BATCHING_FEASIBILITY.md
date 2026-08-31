# Candidate-3 local crop batching feasibility

Date: 2026-08-31

## Question

Can native ComfyUI FLUX.2 evaluate multiple existing 32x32 Blueprint crops in
one model batch while assigning a distinct absolute full-canvas `(y,x)` offset
to each batch element?

This is a correctness gate. No production code, Candidate-3 lifecycle,
geometry, crop plan, assembly, or model execution was changed.

## Native coordinate path

The inspected implementation is
`comfy.ldm.flux.model.Flux.process_img` in the approved ComfyUI development
checkout. For an input `x` with batch size `B`, it:

1. derives one scalar `h_offset` and `w_offset`;
2. adds scalar `rope_options.shift_y` and `shift_x`;
3. constructs one two-dimensional `img_ids` grid with `torch.linspace`;
4. repeats that same grid over the batch with:

```python
repeat(img_ids, "h w c -> b (h w) c", b=bs)
```

`Flux._forward` invokes `process_img(x, transformer_options=...)` once for the
entire batched generated image tensor. `Flux.forward` accepts `x`, timestep,
context, conditioning/control, and transformer options, but exposes no
per-sample `img_ids` argument.

The current Blueprint adapter supplies:

```python
rope_options = {"shift_y": float(region.y), "shift_x": float(region.x)}
```

That dictionary applies to the complete model call, not individual batch
elements.

## Exact-method probe

The experiment called the exact native `Flux.process_img` method with a dummy
batch shaped `[2,128,32,32]`. No diffusion forward or model weights were
required.

With one scalar crop offset `(y=0,x=24)`, both batch elements received the
bit-identical coordinate range:

```text
batch 0: [0,24] -> [31,55]
batch 1: [0,24] -> [31,55]
```

This proves that naively batching crops would silently give every crop the same
absolute position.

Supplying per-batch tensors:

```python
shift_y = tensor([0,24])
shift_x = tensor([0,32])
```

failed at the native coordinate constructor with:

```text
RuntimeError: linspace only supports 0-dimensional start and end tensors,
but got start with 1 dimension(s) and end with 1 dimension(s).
```

Lists or tuples do not solve the scalar arithmetic and grid-construction
contract. There is no adapter-only option that can provide distinct coordinates
without altering the FLUX coordinate backend or adding an explicit per-batch
ID path.

## Consequence

Batching image tensors themselves would keep crops isolated by the batch
dimension during attention. The blocker is not cross-crop attention; it is the
inability to construct distinct full-canvas RoPE IDs for each batch element at
the current public execution boundary.

Spatially concatenating crops would not be equivalent because their image
tokens would attend to each other and the spatial geometry would change. It was
not attempted.

Because the batch-size-2 correctness precondition fails, batch size 4 and the
performance comparison were not run. Timing an incorrectly coordinated batch
would not answer the requested question.

Machine-readable evidence, including the native source hash, method source,
forward signature, coordinate endpoints, and exact exception, is in
`flux2_candidate3_crop_batching_results/report.json`.

## Verdict

CROP BATCHING REQUIRES ADAPTER/BACKEND CHANGE
