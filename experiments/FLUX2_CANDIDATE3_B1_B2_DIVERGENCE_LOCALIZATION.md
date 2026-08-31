# Phase 6c — B=1 versus B=2 divergence localization

## Scope and control

This zero-update probe used the same 32x32 latent crop at absolute offset
`(y=0,x=0)` from the Phase-6b same-crop/same-coordinate control. The FLUX.2
Klein 4B W4A8 checkpoint, prompt, seed `20260831`, sigma `1.0`, conditioning,
model options, and scalar RoPE offset were identical. The B=2 input was an
exact duplication of the B=1 latent, sigma, context, and coordinate grid.

Instrumentation was runtime-only. It wrapped the loaded instance's
`model_sampling.calculate_input` and `Flux.process_img`, attached temporary
forward hooks, and restored every method/hook after the two predictions. No
ComfyUI core file or Blueprint production file was changed.

Full machine-readable telemetry is in
`flux2_candidate3_b1_b2_divergence_results/report.json`.

## Localization result

Every boundary through the input of `img_in` is bit-exact. The first differing
tensor is the output of `Flux.img_in`, the image-token input embedding linear:

| Boundary | B=1 shape | dtype | B2[0] max / RMS | B2[1] max / RMS | Bit exact |
|---|---:|---|---:|---:|---|
| `calculate_input.output` | 1x128x32x32 | fp32 | 0 / 0 | 0 / 0 | yes / yes |
| `process_img.tokens` | 1x1024x128 | bf16 | 0 / 0 | 0 / 0 | yes / yes |
| `process_img.img_ids` | 1x1024x4 | fp32 | 0 / 0 | 0 / 0 | yes / yes |
| `img_in.input` | 1x1024x128 | bf16 | 0 / 0 | 0 / 0 | yes / yes |
| **`img_in.output`** | **1x1024x3072** | **bf16** | **0.00390625 / 3.4613e-6** | **0.00390625 / 3.4613e-6** | **no / no** |

This is an input/embedding-path divergence, before the first transformer block.
It is not caused by patchification, absolute coordinates, RoPE construction,
text preparation, or attention.

The specific operator is `comfy.ops.Linear` at `Flux.img_in`, with a plain
`torch.nn.Parameter` weight of shape `[3072,128]` and dtype bfloat16. Runtime
inspection found `quant_format=None`, `layout_type=None`, no `input_scale`, no
`pre_quant_scale`, and no forced full-precision matrix multiply. The checkpoint
also stores `img_in.weight` directly as bfloat16 and has no `img_in.comfy_quant`
record. Therefore the first divergence is **not inside a quantized operator**
and there is no input/weight quantization scale at this boundary that can
depend on batch contents or batch dimension. It is batch-shape-sensitive
bfloat16 linear execution in the embedding path.

The timestep embedding also diverged independently despite a bit-exact
timestep input (`max=0.0078125`, `RMS=5.5870e-4`), while text projection and
RoPE output remained exact. This is consistent with batch-shape-sensitive
linear execution, not with cross-element interaction: both duplicated B=2
elements had the same difference from B=1 at every reported boundary.

## Propagation

Instrumentation was stopped after the first block plus the final projection,
because the first divergence was already identified:

| Later boundary | B2 element max / RMS versus B1 | dtype |
|---|---:|---|
| first double-block image QKV output | 1.0 / 0.03478 | bf16 |
| first double-block output, image stream | 2.0 / 0.01974 | bf16 |
| final projection output | 0.418861 / 0.029550 | bf16 |
| guider prediction | 0.418861 / 0.029550 | fp32 |

The tiny sparse embedding discrepancy is amplified through the W4A8 model
path into the previously observed prediction drift. This probe does not claim
which CUDA GEMM implementation detail causes the bfloat16 batch-shape
difference; that would require changing or profiling backend kernels, which is
outside this task.

## Conclusion

Per-batch coordinate construction was not the blocker in this control. Native
B=2 does not preserve the qualified B=1 prediction ensemble because execution
already changes at the unquantized bfloat16 image embedding (and separately in
timestep embedding), before transformer attention. Crop batching therefore
cannot be promoted while sequential-equivalence remains a requirement.

**DIVERGENCE IN INPUT/EMBEDDING PATH**
