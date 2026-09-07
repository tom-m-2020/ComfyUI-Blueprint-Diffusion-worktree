# Z-Image Terminal Refine port audit

Date: 2026-09-07

Scope: source/metadata audit only. Diffusion, VAE, and text-encoder inference
calls: zero. Production changes: zero.

## Verdict

The qualified Terminal Refine algorithm is mechanically expressible for native
latent-space Z-Image-Turbo, but it is **not** an adapter-only port and is not yet
semantically qualified. The region planner and transfer/assembly mathematics
can be retained. The current implementations cannot be called unchanged because
they hard-code Klein's 128 channels, Klein pixel/profile tables, exact Klein
schedule, and Klein model validation.

The smallest safe next step is one experiment-only Z-Image-Turbo discriminator
using the ordinary native model, native schedule, native-origin local RoPE, and
one scheduler-member late refinement sigma. Keep the Klein public node frozen.

## Evidence boundary

The inspected ComfyUI checkout is commit
`5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c` (2026-08-19). Relevant native
sources are `comfy/supported_models.py`, `comfy/model_base.py`,
`comfy/model_sampling.py`, `comfy/latent_formats.py`, and
`comfy/ldm/lumina/model.py`.

The locally available native checkpoint is the FP8 E4M3 quantization of
`Tongyi-MAI/Z-Image-Turbo`. Safetensors-header inspection, without tensor
execution, establishes:

- `x_embedder.weight=[3840,64]`: patch size 2 over 16 latent channels;
- final projection `[64,3840]`: 16 output channels over 2x2 patches;
- 30 joint layers, two noise refiners, and two context refiners;
- hidden width 3840, caption projection input 2560, and learned caption/image
  padding tokens.

ComfyUI detection selects 30 heads, 30 KV heads, axes dimensions `[32,48,48]`,
RoPE theta 256, time scale 1000, and token padding to a multiple of 32.
`supported_models.ZImage` retains the Flux latent format: 16 channels, VAE
spatial downscale 8, scale factor 0.3611, and shift factor 0.1159. The model sees
the processed latent; Blueprint transfer remains inside that model-latent
domain, as it does for Klein.

The official Turbo scheduler is flow-match Euler with fixed shift 3.0 and no
dynamic shifting. Current ComfyUI represents this as
`ModelSamplingDiscreteFlow + CONST`, multiplier 1, shift 3.0, noise scale 1.
Thus the handoff construction remains exactly:

```text
W_sigma = model_sampling.noise_scaling(sigma, epsilon, anchor, False)
        = sigma * epsilon + (1 - sigma) * anchor
```

The equation is shared with Klein CONST, but sigma ownership is not. Klein uses
`ModelSamplingFlux` and its separately qualified schedule. Z-Image-Turbo uses a
fixed-shift discrete-flow schedule. ComfyUI's official Turbo workflow currently
uses 8 steps, `res_multistep`, the `simple` scheduler, CFG 1, and a shift-3 model
sampling patch. The resulting current simple sigma list is:

```text
[1.0, 0.9545454383, 0.8999999762, 0.8333333135,
 0.75, 0.6428571343, 0.5, 0.3000000119, 0.0]
```

Do not reuse Klein's four-step sigma list or assume numeric sigma 0.25 is a
qualified Z-Image refinement point. The first discriminator should use the
existing `0.3000000119 -> 0` terminal interval member.

## Useful canvas geometry

Z-Image's VAE downscale 8 and DiT patch size 2 imply model-token dimensions of
`pixel / 16`. Pixel axes must therefore be divisible by 16; latent axes must be
divisible by 2. Token padding to multiples of 32 permits other token counts but
does not remove the patch-divisibility requirement.

The official Turbo application exposes 1024-class and 1280-class aspect-ratio
profiles, including square, portrait, wide, and ultrawide examples. The safest
first bounded model canvas is 1024x1024 pixels = 128x128 latent = 64x64 image
tokens. A 1280-square canvas is also source-exposed but costs 80x80 image
tokens. A 512px/64-latent canvas is mechanically valid; there is no source or
local evidence here that it retains sufficient whole-scene authority, so it is
not a qualified Blueprint size.

For the first discriminator use both `G=128x128` latent and `W=128x128` latent.
This makes each model call native 1024-square geometry. Use an experiment-only
`H=256x256`, `F=64x64`, stride `48x48`: a 2048px destination, 512px footprint,
128px overlap, and 25 row-major W calls. This keeps G/W fixed while making H
larger than either model canvas. It is a hypothesis configuration, not a
production profile.

## RoPE and isolated local calls

`NextDiT.pos_ids_x` constructs three-axis IDs. Caption position occupies axis
0; image axis 0 is the caption-derived stream offset, while image Y/X are
zero-origin patch-grid coordinates. Optional `transformer_options.rope_options`
can scale and shift Y/X, but ordinary native calls omit it. `axes_lens` is not a
lookup-table bound in the current ComfyUI path; RoPE is computed from numeric
IDs. The low theta 256 makes coordinate-policy changes material.

For baseline portability:

- G must use ordinary zero-origin native coordinates for its own bounded canvas;
- each W must also use ordinary zero-origin native coordinates after F-to-W
  enlargement;
- reject incoming `rope_options`, positional patches, or wrappers;
- do not inject destination offsets or footprint compression into W RoPE.

An isolated W call is mechanically valid because it is an ordinary supported
Z-Image T2I canvas and contains no reference/control branch. It is **not yet
semantically qualified** as refinement of a mapped scene region. That is the
central inference gate. Destination-registered RoPE would be a different
positional intervention, not a compatibility requirement, and must not be
smuggled into the first port.

## Guider and excluded conditioning

Z-Image-Turbo's official inference uses guidance scale 0 / Comfy CFG 1. The
minimal port should require ComfyUI `BasicGuider`, CFG exactly 1, and exactly
one positive conditioning branch. This avoids a second model call and matches
the effective positive-only Turbo path. Base Z-Image or other artifacts that
require dual CFG are outside this first adapter.

Fail closed on masks, areas, hooks, GLIGEN, ControlNet/control residuals,
reference latents, reference text contexts, SigLIP/clip-vision inputs, Omni/edit
conditioning, concat/inpaint latents, model/attention patches, diffusion-model
wrappers, context wrappers, and LoRA-modified profiles. These inputs would
change the baseline from ordinary T2I Terminal Refine and some add extra image
streams and positional offsets.

## Reuse audit

| Component | Reuse result |
|---|---|
| `Region` and deterministic row-major planner rule | Reuse unchanged after applying a Z-specific qualified profile. |
| Halo-aware G-to-F coordinate equations | Reuse unchanged mathematically; parameterize expected channels. |
| Nearest F-to-W enlargement | Reuse unchanged mathematically. |
| Area-mean W-to-F restriction | Reuse unchanged mathematically. |
| Normalized streaming overlap weights/order | Reuse unchanged mathematically; assembler shape check must use 16 channels. |
| Deterministic region seed derivation | Reuse unchanged. Generate `[1,16,W_h,W_w]` noise. |
| Accepted G/transient W/no persistent H ownership | Reuse unchanged. |
| Normal guider preparation, cancellation, cleanup | Reuse the same lifecycle principle. |
| Current configurable geometry implementation | Not callable unchanged: hard-coded 128 channels and Klein bounds/profiles. |
| Current initializer/noise/mapper/restrictor/assembler | Not callable unchanged for the same channel-shape reason. |
| Klein schedule and sigma validation | Do not reuse. |
| Frozen-node exact-oracle dispatch | Do not reuse for Z-Image. |

No generalized production abstraction is justified by this audit. An
experiment may use channel-parameterized copies or narrow pure helpers. Only a
passing experiment can justify extracting shared utilities later.

## Smallest `ZImageAdapter` interface

The adapter need not own geometry, sampling, noise, or assembly. It needs the
same two operations as the terminal adapter:

```python
class ZImageAdapter:
    latent_channels = 16
    latent_downscale = 8
    patch_size = 2

    def validate_prepared(
        self, *, guider, model_options, destination,
        model_sampling, destination_hw
    ) -> None: ...

    def predict_native(
        self, *, guider, value, sigma, expected_hw,
        model_options, seed
    ) -> Tensor: ...
```

`validate_prepared` must prove native `comfy.model_base.Lumina2` and native
`comfy.ldm.lumina.model.NextDiT`; batch one `[1,16,H,W]`; patch size 2;
in/out 16; dim 3840; 30 heads; 30 layers; two context/noise refiners;
`axes_dims=[32,48,48]`; theta 256; time scale 1000; padding multiple 32;
native latent-space rather than `ZImagePixelSpace`; fixed-shift
`ModelSamplingDiscreteFlow+CONST`, shift 3, multiplier 1, noise scale 1; and the
positive-only conditioning contract above.

`predict_native` validates `[1,16,*expected_hw]`, even latent axes and finite
sigma, copies model options, rejects rather than removes caller `rope_options`,
calls the prepared guider normally, and validates same-shape finite output.
It must not synthesize conditioning or call `diffusion_model` directly.

## Fail-closed qualification gates

Before any production proposal require:

1. Source/fake-adapter tests for 16-channel initialization, halo mapping,
   restriction, overlap coverage, deterministic seeds, row-major order,
   mutation detection, cancellation, and retry.
2. Exact live model/profile validation against the native checkpoint and
   current ComfyUI revision; reject pixel-space, Nunchaku, GGUF, altered shift,
   LoRA/control/reference/edit paths, CFG other than 1, and unknown Z variants.
3. Confirm the supplied G schedule exactly equals the native 8-step simple
   shift-3 list and the local handoff uses its `0.3000000119` member.
4. Verify G and every W are 128x128 latent, all model calls report that bounded
   geometry, and there are zero H-sized calls.
5. Prove anchor/noise/working hashes, complete coverage, deterministic repeat,
   constant completed-region residency, and normal cleanup/cancellation.
6. Semantic pass: exact S3 object count and placement, continuous horizon,
   no prompt-complete miniature tiles, no new cross-footprint disagreement, and
   credible local structure beyond plain terminal-G resizing.
7. Compare overlap RMS, low-frequency difference, edge/detail measures, wall
   time, call/token counts, and peak allocated/reserved memory. Metrics cannot
   override the semantic gate.

## One minimal first inference discriminator

Use one S3 prompt and seed for three arms; this is one fixed discriminator, not
a sweep:

1. native 1024-square Z-Image-Turbo generation as a model/schedule sanity
   reference;
2. terminal denoised `G=1024` resized to `H=2048`, with no local diffusion;
3. the identical terminal G passed through 25 overlapping F64/stride48 regions,
   enlarged to native W128, initialized once per region with deterministic
   regional noise through `noise_scaling(0.3000000119, ...)`, given one native
   W prediction, area-restricted, and normalized into H256.

G uses the official current 8-step simple schedule and `res_multistep` sampler;
the experiment captures its terminal denoised estimate rather than recreating
an Euler-only Klein schedule. The local `[0.3000000119,0]` interval terminates at
the model's denoised prediction, so no second local sampler policy is introduced.

Advance only if arm 3 preserves S3 and beats arm 2 in credible structure while
remaining deterministic and bounded. Failure means the direct Terminal Refine
port is not qualified for Z-Image-Turbo. Do not respond by changing RoPE,
Blueprint size, sigma, sampler, overlap, or CFG in the same task.
