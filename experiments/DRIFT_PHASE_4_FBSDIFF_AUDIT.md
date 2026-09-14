# Phase 4 - FBSDiff audit and stock-Klein mapping

## Audit result before inference

FBSDiff does **not** transform hidden U-Net activations, attention tensors, or
transformer-block features. Its term *diffusion feature* denotes the spatial
sampler latent `z_t`. The paper explicitly presents the method as requiring no
access to internal denoiser features, and the official sampler applies DCT to
the current reference and generation latent states outside `p_sample_ddim`.

Therefore there is no paper-defined layer ID or before/after-attention
placement. Any Klein hidden-state transfer is an extrapolation and must not be
described as direct FBSDiff.

## Exact paper/source mechanism

FBSDiff uses three trajectories:

1. A 1000-step DDIM inversion maps encoded source latent `z_0` to an inverted
   endpoint `z_Tinv` using empty conditioning.
2. A 50/100-step reference reconstruction trajectory starts from that inverted
   endpoint and advances with empty conditioning and guidance 1.
3. A target-prompt sampling trajectory starts from independent Gaussian noise.

During the early calibration phase, reference and target states at the same
timestep are transformed channel-wise with orthonormal 2D DCT. A binary mask
substitutes selected reference coefficients into the target spectrum, followed
by 2D IDCT. The calibrated target state is then passed to its next denoising
step. The official code performs this state substitution before each paired
reference/target `p_sample_ddim` call. Later steps run without substitution to
release the model from the reference constraint.

For a DCT coordinate `(u,v)`:

- low-FBS: `u+v <= th_lp`; appearance/color/luminance plus layout;
- mid-FBS: `th_mp1 < u+v <= th_mp2`; layout while excluding low appearance and
  high contour bands;
- high-FBS: `u+v > th_hp`; contours with low-frequency appearance left free.

Paper defaults on SD 1.5's `4 x 64 x 64` latent are `th_lp=80`, `th_hp=5`, and
`th_mp1=5, th_mp2=80`; the released inference example uses high-FBS threshold
3 and low-FBS threshold 90. The paper uses roughly the first 45-55% of sampling
as calibration. It reports that one-time substitution becomes incoherent and
full-spectrum substitution suppresses editability.

## Backbone dependency

The published operation is not U-Net-specific: it acts on `[B,C,H,W]` sampler
latents before the denoiser, so its spatial axes exist independently of the
backbone. Stable Diffusion 1.5, DDIM inversion, epsilon prediction, 4-channel
latents, 64x64 grids, 1000 inversion steps, 50/100 reverse steps, and CFG 7.5
are nevertheless material experimental assumptions.

Those assumptions do not transfer directly to Klein:

- Klein is a CONST/flow model, not the paper's DDIM epsilon model.
- The fixed Phase 2 experiment has six Euler evaluations, not 50/100 DDIM
  steps, and no qualified inversion trajectory.
- Klein latents are `[B,128,32,32]`; absolute DCT thresholds have different
  geometric/band fractions.
- Klein patchifies the latent into 1024 tokens and projects each token to 3072
  learned hidden channels; hidden-channel spectra are not equivalent to VAE
  latent-channel spectra.
- Attention globally mixes token content after the input projection, so later
  hidden tokens remain spatially indexed but no longer local feature maps in
  the convolutional sense.
- Empty-conditioning reconstruction under CONST is a controlled source branch,
  not DDIM inversion/reconstruction equivalence.

## Klein forward-path audit

For the active checkpoint and 512x512 decode target, native ComfyUI receives a
`[1,128,32,32]` latent state. `Flux.process_img` patchifies with patch size 1 in
row-major order, producing 1024 generated-image tokens and explicit `(t,y,x)`
IDs. `Flux.forward_orig` then applies `img_in`, yielding
`[1,1024,3072]`, invokes the public `post_input` patch list, constructs RoPE,
and enters double block 0. Twenty-five transformer blocks subsequently mix
image and text globally before the final projection returns the latent-shaped
velocity/denoised result.

The smallest defensible **internal** spatial target is thus the generated-image
token tensor at `post_input/pre_double_block_0`:

- every token maps exactly to one 32x32 latent location;
- source and generation features have identical shapes and coordinate order;
- substitution occurs before attention, avoiding an arbitrary block-depth
  choice;
- model weights and ordinary block code remain unchanged.

The sampler latent itself is more faithful to FBSDiff but is not internal. A
later block was rejected because the paper provides no layer-selection basis
and global mixing weakens the interpretation of a per-channel 2D frequency.

## Preregistered minimal falsifier

The single experimental arm uses the input-projected `[1,1024,3072]` tokens,
reshaped to `[1,3072,32,32]`. It substitutes the reference high band
`u+v>3` during evaluations 0, 1, and 2, then releases evaluations 3-5. The
threshold matches the official high-FBS example because the stated question is
fine contours/identity with appearance divergence; it is not claimed to be
resolution-normalized or paper-equivalent.

The reference trajectory uses the same ordinary Gaussian endpoint, native
source-img2img initialization, empty conditioning, CFG 1, Euler rule, and sigma
schedule. It is precomputed at identical sigma ordinals. The generation branch
uses the target prompt. No sampler state is directly corrected; only the one
preregistered hidden representation is changed immediately before double block
0 during calibration evaluations.

Controls are Gaussian and Phase 1b FSS `r=2`. No FSS or ILVR is combined with
the feature arm. The portrait remains decisive.

## Runtime result

**FBSDIFF-LIKE FEATURE TRANSFER FALSIFIED**

All nine requested output trajectories plus three empty-conditioning reference
trajectories completed finite. The DCT preflight measured `3.00e-7`
orthonormal-identity RMS error and `2.43e-6` roundtrip RMS error. The live feature
shape was exactly `[1,1024,3072]` at `post_input/pre_double_block_0` for every
capture.

| Case / arm | coarse RGB RMS | Chamfer px | edge F1 | appearance RGB RMS | phase rad | gradient diff |
|---|---:|---:|---:|---:|---:|---:|
| Portrait Gaussian | 0.5550 | 7.043 | 0.600 | 0.5589 | 1.759 | 1.436 |
| Portrait FSS r=2 | **0.1832** | **6.350** | **0.760** | 0.1927 | **1.642** | **1.263** |
| Portrait feature band | 0.5374 | 7.693 | 0.619 | 0.5417 | 1.754 | 1.432 |
| Astronaut Gaussian | 0.4881 | 7.189 | 0.712 | 0.4979 | **1.719** | **1.583** |
| Astronaut FSS r=2 | **0.2731** | **4.013** | **0.803** | 0.2896 | **1.634** | **1.511** |
| Astronaut feature band | 0.4794 | 4.941 | 0.732 | 0.4898 | 1.757 | 1.613 |
| Bridge Gaussian | 0.3444 | 7.659 | 0.588 | 0.3607 | 1.740 | 1.474 |
| Bridge FSS r=2 | **0.1598** | **3.502** | **0.742** | 0.1802 | **1.635** | **1.342** |
| Bridge feature band | 0.3140 | 5.736 | 0.603 | 0.3264 | 1.743 | 1.490 |

Lower is better except edge F1. The feature arm remains much closer to Gaussian
than FSS in coarse, phase, and gradient metrics. It improves astronaut/bridge
Chamfer and F1 somewhat over Gaussian but does not approach FSS, and portrait
Chamfer becomes worse than Gaussian.

## Feature-energy instrumentation

The mask substitutes 1014 of 1024 DCT coordinates (`99.023%`) independently in
every one of 3072 channels for evaluations 0-2. Because source and generation
start from the same source-img2img state and diverge only through conditioning,
the first transfer is effectively zero and subsequent differences remain small:

| Case | mean source high-band RMS | mean generated high-band RMS | mean substituted RMS | substituted/generated RMS | energy ratio |
|---|---:|---:|---:|---:|---:|
| Portrait | 0.4894 | 0.4905 | 0.0154 | 3.16% | 0.17% |
| Astronaut | 0.4904 | 0.4914 | 0.0192 | 3.94% | 0.27% |
| Bridge | 0.4917 | 0.4910 | 0.0149 | 3.06% | 0.15% |

Thus the failure is not a simple large-energy overwrite. A small difference in
the input-projected high-band hidden features redirects later global attention
and produces large semantic changes. Conversely, replacing nearly all DCT
positions does not imply preserving source contours because most coefficients
are already similar and the learned hidden channels do not carry the paper's
latent-frequency factorization.

## Visual discriminator

- Portrait: the arm produces a bronze/statue appearance, but depicts an older,
  different person with crossed arms, different body framing, and shifted
  placement. It is no identity improvement over Gaussian or FSS.
- Astronaut: the arm exposes a different face and changes suit silhouette,
  proportions, and stance. It is less source-faithful than FSS.
- Bridge: the arm generates a different tower/cable/deck arrangement and
  perspective. It does not preserve the source bridge contours.

The portrait gate fails decisively while appearance change remains substantial,
so the arm is falsified rather than overconstrained.

## Interpretation and boundary

This result rejects one fixed hidden-token transfer only:
`post_input/pre_double_block_0`, `u+v>3`, evaluations 0-2. It does not falsify
published FBSDiff's sampler-latent/DDIM mechanism. The audit shows why: FBSDiff
does not define an internal layer, whereas Klein's learned 3072-channel token
projection is followed by global text-image transformer mixing.

No alternative layer, threshold, band, calibration duration, reference prompt,
or inversion scheme is authorized or tested. No production code changed.

Artifacts:

- `experiments/DRIFT_PHASE_4_PREREGISTRATION.json`
- `experiments/drift_fbsdiff_like_klein.py`
- `experiments/drift_phase_4_fbsdiff_results/telemetry.json`
- final images, comparison sheets, and selected x0 previews under
  `experiments/drift_phase_4_fbsdiff_results/`
