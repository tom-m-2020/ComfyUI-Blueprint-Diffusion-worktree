# NeuralRemaster / Phase-Preserving Diffusion audit

Date: 2026-09-14  
Scope: source-and-math audit plus one inference-only stock FLUX.2 Klein 4B
falsification. No node, backend patch, production architecture change, ILVR, or
FBSDiff.

## Verdict

**INFERENCE-ONLY FSS PARTIAL / UNCERTAIN**

The fixed stock-Klein probe did not falsify all useful inference-only effects.
The smallest-cutoff FSS arm (`r=2`) retained the source's object count and broad
layout while producing substantially more appearance change than full phase
preservation. Increasing `r` monotonically increased rigidity, but `r=8`,
`r=16`, and full phase preservation mostly reproduced the synthetic source's
flat geometry and color blocks rather than performing a rich remaster. One
synthetic source, one prompt, and one near-full-noise img2img schedule are not
enough to call the method generally promising. The trained paper result remains
a different contract.

## Primary sources and provenance

- Paper: `docs/papers/2512.05106v3_neuralremaster.pdf` and its supplied HTML.
- Authors' project page: <https://yuzeng-at-tri.github.io/ppd-page/>.
- Authors' model/inference repository:
  <https://github.com/zengxianyu/PPD-examples>, inspected at
  `ec698176245ca7b4848f53c6acebf8b1e3d66721`.
- Authors' noise repository:
  <https://github.com/zengxianyu/structured-noise>, inspected at
  `c0d8ea92862b427c1443210782f2fd454e2444e6`.
- Native ComfyUI source:
  `C:/Users/Tom-M/data/a/ai/apps/ComfyUI-dev`, inspected at
  `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`.

The attached `C:/Users/Tom-M/Downloads/untitled.md` was not read because the
repository operating rules explicitly prohibit access to Downloads. It was not
treated as an instruction source.

## Exact construction

For source latent `z`, let

```text
F_z = A_z exp(i phi_z)
epsilon ~ N(0, I)
F_epsilon = A_epsilon exp(i phi_epsilon)
```

Full phase-preserving noise is

```text
F_epsilon_hat = A_epsilon exp(i phi_z)
epsilon_hat = Re(IFFT(F_epsilon_hat)).
```

The source supplies phase and an ordinary Gaussian sample supplies every
Fourier magnitude. With real inputs, conjugate symmetry makes the inverse
effectively real. Parseval gives the same total squared magnitude and RMS as the
Gaussian, up to floating-point error.

FSS defines centered frequency radius `rho=sqrt(u^2+v^2)` and

```text
M(rho) = 1                                      when rho <= r
         exp(-(rho-r)^2 / (2 transition^2))     otherwise

phi_mix = M phi_z + (1-M) phi_epsilon
F_epsilon_hat = A_epsilon exp(i phi_mix).
```

The paper uses transition bandwidth `2`. Larger `r` preserves source phase over
more frequencies and increases rigidity. Radius units are frequency-grid
pixels. Here they are latent-grid pixels on a `32x32` grid; `r={2,8,16}` spans
weak, intermediate, and strong preservation without a tuning sweep.

### Paper/code discrepancy

The authors' released helper literally blends principal angles, performs an
inverse FFT, and takes the real part. It also reflection-pads by default, clips
the 95th percentile of both `image_phases` and noise magnitudes, and replaces
spatial values outside `[-5,5]` with the Gaussian sample. Those extra operations
are not in equations 5--10 and mean the helper does not strictly retain the
sampled Gaussian magnitude.

This harness uses the narrower equation-level FFT construction, no padding,
clipping, or outlier replacement, and the same real projection used by the
authors. Linear principal-angle mixing can introduce a small conjugate-symmetry
error near phase wrapping. It is measured; it was not silently renormalized or
replaced with circular interpolation.

## Pixel versus latent space

The method's classical motivation is a pixel-space phase/structure result, but
the paper applies phase preservation directly to VAE latents. It acknowledges
that nonlinear VAE encoding prevents a complete theoretical transfer argument
and supports the choice empirically with latent magnitude/phase swaps. The
released FLUX script VAE-encodes its input before constructing FSS noise. This
probe follows that latent boundary: `[1,128,32,32]` for a `512x512` image.

## DDPM versus flow matching

For DDPM, ordinary corruption

```text
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
```

uses `epsilon_hat` instead. The epsilon-prediction loss keeps its algebraic
form, but its expectation and target change to source-dependent, non-Gaussian
noise. The paper notes that this violates standard DDPM assumptions and
fine-tunes its working checkpoints under the changed corruption distribution.

For rectified flow, the paper uses

```text
x_t = t epsilon_hat + (1-t) I
v_t = epsilon_hat - I
L = E ||u(x_t,t)-v_t||^2.
```

This matches the algebraic shape of ComfyUI Klein CONST:

```text
x_sigma = sigma * noise + (1-sigma) * source_latent
x0_hat = x_sigma - sigma * model_output.
```

The match is parameterization-level only. Stock Klein learned the vector field
for a Gaussian endpoint independent of the source, not the conditional
structured endpoint and velocity used by Phase-Preserving Diffusion training.

## Training boundary

No architecture change or extra parameter is required, but successful paper
models are not inference-only. The authors fine-tune SD 1.5, FLUX-dev, and
Wan2.2 with phase-preserving noise. SD 1.5 is tested with full and LoRA
fine-tuning; FLUX-dev and Wan use LoRA. Training samples cutoff radii, so the
rigidity range is learned. The released FLUX inference loads a trained PPD LoRA.

Only the mechanical construction/substitution is inference-only. On stock
Klein this is an out-of-distribution probe, not a NeuralRemaster reproduction.

## Fixed Klein interpretation

The focused harness is `experiments/drift_phase_preserving_klein.py`. It keeps
native `ModelPatcher`, conditioning preparation, `sample_custom`, the Klein
schedule, CONST scaling, and Euler recurrence. It changes no model internals or
sigma values.

- supplied FLUX.2 Klein 4B W4A8, Qwen3 encoder, and FLUX.2 VAE;
- deterministic synthetic bridge/lighthouse/tower source, `512x512`;
- fixed prompt and seed `2026091401`;
- Euler, CFG `1.0`;
- native 8-step schedule with fixed img2img `start_index=2`, giving
  `sigma_0=0.9569222331` and six evaluations;
- A: Gaussian endpoint; B: full source-phase endpoint;
- C: FSS at `r=2,8,16`, transition bandwidth `2`.

Every arm initializes through native CONST:

```text
x_sigma0 = sigma_0 * endpoint_noise + (1-sigma_0) * source_latent.
```

Only `endpoint_noise` differs. This differs from the released FLUX.1 example,
which uses structured noise as a pure initial endpoint with a trained PPD LoRA.
The small retained source term here is intentional because A is an ordinary
source-img2img control; it is disclosed rather than treated as paper-equivalent.

## Numerical preflight

`preflight.json` was written before loading the diffusion model.

| endpoint | RMS ratio | variance ratio | imaginary RMS | magnitude relative RMS error | phase RMS to source |
|---|---:|---:|---:|---:|---:|
| B full | 0.99999994 | 0.99998885 | 1.72e-7 | 1.35e-7 | 3.42e-7 |
| C r=2 | 0.99999994 | 0.99998885 | 8.55e-8 | 1.32e-7 | 1.7606 |
| C r=8 | 0.99999994 | 0.99998885 | 3.08e-5 | 1.34e-7 | 1.4884 |
| C r=16 | 0.99999988 | 0.99998879 | 3.18e-4 | 2.07e-6 | 0.5222 |

Gaussian RMS was `0.9995350`, variance `0.9990650`; source RMS was
`0.7417406`. Full preservation meets magnitude and phase properties to numerical
precision. FSS phase deviation falls monotonically. The strongest cutoff's
small real-projection discrepancy is reported, not normalized away.

## Runtime result

All five trajectories completed with finite outputs.

| arm | final RMS vs A | source-relative RMS | phase RMS | coarse 4x RMS | gradient-difference RMS |
|---|---:|---:|---:|---:|---:|
| A Gaussian | 0.000 | 1.09 | 1.76 | 0.62 | 1.18 |
| B full | 1.38 | 0.65 | 0.58 | 0.27 | 0.77 |
| C r=2 | 1.14 | 0.70 | 1.58 | 0.38 | 0.77 |
| C r=8 | 1.26 | 0.62 | 1.17 | 0.32 | 0.63 |
| C r=16 | 1.37 | 0.63 | 0.61 | 0.26 | 0.72 |

At every evaluation the harness records state and denoised/x0 summaries, RMS
from A, source-relative RMS, Fourier phase deviation, 4x pooled deviation, and
finite-difference gradient deviation. Selected first/middle/final state and x0
decodes are saved for A, B, C-r8, and C-r16; final decodes exist for every arm.

Visual inspection is decisive. A makes the richest night photograph but changes
the simple source geometry. B, C-r16, and mostly C-r8 retain geometry by tracing
the source's flat appearance. C-r2 changes the lighthouse, moon glow, bridge
material, lighting, and water while keeping both end structures and the two-span
composition recognizable. This is measurable but weak one-case evidence.

Artifacts are in `experiments/drift_phase_preserving_results/`, principally
`telemetry.json`, `preflight.json`, `COMPARISON.png`, final PNGs, and selected
intermediate state/x0 PNGs.

## Stop boundary

Do not implement a node or modify Blueprint production architecture. Do not
proceed to ILVR or FBSDiff. If explicitly requested, the smallest next step is
the unchanged harness on a preregistered natural/stylized source set. PPD
fine-tuning is a separate trained-contract investigation.
