# Phase 34 — Orthogonal coarse/detail stochastic-state contract

## Executive result

**A — COARSE/DETAIL CONTRACT IS MATHEMATICALLY COHERENT; PROCEED TO ONE FIXED MODEL-MEDIATED DISCRIMINATOR**

An orthonormal `2x2` Haar basis gives a legitimate, full-rank stochastic
coordinate system for a native `64x64` working tensor. It fixes the specific
Phase-32 error: the three detail coordinates are independent Gaussian
coordinates accompanied by an independent coarse coordinate, not a
rank-deficient pixel-space residual presented as ordinary noise.

This does not itself create useful image detail. The current `32x32`
destination state cannot contain the three additional bands. A viable
coarse/detail architecture must persist four `32x32` coefficient fields per
region (or their globally assembled equivalents) and reconstruct a `64x64`
output. If the output remains one `32x32` tensor, the proposal is
information-capacity contradictory.

No diffusion model was loaded or run. Production and ComfyUI core are
unchanged.

## 1. Exact orthonormal transform

For a spatial block `[a b; c d]`, define

```text
C  = ( a + b + c + d) / 2
F1 = ( a + b - c - d) / 2   # vertical variation
F2 = ( a - b + c - d) / 2   # horizontal variation
F3 = ( a - b - c + d) / 2   # diagonal variation
```

The transform matrix is the `4x4` Hadamard matrix divided by `2`; therefore
`T T^T=I`. Its inverse is

```text
a = (C + F1 + F2 + F3) / 2
b = (C + F1 - F2 - F3) / 2
c = (C - F1 + F2 - F3) / 2
d = (C - F1 - F2 + F3) / 2
```

CPU validation over `[2,5,64,64]` tensors measured:

| Check | float32 max abs | float64 max abs |
|---|---:|---:|
| `T^-1(T(W))-W` | `2.3842e-7` | `6.6613e-16` |
| coefficient round trip | `2.3842e-7` | `8.8818e-16` |

Declared tolerances are `1e-5` for float32 and `1e-12` for float64.

Arithmetic mean pooling is not the orthonormal LL coefficient:

```text
avgpool2(W) = (a+b+c+d)/4 = C/2
C = 2 * avgpool2(W)
```

This identity was exact in both tested dtypes.

## 2. Gaussian law

For a block vector `epsilon~N(0,I_4)`, orthogonality gives

```text
T epsilon ~ N(0, T I T^T) = N(0,I_4).
```

Thus `C,F1,F2,F3` are individually standard normal and mutually independent.
Applying disjoint orthogonal blocks preserves independence across spatial
blocks, channels, and batches under the original i.i.d. law. An empirical
250,000-block sanity check found variances `0.99787..0.99969` and maximum
absolute covariance error from identity `0.00377`; sampling is only a sanity
check, while orthogonal Gaussian invariance is the proof.

Phase 32 instead used `(I-P)epsilon`. Its covariance `I-P` has rank three.
Haar detail coordinates alone are also only three-dimensional. Haar avoids
that defect only by retaining the independent LL coordinate together with all
three detail coordinates. Inverse-transforming `(0,F1,F2,F3)` would recreate
the rejected singular law.

## 3. State and sampler semantics

A complete accepted local state is

```text
Z_t = (C_t,F1_t,F2_t,F3_t),    W_t = T^-1(Z_t).
```

All components share `sigma_t`; `Z_t` is accepted state and `W_t` is a
transient model input. For ordinary prediction `x0_W`, let
`(x0_C,x0_F1,x0_F2,x0_F3)=T(x0_W)`. With
`Delta=sigma_next-sigma_t`, CONST-flow Euler is

```text
C_next  = C_t  + Delta*(C_t  - x0_C )/sigma_t
Fj_next = Fj_t + Delta*(Fj_t - x0_Fj)/sigma_t, j=1..3.
```

Linearity gives

```text
T(W_t + Delta*(W_t-x0_W)/sigma_t)
 = Z_t + Delta*(Z_t-T(x0_W))/sigma_t.
```

Therefore coordinate-space and W-space Euler are algebraically identical when
no component is constrained. Measured maximum error was `3.5763e-7` float32
and `8.8818e-16` float64. Atomic acceptance must replace all four bands.

## 4. Blueprint ownership and amplitude

Let `b` be the existing mapped `32x32` Blueprint crop. Exact preservation of
its current nearest-lift amplitude requires

```text
C_blueprint = 2*b.
```

Then

```text
T^-1(2*b,0,0,0) = nearest2(b)
avgpool2(T^-1(2*b,0,0,0)) = b.
```

Both identities validated exactly. Assigning `C=b` would halve the
reconstructed Blueprint amplitude.

## 5. Detail initialization and constrained-W law

At flow sigma `s`, the unique construction preserving the qualified nearest
anchor plus i.i.d. noise law is

```text
epsilon_C, epsilon_F1, epsilon_F2, epsilon_F3 iid ~ N(0,I)
C_s  = (1-s)*(2*b) + s*epsilon_C
Fj_s = s*epsilon_Fj
W_s  = T^-1(C_s,F1_s,F2_s,F3_s).
```

Equivalently,

```text
W_s = (1-s)*nearest2(b) + s*epsilon_W,
epsilon_W = T^-1(epsilon_C,epsilon_F) ~ N(0,I).
```

Validation measured equivalence within `4.7684e-7` float32 and
`8.8818e-16` float64. Each unscaled detail marginal is standard normal; each
detail state has covariance `s^2 I` and zero mean under the zero-detail signal
assumption.

Conditioned on fixed `b`, `W_s` is

```text
N((1-s)*nearest2(b), s^2 I),
```

not `N(0,I)`. This is the current conditional re-noising law around a fixed
denoised anchor, but the pretrained model's fitness for that constrained
anchor remains an empirical assumption. Fixing `C_s=2*b` and sampling only
details would remove coarse noise, create singular covariance, and violate the
ordinary noise-scaling contract; that construction is rejected.

## 6. Model prediction and Blueprint-authority policies

Three policies are distinct:

1. **Unconstrained:** update all predicted bands ordinarily. This is exactly
   W-space Euler but does not guarantee Blueprint authority.
2. **Exact coarse prediction authority:** use
   `x0_Z*=(2*b_x0,x0_F1,x0_F2,x0_F3)`. This is parameter-free and retains the
   model's three predicted detail bands.
3. **Exact accepted-state synchronization:** after an ordinary proposal,
   replace `C_next` with a same-sigma Blueprint-owned accepted coarse state and
   retain proposed `F_next`. This is parameter-free but requires synchronized
   Blueprint state at every sigma.

Soft blending, guidance strength, sigma thresholds, and selective schedules
introduce arbitrary policies and are not qualified. Phase 35 is fixed to
policy 2 at one terminal evaluation; it does not choose between policies 2 and
3 for a trajectory.

## 7. Terminal output and persistent detail capacity

The current path stores only `avgpool2(x0_W)=x0_C/2`; all three detail bands
are discarded. Invertibility cannot recover unretained data.

For detail to survive, overlapping predictions must be assembled into four
global fields

```text
global C, F1, F2, F3: [B,channels,H,W]
```

using identical region weights. Inverse Haar then produces
`[B,channels,2H,2W]`. A persistent trajectory would accept all four fields
atomically. Thus a current `128x256` coarse destination implies a
detail-preserving `256x512` latent output. If final output must remain
`128x256`, the extra degrees of freedom cannot exist in that tensor and the
proposal is contradictory before inference.

## 8. Relationship to current geometry

| Current concept | Coarse/detail interpretation |
|---|---|
| terminal Blueprint `x0` | retained; supplies Blueprint-owned `b` |
| mapped destination Blueprint | retained as coarse-grid signal |
| `32x32` destination region | region in each of four coefficient fields |
| `64x64` working canvas | retained as transient native model input |
| terminal region noise | retained as four independent Haar coordinates |
| ordinary native FLUX prediction | retained unchanged, then transformed |
| `avgpool2` restriction | no longer terminal output; it is `x0_C/2` |
| overlap assembly | applied independently to all four bands |
| one coarse destination latent | replaced by four bands plus inverse-Haar output |

No model block or conditioning semantic changes in the first discriminator;
the architectural change is output/state ownership.

## 9. Compute and memory

For batch one, 128 channels, one region:

| Tensor(s) | Elements | fp16 | fp32 |
|---|---:|---:|---:|
| `C`, `32x32` | 131,072 | 0.25 MiB | 0.50 MiB |
| four bands | 524,288 | 1.00 MiB | 2.00 MiB |
| reconstructed `W`, `64x64` | 524,288 | 1.00 MiB | 2.00 MiB |
| bands plus `W` | 1,048,576 | 2.00 MiB | 4.00 MiB |

Four bands are a coordinate change, not compression. They use `4x` the
storage of one coarse field. Four global `[1,128,128,256]` fields require 32
MiB fp16 or 64 MiB fp32, excluding accumulators and model scratch. The
inverse-Haar `[1,128,256,512]` tensor has the same element count. The transform
is cheap arithmetic, but model token count and forward cost do not change; no
compute or overall VRAM saving is claimed.

## 10. Fixed Phase-35 discriminator

Run one terminal, research-only test on the smallest qualified square scene
(`128x128` coarse destination, 25 regions):

1. reuse persisted terminal Blueprint and qualified nearest/noise `W_0.25`;
2. run one ordinary native `64x64` prediction per region;
3. transform each prediction to four Haar bands;
4. set `x0_C*=2*b` for its mapped Blueprint crop, with no blend;
5. retain all three model-predicted detail bands unchanged;
6. overlap-assemble all four bands with identical weights;
7. inverse Haar once to a `256x256` latent output;
8. verify exact `avgpool2(output)=mapped Blueprint` and preservation of all
   assembled detail bands;
9. compare against a coarse-only inverse-Haar Blueprint and an unconstrained
   full-band assembly made from the same predictions.

The sole empirical question is whether predicted detail bands add credible
native-resolution structure without incompatible local alternatives. No
sigma, strength, transform, pass, or coupling sweep is authorized.

## Unresolved assumptions

- Predicted detail bands may encode aliasing or incompatible local geometry.
- Overlapping detail predictions may disagree despite shared coarse authority.
- Exact coarse replacement may preserve S3 yet remain soft after decoding.
- Two parameter-free trajectory authority policies remain; Phase 35 qualifies
  neither as a full trajectory.
- Doubled latent output geometry is not production-qualified.

## Validation artifacts

- `experiments/flux2_orthogonal_coarse_detail_contract.py`
- `experiments/flux2_orthogonal_coarse_detail_contract_results/report.json`

The validator is CPU-only, deterministic, loads no diffusion model, and
generates no images.
