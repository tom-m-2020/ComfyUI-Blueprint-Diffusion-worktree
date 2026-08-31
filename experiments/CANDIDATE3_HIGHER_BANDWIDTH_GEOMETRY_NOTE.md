# Candidate-3 higher-bandwidth global-state geometry

## Selected setup

Use:

```text
H: 32 x 64
G: 24 x 48

partition H into nonoverlapping 4 x 4 blocks
map each H block to one 3 x 3 G block
```

The operator is a block-local, constant-preserving orthonormal DCT restriction
and its exact zero-padded synthesis right inverse. This is the cleanest
higher-bandwidth extension of the current `2x2 mean / 2x nearest` contract.

## Exact operators

Let `Q_n` be the orthonormal DCT-II analysis matrix of size `n`. For one
`4x4` high-resolution block `X`, define:

```text
C4 = Q4 @ X @ Q4.T
C3 = C4[0:3, 0:3]

D_block(X) = (3/4) * Q3.T @ C3 @ Q3
```

Thus D keeps the lowest `3x3` separable frequency coefficients and converts
them back to an ordinary `3x3` spatial latent block. Applying this to the
`8x16` nonoverlapping H macroblocks produces a regular `24x48` G grid.

For one `3x3` global block `Y`, define:

```text
C3 = Q3 @ Y @ Q3.T
C4 = zeros(4, 4)
C4[0:3, 0:3] = C3

U_block(Y) = (4/3) * Q4.T @ C4 @ Q4
```

Apply U independently to the corresponding `8x16` G macroblocks and reassemble
the resulting `4x4` blocks into `32x64` H geometry.

The scale factors preserve a constant field. For `X = c`, `D_block(X) = c`;
for `Y = c`, `U_block(Y) = c`.

## Right-inverse proof

Because each `Q_n` is orthonormal, coefficient cropping followed by zero
padding is an exact coordinate projection/injection pair. For arbitrary
`Y in R^(3x3)`:

```text
D_block(U_block(Y))
  = (3/4) * (4/3) * Q3.T @ (Q3 @ Y @ Q3.T) @ Q3
  = Y.
```

Therefore:

```text
D(U(G)) = G
```

in exact arithmetic for every G, not only for bandlimited inputs. With fixed
orthonormal matrices, float32 matmuls should be numerically exact to ordinary
roundoff; the runtime experiment must assert the actual maximum error before
model execution and after every coupled acceptance.

The present operator is the same construction at the limiting `2x2 -> 1x1`
case. Keeping only the 2D DC coefficient with scale `1/2` is exactly a `2x2`
mean, and the inverse scale `2` reconstructs a constant `2x2` nearest-repeated
block. The proposed pair therefore changes bandwidth, not the algebraic class
of the coupling contract.

## Token and compute accounting

```text
current G: 16 x 32 =  512 tokens
selected G: 24 x 48 = 1152 tokens
dense H:   32 x 64 = 2048 tokens
```

The selected global state is:

- `2.25x` the current global token count;
- `56.25%` of dense token count;
- 75% of dense spatial sampling in both axes.

For the global forward alone:

- token-linear projection, normalization, residual, MLP, and final-projection
  work grows approximately `2.25x` versus the 512-token branch;
- image-image attention matrix area grows approximately
  `(1152 / 512)^2 = 5.0625x` versus the current global branch;
- relative to dense, token-linear work is 56.25% and image-image attention
  matrix area is `(1152 / 2048)^2 = 31.64%`.

With the unchanged three 1024-token local calls, generated-token execution per
evaluation grows from:

```text
512 + 3 * 1024  = 3584
1152 + 3 * 1024 = 4224
```

or 17.86%. Approximate generated image-image attention matrix work grows from:

```text
512^2  + 3 * 1024^2 = 3,407,872
1152^2 + 3 * 1024^2 = 4,472,832
```

or 31.25%. This slightly exceeds one 2048-token dense image-attention matrix
in total work (`1.066x`) while retaining a 1152-token maximum model-call size.
It is a semantic-bandwidth experiment, not yet a total-compute saving.

The D/U transforms add small blockwise matrix operations over latent channels;
their cost is expected to be minor relative to four transformer forwards but
must be measured rather than assumed in runtime qualification.

## Spatial and interpolation assumptions

- The reduction is symmetric: `4 -> 3` in both axes. It preserves the 2:1
  global grid aspect ratio and does not preferentially discard horizontal or
  vertical samples.
- The retained passband is separable and rectangular. It is not rotationally
  isotropic and can treat diagonal frequencies differently from a circular
  low-pass.
- Every transform is local to one axis-aligned `4x4` macroblock. This avoids a
  global pseudoinverse and global ringing, but it introduces a possible
  `4x4/3x3` block-boundary assumption. Boundary discontinuity/blocking is the
  principal operator-specific failure mode to inspect.
- DCT-II implies even-reflected local boundary behavior within each macroblock.
  G values are ordinary spatial latent samples after inverse `Q3`; they are not
  frequency coefficients passed to FLUX.
- The regular `24x48` G grid should retain the existing full-canvas endpoint
  RoPE convention:

  ```text
  scale_y = (32 - 1) / (24 - 1) = 31/23
  scale_x = (64 - 1) / (48 - 1) = 63/47
  ```

  This treats reconstructed G samples as uniformly spanning the canvas. As in
  the current mean/nearest mapping, the latent restriction has support over
  finite H cells rather than representing literal point samples at every RoPE
  coordinate.

For iid unit H noise, the constant-preserving scale predicts G variance
`(3/4)^2 = 0.5625`, intermediate between the current mean-restricted variance
0.25 and dense variance 1. This is a consequence of retaining 9 of 16
orthonormal block coefficients, not a separate normalization policy.

## Rejected practical alternatives

- Ordinary bilinear/bicubic `32x64 -> 24x48 -> 32x64` does not satisfy
  `D(U(G)) = G` for arbitrary G.
- Defining U as a pseudoinverse of a standard resize can force a numerical
  right inverse, but introduces dense or poorly local inverse weights,
  boundary sensitivity, and a different interpolation algorithm in each
  direction.
- `16x64` or `32x32` supports exact integer mean/repeat with 1024 tokens, but
  halves only one spatial axis, changes the global grid aspect ratio, and
  confounds semantic bandwidth with strong directional anisotropy.
- A whole-canvas Fourier/DCT crop and zero-pad pair is also exact, but it makes
  every output depend on the complete canvas and introduces global boundary and
  ringing assumptions. The block-local pair is a closer controlled extension
  of the qualified current operator.

## Next runtime experiment

Change only global geometry and D/U to the `24x48`, blockwise `4x4 -> 3x3`
DCT pair. Keep the Phase-3c prompt/seed, five variants, four-step sigmas, crops,
overlap, intermediate hard coupling, and terminal release fixed. Assert
`D(U(G))`, initialization statistics, every nonterminal invariant, and D/E
lifecycle identity before running the model.

