# Recurrent/interleaved Blueprint discriminator

## Question

Can an evolving bounded Blueprint trajectory repeatedly initialize bounded local
refinement and improve credible detail without losing the qualified S3 scene?
This is an experiment only.  Candidate-3, Terminal Resampling, and the fixed
persistent-guide experiment remain unchanged.

## Same-sigma contract

The fixed four-interval qualified Blueprint schedule is used without a cadence
or strength sweep.  For interval `i`, both levels consume `sigma_i` and accept a
state at `sigma_(i+1)`:

```text
x0_G_i = model(G_i, sigma_i)
G*_next = Euler(G_i, x0_G_i, sigma_i -> sigma_next)

anchor_W_ir = bounded_transfer(x0_G_i, region_r)
W_ir = model_sampling.noise_scaling(sigma_i, noise(seed, r), anchor_W_ir)
x0_W_ir = model(W_ir, sigma_i)
W*_next_ir = Euler(W_ir, x0_W_ir, sigma_i -> sigma_next)
H*_next = normalized_overlap_assemble(restrict(W*_next_ir))
```

This deliberately uses the evolving denoised estimate `x0_G_i` as W's clean
anchor.  It does not resize a noisy G state, whose interpolation would change
noise statistics.  Native `noise_scaling` establishes W at the active sigma.
Thus G and W updates are trajectory-aligned; no state at one sigma is consumed
as though it belonged to another.

The arms differ only at atomic acceptance:

- **G-authoritative:** `G_next = G*_next`; `H*_next` is diagnostic and does not
  feed back.
- **Bidirectional:** `G_next = interpolate(H*_next, G_hw)`; the accepted H state
  and replacement G state are both at `sigma_next`.

W is transient and reconstructed once per region per accepted interval.  H is
materialized only as the required assembled-state diagnostic and, for the
bidirectional arm, as the source of one bounded downscale.  No model sees H.

## Geometry and controls

- Reuse the qualified square S3 case, seed, conditioning, `45x45` G, `128x128`
  H, twenty-five overlapping `32x32` footprints, and `64x64` W.
- Reuse the halo-aware bounded `G crop -> footprint -> W` bilinear transfer.
- Reuse the four accepted intervals and deterministic region noise policy.
- Compare persisted frozen Terminal, persisted unguided four-interval W,
  G-authoritative recurrent, and bidirectional recurrent.
- Run each recurrent arm twice and require bit-exact final output and interval
  hashes.

## Telemetry and gate

Record accepted input/output G hashes, G predictions, W anchor/initial/accepted
hashes, H hashes, bounded transfer round-trip error, feedback delta, overlap,
gradient and low-frequency metrics, coverage, calls, and inter-region CUDA
allocation barriers.  Visual acceptance requires S3, no new duplicates or
cross-tile disagreement, and credible structural/local detail beyond both
controls.  If neither arm meets that gate, stop without cadence tuning.
