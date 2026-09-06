# Persistent-H shared-provenance discriminator

## Narrow correction to the preceding result

The preceding recurrent experiment falsified **fresh same-sigma W reconstruction
from clean G predictions through regional noise scaling at high/near-one
sigma**.  It did not test or falsify a persistent high-resolution stochastic
trajectory.

## State contract

One destination noise field `H0` is generated once from the case seed.  `G0` is
derived deterministically from that same field by variance-normalized adaptive
area restriction; it has no independent random component.

For each interval:

```text
x0_G = model(G_i, sigma_i)
G* = Euler(G_i, x0_G, sigma_i -> sigma_next)

for each overlapping region r:
    crop_H = view(H_i, footprint_r)
    W_i,r = nearest_lift(crop_H)
    x0_W = model(W_i,r, sigma_i)
    W*_r = Euler(W_i,r, x0_W, sigma_i -> sigma_next)

H_next = normalized_overlap_assemble(restrict(W*_r))
```

There is no regional noise, `noise_scaling`, clean G anchor, or W reconstruction
from a denoised estimate.  W is transient model geometry for the persistent
accepted H trajectory.  `restrict(lift(crop(H_i))) == crop(H_i)` exactly.

Two arms are fixed:

- **G-authoritative:** accept ordinary `G*`.
- **H-feedback:** accept a plain bilinear restriction of `H_next` as `G_next`.

The arms share H0/G0, conditioning, schedule, region order, overlap weights,
and all local operations in interval 0.  H is retained in both arms.  No model
sees the complete destination H.

## Gate

Compare frozen Terminal, unguided four-interval terminal W, rejected fresh-W
recurrent, and both persistent-H arms.  Require deterministic complete coverage,
bounded residency/calls, verified interval-0 shared provenance, S3 composition,
and credible detail gain.  If persistent H still becomes locally complete prompt
interpretations, stop without cadence or policy sweeps.
