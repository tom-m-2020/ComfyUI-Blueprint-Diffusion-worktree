# Persistent-H shared-provenance result

## Verdict

**FAIL — persistent shared-provenance H still fragments; stop this
discriminator.**

This is narrower than the preceding fresh-W result.  It establishes that a
single persistent noisy H canvas and perfectly shared overlapping inputs are not
alone sufficient to preserve S3 when the model evaluates every W as a bounded
local canvas without additional global spatial authority.

## Integrity

- H0 was generated once. G0 was deterministically derived from the same H0 with
  no independent random component.
- At interval 0, every destination crop was a storage-sharing view of accepted
  H0. `restrict(lift(crop))` had exactly zero error, and all overlapping crop
  values agreed with zero maximum error.
- No regional RNG or `noise_scaling()` operation occurred.
- One normalized assembly produced the sole retained H state at every interval.
- Each arm made four bounded `45x45` G calls and 100 bounded `64x64` W calls,
  with zero destination-sized model calls.
- Coverage was complete and CUDA allocation at accepted-interval barriers was
  constant within each arm.
- Both arms reproduced bit-exactly on an independent run.

The final H hash is identical for both arms:
`81a7fc62b848fb6298b8e0166fc81902d4996864f79892728277e9ef8c8fdf43`.
Their G hashes differ, proving H-to-G feedback executed, but under this
discriminator G has no permitted return path into H and therefore cannot alter
the final H.

## Semantic and numerical result

Both arms contain repeated cars, trees, and houses arranged in five stacked
local scene bands.  They fail object count, one-horizon composition, and
cross-tile scene unity.  Local objects are more developed, but this is not a
credible detail improvement because it comes from duplicated local prompt
interpretations.

Final overlap RMS is `0.850843`, versus `0.173451` for frozen Terminal,
`0.190880` for unguided four-interval W, and about `0.943` for rejected fresh-W
recurrent reconstruction.  Gradient RMS is `0.737768`; visual inspection shows
that this high energy includes repeated structures and band boundaries rather
than qualified detail.

## Localization

Shared stochastic provenance fixes inconsistent overlapping noise, but the
bounded W model calls still receive local working-canvas coordinates and the
complete prompt.  They can therefore evolve the shared H crop into locally
complete scenes.  Normalized overlap makes one numerical H trajectory but does
not supply whole-canvas semantic or coordinate authority.

Per the requested gate, stop this discriminator without cadence or policy
sweeps.  This result does not falsify every possible recurrent architecture; it
falsifies persistent shared-provenance H **without a mechanism by which evolving
G information affects W/H predictions or accepted H state**.
