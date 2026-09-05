# Phase 31 — Blueprint-to-native-working-canvas lift discriminator

## Result

**B — LIFT CHANGES OUTPUT BUT DOES NOT SOLVE SOFTNESS**

Replacing the qualified nearest-neighbor `32x32 -> 64x64` lift with bilinear
interpolation (`align_corners=False`) changes every terminal prediction while
preserving S3 semantics, but it does not provide a credible detail improvement.
Across all three cases the bilinear results are slightly smoother, not sharper.

No production or ComfyUI-core code changed. Blueprint trajectories and nearest
controls were reused from Phase 29; only the new bilinear local branches were
executed.

## Integrity and control regression

- All three Phase-29 `sigma=0.25` controls reproduce their Phase-28 selected
  latent bit-exactly (`RMS=0`, `max_abs=0`).
- Persisted Blueprint terminal and mapped hashes validate before reuse.
- Bilinear execution uses the same sigma, region seeds/noise tensors, model,
  conditioning, native coordinates, restriction, weights, and region order.
- New local calls are exactly 25 square, 55 portrait, and 40 landscape; there
  are zero Blueprint recomputations and zero destination-sized model forwards.
- Every branch is finite with complete positive coverage.
- Independent repeat execution is bit-exact for final latent and every
  restricted region prediction in all three cases.
- Post-region CUDA allocation is constant within every branch (barrier range
  `0` bytes), so local state does not accumulate with region ordinal.

## Pre-inference lift diagnostics

For each crop `b`, the two fixed lifts are:

```text
A(b) = nearest(b, scale=2)
B(b) = bilinear(b, scale=2, align_corners=False)
```

Natural `2x2` mean restriction is measured without correction. As expected,
`avgpool2(A(b)) == b` exactly for every region. Bilinear is not an exact right
inverse: its restriction differs from `b`, and its anchor gradient is lower
than nearest's replicated-block boundary gradient.

| Case | mean RMS B-A | mean RMS avgpool(B)-b | maximum abs avgpool(B)-b |
|---|---:|---:|---:|
| square multi-object | 0.06536 | 0.03015 | 0.71039 |
| portrait astronaut | 0.04509 | 0.01684 | 0.22849 |
| landscape bridge | 0.05193 | 0.02108 | 0.44928 |

Every report entry records crop, nearest-anchor, and bilinear-anchor hashes;
restriction error, low-frequency error, and gradient RMS are recorded per
region.

## Output measurements

| Case | Grade A/B | RMS B-A | overlap A -> B | gradient A -> B | BP RMS A -> B | LF RMS A -> B | CUDA / wall B | peak alloc / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| square | S3 / S3 | 0.02889 | 0.17345 -> 0.17059 | 0.23882 -> 0.22473 | 0.13250 -> 0.13775 | 0.06465 -> 0.06839 | 24.75 / 25.66 s | 2.837 / 3.316 |
| portrait | S3 / S3 | 0.01581 | 0.17225 -> 0.17097 | 0.18823 -> 0.18270 | 0.12441 -> 0.12604 | 0.07378 -> 0.07617 | 55.05 / 56.75 s | 2.862 / 3.344 |
| landscape | S3 / S3 | 0.02037 | 0.16277 -> 0.16084 | 0.20241 -> 0.19425 | 0.12333 -> 0.12611 | 0.06427 -> 0.06706 | 39.06 / 40.19 s | 2.850 / 3.332 |

The bilinear latent hashes are:

- square: `fee7e31f9359dde216bef70acb9f9a6df86f9a3cc9261c59dbcc093d49e90472`;
- portrait: `22158020381fb38108dff50441ac48e899bd42b5bee5a4eae6f16b148ff261ce`;
- landscape: `74fb80ad04a51602bd74c7cf58c44680653501a6a834cbf502bd770ad891311f`.

## Semantic/detail review

- **Square:** one car, tree, and house remain correctly organized. Bilinear
  does not improve the car contour/wheels, foliage, roof/window structure, or
  grass texture; it appears marginally smoother.
- **Astronaut:** one body remains coherent. Helmet, shoulders/arms, suit panels,
  shadow, and ground texture are not materially improved; boundaries are
  slightly smoother.
- **Bridge:** one continuous bridge system remains. Tower edges, cables, deck,
  secondary tower, and water detail do not improve; fine lines are marginally
  smoother.

The small overlap reduction is not a detail win. Gradient RMS falls in every
case, while Blueprint and low-frequency discrepancy rise slightly. The visual
review agrees with those directions: bilinear changes the manifold presented to
the local model but does not regenerate missing native detail.

## Interpretation

The zero-order nearest lift is not the primary cause of the remaining softness.
Bilinear interpolation removes the replicated `2x2` discontinuities, but it
also introduces a natural coarse mismatch under the fixed mean restriction and
does not improve visible fidelity. Do not test another interpolation kernel.

The next architecture should investigate one richer Blueprint-derived
working-state construction at the same geometry and sigma, with a declared
information/provenance invariant, rather than another interpolation sweep.

## Artifacts

- Harness: `experiments/flux2_terminal_resampling_lift_discriminator.py`
- Machine report: `experiments/flux2_terminal_resampling_lift_discriminator_results/report.json`
- Comparison: `experiments/flux2_terminal_resampling_lift_discriminator_results/PHASE31_COMPARISON.jpg`
- Detail review: `experiments/flux2_terminal_resampling_lift_discriminator_results/DETAIL_REVIEW.jpg`
- Resumable per-case bilinear latent and telemetry artifacts live under the
  results directory.
