# Phase 6g — Candidate-3 local window scale discriminator

## Verdict

**LARGER WINDOWS PARTIALLY HELP**

The measured crossover depends on target geometry. A 48x48 window is slower
than current at 1024x2048 but faster at 768x1536. A 64x64 window is faster at
1024x2048 and preserves the scene semantically, but raises the final
adjacent-boundary strip metric. No single larger window is qualified as a
general production replacement from these two geometries.

No production or ComfyUI-core code changed.

## Configuration

All variants used the same FLUX.2 Klein backend, prompt/seed per geometry,
CFG 1, four-step Euler schedule, accepted G/H lifecycle, dynamic 4-to-3
block-DCT geometry, hard nonterminal coupling, terminal release, absolute
full-canvas coordinates, sequential crop execution, and normalized overlap
assembly. Only crop size and stride changed:

| Variant | Crop | Stride | Nominal overlap |
|---|---:|---:|---:|
| A CURRENT | 32x32 | 24 | 8 |
| B MEDIUM | 48x48 | 36 | 12 |
| C LARGE | 64x64 | 48 | 16 |

Each geometry received one unmeasured A warm-up. Timed runs reused the loaded
model state. CUDA-event times cover model work; sampling wall time excludes
loading, text encoding, VAE decode, save, and output inspection.

The 1024x512 bridge canvas has latent shape 32x64, so neither 48x48 nor 64x64
fits. It was not padded or rerun because that would add new semantics. The
person/car/tree prompt and seed from Phase 6f were used at 1024x2048 and at the
additional 768x1536 geometry.

## Geometry, work, and performance

| Target | Variant | H | Y starts | X starts | Crops | Local tokens | Unique H | Redundancy | Total forwards | Global CUDA | Local CUDA | Wall | Peak alloc/reserved |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x2048 | A 32 | 64x128 | 0,24,32 | 0,24,48,72,96 | 15 | 15,360 | 8,192 | 1.875x | 64 | 4.616 s | 15.479 s | 20.299 s | 2,978 / 3,510 MiB |
| 1024x2048 | B 48 | 64x128 | 0,16 | 0,36,72,80 | 8 | 18,432 | 8,192 | 2.250x | 36 | 4.632 s | 16.894 s | 21.697 s | 2,982 / 3,510 MiB |
| 1024x2048 | C 64 | 64x128 | 0 | 0,48,64 | 3 | 12,288 | 8,192 | 1.500x | 16 | 4.680 s | 12.113 s | 16.923 s | 2,977 / 3,510 MiB |
| 768x1536 | A 32 | 48x96 | 0,16 | 0,24,48,64 | 8 | 8,192 | 4,608 | 1.778x | 36 | 2.401 s | 8.336 s | 10.847 s | 2,746 / 3,078 MiB |
| 768x1536 | B 48 | 48x96 | 0 | 0,36,48 | 3 | 6,912 | 4,608 | 1.500x | 16 | 2.367 s | 6.216 s | 8.661 s | 2,746 / 3,078 MiB |
| 768x1536 | C 64 | 48x96 | — | — | — | — | 4,608 | — | — | — | — | — | skipped: crop does not fit |

Token counts describe executed image-token instances, not FLOPs. Larger
attention matrices make the cost nonlinear:

- At 1024x2048, B reduces calls but executes 20% more local tokens. Local CUDA
  rises 9.1% and wall time rises 6.9%.
- At 1024x2048, C executes 20% fewer local tokens, local CUDA falls 21.7%, and
  wall time falls 16.6%.
- At 768x1536, B executes 15.6% fewer local tokens, local CUDA falls 25.4%, and
  wall time falls 20.2%.
- Peak allocated changes by at most about 4 MiB and peak reserved is unchanged.
  Sequential-window size did not materially alter allocator peak here.

## Boundary and overlap telemetry

| Target | Variant | Terminal pre-blend overlap RMS | Final H adjacent-strip RMS |
|---|---|---:|---:|
| 1024x2048 | A 32 | 0.2637 | 0.8382 |
| 1024x2048 | B 48 | 0.2071 | 0.8203 |
| 1024x2048 | C 64 | 0.1649 | 0.9580 |
| 768x1536 | A 32 | 0.1965 | 0.9199 |
| 768x1536 | B 48 | 0.1272 | 0.9769 |

All larger-window variants lower ordinary overlap disagreement. B also lowers
the final adjacent-strip metric at 1024x2048. C's final adjacent-strip RMS is
14.3% above A there; B's is 6.2% above A at 768x1536. The adjacent-strip metric
compares neighboring assembled token columns/rows and is not an overlap RMS or
a perceptual seam score.

## Integrity

Every executed run reported `SUCCESS`, finite final output, complete positive
coverage, immutable accepted-H crop views, one global proposal and one
assembled local prediction per interval, one atomic pair acceptance, four
previews, nonterminal `D(H_next)=G_next` within `7.16e-7`, and terminal release
only on the last interval.

## Visual inspection

At 1024x2048 all three variants retain one centered person, one dominant left
red car, one dominant right tree, continuous terrain/horizon, coherent
perspective, and comparable local sharpness. Neither larger window creates the
duplicate shoulder/torso failure seen in Phase 6f zero-overlap. No visible seam
or new object duplication was found. The 64-window result changes small distant
car/terrain details and has the higher boundary-strip metric, but remains
semantically coherent.

At 768x1536, A and B retain the same person/car/tree layout, intact anatomy,
continuous ground and mountains, and comparable detail. No visible seam or
coordinate displacement was found in B.

Five decoded PNGs are in
`experiments/flux2_candidate3_local_window_scale_results/`.

## Interpretation and next qualification

Larger windows are a real optimization at favorable shapes, but a fixed
replacement is not supported. End alignment controls the crop grid and can
make a nominally medium window execute more work than 32x32, while a large
window can cross the runtime crossover despite larger attention matrices.

The next narrowly justified production qualification is **64x64 / stride 48
for latent H=64x128 (1024x2048 decoded output)**. It should be tested across at
least the bridge/train-type long-geometry prompt at a compatible canvas and
additional boundary placements before any geometry-dependent policy is added.
The 48x48 / stride 36 result at H=48x96 is also promising, but does not justify
a universal 48-window policy because it regresses runtime at H=64x128.

This phase does not warrant a production change and does not yet justify
stopping local-geometry research in favor of global cadence: one larger-window
configuration produced a material measured speedup with preserved visible
semantics.
