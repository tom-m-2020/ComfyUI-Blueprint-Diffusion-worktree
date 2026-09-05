# Phase 28 — Geometry-only terminal-resampling generalization

## Decision

Use a small finite set of approximately fixed-2K, aspect-preserving geometry profiles. Every
profile uses the unchanged `32x32` destination regions, stride 24, `64x64`
native local working canvas, fixed terminal sigma `0.25`, nearest-2 lift,
average-2 restriction, and normalized streaming assembly. Blueprint model work
is capped at 2,048 generated image tokens.

The new square, portrait, and less-extreme landscape profiles are deterministic
and preserve S3 whole-scene organization. The original `128x256` destination
remains bit-exact to Phase 27.

## Architecture audit

The audited quantities are:

```text
Blueprint B: bounded native-coordinate model canvas
Destination H: model-external final latent canvas
Region R: fixed 32x32 rectangle in H
Working W: fixed native-coordinate 64x64 model canvas
B -> H: CPU float32 bilinear, align_corners=False
R -> W: nearest-neighbor 2x
W -> R: arithmetic avg_pool2d(2,2)
plan: stride 24, deterministic end alignment, row-major
accumulator: H-sized weighted_sum plus one-channel coverage
```

Candidate policy assessment:

| Policy | Assessment |
|---|---|
| A. Approximately fixed token budget with aspect-derived shapes | Selected within a finite allowlist. Noninteger mappings use deterministic adaptive area restriction plus an analytic per-cell independent-noise scale, preserving unit marginal variance without runtime-statistics tuning. |
| B. Fixed long-axis Blueprint extent | Bounded, but makes square B `64x64`/4,096 tokens while current landscape is 2,048 and changes the initialization scale. It adds work without qualification evidence. Rejected. |
| C. Small finite native-ish shape set | Used as the fail-closed production policy around A: four enumerated aspect-preserving shapes with 1,944–2,048 tokens. Unsupported geometries remain rejected. |
| D. One fixed square Blueprint for every aspect | Bounded, but distorts the model-side aspect ratio and changes the qualified planning semantics. Rejected. |

The selected finite policy is not `B=H/4` as an open-ended rule. It is an
explicit closed set:

| Destination latent / pixels | Blueprint | B tokens | Regions | Accumulator elements |
|---|---:|---:|---:|---:|
| `128x256` / `2048x4096` | `32x64` | 2,048 | 55 | `128*128*256` plus coverage |
| `128x192` / `2048x3072` | `36x54` | 1,944 | 40 | `128*128*192` plus coverage |
| `128x128` / `2048x2048` | `45x45` | 2,025 | 25 | `128*128*128` plus coverage |
| `256x128` / `4096x2048` | `64x32` | 2,048 | 55 | `128*256*128` plus coverage |

Hypothetical destinations outside this table do not increase B: they fail
closed pending separate qualification. Thus the production policy has the
explicit bound `B_tokens <= 2048`; destination growth within future work may
increase region count and accumulator residency, not the currently qualified
global or per-local model budgets.

For the original and portrait profiles, Phase-27 initialization remains the
exact 4x integer-area mean over `m=16` destination-noise positions plus
`sqrt(15/16)` independent Blueprint noise. The square and 3:2 profiles use
PyTorch adaptive area means. For output cell `(i,j)`, the implementation derives
the exact contributor counts from adaptive-pooling bounds and adds independent
noise with scale `sqrt(1 - 1/(n_y(i)n_x(j)))`. This is a deterministic analytic
variance preservation rule, not a measured or tuned normalization.

## Qualification

The original Phase-27 car/tree/house regression was rerun after the geometry
change. Initial B, all four B predictions and accepted states, mapped terminal
B, 55 noise/restricted hashes, final H, and decoded RGB remain bit-exact; there
is no mismatching boundary.

New results:

| Case | B/H | Regions | Repeat hash | Selected | Tiled control | Wall | Peak allocated |
|---|---|---:|---|---|---|---:|---:|
| Square multi-object | `45x45 / 128x128` | 25 | exact | S3 | S0 | 28.52 s | 3,055,350,784 B |
| Portrait astronaut | `64x32 / 256x128` | 55 | exact | S3 | S0 | 59.56 s | 3,089,858,048 B |
| Landscape bridge | `36x54 / 128x192` | 40 | exact | S3 | S0 | 43.16 s | 3,072,888,320 B |

The selected square contains one car/tree/house in the requested order; the
portrait contains one continuous astronaut; and the landscape contains one
dominant continuous bridge and horizon. Each matched direct tiled-local control
is S0 and repeats complete crop-local scenes. The decision is perceptual; the
controls are not rejected from numerical overlap alone.

All new profiles have complete positive coverage, finite tensors, row-major
region/noise provenance, exactly four bounded Blueprint model calls, local
calls equal to region count, zero destination-sized model calls, and identical
latent hashes on repeat execution. Local model inputs remain `64x64` with
ordinary native coordinates. GPU barrier telemetry remains non-growing with
region ordinal.

Artifacts:

- `terminal_resampling_geometry_qualification_results/report.json`
- `terminal_resampling_geometry_qualification_results/semantic_review.json`
- `terminal_resampling_geometry_qualification_results/PHASE28_COMPARISON.jpg`

## Boundary

This qualifies only four enumerated geometry profiles. It does not qualify an
arbitrary aspect ratio, interpolation-based Blueprint initialization, larger
destination, alternate working size, adaptive regions, or user-configurable
geometry policy.

## Verdict

TERMINAL-RESAMPLING GEOMETRY GENERALIZATION QUALIFIED
