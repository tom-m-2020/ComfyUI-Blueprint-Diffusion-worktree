# Phase 44 — Z-Image-Turbo Terminal Refine discriminator

## Verdict

**Reject the direct Z-Image Terminal Refine port and stop.** The fixed arm
preserves the S3 scene, but it does not provide credible structural/detail
improvement over the plain mapped Blueprint. It is slightly smoother overall.
This fails the declared semantic gate even though every causal, geometric,
determinism, overlap, and bounded-execution invariant passes.

## Fixed experiment

- Prompt: exactly one red vintage car on the left, one tall green tree in the
  center, and one small white house on the right, on one field and horizon.
- Seed: `20260921`.
- Native Z-Image-Turbo, 16-channel latent, NextDiT patch 2, CFG 1,
  BasicGuider, fixed-shift CONST flow with shift 3.
- G `128x128`, H `256x256`, F `64x64`, stride `48x48`, W `128x128`.
- 25 row-major regions. G and W calls use zero-origin native coordinates.
- Global sigmas: `[1.0, 0.9545454383, 0.8999999762, 0.8333333135,
  0.75, 0.6428571343, 0.5, 0.3000000119, 0.0]`.
- Local interval: exactly `0.3000000119 -> 0`.
- Nearest F-to-W enlargement and area-mean W-to-F restriction.

## Results

The native 1024 control and plain mapped 2048 control both contain one car,
one tree, one house, and one continuous horizon. Terminal Refine retains the
same counts, left/center/right placement, perspective, and horizon. It shows no
miniature-scene lattice, duplicate objects, separately formed regional scenes,
or visible cross-footprint structural break.

Terminal Refine nevertheless supplies no credible new car contour, branch
structure, house geometry, or ground structure beyond the plain mapped control.
The decoded C-vs-B pixel RMS is `0.04150774`; gradient RMS changes from
`0.07187832` (B) to `0.06470858` (C), a ratio of `0.90025174`.

## Causal and execution evidence

- Exactly 8 G calls and 25 W calls per run; zero H-sized calls.
- Every call shape is `[1,16,128,128]`, corresponding to `64x64` image tokens.
- Every call records absent `rope_options`; no destination offset/scaling exists.
- Coverage min/max: `0.99999988 / 1.00000012`.
- Pre-blend overlap RMS: `0.13594482`.
- Final latent SHA-256:
  `5c25a21f5692d3940e9d1808c2af88db9aae4c4ff503285baa65009aaf1ad6ea`.
- Independent repeat is bit-exact, including per-region hashes.
- Peak CUDA allocated/reserved: `6,735,106,560 / 7,201,619,968` bytes.
- All 25 completed-region barriers are `6,218,021,888` bytes; range is zero.
- Primary model time: 25.996 s global + 67.557 s local = 93.552 s.
- Final H is published only after all 25 proposals succeed.

Complete telemetry is in
`zimage_terminal_refine_discriminator_results/report.json`; the decoded visual
comparison is `zimage_terminal_refine_discriminator_results/COMPARISON.jpg`.

## Stop decision

Do not continue this port with another local sigma, overlap, footprint, W size,
CFG, sampler, destination-aware RoPE, DyPE, guide strength, or multi-step
refinement. Each would be a distinct architecture requiring a new decision.
