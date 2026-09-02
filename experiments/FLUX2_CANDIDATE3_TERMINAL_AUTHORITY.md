# Phase 8c — terminal-authority / lifecycle localization discriminator

## Result

The production terminal release is the direct large-scale failure mechanism in
this fixed 2048x4096 trajectory. Production A returns the severely fragmented
assembled local terminal estimate. Experimental B adds one fresh terminal
global prediction and exact hard projection, restoring one continuous bridge
scene from the identical preterminal state and identical terminal local
predictions. Experimental C projects toward retained synchronized G3 without a
new global call and collapses into noise.

Fresh terminal global authority matters. Retained G3 is a high-sigma state and
is not a valid sigma-zero coarse denoised target.

## Fixed configuration and controls

- H=128x256, production 4→3 DCT G=96x192.
- FLUX.2 Klein, bridge/train prompt, seed 20260901, CFG 1.
- Production four-interval schedule, 55 crops of 32x32 at stride 24.
- Production global/local coordinates, overlap assembly, nonterminal coupling,
  cadence, and atomic acceptance unchanged.
- Intervals 0–2 execute once. All variants share the same accepted H3/G3.
- The 55 terminal crop inputs and 55 terminal predictions execute once and are
  shared by A/B/C. Assembled terminal x0_H and terminal H_star are single
  shared tensors. Only B executes a terminal global prediction.

The production-control final RGB hash is
`e4f624ea0160a5193bea0266942367e7c781896f20fb90dee0a4372bc17c2fbc`,
exactly matching Phases 8a and 8b.

## Terminal Euler algebra

The measured sigma schedule is:

```text
1.000000000
0.999177158
0.997535527
0.992642820
0.000000000
```

At the terminal interval, `sigma_next=0`. Direct numerical comparison gives:

| Check | RMS | Max absolute |
|---|---:|---:|
| `H_star_terminal - x0_H_terminal` | 1.83e-8 | 4.77e-7 |

They are not bit-exact because the ordinary Euler expression performs
floating-point subtraction/division/multiplication, but they are numerically
equal. Production terminal release therefore returns the assembled terminal
local denoised estimate.

## Lifecycle localization

Decoded global x0 remains one coherent bridge at nonterminal intervals 0, 1,
and 2, reproducing Phase 8b. Decoded assembled local x0_H already contains
many independent bridge fragments at interval 0 and remains fragmented at all
later evaluations. The first three H_star and accepted-H states remain visually
near noise because their sigma changes are only about 0.00082, 0.00164, and
0.00489. Hard coupling maintains coarse-state equality but those tiny accepted
updates do not turn the local denoised alternatives into the returned image.

At interval 3, the sigma change is -0.992643. Terminal H_star becomes the
fragmented terminal x0_H to floating-point tolerance, and A publishes it
without global projection. This is where the previously latent local
alternatives become the final decoded scene.

The lifecycle contact sheet is `LIFECYCLE_CONTACT_SHEET.jpg`.

## Terminal variants

### A — production terminal release

`H_final=H_star`. It reproduces the Phase-8a/8b result: many repeated
independent bridges, broken decks, floating spans, repeated tower structures,
and weak train/lighthouse uniqueness.

### B — fresh terminal global hard projection

One terminal global x0_G is evaluated from accepted G3 at sigma 0.992643. Its
Euler proposal G_star is the terminal coarse denoised target. Exact projection
restores a single bridge system with a continuous deck, stable large towers,
one continuous yellow train, coherent horizon/water, and no field of floating
independent bridge fragments.

The image retains strong cable/deck detail, but the projection is extremely
large and produces visible softness/ghosted fine structure around some cables,
deck edges, and the far-right continuation. It is a causal diagnostic, not an
acceptable production policy without separate qualification.

### C — retained-G projection

No terminal global forward is executed. Projecting H_star toward accepted G3
produces a noise-like final image with no usable bridge scene. G3 is synchronized
with H3 at sigma 0.992643; it is not synchronized with a sigma-zero denoised
state. Retaining coarse authority is insufficient unless the coarse target is
also evolved to the terminal sigma.

The final comparison is `FINAL_ABC_COMPARISON.png`.

## Metrics

| Variant | Global forwards | Local forwards | Projection RMS | Projection/H_star | Low-frequency projection RMS | D(final)-target RMS |
|---|---:|---:|---:|---:|---:|---:|
| A release | 3 | 220 | 0 | 0 | 0 | 0.867312 vs terminal G_star |
| B fresh hard projection | 4 | 220 | 0.867312 | 102.51% | 0.867312 | 1.21e-7 |
| C retained-G projection | 3 | 220 | 0.863637 | 102.08% | 0.863637 | 1.26e-7 |

Pairwise final latent RMS is 0.867312 for A/B, 0.863637 for A/C, and
0.815999 for B/C. Projection magnitude exceeds H_star RMS in both projected
variants; output-space hard projection is doing scene-scale reconstruction,
not a small correction.

The shared measured run used 29.37 s of nonterminal global CUDA time, 76.08 s
of local CUDA time, and 9.70 s for B's additional terminal global forward.
Shared wall time including B was 117.62 s. The A/C-equivalent path excludes
that 9.70 s call. Peak before the terminal global call was 2.89 GiB allocated
and 6.23 GiB reserved; after it, peak allocation was 2.97 GiB while reserved
memory was unchanged. B therefore restores composition at the cost of one
full G forward and about 9.7 seconds in this run.

## Interpretation

Fragmentation is first visibly present in assembled local x0_H at interval 0,
not in current global x0_G. It becomes the returned failure specifically when
the dominant sigma-to-zero terminal update gives complete authority to local
x0_H. A fresh terminal global proposal can still recover one global scene,
proving output-space coarse authority is causally sufficient in this probe.

Retained G3 cannot substitute for a fresh terminal estimate. The experiment
therefore distinguishes terminal coarse authority from merely retaining the
last synchronized coarse latent.

This result does not authorize restoring hard terminal projection in
production: its correction RMS is larger than H_star RMS and visible fine
detail damage remains. A separate design task is required before changing the
terminal policy.

## Verdict

TERMINAL RELEASE IS THE LARGE-SCALE FAILURE MECHANISM
