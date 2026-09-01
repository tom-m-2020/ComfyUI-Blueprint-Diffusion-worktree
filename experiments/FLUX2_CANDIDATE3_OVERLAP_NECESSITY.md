# Phase 6f — Candidate-3 overlap necessity discriminator

## Verdict

**OVERLAP STILL REQUIRED**

Zero overlap materially reduced local work and warm sampling time, but failed
the boundary-quality gate in the person/car/tree stress case. The persistent
global trajectory preserved the broad scene, yet it did not prevent a visible
duplicate/offset shoulder and torso structure where the central person crossed
a tile boundary. The bridge case remained broadly coherent, but its measured
boundary discontinuity also increased.

Stride 28 remained visually close to the current result, but deterministic end
alignment produced the same crop count and token executions as stride 24 in
both requested geometries. It is therefore not a useful optimization under the
tested geometry policy. No production change is justified.

## Scope and controls

The experiment used native FLUX.2 Klein, CFG 1, the qualified four-step Euler
schedule, dynamic 4-to-3 block-DCT global state, hard nonterminal coupling,
terminal release, absolute full-canvas coordinates, ordinary sequential crop
execution, and the existing overlap weights. Only the experiment-local crop
stride changed:

| Variant | Crop | Stride | Nominal overlap |
|---|---:|---:|---:|
| A CURRENT | 32x32 | 24 | 8 |
| B REDUCED OVERLAP | 32x32 | 28 | 4 |
| C NO OVERLAP | 32x32 | 32 | 0 |

The production package and ComfyUI core were not modified. The harness is
`flux2_candidate3_overlap_necessity.py`; complete telemetry is in
`flux2_candidate3_overlap_necessity_results/report.json`.

One unmeasured stride-24 run warmed each geometry. A/B/C were then measured in
order from the same loaded model state. CUDA events measured model work;
sampling wall time excludes text encoding, VAE decode, model loading, and file
output. Images were decoded only after all timed sampling runs.

## End-alignment finding

“Nominal overlap” is not the actual overlap at every boundary. The existing
planner appends an end-aligned final start whenever the stride grid misses the
canvas end. This is necessary for exact coverage, but changes the work
interpretation:

| H grid | Stride | Y starts | X starts | Crop grid | Actual adjacent axis overlaps |
|---|---:|---|---|---:|---|
| 32x64 | 24 | 0 | 0,24,32 | 1x3 | X: 8,24 |
| 32x64 | 28 | 0 | 0,28,32 | 1x3 | X: 4,28 |
| 32x64 | 32 | 0 | 0,32 | 1x2 | X: 0 |
| 64x128 | 24 | 0,24,32 | 0,24,48,72,96 | 3x5 | Y: 8,24; X: 8,8,8,8 |
| 64x128 | 28 | 0,28,32 | 0,28,56,84,96 | 3x5 | Y: 4,28; X: 4,4,4,20 |
| 64x128 | 32 | 0,32 | 0,32,64,96 | 2x4 | Y/X: 0 |

Thus B reduces some local overlaps but increases the end-aligned overlap and
does not change total crop work at either tested size.

## Work and performance

All values are actual runtime geometry and one warm measured generation.
Token counts are work accounting, not FLOP or wall-time claims.

| Target | Variant | Crops/int. | Local tokens/int. | Redundancy | Total calls | Global CUDA | Local CUDA | Sample wall | Peak alloc/reserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x512 | A | 3 | 3,072 | 1.500x | 16 | 1.146 s | 3.001 s | 4.202 s | 2,577 / 2,768 MiB |
| 1024x512 | B | 3 | 3,072 | 1.500x | 16 | 1.140 s | 3.030 s | 4.219 s | 2,577 / 2,768 MiB |
| 1024x512 | C | 2 | 2,048 | 1.000x | 12 | 1.143 s | 2.028 s | 3.214 s | 2,577 / 2,768 MiB |
| 1024x2048 | A | 15 | 15,360 | 1.875x | 64 | 4.549 s | 15.305 s | 20.050 s | 2,978 / 3,510 MiB |
| 1024x2048 | B | 15 | 15,360 | 1.875x | 64 | 4.543 s | 15.242 s | 19.996 s | 2,978 / 3,510 MiB |
| 1024x2048 | C | 8 | 8,192 | 1.000x | 36 | 4.532 s | 8.136 s | 12.788 s | 2,975 / 3,510 MiB |

Relative to A, C eliminated 33.3% of local token executions/calls in the
32x64 case and 46.7% in the 64x128 case. Sampling wall time fell 23.5% and
36.2%, respectively. Summed local CUDA time fell 32.4% and 46.8%. Peak
allocated memory changed by less than 4 MiB and peak reserved memory did not
change; sequential crop count affects total work, not the dominant per-call
peak in this setup.

B changed warm wall time by +0.4% and -0.3%, consistent with no meaningful
performance change.

## Integrity

All six runs reported `SUCCESS` and passed finite-state, positive-coverage,
immutable-H crop input, one-global/one-assembly/one-atomic-acceptance lifecycle,
nonterminal coupling, terminal-release, and four-preview checks. Maximum
nonterminal invariant error was `7.15e-7`. The random geometry right-inverse
check was at most `7.15e-7` for 32x64 and `9.54e-7` for 64x128.

## Boundary diagnostics

A/B use ordinary pre-blend overlap disagreement. C has no common prediction
domain, so it cannot have an overlap RMS. For all variants the harness also
measured RMS between one-token-wide strips immediately on either side of each
crop start in the assembled prediction/state. This adjacent-strip metric is
intentionally separate: scene content can change across a boundary, so it is a
comparative discontinuity signal, not a ground-truth seam metric.

### Final accepted H adjacent-strip RMS

| Target | A current | B reduced | C zero |
|---|---:|---:|---:|
| 1024x512 | 0.6379 | 0.6240 | 0.7913 |
| 1024x2048 | 0.8382 | 0.8250 | 0.9971 |

C was 24.1% above A in the bridge case and 18.9% above A in the stress case.
At the first model evaluation, assembled-prediction adjacent-strip RMS rose
from 0.3835 to 1.1604 for the bridge and from 0.6136 to 1.4811 for the stress
case. Coupling reduced but did not erase the boundary-local mismatch.

Ordinary overlap disagreement declined through the trajectory for A/B. At the
terminal evaluation it was 0.2408/0.2049 (A/B) for the bridge and
0.2637/0.2320 for the stress case. Those values are not directly comparable to
C's adjacent-strip metric.

## Decoded semantic inspection

For bridge/train, A and B retain one continuous suspension bridge, two
principal towers, a single central train, the left lighthouse, right stone
tower, and continuous water/horizon. C retains the broad composition and shows
no gross vertical seam, but changes train/deck detail around the central
boundary and has the larger measured boundary discontinuity. This case alone
would not falsify zero overlap.

For person/car/tree, A and B retain one centered person, one dominant left car,
one dominant right tree, a continuous ground plane, and coherent perspective.
C preserves that global arrangement, demonstrating the value of persistent
G/H coupling, but the person lies on the x=32 latent boundary and develops a
visible offset extra left shoulder/torso contour. That is the crop-local
semantic/boundary failure the discriminator was designed to expose. No
postprocessing was applied.

All six decoded PNGs are under
`experiments/flux2_candidate3_overlap_necessity_results/`.

## Conclusion

Candidate-3 global coupling can preserve broad composition without local
overlap, but it does not replace overlap's role in reconstructing objects that
cross local crop boundaries—especially after terminal release grants the local
proposal final authority. Zero overlap is not qualified.

The tested stride-28 layout is quality-safe in these scenes but computationally
useless because end alignment leaves 3 and 15 crops. The recommended
production policy remains deterministic stride 24 / nominal overlap 8. A
production change is not warranted from this experiment.
