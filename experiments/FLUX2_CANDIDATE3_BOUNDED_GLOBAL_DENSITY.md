# Phase 8b — bounded global-state density discriminator

## Result

Bounding the authoritative global state did not restore coherence at the fixed
2048x4096 target. The current 4-to-3 global branch produced a coherent,
continuous bridge in every decoded global x0 estimate, even though final H was
fragmented. The 8-to-4 branch produced disconnected bridge alternatives from
its first global x0, and the 8-to-3 branch lost most bridge geometry from its
first global x0. All three final H images remained severely fragmented.

This rejects the narrow Phase-8a hypothesis that excessive G token density is
the main cause of the observed large-canvas failure. It does not separate
global token count from retained DCT bandwidth or natural mapped-noise
variance.

## Fixed controls

- Output: 2048x4096; H=128x256=32,768 tokens.
- FLUX.2 Klein, Phase-8a bridge/train prompt and seed 20260901.
- CFG 1, four full-denoise Euler intervals.
- Production crop policy: 32x32, stride 24, 55 crops.
- 220 local forwards and 225,280 local token executions per generation.
- Persistent G/H states, hard nonterminal coupling, terminal release, and
  terminal-global omission unchanged.
- Each variant ran in an isolated fresh process. Production and ComfyUI core
  were not modified.

The H0 SHA-256 is identical across all variants:
`98aba634f2bdb0c30e4ebd6acb22636a26c564868fd20812e0c44ef20285e76d`.
Crop rectangles, local coordinates, local-forward count, local token work, and
sigma schedule are identical.

The current-control final RGB hash is
`e4f624ea0160a5193bea0266942367e7c781896f20fb90dee0a4372bc17c2fbc`,
exactly matching the Phase-8a fresh-process 2048x4096 Blueprint control.

## Operators and algebra

For n=4 or 8 and retained size k, each n-by-n H block is transformed with the
orthonormal DCT matrix Qn. Restriction retains the lowest k-by-k coefficients,
reconstructs through Qk, and multiplies by k/n. Prolongation transforms the
k-by-k G block, zero-pads its coefficients to n-by-n, reconstructs through Qn,
and multiplies by n/k.

| Variant | G | Tokens | D(U(G)) max | Constant restriction | Constant prolongation |
|---|---:|---:|---:|---:|---:|
| Current 4→3 | 96x192 | 18,432 | 1.19e-6 | 0 | 1.19e-7 |
| Bounded 8→4 | 64x128 | 8,192 | 7.15e-7 | 0 | 2.38e-7 |
| Bounded 8→3 | 48x96 | 4,608 | 7.15e-7 | 0 | 1.19e-7 |

All pass the 3e-6 experiment tolerance. Restriction/prolongation scales are
3/4 and 4/3, 1/2 and 2, and 3/8 and 8/3 respectively.

## Coordinates

All branches represent the complete H canvas. Endpoint-preserving coordinate
scales are derived independently per axis:

| Variant | scale_y | scale_x | first y,x | last y,x |
|---|---:|---:|---:|---:|
| 4→3 | 1.336842 | 1.335079 | 0,0 | 127,255 |
| 8→4 | 2.015873 | 2.007874 | 0,0 | 127,255 |
| 8→3 | 2.702128 | 2.684211 | 0,0 | 127,255 |

No branch was presented as a physically smaller canvas.

## Initial-state statistics

H0 variance is 0.999606 for all variants.

| Variant | G0 variance | G0/H0 variance |
|---|---:|---:|
| 4→3 | 0.561723 | 0.561945 |
| 8→4 | 0.249373 | 0.249471 |
| 8→3 | 0.140111 | 0.140167 |

These natural operator statistics were not variance-matched. The test changes
token density, retained bandwidth, and mapped variance together, as required.

## Runtime and memory

All timings are isolated fresh-process sampling measurements. CUDA time uses
events; VAE diagnostic decoding is excluded from sampling wall time.

| Variant | Global CUDA by interval | Total global | Local CUDA | Sampling wall | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| 4→3 | 9.87 / 9.50 / 9.53 s | 28.89 s | 74.75 s | 105.39 s | 17.37 GiB | 18.38 GiB |
| 8→4 | 2.86 / 2.69 / 2.70 s | 8.25 s | 74.52 s | 84.35 s | 17.37 GiB | 18.38 GiB |
| 8→3 | 1.75 / 1.20 / 1.21 s | 4.16 s | 74.64 s | 80.38 s | 17.37 GiB | 18.38 GiB |

Bounding G reduced global CUDA time by 71.4% (8→4) and 85.6% (8→3), and
sampling wall time by 20.0% and 23.7%. Local CUDA time stayed within 0.3%,
confirming unchanged local work. Peak allocator memory did not change because
model residency/local execution set the measured peak; this phase establishes
no VRAM saving from bounded G.

## Coupling and overlap

| Variant | Projection RMS intervals 0/1/2 | Projection/H* intervals 0/1/2 | Max invariant error |
|---|---:|---:|---:|
| 4→3 | 0.000726 / 0.001468 / 0.004435 | 0.0727% / 0.1471% / 0.4456% | 9.54e-7 |
| 8→4 | 0.000655 / 0.001324 / 0.004072 | 0.0655% / 0.1326% / 0.4091% | 4.77e-7 |
| 8→3 | 0.000624 / 0.001247 / 0.003766 | 0.0625% / 0.1250% / 0.3784% | 3.58e-7 |

Projection demand decreases modestly with coarser G; it does not grow or
destabilize. Pairwise local overlap RMS follows nearly the same per-interval
trajectory in every variant (about 1.010, 1.014–1.016, 1.030–1.031, then
1.000–1.004). All nonterminal right-inverse invariants pass, coverage is
complete, terminal release is unchanged, and outputs are finite.

## Semantic diagnostics and first divergence

### Current 4→3

Global x0 at intervals 0, 1, and 2 consistently shows one continuous bridge
system across the full canvas, with a stable deck, large left tower, central
tower, right-side structure, continuous water, and a coherent horizon. The
final H nevertheless contains many repeated independent bridge spans and
floating fragments. Thus the Phase-8a final repetition is not already present
as the same failure in the current global model estimate.

### Bounded 8→4

The first global x0 already contains roughly three disconnected bridge
alternatives rather than one continuous deck. This persists through intervals
1 and 2. Final H remains severely fragmented and is not materially more
coherent than the current control.

### Bounded 8→3

The first global x0 is under-specified: only isolated tower-like structures
remain, with almost no usable bridge deck. Later global x0 estimates do not
recover a full scene. Final H again contains many locally invented bridge
structures.

Across final outputs, none provides a continuous train/bridge composition,
unique lighthouse, or controlled tower count. Bounding makes the global model
call much cheaper, but removes semantic bandwidth needed to constrain H.

The comparison sheets are:

- `BOUNDED_GLOBAL_DENSITY_COMPARISON.png`
- `GLOBAL_X0_COMPARISON.png`

Machine-readable telemetry and decoded image hashes are in `report.json`.

## Interpretation

The tested bounded representations do not recover the dense reference's
single organized scene. The 18,432-token current G is not shown to be failing
because it is too dense: its decoded global x0 remains substantially more
coherent than either bounded branch. The final failure therefore occurs
downstream of, or despite, that coherent global estimate.

This does not prove the global trajectory is fully correct, nor does it isolate
positional extrapolation from coupling/local execution. It does show that a
simple global token cap implemented by more aggressive per-block DCT
restriction is not a production direction.

## One next experiment

At this same fixed canvas and with current 4→3 geometry only, capture and decode
the assembled local proposal H*, the hard-coupled accepted H, and the terminal
released H at every interval. This single lifecycle-localization probe should
identify whether fragmentation first enters through local x0 assembly, survives
hard coupling, or appears specifically at terminal release. Do not alter DCT
density, coordinates, or sampling in that probe.

## Verdict

BOUNDING GLOBAL STATE DOES NOT HELP
