# Phase 19 — bounded global-state representation falsifier

Date: 2026-09-04

## Result

Neither a fixed-budget single-scale source nor an equal-budget explicit
two-level whole-canvas source moves the normalized-W result out of the S0
fragmented class. Phase 19 therefore rejects the tested bounded global states
at this fixed terminal state. No trajectory is authorized.

## Controlled state

The experiment reuses the Phase-14/17/18 large bridge case: accepted
`H=128x256`, unchanged accepted G, seed `20260901`, the same terminal sigma and
conditioning, 55 destination regions, reconstructed native-coordinate
`W=64x64`, W-to-H restriction, overlap assembly, and zero accepted updates.

Each arm has exactly one shared 4,096-token source trajectory. The source is
never mutated by W and no W-specific source state exists. At every one of 25
blocks, the source advances normally, all 55 W consumers use that block's
generated K/V, and that K/V is released. All W consumers use identical source
provenance.

## Predeclared representations

### A — existing simple control

`32x128 = 4096` tokens. Every source token is the arithmetic mean of one
nonoverlapping `4x2` H cell:

```text
S[r,c] = mean(H[4r:4r+4, 2c:2c+2])
y = 4r + 1.5
x = 2c + 0.5
```

This exactly reproduces the Phase-14 fixed-budget control.

### B — bounded single scale

`64x64 = 4096` tokens. Every source token is the arithmetic mean of one
nonoverlapping `2x4` H cell:

```text
S[r,c] = mean(H[2r:2r+2, 4c:4c+4])
y = 2r + 0.5
x = 4c + 1.5
```

The entire canvas is represented once with a different fixed model geometry,
not as a crop. For arbitrary larger H, this construction remains a fixed
`64x64` area restriction; cell support expands while model tokens remain 4096.

### C — bounded hierarchy

Two levels are concatenated before the source transformer and jointly interact
as one shared source:

- coarse `16x32 = 512`: exact nonoverlapping `8x8` means, coordinates
  `(8r+3.5, 8c+3.5)`;
- medium `32x112 = 3584`: linear area filtering over the complete H canvas,
  coordinates `(4r+1.5, (16/7)c+9/14)`.

The coarse level reserves one eighth of the budget for explicit whole-scene
layout. The medium level uses seven eighths for complete-canvas structural
detail while retaining substantially greater horizontal sampling for the long
bridge/train relation. Both levels cover the entire destination; neither is a
failure-local region or sparse destination subset. Nearby physical positions
may appear at both scales by design.

The bounded formula is always:

```text
N_global = 16*32 + 32*112 = 4096
```

If destination H grows, only each level's deterministic area-filter support and
coordinate scale change; global model-token count does not.

All three constructions are fixed, linear, parameter-free, prompt-independent,
and decided before execution. They are not persistent sampler states in this
terminal discriminator.

## Work and integrity

Preflight estimated 12 minutes including setup/decode, below the 30-minute stop
limit. Every arm was persisted immediately as fsynced JSON plus tensor artifact
under `flux2_candidate3_bounded_global_state_results/arms/`; compatible resume
is gated by accepted H/G, every W hash, schedule, and representation hashes.

For every arm:

- generated source tokens: 4,096;
- attention: 4,608 text+W queries by 8,704 text+W+source keys;
- source execution: 25 block calls, no source final projection;
- W execution: 55 trajectories, 1,375 block consumptions, 55 final projections;
- current-block source K/V: 48 MiB;
- logical augmented-attention rows: `4,608 * 8,704 * 55 * 25`;
- all context remains device-resident; CPU K/V cache and CPU-to-GPU K/V
  transfer are zero.

Accepted H/G and all W tensors remain hash-identical. Coverage is complete,
all results are finite, and zero state updates occur.

## Measurements

| Arm | Semantic | Overlap RMS | Assembled RMS | RMS vs A | Source CUDA | Local CUDA | Wall | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A simple 32x128 | S0 | 0.784053 | 0.694729 | 0 | 1.063 s | 101.831 s | 103.154 s | 3.80 / 6.02 GiB |
| B single 64x64 | S0 | 0.788349 | 0.702653 | 0.189817 | 1.081 s | 102.639 s | 103.990 s | 3.85 / 6.06 GiB |
| C hierarchy 512+3584 | S0 | 0.780410 | 0.685993 | 0.276856 | 1.058 s | 101.131 s | 102.466 s | 3.85 / 6.11 GiB |

No matching latent from the historical positive direct-crop oracle was
recomputed, so RMS versus that oracle is unavailable. Its saved Phase-8d
decoded reference remains useful only for qualitative contrast; it used direct
32x32 consumers rather than normalized W and is not a numerical control.

## Semantic judgment

All three arms show:

- many independent suspension spans rather than one bridge system;
- duplicated towers, supports, and train-like fragments;
- disconnected decks and cables;
- multiple incompatible horizon/water bands.

B changes local alternatives but remains S0. C changes which larger fragments
dominate and lowers overlap RMS slightly, but it also remains S0. The numerical
compatibility change is not whole-scene semantic improvement and cannot pass
the gate.

## Decision

Gate 1 applies. The tested destination-independent 4K single-scale and explicit
hierarchical states do not carry enough usable whole-scene information through
the frozen generated-K/V consumer contract to organize normalized local working
canvases. There is no Phase 19b trajectory.

The missing information is scene-level relational identity: which bridge deck,
towers, supports, train, horizon, and water belong to one shared structure.
Providing both coarse and medium filtered views does not consolidate those
relations. Further fixed-grid reallocations, deterministic packing, or
interface-depth schedules are not justified. The Blueprint architecture must
be reassessed at the global representation/state level rather than promoted on
overlap or runtime metrics.

**BOUNDED GLOBAL STATE REJECTED**
