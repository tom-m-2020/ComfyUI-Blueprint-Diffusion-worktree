# Phase 9d — native-working-coordinate discriminator

## Question and fixed state

This experiment tests whether Phase-9b/9c semantic repetition comes primarily
from presenting a magnified `64x64` working state as dense samples inside the
original `32x32` destination extent. It is a terminal-only, zero-update probe.
Production and ComfyUI core are unchanged.

All variants reuse the exact Phase-9b C preterminal state:

- FLUX.2 Klein 4B, bridge/train prompt, seed `20260901`, CFG 1;
- `H=64x128`, fixed `G=24x48`, terminal sigma `0.6780259013`;
- fifteen `32x32` destination regions at stride 24;
- identical sigma-consistent `32->64` W tensors;
- identical 2x2-mean x0 restriction and normalized overlap assembly.

The accepted H/G and all W tensors are hash-checked after every variant.
`max_abs(avgpool2(W)-H_crop)` is below `1e-6`.

## Coordinate variants

### A — compressed destination coordinates

The exact Phase-9c A control. Each `64x64` W uses destination-coordinate
spacing `31/63` and is shifted to its destination crop origin.

### B — native local coordinates

The identical W uses ordinary unit-spaced local coordinates `0..63` on both
axes. These coordinates describe a native local model canvas, not 64 target
positions.

### C — native local coordinates plus remapped full-H context

The local W uses the same `0..63` coordinate frame as B. The complete accepted
H is mapped into each crop's working frame by

`working_coordinate = 2 * (H_coordinate - destination_crop_origin)`.

Accordingly, source RoPE uses scale 2 and shifts `(-2*y, -2*x)`. For example,
crop 14 at destination origin `(32,96)` sees full-H source coordinates
`y=-64..62`, `x=-192..62`, while its local W remains `0..63` in both axes.
This keeps source and local K in one coordinate frame.

Because positioned source K differs by crop, C performs one same-sigma full-H
source capture for each of the fifteen crops. This deliberately expensive
construction is a causal upper bound, not an implementation proposal.

## Results

| Variant | Overlap RMS | Assembled RMS | Source CUDA | Local CUDA | Wall | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|
| A compressed | 0.214053 | 0.856812 | 0.000 s | 15.124 s | 15.271 s | 2.890 / 3.353 GiB |
| B native local | 0.234165 | 0.890278 | 0.000 s | 15.188 s | 15.307 s | 2.890 / 3.353 GiB |
| C native + remapped H | 0.191253 | 0.876773 | 50.172 s | 38.146 s | 88.911 s | 3.688 / 4.392 GiB |

Pairwise assembled differences:

| Pair | RMS | Max absolute |
|---|---:|---:|
| A vs B | 0.316471 | 2.330714 |
| A vs C | 0.294763 | 2.694888 |
| B vs C | 0.134098 | 1.105765 |

Representative `x0_W` RMS/max differences are `0.439962/2.812748` for A/B,
`0.415155/2.685618` for A/C, and `0.207882/1.750684` for B/C. The coordinate
change therefore materially changes model predictions; this is not a no-op.

C consumes 8,192 context tokens through all 25 blocks for each crop. Its
sequential per-crop cache is 2.344 GiB; cumulative diagnostic host transfer is
35.156 GiB. These figures are not claimed as practical resource behavior.

## Semantic inspection

- A reproduces the Phase-9c compressed-coordinate control exactly, including
  its repeated bridge/train alternatives.
- B does not materially improve object or bridge uniqueness. It increases
  overlap disagreement by 9.4%, introduces greater crop-relative displacement,
  and retains repeated bridge/train/support interpretations.
- C reduces B's disagreement and makes the long deck somewhat more compatible,
  but repeated train/bridge structures remain clearly visible. It does not
  produce a single unambiguous bridge system.

Native coordinates are therefore neither sufficient alone nor sufficient when
paired with a correctly remapped full-H context upper bound. The full-H context
does help cross-region compatibility, consistent with Phase 9c, but it does
not pass the semantic gate.

This weakens compressed positional geometry as the primary Phase-9 failure
source. It does not prove that all positional choices are irrelevant; it shows
that the specifically requested genuine native local frame does not repair the
fixed W/x0 transport contract. The next causal discriminator should isolate
prediction restriction/transport. Increasing G density or adding more context
is not justified by this result.

All restricted crop predictions, representative `x0_W`, assembled terminal
outputs, coordinate endpoint records, and hashes are under
`flux2_candidate3_native_working_coordinate_results/`.

## Verdict

**NATIVE-WORKING POSITIONAL SEMANTICS DO NOT RESOLVE REPETITION**

No full trajectory and no production change are justified.
