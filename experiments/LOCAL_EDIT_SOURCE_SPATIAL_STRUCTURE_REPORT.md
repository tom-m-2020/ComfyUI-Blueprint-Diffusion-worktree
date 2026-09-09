# Local Edit Source Spatial-Structure Discriminator

Date: 2026-09-09

## Classification

**SCENE-SPATIAL SOURCE SIGNAL.** Correct spatial organization of the interval-0
low-frequency source displacement is required for the strong epsilon-like
editable response. Equal-energy rearrangements substantially weaken and/or
redirect it. The experiment does not show a clean predictable relocation of
bridge structure, so it is not classified as coordinate-transfer.

## Scope and recovered references

The previous latent archives were no longer present, so the experiment reran
only deterministic O and E trajectories to recover the accepted interval-0
source delta. Their evaluation-1 decoded PNGs, and the rerun TRUE_LOW PNG, are
byte-exact to the archived source-coevolution images. Model, prompt, seed,
conditioning, sigma schedule, source encode, mask, and sampler are unchanged.

The low component uses the previously fixed operator exactly:

```text
delta = S * (x_E_after_0 - x_O_after_0)
delta_low = S * box_average_3x3_replicate(delta)
```

Every arm follows O through interval 0, adds one source-only pulse after that
proposal, performs the next ordinary dense evaluation, and receives no repeated
injection.

## Fixed transforms

The source crop is latent columns `16:48`, shape `32x32` spatially.

- TRUE_LOW: unchanged positive control.
- LEFT_RIGHT_SWAP: output source columns are input `[16:32] + [0:16]`.
- VERTICAL_FLIP: reverse all 32 source rows.
- FIXED_BLOCK_PERMUTATION: fixed `2x4` grid of `16x8` blocks, output-to-input
  permutation `[5,2,7,0,3,6,1,4]`.
- TRANSLATED_LOW: toroidal roll inside the source crop by `(dy=8, dx=16)` latent
  positions. The horizontal offset is one quarter of the 64-column canvas.

The fixed block permutation also serves as the required low-frequency
statistical surrogate. It preserves all source-crop values and per-channel
first/second-order statistics while destroying their global arrangement; no
additional surrogate was necessary.

## Mechanical invariants and normalization

For every arm, all of the following have maximum absolute error zero:

- pre-injection state versus O;
- editable state after injection versus O;
- editable pulse values;
- accepted state versus the intended hybrid;
- next model-call input versus that hybrid.

All transforms are bijections of values within the source crop. Relative to
TRUE_LOW:

- source L2 ratios differ from 1 by at most `6.9e-8`;
- maximum per-channel mean difference is at most `3.73e-9`;
- maximum per-channel RMS difference is at most `1.86e-9`.

Consequently source RMS, L2 norm, per-channel mean, and per-channel RMS are
preserved. No norm-matched arms were run. Complete 128-channel values are in
`report.json`.

## Immediate editable response

The full-epsilon editable delta RMS is `0.615694`.

| Pulse | RMS | Magnitude ratio | Cosine vs E | Projection vs E | RMS vs TRUE_LOW x0 | Cosine vs TRUE_LOW |
|---|---:|---:|---:|---:|---:|---:|
| TRUE_LOW | 0.563080 | 0.915 | 0.912 | 0.834 | 0 | 1.000 |
| LEFT_RIGHT_SWAP | 0.414064 | 0.672 | 0.650 | 0.437 | 0.392067 | 0.718 |
| VERTICAL_FLIP | 0.346936 | 0.563 | 0.297 | 0.168 | 0.555900 | 0.329 |
| FIXED_BLOCK_PERMUTATION | 0.309069 | 0.502 | 0.531 | 0.266 | 0.450943 | 0.601 |
| TRANSLATED_LOW | 0.230201 | 0.374 | 0.321 | 0.120 | 0.523972 | 0.368 |

Response does not follow pulse energy: every input norm is the same, while
aligned projection ranges from `0.834` to `0.120`. These finite source-crop
rearrangements were not constrained to preserve the full-canvas Fourier
magnitude, so this experiment isolates spatial arrangement versus value/channel
statistics, not phase versus spectral magnitude by itself.

## Coarse editable response map

The two editable sides are each divided into a fixed `2x2` grid, producing eight
cells. Each cell is `16x8` latent positions. Complete per-cell RMS/projection is
in `report.json`.

TRUE_LOW is strongly aligned throughout: per-cell projection ranges
`0.609-0.964`. Transform effects are spatially nonuniform:

- LEFT_RIGHT_SWAP falls to projection `0.105` in upper inner-left and `0.244`
  in lower inner-left, while right cells retain `0.589-0.859`.
- VERTICAL_FLIP leaves only `0.028/0.044` in the two inner-left cells and
  `0.089-0.112` in lower-right cells.
- TRANSLATED_LOW falls to `0.015/0.070` in inner-left and `0.082/0.127` in
  lower-right cells.

The response changes where spatially rearranged source content is presented,
but it does not form a clean translated/flipped copy of E's editable response.
This is evidence of layout dependence rather than a qualified coordinate-
transfer rule.

## Bridge inspection

TRUE_LOW reproduces the full-epsilon evaluation's enlarged left cable sweep,
changed deck perspective, and train/deck reorganization. LEFT_RIGHT_SWAP retains
some response but substantially retracts that sweep. VERTICAL_FLIP, block
permutation, and translation remain much closer to O's bridge scale and cable
trajectory. None produces a clean spatially relocated version of the source
bridge features.

The fixed block statistical surrogate contains exactly the same values and
channel statistics as TRUE_LOW, yet its aligned projection falls from `0.834`
to `0.266`. This directly rejects a purely coarse-statistical explanation for
the strong response.

## Interpretation and limits

- Energy dependence: falsified as a sufficient explanation.
- Spectral dependence: not isolated here; the preceding test establishes a
  low-frequency carrier, while these transforms test its spatial arrangement.
- Spatial-layout dependence: supported strongly.
- Coordinate correspondence: transformed layouts redirect response
  nonuniformly, but do not establish a predictable transform law.
- Final image quality: not qualified; final decodes are diagnostics only.

This establishes the requested spatial falsification and stops. It does not
identify transformer blocks, define a production algorithm, or authorize
optimization.

Artifacts are under `experiments/local_edit_source_spatial_structure_results/`.
No production code or registration changed.
