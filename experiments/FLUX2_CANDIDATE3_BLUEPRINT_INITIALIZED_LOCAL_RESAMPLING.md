# Phase 22 — Blueprint-initialized local resampling discriminator

## Result

**PASS.** One ordinary late-sigma FLUX refinement initialized from the coherent
native-coordinate Blueprint remains S3 and regenerates modest but measurable
destination detail. It does not return to the independent tiled S0 scene.

This was one terminal/zero-update discriminator. No production or ComfyUI-core
files changed, and no output correction, null-space mixing, K/V context,
strength, filter, or Candidate-3 projection was used.

## Predetermined resampling contract

The fixed resampling sigma was selected and persisted before inference:

```text
sigma = 0.25
W_sigma = (1 - sigma) * nearest2(x0_anchor_crop) + sigma * epsilon_region
        = 0.75 * nearest2(x0_anchor_crop) + 0.25 * epsilon_region
```

`0.25` was chosen as a conservative late-refinement point: it supplies a
nontrivial stochastic component but deliberately retains 75% denoised-anchor
authority. It was not swept.

Each `epsilon_region` is an independent deterministic standard-normal tensor
with seed:

```text
20260901 + 22,000,003 + 1,009 * region.index
```

The exact same 55 noise tensors were used in B and C. Both used identical
conditioning, `64x64` native-local coordinates, region order, working geometry,
model options, prediction restriction, and overlap assembly. Only the clean
anchor used to construct `W_sigma` differed:

- A: saved mapped Phase-20b coherent Blueprint x0; no model call.
- B: saved Phase-20b assembled ordinary S0 local prediction.
- C: saved mapped Phase-20b coherent Blueprint x0.

All anchors came from the same recorded Phase-20b evaluation. Accepted H/G/W
states were not updated or mutated.

## Measurements

| Arm | Semantic | Overlap RMS | gradient RMS | RMS vs Blueprint | low-frequency RMS vs Blueprint | CUDA | wall |
|---|---|---:|---:|---:|---:|---:|---:|
| A Blueprint | S3 | n/a | 0.134299 | 0 | 0 | 0 | 0 |
| B ordinary local | S0 | 0.199548 | 0.486297 | recorded in JSON | recorded in JSON | 55.052 s | 56.286 s |
| C Blueprint-resampled | S3 | 0.166552 | 0.188490 | 0.120775 | 0.065888 | 55.632 s | 56.881 s |

C differs from B by `0.878335` latent RMS. Its gradient RMS is approximately
40.4% above the mapped Blueprint reference, providing a detail-sensitive
numerical signal consistent with the decoded result. The gain is modest: this
is evidence that local detail can regenerate without destroying composition,
not evidence of fully restored native detail.

Peak CUDA allocation/reservation across the measured arms was 3.21/3.61 GiB.
Each arm made 55 ordinary local model calls at 4,096 image tokens per call.
Full-frame diagnostics used ComfyUI's tiled-VAE fallback; all 110 per-region
restricted predictions were also decoded and saved.

## Semantic review

- **A — S3:** one continuous bridge, one centered train, controlled endpoint
  structures, and coherent horizon/water; visibly soft.
- **B — S0:** many independent bridge decks, towers, trains, and incompatible
  water/horizon regions.
- **C — S3:** one dominant bridge and train remain; no distinct alternative
  bridge or tower system survives. Bridge/cable and train contrast are modestly
  sharper than A while the horizon and water remain continuous.

Blueprint initialization therefore changes the local model's semantic basin:
ordinary FLUX refinement at the fixed late sigma adds useful structure without
reconstructing the S0 tiled interpretation. The primary gate passes.

Stop here. The next justified task may test this same operation in a minimal
persistent/multiresolution trajectory. Phase 22 does not qualify a trajectory,
production path, or optimal resampling sigma.

## Integrity and artifacts

- B/C noise provenance matches exactly for all 55 regions.
- Anchor, Blueprint, and W input hashes remain unchanged during inference.
- Coverage is complete and positive; outputs are finite.
- Accepted-state updates: zero.
- Per-arm tensors and telemetry were fsynced before decode.

Artifacts:

- `flux2_candidate3_blueprint_initialized_local_resampling.py`
- `flux2_candidate3_blueprint_initialized_local_resampling_results/report.json`
- `flux2_candidate3_blueprint_initialized_local_resampling_results/B_ORDINARY_LOCAL.pt`
- `flux2_candidate3_blueprint_initialized_local_resampling_results/C_BLUEPRINT_RESAMPLED_LOCAL.pt`
- `flux2_candidate3_blueprint_initialized_local_resampling_results/A_B_C_COMPARISON.png`
- `flux2_candidate3_blueprint_initialized_local_resampling_results/regions/`

