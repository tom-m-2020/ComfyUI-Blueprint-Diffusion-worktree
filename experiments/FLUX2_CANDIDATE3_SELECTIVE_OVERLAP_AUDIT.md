# Phase 6e — Candidate-3 selective overlap execution audit

Date: 2026-09-01

## Verdict

**FAIL.** Cross-crop reuse can mechanically omit genuine generated-token work,
but the reused overlap K/V ceases to represent the current crop after the first
block. The zero-update probe skipped 25% of crop-B generated-token local work,
yet active prediction error reached RMS `0.7050`, max `6.7662`, with a severe
decoded structural failure. Reconstructing crop-B-equivalent overlap context
would require propagating those overlap hidden states through the crop-B block
stack—the work this approach is intended to skip.

No Candidate-3 production or ComfyUI-core file was modified.

## Evidence and scope

Inspected current sources:

- ComfyUI `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`, including
  `comfy/ldm/flux/model.py` and `layers.py`;
- Blueprint production coordinator, FLUX.2 adapter, crop planner, and assembler;
- Phase 6a–6d reports/harnesses;
- local SpotEdit audit and its native sparse executor. The imported experiment
  executor revision is `664b4dd2b809fc7f854504dbb2ab777c1690a97f`.

SpotEdit previously demonstrated the mechanical ability to gather raw active
generated tokens before `img_in`, propagate reduced hidden/query rows through
double and single blocks, attend to full cached K/V, run final projection only
for active tokens, and scatter results. That is reusable executor evidence. It
is not Blueprint correctness evidence: SpotEdit's inactive tokens intentionally
reuse an accepted source/edit trajectory, while two Blueprint crops provide
different spatial contexts for the same overlap positions.

## A. Exact local-crop execution audit

The current production call is
`Flux2Adapter.predict_region -> CFGGuider -> BaseModel/CONST -> Flux._forward`.
Each 32x32 latent crop becomes 1,024 generated image tokens. The adapter adds
absolute `shift_y/shift_x`; text is padded to 512 tokens. FLUX.2 Klein 4B runs
five double-stream blocks, then twenty single-stream blocks, then a 128-channel
final projection and crop-shaped unpack.

| Stage | Classification | Selective requirement / blocker |
|---|---|---|
| Latent crop and patchification | 1, trivially sliceable | `patch_size=1`; raw row-major tokens and IDs can be gathered by active index before projection. Skipped overlap raw tokens are still needed indirectly as context. |
| `img_in` | 1 for active rows; 2 for inactive context | Run only active raw tokens. Inactive tokens need layer-0 K/V derived from their own `img_in`; same-H/same-sigma cross-crop overlap provides an exact initial value. |
| Timestep/vector/global modulation | 1 | Per-evaluation/batch vectors are spatially shared. Their small fixed work is not reduced by image selection. |
| Text preparation | 4 for skipping; full execution retained | Text hidden is updated by joint attention in every double block and depends on the crop image context. The probe recomputes all 512 text rows. Cross-crop text hidden/K/V cannot be assumed reusable. |
| Double-stream image Q/K/V | 2 with valid per-layer full context | Active Q/K/V are projected for E rows. Inactive K/V must correspond to the same crop-B hidden entering that block. Crop-A overlap K/V is exact only at block 0. |
| Double attention | 2 | Query PE is gathered from active absolute IDs; full key PE uses all crop-B IDs. Rectangular Q `(512+E)` versus K `1536` is supported. Full K/V rows remain required. |
| Double image projection, MLP, norms, modulation, residual | 3 | A specialized FLUX.2 executor can run these for E image rows. But skipped overlap hidden cannot advance to produce valid next-block crop-B K/V. |
| Double text path | 3; cannot be skipped here | All text Q/projection/MLP/residual rows remain live and jointly coupled to image context. |
| Transition to single stream | 3 | Reduced order is `[512 text, E active image]`; cached inactive image K/V remains external. Full ordinary hidden `[text,1024 image]` is not retained. |
| Single `linear1` QKV+MLP | 2/3 | Runs for `512+E` query/hidden rows. Inactive image K/V comes from cache; its inactive MLP/residual is skipped, causing context invalidity at later layers if cache came from another crop. |
| Single attention | 2 | Active/text queries attend to full 1,536 K/V rows with independent query/key RoPE. K/V length is not reduced. |
| Single `linear2`, gate, residual | 3 | Truly skipped for inactive image rows, but this also means their current-crop hidden evolution is unavailable. |
| `final_layer` | 1 | Text is removed and final projection runs for E active generated rows only. |
| Unpack/scatter | 1 | Active velocity/x0 tokens scatter into crop/global positions; reusing only a previous final overlap prediction does not supply evolving per-block context. |

Classification key: 1 trivially sliceable; 2 sliceable with full/current cached
context; 3 requires a specialized FLUX.2 executor; 4 cannot safely be skipped
under the tested architecture.

### What an active token needs from skipped overlap tokens

At every attention layer, each active image query requires the skipped tokens'
K/V after those skipped tokens have consumed all preceding crop-local attention,
MLP, modulation, and residual updates. K/V from crop A has correct latent,
sigma, conditioning, and absolute coordinates, but it encodes crop A's left-side
context. Crop B's same overlap positions evolve using crop B's right-side
context. Therefore cache provenance matches state and coordinates but not the
required crop-local hidden trajectory.

Reusing only final predictions supplies none of this layerwise context.
Recomputing skipped hidden states through every block would restore it, but
would also retain their norms, QKV, MLP, residual, and later K/V work and hence
is not selective execution.

## B. Redundancy and theoretical opportunity

Full accounting is in `work_accounting.json`.

| Target | Crops | Local image executions | Unique H | Redundant | Max image-linear/final reduction | Max combined query/QxK reduction | Context K/V reduction | Forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| latent 32x64 / decoded 512x1024 | 3 | 3,072 | 2,048 | 1,024 | 33.3% | 22.2% | 0% | 3 -> 3 |
| latent 64x128 / decoded 1024x2048 | 15 | 15,360 | 8,192 | 7,168 | 46.7% | 31.1% | 0% | 15 -> 15 |

The image-linear ceiling assumes every H position is fully propagated once.
Text's 512 rows are still executed for every crop, reducing the combined-row
saving. Full local context remains 1,536 K/V rows per layer per crop. Attention
QxK counts fall only through fewer query rows; these are not FLOP or speed
claims. Model-forward count is unchanged.

## C. Zero-update runtime probe

### Construction

The probe used the initial accepted Candidate-3 H at sigma 1.0, bridge/train
prompt and seed `20260829`, with zero sampler updates:

- crop A: x `0..31`;
- crop B: x `24..55`;
- shared overlap: 32x8 = 256 positions;
- crop-B active/new subset: local x `8..31`, 768 positions.

An ordinary full crop-A pass captured per-layer K/V. The experiment remapped
crop A's rightmost eight columns to crop B's leftmost eight cache slots using
the same absolute coordinates. Sparse crop B propagated only 768 generated
hidden rows, while recomputing all text rows and attending to full 1,536-token
K/V at all 25 blocks. Cache provenance was the same accepted H, sigma,
conditioning, model, and absolute positions; no stale-step or mismatched-sigma
features were used.

This genuinely skipped for 256 generated rows:

- `img_in`;
- image QKV, attention-output projection, image MLP/norm/modulation/residual in
  all five double blocks;
- single-stream `linear1` QKV+MLP, `linear2`, norm/modulation/residual in all
  twenty blocks;
- final-layer projection.

It did not skip text work, full context K/V storage/reassembly, or attention to
the full key set.

### Correctness and first divergence

| Boundary | Active hidden RMS | Max | Exact |
|---|---:|---:|---|
| double block 0 output | 0 | 0 | yes |
| double block 1 output | 0.13243 | 6.0 | no |
| double block 4 output | 1.119 | 48.0 | no |
| single block 0 output | 2.174 | 78.5 | no |
| single block 19 output | 26.79 | 1,426 | no |
| active final x0/prediction | **0.70501** | **6.7662** | no |
| 4x4-low-pass active prediction | **0.48458** | **4.7997** | no |

The cross-crop overlap cache confirms the cause. Crop-A versus crop-B overlap
K/V is bit-exact at double block 0, then diverges at double block 1: key RMS
`0.716`, max `7.89`; value RMS `2.65`, max `33.02`. Thus block 0 active output
is exact, and the first captured divergence at block 1 follows directly from
using overlap features that evolved under crop A rather than crop B.

Ordinary crop A and crop B also disagree strongly on their final predictions
for the same overlap positions (RMS `1.0762`, max `8.0313`). This is direct
evidence that final-prediction reuse is not a substitute for crop-B context.

The decoded selective diagnostic loses the right bridge support/tower and
introduces a rectangular structural discontinuity at the active/overlap
boundary. The failure is semantic and geometric, not a small numerical drift.

### Measured core timing and memory

Three post-warmup CUDA-event measurements:

| Path | Mean core CUDA time | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| ordinary full crop B | 247.58 ms | 4.144 GiB | 4.414 GiB |
| selective 768Q/full-KV | 237.74 ms | 4.603 GiB | 4.695 GiB |

The invalid selective approximation is only 4.0% faster while increasing peak
allocated memory by 0.459 GiB and peak reserved memory by 0.281 GiB. Cache
reassembly/cloning and full K/V attention remain substantial. This timing does
not qualify an optimization because the correctness gate failed.

## D. Interpretation

The SpotEdit mechanics are real: 25% of generated-token-local work was omitted,
block 0 was exact, and runtime changed. Blueprint overlap reuse fails for a
different reason than Candidate 2: a crop is not merely a set of independently
evolving absolute tokens. Its overlap hidden trajectory depends on the crop's
non-overlap context at every depth.

An exact active-query path would need crop-B-current K/V for skipped overlap
tokens at each layer. Producing those K/V requires evolving crop-B overlap
hidden through prior attention, projection, MLP, modulation, and residual
updates. That is essentially the complete repeated overlap-token computation.
Computing all hidden rows and discarding their final predictions is explicitly
not selective execution.

No further cross-crop K/V layout/density sweep is justified. A future
optimization phase should select a different measured lever rather than
productionizing this executor. No next selective-overlap runtime experiment is
recommended under the fixed Candidate-3 crop contract.

## Final verdict

**FAIL**
