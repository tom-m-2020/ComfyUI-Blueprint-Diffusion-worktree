# Phase 8i — specialized FLUX block executor feasibility

## Question and boundary

This probe asks whether the exact Phase-8d full-context terminal computation
can be executed block-major while retaining only block-to-block hidden state
for 55 crops. It uses the fixed 2048×4096 bridge case, accepted H3/G3,
96×192 current-G source, 18,432 global K/V positions, all 25 blocks, crop
32/stride 24, and terminal release. Production and ComfyUI core are unchanged.

The implementation is experiment-local. It reuses the live native FLUX.2
`img_in`, text/timestep/vector preparation, five `DoubleStreamBlock` modules,
twenty `SingleStreamBlock` modules, RoPE implementation, attention patches,
and `final_layer`. It changes orchestration only: each native block returns
before the next crop starts, so its Q/K/V, modulation, attention, MLP, and
residual temporaries leave scope. No ordinary model forward is suspended.

## Staged native-equivalence gates

### Gate 1 — ordinary local crop

Crop 0 was evaluated both through ordinary native FLUX.2 and through the
explicit executor with no external context. The following boundaries are
bit-exact (`RMS=0`, `max_abs=0`):

- `img_in` and `txt_in` outputs;
- double-block outputs 0–4, both image and text streams;
- single-block outputs 0–19;
- final projected generated tokens;
- returned float32 x0 crop.

This establishes that explicit block entry/exit itself does not alter native
FLUX.2 arithmetic for the fixed case.

### Gate 2 — one crop with full current-G context

An ordinary full 96×192 current-G source captured the Phase-8d K/V contract.
The explicit source and local crop were then advanced block by block.

- every source K and V tensor is bit-exact at all 25 blocks;
- every context-conditioned local hidden output is bit-exact;
- final crop x0 is bit-exact.

The source still performs ordinary dense global interaction. Local text
queries retain ordinary attention, while generated queries see text, local
generated, and all 18,432 global generated K/V positions exactly as in Phase
8d.

### Gate 3 — all 55 crops

All 55 explicit crop predictions are bit-exact against a same-process
CPU-offloaded Phase-8d reference. The first non-bit-exact boundary is overlap
assembly because the reference predictions were stored/assembled on CPU while
the explicit predictions remained on GPU:

| Comparison | RMS | max abs |
|---|---:|---:|
| assembled x0_H vs same-process reference | 2.016e-8 | 4.768e-7 |

This is numerically negligible and follows 55/55 bit-exact transformer
predictions. Aggregate overlap RMS is 0.2812379193 versus the recorded
0.281238. Assembled x0_H RMS is 0.7098370194, matching the Phase-8d reference.

The decoded RGB hash is not the historical hash because ordinary VAE decode
OOMed after the measurement and ComfyUI retried tiled decode. Direct visual
comparison is indistinguishable, and the transformer/latent equivalence gate
does not rely on that decode hash.

## Memory ownership and residency

At every clean barrier the executor owns one `ExplicitState` per crop. No crop
state requires gradients; all are inference tensors. The container retains
only image/text hidden state, modulation inputs, position embedding, and
options needed by the next block.

| Resident category | Double blocks | Single blocks |
|---|---:|---:|
| all 55 crop hidden state | 495.0 MiB | 660.0 MiB |
| global hidden state | 111.0 MiB | 114.0 MiB |
| all source/crop positional embeddings | 101.0 MiB | 101.0 MiB |
| one source block full K/V | 216.0 MiB | 216.0 MiB |

Measured maximum CUDA allocation was 3.322 GiB and maximum reserved memory was
6.338 GiB. The first measured crop returns to the same allocated baseline as
before it entered the block; allocation does not rise with crop ordinal.
Maximum measured per-crop temporary increment was about 387 MiB.

This replaces the failed Phase-8h ownership pattern, where 46 suspended native
call frames reached 16.30 GiB before block 0 completed.

## Transfer, time, and semantic result

| Metric | CPU-offloaded reference | Explicit block-major |
|---|---:|---:|
| all-block CPU source cache | 5,662,310,400 B | 0 B |
| aggregate CPU→GPU K/V transfer | 311,427,072,000 B | 0 B |
| one-block GPU K/V | transferred repeatedly | 226,492,416 B |
| terminal source + local CUDA | ~106.39 s | 65.72 s |
| terminal source + local wall | ~106.71 s | 65.73 s |

The reference total combines its measured 11.60 s source forward with 94.79 s
local CUDA. The same-process reference local wall was 91.69 s; the table uses
the Phase-8h reference for the complete source-plus-local comparison.

The decoded explicit result preserves the qualified one-dominant-bridge
scene, continuous deck/cables, centered train, controlled towers, and coherent
horizon/water. It is visually identical to the Phase-8d full-context image.

## Conclusion

The exact semantic mechanism is executable on this device without K/V
compression, CPU hidden offload, or backend/core changes. Explicit native
block orchestration removes both the 5.27-GiB host cache and roughly 290-GiB
transfer while lowering measured source-plus-local terminal time by about
38%. The remaining persistent crop-hidden residency is well below the device
limit in this fixed case.

This is experiment evidence only. It does not authorize production
integration or generalization beyond native FLUX.2 Klein, CFG 1, this terminal
context contract, or this geometry.

**SPECIALIZED FLUX EXECUTOR QUALIFIED**
