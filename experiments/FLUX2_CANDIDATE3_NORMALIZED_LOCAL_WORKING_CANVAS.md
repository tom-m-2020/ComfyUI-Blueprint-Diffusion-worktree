# Phase 9 — normalized local working-canvas falsifier

## Result

The simplest fixed-budget/fixed-working-geometry formulation is not qualified.
It demonstrates the desired resource-scaling invariant and a large measured
runtime reduction, but does not satisfy the primary image criterion: the
normalized local model produces a recognizable single bridge at the final
canvas while losing most useful local detail and retaining train/structure
ghosting.

This falsifies the tested bilinear state/prediction resampling rule. It does
not falsify every possible normalized local working-canvas architecture.

## Controlled setup

- Native FLUX.2 Klein 4B, CFG 1, deterministic four-step CONST-flow Euler.
- Output 1024×2048; persistent destination `H=64×128` (8,192 tokens).
- Phase-8 bridge/train prompt, seed 20260901.
- Three deterministic destination regions, each 64×64, stride 48.
- Identical destination rectangles, normalized overlap weights, accepted-state
  lifecycle, hard nonterminal coupling, terminal release, prompt, seed and
  schedule.
- No production or ComfyUI-core changes.

Variants:

1. `A_CURRENT_DIRECT`: production 4→3 DCT, `G=48×96` (4,608 tokens), direct
   64×64 local model calls.
2. `B_FIXED_GLOBAL_DIRECT`: experimental 8→3 DCT, fixed `G=24×48` (1,152
   tokens), direct 64×64 local calls.
3. `C_FIXED_GLOBAL_NORMALIZED`: same fixed `G=24×48`; each 64×64 destination
   region is bilinearly restricted to a 32×32 working latent, evaluated using
   endpoint-preserving full-canvas coordinates, and its denoised estimate is
   bilinearly prolonged to 64×64 before ordinary assembly.

For C, the accepted H crop is the spatially corresponding Blueprint state: H
is initialized consistently from `G=D(H)` and satisfies `D(H)=G` after every
nonterminal acceptance. No untrained extra conditioning channel or Candidate-2
K/V mechanism was introduced.

## Resource invariant and work

| Variant | G tokens | Tokens/local forward | Local forwards | Local token executions | Global CUDA | Local CUDA | Sampling wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A current direct | 4,608 | 4,096 | 12 | 49,152 | 4.071 s | 11.803 s | 16.358 s |
| B fixed-G direct | 1,152 | 4,096 | 12 | 49,152 | 1.004 s | 11.845 s | 13.020 s |
| C fixed-G normalized | 1,152 | 1,024 | 12 | 12,288 | 1.025 s | 3.099 s | 4.303 s |

C holds global model tokens at 1,152 and per-local-forward tokens at 1,024.
Those budgets are independent of destination H in the tested construction;
larger H would change region count, not either per-forward budget. Token counts
are not treated as FLOP or wall-time ratios.

Relative to B, C reduces local token executions by 75%, measured local CUDA by
73.8%, and sampling wall time by 67.0%. It retains the same 15 total model
forwards (three global and twelve local). The fixed-G change alone reduces
wall time by 20.4% versus A.

| Variant | Peak allocated | Peak reserved |
|---|---:|---:|
| A | 2.885 GiB | 3.420 GiB |
| B | 2.834 GiB | 3.244 GiB |
| C | 2.530 GiB | 2.699 GiB |

## Coupling and overlap

All outputs are finite and coverage is complete. Maximum nonterminal
`D(H)-G` error is `7.15e-7` for A and `4.77e-7` or lower for B/C.

Terminal overlap RMS is:

- A: 0.14544
- B: 0.14701
- C: 0.12835

C's lower overlap disagreement is not evidence of improved detail; much of it
comes from the strong low-pass effect of restriction/prolongation.

Nonterminal projection RMS for C is `0.02584 / 0.03206 / 0.05158`, versus
`0.02980 / 0.03472 / 0.07386` for B. The normalized path does not destabilize
the exact coarse invariant.

## Semantic and trajectory inspection

A produces the strongest result: one continuous detailed bridge, centered
train, left lighthouse, right stone tower, continuous water and horizon.

B remains globally legible but the 1,152-token 8→3 global branch changes the
scene: duplicate yellow train-like structures and weaker bridge organization
appear. This is consistent with earlier evidence that aggressive DCT bounding
changes both semantic bandwidth and mapped-state statistics.

C's assembled local x0 is already visibly smeared at interval 0. The later
trajectory does not recover the discarded high spatial frequencies. Its final
image has one coarse bridge/deck and a coherent horizon, but cables, towers,
train and shore structures are horizontally ghosted and local texture is
largely absent. Consequently it does not demonstrate that fixed native-size
local working canvases can add regional fidelity to a fixed global plan.

Artifacts:

- `flux2_candidate3_normalized_local_working_canvas_results/report.json`
- `FINAL_COMPARISON.png`
- `TRAJECTORY_COMPARISON.png`
- per-variant/per-interval decoded images and JSON telemetry in the same folder

## Verdict

**NORMALIZED LOCAL WORKING CANVAS FAILS LOCAL-FIDELITY GATE**

No production change is justified. The next design, if separately authorized,
must avoid treating a bilinearly downsampled noisy latent and bilinearly
upsampled x0 as sufficient high-resolution local-state transport; repeating
this resampling formulation at other sizes is not warranted.
