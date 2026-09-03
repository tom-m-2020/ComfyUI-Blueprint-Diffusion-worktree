# Phase 9b — native-resolution local magnification falsifier

## Result

Sigma-consistent local magnification behaves materially better than naïve
interpolation and provides partial evidence for the native-working-canvas
hypothesis. It preserves a coherent whole-canvas bridge and recovers useful
fine bridge/water structure. It is not qualified: repeated train/bridge
elements and structural ghosting remain, and it costs roughly four times the
local model work of direct destination-scale execution.

No production or ComfyUI-core code was changed.

## Fixed experiment

- Native FLUX.2 Klein 4B, T2I, CFG 1.
- Output 1024×2048; persistent `H=64×128`.
- Fixed global `G=24×48` using the experimental exact 8→3 block-DCT pair:
  1,152 model tokens for all variants.
- Four-step full-denoise CONST-flow Euler, seed 20260901, Phase-9 bridge/train
  prompt.
- Destination regions are 32×32, stride 24 with deterministic end alignment:
  15 crops covering H with destination redundancy 1.875×.
- Identical global branch, global budget, destination regions, overlap
  assembly, hard nonterminal coupling, terminal release and coordinates.

Variants:

1. **A destination scale:** ordinary 32×32 local execution.
2. **B naïve magnified:** bilinear 32→64 working state; 64×64 model execution;
   nonoverlapping 2×2 mean restriction of x0 back to 32×32.
3. **C sigma-consistent magnified:**

   ```text
   W = nearest2(H_crop)
       + sigma * (N - nearest2(avgpool2(N)))
   ```

   followed by the same 64×64 model call and the same x0 mean restriction as
   B. Added degrees of freedom therefore have exactly zero 2×2-cell mean.

For B/C, the 64 working positions cover the original 32-position destination
extent using endpoint-preserving coordinate scale `31/63`; they are not
presented as a physically larger crop.

## State construction controls

Across all 60 C working states, `max_abs(avgpool2(W)-H_crop)` is
`4.77e-7`. Initial working high-band RMS is 0.86576 and remains 0.58756 at the
terminal evaluation, so C supplies genuine sigma-scaled additional degrees of
freedom rather than interpolation-only content.

B does not preserve the accepted destination state under the chosen 2×2 mean
restriction: maximum coarse error reaches 2.448. Its high-band RMS is only
0.28248 initially and 0.20916 terminal. This is direct evidence that naïve
state interpolation is itself a major failure source.

## Work and performance

| Variant | G tokens | Tokens/local | Local calls | Local token executions | Global CUDA | Local CUDA | Wall |
|---|---:|---:|---:|---:|---:|---:|---:|
| A direct | 1,152 | 1,024 | 60 | 61,440 | 1.085 s | 15.512 s | 16.843 s |
| B naïve 64 | 1,152 | 4,096 | 60 | 245,760 | 1.031 s | 59.468 s | 61.032 s |
| C sigma-consistent 64 | 1,152 | 4,096 | 60 | 245,760 | 1.060 s | 59.613 s | 61.248 s |

The architectural token budgets are fixed: G is always 1,152 tokens and a
magnified local forward always 4,096 tokens. Increasing destination H would
increase crop count, not either per-forward budget. This experiment does not
claim a compute saving: native magnification deliberately spends 4× the local
token executions and measures 3.64× A's wall time.

| Variant | Peak allocated | Peak reserved |
|---|---:|---:|
| A | 2.530 GiB | 2.705 GiB |
| B | 2.839 GiB | 3.252 GiB |
| C | 2.844 GiB | 3.252 GiB |

## Lifecycle and boundaries

All outputs are finite, coverage is complete, and all nonterminal `D(H)=G`
errors are at most `3.58e-7`. Terminal overlap RMS is 0.25894 / 0.21616 /
0.21405 for A/B/C. Lower overlap disagreement alone is not treated as a pass;
B demonstrates that blur can lower it.

## Semantic inspection

- **A** gives the clearest destination result: a continuous bridge and horizon,
  detailed truss/cables, shores and water. Some prompt-count ambiguity remains
  in small yellow train-like elements.
- **B** is strongly low-pass and horizontally ghosted. Towers, train and cable
  structure are blurred. Naïve interpolation fails.
- **C** is substantially sharper than B. Its representative terminal x0_W
  contains detailed truss, cables, supports and water texture, and the assembled
  result retains one principal bridge across the canvas. However, multiple
  train-like yellow structures, repeated supports/cable alternatives and
  residual ghosting remain. It is useful detail, but not a clean single-scene
  qualification.

## Causal interpretation

- **Naïve state interpolation:** fails. It is not even coarse-consistent with
  the accepted destination crop under the shared restriction operator.
- **Prediction restriction:** not sufficient to explain failure. B and C share
  identical 2×2 x0 restriction, yet C recovers much more structure. Restriction
  may still limit which high-resolution improvements reach H.
- **Coordinate mapping:** not the sole failure. B and C use identical `31/63`
  endpoint coordinates and differ materially.
- **Broader native-working-canvas hypothesis:** not falsified, but not
  qualified. Sigma-consistent W demonstrates that native-resolution local
  computation can contribute destination-visible detail while preserving a
  coarse global scene; semantic uniqueness and cross-region compatibility are
  unresolved.

Artifacts and complete telemetry are under
`flux2_candidate3_native_local_magnification_results/`, including decoded W,
x0_W, restricted representative predictions, assembled interval estimates,
final outputs, and `FINAL_COMPARISON.png`.

## Verdict

**NATIVE LOCAL MAGNIFICATION PARTIALLY SUPPORTED**

No production change is justified.
