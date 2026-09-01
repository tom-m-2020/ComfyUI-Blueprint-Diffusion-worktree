# Phase 6h — geometry-dependent local-window production qualification

## Decisions

**64x64 POLICY QUALIFIED FOR H=64x128**

**48x48 POLICY QUALIFIED FOR H=48x96**

A production policy change was made, limited to the two exact latent shapes:

```text
H=(64,128) -> crop 64x64, stride 48
H=(48,96)  -> crop 48x48, stride 36
otherwise  -> crop 32x32, stride 24
```

No generic largest-fitting-window rule was introduced. Candidate-3 sampling,
G geometry, D/U, hard nonterminal coupling, terminal release, Euler lifecycle,
conditioning, coordinate handling, adapter, and assembler are unchanged.

## Experiment design

The experiment compared baseline and candidate from identical prompt, seed,
noise, conditioning, sigma schedule, accepted-state lifecycle, and model state.
Every primary trajectory was recomputed once; all 28 primary/repeat final
latents were bit-exact (`RMS=0`, `max_abs=0`).

### H=64x128 scenes

1. qualified person/car/tree composition;
2. one full-width suspension bridge and train;
3. one centered astronaut whose body crosses candidate boundary x=64;
4. one giraffe centered within candidate overlap x=48..64.

### H=48x96 scenes

1. qualified person/car/tree composition;
2. one centered robot crossing candidate boundary x=48;
3. one continuous passenger train spanning the frame.

The initial boundary prompts named a “region boundary”, which FLUX rendered as
a literal vertical line. Those confounded results were discarded. The final
matrix positions subjects centrally without naming the implementation concept.

## Integrity

Every run reported success and verified:

- one global proposal per interval;
- immutable accepted H for all sequential local calls;
- one normalized assembled x0_H;
- one atomic G/H acceptance;
- the same four-sigma Euler lifecycle;
- complete positive crop coverage;
- absolute FLUX.2 crop coordinates;
- nonterminal `D(H_next)=G_next` within `7.16e-7`;
- terminal release only on interval four;
- finite output and four previews.

The global branch is structurally identical across each A/B pair: G is
initialized only from the common H noise, evolves independently through the
same model/sigma/coordinate path, and is never derived from crop layout.

## Work and performance

Values are means across the four H=64x128 scenes and three H=48x96 scenes.
Each geometry was warmed before measurement. CUDA events and sampling-only wall
timing follow Phase 6a/6g. Token counts are not FLOP claims.

| H | Policy | Crops/int. | Local tokens/int. | Redundancy | Total forwards | Global CUDA | Local CUDA | Sample wall | Peak alloc/reserved |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64x128 | 32/24 | 15 | 15,360 | 1.875x | 64 | 4.677 s | 15.660 s | 20.543 s | 5,510 / 6,069 MiB |
| 64x128 | 64/48 | 3 | 12,288 | 1.500x | 16 | 4.699 s | 12.190 s | 17.015 s | 5,509 / 6,069 MiB |
| 48x96 | 32/24 | 8 | 8,192 | 1.778x | 36 | 2.435 s | 8.418 s | 10.971 s | 5,209 / 5,458 MiB |
| 48x96 | 48/36 | 3 | 6,912 | 1.500x | 16 | 2.442 s | 6.400 s | 8.931 s | 5,209 / 5,458 MiB |

At H=64x128 the candidate reduces mean local CUDA time 22.2% and wall time
17.2%. At H=48x96 it reduces local CUDA time 24.0% and wall time 18.6%.
Per-scene wall gains were stable: 16.9-17.4% and 18.3-18.9%, respectively.
Peak allocated differs by about 1 MiB or less and peak reserved is unchanged.

## Boundary telemetry

Candidate terminal pre-blend overlap RMS was lower in every scene. Final
adjacent-strip RMS was sometimes higher, especially on centered subjects:

| Scene | Baseline final strip | Candidate final strip | Stressed coordinate baseline -> candidate |
|---|---:|---:|---|
| H64 person/car/tree | 0.8382 | 0.9580 | x64: 1.0232 -> 1.0502 |
| H64 bridge/train | 0.9258 | 0.8785 | x48: 0.7955 -> 0.8600; x64: 0.8549 -> 0.8967 |
| H64 centered astronaut | 0.9077 | 1.0818 | x64: 1.1260 -> 1.1658 |
| H64 overlap giraffe | 0.9360 | 1.0129 | x48: 1.0120 -> 1.0499 |
| H48 person/car/tree | 0.9199 | 0.9769 | x48: 1.0431 -> 1.0706 |
| H48 centered robot | 0.9064 | 1.1165 | x48: 1.1725 -> 1.1914 |
| H48 full-width train | 0.9268 | 0.9968 | x48: 0.9581 -> 0.9895 |

These increases did not correspond to visible seams, displacement, missing
parts, or semantic duplication in the decoded comparisons. As specified,
adjacent-strip RMS was treated as diagnostic rather than the perceptual gate.

## Semantic inspection

The H=64x128 candidate preserves one centered person, left car, right tree,
continuous terrain, and local detail. It preserves the full bridge deck,
cables, towers, central train, horizon, and water. The centered astronaut has
one intact head, torso, two arms, and two legs across x=64. The overlap-centered
giraffe has one continuous neck/body and four coherent legs without duplication.

The H=48x96 candidate preserves the person/car/tree scene, the centered robot's
head/torso/limbs across x=48, and one continuous multi-car train and railway
across all local regions. No texture seam, ground/horizon break, crop-coordinate
shift, missing object part, or material sharpness loss was visible.

Fourteen decoded A/B images and machine telemetry are under
`experiments/flux2_candidate3_geometry_window_qualification_results/`.

## Production implementation and tests

Only `FixedCropPlanner` changed. It contains two explicit qualified mappings;
`Region` already represented variable height/width and `OverlapAssembler`
already assembled sized regions. Two unit tests cover exact mappings, coverage,
normalized constant assembly, and fallback behavior. All 17 production tests
pass.

A fresh ComfyUI 0.33.0 process loaded the junctioned production custom node.
Both exact geometries completed through `BasicGuider`, four-step scheduler,
`SamplerCustomAdvanced`, four real preview callbacks, normal VAE decode, and
save. Repeated queue executions returned identical decoded RGB hashes; the
second executions were normal ComfyUI cache hits. Machine evidence is in
`experiments/live_comfyui_candidate3_geometry_window_results/report.json`.

## Conclusion

Both exact geometry policies meet the semantic, lifecycle, deterministic, and
material wall-time gates. The production planner now selects them only for
their qualified latent shapes and retains 32x32/stride24 everywhere else.
