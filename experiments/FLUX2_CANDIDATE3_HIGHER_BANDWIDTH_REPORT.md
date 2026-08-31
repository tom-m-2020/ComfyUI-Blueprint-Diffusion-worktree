# FLUX.2 Candidate-3 higher-bandwidth global-state discriminator

## Verdict

The selected 24x48 block-DCT global state resolves the Phase-3c secondary
duplication in this controlled case.

Terminal release produces exactly one central person, one dominant red car on
the left, and one dominant tree on the right. The small second car and two thin
extra trees from the 16x32 Phase-3c trajectory are absent. Perspective, ground,
horizon, and local person/car/tree detail remain coherent. No decoded
`4x4 H / 3x3 G` macroblock artifacts are visible in the sky, smooth ground,
horizon, tree boundary, or large gradients.

The result qualifies the complete selected geometry/operator contract. It does
not isolate token density from the mapped-noise distribution change: the
constant-preserving block-DCT restriction raises initial G variance from the
old approximately 0.25 to 0.563112, as predicted.

## Fixed experiment

Implementation and raw outputs:

- `flux2_candidate3_hard_global_anchor.py`
- `flux2_candidate3_higher_bandwidth_results/report.json`
- `flux2_candidate3_higher_bandwidth_results/FINAL_COMPARISON.png`
- `flux2_candidate3_higher_bandwidth_results/TERMINAL_RELEASE_COMPARISON.png`

Command:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate3_hard_global_anchor.py --output-dir experiments\flux2_candidate3_higher_bandwidth_results --global-geometry block_dct_24x48 --seed 20260831 --prompt "<exact Phase-3c prompt>"
```

The Phase-3c model, prompt, seed, CFG, four Euler intervals, sigma schedule,
1024 x 512 H canvas, three 512 x 512 crops, overlap weights, full-canvas crop
coordinates, hard nonterminal coupling, and terminal release are unchanged.

The only algorithmic change is the selected G representation:

```text
G: 24 x 48 = 1152 tokens
D: each 4x4 H block -> orthonormal DCT -> retain lowest 3x3
   -> inverse 3x3 DCT -> scale 3/4
U: each 3x3 G block -> orthonormal DCT -> zero-pad to 4x4
   -> inverse 4x4 DCT -> scale 4/3
```

All dual variants C/D/E use the same 24x48 mapped G. D/E use:

```text
nonterminal: G_next=G*; H_next=H*+U(G*-D(H*))
terminal:    G_next=G*; H_next=H*
```

No soft strength, alternate operator, external K/V, cache, schedule/prompt
change, geometry sweep, or production code was introduced.

## Operator and lifecycle integrity

### Synthetic right inverse before FLUX

On a random `1x3x24x48` tensor:

```text
RMS(D(U(G)) - G)      = 1.52364e-7
mean absolute error   = 1.12641e-7
maximum absolute error= 7.15256e-7
```

The model was not loaded until this assertion passed.

### Accepted-state invariants

| Step | sigma | invariant RMS | invariant max |
|---:|---:|---:|---:|
| 0 | 1.000000 | 8.47e-8 | 7.15e-7 |
| 1 | 0.961437 | 8.10e-8 | 5.96e-7 |
| 2 | 0.892594 | 7.51e-8 | 4.77e-7 |
| 3 hard control | 0.734760 | 1.12e-7 | 9.54e-7 |

Every nonterminal D/E acceptance therefore satisfies `D(H)=G` to float32
roundoff. E intentionally releases the terminal state and ends with:

```text
RMS(D(H_final)-G_final) = 0.329385
mean absolute           = 0.231368
max absolute            = 3.778430
```

### Causal controls

- C.H equals B.H bit-exactly at all four accepted states.
- C.G, D.G, and E.G are bit-exact at every accepted state.
- D/E H inputs, model predictions, and Euler proposals are bit-exact through
  the terminal proposal.
- D/E accepted H differs only at terminal acceptance.
- C/D/E each record four atomic pair acceptances, one 1152-token global plus
  three 1024-token local forwards per evaluation, and zero crop state updates.
- Actual global RoPE is `scale_y=31/23=1.34782608696` and
  `scale_x=63/47=1.34042553191`, covering y `0..31` and x `0..63`.

## Initialization

| State | RMS | standard deviation | variance ratio to H |
|---|---:|---:|---:|
| H noise | 0.999234 | 0.999236 | 1.0 |
| mapped 24x48 G | 0.749832 | 0.749834 | 0.563112 |

The measured ratio is close to the predicted `0.5625`. Adjacent correlations
remain near zero (`-0.00160` horizontal, `-0.00253` vertical). This is not
equivalent to Phase 3c's 16x32 mean-restricted initialization, whose variance
ratio was approximately 0.250565.

The decoded 24x48 G trajectory is independently valid: one centered person,
one left car, one right tree, continuous ground and horizon, coherent scale and
perspective, and no secondary duplicate car/tree structures.

## Projection telemetry

| Step | projection RMS | projection / proposed-H RMS | pre-blend overlap RMS |
|---:|---:|---:|---:|
| 0 | 0.033695 | 3.47% | 1.262135 |
| 1 | 0.033817 | 3.65% | 0.621851 |
| 2 | 0.070745 | 8.44% | 0.400574 |
| 3 | 0.329385 | 35.01% | 0.242493 |

Terminal D/E overlap is identical because their local calls and proposals are
identical. Tiled-only terminal overlap is 0.261909. Higher-bandwidth coupling
therefore reduces terminal cross-crop disagreement modestly before blending.

The hard terminal projection still damages fine structure, although its ratio
falls from 36.87% under Phase 3c's 16x32 geometry to 35.01%. Terminal release
remains necessary.

## Numerical comparison

### Final latent versus dense

| Variant | RMS | low-frequency RMS |
|---|---:|---:|
| B tiled-only | 0.756690 | 0.444719 |
| D hard 24x48 | 0.625397 | 0.273259 |
| E terminal release 24x48 | **0.549170** | **0.245447** |

The earlier 16x32 Phase-3c terminal release measured RMS `0.676254` and
low-frequency RMS `0.354860`. The 24x48 result improves those by 18.79% and
30.83%, respectively. These are numerical comparisons under the full selected
operator contract, not a claim that token density alone caused the change.

## Semantic and artifact inspection

### A dense

One central person, one left red car, one right tree, continuous ground, and no
meaningful duplicate objects.

### B tiled-only and C uncoupled

Two people, two cars, and two large trees form incompatible crop-local scene
alternatives. B/C remain bit-exact.

### D hard 24x48

The secondary car and thin extra trees are removed, leaving the requested three
primary subjects. The terminal projection visibly doubles/smears person and car
edges, confirming that higher bandwidth does not remove the need for terminal
release.

### E terminal release 24x48

- one centered person, with clean face/body/clothing detail;
- one dominant red car on the left, with no second background car;
- one dominant tree on the right, with no thin extra trees;
- coherent relative scale and perspective;
- continuous horizon and ground plane;
- no crop-local alternative scene or visible seam;
- sharp local detail retained.

### Block-DCT artifact check

At decoded resolution, no repeating 4-cell boundary pattern, 3-cell global
pattern, staircase, checkerboard, or block discontinuity is visible in:

- the smooth sky gradient;
- horizontal mountain/horizon contours;
- the dry ground texture and illumination gradient;
- the outer tree silhouette or internal foliage;
- person and car boundaries.

This does not prove artifacts cannot occur in other prompts or under numerical
analysis, but operator artifacts do not dominate this result.

## Runtime and CUDA allocation

| Variant | seconds | peak allocated GiB |
|---|---:|---:|
| A dense | 9.233 | 2.6011 |
| B tiled-only | 3.149 | 2.4951 |
| C uncoupled 24x48 | 4.419 | 2.5135 |
| D hard 24x48 | 4.427 | 2.5135 |
| E release 24x48 | 4.419 | 2.5135 |

For the same prompt/seed, Phase 3c's 16x32 E took 3.898 seconds with 2.4987
GiB peak allocation. The 24x48 E is 13.35% slower and adds approximately
0.0148 GiB peak allocation in this run. Timing remains secondary and is not a
formal benchmark; peak allocation is not total residency.

Each 24x48 Candidate-3 evaluation executes 4224 generated tokens versus 3584
for 16x32. Semantic success therefore comes with the predicted higher global
work and does not establish a total-compute advantage over dense attention.

## Conclusion

The complete 24x48 block-DCT Candidate-3 geometry resolves the specific
secondary duplication that remained under 16x32 mean/nearest coupling, while
preserving the terminal-release detail benefit and avoiding visible
operator-specific macroblocking in this scene. It becomes the leading
Candidate-3 research geometry for this model/setup.

The evidence does not separate greater spatial bandwidth from the changed
mapped-noise variance, does not generalize beyond the tested scenes, and does
not justify production implementation or a geometry sweep.

**HIGHER BANDWIDTH RESOLVES SECONDARY DUPLICATION**
