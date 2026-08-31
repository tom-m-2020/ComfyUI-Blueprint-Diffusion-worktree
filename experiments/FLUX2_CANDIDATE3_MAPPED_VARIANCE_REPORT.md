# FLUX.2 Candidate-3 mapped-noise-variance discriminator

## Verdict

Both mapped-state statistics and spatial/operator bandwidth contribute, with
the stronger remaining evidence favoring bandwidth/operator content.

Variance matching the 16x32 branch materially improves numerical agreement and
suppresses some of the Phase-3c secondary structure, but it does not reproduce
the clean 24x48 result. The variance-matched final retains one thin extra tree
and a small ambiguous orange vehicle-like remnant near the horizon. Only the
24x48 block-DCT branch cleanly yields one person, one car, and one tree.

## Controlled setup

Implementation and outputs:

- `flux2_candidate3_mapped_variance_discriminator.py`
- `flux2_candidate3_mapped_variance_results/report.json`
- `flux2_candidate3_mapped_variance_results/FINAL_COMPARISON.png`
- `flux2_candidate3_mapped_variance_results/GLOBAL_COMPARISON.png`

The experiment retained the exact Phase-3c prompt, seed `20260831`, FLUX.2
model, CFG, four-step Euler sigmas, H geometry, crops, overlap, hard
nonterminal coupling, full-canvas coordinates, and terminal release.

One dense run was used only as the requested numerical reference. Exactly
three Candidate-3 variants were compared:

```text
A ORIGINAL 16x32
  D = avg_pool2d(H, 2)
  U = nearest_2x(G)

B VARIANCE-MATCHED 16x32
  D = 1.5 * avg_pool2d(H, 2)
  U = (2/3) * nearest_2x(G)

C BLOCK-DCT 24x48
  selected 4x4 -> 3x3 DCT pair
```

No other scaling, geometry, schedule, prompt, coupling, K/V, or production
mechanism was added.

## Right-inverse and lifecycle integrity

Synthetic random-tensor results before model execution:

| Variant | RMS `D(U(G))-G` | maximum absolute error |
|---|---:|---:|
| A original | 0 | 0 |
| B variance-matched | 4.62e-8 | 2.38e-7 |
| C block-DCT | 1.57e-7 | 9.54e-7 |

All three trajectories recorded exactly four atomic pair acceptances, one
global plus three local forwards per evaluation, zero crop updates, and shared
sigma/conditioning. Every nonterminal accepted-state invariant passed:

```text
A maximum error: 2.38e-7
B maximum error: 4.77e-7
C maximum error: 7.15e-7
```

Terminal release intentionally leaves final coarse mismatch:

| Variant | RMS `D(H_final)-G_final` | maximum absolute error |
|---|---:|---:|
| A | 0.344136 | 3.086852 |
| B | 0.517389 | 3.830222 |
| C | 0.329385 | 3.778430 |

The larger B mismatch follows its scaled state convention and is not an
invariant failure; no later evaluation consumes the released terminal state.

## Initialization

| Variant | G tokens | measured variance ratio | predicted |
|---|---:|---:|---:|
| A original 16x32 | 512 | 0.250565 | 0.25 |
| B scaled 16x32 | 512 | 0.563771 | 0.5625 |
| C block-DCT 24x48 | 1152 | 0.563112 | 0.5625 |

B and C therefore match mapped-G variance closely while retaining different
spatial density/operator content. All mapped initial states are finite and all
three decoded global trajectories are valid: each independently contains one
centered person, one left car, one right tree, and coherent ground/perspective.

## Numerical results

### Final latent versus dense

| Variant | RMS | low-frequency RMS |
|---|---:|---:|
| A original 16x32 | 0.676254 | 0.354860 |
| B scaled 16x32 | 0.645287 | 0.313597 |
| C block-DCT 24x48 | **0.549170** | **0.245447** |

Matching variance at fixed 16x32 improves RMS by 4.58% and low-frequency RMS
by 11.63% versus A. C then improves a further 14.90% and 21.73% versus B.
Mapped-state scale therefore matters, but it does not account for most of C's
remaining numerical advantage.

### Terminal overlap disagreement

| Variant | aggregate RMS |
|---|---:|
| A original 16x32 | 0.248414 |
| B scaled 16x32 | 0.288335 |
| C block-DCT 24x48 | **0.242493** |

B's global-state scaling improves dense-reference error while worsening
cross-crop agreement. C is the only candidate that improves both the semantic
result and terminal overlap relative to A.

## Projection telemetry

Projection is available at every step, but terminal release applies zero at
step 3. Ratios below use the available correction, including terminal.

| Step | A RMS / H* | B RMS / H* | C RMS / H* |
|---:|---:|---:|---:|
| 0 | 0.035395 / 3.65% | 0.028318 / 2.92% | 0.033695 / 3.47% |
| 1 | 0.031219 / 3.37% | 0.029072 / 3.14% | 0.033817 / 3.65% |
| 2 | 0.069132 / 8.26% | 0.069149 / 8.28% | 0.070745 / 8.44% |
| 3 | 0.344136 / 36.87% | 0.344926 / 37.96% | 0.329385 / 35.01% |

Variance matching reduces the first two corrections but provides no terminal
advantage. C has the smallest terminal relative correction.

## Semantic observations

### A original 16x32

- one main centered person;
- one dominant left car plus a clearly visible small second red car;
- one dominant right tree plus two thin extra trees;
- coherent perspective and local detail otherwise.

### B variance-matched 16x32

- one centered person and one dominant left car;
- one dominant right tree;
- one thin extra tree remains near center-left;
- a very small orange vehicle-like remnant remains at the distant horizon;
- fewer/less severe extras than A, but exact secondary uniqueness is not met;
- no broad instability or invalid global decode.

### C block-DCT 24x48

- exactly one centered person;
- exactly one left red car;
- exactly one right tree;
- no secondary car/tree remnants;
- best perspective, overlap consistency, and dense-reference agreement;
- no visible macroblock artifacts.

## Runtime and CUDA allocation

| Variant | seconds | peak allocated GiB | generated tokens/evaluation |
|---|---:|---:|---:|
| A original 16x32 | 3.906 | 2.4987 | 3584 |
| B scaled 16x32 | 3.909 | 2.4987 | 3584 |
| C block-DCT 24x48 | 4.419 | 2.5135 | 4224 |

As expected, scale matching alone has negligible runtime/VRAM effect. C costs
more because it executes the higher-density global model branch. These are
run-specific telemetry rather than general performance claims.

## Interpretation

The result matches the specified mixed interpretation:

```text
16x32 variance-matched improves but does not fully resolve
-> both bandwidth/operator content and mapped-state statistics contribute
```

The variance change explains part of Phase 3d's numerical and semantic gain,
but not the clean removal of all secondary objects. Since B and C have nearly
identical initial variance while only C fully resolves duplication and improves
overlap, Phase 3d retains strong evidence that additional spatial bandwidth or
the block-DCT retained content matters.

This experiment does not separate raw token density from DCT operator content.
Do not continue automatically to another scale or geometry sweep.

