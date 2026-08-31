# FLUX.2 Candidate-3 terminal-release generalization qualification

## Verdict

Terminal release partially generalizes to the second composition-stress case.

The fixed lifecycle again preserves the major low-density plan and gives local
high-resolution authority at terminal decode without restoring the tiled-only
alternative scene. It produces one centered person, the requested dominant red
car on the left, the requested dominant tree on the right, a coherent ground
plane, and substantially sharper detail than the all-step hard anchor.

It does not satisfy exact secondary-object uniqueness: a small second red car
and two thin extra trees remain. Those extras are already present in D's
bit-identical terminal proposal and survive D's hard projection, so terminal
release does not cause their return. The fixed intermediate coarse-consistency
policy is insufficient to remove them in this prompt.

## Fixed setup and changed variables

Implementation and outputs:

- `flux2_candidate3_hard_global_anchor.py`
- `flux2_candidate3_terminal_release_generalization_results/report.json`
- `flux2_candidate3_terminal_release_generalization_results/FINAL_COMPARISON.png`
- `flux2_candidate3_terminal_release_generalization_results/TERMINAL_RELEASE_COMPARISON.png`

Only prompt and seed changed:

```text
seed: 20260831

A cinematic wide-angle photograph of exactly one large full-body woman
standing centered in the foreground, occupying most of the image height;
exactly one red vintage car parked on the far left; exactly one tall green
tree on the far right; asymmetric left-center-right composition, continuous
dry ground plane, coherent perspective and scale, distant low hills, sunset
light, no duplicate people, no duplicate cars, no duplicate trees
```

Command:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate3_hard_global_anchor.py --output-dir experiments\flux2_candidate3_terminal_release_generalization_results --seed 20260831 --prompt "<prompt above>"
```

Model, 1024 x 512 target, 512 x 256 global branch, mapped-noise
initialization, `D/U`, four Euler intervals, sigmas, three 512 x 512 crops,
128-pixel overlap, normalized weights, coordinates, coupling strength, and
terminal-release lifecycle are unchanged.

## Integrity checks

- C.H equaled B.H bit-exactly at every accepted state.
- D and E had bit-exact H inputs, model predictions, and Euler proposals at all
  four evaluations.
- D/E accepted H was bit-exact after intervals 0, 1, and 2.
- D/E accepted G was bit-exact after every interval.
- Only final H acceptance differed (`RMS 0.344136`, max `3.086852`).
- All nonterminal D/E `D(H)=G` invariants passed with maximum error at most
  `2.38419e-7`.
- E intentionally ended with `RMS(D(H_final)-G_final)=0.344136` and maximum
  mismatch `3.086852`.
- Every Candidate-3 evaluation retained one global plus three local forwards,
  one atomic pair acceptance, and zero crop state updates.

Mapped-noise global variance was 0.250565 of H noise variance. The decoded G
trajectory was independently coherent: one person near center, one car left,
one tree right, and a continuous ground plane.

## Numerical measurements

### Final latent versus dense

| Variant | RMS | low-frequency RMS |
|---|---:|---:|
| tiled-only | 0.756690 | 0.444719 |
| hard global anchor | 0.755579 | 0.449037 |
| terminal release | **0.676254** | **0.354860** |

Terminal release improves RMS by 10.5% versus hard anchoring and 10.6% versus
tiled-only. It improves low-frequency RMS by 21.0% versus hard anchoring and
20.2% versus tiled-only.

### Terminal lifecycle

| Measurement | Value |
|---|---:|
| tiled-only pre-blend overlap RMS | 0.261909 |
| D/E pre-blend overlap RMS | 0.248414 |
| hard terminal projection RMS | 0.344136 |
| hard projection / proposed-H RMS | 36.87% |
| E final coarse mismatch RMS | 0.344136 |
| E versus D final low-frequency RMS | 0.258904 |

D and E share terminal overlap exactly because their crop predictions and
proposal are identical. The final numerical and visual differences arise only
from accepting or omitting the terminal projection.

## Semantic observations

### A dense

- one centered woman;
- one red car on the left;
- one large tree on the right;
- correct asymmetric scale and continuous ground perspective;
- no meaningful duplicate subject.

### B tiled-only and C uncoupled

- two people occupying separate crop-local center hypotheses;
- two red cars, one at each side;
- two large trees in different regions;
- locally plausible ground and perspective, but an incompatible combined
  object-count/layout plan.

### D hard global anchor

- one main centered person, but face/body detail is visibly doubled;
- one dominant left car plus a small second background car;
- one dominant right tree plus two thin extra trees near center-left;
- strong car/person/tree edge ghosting from terminal projection;
- continuous ground and horizon.

### E terminal release

- one sharp, centered person at a coherent scale;
- one dominant sharp red car in the requested left position;
- one dominant detailed tree in the requested right position;
- continuous ground plane, horizon, and coherent perspective;
- no return of tiled-only's second person or second large competing tree;
- a small second red car remains in the background;
- two thin extra trees remain near center-left;
- substantially less person, car, tree, and ground ghosting than D.

## Interpretation

The lifecycle-specific Phase-3b result generalizes in its central causal claim:
intermediate hard coupling can organize later local execution, and final local
authority can remove projection damage without immediately restoring the full
tiled-only alternative scene. The improvement is both visible and numerical
in a non-architectural, asymmetric subject/object scene.

Generalization is only partial because exact secondary-object count is not
preserved. Since D and E share the same terminal proposal and D also retains
the extras, the remaining failure is not caused by terminal release. It shows
that fixed `2x2` coarse state equality does not fully exclude small semantic
alternatives from H even across hard-coupled intermediate states.

Do not proceed automatically to production or soft anchoring. This result
qualifies terminal release as preferable to terminal projection in two cases,
but leaves Candidate 3's object-uniqueness contract unresolved.

**TERMINAL RELEASE PARTIALLY GENERALIZES**
