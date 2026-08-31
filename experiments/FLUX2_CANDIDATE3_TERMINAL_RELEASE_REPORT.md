# FLUX.2 Candidate-3 terminal-release lifecycle discriminator

## Verdict

Terminal release is sufficient in this controlled four-step case.

Hard coupling through the three intermediate accepted states establishes and
preserves the low-density trajectory's single-scene organization. Accepting the
unchanged terminal local Euler proposal without coarse projection retains one
continuous bridge, one centered train, one left lighthouse, one right stone
tower, and a continuous horizon/water field while removing most of the hard
final's cable, deck, train, and support ghosting.

## Controlled change

Implementation and raw evidence:

- `flux2_candidate3_hard_global_anchor.py`
- `flux2_candidate3_terminal_release_results/report.json`
- `flux2_candidate3_terminal_release_results/TERMINAL_RELEASE_COMPARISON.png`

Command:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate3_hard_global_anchor.py --output-dir experiments\flux2_candidate3_terminal_release_results
```

The run retained A dense, B tiled-only, C uncoupled dual, and D hard global
anchor unchanged, and added exactly:

```text
E TERMINAL-RELEASE

if sigma_next > 0:
    G_next = G*
    H_next = H* + U(G* - D(H*))
else:
    G_next = G*
    H_next = H*
```

Model, prompt, seed, CFG, sigmas, mapped-noise initialization, `D/U`, crop
geometry, overlap weights, full-canvas coordinates, and Euler equations are
identical to Phase 3. No soft strength, normalized noise, consensus, alternate
operator, external K/V, cache, or schedule variation was added.

## Causal integrity

D and E were bit-exact for:

- `H` before all four evaluations;
- local model predictions at all four evaluations;
- Euler `H*` proposals at all four evaluations;
- accepted `H` after steps 0, 1, and 2;
- accepted `G` after all four intervals.

The step-3 terminal prediction and proposal therefore match exactly before D
applies its projection. Only final `H` acceptance differs:

| D versus E | Steps 0–2 maximum difference | Terminal maximum difference | Terminal RMS difference |
|---|---:|---:|---:|
| accepted `H` | 0 | 3.423378 | 0.349332 |
| accepted `G` | 0 | 0 | 0 |

Both variants recorded four atomic pair acceptances. Each evaluation used one
512-token global plus three 1024-token local forwards, the same sigma, no crop
state updates, and one assembled high-resolution prediction.

For E, all intermediate invariants retain the Phase-3 hard-anchor tolerance.
The terminal lifecycle intentionally leaves:

```text
RMS(D(H_final) - G_final)      = 0.349332
mean_abs                       = 0.254759
max_abs                        = 3.423378
```

This mismatch exactly equals the coarse correction that D applies. It is an
intentional terminal state property rather than an integrity failure.

## Required measurements

### Final comparison against dense

| Variant | RMS | low-frequency RMS |
|---|---:|---:|
| tiled-only | 0.863924 | 0.501593 |
| hard global anchor | 0.808925 | 0.476675 |
| terminal release | **0.727491** | **0.381146** |

Terminal release improves absolute RMS by 10.1% relative to hard anchoring and
15.8% relative to tiled-only. Its low-frequency RMS improves by 20.0% relative
to hard anchoring and 24.0% relative to tiled-only.

### Terminal evaluation

- D and E terminal pre-blend overlap disagreement: identical aggregate RMS
  `0.260869`.
- Available/applied D terminal projection RMS: `0.349332`, 38.37% of proposed-H
  RMS.
- E terminal projection applied: exactly zero.
- E versus D final latent: RMS `0.349332`, low-frequency RMS `0.264918`.
- D/E terminal model prediction and Euler proposal: bit-exact.

Thus the improvement cannot come from a different terminal crop execution,
overlap assembly, input state, sigma, or proposal. It is caused solely by final
acceptance policy.

## Qualitative scoring

| Criterion | Hard projected | Terminal release |
|---|---|---|
| one continuous bridge | yes, but doubled edges | yes, clean continuous span |
| one centered train | yes, visibly smeared/ghosted | yes, substantially sharper |
| left lighthouse | present, doubled/blurred | present and clean |
| right stone tower | present, edge ghosting | present and detailed |
| duplicate semantic structures | projection-like duplicates/ghosts | no returned crop-local alternative scene |
| cable sharpness | poor; multiple displaced cable traces | strong |
| train sharpness | poor | strong |
| deck/support ghosting | severe | largely removed |
| horizon/water continuity | continuous but locally smeared | continuous and detailed |

The terminal-release image is the terminal local proposal by construction. It
retains the plan established by the prior coupled trajectory and contains much
more local structure than native low-density `G`. Omitting the final projection
does not immediately restore the tiled-only multiple-span/train/lighthouse
alternatives.

## Interpretation

This is positive evidence for a lifecycle-specific Candidate-3 contract:

```text
hard coarse coupling for intermediate states consumed by later evaluations
+
local high-resolution authority for the terminal decoded state
```

The global branch is acting as a trajectory organizer, not as a mandatory
final latent constraint. Exact low/high equality remains useful while it can
affect a later local model evaluation; enforcing it after the final model call
adds no planning opportunity and damages detail in this case.

This is one prompt/seed/model/geometry result. It does not establish broad
quality, efficiency, or a production lifecycle, and it does not justify moving
automatically to soft anchoring.

**TERMINAL RELEASE SUFFICIENT**
