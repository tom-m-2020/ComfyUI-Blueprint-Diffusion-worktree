# FLUX.2 Candidate-3 hard-global-anchor falsifier

## Verdict

The hard global anchor controls composition but overconstrains detail in this
four-step case.

The mapped-noise low-density branch is independently valid: it produces a
coherent single suspension bridge, one centered train, one left lighthouse,
one right stone tower, and a continuous horizon/water field. Hard coupling
causes the high-resolution local trajectory to adopt that broad organization.
The high-resolution proposal before the terminal projection contains useful
local train, cable, deck, tower, lighthouse, and water detail.

However, exact accepted-state projection becomes progressively larger and the
terminal projection visibly doubles/smears those structures. The final hard-
anchored image retains the global layout but contains ghosted bridge cables,
deck/support edges, train detail, and lighthouse/tower structure. Exact
`D(H) = G` at every accepted state is therefore too strong as the complete
detail-preserving contract tested here.

## Controlled implementation

Implementation:

- `flux2_candidate3_hard_global_anchor.py`
- `flux2_candidate3_hard_global_anchor_results/report.json`

Command:

```powershell
C:\Users\Tom-M\miniconda3\envs\comfydev\python.exe experiments\flux2_candidate3_hard_global_anchor.py
```

The experiment reused the exact Phase-2e setup:

- FLUX.2 Klein 4B W4A8;
- prompt, seed `20260829`, CFG 1, and four-step zero-churn Euler schedule;
- sigmas `1.0, 0.961436868, 0.892594397, 0.734759688, 0.0`;
- `H: [1,128,32,64]`, corresponding to 1024 x 512;
- `G: [1,128,16,32]`, corresponding to 512 x 256;
- three `32x32` local crops at x offsets `0`, `24`, and `32`;
- the existing 128-pixel overlap ramps and normalized assembly.

Exactly four variants ran:

```text
A DENSE
B TILED-ONLY
C UNCOUPLED DUAL
D HARD GLOBAL-ANCHOR
```

No normalized noise, soft coupling, consensus, external K/V, cache, sweep, or
production infrastructure was introduced.

## State operators and coordinates

Initialization and coupling exactly followed the design note:

```text
G_0 = D(H_0)
D    = avg_pool2d(kernel_size=2, stride=2)
U    = nearest interpolate(scale_factor=2)

G* = Euler(G_i, x0_G, sigma_i, sigma_next)
H* = Euler(H_i, assembled_x0_H, sigma_i, sigma_next)

G_next = G*
H_next = H* + U(G* - D(H*))
```

The low-density branch used the same endpoint-spanning full-canvas convention
as Phase 2:

```text
y_global -> y_global * (31 / 15) = y_global * 2.0666666667
x_global -> x_global * (63 / 31) = x_global * 2.0322580645
```

Thus global coordinates `y=0..15` map to target `0..31`, and global
coordinates `x=0..31` map to target `0..63`. Global tokens are not treated as
a crop.

## Initialization qualification

The documented mapped-noise distribution shift occurred:

| State | RMS | standard deviation | horizontal correlation | vertical correlation |
|---|---:|---:|---:|---:|
| `H_0` | 0.999933 | 0.999935 | -0.00150 | -0.00215 |
| `G_0 = D(H_0)` | 0.498630 | 0.498633 | -0.00434 | -0.00853 |

The global-to-high variance ratio was `0.248667`, close to the expected 0.25.
`G_0` equaled `D(H_0)` exactly.

This shift did not invalidate the global trajectory in the tested case. The
final decoded `G` is finite, sharp at its native scale, compositionally
coherent, and prompt-consistent: it has one continuous bridge, one train, one
left lighthouse, and one right dark tower. The experiment is therefore
interpretable for coupling and does not receive the global-invalid verdict.

## Lifecycle and integrity checks

### Uncoupled control

For every accepted state, `C.H` and `B.H` were bit-exact:

| Accepted step | RMS | maximum absolute difference |
|---:|---:|---:|
| 0 | 0 | 0 |
| 1 | 0 | 0 |
| 2 | 0 | 0 |
| 3 | 0 | 0 |

The independently evolved `C.G` and `D.G` states were also bit-exact at all
four accepted steps. Executing `G` alone therefore had no hidden effect on the
high-resolution trajectory.

### Atomic acceptance

- C and D each recorded exactly four atomic `(G,H)` acceptances.
- Every Candidate-3 evaluation recorded exactly one 512-token global forward
  plus three 1024-token local forwards: four forwards and 3584 executed image
  tokens.
- All three crops read views backed by the same current accepted `H` storage.
- Crop-state update count was zero.
- One assembled high-resolution prediction and one global prediction formed
  the two Euler proposals before the single pair acceptance.
- Every global and local call within an evaluation recorded the same sigma.

### Hard invariant

After every D acceptance, `max_abs(D(H_next) - G_next)` was
`2.38419e-7`. The synthetic `D(U(G))` check was exact, and the synthetic
accepted-state invariant was within `5.96e-8`.

## Projection and proposal interaction

| Step | sigma | pre-projection RMS | projection/H* RMS | post-projection max | global/local coarse-update cosine | projection cosine vs previous |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 0.025974 | 0.02675 | 2.38e-7 | 0.84095 | — |
| 1 | 0.961437 | 0.041440 | 0.04496 | 2.38e-7 | 0.74001 | 0.199 |
| 2 | 0.892594 | 0.075922 | 0.09243 | 2.38e-7 | 0.84552 | 0.538 |
| 3 | 0.734760 | 0.349332 | 0.38374 | 2.38e-7 | 0.85687 | 0.770 |

The local and global coarse proposal updates point broadly in the same
direction (`cosine 0.74–0.86`), so they are not simply proposing opposite
scenes. They nevertheless disagree increasingly in absolute state. Projection
RMS grows at every step and reaches 38.4% of the proposed H RMS on the long
terminal Euler interval. Consecutive corrections also become more aligned,
showing a persistent systematic mismatch rather than alternating cancellation.

The branches therefore repeatedly fight in magnitude, especially at the final
acceptance, even though their update directions are positively correlated.

## Numerical comparison

Final latent comparison against dense:

| Variant | RMS | low-frequency RMS |
|---|---:|---:|
| tiled-only | 0.863924 | 0.501593 |
| hard anchor | 0.808925 | 0.476675 |

Hard anchoring is modestly closer to dense numerically. The same trend appears
by accepted step: hard-anchor RMS versus dense is equal at the first evaluation
and lower than tiled-only at the middle and final evaluations. This numerical
improvement is not sufficient to qualify the result, because the final
projection visibly damages fine structure.

Pre-blend overlap disagreement also improves late in the coupled trajectory:

| Step | tiled-only | hard anchor |
|---:|---:|---:|
| 0 | 0.849 | 0.849 |
| 1 | 0.730 | 0.739 |
| 2 | 0.548 | 0.484 |
| 3 | 0.299 | 0.258 |

## Semantic and visual observations

### Dense

The dense reference contains one coherent bridge with two principal suspension
towers, one centered train, one left lighthouse, and one right stone tower.

### Tiled-only and uncoupled dual

These are visually identical, as required. They independently invent multiple
bridge spans/towers and train segments. The bridge slope and support system
change across crop regions, and extra lighthouse/pier-like structures appear.

### Low-density G

The global trajectory resolves the requested asymmetric plan cleanly: one
continuous bridge, one centered train, one lighthouse at the left, one dark
tower at the right, and a continuous horizon. It lacks high-resolution texture
by construction but is not numerically or visibly invalid.

### Hard-anchor H

Prior hard-coupled acceptances materially reorganize the local trajectory. The
step-3 local proposal, before the terminal projection, follows the global
single-bridge plan and contains substantially more cable, deck, train, stone,
lighthouse, and water detail than native `G`.

The terminal exact projection then introduces strong doubled/ghosted edges in
the cables, deck, train, bridge supports, lighthouse, and tower. The final is
not merely an upscaled `G`, because local texture remains, but that texture is
not cleanly compatible with the imposed coarse latent state.

## Interpretation and recommendation

This experiment supplies positive evidence for persistent two-state coupling:
the low-density trajectory can causally steer later local predictions toward a
coherent whole-canvas plan, and the local model can propose useful additional
detail. It rejects the stronger rule that exact coarse latent equality should
be imposed after every Euler proposal, including the terminal proposal.

Candidate 3 itself is not rejected. The smallest next discriminator should
retain the same hard anchor for the accepted states that will be consumed by a
later model evaluation, but release the terminal `sigma_next=0` local proposal
without projecting it. That single lifecycle change would test whether the
global branch is needed to organize the trajectory while a final local model
evaluation must remain authoritative for detail. It should not be combined
with a strength sweep or another coupling rule.

**HARD GLOBAL ANCHOR CONTROLS COMPOSITION BUT OVERCONSTRAINS DETAIL**
