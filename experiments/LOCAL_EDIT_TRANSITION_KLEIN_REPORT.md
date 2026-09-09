# Local Edit two-column transition discriminator

## Verdict

**Fail and stop scalar transition guidance.** The two-column editable-side
linear source pull preserves the exact hard lock but regresses the primary S3
gate. It changes the former seam into a wider near-black vertical barrier while
the generated side still restarts a smaller, disconnected bridge.

The optional alternate falloff was not run because the required first arm did
not move geometry meaningfully in the correct direction. No width, strength,
schedule, seed, or projection sweep is authorized by this result.

## Fixed contract

The experiment reuses the previous bridge source, Klein model, prompt, seed,
four-step sigma schedule, 1024x512 canvas, and left-75-percent hard lock. The
locked region uses the unchanged source-authoritative CONST derivative and
accepted-state restoration. The two editable columns nearest the lock have
deterministic linear weights `[1.0, 0.0]`; no transition coordinate is restored
after an Euler proposal.

Seven pre-inference tests pass:

- `T` is disjoint from `L` and contained in `M`;
- locked derivative and accepted state are bit-exact to hard-source behavior;
- editable interior derivative is bit-exact to ordinary model Euler;
- weights are deterministic and monotonic;
- transition coordinates are not source-restored;
- terminal locked latent equals `y`.

## Runtime integrity and telemetry

S1 remains exact. Every locked input/accepted RMS and maximum trajectory error
is zero, terminal locked error is zero, and the final locked latent is bit-exact
to the frozen hard-source baseline. The editable interior derivative change is
zero at every interval. The independent transition repeat is bit-exact in both
latent and decoded pixels. The first accepted-state divergence from the frozen
baseline is step 0, the intended transition-policy boundary.

| Step | sigma to sigma_next | transition d RMS | correction RMS | accepted transition RMS |
| ---: | --- | ---: | ---: | ---: |
| 0 | 1.000000 to 0.961437 | 1.286924 | 0.920788 | 0.968907 |
| 1 | 0.961437 to 0.892594 | 1.209390 | 0.862170 | 0.914156 |
| 2 | 0.892594 to 0.734760 | 1.266658 | 0.735910 | 0.806290 |
| 3 | 0.734760 to 0.000000 | 1.296072 | 0.642278 | 0.943148 |

Final latent RMS versus the frozen baseline is `0.160385`; within T it is
`0.848558`, and in the editable core it is `0.121357`. The correction therefore
substantially perturbs the transition and propagates through later full-canvas
predictions, rather than being too weak to test.

## Gates

- **S1 exact source trajectory — PASS.** All locked errors are zero.
- **S3 geometric continuity — FAIL/REGRESSION.** A dark vertical barrier forms
  at the join. The right side still contains the independently restarted small
  bridge, with no continuous deck, cable, horizon, or scale relationship.
- **S4 edit freedom — PASS.** Editable-region decoded change RMS is `0.685876`.
- **No source copying — PASS narrowly.** Most of the editable region remains
  generated, but the first transition column follows the source pull strongly.
- **No wider artifact — FAIL.** The visible dark barrier is wider and more
  objectionable than the prior hard-source seam.
- **Determinism — PASS.** Independent latent and decoded repeats are bit-exact.

The cross-boundary seam-gradient RMS increases from `0.067718` in the frozen
baseline to `0.308739`; the adjacent editable gradient reaches `0.398623`.
These metrics agree with, but do not replace, the visual failure judgment.

## Zero-diffusion VAE locality diagnostic

For each diagnostic, locked latent values remain exactly equal to the clean
source latent. Only editable columns beyond an unchanged guard are replaced by
the terminal transition latent, then the full canvas is decoded.

| Unchanged editable guard | Locked 0-16 px MAE | Locked 64-128 px MAE | Locked 256-768 px MAE |
| ---: | ---: | ---: | ---: |
| 0 columns | 0.065095 | 0.043356 | 0.035560 |
| 1 column | 0.065095 | 0.043356 | 0.035560 |
| 2 columns | 0.054598 | 0.040998 | 0.033382 |
| 4 columns | 0.043952 | 0.035830 | 0.028981 |
| 8 columns | 0.030868 | 0.024938 | 0.020348 |
| 16 columns | 0 | 0 | 0 |

The effect is not localized to a narrow decoder receptive field at the latent
boundary: editable latent changes alter decoded pixels throughout the locked
source side. Increasing the unchanged editable guard reduces the error
monotonically, and replacing no editable columns (16-column guard) returns the
identical decode. The first one-column guard is identical to guard zero because
the weight-1 transition column already ends on the clean source value. This
diagnostic establishes broad VAE decode coupling for this full-canvas case; it
does not modify the Local Edit algorithm or identify the internal decoder layer
responsible.

## Decision boundary

The tested scalar `d_model -> d_source` transition does not improve S3, so this
epsilon-like transition branch stops. The next research branch should audit a
model-visible or prediction-refresh boundary mechanism while preserving the
hard locked trajectory unchanged.

Implementation, raw telemetry, decoded arms, accepted previews, and guard
diagnostics are under `experiments/local_edit_transition_klein_results/` and
`experiments/local_edit_transition_klein.py`. No production source changed.
