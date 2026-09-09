# Local Edit Source Co-Evolution Pulse Report

Date: 2026-09-09

## Classification

**SOURCE CO-EVOLUTION LOCALIZED.** The first accepted source displacement is a
causally sufficient, isolated pulse for reproducing 100% of full-epsilon's next
editable raw-x0 change in all three cases. On the rigid bridge, most of that
immediate effect comes from the spatially distant source interior and from the
fixed low-frequency component, not from the two-column source-side boundary
halo or high-frequency residual.

This is localization of a causal prediction effect, not qualification of a
one-pulse image-generation algorithm. A single pulse persists naturally but
does not remain identical to full epsilon after later corrections are omitted.

## Definitions and integrity

Fresh ordinary O and epsilon-projection E trajectories use the exact prior
8-step deterministic Klein/RES4LYF configuration, prompts, source encodes,
masks, conditioning, seed, and no-noise-mask semantics. Every accepted state
and raw x0 from both fresh references is bit-exact to the previous epsilon
mechanism archive (`max_abs=0`) for bridge, tree, and desert.

For pulse interval `i`:

```text
delta_S_i = S * (x_E_after_i - x_O_after_i)
x_pulse_after_i = M * x_O_after_i + S * x_E_after_i
```

The pulse then runs ordinary sampling with no further injection. Independent
trajectories cover intervals 0 through 6; interval 7 has no following model
evaluation. Across every pulse and case:

- pre-injection state versus O: bit-exact;
- editable state after injection versus O: bit-exact;
- source state after injection versus E: bit-exact;
- next model-call input versus intended hybrid: bit-exact;
- model, conditioning, prompt, noise, sigma, and weights are unchanged;
- no direct editable modification occurs.

## Experiment 1 — Immediate pulse response

Each row measures the editable raw x0 at evaluation `i+1`.

### Rigid bridge

| Pulse i | RMS pulse-O | RMS E-O | Recovery R | Cosine | Projection |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.615694 | 0.615694 | 1.000 | 1.000 | 1.000 |
| 1 | 0.803846 | 0.941630 | 0.854 | 0.675 | 0.576 |
| 2 | 0.616016 | 0.970991 | 0.634 | 0.390 | 0.247 |
| 3 | 0.477926 | 1.005004 | 0.476 | 0.331 | 0.157 |
| 4 | 0.316495 | 1.022845 | 0.309 | 0.229 | 0.071 |
| 5 | 0.224954 | 1.038306 | 0.217 | 0.180 | 0.039 |
| 6 | 0.143791 | 1.031777 | 0.139 | 0.165 | 0.023 |

### Organic tree

| Pulse i | RMS pulse-O | RMS E-O | Recovery R | Cosine | Projection |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.394856 | 0.394856 | 1.000 | 1.000 | 1.000 |
| 1 | 0.593851 | 0.635695 | 0.934 | 0.899 | 0.839 |
| 2 | 0.613428 | 0.709852 | 0.864 | 0.817 | 0.706 |
| 3 | 0.470260 | 0.757819 | 0.621 | 0.538 | 0.334 |
| 4 | 0.350571 | 0.798064 | 0.439 | 0.391 | 0.172 |
| 5 | 0.239843 | 0.822348 | 0.292 | 0.264 | 0.077 |
| 6 | 0.131416 | 0.818917 | 0.160 | 0.171 | 0.027 |

### Photometric desert

| Pulse i | RMS pulse-O | RMS E-O | Recovery R | Cosine | Projection |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.287673 | 0.287673 | 1.000 | 1.000 | 1.000 |
| 1 | 0.424992 | 0.443774 | 0.958 | 0.980 | 0.939 |
| 2 | 0.444920 | 0.515465 | 0.863 | 0.878 | 0.757 |
| 3 | 0.384855 | 0.581331 | 0.662 | 0.703 | 0.466 |
| 4 | 0.302858 | 0.629767 | 0.481 | 0.526 | 0.253 |
| 5 | 0.204283 | 0.652377 | 0.313 | 0.397 | 0.124 |
| 6 | 0.123983 | 0.653705 | 0.190 | 0.297 | 0.056 |

Interval 0 is exact because E and O have identical editable accepted states
after the first Euler proposal and differ only in source coordinates. Replacing
O's source with E's source therefore recreates the complete E state at that
boundary. Later pulses combine E source with an already-different O editable
history, so their recovery decreases. This confirms the causal chain while also
showing accumulated trajectory dependence after the first interval.

Source-coordinate next predictions are also strongly causal. At interval 0,
pulse and E source raw x0 are identical. At later intervals, source response
remains highly directionally aligned while editable alignment falls sooner;
for example bridge source cosine is `0.902` at pulse 1 and `0.995` at pulse 6.

## Experiment 2 — Persistence

The interval-0 pulse receives no later epsilon correction. Its editable effect
does not vanish:

| Case | Eval 1 projection | Eval 2 | Eval 4 | Eval 7 | Eval 7 cosine | Eval 7 recovery |
|---|---:|---:|---:|---:|---:|---:|
| bridge | 1.000 | 0.470 | 0.453 | 0.469 | 0.526 | 0.892 |
| tree | 1.000 | 0.425 | 0.303 | 0.297 | 0.426 | 0.697 |
| desert | 1.000 | 0.324 | 0.274 | 0.297 | 0.428 | 0.694 |

The perturbation persists and its magnitude remains substantial, but it
redirects away from the complete E trajectory after evaluation 1. Full epsilon
therefore contains both a powerful early displacement and cumulative later
source corrections. Magnitude recovery alone overstates reproduction: bridge
ends near `R=0.892` but only `0.469` of E's directionally aligned component.

## Experiment 3 — Spatial source support

The selected interval is fixed at 0. The source region is latent columns
`16:48`. The source-side boundary halo is exactly two latent columns at each
edge (`16:18` and `46:48`); distant source is the remaining columns `18:46`.

| Support | Editable RMS | Recovery R | Cosine | Projection |
|---|---:|---:|---:|---:|
| full source | 0.615694 | 1.000 | 1.000 | 1.000 |
| boundary halo | 0.099428 | 0.161 | 0.219 | 0.035 |
| distant source | 0.524397 | 0.852 | 0.898 | 0.765 |

Useful communication is not predominantly local boundary negotiation. The
distant source interior carries most of the aligned response. In the decoded
bridge x0 sheet, boundary-only remains close to O's bridge scale and cable
layout; distant-source shifts deck/cable perspective toward E. Full source is
still required for exact reproduction, so the supports are not assumed additive
through the nonlinear model.

## Experiment 4 — Frequency content

At interval 0, the source delta is split with a fixed `3x3` box average using
replicated latent borders, remasked to the source. The complementary component
is `high = full - low`, so reconstruction is exact. No scale was swept.

| Component | Editable RMS | Recovery R | Cosine | Projection |
|---|---:|---:|---:|---:|
| full | 0.615694 | 1.000 | 1.000 | 1.000 |
| low | 0.563080 | 0.915 | 0.912 | 0.834 |
| high | 0.146045 | 0.237 | 0.298 | 0.071 |

The causal signal is predominantly low-frequency at this fixed decomposition.
Low-frequency injection visibly reproduces E's larger perspective/cable sweep;
high-frequency injection remains closer to O and does not reproduce the same
geometric change. This is one deterministic latent split, not a general claim
that high frequencies are irrelevant.

## Interpretation and limits

- Causal influence: proven exactly by hybrid-state invariants and next-call
  response.
- Directional reproduction: exact for interval 0; progressively weaker for
  later isolated pulses.
- Persistence: strong but partially redirected without repeated correction.
- Spatial locality: predominantly distant/distributed source, not the narrow
  boundary halo.
- Frequency: predominantly the fixed low-frequency component.
- Final-image quality: not evaluated or qualified by this mechanism study.

The classification is LOCALIZED because a particular interval (0), spatial
support (distant source carrying `0.765` aligned projection), and component
(low-frequency carrying `0.834`) reproduce substantial directionally consistent
fractions of epsilon's editable effect. This does not imply that a single-pulse
sampler can replace full epsilon: later trajectory agreement is incomplete.

Unresolved questions are whether the interval-0 localization survives other
seeds/geometries and how repeated later source corrections contribute to final
quality. Those require separately authorized discriminators, not optimization.

Artifacts and full telemetry are under
`experiments/local_edit_source_coevolution_results/`. No production code or
registration changed.
