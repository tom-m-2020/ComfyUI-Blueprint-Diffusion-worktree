# Phase 37 — FLUX.2 Klein local-refinement schedule provenance audit

## Executive result

**B — CANONICAL FULL/PARTIAL SCHEDULE EXISTS, BUT LOCAL STEP COUNT REMAINS FREE.**

Authoritative sources establish a flow coordinate, a model-associated shifted
full schedule, Euler integration, and (in Diffusers' Klein inpaint pipeline) an
index-truncation rule for `strength`. They do **not** determine a unique short
trajectory beginning at the project's exact `sigma=0.25`. Selecting the local
interval count remains an empirical policy, and exact `0.25` is not generally a
member of the official discretized curve.

No Phase-38 schedule is proposed. No diffusion model was invoked.

## Evidence hierarchy and revisions

1. Black Forest Labs' official `black-forest-labs/flux2` inference repository,
   inspected 2026-09-06.
2. The BFL `FLUX.2-klein-4B` Diffusers package and scheduler configuration,
   inspected 2026-09-06.
3. Hugging Face Diffusers' `Flux2KleinPipeline` and
   `Flux2KleinInpaintPipeline`, inspected 2026-09-06.
4. Native ComfyUI checkout commit
   `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`.

The BFL repository identifies distilled Klein 4B as step- and
guidance-distilled, fixes its production default to four steps, and contrasts
that with 50-step Base inference. Its CLI creates a random T2I state, calls
`get_schedule(num_steps, image_seq_len)`, and treats supplied images as
reference tokens rather than as a partially noised starting latent. There is
no `strength` or init-latent partial-denoise path in that official CLI.

## 1. Model/native parameterization

BFL defines a continuous flow coordinate `t` descending from one to zero. For
an unshifted coordinate `u`, its schedule uses

```text
shift(u; mu) = exp(mu) / (exp(mu) + (1/u - 1))
```

where `mu = compute_empirical_mu(image_seq_len, num_steps)`. The full schedule
is

```text
u_i = 1 - i / num_steps,  i = 0..num_steps
t_i = shift(u_i; mu)
```

and the official denoiser advances with explicit first-order Euler:

```text
x_next = x + (t_next - t_current) * model(x, t_current, conditioning)
```

The schedule therefore depends on both image sequence length **and the
caller-supplied step count**. BFL's implementation is authoritative evidence
for a full-generation curve and solver, not for a unique late refinement
discretization. See the official
[`get_schedule` and `denoise` implementation](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/sampling.py)
and [Klein model defaults](https://github.com/black-forest-labs/flux2/blob/main/src/flux2/util.py).

Native ComfyUI represents Klein through `CONST + ModelSamplingFlux`:

```text
x0 = x_t - sigma * model_output
x_t = sigma * noise_scale * noise + (1 - sigma) * latent_image
timestep(sigma) = sigma
```

For the current Flux2 profile, ComfyUI records `shift=2.02`. This establishes
that `0.25` is a valid continuous model/sample coordinate under the qualified
ComfyUI contract. It does not establish `0.25` as an official img2img strength
or choose later integration points.

## 2. Official inference schedule construction

The official BFL implementation prescribes its shifted-linear flow schedule
and Euler solver for ordinary inference. The official Klein 4B Diffusers model
package names `FlowMatchEulerDiscreteScheduler`, with dynamic shifting enabled,
1000 training timesteps, exponential shift type, and no Karras, beta, or
exponential-sigma alternative. See the model's
[`scheduler_config.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/scheduler/scheduler_config.json)
and [`model_index.json`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/main/model_index.json).

Diffusers' Klein T2I pipeline still accepts either `num_inference_steps` or an
explicit custom `sigmas` list. Its default curve starts from a caller-sized
linear sigma grid and applies the empirical image-length/step-count shift.
Consequently, even inside the model-associated scheduler family, the number of
points is not inferred from a starting strength. See
[`Flux2KleinPipeline`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein.py).

For a 4096-position local canvas, reproducing BFL's formula gives:

```text
4 steps: [1.0, 0.9673839882, 0.9081439225, 0.7671999639, 0.0]
8 steps: [1.0, 0.9854255070, 0.9666411017, 0.9415147849,
          0.9061825955, 0.8528416678, 0.7630145132, 0.5798067880, 0.0]
```

Neither contains `0.25`. Raising the step count until a point happens to
approach `0.25`, or inserting `0.25`, would be an empirical discretization or
splice forbidden by this audit.

## 3. Partial denoising and img2img

BFL's official FLUX.2 repository supports image editing by concatenating image
reference tokens while the generated state starts from noise. It does not
define an img2img `strength -> partially noised generated latent` path.

Diffusers does provide a Klein inpaint pipeline with a `strength` parameter.
Its rule is:

```text
init_timestep = min(num_inference_steps * strength, num_inference_steps)
t_start = int(max(num_inference_steps - init_timestep, 0))
partial_timesteps = full_timesteps[t_start * scheduler.order:]
```

That implementation explicitly copies the generic Stable Diffusion 3 img2img
truncation policy. It first constructs a full schedule from the separately
supplied `num_inference_steps`, then truncates by index. Thus `strength` selects
how many already-discretized points survive; it does not uniquely determine
their locations or the original step count. See
[`Flux2KleinInpaintPipeline.get_timesteps`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux2/pipeline_flux2_klein_inpaint.py).

This is enough to classify the provenance as a canonical full/partial family,
but not enough to derive a parameter-free Phase-38 schedule. It is also weaker
evidence for this project than a BFL Klein-4B latent img2img implementation:
the official BFL CLI does not expose the same operation, and Diffusers' rule is
an inherited generic truncation policy.

## 4. Denoise strength versus sigma

They are not interchangeable:

- `sigma`/`t` is the continuous flow coordinate used by the model and state
  interpolation.
- `strength` is a fraction used to choose an index into a schedule that was
  already discretized using `num_inference_steps`.
- The selected schedule entry is the actual starting sigma. It need not equal
  the numeric strength.

Therefore `strength=0.25` does not mean `sigma_start=0.25`. Under a four-step
official full curve it retains one denoising interval whose start is the last
positive schedule value (about `0.7672` for 4096 positions), not `0.25`.

The project's `sigma=0.25` has a legitimate and exactly tested meaning under
ComfyUI's CONST law:

```text
W_0.25 = 0.75 * anchor + 0.25 * epsilon
```

But it is an **empirically qualified Blueprint terminal-resampling constant**,
not a value selected by official Klein partial-denoise provenance.

## 5. Four distinct layers

| Layer | Provenance result |
|---|---|
| Model sampling law | Continuous CONST/flow coordinate; `x_t=(1-t)x0+t epsilon`; model output is the flow field. |
| Scheduler family | BFL supplies an image-length/step-count shifted linear flow curve; the Diffusers package selects FlowMatch Euler with dynamic shift. Native ComfyUI exposes nine caller-selectable families rather than enforcing the BFL curve. |
| Numerical solver | BFL and the packaged Diffusers scheduler use first-order Euler for the cited path. ComfyUI solver choice remains caller-controlled in general. |
| Partial-denoise policy | Diffusers Klein inpaint truncates a precomputed schedule by strength/index; BFL's own FLUX.2 CLI has no generated-latent img2img truncation path. |

## 6. Why no unique short schedule follows

Starting from exact `sigma=0.25` leaves at least one independent free choice:
the number of local integration intervals. In the authoritative formulas,
`num_steps` is an input and even changes `mu`; it is not derivable from the
starting sigma. The official partial policy cannot remove that freedom because
it truncates a full schedule whose count was already chosen.

There is consequently no deterministic mapping

```text
exact sigma_start=0.25 -> uniquely located intermediate sigmas -> 0
```

without adding an empirical step-count/discretization policy. The model's
step-distilled four-step provenance also does not solve this: its four-step
curve begins at one and has no `0.25` point, so truncation changes the start
state rather than extending the qualified `[0.25,0]` contract.

## 7. Training-time evidence boundary

The inspected official inference repository and shipped scheduler configuration
identify flow matching, dynamic timestep shifting, and a 1000-point training
timestep domain. They do not publish a training-noise density that uniquely
induces a finite late local-refinement discretization. Training-time continuous
or weighted timestep exposure would not, by itself, select a numerical
inference step count.

## 8. Integrity

- Diffusion model forwards: `0`
- Local forwards: `0`
- Destination-sized forwards: `0`
- Decoded images: `0`
- Production changes: none
- ComfyUI-core changes: none

The companion CPU-only script validates local source markers/hashes and
recomputes representative official BFL schedule curves. Machine-readable
results are in
`flux2_klein_refinement_schedule_provenance_results/report.json`.

## Decision

**B — CANONICAL FULL/PARTIAL SCHEDULE EXISTS, BUT LOCAL STEP COUNT REMAINS
FREE.**

Phase 38 may test short local trajectories only if it openly authorizes one
fixed empirical schedule policy, including a fixed interval count and exact
construction from `0.25` to zero. That future schedule must be described as a
new research contract, not as model-mandated Klein provenance.

**SHORT LOCAL TRAJECTORY REQUIRES AN EXPLICIT EMPIRICAL SCHEDULE POLICY**
