# Phase 7 - Production boundary audit for Drift-Constrained Sampling

## Outcome

**PRODUCTION BOUNDARY QUALIFIED — IMPLEMENTATION DEFERRED**

The installed ComfyUI interfaces provide the required separation without a
replacement for `SamplerCustomAdvanced` and without a MODEL patch. FSS belongs
in a source-aware `NOISE`; interval and pre-evaluation interventions belong in
a fixed Euler `SAMPLER`; immutable configuration and source ownership belong in
a compositional `DRIFT_CONSTRAINT` policy.

Implementation is deliberately deferred. The sampler-latent FBSDiff-like mode
adds one required contract not present in the initial diagram: an explicit
empty/reference `CONDITIONING` must be carried by the policy and prepared inside
the already-loaded sampler lifecycle. Landing only FSS/ILVR first would leave
the promised mode/socket contract incomplete, while substituting target
conditioning would change the falsified Phase 5 experiment.

No production files were changed in Phase 7.

## Audited native snapshot

The audit used `C:/Users/Tom-M/data/a/ai/apps/ComfyUI-dev`:

- `comfy_extras/nodes_custom_sampler.py`: `Noise_RandomNoise` lines 724-731 and
  `SamplerCustomAdvanced` lines 1020-1068;
- `comfy/sample.py`: noise generation lines 9-43 and `sample_custom` lines
  86-89;
- `comfy/samplers.py`: `KSAMPLER` lines 977-1007 and `CFGGuider` sampling
  lifecycle lines 1188-1274;
- `comfy/k_diffusion/sampling.py`: Euler lines 189-212;
- `comfy/model_sampling.py`: `CONST` lines 86-101.

The checkout commit could not be queried because Git rejected its ownership as
unsafe. Conclusions are tied to these inspected source files, not an asserted
commit ID.

## Native call and state lifecycle

```text
SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)
  +-- fix_empty_latent_channels(...)
  +-- read optional latent noise_mask
  +-- noise.generate_noise(full LATENT dict)       [raw endpoint]
  `-- guider.sample(raw_noise, latent.samples, sampler, sigmas, ...)
        +-- prepare_sampling(MODEL, shape, guider conditions)
        +-- move noise, latent, sigmas to model device/float32
        +-- process source latent and guider conditions
        `-- sampler.sample(
              prepared guider, complete SIGMAS, model options/seed,
              callback, raw noise, processed source, optional mask)
              +-- model_sampling.noise_scaling(sigma_0, noise, source, ...)
              +-- sampler owns model evaluations and accepted updates
              `-- inverse_noise_scaling(sigma_final, final state)
```

`NOISE.generate_noise()` runs before the guider/model is prepared. It sees the
full LATENT dictionary, including `samples` and optional `batch_index`, but not
MODEL, SIGMAS, conditioning, or CFG.

The `SAMPLER` receives raw noise, the processed source latent, complete sigmas,
prepared guider callable, model options, seed, mask, and callback. It can call
the same native `noise_scaling` and owns both `sigma_i` and `sigma_next`.

Native Euler evaluates `denoised=model(x,sigma_i)`, forms
`d=(x-denoised)/sigma_i`, and accepts
`x_next=x+d*(sigma_next-sigma_i)`. Native Euler's callback occurs before that
update. The drift sampler must instead report its corrected accepted state in
the callback, matching the research harnesses.

For CONST, initialization remains:

```text
x(sigma_0) = sigma_0 * (noise_scale * endpoint)
           + (1 - sigma_0) * source_latent
```

Custom NOISE returns an endpoint only and never applies sigma scaling.

## Boundary questions resolved

1. Noise is generated before `guider.sample`; scaling happens later inside the
   sampler.
2. SAMPLER receives the guider/model wrapper, full sigmas, options/seed,
   callback, raw endpoint, processed source, and mask.
3. It can modify every accepted state only by owning the update algorithm. An
   arbitrary-sampler wrapper cannot assume one evaluation/acceptance per sigma.
4. It has the full tensor and therefore `sigma_i` and `sigma_next`.
5. ILVR constructs the matched source at `sigma_next` with native
   `noise_scaling(sigma_next, endpoint, source, False)` and needs no MODEL patch.
6. Sampler-latent DCT substitution can occur immediately before selected target
   model evaluations.
7. FSS is a clean NOISE operation: it sees source samples and native
   `batch_index` seed semantics before phase substitution.
8. NOISE need not know CONST/sigma; native scaling of its endpoint preserves the
   tested semantics.
9. A frozen opaque policy can carry an owned source snapshot and scalar configs
   without invoking Comfy list mapping.
10. Tensor batching is unambiguous only with exact source/target/noise
    `[B,C,H,W]` equality and reference row `b` owned by target row `b`.

Nested/heterogeneous batches and noise masks were absent from Phases 1-5 and
must be rejected in the first slice.

## Mechanism ownership

| Mechanism | Exact intervention | Owner |
|---|---|---|
| none | Native CONST initialization and Euler intervals | fixed Euler SAMPLER |
| FSS | Substitute source-dependent Fourier phase into Gaussian endpoint before sigma scaling | NOISE |
| ILVR | Correct accepted proposal at `sigma_next>0` with matching-sigma low-pass source | SAMPLER |
| FSS + ILVR | Use the FSS endpoint for initialization and every ILVR matching source | NOISE + SAMPLER |
| sampler-latent FBSDiff-like | Substitute reference DCT band before selected target evaluations | SAMPLER |

The public sampler is Euler-only. It must not claim arbitrary-sampler support.

## FBSDiff reference branch

Phase 5 used the same ordinary endpoint and source, empty conditioning, CFG 1,
identical sigmas, and ordinary Euler for the reference. Production can evolve
it in lockstep:

```text
reference_i, target_i share sigma_i
  +-- optional DCT substitution reference_i -> target_i
  +-- empty-conditioned reference evaluation
  +-- target-guider evaluation
  `-- independent Euler proposals to sigma_next
```

The reference CONDITIONING must be an explicit policy input. The sampler must
convert/process it against the already-loaded inner model with the same batch,
device, seed, and options. The first implementation should reject ControlNet,
GLIGEN, hooks, or other stateful reference additions. Substituting target
conditioning is not equivalent. This preserves Phase 5, including its known
limitation: it is not DDIM inversion and was falsified.

## MODEL-patch alternative

A MODEL patch sees current input and current sigma for each forward. It does not
receive `sigma_next`, sampler derivative/update semantics, accepted post-update
state, or a reliable acceptance event. Multistage, ancestral, or adaptive
samplers may evaluate multiple times or reject proposals.

- FSS in MODEL would have to detect the first call and undo/emulate native
  initialization.
- ILVR requires the post-Euler proposal at `sigma_next`, which MODEL never sees.
- FBSDiff could alter a pre-forward input, but MODEL cannot identify accepted
  target evaluations or synchronize an independently accepted reference path
  for arbitrary samplers.
- The hybrid inherits both ownership failures.

MODEL patching therefore cannot implement all modes while genuinely reusing
arbitrary samplers. MODEL and its forward remain ordinary.

## Proposed nodes and sockets

### Drift-Constrained Sampling

Inputs:

- `source: LATENT`
- `mode: COMBO`: `none`, `fss`, `ilvr`, `fss_ilvr`, `fbsdiff`
- `noise_seed: INT`
- FSS `radius: FLOAT`, `transition_bandwidth: FLOAT`
- ILVR `downsample_factor: INT`, `alpha: FLOAT`
- FBSDiff `normalized_threshold: FLOAT`, calibration `start/end: FLOAT`
- optional `reference_conditioning: CONDITIONING`, required for `fbsdiff`

Outputs: `constraint: DRIFT_CONSTRAINT` and `noise: NOISE`.

Paired outputs bind source snapshot, seed, configuration, and endpoint
provenance. For `none`, `ilvr`, and `fbsdiff`, NOISE is numerically native random
noise. For FSS modes it applies FSS. Keeping them paired prevents a policy from
silently receiving an unrelated endpoint.

### Drift-Constrained Euler Sampler

Input `constraint: DRIFT_CONSTRAINT`; output `sampler: SAMPLER`. Connect both
outputs, an ordinary GUIDER, ordinary SIGMAS, and the same source LATENT to
native `SamplerCustomAdvanced`.

The sampler validates source fingerprint and endpoint provenance before
initialization. It rejects masks, nested/non-4D latents, non-CONST sampling,
invalid sigmas, cardinality mismatch, and a source differing from the policy
snapshot. No custom GUIDER, MODEL, SIGMAS, VAE, or Advanced sampler is needed.

## Internal policy

Use immutable composition rather than mode-shaped implementation:

```text
DriftConstraintPolicy
  source: owned tensor snapshot
  source_fingerprint
  noise_seed
  fss: FSSConfig | None
  ilvr: ILVRConfig | None
  fbsdiff: FBSDiffConfig | None
  endpoint_provenance
```

UI modes map only to permitted compositions. Reject FSS+FBSDiff, ILVR+FBSDiff,
and three-way combinations. There is no geometry drift threshold because Phase
6 found no qualified signal.

Controls retain their own meanings:

- FSS radius is in latent-frequency pixels; transition bandwidth defaults to
  tested 2.0.
- ILVR factor is area-downsample/nearest-upsample; alpha is explicit blending.
  Defaults are tested factor 4/alpha 1. Initially apply every nonterminal
  interval; defer untested ILVR start/end controls.
- FBSDiff exposes high-FBS only. Threshold is normalized frequency with tested
  default `5/63`, not an absolute 32-grid integer. Calibration is a half-open
  normalized evaluation span with tested default `[0,0.5)`. Low/mid bands are
  not Klein-qualified.

## Batch/cardinality contract

Normal tensor batching is one execution and always one-to-one:

```text
source[b] + endpoint[b] + target_state[b] + reference_state[b]
```

- Source/latent/noise batch, channel, and spatial dimensions must match exactly.
- Never repeat, modulo, crop, or singleton-broadcast spatial source states.
- FFT/DCT and ILVR operate independently per row and channel.
- Forward `batch_index` to native `prepare_noise`; intentional shared Gaussian
  indices do not share reference rows.
- Native conditioning broadcasting may remain, but source states may not.
- Reject Comfy list inputs as a substitute for tensor batching.
- Clone/detach the policy source so later LATENT mutation cannot retarget it.

## Migration map

| Research harness | Reusable logic | Experiment-only |
|---|---|---|
| `drift_phase_preserving_klein.py` | frequency mask and structured endpoint FFT math | synthetic source, diagnostics, sheets |
| `drift_ilvr_klein.py` | factor projection and nonterminal matching-sigma correction | metrics, captures, cases |
| Phase 3 hybrid | same FSS endpoint for initialization and ILVR source formula | comparisons/telemetry |
| `drift_sampler_latent_fbsdiff_klein.py` | DCT/IDCT, normalized high mask, paired substitution | control deltas and exports |
| Phase 1/2 Euler samplers | initialization, call, proposal, callback, inverse scaling | diagnostic snapshots |

Production math should live in one narrow target module. Tests should import it,
not experiment scripts.

## Implementation and equivalence gate

Before node registration:

1. Pure tests: FFT magnitude/RMS and deterministic seed; DCT roundtrip; ILVR
   projection; schedule activation.
2. Socket tests: exact types, frozen policy, five modes, paired provenance.
3. Batch tests: batch 1/2 one-to-one; reject mismatch, nested, mask, and swapped
   source.
4. Fake-CONST lifecycle tests: correction after proposal, no sigma-zero
   correction, DCT before target evaluation, matching reference ordinal, and
   callback receives corrected accepted state.
5. Numerical equivalence against each Phase 1-5 harness mechanism using
   identical tensors/settings.
6. One real stock-Klein batch-one run per mode through native
   `SamplerCustomAdvanced`, checking trajectory/final latent tolerance.

Expected results are migration parity, not renewed perceptual qualification.

## Risks and compatibility limitations

- First release is Euler- and CONST/Klein-only.
- Phase 5's empty reference is not DDIM inversion and remains a formulation
  mismatch; FBSDiff mode is experimental and falsified.
- FBSDiff doubles model evaluations during calibration.
- Dynamic hooks, ControlNet, masks, and nested latents are outside the first
  slice.
- Callback timing differs from native Euler and requires an explicit test.
- Large batches multiply reference compute/memory; batch must never collapse.
- FSS/ILVR did not preserve portrait identity reliably and must not be defaults.
