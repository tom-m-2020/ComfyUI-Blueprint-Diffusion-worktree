# Drift-Constrained Sampling production implementation slice

## Result

**DRIFT PRODUCTION SLICE IMPLEMENTED — PARITY VERIFIED**

The complete qualified five-mode contract is implemented without replacing
native `SamplerCustomAdvanced`, GUIDER, MODEL, SIGMAS, or VAE behavior. Public
modes are exactly `none`, `fss`, `ilvr`, `fss_ilvr`, and `fbsdiff`; no other
combination is accepted or exposed.

## Public nodes

### Drift-Constrained Sampling

Inputs are source LATENT, mode, noise seed, FSS radius/bandwidth, ILVR
factor/alpha, normalized FBSDiff threshold, calibration start/end, and optional
reference CONDITIONING. The socket is optional at declaration level so other
modes need no dummy connection, but `fbsdiff` rejects a missing value.

Outputs are paired `DRIFT_CONSTRAINT` and `NOISE`. The frozen policy owns a
detached source snapshot, fingerprints, batch index, seed, and compositional
configs. A shared run-local provenance record binds the generated endpoint to
the policy; the sampler rejects missing or mismatching provenance.

### Drift-Constrained Euler Sampler

Input is `DRIFT_CONSTRAINT`; output is `SAMPLER`, connected directly to native
`SamplerCustomAdvanced`. It is explicitly Euler- and CONST-only.

```text
Drift-Constrained Sampling
    +-- NOISE -----------------------------------+
    `-- DRIFT_CONSTRAINT                         |
                 |                               v
       Drift-Constrained Euler          SamplerCustomAdvanced
                 `-- SAMPLER --------------------^
```

## Preserved semantics

- `none`: native `prepare_noise`, native CONST initialization, ordinary Euler.
- `fss`: Phase 1 FFT endpoint, default radius 2/bandwidth 2; sigma scaling stays
  native and later.
- `ilvr`: Phase 2 factor-4 area/nearest projection and alpha 1 correction after
  each accepted proposal with `sigma_next>0`; never at sigma zero.
- `fss_ilvr`: the same FSS endpoint initializes CONST and constructs every
  matching-sigma ILVR source.
- `fbsdiff`: Phase 5 channel-wise orthonormal DCT, normalized high threshold
  `5/63`, and half-open calibration `[0,0.5)`, immediately before target
  evaluation. The explicit reference conditioning drives an independent
  CFG-1-equivalent Euler branch at identical sigmas from the same ordinary
  endpoint/source.
- Completion calls native inverse noise scaling.
- Callbacks receive corrected accepted state, matching the harness rather than
  native Euler's pre-update callback placement.

Reusable math lives only in `drift_constraints.py`; production imports no
experiment scripts or research instrumentation.

## Fail-closed behavior and batching

The slice rejects non-CONST sampling, invalid sigmas, masks, nested/non-4D
latents, dimensional or batch mismatch, swapped/mutated sources, unrelated
endpoint provenance, unsupported modes, nonsquare FBSDiff grids, and FBSDiff
reference conditions containing controls, GLIGEN, hooks, masks, or declared
stateful fields.

No source/reference is broadcast, repeated, modulo-mapped, cropped, or shared
between rows. Tensor row `b` owns endpoint/source/reference row `b`. Native
`batch_index` is passed unchanged to `prepare_noise`.

## Automated validation

`tests/test_drift_constraints.py` has 17 tests covering deterministic Gaussian
and `batch_index`; FSS parity/magnitude/RMS/row independence; ILVR projection;
DCT roundtrip and the 1018/1024 mask; five modes/socket types; frozen policy,
provenance, fingerprints; B=1/B=2 ownership; fake-CONST initialization/Euler;
accepted callbacks; terminal ILVR skip; hybrid endpoint use; FBSDiff ordering;
and synchronized reference ordinals.

Focused result: 17/17 passed. Repository result: 99/99 passed. Known optional
Triton/xformers and deprecated pynvml warnings did not affect success.

## Real stock-Klein migration parity

`drift_production_slice_runtime_validation.py` used the fixed Phase 1-5 Klein
4B, VAE, text encoder, portrait, prompt, seed, CFG, six-evaluation schedule, and
Euler settings. Each mode ran through `SamplerCustomAdvanced.execute` and was
compared with its corresponding research sampler at tolerance `2e-5` RMS.

| Mode | evaluations | final RMS error | max trajectory RMS error | finite |
|---|---:|---:|---:|---:|
| none | 6 | 0 | 0 | yes |
| fss | 6 | 0 | 0 | yes |
| ilvr | 6 | 0 | 0 | yes |
| fss_ilvr | 6 | 0 | 0 | yes |
| fbsdiff | 6 target plus synchronized reference | 0 | 0 | yes |

Trajectory error includes denoised estimates and accepted states. This proves
migration parity, not new perceptual qualification. Machine-readable evidence:
`experiments/drift_production_slice_validation/results.json`.

## Qualification labels

- `none`: ordinary baseline.
- `fss`: experimental; useful broad structure preservation in several cases,
  but failed portrait identity.
- `ilvr`: experimental/partial; coarse restraint did not preserve decisive
  portrait identity.
- `fss_ilvr`: experimental/nonredundant, but still failed portrait identity.
- `fbsdiff`: experimental; the tested stock-Klein adaptation was falsified.

None is a reliable identity-preservation claim. FBSDiff's empty-conditioned
CONST/Euler reference is not published DDIM inversion and doubles evaluations
during calibration. External correspondence, adaptive detection, masks,
arbitrary samplers, non-CONST models, stateful reference conditioning, and
additional combinations remain excluded.

## Changed files

- `target/ComfyUI-Blueprint-Diffusion/drift_constraints.py`
- `target/ComfyUI-Blueprint-Diffusion/nodes.py`
- `tests/test_drift_constraints.py`
- `experiments/drift_production_slice_runtime_validation.py`
- `experiments/drift_production_slice_validation/results.json`
