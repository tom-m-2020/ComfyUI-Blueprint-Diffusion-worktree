# FLUX.2 intermediate global-density discriminator

## Verdict

**INTERMEDIATE DENSITY INSUFFICIENT.**

At the exact late Phase-2e state used by Phase 2f, increasing the global branch
from `16 x 32` (512 tokens) to `24 x 48` (1152 tokens) improved numerical
agreement with the dense prediction, but did not remove the duplicate
lighthouse-like object. The `32 x 64` (2048-token) dense-global context did
remove that duplicate through the identical external-K/V integration path.

The tested intermediate density is therefore not a useful semantic threshold
for this failure. This conclusion is narrow: it rejects this one uniformly
resized 1152-token representation at this state, not every representation with
1152 tokens.

## Controlled boundary

The experiment reproduced the Phase-2e accepted context trajectory through
exactly two Euler updates and stopped before evaluation 2:

```text
sigma:                 0.8925943970680237
accepted latent:       [1,128,32,64]
accepted latent RMS:   0.9205285906791687
affected center crop:  x=24..55, y=0..31 latent coordinates
diagnostic updates:    0
```

All GPU-side accepted-state statistics matched the Phase-2f reproduction
exactly (`max stat difference = 0`). The diagnostic sampler returned the
accepted latent unchanged (`max_abs = 0`). Prompt, conditioning, local crop,
sigma, local RoPE, local hidden/MLP path, all 25 modified FLUX.2 blocks,
attention integration, and final projection were fixed.

Only the global input density and corresponding full-canvas RoPE changed:

```text
A  16 x 32  =  512 global generated tokens
B  24 x 48  = 1152 global generated tokens
C  32 x 64  = 2048 global generated tokens
```

## Quantitative result

| Global context | Local Q x K | Changed QK work | RMS vs dense crop | Low-frequency RMS | Prediction RMS |
|---|---:|---:|---:|---:|---:|
| 512 tokens | 1536 x 2048 | 3,145,728 (1.000x) | 0.372353 | 0.187836 | 0.819064 |
| 1152 tokens | 1536 x 2688 | 4,128,768 (1.3125x) | 0.327636 | 0.162362 | 0.819669 |
| 2048 tokens | 1536 x 3584 | 5,505,024 (1.750x) | 0.179977 | 0.098817 | 0.846498 |

The 1152-token branch reduced total RMS by 12.0% and low-frequency RMS by
13.6% relative to 512 tokens. It closed only 23.2% of the total-RMS gap and
28.6% of the low-frequency gap between compact and dense context. Prediction
norms remained stable, so the result is not explained by gross scale collapse.

The QK figures are only the changed local-query attention-context products per
block. They do not include the global capture forward, projections, MLPs,
residual paths, or total-model FLOPs.

## Decoded semantic observations

- **Lighthouse duplication:** both 512 and 1152 retain the small extra
  center-right lighthouse beside the dark stone structure. Dense context
  removes the lighthouse.
- **Stone structure:** both compact and intermediate retain a prominent dark
  stone object. Dense context reduces it to a smaller tower-like structure but
  is still not identical to the dense reference crop.
- **Train:** intermediate context modestly changes carriage spacing and
  continuity toward dense, but remains visibly different from the dense train.
- **Bridge:** intermediate context gives an incremental deck/cable alignment
  improvement; dense context is substantially closer to the reference.
- **Horizon/water:** intermediate improves the distant horizon locally, but
  retains the same false object pair. Dense context has the best agreement.
- **Local stability:** none of the densities destroyed fine bridge, train, or
  water detail.

The surrounding-context composites are diagnostic visualizations: the tested
center crop is placed into the unchanged dense reference surroundings so its
semantic differences are easy to locate. They are not assembled generation
outputs.

## Implementation and evidence

- `flux2_candidate2_intermediate_density.py`
- `flux2_candidate2_intermediate_density_results/report.json`
- three decoded crop predictions
- three dense-surrounding context composites
- three prediction-error magnitude maps
- dense crop and full-canvas decoded references

The JSON report records all 25 source-capture and context-injection block
records for every density, exact generated/text positional-ID extents, per-call
tensor statistics, timing, and the zero-update/state-reproduction assertions.

## Interpretation and next experiment

A single uniform 24-by-48 global grid spends 31.25% more changed local
attention-context work than the 512-token branch yet does not solve the target
semantic failure. A further uniform-density sweep would have low information
value and risks merely approaching full dense cost.

The highest-value next experiment is a single-state **multiscale representation
probe at roughly the same token budget**: retain the 512-token whole-canvas grid
and add a fixed higher-density global token region covering the center-right
object/bridge overlap, then use the same 25-block external-K/V path. Compare it
against the uniform 1152-token control. This directly tests whether the missing
evidence is spatially localized and recoverable without a full dense global
forward; it must remain a diagnostic, not adaptive scheduling infrastructure.

**INTERMEDIATE DENSITY INSUFFICIENT**
