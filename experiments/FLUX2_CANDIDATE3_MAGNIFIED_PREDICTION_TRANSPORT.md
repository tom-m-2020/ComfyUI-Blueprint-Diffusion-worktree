# Phase 9e — magnified-prediction transport discriminator

## Result

The requested transport rules are algebraically identical under the qualified
Phase-9b construction. Runtime differences are limited to float32 operation
ordering (`3.65e-8` to `5.20e-8` assembled RMS), and all four decoded outputs
retain the same repeated bridge/train alternatives.

This terminal-only experiment changes no production or ComfyUI-core code.

## Fixed controls

- native FLUX.2 Klein 4B;
- exact Phase-9b C accepted preterminal state;
- `H=64x128`, fixed `G=24x48`, sigma `0.6780259013`;
- bridge/train prompt, seed `20260901`, CFG 1;
- fifteen destination regions of `32x32`, stride 24;
- identical sigma-consistent `32->64` W tensors and hashes;
- native-local unit coordinates `0..63` from Phase 9d B;
- no global K/V, no state update;
- the same linear nonoverlapping 2x2 mean restriction `D` and overlap assembly.

All variants share the same fifteen W model evaluations. Model CUDA time is
`15.473 s`; peak allocated/reserved CUDA is `2.883/3.373 GiB`.

## Exact algebra

The live ComfyUI `CONST` model sampling relation is

```text
x0_W = W - sigma * v_W
v_W  = (W - x0_W) / sigma
```

The Phase-9b W construction establishes `D(W)=H_crop` to numerical precision,
and D is linear.

### A — x0 mean

```text
x0_A = D(x0_W)
```

### B — velocity mean

```text
v_crop = D(v_W)
x0_B   = H_crop - sigma * v_crop
       = D(W) - sigma * D(v_W)
       = D(W - sigma*v_W)
       = D(x0_W)
```

### C — delta transport

The exact working-frame denoising correction is
`delta_W=x0_W-W=-sigma*v_W`:

```text
x0_C = H_crop + D(delta_W)
     = D(W) + D(x0_W-W)
     = D(x0_W)
```

No independent heuristic scale exists under this parameterization.

### D — same-sigma re-noise/restrict

This control is mathematically defined:

```text
R_W  = x0_W + sigma*v_W = W
R_H  = D(R_W) = H_crop
x0_D = R_H - sigma*D(v_W) = D(x0_W)
```

It therefore cannot create a fourth transport behavior.

## Numerical results

| Variant | Assembled RMS | Overlap RMS |
|---|---:|---:|
| A X0_MEAN | 0.890278 | 0.234164910 |
| B VELOCITY_MEAN | 0.890278 | 0.234164911 |
| C DELTA_TRANSPORT | 0.890278 | 0.234164912 |
| D ORACLE_RENOISE_RESTRICT | 0.890278 | 0.234164910 |

| Pair | Assembled RMS difference | Max absolute |
|---|---:|---:|
| A/B | 4.887e-8 | 7.153e-7 |
| A/C | 4.954e-8 | 7.153e-7 |
| A/D | 5.143e-8 | 7.153e-7 |
| B/C | 3.652e-8 | 7.153e-7 |
| B/D | 3.983e-8 | 4.768e-7 |
| C/D | 5.196e-8 | 7.153e-7 |

Per crop, `max_abs(D(W)-H_crop)=2.384e-7`. Same-sigma reconstruction has
`max_abs(R_W-W)=3.576e-7`, and its restricted state differs from H by at most
`2.384e-7`. Each alternative destination prediction differs from A by at most
`4.768e-7` before assembly.

Decoded RGB hashes differ because these sub-micro latent rounding differences
cross final pixel quantization boundaries. Direct visual inspection shows no
semantic or perceptual transport difference. Every output retains the repeated
train/bridge/support systems of Phase 9d B.

## Interpretation

Velocity/update transport cannot improve this experiment because it is not a
different operation under linear D and exact coarse consistency. The same is
true for delta transport and same-sigma re-noise. Obtaining a distinct result
would require changing an explicitly fixed assumption, such as the restriction
operator, W/H correspondence, or the evolution of W through time.

No terminal transport passes the semantic gate. Terminal one-shot
magnification is therefore insufficient under the qualified contract. The
next architectural experiment should maintain persistent per-region
native-scale working states across the trajectory, rather than regenerate W
from accepted H crops at each evaluation and seek a different terminal map.

All 60 transported crop decodes, shared `x0_W` diagnostics, assembled outputs,
hashes, and consistency telemetry are under
`flux2_candidate3_magnified_prediction_transport_results/`.

## Verdict

**NO TRANSPORT RULE MATERIALLY IMPROVES — TERMINAL ONE-SHOT MAGNIFICATION REJECTED**

No full trajectory and no production change are justified.
