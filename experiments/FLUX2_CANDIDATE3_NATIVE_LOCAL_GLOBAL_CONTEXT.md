# Phase 9c — native-local global-context discriminator

## Scope

This zero-update terminal probe reuses the exact Phase-9b sigma-consistent
accepted trajectory and tests whether whole-scene transformer context repairs
the remaining semantic divergence of magnified local predictions. Production
and ComfyUI core are unchanged.

Fixed controls:

- FLUX.2 Klein 4B, bridge/train prompt, seed `20260901`, CFG 1;
- `H=64x128`, fixed `G=24x48`, four-step CONST-flow schedule;
- accepted Phase-9b C state through interval 2;
- fifteen destination regions, `32x32`, stride 24;
- working canvas `W=64x64` constructed as
  `nearest2(H_crop) + sigma*(N-nearest2(avgpool2(N)))`;
- working coordinates span each destination region with endpoint scale
  `31/63` on both axes;
- identical 2x2-mean prediction restriction and normalized assembly.

The terminal sigma is `0.6780259013`. All variants consume identical accepted
`H/G`, region inputs, working tensors, order, coordinates, and overlap weights.
`avgpool2(W)==H_crop` holds to `1e-6`. No variant updates sampling state.

## Variants and context construction

| Variant | Source | Context tokens/block | Context depth |
|---|---|---:|---:|
| A MAGNIFIED_LOCAL_ONLY | none | 0 | 0 |
| B MAGNIFIED_FULL_H_CONTEXT | accepted current `H` | 8,192 | all 25 blocks |
| C MAGNIFIED_FIXED_G_CONTEXT | accepted fixed `G` | 1,152 | all 25 blocks |

B and C run one ordinary same-sigma globally interacting source forward and
reuse the previously qualified pre-RoPE generated K/V integration. Local text
queries retain native behavior. Local K uses mapped destination coordinates;
source H uses its native complete-canvas coordinates, while G uses
endpoint-preserving scales `63/23` and `127/47`. The diagnostic CPU-offloaded
cache is intentionally not a proposed production implementation.

## Numerical results

| Variant | Overlap RMS | Assembled RMS | Change from A (RMS/max) | Source CUDA | Local CUDA | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|
| A local only | 0.214053 | 0.856812 | — | 0.000 s | 14.912 s | 2.890 / 3.353 GiB |
| B full-H context | 0.134323 | 0.855734 | 0.126025 / 1.422927 | 3.719 s | 38.101 s | 3.657 / 4.392 GiB |
| C fixed-G context | 0.200175 | 0.846930 | 0.077429 / 0.947349 | 0.400 s | 24.137 s | 3.001 / 3.472 GiB |

For the representative working crop, B differs from A by RMS/max
`0.180084/1.908272`; C differs by `0.112169/1.340161`. Full-H context reduces
overlap disagreement by 37.2%, while fixed-G context reduces it by only 6.5%.

The full-H diagnostic retains 2.344 GiB of CPU K/V and transfers 35.156 GiB
across the fifteen crop calls. Fixed-G retains 0.330 GiB and transfers 4.944
GiB. These costs are diagnostic overhead, not an efficiency result.

## Semantic result

The first material change occurs inside the context-conditioned local model
predictions: B aligns the bridge deck and crop transitions much more strongly
than A, consistent with the overlap reduction. It nevertheless does not remove
the repeated train/bridge alternatives. The decoded assembled terminal result
still contains a long sequence of competing/repeated structural spans rather
than one unambiguous bridge/train system.

C changes the local predictions and slightly improves overlap agreement, but
remains visually close to A's semantic failure. The 1,152-token fixed G does
not impose object uniqueness or a single bridge interpretation on the
magnified local calls.

Because the full accepted-H upper bound also fails the primary semantic gate,
this probe does **not** isolate the remaining problem as merely insufficient
fixed-G density. It also does not justify increasing G density. The evidence
instead points to the combination of native-working-canvas scale transport,
its positional semantics, and/or 2x2 x0 restriction as the next causal
boundary. Full-H context demonstrates improved compatibility, but not semantic
resolution.

All fifteen restricted crop diagnostics, representative `x0_W`, assembled
terminal outputs, hashes, and detailed metrics are in
`flux2_candidate3_native_local_global_context_results/`.

## Verdict

**FULL-H CONTEXT IMPROVES AGREEMENT BUT LOCAL TERMINAL FAILURE PERSISTS**

Fixed-G context does not qualify for a complete four-step trajectory. No
production change is justified.
