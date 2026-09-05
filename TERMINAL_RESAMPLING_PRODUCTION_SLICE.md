# Phase 27 — Terminal-resampling production slice

## Outcome

The exact Phase-25 terminal-resampling mechanism is implemented as a separate
production node, `BlueprintTerminalResampling` (`Blueprint Terminal
Resampling`). The legacy `Blueprint Candidate-3 Euler Sampler` remains present
and its sampler/state/geometry/policy/executor files are unchanged.

The new node is deliberately fail-closed. It supports only native ComfyUI
FLUX.2 Klein 4B, BasicGuider with one positive CFG-1 branch, batch-one empty
T2I latent `[1,128,128,256]`, and the exact five-value qualified schedule.

## Production lifecycle

The dedicated node enters normal ComfyUI preparation, model management,
conditioning, wrappers, and cleanup through `guider.sample()`, using a private
procedure sampler. That procedure owns an immutable Blueprint-only accepted
state and executes:

1. exact CPU-seeded Phase-25 initialization of `B0=[1,128,32,64]`;
2. four ordinary native-coordinate Blueprint Euler predictions;
3. the qualified CPU `float32` terminal bilinear mapping to `128x256`;
4. 55 row-major, end-aligned terminal regions;
5. one deterministic CPU-seeded `64x64` working state per region at sigma
   `0.25`, through the live CONST `noise_scaling()` API;
6. one ordinary native-coordinate FLUX prediction per working state;
7. exact `2x2` average restriction and immediate normalized overlap
   accumulation.

Only the Blueprint is accepted trajectory state. Working states, local
predictions, and the destination accumulator are transient. No destination-
sized model forward occurs.

## Qualification evidence

### Unit and legacy regression

- Focused terminal-resampling tests: **14/14 passed**.
- Complete custom-node suite: **36/36 passed**.
- Coverage includes registration coexistence, exact geometry/order/seeds,
  initialization, CONST construction, mapping/lift/restriction, streaming
  assembly equivalence, immutable H-free state, schedule/batch rejection,
  injected failure/nonfinite handling, and cancellation followed by a
  deterministic fresh retry.
- The existing Candidate-3 tests remain green. No legacy Candidate-3 sampling,
  state, geometry, DCT, policy, adapter, or executor file changed.

### Exact Phase-25 tensor regression

The persisted Phase-25 car/tree/house case is bit-exact at every available
boundary:

- initial Blueprint;
- all four Blueprint x0 predictions;
- all four accepted Blueprint states;
- terminal Blueprint x0;
- mapped terminal Blueprint;
- all 55 noise hashes;
- all 55 restricted prediction hashes;
- coverage and final H;
- decoded RGB hash.

The decoded RGB hash is
`7b156a83066b16ea627e715140794413633931ab0af7dfa4d5cbffd5a75d9bf6`.
The first trial localized a non-bit-exact CUDA-versus-CPU bilinear mapping
boundary (`3.19e-8` RMS). Restoring the research path's CPU float32 mapping
removed that divergence; the accepted qualification has no mismatch.

The astronaut path is also bit-exact to its Phase-25 artifact. The bridge
production rerun is not bit-exact to its reused Phase-24 artifact
(`0.024750` latent RMS), because that control predates the Phase-25 frozen
initialization provenance. Its freshly decoded production output nevertheless
retains the required S3 single bridge/train scene. This historical-provenance
difference is not treated as an exact Phase-25 implementation mismatch.

### Live node execution

Two fresh executions through the dedicated node and real BasicGuider produced
identical final latent hashes and identical decoded RGB hashes. Normal VAE
decode/save completed (with ComfyUI's automatic tiled-VAE fallback on this
canvas). Each execution recorded exactly four Blueprint predictions, 55 local
predictions, and zero destination-sized predictions.

The repeated decoded RGB hash is
`1397671637dba1552aac32c871efa03c4100cadeb5d869069e8876cbee0b4db8`.

The live first run recorded:

| Metric | Result |
|---|---:|
| Blueprint CUDA | 2.599 s |
| Local CUDA | 57.888 s |
| Node wall time | 68.874 s |
| Peak allocated | 1,597,753,344 bytes (1.488 GiB) |
| Peak reserved | 2,199,912,448 bytes (2.049 GiB) |
| Region barrier allocation | 1,071,588,352 bytes, identical for all 55 regions |

The flat barrier allocation directly verifies that completed W/x0/prediction
tensors do not accumulate with region ordinal.

### Semantic acceptance

| Fixed case | Production result | Evidence |
|---|---|---|
| Bridge/train | S3 | One dominant continuous bridge/train system; coherent horizon/water |
| Car/tree/house | S3 | Bit-exact Phase-25 selected output |
| Astronaut | S3 | Bit-exact Phase-25 selected output; one coherent centered body |

Expected softness from the fixed `0.25` refinement remains visible and is not
a Phase-25 semantic failure.

## Failure and cancellation behavior

No result or telemetry is published until all 55 predictions and final
coverage/finite checks pass. Interruption is checked before model work,
between Blueprint intervals, between local calls, and before final
publication. Injected model failure, nonfinite output, and cancellation leave
no successful destination; a fresh retry is deterministic. The implementation
uses no node-global mutable run state.

## Exclusions verified by telemetry

Every qualified run reports zero calls/instances for BlockDCT, persistent H,
hard terminal policy, `Flux2BlockExecutor`, global K/V context, periodic
resampling, post-anchor, and destination-sized model prediction.

## Files and core boundary

Production changes are limited to the custom-node target: the new staged
procedure, strict ordinary-call FLUX adapter, and node registration. ComfyUI
core was inspected and executed but not modified.

## Verdict

TERMINAL-RESAMPLING PRODUCTION SLICE QUALIFIED
