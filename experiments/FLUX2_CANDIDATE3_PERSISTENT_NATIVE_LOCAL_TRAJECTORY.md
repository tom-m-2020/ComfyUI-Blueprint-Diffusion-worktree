# Phase 10 — persistent native-scale local trajectory falsifier

## Verdict

Persistent independent W trajectories are substantially more fragmented than
the reconstructed-W control. Variant C was not run because B answers the
causal question cleanly: unconstrained native-scale persistence does not reduce
Phase-9 semantic repetition and its W/H drift is large, monotonic, and directly
accompanied by cross-region divergence.

**PERSISTENT W IS MORE FRAGMENTED — PERSISTENCE ALONE IS INSUFFICIENT**

Production and ComfyUI core are unchanged.

## Fixed configuration

- native FLUX.2 Klein 4B;
- bridge/train prompt, seed `20260901`, CFG 1;
- qualified four-step CONST-flow Euler schedule;
- destination `H=64x128`, fixed `G=24x48` through the existing experimental
  8-to-3 block-DCT Candidate-3 lifecycle;
- fifteen `32x32` regions at stride 24;
- one `64x64` W per region, native local coordinates `0..63`;
- no external/global K/V context;
- identical global calls, crop order, restriction, overlap assembly, hard
  nonterminal H/G coupling, and terminal release.

At initialization, both variants construct the same sigma-consistent W values
from H0. Across all regions, `D(W0)=H0_crop` within `1e-6`.

## State contracts

### A — reconstructed W control

At every interval and for every crop, A creates W from the currently accepted
H crop using the Phase-9b coarse-preserving lift and the deterministic
step/region noise. It evaluates W, restricts x0_W, assembles x0_H, and follows
the ordinary Candidate-3 H/G acceptance. W has no accepted persistence.

### B — persistent independent W

Fifteen W tensors are created once. For interval i, every model call reads its
immutable accepted `W_r,i`. After all local predictions and the candidate H/G
state exist, each W proposal is computed by ordinary Euler:

```text
W_r,* = W_r,i + (sigma_next-sigma) * (W_r,i-x0_W_r) / sigma
```

Only after all H/G/W proposals are finite and the accepted hashes remain
unchanged are the new H/G/W references published together. Each W has a
recorded parent object id/hash, prediction hash, and child object id/hash.
Every region receives exactly four Euler updates. No W is regenerated after
initialization and no crop commits independently.

## Runtime and work

| Variant | Global forwards | Local forwards | Local token executions | Global CUDA | Local CUDA | Wall | Peak alloc/reserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| A reconstructed | 3 | 60 | 245,760 | 1.067 s | 60.438 s | 62.972 s | 2.903 / 3.291 GiB |
| B persistent | 3 | 60 | 245,760 | 1.147 s | 60.373 s | 64.359 s | 2.939 / 3.299 GiB |

Persistence does not reduce model work in this experiment. Its extra state
storage slightly increases peak allocation. Timing is characterization only;
the semantic gate is decisive.

## Trajectory diagnostics

### Destination-prediction overlap

| Accepted interval | A reconstructed | B persistent |
|---:|---:|---:|
| 0 | 0.913870 | 0.913870 |
| 1 | 0.653408 | 0.817340 |
| 2 | 0.491292 | 0.842901 |
| 3 | 0.259399 | 0.803052 |

The identical interval-0 values confirm the shared W initialization and model
path. A's independently reconstructed predictions become progressively more
compatible. B stops converging after the first update and remains highly
inconsistent at terminal sigma.

### Persistent W/H drift

| Accepted state after interval | Mean regional RMS | Maximum regional RMS | D(W) overlap RMS |
|---:|---:|---:|---:|
| 0 | 0.050 | 0.0526 | 0.0526 |
| 1 | 0.11 | 0.1186 | 0.10 |
| 2 | 0.27 | 0.2874 | 0.25 |
| 3 terminal | 0.43 | 0.5234 | 0.8031 |

The first nonzero drift appears after the first independently proposed W and
coupled H acceptance. It grows every interval. Nonterminal H/G projection RMS
also grows more in B (approximately `0.04, 0.07, 0.17`) than A
(`0.04, 0.05, 0.12`). All nonterminal `D(H)=G` invariants remain within the
qualified tolerance, so this is specifically W-versus-destination divergence,
not corruption of the existing Candidate-3 state.

## Semantic inspection

A retains a principal horizontal bridge/horizon but still shows the known
repeated train/support alternatives and terminal artifacts.

B is qualitatively much worse. By the final state it contains multiple
independent bridge scenes at different vertical positions and scales, repeated
towers and supports, incompatible water/sky bands, and severe cross-region
ghosting. Local texture remains present, but it belongs to mutually divergent
regional trajectories rather than one scene.

The decoded trajectory and restricted D(W) views show fragmentation developing
after persistence begins, matching the numerical drift and overlap evidence.
This is not a borderline case where reconciliation is needed merely to make B
interpretable. Therefore the conditional C experiment is not authorized by the
task's gate.

## Interpretation

Reconstructing W at each evaluation was not the sole missing state mechanism.
Native-scale persistence without cross-region/global synchronization allows
each local trajectory to establish its own semantic scene. The result supports
neither advancing independent persistent W nor treating minimal post-step
coarse reconciliation as already justified.

Training-free fixed-working geometry remains unqualified. If this direction is
pursued further, it requires an explicitly designed global/cross-region
coupling contract for persistent W states; persistence alone is rejected.

All accepted H images, representative W states, every restricted D(W) view,
lineage telemetry, and machine-readable metrics are under
`flux2_candidate3_persistent_native_local_trajectory_results/`.
