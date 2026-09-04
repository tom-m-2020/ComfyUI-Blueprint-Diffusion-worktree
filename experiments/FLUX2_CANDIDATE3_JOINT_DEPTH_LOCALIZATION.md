# Phase 18b — bounded joint-interface depth localization

## Question

Does the fixed-4096 bidirectional source/W interface contain a transient
coherent whole-scene solution which later transformer depth destroys, or can a
late joint tail repair an externally conditioned fragmented state?

This is a terminal-only, zero-update discriminator at the exact Phase-17
accepted state. Production and ComfyUI core are unchanged.

## Fixed contract

- accepted `H=128x256`; accepted `G` retained and immutable;
- Phase-14 `4x2` area-mean source, `32x128 = 4096` positions;
- 55 reconstructed native-coordinate `W=64x64` consumers;
- same prompt, seed `20260901`, conditioning, terminal sigma, coordinates,
  final W projection, restriction, and overlap assembly as Phase 17;
- native token layout is `512 text + 4096 source + 4096 W` during joint
  execution;
- no accepted-state update.

The full-joint reference was evaluated once. The bounded matrix contained only:

- prefix-joint/external-tail at S0, S9, and S19;
- external-prefix/joint-tail at D4, S0, S9, and S14.

D0/D4 prefix arms were not rerun. The aborted exploratory run had already
shown their legitimate but prohibitive costs (`1491.96 s` and `1157.48 s`).

## Resumable harness

Every reference/arm writes two artifacts before the next arm begins:

- a JSON completion record with configuration fingerprint and telemetry;
- a `torch.save` tensor artifact containing assembled output, all regional
  predictions, and representative diagnostics needed for later decode.

Both files are flushed and `fsync`ed, then atomically renamed. Resume rebuilds
the exact accepted state, validates a fingerprint covering the sigma schedule,
accepted H/G, source, every W tensor, and checkpoint list, then skips only
complete compatible artifacts. The final report and sheets were produced after
a decode-only resume which skipped all 8 completed model items.

## Results

Semantic scale: S0 is many small fragments; S1 is several larger bridge
systems; S2 is one dominant bridge with residual alternatives; S3 is one
coherent scene.

| Arm | Semantic | Overlap RMS | RMS vs full joint | Mean region RMS vs full | CUDA s | Wall s | Peak alloc/reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full joint reference | S1 | 0.766685 | 0 | 0 | — | 146.57 | — |
| Prefix joint through S0, external tail | S1 | 0.770050 | 0.223708 | 0.242092 | 154.60 | 154.61 | 9.80 / 10.99 |
| Prefix joint through S9, external tail | S1 | 0.773796 | 0.101851 | 0.109421 | 150.95 | 150.96 | 9.80 / 10.99 |
| Prefix joint through S19 | S1 | 0.766685 | 0 | 0 | 149.85 | 149.86 | 9.80 / 10.75 |
| External through D4, joint tail | S1 | 0.780861 | 0.479390 | 0.500202 | 139.03 | 139.39 | 5.82 / 9.79 |
| External through S0, joint tail | S1 | 0.772765 | 0.484318 | 0.508554 | 138.93 | 139.30 | 6.62 / 9.16 |
| External through S9, joint tail | S0 | 0.772313 | 0.512571 | 0.537726 | 121.74 | 122.13 | 6.62 / 9.16 |
| External through S14, joint tail | S0 | 0.779906 | 0.515917 | 0.541339 | 110.85 | 111.23 | 6.62 / 9.16 |

All outputs are finite, coverage is complete, accepted H/G and every prepared W
remain hash-identical, and no state was accepted. Prefix S19 is exactly the
full-joint reference (`RMS=0`, `max_abs=0`).

Representative full-joint source/W hidden RMS grows from `12.767/13.009` at S0
to `18.066/18.438` at S9 and `73.436/68.271` at S19. The checkpoint direct
`final_layer` decodes are explicitly diagnostic and out-of-distribution before
the final block; they do not show a hidden S2/S3 assembled solution.

## Semantic interpretation

Prefix S0 and S9 preserve the same weak S1 class seen at the full-joint
endpoint, but neither contains one dominant scene. Thus later joint depth is
not observed destroying a previously coherent S2/S3 organization.

An external prefix through D4 or S0 followed by a joint tail can reach the same
broad S1 category, so early joint double-stream interaction is not necessary
merely to reproduce Phase 17's limited improvement. It is not a useful success:
the result still contains several independent bridge/train systems. Delaying
the switch through S9 or S14 worsens the result to S0, showing that a very late
tail cannot repair fragmentation.

Overlap RMS is nearly flat and does not track these semantic differences; it
remains only a compatibility diagnostic.

## Decision

No bounded arm exceeds the full-joint S1 result and no S2/S3 transient exists.
The transformer-depth scheduling hypothesis is closed. Do not run another
joint block schedule or treat D4/S0 late-start parity with weak S1 as an
optimization qualification. The next architectural work should reassess what
information the fixed global representation/state must carry, rather than
searching for a joint-interface depth schedule.

Artifacts:

- `flux2_candidate3_joint_depth_localization_results/report.json`
- `flux2_candidate3_joint_depth_localization_results/arms/`
- `flux2_candidate3_joint_depth_localization_results/PREFIX_JOINT_EXTERNAL_TAIL.png`
- `flux2_candidate3_joint_depth_localization_results/EXTERNAL_PREFIX_JOINT_TAIL.png`
- `flux2_candidate3_joint_depth_localization_results/FULL_JOINT_W_CHECKPOINTS.png`

