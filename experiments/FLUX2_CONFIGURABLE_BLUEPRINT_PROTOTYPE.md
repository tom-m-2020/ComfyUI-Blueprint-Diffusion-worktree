# Configurable Blueprint prototype qualification

## Verdict

**ACCEPT — the minimal configurable Terminal-Resampling sibling is qualified.**

`Blueprint Configurable Prototype` preserves the simple terminal-denoised
handoff architecture while exposing bounded G/F/stride/W geometry and one local
refinement sigma in `[0.10,0.50]`. The frozen Terminal node remains unchanged.

## Qualified production slice

- H dimensions come from the empty destination latent and may be `16..512` per
  latent axis.
- G axes are `16..64`, with at most 4096 model cells.
- W axes are `32..64`, with at most 4096 model cells.
- F must fit H; stride is the planner primitive and must be `1..F`.
- W must be an integer nearest-neighbor upscale of F on each axis. Restriction
  uses the matching area mean.
- G-to-F uses halo-aware bounded bilinear sampling. It does not construct a
  mapped destination anchor.
- The G schedule remains the frozen four-interval CONST schedule. Local
  refinement remains exactly one `[sigma,0]` interval.
- Only terminal-denoised handoff, native Klein 4B BasicGuider, CFG 1,
  positive-only empty-latent batch-one T2I, and unpatched conditioning are
  accepted.

Interpolation choice, multi-step local schedules, noisy handoff, masks,
reference/edit conditioning, batches, other models/schedulers, and W smaller
than F remain fail-closed rather than nominally generalized.

## Exact oracle regression

The exact Phase-25 configuration delegates to the frozen
`TerminalResamplingProcedure`. Through the full `guider.sample()` node path it
reproduced the persisted Phase-27 latent bit-exactly:

`f0fc754078f56372043e29af20fc85449fb7025e00ce764bf123344d71aca3bd`

RMS and maximum error are both zero. Candidate-3, Terminal Resampling, and the
specialized executor were not modified.

## New configurations

### Square, denser overlap

```text
G 45x45; H 128x128; F 32x32; stride 16x16; W 64x64
49 regions; overlap 16x16; sigma 0.25 -> 0
```

- Hash: `09ee4f8be0f88bb577a0a142ea9c5a298845019be552b09b380fd57c3acb75b4`
- Coverage: `1.0 .. 1.0000001192`
- Overlap RMS: `0.173297`
- Calls: 4 G, 49 W, 0 H-sized
- Sampling wall time: `51.80 s`
- Accepted-region allocation range: `0 bytes`
- S3: one car/tree/house, continuous field/horizon, no tiled repetition.
- Refinement raises gradient RMS from the plain Blueprint's `0.194260` to
  `0.221675` while retaining composition.

### Larger destination scaling case

```text
G 48x48; H 256x256; F 32x32; stride 24x24; W 64x64
121 regions; overlap 8x8; sigma 0.25 -> 0
```

- Hash: `ade54818b68d0092a640e358b66e3738935c6548b39500bd8e29bb9bb58934f4`
- Coverage: `0.9999998808 .. 1.0000001192`
- Overlap RMS: `0.165641`
- Calls: 4 G, 121 W, 0 H-sized
- Sampling wall time: `124.02 s`
- Accepted-region allocation range: `0 bytes`
- S3: one car/tree/house and one horizon, without miniature-scene repetition.
- Refinement raises gradient RMS from the plain Blueprint's `0.109416` to
  `0.167996`, with visible car, tree, house-edge, and grass development.

Both new outputs repeated bit-exactly. Increasing H from 128-square to
256-square increased local calls and sampling time, while model W stayed
`64x64`. Peak reserved memory stayed `3,755,999,232` bytes; peak allocation rose
from `3,273,531,904` to `3,350,625,792` bytes (about 77 MB) for the larger H
assembly. Region-barrier allocation was flat throughout each run.

## Next boundary

The next sibling feature should be a separately qualified local-schedule policy
(more than one sigma-to-zero interval), because the current prototype already
has explicit geometry, handoff, and noise-policy boundaries. It should not be
added implicitly to this qualification or conflated with noisy handoff.
