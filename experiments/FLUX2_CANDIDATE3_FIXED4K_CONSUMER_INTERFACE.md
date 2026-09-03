# Phase 17 — fixed-4K consumer-interface discriminator

Date: 2026-09-04

## Question

Does the same Phase-14 fixed 4,096-position source become semantically useful
when normalized W consumes it through stronger bidirectional native transformer
interaction rather than frozen external generated K/V?

This terminal-only, zero-update experiment preserves accepted `H=128×256`, G,
sigma, prompt, seed, conditioning, the deterministic `4×2` area-mean source,
full-canvas source coordinates, 55 reconstructed native `W=64×64` states,
assembly, and all provenance. Production and ComfyUI core are unchanged.

## Interfaces

### A — external-K/V control

Phase-15 C0 evidence is reused. One 4K source trajectory is shared. At every
block, W generated queries see ordinary text/local K/V plus frozen current-block
source generated K/V; ordinary text-query behavior is restored.

```text
source:  (512 + 4096) × (512 + 4096)
consumer augmented Q×K: 4608 × 8704
```

### B — bidirectional joint-source/W oracle

Source and W are separately prepared with their own exact positional embeddings,
then their generated hidden streams are concatenated behind one shared native
text stream. Every native double and single block updates source, W, and text
together. Source and W may query each other directly; only the W slice reaches
`final_layer`.

```text
ordering: text[512], source[4096], W[4096]
native joint Q×K: 8704 × 8704
```

Token identity and source/W coordinate systems remain explicit. Because source
hidden state becomes W-specific, the oracle evolves 55 separate source/W
trajectories. This is deliberately not an efficiency proposal.

## Mechanical gate and divergence

One representative region was run in lockstep through A and B before the full
assembly. At block 0, source and W K/V are bit-identical before attention; the
first difference therefore arises from the stronger attention output, not input
preparation or coordinates.

| Boundary | source hidden RMS diff | W hidden RMS diff | source K/V RMS diff | W K/V RMS diff |
|---|---:|---:|---:|---:|
| double 0 | 0.7148 | 0.0149 | 0 / 0 | 0 / 0 |
| double 4 | 2.8537 | 1.0064 | 0.6886 / 5.1589 | 0.1834 / 1.1960 |
| single 0 | 5.3997 | 2.4128 | 0.7810 / 0.3521 | 0.2282 / 0.1077 |
| single 9 | 6.8841 | 4.9991 | 1.0139 / 0.3769 | 0.6032 / 0.2545 |
| single 19 | 41.6351 | 42.9674 | 0.9529 / 0.9865 | 0.8334 / 0.8645 |

The representative final W prediction changes strongly: RMS `0.789191`, max
absolute `5.83178`, versus external K/V. The interface is causally active, and
source/W trajectories increasingly diverge with depth.

## Semantic result

The comparison is
`flux2_candidate3_fixed4k_consumer_interface_results/EXTERNAL_VS_JOINT.png`.

B materially changes organization. Compared with A's many small independent
fragments, B promotes larger, more internally continuous bridge/deck/cable
systems and reduces some small-fragment clutter. The stronger interface thus
extracts useful organizing behavior from the same source.

It does **not** pass the required semantic gate. B still contains several
independent bridge and train interpretations arranged in horizontal bands,
multiple towers/support systems, and discontinuous horizon/water. There is no
single dominant whole-canvas bridge system.

| Metric | A external K/V | B joint |
|---|---:|---:|
| terminal overlap RMS | 0.784053 | 0.766685 |
| assembled prediction RMS | 0.694729 | 0.709500 |

Overlap improves modestly but remains secondary to the visual failure.

## Cost and memory

| Metric | A external K/V | B joint oracle |
|---|---:|---:|
| terminal wall | 104.211 s | 142.178 s |
| transformer CUDA | external source/local 103.625 s | joint 141.956 s |
| W final projection CUDA | included in A total | 0.202 s |
| peak allocated | 3.79 GiB | 4.93 GiB |
| peak reserved | 5.89 GiB | 9.82 GiB |

Source/local CUDA cannot be separated for B: native joint QKV, attention,
residual, and MLP arithmetic update both streams in one block. Any split would
be fictitious. The machine report records all 25 block barriers and confirms
complete normalized coverage.

## Integrity

- Accepted H/G and every W hash match Phase 15.
- Source remains exactly 4,096 positions with unchanged complete provenance.
- 55 W-specific joint states execute exactly 25 blocks.
- Only W is final-projected; source never produces a prediction.
- Accepted H/G and W remain immutable; no terminal update occurs.
- Output is finite and coverage complete.

## Decision

**Stronger source↔W interaction materially improves organization but remains
fragmented.** The external-K/V consumer interface is not exonerated, but it is
also not the only bottleneck. Both fixed-source capacity/representation and the
normalized-W interface likely contribute.

Do not return to density or deterministic representation sweeps. The one
justified next discriminator is depth localization of the fixed joint interface:
identify where larger shared structures emerge and whether later depth loses or
fails to consolidate them.
