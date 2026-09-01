# Phase 6d — batched-trajectory semantic qualification

## Result

Experiment-only B=2 local-crop scheduling preserved the visible Candidate-3
scene semantics in both qualified stress scenes, despite the known numerical
B=1/B=2 model drift. It did not provide a meaningful warm wall-time benefit at
this three-crop geometry and increased peak CUDA allocator memory. Production
was not changed.

Machine-readable results and final decodes are under
`flux2_candidate3_batched_trajectory_results/`.

## Controlled implementation

The harness subclasses the Phase-3 terminal-release research sampler and
changes only local model-call scheduling:

- sequential: three B=1 crop calls at offsets `(0,0)`, `(0,24)`, `(0,32)`;
- batched: one B=2 call for `(0,0)` and `(0,24)`, then one B=1 call for
  `(0,32)`;
- the already-proven scoped `Flux.process_img` override supplies a distinct
  absolute image-ID/RoPE grid to every batch element and is restored afterward;
- all crops read the same immutable accepted H; only the assembled x0_H enters
  the Euler proposal;
- the low-density global call remains B=1 and unchanged;
- 24x48 block-DCT D/U, normalized overlap, four sigmas, hard nonterminal
  coupling, terminal release, and atomic `(G,H)` acceptance are unchanged.

The batched path executes the same 3,072 local image tokens per interval. It
reduces actual local invocations from three to two, not model arithmetic.

## Per-interval numerical comparison

### Bridge/train

| Step | assembled x0_H RMS / max | accepted H RMS / max | accepted G max | D(H)-G max | overlap RMS seq -> batch | projection RMS seq -> batch |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.03098 / 0.3320 | 0.000439 / 0.00570 | 0 | 7.15e-7 | 0.8490 -> 0.8516 | 0.02706 -> 0.02698 |
| 1 | 0.05649 / 2.2318 | 0.002546 / 0.08449 | 0 | 7.15e-7 | 0.6260 -> 0.6302 | 0.03932 -> 0.03935 |
| 2 | 0.04001 / 0.8120 | 0.004863 / 0.12599 | 0 | 4.77e-7 | 0.3977 -> 0.3956 | 0.07212 -> 0.07203 |
| 3 | 0.03624 / 2.1564 | 0.03624 / 2.1564 | 0 | terminal release | 0.24084 -> 0.24087 | 0.32006 -> 0.32030 |

### Person/car/tree

| Step | assembled x0_H RMS / max | accepted H RMS / max | accepted G max | D(H)-G max | overlap RMS seq -> batch | projection RMS seq -> batch |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.02664 / 0.3438 | 0.000442 / 0.00837 | 0 | 7.15e-7 | 1.2875 -> 1.2843 | 0.04177 -> 0.04166 |
| 1 | 0.04434 / 1.1767 | 0.002012 / 0.04454 | 0 | 5.96e-7 | 0.7577 -> 0.7568 | 0.03584 -> 0.03589 |
| 2 | 0.03714 / 0.8759 | 0.004519 / 0.09933 | 0 | 4.77e-7 | 0.4508 -> 0.4538 | 0.07448 -> 0.07452 |
| 3 | 0.03760 / 1.9778 | 0.03760 / 1.9778 | 0 | terminal release | 0.27143 -> 0.27225 | 0.36189 -> 0.36125 |

Accepted G is bit-exact across scheduling variants at every step because the
global trajectory is unchanged and independent of local execution. Hard
coupling reduces the local prediction drift strongly at nonterminal accepted
states while preserving the qualified D(H)=G tolerance. The terminal interval
intentionally releases H, so final latent RMS equals terminal assembled-x0
drift and D(H_final)=G_final is not required.

## Semantic inspection

The sequential and batched final decodes are visually extremely close.

- Bridge/train: both retain one continuous suspension bridge, one continuous
  centered train, the left lighthouse, the right stone tower, aligned deck and
  cables, and continuous water/horizon. No new duplicate, seam, or coordinate
  displacement is visible in the batched result.
- Person/car/tree: both retain one centered person, one dominant red car on the
  left, one dominant tree on the right, the same small distant vegetation, and
  coherent ground perspective. No new person/car/tree duplication, crop
  displacement, seam, or obvious fine-detail loss is visible.

This is positive semantic evidence for these two cases, not bit-exact or broad
perceptual equivalence.

## Performance and memory

| Scene/run order | Sequential wall | Batched wall | Apparent speedup | Peak alloc seq -> batch | Peak reserved seq -> batch |
|---|---:|---:|---:|---:|---:|
| Bridge/train, first-use sequential then batched | 11.87 s | 4.39 s | 2.70x | 6.14 -> 6.29 GiB | 6.36 -> 6.61 GiB |
| Person/car/tree, both after model warm-up | 4.48 s | 4.35 s | **1.03x** | 6.30 -> 6.45 GiB | 6.40 -> 6.69 GiB |

The bridge timing is not a valid steady-state speed claim because the
sequential run was the first trajectory after model load and absorbed first-use
kernel/setup effects. The subsequent warm comparison is the useful
discriminator: approximately 3% wall-time reduction is not material, while
peak allocated memory rises by about 155 MiB and peak reserved rises by about
297 MiB. At only three crops, one unchanged global call plus two local calls
leaves too little invocation overhead to amortize the larger B=2 execution.

## Conclusion

B=2 scheduling is semantically acceptable in the two requested scenes, but
the measured warm speed benefit is negligible and memory moves in the wrong
direction. This does not justify production integration at the qualified
1024x512 three-crop geometry.

**BATCHED EXECUTION NO MEANINGFUL SPEED BENEFIT**
