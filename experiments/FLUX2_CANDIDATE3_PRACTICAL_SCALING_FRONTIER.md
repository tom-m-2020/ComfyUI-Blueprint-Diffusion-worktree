# Phase 8a — Candidate-3 practical scaling frontier

## Scope

This phase characterized the unchanged production Candidate-3 sampler above
the previously qualified 1024x2048 canvas. It did not change crop policy,
global geometry, DCT operators, coupling, terminal release, sampler, adapter,
or backend. Every generation used FLUX.2 Klein, CFG 1, the same four-interval
full-denoise Euler contract, and matched prompt/seed/model state.

The ladder stopped at 2048x4096. Existing records and decoded outputs were
used; the ladder was not extended.

## Geometry and work growth

Dimensions are height x width. FLUX image tokens use latent grid `H`, and
production constructs `G=(3H_y/4, 3H_x/4)` through the qualified 4-to-3
block-DCT operator.

| Target | H shape | H tokens | G shape | G tokens | G/H | Crops | Local tokens / interval | Redundancy | BP forwards |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1536x2048 | 96x128 | 12,288 | 72x96 | 6,912 | 56.25% | 20 | 20,480 | 1.667x | 3 global + 80 local |
| 2048x2048 | 128x128 | 16,384 | 96x96 | 9,216 | 56.25% | 25 | 25,600 | 1.563x | 3 global + 100 local |
| 1536x3072 | 96x192 | 18,432 | 72x144 | 10,368 | 56.25% | 32 | 32,768 | 1.778x | 3 global + 128 local |
| 2048x3072 | 128x192 | 24,576 | 96x144 | 13,824 | 56.25% | 40 | 40,960 | 1.667x | 3 global + 160 local |
| 2048x4096 | 128x256 | 32,768 | 96x192 | 18,432 | 56.25% | 55 | 56,320 | 1.719x | 3 global + 220 local |

The global policy has no fixed token budget. Scaling each axis by 3/4 means G
always contains 9/16 of the H tokens. At 2048x4096, G has 18,432 tokens: more
than the complete H state at 1536x2048 (12,288), and more than twice the 8,192
tokens in the previously qualified 1024x2048 H state.

## Execution frontier

Times are sampling-only. The first four rows are warm. The final row uses
successful fresh-process/cold executions because immediate repeated execution
failed for both methods as described below. Memory is CUDA allocator GiB, not
physical free VRAM.

| Target | Dense status / time | Dense peak alloc / reserved | Blueprint status / time | BP peak alloc / reserved | BP global / local CUDA time |
|---|---:|---:|---:|---:|---:|
| 1536x2048 | SUCCESS / 19.29 s | 3.66 / 5.00 | SUCCESS / 27.13 s | 3.13 / 3.93 | 6.09 / 20.84 s |
| 2048x2048 | SUCCESS / 31.26 s | 4.08 / 5.85 | SUCCESS / 36.05 s | 3.38 / 4.41 | 9.62 / 26.17 s |
| 1536x3072 | SUCCESS / 38.12 s | 4.30 / 6.28 | SUCCESS / 45.32 s | 3.50 / 4.65 | 11.39 / 33.62 s |
| 2048x3072 | SUCCESS / 61.49 s | 4.93 / 7.55 | SUCCESS / 60.04 s | 3.87 / 5.38 | 17.57 / 42.05 s |
| 2048x4096 | SUCCESS / 102.72 s cold | 4.30 / 9.29 | SUCCESS / 104.40 s cold | 2.67 / 6.15 | 27.76 / 75.47 s |

Both methods completed the requested ladder in a fresh model state. No OOM
frontier was reached at or below 2048x4096. Blueprint used lower peak allocated
and reserved memory in every row. Runtime was slower through 1536x3072 and
approximately even at the two largest points; this is not a speed frontier.
Per-forward events, allocator baselines, interval telemetry, hashes, and crop
starts are retained in `report.json`. Token counts are not FLOP estimates.

### Warm-reuse/backend failure

After successful fresh 2048x4096 executions, an immediate repeated run of both
dense and Blueprint failed with:

```text
RuntimeError: Cannot set version_counter for inference tensor
```

This was not OOM. It is a repeated backend/model-reuse issue and does not
define either method's execution or semantic frontier. Successful fresh runs
provide the required timing and allocator telemetry.

## Integrity

Every successful Blueprint run had finite state, complete positive coverage,
one atomic acceptance per interval, three fresh nonterminal global forwards,
the expected local-forward count, terminal local-only release, and
nonterminal `D(H)=G` errors within production tolerance. Every decode had the
requested dimensions and a recorded RGB hash.

## Semantic frontier

- **1536x2048:** Blueprint remains coherent: one continuous bridge scene with
  compatible train, lighthouse, tower, horizon, and water organization.
- **2048x2048:** degradation begins, with competing bridge/cable fragments and
  alternative structures.
- **1536x3072:** global fragmentation is clear, including repeated independent
  bridge spans and floating structural alternatives.
- **2048x3072:** severe repeated bridge structures and incompatible spans are
  present.
- **2048x4096:** the bridge scene collapses into many repeated independent
  structures. The boundary-subject control similarly produces repeated
  independent astronauts and scene elements rather than one spanning subject.

Matched dense generations remain substantially more globally coherent at the
same sizes. The saved review sheets are `frontier_bridge_contact_sheet.png`
and `largest_semantic_contact_sheet.png`.

## Interpretation

This evidence does not isolate local crops as the cause. Crop count and local
work grow, but the observed failure is repetition and fragmentation of the
whole-scene plan, while Candidate-3 makes G authoritative at every
nonterminal acceptance through exact coarse coupling.

The next hypothesis is that the authoritative global trajectory has moved too
far beyond FLUX.2 Klein's useful spatial/token regime. G is globally
interacting, but grows without bound: 6,912, 9,216, 10,368, 13,824, then
18,432 tokens. More global tokens are not necessarily a better global plan
when the model is extrapolated beyond its qualified canvas.

This is a semantic architectural limit, not an execution OOM. Blueprint shows
real allocator-memory savings, but does not extend the practically usable
resolution frontier when its authoritative global plan fragments first.

## Next experiment

Run one bounded/adaptive global-state-density discriminator at a known failing
canvas while leaving H, crops, coupling, and terminal release unchanged.
Compare current unbounded G against one explicitly bounded whole-canvas G with
a mathematically exact D/U contract. Test whether keeping the global trajectory
inside a useful token regime restores one scene; do not run another ladder.

## Verdict

BLUEPRINT GLOBAL/LOCAL COHERENCE FAILS AT LARGE SCALE
