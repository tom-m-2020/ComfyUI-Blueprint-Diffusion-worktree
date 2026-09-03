# Phase 8j — terminal specialized FLUX executor production qualification

## Result

The terminal-only specialized path is integrated at the fixed production
boundary selected in `PRODUCTION_SPECIALIZED_EXECUTOR_ARCHITECTURE.md`.
`BlueprintCoordinator` still owns accepted state, Euler arithmetic, coupling,
assembly, preview, validation, and commit. `Flux2Adapter.predict_regions()` is
an ordered loop for every ordinary evaluation and dispatches to a private
`Flux2BlockExecutor` only for terminal H=128×256 / G=96×192 with the qualified
55-crop layout.

The live native run completed twice with identical output-latent and decoded
RGB hashes. The decoded scene contains one dominant continuous bridge system,
a continuous deck/train, controlled major towers, and coherent water/horizon.
It does not reproduce the historical untiled-VAE RGB hash because normal VAE
decode exhausted available memory and ComfyUI selected its tiled fallback;
transformer/latent evidence is authoritative.

## Implementation contract

- 56 scoped descriptor preparations (one source, 55 crops) use normal ComfyUI
  CFG-1 condition preparation and a copied model-options dictionary. A scoped
  `DIFFUSION_MODEL` wrapper captures exactly one already-prepared native call
  and returns a discarded shape-correct sentinel.
- Source and all crop hidden states are advanced explicitly through five
  native double blocks and twenty native single blocks.
- Every local generated query sees text, its own 1,024 generated tokens, and
  all 18,432 positioned current-G K/V tokens. Text-query attention is restored
  to the ordinary native result.
- Only current-block source K/V is retained. The source final layer is omitted.
  No host K/V cache, host transfer, batch, pooling, or density reduction exists.
- The operation returns a complete `RegionPredictionSet` or raises. Coordinator
  assembly and atomic terminal acceptance happen only afterward.

## Correctness evidence

| Gate | Result |
|---|---:|
| Existing + focused unit tests | 22/22 pass |
| Exact terminal dispatch | pass |
| Nonterminal/other-geometry ordinary dispatch | pass |
| Injected bulk-evaluation failure leaves accepted state unchanged | pass |
| Four previews | pass |
| Nonterminal global forwards | exactly 3 |
| Terminal context source blocks | exactly 25 |
| Terminal global prediction/source final projection | neither performed |
| Nonterminal D(H)=G invariants | pass, ≤2e-6 |
| Final finite latent | pass |
| Repeated fresh-run latent hash | identical |
| Repeated decoded hash | identical |

The production final latent has RMS `0.709837019443512`, mean
`-0.06010382995009422`, and max absolute value `4.921250820159912`, exactly
matching the qualified Phase-8i assembled terminal result at all recorded
statistics. Phase 8i established bit-exact native execution at every ordinary
block, bit-exact source K/V and local hidden outputs for a context crop, and
55/55 bit-exact crop predictions. Production retains the same native module
ordering and arithmetic; only descriptor acquisition moved to the scoped
ComfyUI wrapper boundary.

## Runtime and memory

Second fresh-process qualification run:

| Measurement | Value |
|---|---:|
| Whole four-step sampling wall time | 152.45 s |
| Specialized source + terminal locals CUDA | 64.43 s |
| Peak allocated, whole sampling | 5,149,713,408 B (4.80 GiB) |
| Peak reserved, whole sampling | 7,174,356,992 B (6.68 GiB) |
| Current-block source K/V | 226,492,416 B (216 MiB) |
| All-layer CPU K/V cache | 0 |
| CPU→GPU K/V transfer | 0 |

Allocation is stable across crop consumption within each depth regime. The
expected one-time increase occurs at the double-to-single transition when each
retained crop state becomes a concatenated single-stream state. No per-crop
suspended call frame remains.

## Fail-closed scope

The specialized dispatch rejects a non-native model/base, any profile other
than native Klein 4B, non-single prepared positive conditioning, references,
ControlNet, attention masks, context handlers, transformer patches,
replacements or wrappers, and coordinate/shape drift. It never retries through
local-only terminal execution.

The live qualification exercised native sampling, previews, normal VAE decode
fallback/save, deterministic fresh-process repetition, and compact telemetry.
Cancellation and injected execution failures use the same exceptional path:
run-local references are cleared in `finally`, no prediction set is returned,
and the coordinator cannot assemble, preview, or commit. Unit injection verifies
accepted-state immutability and no interval publication.

Machine-readable evidence is in
`flux2_candidate3_terminal_specialized_production_results/report.json`.

## Verdict

TERMINAL SPECIALIZED EXECUTOR PRODUCTION QUALIFIED
