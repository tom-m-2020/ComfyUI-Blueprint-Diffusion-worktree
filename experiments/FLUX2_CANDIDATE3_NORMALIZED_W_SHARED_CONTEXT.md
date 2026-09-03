# Phase 11 — normalized native working canvas with shared transformer context

Date: 2026-09-03

## Question

Does fresh, same-sigma whole-scene context inside every local FLUX.2 block
resolve the semantic repetition of reconstructed native-scale working canvases,
and is the fixed 24×48 Blueprint state sufficient for that role?

This experiment changes no production or ComfyUI-core code. It keeps the exact
Phase-10b reconstructed-W lifecycle and compares only the source of external
generated-image context.

## Fixed contract

- Native ComfyUI FLUX.2 Klein 4B, CFG 1, bridge/train prompt, seed `20260901`.
- Four-step CONST-flow Euler sigmas:
  `1.0, 0.9498810172, 0.8633416295, 0.6780259013, 0.0`.
- Destination `H=64×128`; fixed Blueprint `G=24×48`.
- Fifteen destination regions, `32×32`, stride 24, deterministic end alignment.
- Each region reconstructs a fresh `64×64` W at every evaluation with the
  unchanged sigma-consistent Phase-9b lift.
- W uses native unit coordinates `0..63` on both axes.
- W prediction restriction is unchanged 2×2 mean; overlap assembly is
  unchanged.
- Candidate-3 G/H Euler proposals, hard nonterminal coupling, atomic
  acceptance, and terminal release are unchanged.
- The external-context mechanism is the previously qualified all-25-block
  generated-image K/V path. Ordinary text-query behavior is retained.

| Variant | Context source | Source tokens/block | Context policy |
|---|---:|---:|---|
| A_LOCAL_ONLY | none | 0 | ordinary reconstructed W |
| B_FIXED_G_CONTEXT | current accepted G | 1,152 | fresh source each interval, all 25 blocks |
| C_FULL_H_CONTEXT_ORACLE | current accepted H | 8,192 | fresh source each interval, all 25 blocks |

For B, G represents the complete H canvas with endpoint-preserving coordinate
scales `y=63/23` and `x=127/47`. For C, source H uses its native complete-canvas
coordinates. W always keeps its independent native local frame. Source x0 is
discarded; only restricted local W predictions enter H assembly.

## Provenance and integrity

- Initial H and G hashes are identical across A/B/C.
- Every B/C interval constructs a new context probe from the variant's current
  accepted source state and exact current sigma. The four probe identities are
  distinct in each run; no context survives to the next accepted interval.
- Each source captures 25 block K/V sets, and all 15 crop calls consume the
  matching context at every block (`25×15=375` consumptions per interval).
- Every recorded source hash equals its accepted G hash (B) or accepted H hash
  (C) before evaluation.
- W construction, region order, transport, coordinates, and assembly are common
  code paths. State hashes are checked before publication and all outputs are
  finite.
- There are three Candidate-3 global predictions, four context-source
  predictions for B/C, and 60 local W predictions per variant.

## Quantitative results

### Prediction overlap RMS by interval

| Variant | i0 | i1 | i2 | terminal i3 |
|---|---:|---:|---:|---:|
| A_LOCAL_ONLY | 0.913870 | 0.653408 | 0.491292 | 0.259399 |
| B_FIXED_G_CONTEXT | 0.846419 | 0.631362 | 0.469160 | 0.250149 |
| C_FULL_H_CONTEXT_ORACLE | 0.824706 | 0.636335 | 0.471722 | 0.266542 |

Both context sources materially change predictions. Final accepted-H RMS is
`0.278933` for A versus B, `0.514368` for A versus C, and `0.440736` for B
versus C. Terminal assembled-x0 has the same pairwise RMS because the terminal
Euler interval ends at zero and releases the local proposal.

Full-H context improves early overlap agreement most strongly, but terminal
overlap RMS alone does not predict semantic quality: C's terminal RMS is
slightly above A while its decoded global scene is substantially more coherent.

### Runtime and memory

| Variant | Candidate-3 global CUDA | Context-source CUDA | Local CUDA | Sampling wall | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| A_LOCAL_ONLY | 0.875 s | 0 | 59.323 s | 62.328 s | 2.874 GiB | 3.260 GiB |
| B_FIXED_G_CONTEXT | 0.882 s | 1.825 s | 97.100 s | 101.010 s | 2.990 GiB | 3.436 GiB |
| C_FULL_H_CONTEXT_ORACLE | 0.884 s | 13.114 s | 153.212 s | 168.559 s | 3.665 GiB | 4.373 GiB |

All variants execute 245,760 local W image tokens (`60×4096`). Context does
not reduce local token work. B stores about 0.330 GiB of CPU K/V per interval
and transfers about 19.78 GiB across the run. C stores about 2.344 GiB per
interval and transfers about 140.63 GiB. These are diagnostic costs, not a
practical architecture.

## Semantic observations

The final comparison is
`flux2_candidate3_normalized_w_shared_context_results/FINAL_COMPARISON.png`.

- **A_LOCAL_ONLY:** one broad bridge placement is visible, but the deck/train
  band contains repeated independently resolved truss, train, and support
  alternatives.
- **B_FIXED_G_CONTEXT:** modest cleanup and lower overlap disagreement are
  visible, but repeated red truss/train structures remain. It does not approach
  the semantic oracle.
- **C_FULL_H_CONTEXT_ORACLE:** the local predictions converge on one dominant
  continuous bridge system with controlled end towers, a much cleaner single
  deck/train organization, and coherent water/horizon. It is visibly smoother
  in open regions, but it passes the principal whole-scene uniqueness gate.

The first meaningful semantic reduction is already visible in C's interval-0
assembled prediction and persists through the accepted trajectory. B changes
interval-0 predictions and overlap statistics but does not cross the semantic
threshold.

## Conclusion

Inside-transformer shared global context is a viable mechanism for normalized
native working canvases. The full accepted-H oracle demonstrates that the
training-free scale transformation and reconstructed-W lifecycle can produce a
coherent whole-canvas result when sufficient current global information is
available.

The fixed 24×48 G representation does not carry enough information through the
same interface. This is decision gate 2: normalized local working canvases
remain viable, but **global representation compression is the current
bottleneck**. This does not authorize denser G, a compression sweep, or
production integration. It narrows the next research task to a bounded global
representation that approaches the full-H oracle while retaining a fixed model
token budget.
