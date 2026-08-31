# Candidate-3 live ComfyUI workflow qualification

Date: 2026-08-31

## Scope

This qualification installed the Phase-4b package as an ordinary custom node
in the approved ComfyUI development instance and exercised it through the real
HTTP queue, websocket preview/events path, `SamplerCustomAdvanced`, normal
guider, and normal VAE decode/save path. The Candidate-3 coupling algorithm was
not changed.

Installation used a directory junction at:

```text
C:\Users\Tom-M\data\a\ai\apps\ComfyUI-dev\custom_nodes\ComfyUI-Blueprint-Diffusion
```

pointing to the production package under this repository's `target/` tree.
ComfyUI 0.33.0 was started on `127.0.0.1:8191` with automatic previews.

The machine-readable evidence is in
`live_comfyui_candidate3_results/report.json`. The executable qualification
harness is `live_comfyui_candidate3_qualification.py`.

## Registration and ordinary workflow

Live `/object_info` reported:

- node name: `BlueprintCandidate3EulerSampler`;
- display name: `Blueprint Candidate-3 Euler Sampler`;
- output: ordinary `SAMPLER`;
- category: `sampling/custom_sampling/samplers`;
- module: `custom_nodes.ComfyUI-Blueprint-Diffusion`.

The valid graph used `UNETLoader` (native FLUX.2 Klein 4B), `CLIPLoader`,
`CLIPTextEncode`, `BasicGuider`, `RandomNoise`, `Flux2Scheduler` with four
steps, `EmptyFlux2LatentImage` at 1024x512, the Blueprint sampler,
`SamplerCustomAdvanced`, `VAELoader`, `VAEDecode`, and `SaveImage`.

The ordinary loadable UI workflow is
`BLUEPRINT_CANDIDATE3_PHASE3C_WORKFLOW.json`. Its 12 node types were checked
against the running server and its 11 links were validated.

## Valid execution and reproducibility

| Run | Result | Preview frames | Wall time | Decoded RGB SHA-256 |
| --- | --- | ---: | ---: | --- |
| first valid | success | 4 | 29.28 s | `56419115731865bd530581c6d5adf4fb6c2b21f6cf10c9f2e91a90a31dfb65ed` |
| after `/free` unload/free-memory | success | 4 | 26.95 s | same |
| after invalids and cancellation | success | 4 | 5.68 s | same |

The hash is over decoded RGB pixels, not PNG container bytes, because ComfyUI
embeds run metadata in saved PNGs. All three executions therefore reproduced
the same decoded image exactly. Each valid run emitted four real websocket
binary preview frames, one for every Euler interval, and completed through the
normal VAE/save nodes.

The unload/reload control called ComfyUI's ordinary `/free` endpoint with model
and node-cache unloading before the second valid queue execution.

## Fail-closed workflows

Every invalid workflow failed with zero preview frames, before any accepted
sampling interval could be exposed or committed:

| Invalid workflow | Live error |
| --- | --- |
| wrong resolution | `Blueprint requires latent grid (32, 64), got (32, 48).` |
| CFG not 1 | `Blueprint first slice requires CFG exactly 1.0.` |
| wrong model family | `Blueprint first slice requires native ComfyUI Flux2, got Lumina2.` |
| mask/inpainting | `Blueprint first slice does not support masks.` |
| nonempty latent | `Blueprint first slice supports only empty-latent T2I.` |
| partial denoise | `Blueprint first slice requires sigma[0] exactly 1.0; partial denoise schedules are unsupported.` |
| wrong sigma count | `Blueprint first slice requires exactly four Euler intervals.` |
| incompatible/reversed schedule | `Blueprint sigma schedule must terminate at exactly zero.` |
| spatial conditioning | `Blueprint does not support positive conditioning keys ['area'].` |

The spatial-conditioning control used ComfyUI's real `ConditioningSetArea`
path. This directly qualifies rejection of extra spatial conditioning keys;
ControlNet and reference/image conditioning remain unsupported by the same
fail-closed conditioning-key boundary rather than being approximated.

Live qualification exposed one integration validation gap: the inherited
ComfyUI `max_denoise` heuristic was broad enough to accept a truncated schedule.
The fixed slice now additionally requires `sigma[0]` to be exactly 1.0. This is
a user-boundary validation change only; Euler, DCT, coupling, crop assembly,
and terminal release are unchanged.

## Cancellation and cleanup

The cancellation run was interrupted through ComfyUI's `/interrupt` endpoint
after its first binary preview. It produced `execution_interrupted`, no final
output, and no success event. A subsequent valid queue execution completed
with four previews and the same decoded-pixel hash as both earlier valid runs.
No sampler state is stored globally, so an interrupted or failed invocation
cannot publish a partial `(G,H)` state into a later queue execution.

## Focused tests

`python -m unittest tests.test_candidate3_production -v` passes all 10 tests.
The suite covers the DCT right inverse, constant preservation, normalized crop
coverage, immutable state/model inputs, nonterminal and terminal lifecycle,
the fixed schedule (including partial-denoise rejection), and unsupported
cases.

## Verdict

The exact fail-closed Phase-4b slice is usable as an ordinary ComfyUI sampler
node with the qualified native FLUX.2 Klein CFG-1 workflow. This result does
not qualify other geometry, samplers, CFG, editing, reference conditioning,
model families, optimization, or Nunchaku.

`LIVE WORKFLOW QUALIFIED`
