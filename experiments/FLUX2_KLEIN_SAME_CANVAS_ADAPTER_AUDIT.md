# Phase 40 — FLUX.2 Klein 4B same-canvas adapter availability audit

## Executive result

**C — ONLY INCOMPATIBLE / SEPARATE-CANVAS ADAPTERS FOUND**

The bounded search found trained FLUX.2 Klein 4B RefControl LoRAs, but those
learn roles for two separately indexed Klein edit images: a control map and a
reference image. They do not expose an aligned latent/feature plane whose
trained meaning is “authoritative structure at these generated-canvas
coordinates.” Their documented LoRA coupling is also a range (`0.8–1.0`), not
an inherent or canonical fixed Blueprint coupling.

True same-canvas ControlNet releases found in this audit target FLUX.2-dev or
FLUX.1, while the available Klein ControlNet research implementation targets
9B and does not provide a qualified released 4B checkpoint. No candidate clears
all Phase-40 gates. Diffusion inference was therefore not run.

## Scope and method

The audit started from Phase 39's established boundary: native ComfyUI/Klein
hooks are not trained conditioning semantics. It searched official BFL
materials first, then Hugging Face and GitHub for released ControlNet-like,
residual, spatial-control, and same-canvas adapters.

This is an availability result as of 2026-09-06, not proof that no private or
future checkpoint can exist. A repository containing an injection class was
not counted as a trained release.

The deterministic audit harness validates the persisted qualified square
control before writing its result:

- mapped Blueprint SHA-256:
  `8a1ae79beeb93baa0f555cff7a65bd38774b254502649d898fc6a512c4143d05`;
- Phase-29 sigma-0.25 output SHA-256:
  `1b61a401451c5838cd0370897c9d9d4e838a23f497c76490f4549e68aecd1de3`;
- Phase-28 regression remains marked bit-exact;
- Phase-39 verdict and control fingerprints match.

## Candidate audit

| Candidate | Actual base | Trained | Representation and injection | Same-canvas authority? | Strength | Native qualified-path compatibility | Result |
|---|---|---:|---|---:|---|---|---|
| Official BFL Klein 4B / Base 4B | FLUX.2 Klein 4B | Yes | Text plus separately indexed reference-image tokens in native joint attention | No | Native edit behavior, but no same-canvas control coupling | Native references work, but are the rejected separate-canvas relation | Reject |
| `thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora` | Klein Base 4B LoRA | Yes | Depth control image + identity reference image via Klein edit inputs; LoRA-modified DiT | No | Recommended LoRA weight `0.8–1.0` | Requires reference-edit contract, not qualified BasicGuider T2I-only local path | Reject |
| `xocialize/refcontrol-FLUX.2-klein-4B-pose-lora` | Klein Base 4B LoRA | Yes | COCO-18 skeleton + identity reference through two edit images; rank-32 DiT LoRA | No | Catalog recommends `0.8–1.0` | Base-trained LoRA and edit references; not an aligned Blueprint latent interface | Reject |
| `alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union` | FLUX.2-dev 32B | Yes | Canny/HED/depth/pose/etc.; residuals on four double blocks | Yes | Recommended `0.65–0.80` | Checkpoint architecture incompatible with Klein 4B | Reject |
| `ReyChiaro/flux.2-klein-controlnet` | FLUX.2 Klein 9B research code | No qualifying 4B release | Aligned packed control embedding and zero-initialized block residual projections | Mechanically yes | Code default `1.0`; no qualifying 4B weights | Diffusers 9B research implementation, not native Klein 4B | Reject |
| FLUX.1 ControlNet/IP-Adapter families | FLUX.1 | Yes | FLUX.1-specific residual or attention adapters | Varies | Generally user-selected | Architecture incompatible | Reject |

### Architecture-matched RefControl details

The 4B RefControl releases are the closest candidates because their LoRA keys
target Klein Base 4B and the LoRA is genuinely trained. They still fail three
independent gates:

1. Klein's documented input format is **separate images** (`control →
   reference`), so using a Blueprint as if it were an aligned same-canvas
   latent would reinterpret the trained contract.
2. The controls are task-specific rendered representations such as depth or a
   COCO-18 pose skeleton. Converting a 128-channel Blueprint latent into one of
   those controls would require a new decoder/projection and semantic contract.
3. Coupling remains a documented LoRA-weight range rather than one fixed
   Blueprint authority value.

They therefore do not justify Phase-41 inference.

### Same-canvas ControlNet details

The FLUX.2-dev Fun Union checkpoint is a genuine trained spatial ControlNet. It
accepts full-frame Canny, HED, depth, pose, MLSD, scribble, gray, and inpaint
conditions and adds residuals to four double blocks. It is not a Klein 4B
checkpoint, however, and its model card recommends a conditioning-scale range.
Neither its weights nor its normalization can be transplanted to Klein 4B
without a new training/compatibility claim.

The Klein ControlNet GitHub implementation defines a plausible aligned control
branch, but its declared target is Klein 9B. Code and zero-initialized layers do
not establish a trained 4B interface; no qualifying released 4B checkpoint and
canonical preprocessing contract was found.

## Qualification-gate result

No candidate simultaneously establishes:

- released trained weights;
- FLUX.2 Klein 4B compatibility;
- same-canvas spatial alignment;
- exact recoverable preprocessing/normalization;
- a known injection path;
- direct Blueprint compatibility without a new projection;
- no arbitrary guidance metric;
- no reinterpretation of `reference_latents`;
- inherent or canonical fixed coupling.

Consequently there is no fixed Phase-41 discriminator to specify or execute.

## Compute and compatibility implications

- RefControl adds LoRA work and attention over extra reference-image tokens;
  it does not provide a cheap aligned control plane.
- FLUX.2-dev Fun ControlNet executes an additional residual-producing control
  branch and stores block residuals; it is both compute-positive and
  architecture-incompatible.
- The Klein 9B research ControlNet similarly implies an additional transformer
  branch and residual storage, even if appropriate weights later exist.

These are descriptive only. No runtime or VRAM claim was measured because all
candidates failed before inference.

## Integrity

- diffusion model forwards: `0`;
- local forwards: `0`;
- destination-sized forwards: `0`;
- decoded images: `0`;
- production changes: none;
- ComfyUI-core changes: none.

Machine-readable details are in
`experiments/flux2_klein_same_canvas_adapter_audit_results/report.json`.

## Decision and next phase

Close the Blueprint-as-guide adapter family for the base Klein 4B plus released
resources audited here. The exact next major phase should be a **separate
interleaved global↔local resampling architecture design discriminator**. It
must first define global/local accepted-state ownership, feedback mapping,
atomicity, and bounded-compute invariants; it should not silently reuse the
failed reference, K/V, projection, or terminal-resampling guide contracts.

## Sources

- [Official BFL FLUX.2 repository](https://github.com/black-forest-labs/flux2)
- [Official Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- [RefControl catalog and input contract](https://github.com/thedeoxen/refcontrol)
- [Klein 4B reference-depth RefControl](https://huggingface.co/thedeoxen/refcontrol-FLUX.2-klein-4B-reference-depth-lora)
- [Klein 4B pose RefControl](https://huggingface.co/xocialize/refcontrol-FLUX.2-klein-4B-pose-lora)
- [FLUX.2-dev Fun ControlNet Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
- [Klein 9B ControlNet research implementation](https://github.com/ReyChiaro/flux.2-klein-controlnet)

## Verdict

**C — ONLY INCOMPATIBLE / SEPARATE-CANVAS ADAPTERS FOUND**
