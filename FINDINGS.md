# Findings

## 2026-08-29 — Phase 1 architecture and feasibility audit

Evidence and detailed reasoning are in `BLUEPRINT_ARCHITECTURE_AUDIT.md`.

### Confirmed current ComfyUI behavior

- Native sampling retains one full latent through guider/model prediction and
  sampler update. Model wrappers can coordinate an evaluation, but post-CFG or
  output masking does not reduce diffusion-model compute.
- Dense spatial cost begins at patch/input projection. Image QKV/projections,
  MLPs, normalization, modulation, residual arithmetic, and final projection
  scale with image-token count in addition to attention.
- Cache/refresh state cannot use raw model-call count as accepted-step identity;
  samplers such as Heun can perform multiple evaluations for one update.

### Confirmed model-source behavior

- FLUX.2 uses explicit generated/reference spatial IDs, double-stream then
  single-stream joint processing, and per-block `patches_replace["dit"]`.
- Z-Image uses a Lumina-derived single joint stream with context/noise refiners,
  token padding, 3-axis RoPE, and optional latent/SigLIP reference tokens. It
  requires a separate executor adapter rather than FLUX-shaped blocks.
- Anima uses image self-attention, separate text cross-attention, MLPs, 3-D
  RoPE, and an Anima-specific text adapter. It also requires a family adapter.
- Source classification supports a common sampler/global-local abstraction for
  all three families. Z-Image and Anima selected-token execution remain runtime
  unqualified.

### External research findings

- Mixture of Diffusers and MultiDiffusion preserve a shared canvas update and
  fuse crop predictions, but overlap/shared optimization is not whole-scene
  model context.
- ElasticDiffusion supplies evidence for low-resolution global plus local signal
  separation in Stable Diffusion U-Nets. Its conditional-minus-unconditional
  decomposition is not established for CFG-1 distilled joint DiTs.
- DyPE addresses dense high-resolution positional extrapolation and is
  complementary; it does not reduce target-canvas model compute or VRAM.

### Architectural inference and open hypothesis

- T2I and reference/edit can plausibly share lifecycle, geometry, fusion, and
  executor interfaces while using different global-context policies.
- Reference tokens are conditioning, not the same-canvas evolving trajectory.
  Successful editing therefore does not establish T2I global planning.
- The central unproven hypothesis is that a cheap low-density whole-canvas
  prediction can remain authoritative enough to prevent local high-resolution
  regions from inventing incompatible composition.
- Ranked experiments: (1) coarse global prediction plus local residuals;
  (2) low-density global K/V plus selected local queries; (3) coupled persistent
  low/high-resolution trajectories.

### Provenance

- ComfyUI source revision:
  `5ab2f7a2d676c1fb7b410c22e82e2ed8f217b56c`.
- Prior native FLUX.2 SpotEdit evidence is reused only for sparse executor,
  K/V-cache, coordinate, and lifecycle feasibility; it is not treated as a T2I
  algorithm or image-quality result.
