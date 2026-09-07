# Blueprint Diffusion user-facing integration qualification

## Verdict

**ACCEPT — ready for ordinary user testing inside the documented contract.**

The public node is `Blueprint Diffusion (Terminal Refine)`. It is a pixel-based
wrapper over the already-qualified `ConfigurableResamplingProcedure`, not a new
algorithm. The existing `Blueprint Configurable Prototype` registration and
latent-grid interface remain unchanged for saved-workflow compatibility.

## User contract

Destination width/height are derived from the connected empty FLUX.2 latent.
Users select `auto` or `manual`, seed, and refinement sigma. Manual mode exposes
pixel-space Blueprint width/height, tile footprint width/height, tile overlap
X/Y, and working-canvas width/height. Internal stride is `footprint-overlap`.
Region plan/order, coverage, interpolation, restriction, and the single local
interval remain automatic and fixed.

Every pixel geometry value must divide exactly by FLUX.2's 16-pixel latent
scale. Invalid relationships fail with specific errors rather than rounding.

Auto is an explicit five-profile allowlist:

| Destination pixels | G latent | F latent | Overlap latent | W latent |
|---|---:|---:|---:|---:|
| 2048x2048 | 45x45 | 32x32 | 16x16 | 64x64 |
| 4096x4096 | 48x48 | 32x32 | 8x8 | 64x64 |
| 4096x2048 | 64x32 | 32x32 | 8x8 | 64x64 |
| 2048x3072 portrait | 32x48 | 32x32 | 8x8 | 64x64 |
| 3072x2048 wide | 48x32 | 32x32 | 8x8 | 64x64 |

Widths precede heights in this user-facing table; telemetry stores H/W tensor
axes as height then width.

## Regression and live evidence

- Auto frozen-oracle configuration reproduces Phase 27 bit-exactly:
  `f0fc754078f56372043e29af20fc85449fb7025e00ce764bf123344d71aca3bd`.
- Existing 2048-square and 4096-square live fingerprints and geometry reports
  were revalidated unchanged.
- Portrait 2048x3072: `G=48x32` H/W tensor order, 40 regions, hash
  `50b0018732836797bea8f660af231d1f56d2ab8cba0c987fdb1d3fef4c5c400c`.
- Wide 3072x2048: `G=32x48` H/W tensor order, 40 regions, hash
  `446f12fe16e70329983f1ff5c6d00be0b9ff6ede2e1edcdfc7adb336091c9eb1`.
- Portrait and wide independently repeat bit-exactly and retain S3: one car,
  one tree, one house, continuous horizon, and no tiled miniature scenes.
- Both use four G calls, 40 `64x64` W calls, zero H-sized calls, complete
  normalized coverage, and zero-byte completed-region allocation range.
- Peak allocated/reserved memory is `3,286,629,888 / 3,741,319,168` bytes for
  both new aspect-ratio cases. Wall time is 43.93 s portrait and 42.35 s wide.

## Remaining limits

Klein 4B/CONST, BasicGuider CFG 1, batch-one unmasked empty-latent T2I, the
qualified four-step G schedule, terminal-denoised handoff, fixed interpolation,
integer F-to-W enlargement, sigma `0.10..0.50`, and one local `[sigma,0]`
interval remain mandatory. Unsupported auto destinations require manual mode;
manual geometry still fails outside the existing qualified bounds.
