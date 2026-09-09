# Local Edit Fixed Multiband Compositor Discriminator

Date: 2026-09-09

## Question

Can a fixed multiband policy reconcile low-frequency source/generated disagreement over broader inward support while keeping high-frequency structural ownership confined to the qualified 24-pixel transition?

## Scope and fixed policy

This was a zero-diffusion, zero-VAE experiment on the previously qualified `rigid_bridge`, `organic_tree`, and `photometric_desert` artifacts. No affine/color normalization, geometric warp, mask optimization, or per-image tuning was performed.

The policy was fixed before outcome inspection: source footprint `[256,768)`, Gaussian radii 2 and 8 pixels, and inward raised-cosine supports of 24/64/128 pixels for high/mid/lowest-frequency bands. Source boundary values were edge-extended only for pyramid construction. The exact-source interior is `[384,640)` and generated pixels outside the source footprint are copied byte-for-byte. Controls were A generated, B hard composite, C fixed 24-pixel composite, and D multiband.

## Mechanical validation

Four focused tests pass: fixed policy constants and containment, deterministic monotonic inward weights, exact interior plus byte-exact exterior, and identical-input identity. All D outputs have exact-interior fraction `1.0`, generated-exterior bit exactness `true`, and exterior maximum error `0`.

## Results

| Case | Arm | Transition MAE | Left LF RMS | Right LF RMS | Left edge p95 | Right edge p95 |
|---|---:|---:|---:|---:|---:|---:|
| bridge | C | 0.006643 | 0.012584 | 0.016690 | 0.137120 | 0.095792 |
| bridge | D | 0.016228 | 0.011076 | 0.015315 | 0.138155 | 0.096379 |
| tree | C | 0.006597 | 0.068854 | 0.128607 | 0.101943 | 0.284756 |
| tree | D | 0.015999 | 0.074930 | 0.130035 | 0.102701 | 0.284814 |
| desert | C | 0.004704 | 0.012860 | 0.019044 | 0.049920 | 0.075306 |
| desert | D | 0.014864 | 0.018183 | 0.011024 | 0.049879 | 0.076043 |

D modestly reduces the bridge residuals and desert-right residual, but worsens desert-left by about 41%, worsens both tree residuals, and raises transition MAE by roughly 2.4-3.2 times. Edge p95 is nearly unchanged.

Visual inspection finds no new doubled bridge/tree contour or material cable/tree blur. However, D does not clearly improve either structural case and the desert's obvious full-height tonal band remains. The broader low-frequency support redistributes the mismatch farther inward rather than reconciling sky and sand. There is no generated-exterior change or loss of exact source beyond the declared supports.

## Verdict

**Fail.** D does not clearly improve the desert band over C and therefore fails the primary gate. Stop post-decode frequency-domain blending without tuning pyramid depth, radii, supports, or weights. The next eligible experiment is one fixed gradient-domain compositor under the same ownership contract.

Artifacts are in `experiments/local_edit_multiband_compositor_results/`; implementation and tests are `experiments/local_edit_multiband_compositor.py` and `experiments/test_local_edit_multiband_compositor.py`. No production code or registration changed.
