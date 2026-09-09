# Local Edit Fixed Gradient-Domain Compositor Discriminator

Date: 2026-09-09

## Question and scope

Can bounded gradient-domain reconstruction remove the low-frequency seam while preserving source ownership and structural continuity better than direct RGB blending?

This zero-diffusion, zero-VAE test used the same eligible `rigid_bridge`, `organic_tree`, and `photometric_desert` artifacts. Controls were A generated, B hard source composite, C fixed 24-pixel inward raised-cosine composite, and D gradient-domain composite.

## Fixed policy

One policy was declared before inspection and shared by all cases: independent 64-pixel inward regions; a 2D per-channel Poisson solve; generated exterior and exact source interior as lateral Dirichlet boundaries; zero-normal handling at image top/bottom; and per-edge, per-channel selection of the larger absolute source/generated finite-difference gradient. Source values were edge-extended outside their footprint for guidance only. Exact source occupies `[320,704)` and generated exterior outside `[256,768)` is byte-exact.

There was no affine/color fitting, multiband processing, warp, tuning, mask optimization, or learned processing.

## Validation

Five tests pass: fixed support, mixed-gradient selection, unsigned-difference sign safety, exact ownership, and constant identical-input identity. An initial diagnostic run exposed unsigned subtraction overflow; it was corrected before perceptual interpretation and covered by a regression test.

All final D arms have exact-source-interior fraction `1.0`, generated-exterior bit exactness `true`, and exterior maximum error `0`.

## Measurements

| Case | Arm | Transition MAE | Left LF RMS/max | Right LF RMS/max | Left/right derivative discontinuity |
|---|---:|---:|---:|---:|---:|
| bridge | C | 0.013287 | 0.012584 / 0.031952 | 0.016690 / 0.029998 | 0.108146 / 0.065170 |
| bridge | D | 0.056821 | 0.012783 / 0.036543 | 0.053912 / 0.120776 | 0.109375 / 0.203874 |
| tree | C | 0.013194 | 0.068854 / 0.149351 | 0.128607 / 0.193603 | 0.035618 / 0.208309 |
| tree | D | 0.086055 | 0.071119 / 0.149351 | 0.133282 / 0.261068 | 0.040783 / 0.456322 |
| desert | C | 0.009408 | 0.012860 / 0.027651 | 0.019044 / 0.048982 | 0.031725 / 0.036199 |
| desert | D | 0.069332 | 0.016048 / 0.028566 | 0.108097 / 0.271382 | 0.038479 / 0.266339 |

Desert-right low-frequency RMS increases 5.68 times, maximum deviation 5.54 times, and derivative discontinuity 7.36 times. Bridge-right and tree-right discontinuities also increase strongly.

## Visual result and verdict

D produces an obvious dark full-height desert-right band and left tonal ramp. The tree acquires strong color/brightness contamination and a vertical halo. The bridge develops broad ramps and a conspicuous right discontinuity through sky, deck, and water. Exact ownership remains mechanically correct, but perceptual and structural gates fail.

**Fail.** Do not tune width, gradient rule, or boundary handling. Close post-decode compositor research completely and return the Local Edit investigation to diffusion-stage epsilon/projection. Do not productionize.

Artifacts are in `experiments/local_edit_gradient_compositor_results/`; implementation and tests are `experiments/local_edit_gradient_compositor.py` and `experiments/test_local_edit_gradient_compositor.py`.
