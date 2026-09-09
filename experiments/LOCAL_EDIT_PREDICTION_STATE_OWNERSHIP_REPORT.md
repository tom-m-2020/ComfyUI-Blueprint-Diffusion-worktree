# Local Edit Prediction Consistency vs State Ownership

Date: 2026-09-09

## Question

Does constraining both the locked-region prediction and accepted CONST trajectory make the editable region converge to geometry and photometry compatible with the source?

## Exact current execution order

The experiment traced and used the installed RES4LYF/ComfyUI path:

```text
accepted x_sigma
→ native Klein model input conversion and prediction
→ CFG/guider combination (CFG=1)
→ Comfy denoised x0_model
→ RES4LYF CONST derivative d_model=(x_sigma-x0_model)/sigma
→ Clown epsilon_projection (A/B)
→ C locked derivative replacement
→ deterministic Euler proposal
→ B/C exact locked source-state restoration
→ accepted x_sigma_next
```

Clown guidance remains a post-prediction sampler-derivative intervention. The model does not see the guide directly. Its projected correction is multiplied by the locked/source guide mask, so editable derivative coordinates remain the ordinary model derivative.

## Fixed experiment

The three previously qualified cases were rerun with their saved source canvases, prompts, seed 0, FLUX.2 Klein 4B W4A8, CFG 1, eight-step Flux2 schedule, RES4LYF deterministic `linear/euler`, and no noise mask.

- A `A_EPSILON_BASELINE`: current full-canvas `epsilon_projection`, no state restoration.
- B `B_STATE_RESTORATION`: unchanged `epsilon_projection`; restore locked coordinates to `(1-sigma_next)y + sigma_next z_L` after every Euler proposal and to clean `y` terminally.
- C `C_PREDICTION_AND_STATE`: before Euler, replace the locked derivative using

```text
d_C = M*d_model + (1-M)*(x_sigma-y)/sigma
```

  then apply exactly the same accepted-state restoration as B. `M=1` is editable and `L=1-M` is locked. `z_L` is captured once from the initial accepted state and reused.

No feathering, dilation, compositing, color correction, reference conditioning, K/V reuse, or sparse execution was used.

## Mechanical checks

Five tests pass: mask polarity, fixed-noise CONST endpoints, authoritative derivative-to-clean-source identity, Euler source-trajectory identity, and fixed two-latent-column boundary telemetry.

For B and C, locked accepted-state error is exactly zero after every acceptance and terminal locked latent error is zero in all cases. In C, locked predicted-`x0` and locked derivative error are zero at every sigma. B retains nonzero locked prediction error despite its exact accepted states; locked predicted-`x0` RMS ranges approximately `0.283-0.551` across cases/steps.

## Decisive B/C equivalence

B and C produce:

- identical raw model `x0` PNGs at every step in all three cases (24/24 comparisons);
- identical final decoded PNGs in every case;
- identical final decoded metrics;
- identical editable accepted trajectories by construction.

This follows directly from the lifecycle. B and C enter each model call with the same accepted state. C changes only locked derivative coordinates. The post-Euler restoration then replaces those same locked proposal coordinates in both arms with the identical source trajectory. Since the Clown projection correction in B is also spatially applied only on the locked mask, editable proposal coordinates are equal. Therefore both accepted states are equal, induction preserves equal future model predictions, and C cannot influence the editable region beyond B.

## Per-sigma prediction evidence

At step 0, A and C consume the same initial state, so editable effective-`x0` difference is exactly zero. After B/C restoration changes the next model-visible state, editable A↔C prediction RMS becomes nonzero at step 1 and grows:

| Case | Step 1 | Peak over steps 1-7 | Step 7 |
|---|---:|---:|---:|
| bridge | 0.345849 | 0.656890 | 0.647648 |
| tree | 0.147030 | 0.318779 | 0.318779 |
| desert | 0.099562 | 0.319224 | 0.318144 |

This is an effect of accepted-state ownership relative to free-running A, not an effect of C's source-authoritative derivative relative to B. B and C raw predictions remain identical.

Every raw and effective `x0` was decoded at all eight sigmas. The C effective predictions show the incompatible solution forming early and persisting:

- bridge: independent side bridge/tower geometry is already apparent in the step-0 prediction; by steps 3 and 7 it resolves into disconnected/restarted bridge sections at both source boundaries;
- tree: side illumination/ground statistics differ strongly at step 0 and settle into visible vertical ground/sky changes; right-side tree geometry grows independently of the locked tree edge;
- desert: side sky/sand luminance differs at step 0 and remains a vertical photometric ownership boundary through step 7.

C decoded boundary low-frequency RMS does not converge toward zero. At step 7 it is left/right `0.127089/0.155967` for bridge, `0.241806/0.194904` for tree, and `0.225758/0.220792` for desert.

## Final perceptual result

- A retains the previously qualified globally continuous epsilon-projection compositions, with source drift.
- B and C are visually identical.
- Bridge B/C restart bridge sections independently on both sides and show hard vertical joins.
- Tree B/C preserve the center source but create differing side illumination/ground and an independently resolved right tree continuation.
- Desert B/C preserve the astronaut/source center but retain obvious sky/sand photometric discontinuities and unrelated right-side shadow structure.

Decoded source-region MAE is not zero despite exact terminal source latent due to full-canvas VAE coupling, consistent with the earlier locality diagnostic.

## Verdict

**Fail: locked prediction consistency plus locked accepted-state ownership does not make the editable prediction compatible with the source.** C proves exact locked predicted `x0`, derivative, trajectory, and terminal latent authority, but adds no editable-region effect beyond B.

Hard projection alone is insufficient. The next research branch may investigate a model-level mechanism that makes editable prediction consume source context—such as a narrowly defined source-context/KV discriminator—without reopening pixel compositing, feathering, or additional sampler families. No production code changed.

Artifacts, final sheets, all raw/effective per-sigma `x0` decodes, and machine-readable telemetry are in `experiments/local_edit_prediction_state_ownership_results/`.
