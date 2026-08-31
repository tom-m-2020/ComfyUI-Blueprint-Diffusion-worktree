# FLUX.2 runtime per-batch crop-coordinate audit

Date: 2026-08-31

## Objective

Determine whether Blueprint can supply distinct absolute `(y,x)` coordinates
to local crops in one FLUX.2 batch through a narrowly scoped runtime override,
without editing ComfyUI core files or changing Candidate-3 semantics.

## Narrow override boundary

`Flux._forward` calls `self.process_img(...)` before transformer execution and
then consumes an `img_ids` tensor already shaped `[B,T,axes]`. The transformer
path therefore does not inherently require every batch element to share IDs.

The experiment temporarily replaced `process_img` on only the loaded FLUX.2
instance. A Blueprint-only transformer option carried one `(y,x)` pair per
batch element. The override preserved native patchification and constructed
one coordinate grid per element. It did not modify attention, transformer
blocks, weights, ComfyUI files, Candidate-3 state, or sampler lifecycle.

The override was active for one batched call inside a context manager and the
original bound method was restored in `finally`.

## Coordinate correctness

The probe used the Phase-6a 1024x2048 state at sigma 1.0 and selected:

```text
crop 0: y=0,  x=0   endpoints [0,0]   -> [31,31]
crop 4: y=24, x=24  endpoints [24,24] -> [55,55]
```

The batch input was `[2,128,32,32]`. For both crops, the runtime override's
`img_ids` matched the corresponding ordinary sequential `process_img` IDs
bit-exactly with maximum absolute difference zero. Per-batch absolute
coordinates are therefore technically expressible through a narrow
instance-level coordinate-preparation override.

## Prediction equivalence failure

Despite exact IDs, batched model predictions did not match the two sequential
predictions:

| Crop | Max absolute | RMS | Bit exact |
| --- | ---: | ---: | --- |
| 0 | 0.418861 | 0.029550 | no |
| 4 | 0.429688 | 0.035616 | no |

These differences are far beyond the fixed `2e-5` numerical-equivalence
tolerance and cannot be described as harmless operation-order drift.

An additional control batched the exact same crop twice with the same ordinary
scalar coordinates, without distinct IDs. Both batched outputs differed from
the sequential prediction by maximum `0.418861`, RMS `0.029550`. Therefore the
prediction failure is not caused by the coordinate override. Changing native
execution from batch size one to batch size two changes the current
guider/model/backend result materially even when input and coordinates repeat.

The current W4A8 path may make activation quantization or another
batch-sensitive backend operation relevant, but this probe does not attribute
the cause. Doing so requires a deeper guider/model/backend audit and cannot be
corrected in `Flux2Adapter` merely by preparing different IDs.

## Boundary and conclusion

The coordinate-preparation portion is narrowly runtime-patchable. The required
end-to-end claim—one batched call reproduces the existing sequential crop
prediction ensemble—is false for the qualified runtime.

Obtaining sequential-equivalent batched predictions would require changing or
special-casing execution below the coordinate adapter, or accepting a new
numerical/semantic algorithm. Either exceeds this task's permitted narrow
override. No production patch, trajectory, or performance benchmark was made.

Machine-readable evidence is in
`flux2_candidate3_runtime_coordinate_batch_results/report.json`.

## Verdict

Per-batch coordinate preparation is feasible, but equivalent crop batching is
not feasible through `Flux2Adapter` alone. Deeper guider/model/backend work is
required, so this investigation stops at the requested boundary.
