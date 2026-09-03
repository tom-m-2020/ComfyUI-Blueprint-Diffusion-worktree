# FLUX.2 Candidate-3 block-major full-context streaming feasibility

## Verdict

**EXACT STREAMING REQUIRES A SPECIALIZED FLUX EXECUTOR.**

The exact block-major schedule is expressible at the existing attention patch
boundary, but it is not executable on the 12 GiB device by suspending 55
ordinary native FLUX.2 forwards. The probe OOMs in double block 0 before all
local crops reach the first clean barrier. Native call-frame suspension retains
Q/K/V, normalization/modulation, attention, residual, and other block-local
temporaries for every crop—not merely the intended block-to-block hidden state.

The one-block full-G K/V itself is only 216 MiB. The failed native execution
reaches 16.30 GiB peak allocated, a 15.31 GiB increase over its measured
streaming baseline. This establishes that exact streaming needs an executor
which explicitly owns source and crop hidden states at block boundaries rather
than pausing complete model calls.

No crop-hidden CPU offload, semantic compression, alternate context policy, or
production/core modification was attempted.

## Fixed controls

```text
output:                 2048×4096
H / G:                  128×256 / 96×192
seed:                   20260901
CFG / schedule:         1 / qualified four-step CONST-flow Euler
accepted state:         Phase-8d–8g H3/G3
terminal crops:         55, 32×32, stride 24
source context:         all 18,432 positions
context depth:          double 0–4 and single 0–19
```

Production intervals 0–2 execute once. Runtime hashes for accepted H3/G3 and
all terminal crop inputs are stored in `report.json`.

## CPU-offloaded reference

The reference reruns the exact Phase-8d mechanism before the streaming attempt:

```text
source x0_G RMS:                   0.721158
assembled terminal x0_H RMS:      0.709837
overlap RMS:                       0.281238
coverage:                          0.99999994 .. 1.00000012
CPU source cache:                  5,662,310,400 bytes / 5.27 GiB
aggregate CPU→GPU K/V transfer:  311,427,072,000 bytes / 290.04 GiB
source CUDA:                       11.60 s
terminal-local CUDA:              94.79 s
terminal-local wall:              95.11 s
```

Its tensor statistics and controls reproduce the established full-context
reference. The qualified decoded RGB hash remains:

```text
53cad700c9378317278ee3e609a00f8a0d906b3e1db243e3de971b8256f259ce
```

## Experiment-local block-major mechanism

The probe uses native model calls and existing experiment-only attention
patches; ComfyUI and production source are unchanged.

1. One source thread advances until a source block's ordinary pre-attention
   Q/K/V boundary.
2. It applies the unchanged source RoPE and publishes only that block's full
   18,432 generated K/V on GPU.
3. Fifty-five local threads advance native crop calls to the matching block.
4. Each computes ordinary local/text attention for text restoration and the
   exact Phase-8d augmented attention for generated queries.
5. The source waits for all 55 consumers before releasing K/V and advancing.

Thread-local ordinary-attention state prevents crops from sharing patch state.
Every worker uses inference mode, the accepted H3 view, identical sigma and
conditioning, original local coordinates, and the same source K/V objects.
There is no CPU source cache or CPU→GPU K/V transfer in the streaming path.

This arrangement is a feasibility probe, not a proposed runtime architecture.

## Failure boundary

The native-call strategy fails during double block 0:

```text
source blocks reached:                  1 / 25
local attention calls reached:         46 / 55 for block 0
clean all-crop barriers completed:      0
one-block full-context K/V residency:   226,492,416 bytes / 216 MiB
streaming baseline allocated/reserved:  1.00 / 1.10 GiB
peak allocated/reserved:               16.30 / 16.71 GiB
peak increment over baseline:           15.31 GiB
wall/CUDA until failure:                27.22 / 27.21 s
```

The first recorded allocation failures request 118 MiB; subsequent workers
also fail requesting 36 MiB and 18 MiB. CUDA reports no physically free memory
on the 12 GiB GPU. The reported PyTorch allocation exceeds physical capacity
under the backend's managed/offloaded execution, but the causal comparison is
clear: the 216 MiB source K/V is not the dominant resident object.

All 55 local workers eventually terminate due to OOM or the propagated source
failure. No sampler state is committed and no output is decoded.

## Resident-memory interpretation

The existing attention hook occurs inside a complete native block call. Pausing
a crop there retains more than its persistent hidden state:

- local image and text hidden inputs;
- modulation/normalization products;
- projected Q/K/V;
- ordinary-attention output retained for text restoration;
- concatenated augmented K/V and attention temporaries;
- native block call-frame residual/MLP intermediates;
- the source call frame and one block's global K/V.

Because OOM occurs before the first completed block barrier, persistent crop
hidden memory, text-state memory, and temporary attention memory cannot be
measured independently at a clean boundary. The directly measured aggregate
increment is 15.31 GiB. Claiming a smaller persistent-only number from tensor
shape estimates would violate the experiment's measurement requirement.

## Correctness status

The intended semantics were not contradicted; they were not reached. The probe
publishes the first source block's full 18,432-token K/V and 46 crop attention
calls consume that exact object, with zero host transfer. It cannot complete
block 0, final projections, assembly, or decoding. Therefore:

- semantic equivalence: **not established**;
- assembled/per-crop differences: **unavailable due OOM**;
- memory-residency reduction: **source cache/transfer eliminated, but aggregate
  native-frame residency is unacceptable**;
- wall-clock acceleration: **not measurable**;
- VRAM qualification: **failed**.

## Architectural implication

An exact implementation would require a specialized FLUX.2 executor with
explicit block entry/exit boundaries. It must retain only each crop's required
block-to-block image/text hidden state, execute one crop at a time within the
current block, release its block-local Q/K/V/MLP temporaries, then move to the
next crop. The current monolithic native forward API cannot expose that
lifecycle without retaining complete suspended call frames.

This conclusion does not authorize that executor. Its design, numerical
equivalence, memory accounting, backend compatibility, and provenance would be
a separate task.

## Artifacts

- `flux2_candidate3_block_major_context_streaming.py`
- `flux2_candidate3_block_major_context_streaming_results/report.json`

No comparison image was created because the streaming path did not produce an
assembled output.

**EXACT STREAMING REQUIRES A SPECIALIZED FLUX EXECUTOR**
