# FLUX.2 Candidate-3 performance characterization

Date: 2026-08-31

## Scope and controls

This Phase-6a run measures the existing Candidate-3 implementation without
changing its geometry, crop plan, coupling, terminal release, sampler lifecycle,
or execution strategy. The dense control and Blueprint used the same native
ComfyUI FLUX.2 Klein 4B W4A8 checkpoint, text conditioning, prompt, seed,
CFG 1, four-step Flux2 schedule, CONST/Euler equations, device, backend, and
process configuration.

Hardware/runtime:

- NVIDIA GeForce RTX 3060, 12 GiB;
- CUDA device 0;
- PyTorch 2.11.0+cu130;
- native ComfyUI FLUX.2 path with the same dynamic model residency/offload
  policy for both variants.

The experiment preloaded the same model object before each variant pair so
checkpoint loading is outside the sampler measurement. `cold` means the first
sample after unload/reload and allocator cleanup; `warm` is the immediately
following sample with the same loaded state. CUDA/kernel caches are
process-global, so only the first process run can include every first-use
kernel effect. Warm results are the primary comparison.

Sampling-only wall time begins inside the sampler immediately before initial
noise scaling and ends after inverse noise scaling. It excludes CLIP encoding,
checkpoint loading, VAE decode, and image saving. Model and CUDA tensor phases
use CUDA events; the wall boundary and event resolution synchronize CUDA.
Allocator baseline and peak allocated/reserved bytes are recorded separately.
These are PyTorch allocator measurements, not physical free VRAM.

The measurement implementation is experiment-only. Production Candidate-3
source was not modified in Phase 6a.

## Primary scaling table

Peak values are `allocated/reserved`. Token counts are executed spatial token
counts, not FLOP estimates.

| Target | H tokens | Blueprint G tokens | Crops | Executed local tokens | Redundancy | Dense forwards/int. | BP forwards/int. | Dense warm sample time | BP warm sample time | Dense peak alloc/reserved | BP peak alloc/reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 1024 | 576 | 1 | 1024 | 1.000 | 1 | 2 | 0.989 s | 1.754 s | 2.493/2.646 GiB | 2.499/2.684 GiB |
| 1024x512 | 2048 | 1152 | 3 | 3072 | 1.500 | 1 | 4 | 1.825 s | 4.103 s | 2.600/2.869 GiB | 2.518/2.707 GiB |
| 1280x512 | 2560 | 1440 | 3 | 3072 | 1.200 | 1 | 4 | 2.226 s | 4.382 s | 2.652/2.975 GiB | 2.551/2.742 GiB |
| 1024x2048 | 8192 | 4608 | 15 | 15360 | 1.875 | 1 | 16 | 10.233 s | 19.475 s | 3.238/4.154 GiB | 2.915/3.428 GiB |

At 512x512 Blueprint has no memory advantage: its peak allocation is 5.8 MiB
higher. At 1024x512, 1280x512, and 1024x2048, peak allocation is respectively
83.6, 103.7, and 331.0 MiB lower than dense. Peak reserved memory is 166, 238,
and 744 MiB lower at those geometries. The relative peak allocated reduction at
1024x2048 is 10.0%; reserved reduction is 17.5%.

Blueprint warm sampling is respectively 1.77x, 2.25x, 1.97x, and 1.90x the
dense warm time. The current implementation therefore demonstrates a modest,
growing peak-memory reduction, but no wall-clock acceleration.

## Cold and warm controls

Baseline and peak are `allocated/reserved GiB` for the warm run.

| Target | Variant | Cold sample | Warm sample | Warm baseline | Warm peak |
| --- | --- | ---: | ---: | ---: | ---: |
| 512x512 | dense | 1.888 s | 0.989 s | 2.325/2.646 | 2.493/2.646 |
| 512x512 | Blueprint | 1.861 s | 1.754 s | 2.325/2.684 | 2.499/2.684 |
| 1024x512 | dense | 1.833 s | 1.825 s | 2.326/2.869 | 2.600/2.869 |
| 1024x512 | Blueprint | 4.106 s | 4.103 s | 2.326/2.707 | 2.518/2.707 |
| 1280x512 | dense | 2.223 s | 2.226 s | 2.327/2.975 | 2.652/2.975 |
| 1280x512 | Blueprint | 4.381 s | 4.382 s | 2.327/2.742 | 2.551/2.742 |
| 1024x2048 | dense | 10.221 s | 10.233 s | 2.334/4.154 | 3.238/4.154 |
| 1024x2048 | Blueprint | 19.447 s | 19.475 s | 2.334/3.428 | 2.915/3.428 |

Baseline allocated residency differs by no more than 9 MiB across the matrix.
Reserved baselines reflect allocator retention from the preceding cold sample;
they are reported rather than treated as model-weight residency.

## Blueprint warm breakdown

All values are CUDA-event milliseconds across the complete four-interval
generation. `DCT` includes one-time qualification/initial restriction plus
coupling D/U and nonterminal invariant D. `Projection/coupling` contains the
remaining coarse delta and acceptance arithmetic, so it does not double-count
D/U. `Other coordinator` is synchronized wall time minus all recorded CUDA
ranges and includes CPU validation, scalar telemetry synchronization, callback,
event overhead, and crop planning/extraction.

| Target | Global forward | Sum local forwards | DCT | Assembly | Projection/coupling | Other coordinator | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512x512 | 744.1 | 982.1 | 18.3 | 2.2 | 0.0 | 6.5 | 1754.3 |
| 1024x512 | 1118.0 | 2941.1 | 29.9 | 4.9 | 0.1 | 7.4 | 4102.7 |
| 1280x512 | 1379.1 | 2954.6 | 35.2 | 4.7 | 0.1 | 7.2 | 4382.1 |
| 1024x2048 | 4439.6 | 14908.4 | 96.3 | 20.9 | 0.2 | 7.9 | 19474.6 |

Euler proposal arithmetic totals 0.95, 1.00, 0.86, and 1.08 ms respectively
and is included in each total but shown separately here to avoid rounding it
into the table's near-zero coupling column. Model forwards account for 98.4%
to 99.3% of Blueprint warm sampling time.

At 1024x2048, local forwards are 76.6% of measured Blueprint model time and
global forwards are 22.8% of total sampling time. DCT, assembly, Euler,
projection, and remaining coordinator work together are approximately 0.65%.
The dominant cost is therefore local crop model execution, not coupling.

## Per-interval warm CUDA telemetry

Values are milliseconds. Blueprint entries are `global / summed local`; dense
entries are the single full-canvas forward. Raw ranges, every individual crop
forward, sigma, coverage, projection RMS, and invariant result remain in
`report.json`.

| Target | Variant | Interval 0 | Interval 1 | Interval 2 | Interval 3 |
| --- | --- | ---: | ---: | ---: | ---: |
| 512x512 | dense | 247.25 | 245.89 | 245.51 | 248.24 |
| 512x512 | Blueprint | 186.62 / 245.51 | 185.65 / 246.43 | 185.65 / 244.87 | 186.15 / 245.26 |
| 1024x512 | dense | 458.16 | 453.48 | 456.62 | 454.73 |
| 1024x512 | Blueprint | 278.26 / 735.47 | 278.03 / 734.74 | 283.29 / 736.61 | 278.45 / 734.25 |
| 1280x512 | dense | 553.45 | 557.53 | 556.56 | 556.15 |
| 1280x512 | Blueprint | 345.57 / 738.34 | 344.55 / 739.34 | 344.68 / 739.33 | 344.30 / 737.59 |
| 1024x2048 | dense | 2576.61 | 2545.11 | 2549.63 | 2558.81 |
| 1024x2048 | Blueprint | 1110.12 / 3718.11 | 1109.42 / 3719.80 | 1108.30 / 3741.42 | 1111.76 / 3729.08 |

No meaningful late-step model-time decline appears under the fixed four-step
policy. The local/global balance is stable across intervals.

## Attention dimensions and token-work interpretation

The actual combined text/image attention dimensions were:

| Target | Dense QxK | Blueprint global QxK | Each local QxK |
| --- | ---: | ---: | ---: |
| 512x512 | 1536x1536 | 1088x1088 | 1536x1536 |
| 1024x512 | 2560x2560 | 1664x1664 | 1536x1536 |
| 1280x512 | 3072x3072 | 1952x1952 | 1536x1536 |
| 1024x2048 | 8704x8704 | 5120x5120 | 1536x1536 |

These dimensions and spatial token executions are not FLOP ratios. FLUX.2
also executes token-linear projections, MLPs, normalization, modulation,
residual operations, text tokens, backend kernels, and model-management work.
Sequential crops bound peak activation size but repeat full model overhead and
overlap tokens. The measured CUDA and allocator results, rather than token
counts, establish the performance conclusion.

## Integrity

All 16 samples reported `SUCCESS`. Every dense and Blueprint run produced four
callbacks and finite final state. Every Blueprint run retained positive full
coverage, all nonterminal `D(H_next)=G_next` checks within tolerance, and
terminal release only on interval three. Dense and Blueprint warm latents for
all four targets decoded to finite images after measurements completed.

Visual inspection of the 1024x2048 stress outputs found that Blueprint retained
one centered subject, one left car, one right tree, and a continuous ground/
horizon scene without obvious crop seams or duplicate major objects. Dense was
also coherent. This confirms that performance instrumentation did not break the
qualified composition mechanism; it is not a new multi-prompt quality study.

The 1024x2048 stress case did not OOM on this 12 GiB setup. No settings were
changed to make it complete.

## Interpretation and Phase-6b target

The current architecture buys a measurable peak-memory reduction at larger
canvases, growing to 331 MiB allocated and 744 MiB reserved at 1024x2048, while
retaining the previously qualified composition mechanism. It does not buy
compute or wall-clock reduction: repeated crop execution makes it roughly 1.9x
slower than dense at the stress geometry.

The single justified Phase-6b optimization target is **local crop model work**:
the number of local forwards and repeated local/overlap token execution. This
statement selects the measured cost center, not an optimization mechanism.
Global cadence, DCT/coupling, and assembly should not be the first target on
these measurements.
