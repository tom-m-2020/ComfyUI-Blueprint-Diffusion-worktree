# Operating rules

## Project scope

This repository investigates and develops **Blueprint Diffusion** for
ComfyUI.

The central objective is to make high-resolution or arbitrary-size
diffusion more computationally practical without giving up coherent
whole-scene composition.

The working principle is:

1.  maintain a representation of the complete target canvas and its
    spatial relationships;
2.  obtain global composition/context substantially more cheaply than a
    normal dense full-resolution diffusion-model forward;
3.  spend expensive model computation selectively on local or
    higher-detail regions;
4.  allow local computation to remain informed by the global scene so
    independently processed regions do not drift into incompatible
    compositions;
5.  preserve a coherent persistent whole-canvas diffusion state across
    local/global updates;
6.  qualify image quality, semantic coherence, compute, wall-clock time,
    and VRAM separately.

This project is **research-first**. Do not assume that the correct
solution is ElasticDiffusion, MultiDiffusion, Mixture of Diffusers,
tiled diffusion, sparse-token execution, reduced-resolution guidance,
cached global features, a pyramid, or any other specific mechanism until
evidence supports it.

Blueprint Diffusion is not defined as a port of one upstream algorithm.

Do not broaden the project into unrelated image editing, model loading,
quantization, sampling, or optimization features without an explicit
task.

## Core definition

For this project, **Blueprint Diffusion** means the design space:

``` text
cheap whole-canvas planning/context
+
selective or local expensive diffusion
+
persistent coherent whole-canvas state
```

It is not sufficient to process independent tiles and blend them
afterward.

It is not sufficient to run an ordinary full-resolution global DiT
forward plus local forwards if the global forward still dominates
compute and memory.

A candidate method must explain:

-   what information represents the whole scene;
-   how that representation preserves spatial coordinates and long-range
    relationships;
-   how local/high-detail computation consumes global information;
-   how local results update the persistent whole-canvas state;
-   when and how often global information is recomputed;
-   why the global mechanism is materially cheaper than ordinary
    full-canvas diffusion.

## Writable paths

Codex may modify:

-   `./target/ComfyUI-Blueprint-Diffusion/`
-   `STATUS.md`
-   `FINDINGS.md`
-   `DECISIONS.md`
-   `docs/notes/`
-   `experiments/`

If a backend implementation directory is added later, it is not writable
merely because it exists. Its writable scope must be explicitly added
here or authorized by the user.

## Read-only paths

Never modify:

-   `./references/`
-   `../ComfyUI-SpotEdit/`
-   `../ComfyUI-Local-Edit/`
-   `../ComfyUI-Nunchaku-Klein/`
-   any sibling project unless explicitly authorized;
-   the approved ComfyUI development checkout;
-   runtime model files.

A junction or symlink into a read-only repository remains read-only.

## Previous-project knowledge

`ComfyUI-SpotEdit`, `ComfyUI-Local-Edit`, and `ComfyUI-Nunchaku-Klein`
are approved read-only technical references.

Codex may directly inspect, when relevant:

-   their `FINDINGS.md`;
-   their `DECISIONS.md`;
-   their `STATUS.md`;
-   their `docs/`;
-   their `experiments/`;
-   target source;
-   Nunchaku/backend source where applicable;
-   equivalent paths exposed through junctions under `./references/`.

Reuse established conclusions when they directly apply instead of
repeating expensive investigations unnecessarily.

Relevant prior knowledge may include:

-   native ComfyUI FLUX.2 execution boundaries;
-   FLUX.2 generated/reference token preparation and positional IDs;
-   rectangular token geometry;
-   sampler/model wrapper lifecycle;
-   SpotEdit sparse active-token execution;
-   rectangular query versus full-context K/V attention;
-   K/V cache topology and lifecycle;
-   Nunchaku FLUX.2 execution and attention limitations;
-   Local-Edit research concerning local/global context and spatial
    constraints.

However:

-   distinguish previous-project evidence from evidence established
    here;
-   check whether version-sensitive conclusions still apply;
-   do not silently import previous architectural decisions;
-   do not assume SpotEdit's sparse execution is automatically the
    correct Blueprint execution model;
-   do not modify previous projects while working here;
-   when prior evidence materially determines a design, record its
    provenance in this project's `FINDINGS.md` or `DECISIONS.md`.

## Research objective and terminology

Always distinguish the following concepts:

-   **full-canvas state** --- the complete latent/sample state being
    generated;
-   **global representation** --- information made available for
    whole-scene reasoning;
-   **global computation** --- model work used to construct or update
    that representation;
-   **local computation** --- expensive model work applied to a subset,
    crop, region, or selected token set;
-   **global coherence** --- consistency of composition, geometry,
    object relationships, perspective, lighting, and semantics across
    the complete canvas;
-   **local fidelity** --- detail and quality within high-resolution
    regions;
-   **token density** --- how densely the global canvas is represented
    spatially;
-   **compute reduction** --- reduction in executed operations/FLOPs or
    token work;
-   **VRAM reduction** --- reduction in peak or persistent GPU memory;
-   **wall-clock acceleration** --- actual end-to-end runtime
    improvement.

Do not collapse these into one claim such as "efficient" or "high
resolution."

A full-resolution latent stored in RAM or VRAM is not automatically a
useful global representation. If a model call sees only a local tile,
distant information is absent unless some explicit mechanism provides
it.

Likewise, a low-resolution global image is only one possible global
representation. Other candidates may include sparse global tokens,
cached features/K/V, hierarchical representations, semantic controls,
compressed features, or combinations.

## Reference hierarchy

This project has no single canonical upstream algorithm.

For research and implementation questions, use this order of authority:

1.  original papers and official implementations for the specific method
    being studied;
2.  current ComfyUI source for sampling, patching, model management,
    conditioning, and model contracts;
3.  original model implementations for model-specific architecture and
    conditioning;
4.  peer-reviewed or official research on arbitrary-size generation,
    global/local diffusion, tiled diffusion, sparse/token-reduced
    diffusion, multiscale generation, and related methods;
5.  previous local projects for already-established ComfyUI, FLUX.2,
    SpotEdit, Local-Edit, and Nunchaku behavior;
6.  third-party ComfyUI implementations as integration references;
7.  Nunchaku source for Nunchaku-specific execution behavior.

Report meaningful differences between papers, official implementations,
and ports.

Do not treat ElasticDiffusion, MultiDiffusion, or Mixture of Diffusers
as the definition of Blueprint Diffusion. They are reference methods and
baselines.

## Baseline and reference methods

Initial methods worth studying include:

-   ordinary full-canvas generation;
-   ordinary tiled diffusion/upscaling;
-   Mixture of Diffusers;
-   MultiDiffusion;
-   ElasticDiffusion;
-   relevant modern arbitrary-resolution or token-reduction methods;
-   SpotEdit's sparse transformer machinery as an engineering reference,
    not an algorithmic baseline for generation.

When evaluating a method, identify separately:

-   what constitutes its global information;
-   whether global information is computed from the complete canvas;
-   its spatial resolution or token density;
-   whether local regions interact directly or only through shared
    global information;
-   whether model predictions are computed densely or selectively;
-   how overlapping/local predictions are merged;
-   how diffusion state remains consistent across regions;
-   how often global information is recomputed;
-   whether the method depends on U-Net-specific or DiT-specific
    behavior;
-   whether it requires training or model modification.

## Candidate architecture classes

Treat the following as hypotheses to investigate, not predetermined
requirements:

-   reduced-resolution global latent or prediction;
-   hierarchical/pyramid global representations;
-   sparse or strided full-canvas tokens;
-   cached global hidden states or K/V;
-   global computation only during early denoising;
-   periodic/adaptive global refresh;
-   local queries attending to compact global context;
-   semantic or geometric scene representations alongside latent
    context;
-   combinations of global planning and selective token execution.

Do not implement several speculative architectures at once. Use
experiments to narrow the design space first.

## Efficiency requirement

A core project requirement is that the global mechanism must be
materially cheaper than ordinary dense full-resolution diffusion at the
target canvas size.

For every candidate, account for at least:

-   number of global tokens;
-   number of local/active tokens;
-   attention query/key dimensions;
-   number and frequency of model forwards;
-   projection/MLP work;
-   cache/storage cost;
-   transfer cost between CPU and GPU where applicable;
-   peak and persistent VRAM;
-   end-to-end step and sampling time.

Do not infer wall-clock or VRAM savings from skipped FLOPs.

Do not call a method successful merely because each local call is small
if the total work across all local calls equals or exceeds ordinary
full-canvas generation.

## Runtime model files

Read-only:

-   `C:/Users/Tom-M/data/a/ai/models-1/models/`

Codex may read and load tensor data from model files for controlled
runtime validation.

Codex may:

-   load diffusion models, text encoders, VAEs, LoRAs, and required
    model components;
-   construct native ComfyUI and qualified Nunchaku models;
-   execute real forward passes and sampling;
-   inspect tensor values, shapes, strides, dtypes, devices, hashes, and
    residency;
-   measure CUDA memory and execution time;
-   save diagnostic outputs, logs, metrics, and test images under
    `experiments/`.

Codex must not:

-   move, rename, modify, overwrite, convert, or delete model files;
-   write tensor data or metadata back into model files;
-   redistribute model weights.

Use the smallest suitable model, canvas, tile, token grid, and step
count first when a full run is unnecessary.

## Approved development environment

Codex may use:

-   ComfyUI checkout: `C:/Users/Tom-M/data/a/ai/apps/ComfyUI-dev`
-   Python: `C:/Users/Tom-M/miniconda3/envs/comfydev/python.exe`
-   optionally, portable ComfyUI:
    `C:/Users/Tom-M/data/a/ai/apps/ComfyUI_windows_portable`

Codex may import and execute ComfyUI directly from this checkout.

Codex may run native samplers, schedulers, guiders, conditioning paths,
VAE encode/decode, text encoders, diffusion models, controlled
native-versus-Nunchaku comparisons, and frontend/dev-server inspection
when required.

Do not modify the ComfyUI-dev checkout unless explicitly requested.

Do not install or upgrade packages without explicit approval.

## Web access

Web research is allowed when it materially helps with:

-   arbitrary-size or arbitrary-aspect diffusion research;
-   global/local, tiled, multiscale, sparse-token, or hierarchical
    diffusion methods;
-   current ComfyUI APIs;
-   model/backend documentation;
-   issues, pull requests, and implementation discussions;
-   licensing and provenance.

Prefer local source for exact implementation behavior when the relevant
revision is already available.

Record version-sensitive or research-derived conclusions in
`FINDINGS.md` when they affect design.

Do not download and execute arbitrary code without explicit approval.

## Development rules

1.  Inspect the complete relevant execution path before modifying it.
2.  Trace imports and call sites recursively when ownership or execution
    boundaries are unclear.
3.  Distinguish:
    -   algorithm/research policy;
    -   global-state representation;
    -   local-region selection/scheduling;
    -   model/transformer execution;
    -   sampler lifecycle;
    -   native ComfyUI functionality;
    -   model-specific functionality;
    -   backend-specific machinery;
    -   optional sparse/optimization machinery;
    -   experimental/debugging code.
4.  Prefer public ComfyUI APIs and patching boundaries when they can
    express the required behavior correctly.
5.  Do not create a custom sampler merely because it is easier if native
    sampling can support the required lifecycle correctly.
6.  Establish a simple full-compute or reference baseline before
    optimizing a candidate architecture where practical.
7.  Do not copy whole modules or directories without proving they are
    required.
8.  Check license and provenance before copying or adapting
    implementation code.
9.  Keep investigation code in `experiments/` until it demonstrates a
    production need.
10. Keep functional changes separate from cleanup and unrelated
    refactoring.
11. Do not claim semantic, numerical, performance, or VRAM equivalence
    without evidence.
12. Do not build generic infrastructure solely for hypothetical future
    methods.

## Architectural rules

Do not assume:

-   independent tiled denoising preserves global composition;
-   overlap and blending alone provide global reasoning;
-   downsampling automatically preserves all global information needed
    by the model;
-   fewer channels solve token-count/attention cost;
-   sparse tokens preserve geometry merely because they retain
    coordinates;
-   cached global features remain valid across arbitrary denoising
    intervals;
-   composition becomes fixed after a predetermined number of early
    steps;
-   SpotEdit's active-token execution can be reused unchanged for
    generation;
-   a high-resolution persistent latent means the model has
    high-resolution global context.

Each assumption must be tested or supported by source/research evidence.

Prefer architectures where global/local information flow is explicit and
inspectable.

## Native-first qualification

For a model family supported by native ComfyUI, establish native
behavior before implementing or qualifying a specialized backend where
practical.

FLUX.2 Klein is a useful initial DiT research target because previous
projects provide substantial execution knowledge, but do not hard-code
the entire project architecture around Klein unless evidence or the
current task requires it.

For Nunchaku support:

-   establish the algorithm and required execution contract natively
    first where feasible;
-   reuse prior Nunchaku knowledge instead of repeating solved audits;
-   do not assume existing attention callbacks can express selective
    execution;
-   keep backend modifications as narrow as the measured requirement
    allows.

## Code design philosophy

Prefer the smallest correct implementation.

Optimize for:

1.  global compositional coherence;
2.  local image quality;
3.  correctness of diffusion-state updates;
4.  measurable computational advantage;
5.  readability;
6.  debuggability;
7.  compatibility with native ComfyUI;
8.  minimal dependency and modification surface;
9.  backend optimization only after the algorithmic requirement is
    established.

Do not optimize for architectural sophistication or abstraction for its
own sake.

## Simplicity rules

-   Make the smallest change that satisfies the current verified
    requirement.
-   Reuse existing ComfyUI mechanisms before introducing parallel
    infrastructure.
-   Do not introduce a class when a function is sufficient.
-   Avoid factories, registries, dependency-injection systems, plugin
    frameworks, and generic dispatch layers without a current concrete
    requirement.
-   Do not create abstractions solely for hypothetical future features.
-   Prefer explicit control flow and ordinary Python data structures.
-   Avoid hidden global state and import-time side effects.
-   Avoid broad exception handling and silent fallback behavior.
-   Keep global/local sampling state visible and testable.

If a nontrivial abstraction is required, document why in `DECISIONS.md`.

## Human debuggability

Code must remain inspectable with ordinary source inspection, Python
tracebacks, breakpoints, logging, tensor inspection, intermediate
image/latent inspection, profilers, and CUDA-memory measurements.

Therefore:

-   make global and local token/grid mappings explicit;
-   make coordinate transforms explicit;
-   make global refresh and cache lifecycle explicit;
-   make full/global/local execution transitions observable;
-   validate token counts and mappings near their use;
-   log canvas, global, local, and token geometry where relevant;
-   include actionable tensor shapes, block names, sigma/step, and
    execution mode in errors;
-   preserve original exceptions with `raise ... from` when adding
    context.

## Testing

Before claiming a component works, choose tests appropriate to the
claim.

At minimum where relevant:

-   compile/import and node registration;
-   ModelPatcher clone/branch behavior;
-   square and rectangular canvases;
-   local-region coordinate mapping;
-   global-to-local mapping;
-   overlap/merge behavior;
-   real model forward execution;
-   native KSampler/SamplerCustom execution;
-   multiple denoising-step lifecycle;
-   full-versus-candidate controlled comparisons;
-   unload/reload and CUDA cleanup.

For image-quality qualification, test scenes that stress global
relationships:

-   one large subject crossing local-region boundaries;
-   multiple spatially separated subjects;
-   long straight structures or horizons crossing tiles;
-   perspective-sensitive architecture/interiors;
-   wide and tall aspect ratios;
-   repeated patterns where tiled methods can duplicate content;
-   large canvases where ordinary generation shows composition drift;
-   fine details requiring local high-resolution refinement.

Inspect separately:

-   global layout;
-   object count and placement;
-   subject continuity across local boundaries;
-   pose/anatomy continuity;
-   perspective;
-   lighting/color consistency;
-   seams;
-   local detail;
-   prompt adherence.

For performance claims additionally measure:

-   global/local executed token counts;
-   model forwards per diffusion step;
-   attention dimensions;
-   projection/MLP work where measurable;
-   peak and persistent VRAM;
-   cache memory;
-   data-transfer overhead;
-   step time;
-   end-to-end sampling time.

Control model, prompt, seed, sigma schedule, resolution, and sampling
configuration as tightly as practical when comparing methods.

## Research workflow

Before production implementation of a new architecture:

1.  state the exact limitation or hypothesis being investigated;
2.  identify the relevant baseline;
3.  separate confirmed facts from hypotheses;
4.  inspect original research and implementations;
5.  determine whether the mechanism depends on U-Net, DiT, sampler, or
    model-specific behavior;
6.  design the smallest experiment capable of falsifying the hypothesis;
7.  prefer instrumentation or experiment-only patches before production
    code;
8.  record reusable results in `FINDINGS.md`;
9.  record a design choice in `DECISIONS.md` only after evidence
    supports choosing it.

Do not commit to one Blueprint architecture before the design space has
been narrowed by evidence.

## Change discipline

Before production edits:

1.  identify the observed requirement;
2.  identify the minimum files that must change;
3.  identify existing code/API that can be reused;
4.  identify every proposed new abstraction;
5.  justify abstractions necessary now;
6.  state which baseline behaviors must remain unchanged.

After production edits:

1.  report files changed;
2.  summarize the behavioral change;
3.  identify new dependencies and abstractions;
4.  report whether baseline behavior changed;
5.  show or summarize the relevant diff;
6.  run focused tests before broad tests;
7.  report remaining unvalidated assumptions.

If implementation scope expands substantially beyond the evidence, stop
and re-evaluate.

## Documentation

Maintain three distinct project records.

### `STATUS.md`

Current state:

-   completed work;
-   active work;
-   blockers;
-   validation status;
-   next concrete milestones.

Do not turn it into a full research notebook.

### `FINDINGS.md`

Reusable technical facts supported by source inspection, research, or
experiments.

Clearly distinguish:

-   confirmed source behavior;
-   runtime evidence;
-   external research findings;
-   architectural inference;
-   hypotheses;
-   version-dependent behavior.

When carrying conclusions over from another project, identify
provenance.

### `DECISIONS.md`

Architectural/project choices and why alternatives were rejected.

Do not record every implementation detail as a decision.

After each meaningful task:

-   update `STATUS.md`;
-   add reusable discoveries to `FINDINGS.md`;
-   add architectural choices to `DECISIONS.md` when a real choice was
    made.

Do not rewrite historical conclusions merely to make the documents
cleaner. Append or narrowly amend them when new evidence changes a
conclusion.
