# ComfyUI-Blueprint-Diffusion

## Project purpose

This project investigates and develops **Blueprint Diffusion**: a
training-free global/local diffusion strategy intended to make large,
high-resolution, or unusual aspect-ratio image generation more
computationally practical while retaining coherent whole-scene
composition.

The project begins as a research project. It is not initially a
commitment to one published algorithm or one implementation
architecture.

## Problem

Ordinary diffusion models process the complete image representation
through the model at every denoising evaluation.

At large resolutions this is expensive, especially for transformer-based
diffusion models where increasing the number of spatial tokens can make
attention and other token-wise computation costly.

A straightforward tiled approach reduces the size of each model call,
but each tile has limited knowledge of distant parts of the scene. This
can cause:

-   duplicated or missing subjects;
-   incompatible object placement;
-   inconsistent pose or anatomy across regions;
-   perspective or horizon discontinuities;
-   locally plausible regions that do not form one coherent image;
-   seams or transitions that require post-hoc blending rather than
    model-level consistency.

The central question is therefore not merely how to divide an image into
smaller pieces.

It is:

> Can a pretrained diffusion model maintain coherent whole-canvas
> geometry and semantics using a cheap global representation while
> expensive denoising computation is concentrated on local or
> higher-detail regions?

## Working definition

Blueprint Diffusion is the design space:

``` text
cheap whole-canvas planning/context
        ↓
globally informed selective/local diffusion
        ↓
persistent coherent whole-canvas state
```

The global representation does **not** have to be a low-resolution
image.

It only needs to be cheap enough to remain practical while preserving
enough whole-scene information for local computation.

Possible representations include:

-   reduced-resolution latent or model prediction;
-   sparse or strided full-canvas tokens;
-   hierarchical/pyramid features;
-   cached global hidden states or K/V;
-   compressed learned scene tokens;
-   semantic/geometric scene information;
-   combinations of these.

These are research candidates, not project requirements.

## What this project is not

Blueprint Diffusion is not defined as:

-   ordinary tiled diffusion;
-   tiled upscaling;
-   independent crop generation followed by blending;
-   an ElasticDiffusion port;
-   a MultiDiffusion port;
-   a Mixture of Diffusers port;
-   SpotEdit applied to text-to-image generation;
-   a specific sparse-attention implementation.

Existing methods are references, baselines, and sources of mechanisms.

The project may eventually reproduce or adapt parts of them, but the
architecture must be selected from evidence.

## Primary goals

### 1. Preserve global composition

Local/high-resolution computation must remain informed by the complete
scene strongly enough to preserve:

-   object count and placement;
-   relative scale;
-   pose and anatomy;
-   perspective;
-   horizon and large structural lines;
-   semantic relationships between distant regions;
-   broad lighting and color organization.

### 2. Reduce expensive full-resolution computation

The global mechanism must be materially cheaper than running the normal
diffusion model densely over the complete target canvas.

A design such as:

``` text
full 4096×4096 model forward
+
local 1024×1024 model forward
```

does not solve the main problem if the full-canvas forward still
dominates cost.

### 3. Preserve local generative quality

Reducing global computation must not turn the final image into a merely
upscaled or blurred global draft.

Local computation should be able to produce high-frequency detail and
refine regional content at the desired output resolution.

### 4. Maintain one diffusion state

The final image should behave as one generated canvas rather than a
collage of independently denoised images.

The project must investigate how global and local predictions interact
with the persistent latent/sample state across denoising steps.

### 5. Remain practical in ComfyUI

Where possible, the implementation should preserve normal ComfyUI model
management, sampling, conditioning, and patching behavior.

Model/backend-specific execution changes should be introduced only when
the required algorithm cannot be expressed correctly through existing
interfaces.

## Secondary goals

After the basic method is qualified:

-   reduce wall-clock generation time;
-   reduce peak VRAM;
-   allow canvases larger than ordinary dense execution permits;
-   support arbitrary aspect ratios robustly;
-   investigate adaptive allocation of computation;
-   investigate whether global reasoning can be concentrated in early
    denoising steps;
-   investigate whether global context can be cached or refreshed
    periodically;
-   investigate native and specialized/quantized backends.

Compute, wall-clock time, and VRAM are separate metrics and must be
measured separately.

## Initial research references

The initial literature/implementation audit should include at least:

### Mixture of Diffusers

Useful as a reference for composing spatially separated diffusion
processes into a shared canvas and for understanding the memory
advantages and limitations of regional generation.

### MultiDiffusion

Useful as a baseline for coordinating multiple diffusion processes over
overlapping regions and reconciling them into a shared result.

### ElasticDiffusion

A particularly relevant reference because it explicitly separates global
and local content for training-free arbitrary-size generation.

Its mechanisms should be studied carefully, including what is specific
to its original Stable-Diffusion/U-Net formulation and what principle
may transfer to modern DiTs.

### Modern DiT execution

The project should determine how the global/local principle changes for
transformer models where image regions correspond naturally to token
subsets and where attention can potentially expose global context
separately from expensive local query computation.

### Previous local projects

`ComfyUI-SpotEdit`, `ComfyUI-Local-Edit`, and `ComfyUI-Nunchaku-Klein`
provide engineering evidence about ComfyUI/FLUX.2 execution, sparse
token computation, rectangular attention, caches, sampling lifecycle,
and local/global spatial constraints.

They are technical references, not algorithmic authority for Blueprint
Diffusion.

## Initial model focus

FLUX.2 Klein is a useful initial research target because substantial
execution knowledge already exists from previous projects.

This does not mean Blueprint Diffusion is defined as a FLUX.2-only
technique.

The first architecture should avoid unnecessary Klein-specific
assumptions unless they are required to prove the concept.

## Research questions

The initial research phase should answer the following.

### Global representation

1.  What information must be retained for reliable whole-scene
    composition?
2.  How low can global spatial/token density become before geometry
    deteriorates?
3.  Is a resized latent sufficient, or are transformer features/tokens
    more useful?
4.  Can full-canvas coordinates be retained while reducing token
    density?
5.  Is a multiscale representation materially better than one coarse
    global scale?

### Information flow

1.  How should local tokens consume global context?
2.  Is model-prediction guidance sufficient?
3.  Do local queries need attention to global K/V?
4.  Can global features be injected without retraining?
5.  How should local and global predictions be reconciled in the shared
    latent state?

### Denoising schedule

1.  Is global computation required at every denoising step?
2.  Can it be concentrated in high-noise/early steps?
3.  Can global state be refreshed periodically?
4.  Can refresh frequency be selected adaptively?
5.  How quickly does stale global context become harmful?

### Efficiency

1.  What work is actually avoided?
2.  Does the candidate reduce total transformer token computation?
3.  Does attention remain the dominant cost?
4.  What is the cost of caches and global/local transfers?
5.  Does theoretical compute reduction become wall-clock acceleration?
6.  Does the architecture reduce VRAM, increase VRAM through caches, or
    merely allow different memory scheduling?

### Generalization

1.  Does the method work on rectangular canvases?
2.  Does it work for both moderate and extreme target sizes?
3.  Does it preserve multiple distant subjects?
4.  Does it handle structures crossing local boundaries?
5.  Which parts are model-family-specific?

## Candidate design families

The following should initially be treated as separate hypotheses.

### A. Reduced-resolution global branch

Run a whole-canvas representation at substantially reduced spatial
resolution and use it to guide high-resolution local computation.

This is conceptually closest to ElasticDiffusion.

### B. Sparse global-token branch

Retain the full coordinate system but use a lower density of tokens for
global reasoning.

This may preserve spatial positions differently from ordinary image
downsampling.

### C. Hierarchical global pyramid

Maintain several global scales so coarse composition, medium-scale
geometry, and local detail are represented separately.

### D. Cached or intermittent global computation

Compute global features only on selected denoising steps and reuse them
between refreshes.

### E. Local-query/global-context transformer execution

Compute expensive query-side updates only for active/local tokens while
allowing those queries to attend to a compact or cached global context.

Previous SpotEdit engineering may be relevant here, but generation
semantics must be established independently.

These families may eventually be combined, but early experiments should
isolate them where possible.

## Initial milestones

### Phase 0 --- Repository and evidence setup

Create and maintain:

``` text
AGENTS.md
PROJECT.md
STATUS.md
FINDINGS.md
DECISIONS.md
docs/notes/
experiments/
references/
target/ComfyUI-Blueprint-Diffusion/
```

Populate `references/` with read-only source/paper repositories or
junctions as appropriate.

No production algorithm should be assumed at this phase.

### Phase 1 --- Comparative architecture audit

Study:

-   ordinary full-canvas diffusion;
-   tiled diffusion;
-   Mixture of Diffusers;
-   MultiDiffusion;
-   ElasticDiffusion;
-   relevant newer global/local or arbitrary-resolution methods;
-   modern FLUX.2/DiT execution boundaries.

For each, record:

-   global information source;
-   local processing unit;
-   diffusion-state ownership;
-   merge/reconciliation method;
-   training requirements;
-   computational scaling;
-   memory behavior;
-   assumptions tied to U-Net or DiT architecture.

Deliverable: a shortlist of Blueprint architectures worth testing.

### Phase 2 --- Minimal falsification experiments

Before production nodes, implement experiment-only prototypes that
answer the highest risk questions.

Examples:

-   how much composition survives global downsampling;
-   whether local FLUX.2 queries can use reduced/cached global context;
-   whether early-only global computation is viable;
-   whether global context can remain stale for several steps;
-   whether a local update can be scattered into a full-canvas
    trajectory without destabilizing sampling.

Deliverable: evidence for or against each candidate.

### Phase 3 --- Reference prototype

Implement the smallest full-compute or minimally optimized architecture
that demonstrates the desired global/local behavior.

Correctness and image behavior take priority over optimization.

Deliverable: reproducible ComfyUI-compatible prototype and controlled
baselines.

### Phase 4 --- Efficiency implementation

Only after the architecture is qualified, investigate:

-   sparse token execution;
-   cached global K/V/features;
-   global refresh scheduling;
-   CPU/GPU state placement;
-   reduced global token density;
-   backend-specific acceleration.

Deliverable: measured reduction in work and/or memory without
unacceptable quality loss.

### Phase 5 --- Backend qualification

Where useful, qualify specialized backends such as Nunchaku only after
the native execution contract is understood.

Backend changes should expose the minimum generic capability required by
the proven Blueprint architecture.

## Evaluation

Image-quality and performance claims must be evaluated separately.

### Image behavior

Use tests that expose failures hidden by ordinary tiled landscapes:

-   one subject spanning region boundaries;
-   multiple distant people/objects;
-   long structures crossing the image;
-   strong perspective;
-   interiors with repeated geometry;
-   panoramic scenes;
-   tall portrait scenes;
-   compositions with a clear central subject and distant contextual
    objects.

Inspect:

-   composition;
-   count;
-   placement;
-   relative scale;
-   continuity;
-   anatomy;
-   perspective;
-   lighting;
-   seams;
-   fine detail;
-   prompt adherence.

### Performance

Measure:

-   target canvas/token count;
-   global representation size/token count;
-   active/local token count;
-   number of model forwards per step;
-   attention dimensions;
-   model step time;
-   end-to-end sampling time;
-   peak VRAM;
-   persistent VRAM;
-   cache memory;
-   CPU/GPU transfer overhead.

A method must not be described as cheaper merely because its individual
local forwards are smaller.

## Success criteria

A first successful Blueprint Diffusion implementation should demonstrate
all of the following:

1.  a target canvas larger or more difficult than the model's ordinary
    comfortable generation regime;
2.  materially better global coherence than naive independent tiled
    diffusion;
3.  local detail competitive with high-resolution local generation;
4.  a global mechanism substantially cheaper than ordinary dense
    full-canvas diffusion;
5.  a coherent shared denoising trajectory rather than post-hoc image
    assembly;
6.  reproducible integration with ComfyUI;
7.  measured compute/runtime/VRAM behavior with no unsupported
    efficiency claims.

The project does not require every candidate architecture to satisfy
these criteria. Research that conclusively rejects an approach is a
useful result and should be recorded in `FINDINGS.md`.

## Documentation policy

Use:

-   `STATUS.md` for current progress and next milestones;
-   `FINDINGS.md` for reusable evidence and technical facts;
-   `DECISIONS.md` for architecture/project choices supported by
    evidence;
-   `docs/notes/` for longer investigations;
-   `experiments/` for disposable or research-only validation code and
    outputs.

Do not convert hypotheses into decisions merely because they appear
promising.
