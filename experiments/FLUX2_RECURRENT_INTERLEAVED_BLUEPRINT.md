# Recurrent/interleaved Blueprint result

## Verdict

**FAIL — both fresh-W recurrent variants lose S3.**

The fixed four-interval same-sigma discriminator is deterministic and bounded,
but both arms replace the qualified single car/tree/house composition with a
repeated lattice of locally generated cars, trees, and houses.  Recognizable
local objects do not count as credible detail improvement when the whole-scene
composition is destroyed.

## Executed contract

At each qualified Blueprint interval, the live `x0_G` prediction was transferred
with the halo-aware bounded crop primitive.  Each W was constructed from that
clean anchor and deterministic regional noise through active CONST
`noise_scaling(sigma, noise, anchor)`.  G and W each accepted one Euler update
from the same `sigma` to the same `sigma_next`.

- G-authoritative accepted the ordinary low-resolution `G*` and discarded local
  feedback.
- Bidirectional replaced `G*` with a bilinear downscale of the accepted assembled
  H at the same next sigma.

No H-sized model call occurred.  Each arm made four `45x45` G calls and 100
`64x64` W calls.  H was materialized only for normalized assembly, metrics, and
the explicit H-to-G feedback operation.

## Evidence

- Independent repeats are bit-exact for both arms.
- Final latent hashes are
  `9815edcbbbec6e7ccbf19946a18488568face800939df006dd891b04ab4ef711`
  (G-authoritative) and
  `99a2c628659b76a0614d43697a6533774d6d20d5d02503b2b78099a5c8936eef`
  (bidirectional).
- Coverage is positive and normalized (`1.0` to `1.0000001192`) at every
  interval.
- Inter-interval CUDA allocation barriers are constant within each arm.
- Final overlap RMS is `0.943350` and `0.942238`, far worse than frozen
  Terminal's `0.173451` and the unguided four-interval control's `0.190880`.
- Final gradient RMS rises to `0.511688` and `0.508671`, but inspection shows
  that this energy is repeated object structure and band boundaries, not an
  accepted local-detail gain.
- The arms share the same initial G and identical first-interval results.
  Bidirectional feedback differs substantially from the ordinary G proposal in
  interval 0 (`RMS 1.045159`), then closely tracks the local path in intervals 1
  and 2.  Both nevertheless share the same qualitative fragmentation.  H-to-G
  feedback therefore neither causes the initial local duplication nor repairs it.

## Failure localization

The fixed Blueprint schedule spends its first three evaluations near sigma one.
Freshly noise-scaled W regions at those sigmas have enough independent
generative freedom to instantiate the full prompt locally.  The evolving
low-resolution `x0_G` anchor does not exert sufficient spatial authority over
those independent W generations.  Overlap normalization blends boundaries but
cannot recover one global object layout.

This narrowly rejects fresh same-sigma W reconstruction from clean G predictions
through regional `noise_scaling` at the qualified high/near-one sigmas.  It does
not reject recurrent/interleaved multiresolution sampling generally or a
persistent H trajectory with shared stochastic provenance.
