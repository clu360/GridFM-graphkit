# RQ1 Frozen M0 Evaluator Study

## Status

```text
RQ1_PUBLICATION_PASS
```

The study applies only released GridSFM M0 to 54 unique fixed decisions drawn
from 75 refined-study provenance rows. Canonical deduplication includes full
branch status, alpha-effective, scaled active demand, and scaled reactive
demand. Repetitions are measurement noise; the decision is the statistical
unit.

## Timing Boundary

The primary evaluator boundary begins when an initialized evaluator receives a
fixed candidate and ends when a usable electrical state is returned. It
includes candidate mutation, model/formulation construction, computation, and
state extraction. It excludes downstream wildfire scoring, publication I/O,
and one-time environment/model loading.

Reference A retains its existing default initialization: `V=1`, `theta=0`,
`Pg=(Pmin+Pmax)/2`, and `Qg=0`. It is a locally solved fixed-decision AC OPF
reference under the declared tolerances, not a global-optimality certificate.

## Result

Median paired evaluator speedup was **3.59x**
(95% paired bootstrap interval
3.48-3.68x);
geometric-mean speedup was
**3.61x**. Median core forward-versus-IPOPT
speedup was **3.35x** (95% interval
3.31-3.45x).

This supports the computational motivation for approximate M0 screening before
a smaller exact finalist audit, on this GOC-500 implementation and CPU. It does
not imply the complete OPS algorithm is accelerated by the same factor.

## Illustration

The publication illustration is an actual M0-selected finalist:
`m0:s1_l0p8` / `RQ1U031`. GridSFM
evaluates this fixed decision; the outer OPS search selected the de-energized
branch set `(276, 473)`.
The figure reports the M0-predicted state with an AC physics infeasibility
diagnostic, not an exact-feasibility claim.
