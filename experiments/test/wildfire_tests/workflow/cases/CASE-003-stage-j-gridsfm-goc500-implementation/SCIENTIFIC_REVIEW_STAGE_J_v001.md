# Scientific Review Stage J v001

```yaml
artifact_id: CASE-003-SCIENTIFIC-REVIEW-STAGE-J
artifact_version: v001
created_local: 2026-08-12T22:58:28.1336018-04:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: INITIAL_REVIEW_AFTER_FORMULATION_LOCK
sha256_when_frozen: null
```

## Review Scope

This review checks whether the Stage J design is scientifically coherent with Caleb's locked intent before S1-S3 implementation.

## Findings

`OK`: The decision space is explicitly wildfire-side topology plus full per-load service:

```text
z_l in {0,1}
alpha_i in [0,1] for every load i in D
```

`OK`: The primary comparison is controlled: Guided-DC and Guided-GridSFM receive the same external `(z, alpha)` candidates and differ only in electrical recourse/evaluation once the adapter is built.

`OK`: The GridSFM objective distinction is scientifically clean:

```text
J_trade = lambda_R R_norm + (1 - lambda_R) L_shed_total
J_total^SFM = J_trade^SFM + rho_phys PAC_total^SFM
```

Pareto interpretation remains in `(L_shed_total, R_norm)`, while `J_total` is a physics-aware search merit for GridSFM.

`OK`: Load shedding is model-independent and decomposed into:

```text
L_shed_total = L_shed_control + L_shed_island
```

This distinguishes intentional alpha curtailment from topology-forced source-less island shedding.

`OK`: Reference A and Reference B are distinct:

```text
Reference A = fixed-z, fixed-alpha economic AC-OPF
Reference B = fixed-z Rhodes-style AC redispatch / MLD
```

This prevents AC state fidelity, feasibility restoration, and maximum-load-delivery interpretation from being collapsed into one metric.

`OK_WITH_LIMITATION`: J0.5 validated the official GridSFM package/checkpoint/sample inference path and the PowerModels/IPOPT backend, but the experiment-specific GOC-500 mutation/identity adapter and exact AC baseline are not yet implemented.

## Required Scientific Guardrails Before Results

1. Do not replace full per-load alpha with grouped alpha without an `APPROXIMATION REQUIRED` decision.
2. Do not compute wildfire risk without verified finite positive `rateA`, compatible units, and positive `R_base`.
3. Do not tune PAC weights after inspecting S1-S3 comparative winners.
4. Do not call the method security-constrained unless contingency/security constraints are implemented.
5. Do not treat GridSFM predicted feasibility as exact AC feasibility; use Reference A/B audits for exact claims.
