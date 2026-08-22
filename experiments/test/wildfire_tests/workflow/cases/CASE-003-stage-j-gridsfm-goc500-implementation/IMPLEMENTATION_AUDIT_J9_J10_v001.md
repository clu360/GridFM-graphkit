# Implementation Audit: J9/J10

artifact_id: `IMPLEMENTATION_AUDIT_J9_J10_v001`

created_utc: `2026-08-17`

execution_mode: `separate_codex_contexts`

blind_context_enforced: `false`

reviewer_prior_outputs_visible: `true`

status: `PASS_WITH_LIMITATIONS`

## Scope

Read-only audit of the Guided-DC and Guided-GridSFM finalist smoke in:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/
  j9_j10_smoke/s1_l08_v005/
```

Reviewed implementation:

```text
stage_j_ac_reference.jl
run_j9_j10_reference_smoke.py
STAGE_J_PRIMARY_DESIGN.md, Reference A/B and warm-start sections
```

## Confirmed Alignment

1. Reference A fixes the selected topology and full effective-alpha vector,
   retains original generator bounds, and solves exact economic AC-OPF.
2. Reference B fixes only topology, enforces source-less loads, uses fixed
   shunts, applies `0 <= Pg <= Pgmax` only in restoration, and implements B1
   maximum load delivery followed by B2 fuel-cost minimization.
3. All warm-start policies solve the same fixed-z/fixed-alpha Reference A
   instance. DC and GridSFM starts are intentionally partial (`Pg, theta`),
   while GT uses the prior exact Reference A state.
4. GridSFM state fidelity includes Pg, Qg, V, aligned theta, and both-end P/Q
   branch flows. DC is explicitly `N/A` for quantities it does not model.
5. J9/J10 scope is limited to Guided-DC and Guided-GridSFM, as locked. TH is
   excluded from the post-hoc exact-reference smoke.

## Findings And Limitations

1. `PASS_WITH_LIMITATIONS`: PowerModels/Ipopt iteration counts are unavailable
   through the current wrapper and are recorded as unavailable rather than
   inferred.
2. GridSFM start-construction time is measured end-to-end but not yet split
   into preprocessing and forward-inference components.
3. B2 uses an explicit `1.0e-5` Pd-unit service tolerance. The final smoke
   reports both the raw service-lock residual (`1.7773e-6`) and a separate
   solver-tolerance-qualified verification flag. Do not suppress either field.
4. The detailed family-level state-fidelity metrics are primary. No composite
   `D_state_to_AC` is reported because its weights are not locked.

## Assessment

No critical formulation violation was found. The v005 smoke is usable to
validate the Reference A/B and component-fidelity plumbing. It is not yet a
multi-scenario warm-start performance study.
