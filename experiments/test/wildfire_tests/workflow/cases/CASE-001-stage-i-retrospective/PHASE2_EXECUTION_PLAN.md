# Phase 2 Execution Plan

```yaml
artifact_id: CASE-001-PLAN
artifact_version: v001
created_utc: 2026-07-30T23:55:00+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: PREPARED_NOT_EXECUTED
sha256_when_frozen: recorded_in_workflow/cases/CASE-001-stage-i-retrospective/FROZEN_PREP.csv
```

## Review Roles

Phase 2 should use separate Codex contexts where feasible:

| Role | Responsibility | Initial Output |
| --- | --- | --- |
| Case coordinator | Assign inputs, enforce routing, maintain evidence ledger, prevent premature reconciliation. | coordinator intake and evidence availability ledger |
| Scientific reviewer | Evaluate formulation, physics, method fairness, literature alignment, and result-to-claim validity. | frozen initial scientific review |
| Implementation/results auditor | Evaluate intent-to-code, code-to-test, test-to-result, reproducibility, provenance, and result completeness. | frozen initial implementation/results audit |
| Post-freeze reconciliation reviewer | Compare frozen initial reports after both are complete. | reconciliation report and unresolved questions |

## Independence Rule

The scientific reviewer and implementation/results auditor must produce and freeze their initial reports independently before reconciliation.

If separate contexts are not used, every report must record:

```text
execution_mode = single_context_role_simulation
blind_context_enforced = false
reviewer_prior_outputs_visible = false
```

## Review Dimensions

Phase 2 evaluates:

- intent-to-formulation;
- intent-to-code;
- code-to-test;
- test-to-result;
- result-to-claim;
- physics alignment;
- literature alignment;
- method fairness;
- reproducibility.

## Finding Rule

Phase 2 produces findings only. It must not implement remediation without a separate Caleb-approved decision.

