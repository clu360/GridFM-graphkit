# Role Context Protocol

```yaml
artifact_id: CASE-001-ROLE-CONTEXT
artifact_version: v001
created_utc: 2026-07-30T23:55:00+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: PREPARED_NOT_EXECUTED
sha256_when_frozen: recorded_in_workflow/cases/CASE-001-stage-i-retrospective/FROZEN_PREP.csv
```

## Preferred Mode

Use `separate_codex_contexts` where feasible for:

- scientific reviewer;
- implementation/results auditor;
- post-freeze reconciliation reviewer.

## Fallback Mode

If Phase 2 uses one context, record `single_context_role_simulation` and do not claim true reviewer independence.

## Visibility Rules

Before initial reports are frozen:

- the scientific reviewer must not see the implementation/results auditor's draft conclusions;
- the implementation/results auditor must not see the scientific reviewer's draft conclusions;
- neither reviewer may receive desired outcomes, self-scores, or persuasive claims about correctness.

After both initial reports are frozen, the reconciliation reviewer may inspect both.

