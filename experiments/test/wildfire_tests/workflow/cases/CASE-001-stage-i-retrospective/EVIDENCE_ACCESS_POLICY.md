# Evidence Access Policy

```yaml
artifact_id: CASE-001-EVIDENCE-POLICY
artifact_version: v001
created_utc: 2026-07-30T23:55:00+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: PREPARED_NOT_EXECUTED
sha256_when_frozen: recorded_in_workflow/cases/CASE-001-stage-i-retrospective/FROZEN_PREP.csv
```

## Evidence Labels

Every evidence item used in Phase 2 must be labeled exactly one of:

```text
AVAILABLE
UNAVAILABLE
NOT_INSPECTED
```

## Live Repository Artifacts

Phase 2 may use live-repository artifacts referenced by v002 package indexes. Each live artifact access must record:

- artifact path;
- evidence label;
- pre-access SHA-256;
- post-access SHA-256;
- access timestamp;
- role accessing the file;
- downstream finding or report using the evidence.

## Verification Rule

No finding may be marked verified when required evidence is unavailable.

## Package Boundary

Use `stage_i_workflow_review_package_v002` as the active evidence package. Keep the original package visible only as failed historical evidence.

