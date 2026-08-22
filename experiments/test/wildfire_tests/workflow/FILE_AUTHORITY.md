# File Authority

```yaml
artifact_id: AUTH-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


## Authority Order

1. Caleb's explicit current instruction
2. Caleb-approved decision record
3. Approved proposal or approved amendment
4. `CURRENT_STATE_SUMMARY.md`
5. Active task-specific formulation or handoff
6. Workflow architecture and role instructions
7. Validated implementation, tests, and result metadata
8. `HISTORY.md`
9. Archived, superseded, rejected, or deferred plans
10. Unapproved agent suggestions and informal notes

## Conflict Classes

`TEMPORAL_SUPERSESSION`, `SCOPE_DIFFERENCE`, `NAMING_DRIFT`, `FORMULATION_CONFLICT`, `IMPLEMENTATION_DEVIATION`, `RESULT_INTERPRETATION_CONFLICT`, and `UNRESOLVED_AUTHORITY`.

## Controlled Phase 1 Authority Tests

| Test | Conflict Class | Phase 1 Resolution Behavior |
| --- | --- | --- |
| Stage H versus Stage I naming | `NAMING_DRIFT` | Preserve both names and document exact directory/script relationship. |
| Stage D inclusion/exclusion | `SCOPE_DIFFERENCE` or `TEMPORAL_SUPERSESSION` | Use current approved Stage H plotting scope; preserve historical context. |
| Coupled versus decoupled proxy/inner lambda | `SCOPE_DIFFERENCE` | Treat main sweep, MLD, and proxy-inner sweep as separate study scopes. |
| Unresolved baseline commit | `UNRESOLVED_AUTHORITY` | Do not invent a baseline; escalate for Caleb decision if needed. |
