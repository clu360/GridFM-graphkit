# GridFM Research Workflow Architecture

```yaml
artifact_id: ARCH-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


## Purpose

This workflow governs future GridFM wildfire-aware optimization research through file-mediated collaboration, documented authority, role separation, and auditable validation.

Phase 1 establishes the workflow mechanics only. It does not accept, reject, or scientifically certify the completed Stage I/Stage H research.

## Enforcement Model

The Phase 1 enforcement model is `procedural_and_auditable_role_separation`.

This means roles are separated by documented responsibilities, file routing, versioned artifacts, freeze hashes, and contamination audits. Phase 1 does not implement OS-level permissions, separate credentials, separate filesystem ACLs, or true security isolation.

Default execution metadata:

```text
execution_mode = single_context_role_simulation
blind_context_enforced = false
reviewer_prior_outputs_visible = false
```

## Lifecycle

Proposal lifecycle:

```text
DRAFT -> UNDER_SCIENTIFIC_REVIEW -> REVISION_REQUIRED or APPROVAL_PENDING -> APPROVED -> IMPLEMENTING -> IMPLEMENTED -> UNDER_AUDIT -> RESULTS_REVIEW -> ACCEPTED / ACCEPTED_WITH_LIMITATIONS / REMEDIATION_REQUIRED / DEFERRED
```

Only Caleb can authorize transitions into `APPROVED`, `ACCEPTED`, `ACCEPTED_WITH_LIMITATIONS`, `REJECTED`, or `DEFERRED`.

Finding lifecycle:

```text
OPEN -> ACKNOWLEDGED -> DISPOSITION_PENDING -> ACCEPTED / REJECTED / DEFERRED -> REMEDIATION_PLANNED -> REMEDIATED -> VERIFIED -> CLOSED
```

No agent may close its own finding.

## Phase 2 Placeholder

The proposed Phase 2 retrospective case path is:

```text
workflow/cases/CASE-001-stage-i-retrospective/
```

It is intentionally not created or executed in Phase 1.
