# Permission Matrix

```yaml
artifact_id: PERM-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


This matrix is procedural and auditable, not OS-enforced.

| Resource | Primary Agent | Scientific Reviewer | Implementation Auditor |
| --- | --- | --- | --- |
| Active source code | Read/write only after Caleb-approved implementation | Read only | Read only |
| Active tests | Read/write only after Caleb-approved implementation | Read only | Read only |
| Official results | Generate; never rewrite silently | Read only | Read only |
| Approved proposals | Read only after approval | Read only | Read only |
| Draft proposals | Create/revise | Comment through review | Read |
| Scientific reviews | Read only | Create new versions | Read only |
| Audit reports | Read only | Read only | Create new versions |
| Rubric | Read only | Read only | Read only |
| Decision records | Read | Read | Read |
| Literature index | Propose additions | Validate/categorize | Read |
| Workflow architecture | Read | Read | Read |
