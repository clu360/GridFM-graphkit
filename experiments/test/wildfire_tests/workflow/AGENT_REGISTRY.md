# Agent Registry

```yaml
artifact_id: REG-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


| Role | Directory | Phase 1 Execution Mode | Write Authority |
| --- | --- | --- | --- |
| Primary research agent | `workflow/primary_agent/` | `single_context_role_simulation` | Draft proposals, implementation handoffs, responses to findings. |
| Scientific reviewer | `workflow/scientific_reviewer/` | `single_context_role_simulation` | New scientific review versions only. |
| Implementation/results auditor | `workflow/implementation_auditor/` | `single_context_role_simulation` | New audit versions only. |
| Caleb | decision records | human authority | Final approval, acceptance, rejection, deferral, and workflow changes. |

Phase 1 simulates role separation in one Codex context and records that limitation explicitly.
