# Research Invariants

```yaml
artifact_id: INV-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


| ID | Statement | Authority Source | Scope | Status |
| --- | --- | --- | --- | --- |
| INV-001 | Physical branch decisions use canonical physical branches. | Stage I package and Stage H handoff records | topology decisions | `ACTIVE` |
| INV-002 | Duplicate directed GridFM edges are not independent shutoff decisions. | Stage I package | topology decisions | `ACTIVE` |
| INV-003 | MATPOWER `rateA` supplies thermal ratings where specified. | approved Stage H plan | DC/common diagnostics | `ACTIVE` |
| INV-004 | Source-less islanded load is treated as unserved. | approved Stage H plan | load shedding | `ACTIVE` |
| INV-005 | GridFM predictions are not AC-feasibility certificates. | Stage H findings | GridFM evaluation | `ACTIVE` |
| INV-006 | DC feasibility is not AC feasibility. | Stage I methodology | DC approximations | `ACTIVE` |
| INV-007 | Primary K<=2 comparison methods obey the approved topology budget. | approved Stage H plan | comparisons | `ACTIVE` |
| INV-008 | `PAC_total`, DC residuals, common operational diagnostics, and AC projection distance must not be conflated. | validation proposal | diagnostics | `ACTIVE` |
