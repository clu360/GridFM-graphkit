# AUD-FINDING-0002-v001

```yaml
finding_id: AUD-FINDING-0002
version: v001
category: DOCUMENTATION_GAP
severity: MINOR
reviewer_role: implementation_results_auditor
status: OPEN
confidence: HIGH
execution_mode: separate_codex_contexts
created_at: 2026-07-31T00:24:00+00:00
```

Progress prose contains numeric drift: MLD finite projection distances are stated as `32/60`, but the live table has `31/60`; main augmented evaluated points are stated as `10364`, while refresh summary has `10414`.

Evidence: `STAGE_H_DC_COMPARISON_PROGRESS.md`, MLD `ac_projection_distances.csv`, main `stage_i_b_solution_pool_refresh_summary.json`.

