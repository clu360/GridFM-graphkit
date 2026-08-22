# AUD-FINDING-0003-v001

```yaml
finding_id: AUD-FINDING-0003
version: v001
category: PHYSICS_ISSUE; RESULT_EVIDENCE_GAP
severity: MODERATE
reviewer_role: implementation_results_auditor
status: OPEN
confidence: HIGH
execution_mode: separate_codex_contexts
created_at: 2026-07-31T00:24:00+00:00
```

AC projection uses broad Qg bounds and records `Qg_bounds_available=False`. Projection success should be described as fixed-topology AC balance/thermal projection with relaxed Qg limits, not full generator reactive capability feasibility.

Evidence: `ac_projection.py`, main/MLD `ac_projection_distances.csv`, `INT-011`, `INT-012`.

