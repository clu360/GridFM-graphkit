# AUD-FINDING-0004-v001

```yaml
finding_id: AUD-FINDING-0004
version: v001
category: TEST_COVERAGE_GAP
severity: MODERATE
reviewer_role: implementation_results_auditor
status: OPEN
confidence: HIGH
execution_mode: separate_codex_contexts
created_at: 2026-07-31T00:24:00+00:00
```

Automated tests are mostly structural/helper-level and do not cover several core promises: nontrivial tap/shift fixtures, Stage I-a no-revisit behavior, Stage I-b MIQP optimality behavior, and AC projection semantics.

Evidence: `tests/test_wildfire_stage_i_dc_comparison.py`, `tests/test_wildfire_stage_h_heuristic_comparison.py`, `tests/test_wildfire_stage_g_revised_continuous.py`.

