# Stage I Workflow Review Package

This package is a local read-only evidence snapshot for a retrospective review
of the GridFM wildfire-aware Stage I / Stage H DC approximation and AC
projection workflow.

It exists so an external planning/review agent can inspect intent, code, tests,
results, git state, and reproducibility evidence without modifying the active
research directory.

## Snapshot

- Created UTC: `2026-07-30T02:59:57.735979+00:00`
- Current commit: `831d76738d5a64f934430a8ebf1ac281e6bccd7f`
- Baseline commit: unresolved
- Primary run selected: `main_results/r11`
- Companion runs: `MLD/r5`, `proxy_inner_lambda_sweep/r2`
- Package status: local and uncommitted

## What It Contains

- Current/historical state documents.
- Stage I/DC comparison source surface and materially imported dependencies.
- Relevant tests and current test execution output.
- Primary and companion result artifacts, with large CSVs indexed instead of copied.
- Git, environment, and reproducibility evidence.
- Review scope, checklists, method matrix, and traceability matrix.
- Literature placeholders.

## What It Does Not Contain

- No edits to active research code or results.
- No reviewer scores or judgments of correctness.
- No supplied literature papers yet.
- No proxy-inner AC projection distances.

## Recommended Reading Order

1. `PACKAGE_README.md`
2. `REPOSITORY_SNAPSHOT.md`
3. `01_authoritative_state/CURRENT_STATE_SUMMARY.md`
4. `02_planning_and_intent/INTENT_RECONSTRUCTION.md`
5. `09_review_inputs/REVIEW_SCOPE.md`
6. `09_review_inputs/METHOD_COMPARISON_MATRIX.md`
7. `03_stage_i_source/SOURCE_INDEX.md`
8. `04_tests/TEST_EXECUTION_SUMMARY.md`
9. `05_results/RESULT_COMPLETENESS_AUDIT.md`
10. `06_git_and_changes/CHANGE_SUMMARY.md`
11. Literature materials when provided.
