# Texas2k Results

This directory is the canonical local results package for the Stage K Texas2k
case study. Methodology, configuration, implementation, and runbooks remain in
`../stage_k_case_study/`; immutable or retrieved study outputs live here.

## Contents

- `environment_snapshot_tau0p50/`: frozen June 23, 2023, 16:00 CDT cumulative
  environmental hazard snapshot and its figures/reports.
- `smoke_test/`: complete Gate 4 PACE smoke-test outputs, including the final
  corrected result package and retained diagnostic history.
- `full_run/`: destination for the retrieved Stage K production run. The
  production package is added only after Slurm aggregation and validation
  complete successfully.

Result trees are preserved rather than rewritten so their internal manifests,
hashes, diagnostics, and provenance remain auditable.

The required local analysis and figure package to construct after production
retrieval is frozen in `POST_RUN_ANALYSIS_PLAN.md`.
