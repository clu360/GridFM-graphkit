# Improved Optimization Progress

## Completed so far

- Created the new `experiments/test/improved_optimization/` package structure.
- Added reusable config, IO, wildfire metric, objective, selection, validation, reporting, plotting, and optimization modules.
- Implemented targeted control selection using surrogate finite differences.
- Implemented a compact selected-control decision specification with deterministic ordering.
- Added runnable scripts for:
  - single improved optimization runs
  - Pareto sweeps
  - objective sensitivity analysis
- Preserved reuse of legacy scenario loading, surrogate wrappers, and checkpoint loading.

## Remaining validation / cleanup tasks

- Run and verify the new scripts end-to-end.
- Archive or remove stale markdown files from `experiments/test/`.
- Tighten any issues surfaced by the first full executions.
- Update methodology/report outputs with verified findings.
- Ensure the repo is clean and ready for commit.

## Notes for continuation

- The improved workflow lives at `experiments/test/improved_optimization/`.
- GPS loading still depends on the inferred architecture logic in `experiments/test/pipeline_utils.py`.
- The legacy notebook and pipeline changes in `experiments/test/ieee30_optimization_validation.ipynb` and `experiments/test/pipeline_utils.py` remain in the worktree and should be preserved unless explicitly reverted.
