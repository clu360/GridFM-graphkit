# Integration Audit: Upstream Main Into Wildfire Experiments

Date: June 10, 2026

This audit previews integrating the updated `main` branch into
`wildfire-experiments` without performing the merge. It exists to protect the
current wildfire predict-to-optimize workflow, preserved checkpoints, and local
IEEE-30 artifacts before any branch integration.

## Current Branch State

- Active branch: `wildfire-experiments`
- Current `HEAD`: `c0932eb`
- Updated `main`, `origin/main`, and `upstream/main`: `2cdd791`
- Merge base between `wildfire-experiments` and `main`:
  `acbf7be5dc7edac4d2dd22df39108a60c58659a5`
- Ahead/behind relationship:

```text
main...wildfire-experiments: 21 commits on main, 17 commits on wildfire
```

The working tree was not clean before this audit:

```text
M  experiments/test/wildfire_tests/HISTORY.md
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/auto/bal/run_20260531_211830/
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/auto/risk/run_20260531_211828/
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/auto/svc/run_20260531_211832/
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/lgh/bal/run_20260531_211842/
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/lgh/risk/run_20260531_211841/
?? experiments/test/wildfire_tests/results/stage_d_deenergization/t0p30/gps/lgh/svc/run_20260531_211844/
```

No merge was performed during this audit.

## Non-Mutating Merge Preview

Command used:

```powershell
git merge-tree $(git merge-base HEAD main) HEAD main
```

Result:

- No conflict markers were reported.
- No `CONFLICT` lines were reported when searching the merge preview output.
- The preview shows the upstream changes can be mechanically merged with the
  current branch state.

Important interpretation note:

```powershell
git diff HEAD..main
```

shows many wildfire files as deleted because it asks what would change if
`HEAD` were replaced by `main`. That is not the same operation as merging
`main` into `wildfire-experiments`. The merge preview indicates branch-local
wildfire files should be preserved by a normal merge.

## Actual Upstream Delta Since Merge Base

The real upstream-side file delta from the merge base to updated `main` is:

```text
M  .github/workflows/ci-build.yaml
M  .github/workflows/release.yaml
M  gridfm_graphkit/__main__.py
M  gridfm_graphkit/cli.py
M  gridfm_graphkit/datasets/hetero_powergrid_datamodule.py
M  gridfm_graphkit/models/gnn_heterogeneous_gns.py
```

No upstream-side changes since the merge base touched:

- `experiments/test/`
- `experiments/test/wildfire_tests/`
- `examples/models/`
- `tests/config/gridFMv0.1_dummy.yaml`
- `tests/data/`

## Preserved Research Dependencies

The wildfire workflow currently relies on these preserved branch/local assets.

Tracked on `wildfire-experiments`, absent from updated `main`:

```text
examples/models/GridFM_v0_1.pth
examples/models/GridFM_v0_2.pth
experiments/test/pipeline_utils.py
experiments/test/neural_solver.py
experiments/test/scenario_data.py
experiments/test/overload_penalty.py
experiments/test/pv_dispatch.py
experiments/test/wildfire_tests/
tests/config/gridFMv0.1_dummy.yaml
tests/test_wildfire_*.py
```

At initial audit time, these were local-only and ignored by Git:

```text
tests/data/case30_ieee/processed/*.pt
tests/data/case30_ieee/processed/*.done
```

The `.gitignore` file ignores `*.pt` and `*.done`, which explains why the
local IEEE-30 processed tensors do not appear in Git status or `git ls-files`.
This is a portability risk: a fresh clone of the branch will not necessarily
have the IEEE-30 processed data needed by the current wildfire tests unless the
data is regenerated or preserved outside Git.

Follow-up preservation action:

- Because the current workflow does not have a known regeneration process for
  these IEEE-30 processed tensors, the 26 files under
  `tests/data/case30_ieee/processed/` were intentionally force-added with
  `git add -f`.
- Total size is about 181 KB, so tracking these files is reasonable for
  reproducibility.
- This turns the IEEE-30 processed test artifacts from fragile local state into
  explicit branch state for `wildfire-experiments`.

## Methodology Risk Review

### Low-Risk Upstream Changes

`.github/workflows/ci-build.yaml`

- Adds `MLFLOW_ALLOW_FILE_STORE=true` for pytest.
- Expands ignored pip-audit vulnerabilities.
- Temporarily disables the pre-commit job.
- Does not affect local wildfire objective semantics or model inference.

`.github/workflows/release.yaml`

- Changes release publishing to tag-triggered build and upload.
- Does not affect local wildfire experiments.

`gridfm_graphkit/__main__.py`

- Adds `--mp_context` CLI argument to train, finetune, evaluate, predict, and
  benchmark commands.
- Adds a Linux warning when multiprocessing context is unset, `fork`, or
  `forkserver`.
- The wildfire objective loop does not call the GridFM CLI, so this should not
  directly affect the predict-to-optimize workflow.

`gridfm_graphkit/cli.py`

- Passes `mp_context` into the datamodule.
- Adds device summary logging.
- Disables Triton dynamic graph support for `torch.compile`.
- Changes DDP strategy from `find_unused_parameters=True` to
  `find_unused_parameters=False`.
- The wildfire objective loop loads GridFM in memory through
  `experiments/test/pipeline_utils.py` and `NeuralSolverWrapper`, not through
  `main_cli`, so these are not expected to change current optimization
  objective values.

`gridfm_graphkit/datasets/hetero_powergrid_datamodule.py`

- Adds optional `multiprocessing_context`.
- Removes the previous automatic Linux `fork` override.
- Current wildfire config uses `workers: 0`, so the dataloader
  multiprocessing path should not be active in the current small IEEE-30 test
  workflow.

### Medium-Risk Upstream Change

`gridfm_graphkit/models/gnn_heterogeneous_gns.py`

- For `StateEstimation`, freezes `mlp_gen`, `physics_mlp`, the last generator
  normalization block, and the last bus-to-generator convolution.
- The current wildfire config in `tests/config/gridFMv0.1_dummy.yaml` uses:

```yaml
task:
  task_name: PowerFlow
```

- Therefore this should not affect the current PowerFlow wildfire run path.
- It could affect future wildfire experiments if they switch to a
  `StateEstimation` task or rely on train-time behavior rather than inference
  from existing checkpoints.

### Explicit Predict-To-Optimize Impact Check

Current wildfire execution path:

```text
run_* scripts
  -> GridFMRunner
  -> experiments.test.wildfire_tests.gridfm_support.pipeline_utils.load_gnn_model/load_gps_model
  -> experiments.test.wildfire_tests.gridfm_support.neural_solver.NeuralSolverWrapper
  -> in-memory model prediction
```

The active objective loop does not call the package CLI inside optimizer
iterations. Upstream CLI, release, and CI changes are therefore not expected to
change the scalar objective, wildfire risk computation, line indexing, service
metric, or decision vector semantics.

## Baseline Backtest

Command run before any merge:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py tests/test_wildfire_stage_d_deenergization.py -q
```

Result:

```text
30 passed, 3 external deprecation warnings
```

This is the baseline to compare against after a future merge.

## Isolation Recommendations

Before merging, preserve the current dirty research state:

1. Commit or otherwise intentionally preserve the current `HISTORY.md` update.
2. Decide whether the untracked Stage D `run_20260531_2118xx` directories are
   obsolete failed/early artifacts or should be retained.
3. Keep the force-added IEEE-30 processed tensors in the next preservation
   commit unless a better regeneration path is created.

For future-proof organization:

1. Keep wildfire-only code under `experiments/test/wildfire_tests/`.
2. Keep wildfire support wrappers under `experiments/test/` unless they become
   generally useful enough to upstream.
3. Consider moving checkpoint path resolution behind an experiment-local
   resolver so future upstream deletion or relocation of `examples/models/`
   does not break the workflow silently.
4. Consider adding an experiment-local README or manifest that states how to
   obtain/regenerate or verify:
   - `examples/models/GridFM_v0_1.pth`
   - `examples/models/GridFM_v0_2.pth`
   - `tests/data/case30_ieee/processed/*.pt`
5. After any merge, rerun the baseline wildfire tests and at least one small
   GPS/GNN smoke run before regenerating larger result sets.

## Known Difficulty

`git diff HEAD..main` is misleading for this situation because it shows all
branch-local wildfire files and checkpoints as deletions. The correct preview
tool for this step is `git merge-tree` or a temporary merge in a throwaway
worktree/branch. Based on the non-mutating preview, a normal merge of updated
`main` into `wildfire-experiments` should preserve the branch-local wildfire
files. The IEEE-30 processed tensors were initially outside Git protection but
have now been force-added for reproducibility.
