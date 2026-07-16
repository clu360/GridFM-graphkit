# Wildfire Tests

This folder is the active research harness for wildfire-aware GridFM
predict-to-optimize experiments.

## Workflow

Use two verification phases after refactors:

1. Legacy parity first: run legacy-equivalent settings and compare the new
   outputs against archived `methodology_testing_results/old` artifacts.
2. Canonical regeneration second: run the standardized lambda cases and treat
   those outputs as current only after parity checks pass.

Canonical lambda cases use `lambda_R` for wildfire risk and
`lambda_L = 1 - lambda_R`:

```text
risk_leaning:    lambda_R = 0.9, lambda_L = 0.1
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.1, lambda_L = 0.9
```

## Layout

- `gridfm_support/`: GridFM model loading, scenario extraction, and in-memory
  surrogate inference.
- `shared/`: shared objective, risk, plotting, reporting, and path utilities.
- `stage_a_first_pass/`: fixed-topology single-group first-pass runs.
- `stage_b_multigroup/`: multi-group threshold and multistart runs.
- `stage_c_psps_baseline/`: deterministic PSPS baseline runs.
- `stage_d_deenergization/`: limited enumerated de-energization runs.
- `stage_e_gurobi_implementation/`: Gurobi proxy-master candidate generation with GridFM true evaluation.
- `analysis/`: objective analysis, AC-OPF exploration, and parity comparison.
- `results/`: newly generated refactored outputs.
- `methodology_testing_results/`: archived old methodology-test outputs, including legacy exploratory results retained for reference.

The old `wildfire_initial_tests` package remains only as a compatibility
surface while we decide whether it can be removed.

The IEEE-30 processed tensors are intentionally force-tracked because the
current workflow does not have a known regeneration process for them.

## Environment

Do not treat a repository-local `.venv` as part of the wildfire methodology.
Use the active Python environment for the current machine, for example a Conda
environment or system Python with the project dependencies installed. Run
experiment modules with `python -m ...` from the repository root.

Gurobi runs must be launched from a shell/user context that matches the active
license on that device. A virtual environment does not resolve a Gurobi
username/license mismatch.

On Windows, deeply nested result folders under OneDrive can exceed ordinary
tool path handling even when the files are valid. If reads or writes fail with
`FileNotFoundError` on paths that appear in directory listings, map the
repository to a short temporary drive before running or inspecting results:

```powershell
subst G: "C:\path\to\GridFM-graphkit"
G:
python -m experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality --help
```

Remove the temporary mapping when finished:

```powershell
subst G: /D
```

## Canonical Commands

Run modules from the repository root:

```powershell
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_connected_corridor_tradeoffs
python -m experiments.test.wildfire_tests.stage_b_multigroup.run_multi_group_threshold_sensitivity
python -m experiments.test.wildfire_tests.stage_b_multigroup.run_multistart_optimization --tradeoff-sets
python -m experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline
python -m experiments.test.wildfire_tests.stage_d_deenergization.run_stage_d_deenergization
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm_unconstrained
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_unconstrained_frontier
python -m experiments.test.wildfire_tests.analysis.ac_opf_experiment
```

Stage E writes under `results/leq/stage_e/gurobi_gridfm/`. It keeps Stage D
enumeration intact for comparison. Gurobi is used only as a proxy topology
candidate generator; GridFM evaluates the true post-topology state. The true
wildfire exposure is `sum_l z_l * p_env_l * loading_l^2` over the selected
candidate/group scope and does not multiply by the single-line consequence
score `c_l`. `c_l` appears only in the proxy master load-consequence term.

The first real Stage E experiment summary is under:

```text
results/leq/stage_e/gurobi_gridfm/experiment_summaries/
```

Use the revised-objective comparison helper before comparing Stage E to Stage D:

```powershell
python -m experiments.test.wildfire_tests.analysis.run_stage_e_real_experiment_analysis --models gps --cases auto_env largest_group_high --max-deenergized-lines 1 2
```

This helper rebuilds Stage D enumeration and scores it with the Stage E
exposure-only objective, leaving Stage D result folders untouched.

The unconstrained Stage E runner writes under
`results/leq/stage_e/unconstrained/`. It removes the `K` cardinality constraint,
uses no-good cuts to evaluate unique proxy-ranked topologies, and is intended as
a standalone Stage E study rather than a Stage D comparison.

The unconstrained frontier runner writes under
`results/leq/stage_e/unconstrained_frontier/`. It sweeps
`lambda_R = 0.00..1.00` in increments of `0.05`, evaluates 100 unconstrained
topologies per lambda/case, and writes the requested Pareto scatterplot at
`figures/unconstrained_pareto_frontier_scatter.png`.

Expected per-run artifacts:

```text
optimization_summary.json
objective_trace.csv
config.yaml
visualization_summary.json
figures/optimization_behavior.png
figures/ieee30_network_changes.png
```

AC-OPF writes the same artifact surface where meaningful. When no optimized
de-energization artifact exists, the topology plot is still expected to show
the wildfire scenario context and report that no de-energized lines were
provided.

## Verification

Focused wildfire tests:

```powershell
$files = Get-ChildItem tests -Filter 'test_wildfire*.py' | ForEach-Object { $_.FullName }; pytest $files -q
```

Refactor smoke checks:

```powershell
python -m compileall experiments/test/wildfire_tests tests/test_wildfire_refactor_structure.py
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --help
python -m experiments.test.wildfire_tests.stage_d_deenergization.run_stage_d_deenergization --help
```

Use `experiments.test.wildfire_tests.analysis.parity_compare` to compare a
legacy run directory against a refactored legacy-equivalent run directory
before treating standardized-weight outputs as canonical.
