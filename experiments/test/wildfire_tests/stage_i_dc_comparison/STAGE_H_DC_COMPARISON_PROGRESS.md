# Stage H DC Approximation + Baseline Heuristic Comparison Progress

## Current status

Stage H K<=2 implementation, smoke checks, full main results, corrected MLD,
proxy-inner lambda dimensionality study, plots, AC projection attempts, and
methodology fidelity audits are complete.

Important update: the implementation now distinguishes `lambda_R_proxy` from the
inner/continuous `lambda_R`. The main sweep remains methodologically valid
because it intentionally uses coupled lambdas. The corrected MLD run at `MLD/r5`
uses:

```text
lambda_R_proxy = 1
lambda_R       = 0
```

Implemented package:

```text
experiments/test/wildfire_tests/stage_i_dc_comparison/
```

Primary runner:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py
```

Final result roots:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/
    main_results/
    MLD/
```

Because the requested folder name is long and the repo is under a long OneDrive
path, run folders are intentionally short (`r`, `r1`, `r2`, ...). Use explicit
Windows long-path handling (`\\?\`) when auditing deeply nested plots.

## Implemented methodology pieces

- Stage E K2 GridFM reference rows reused from:

```text
tmp/stage_h_dc_stage_e_reference/full/run_20260714_015005
```

- Stage I-a fixed-topology DC recourse with independent Stage-E-style proxy
  topology generation and no-good/no-revisit cuts.
- Stage I-b direct DC MIQP with `sum y_l <= 2`, `MIPGap=1e-4`, and
  `TimeLimit=600`.
- MATPOWER branch audit using `rateA`, `x`, taps, and shifts.
- Shared stored-baseline denominator through the existing Stage G scenario
  baseline plumbing.
- Budget-compatible TH top-1, TH top-2, and AH K2 heuristic points.
- Common operational diagnostic components for DC rows.
- Separate target recall and precision metrics in
  `expected_vs_selected_by_rho.csv`.
- SciPy/SLSQP v1 AC projection attempts for selected finalists only, with
  nonfatal failure logging and cache-by-solution identity.
- Long-path-safe table, JSON, and plot writes under the requested Stage H root.

## Smoke checks completed

Unit tests:

```text
python -m pytest tests/test_wildfire_stage_i_dc_comparison.py -q
```

Result:

```text
3 passed
```

Main smoke:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py --smoke
```

Latest completed smoke:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/main_results/r9
```

MLD smoke:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py --smoke --mld
```

Latest completed smoke:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/MLD/r3
```

Both smoke runs included Stage E K2, Stage I-a, Stage I-b, TH top-1/top-2,
and AH K2, and both passed all hard methodology checks.

## Full main result

Run:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/main_results/r11
```

Summary:

```text
status: complete
num_best_rows: 300
num_projection_rows: 300
num_methodology_checks: 12
num_failed_checks: 0
```

Expected row structure:

```text
5 scenarios x 5 lambda_R values x 2 rho values x 6 methods = 300 best rows
```

Method counts:

```text
stage_e_k2              50
stage_i_a_dc_k2         50
stage_i_b_dc_miqp_k2    50
th_top1                 50
th_top2                 50
ah_k2_budgeted          50
```

Additional table counts:

```text
stage_i_a_topology_pool.csv    2500
dc_recourse_results.csv        5200
ac_projection_distances.csv     300
```

Projection outcomes:

```text
DC solved                                    150
DC solver_returned_infeasible_or_residual    100
GridFM solved                                 10
GridFM solver_returned_infeasible_or_residual 40
finite projection distances                  160 / 300
```

The projection failures are expected to be nonfatal under the plan. They are
logged with status/message fields and `D_proj_total = NaN` where no feasible
projection was certified.

## Full MLD result, corrected

Run:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/MLD/r5
```

This run replaced the stale pre-correction MLD artifact. It uses the corrected
MLD framing:

```text
lambda_R_proxy = 1.0
lambda_R       = 0.0
```

Summary:

```text
status: complete
num_best_rows: 60
num_projection_rows: 60
num_methodology_checks: 12
num_failed_checks: 0
```

Expected row structure:

```text
5 scenarios x 1 lambda_R value x 2 rho values x 6 methods = 60 best rows
```

Method counts:

```text
stage_e_k2              10
stage_i_a_dc_k2         10
stage_i_b_dc_miqp_k2    10
th_top1                 10
th_top2                 10
ah_k2_budgeted          10
```

Additional table counts:

```text
stage_i_a_topology_pool.csv     500
dc_recourse_results.csv        1040
ac_projection_distances.csv      60
```

Projection outcomes:

```text
DC solved                                     32
DC solver_returned_infeasible_or_residual     18
GridFM solver_returned_infeasible_or_residual 10
finite projection distances                   32 / 60
```

Audit:

```text
best rows have lambda_R       = [0.0]
best rows have lambda_R_proxy = [1.0]
max num_shutoff_lines         = 2
methodology failed checks     = 0
```

## Proxy-inner lambda dimensionality study

Run:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2
```

Runner:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_proxy_inner_lambda_sweep.py
```

Purpose:

```text
Compare Stage E K2 GridFM and Stage I-a DC guided K2 only.
Hold rho = 0.
Generate topology pools by lambda_R_proxy.
Evaluate each proxy-generated topology across the inner lambda_R sweep.
Use J = lambda_R R_norm + (1 - lambda_R) L_shed for fair comparison.
```

Summary:

```text
status: complete
num_all_evaluated_rows: 22830
num_best_rows: 250
num_frontier_rows: 970
num_methodology_checks: 6
num_failed_checks: 0
num_plot_files: 90
runtime_seconds: 22267.61
```

Expected row structure:

```text
5 scenarios x 5 lambda_R_proxy values x 5 inner lambda_R values x 2 methods
= 250 best rows
```

Key table counts:

```text
stage_e_proxy_topology_pool.csv          2500
stage_e_topology_pool.csv               12500
stage_e_continuous_recourse_results.csv 12500
stage_i_a_proxy_topology_pool.csv        2500
stage_i_a_topology_pool.csv             12500
stage_i_a_dc_recourse_results.csv       12500
all_evaluated_proxy_inner_points.csv    22830
best_by_scenario_proxy_inner_stage.csv    250
pareto_frontier_points_by_proxy.csv       970
```

Required values were verified:

```text
scenarios = S1, S2, S3, S4, S5
lambda_R_proxy = [0.0, 0.2, 0.5, 0.8, 1.0]
lambda_R       = [0.0, 0.2, 0.5, 0.8, 1.0]
rho_phys       = [0.0]
stages         = stage_e_k2, stage_i_a_dc_k2
```

Plot structure:

```text
plots/by_scenario/S*/proxy_lambda_*/pareto_frontier_scatter.png
plots/by_scenario/S*/proxy_lambda_*/traditional_lambda_objective_convergence.png
plots/by_scenario/S*/proxy_lambda_*/best_metrics_by_inner_lambda.png
plots/by_scenario/S*/best_J_no_physics_heatmap.png
plots/by_scenario/S*/best_L_shed_heatmap.png
plots/by_scenario/S*/best_R_norm_heatmap.png
```

Implementation note: Stage E live checkpoints for this long run were written to
the short path `tmp/stage_h_proxy_inner_checkpoints/r2/` to avoid Windows
long-path checkpoint failures, then final tables/plots were written under the
requested Stage H result folder.

## Generated plots

Key interpretation finding:

- The DC comparison is now an important grounding check on the current GridFM
  implementation. Stage E GridFM can produce very large and physically
  unrealistic predicted loading, which directly inflates `R_norm` when
  `lambda_R > 0`. More importantly, this surrogate physical-realism issue can
  indirectly affect recourse quality even when `lambda_R = 0`, because the
  GridFM-predicted post-topology state still mediates load-shedding estimates
  and controlled-service behavior. This suggests that observed Stage E
  performance gaps should be interpreted partly as limitations of the current
  learned surrogate under topology/control interventions, not only as
  weaknesses of the outer/inner optimization methodology.

The full main run `main_results/r11` was regenerated after the initial run to
improve the per-rho diagnostic plots. The updated `r11` plot semantics are:

- `expected_vs_selected_shutoff_lines.png` includes the scenario target set in
  the title and each cell reports selected lines, target hits, missed targets,
  non-target selected lines, recall, precision, and source-less-island status.
- `pareto_frontier_scatter.png` uses all finite evaluated solutions, not only
  final selected rows. It overlays evaluated points and nondominated points for
  the six Stage H families.
- `traditional_lambda_objective_convergence.png` was added under every
  `plots/per_rho/rho*/S*/` folder. It uses:

```text
J = lambda_R R_norm + (1 - lambda_R) L_shed
```

and intentionally excludes the `rho_phys * PAC_total` physics penalty so rho=2
comparisons remain fair against DC methods.

Updated full-main audit tables:

```text
tables/all_evaluated_stage_h_points.csv
tables/pareto_frontier_points_all_evaluated.csv
tables/traditional_lambda_objective_convergence.csv
```

Required plot families include:

```text
plots/per_rho/rho*/S*/pareto_frontier_scatter.png
plots/per_rho/rho*/S*/effective_load_shedding_by_lambda.png
plots/per_rho/rho*/S*/common_operational_diagnostic_by_lambda.png
plots/per_rho/rho*/S*/expected_vs_selected_shutoff_lines.png
plots/per_rho/rho*/S*/traditional_lambda_objective_convergence.png
plots/projection_distance_by_lambda.png
plots/stage_i_b_solver_diagnostics.png
```

Additional no-GridFM comparison plots were generated for both `main_results/r11`
and corrected `MLD/r5` to make the DC and heuristic families easier to compare
without Stage E's large GridFM-driven scale dominating the axes:

```text
plots/per_rho/rho*/S*/pareto_frontier_scatter_no_gridfm.png
plots/per_rho/rho*/S*/traditional_lambda_objective_convergence_no_gridfm.png
```

These exclude `stage_e_k2` and include:

```text
stage_i_a_dc_k2
stage_i_b_dc_miqp_k2
th_top1
th_top2
ah_k2_budgeted
```

Generation audit:

```text
main_results/r11: 10 no-GridFM pareto plots + 10 no-GridFM convergence plots
MLD/r5:          10 no-GridFM pareto plots + 10 no-GridFM convergence plots
```

Stage I-b MIQP solution-pool refresh was added after the first no-GridFM plots.
This does not trace the chronological branch-and-bound incumbent stream.
Instead, it reruns each Stage I-b MIQP with Gurobi solution-pool settings and
uses the deduplicated pool solutions as additional evaluated Stage I-b topology
points. This is useful for Pareto/topology-set comparison, but should be labeled
as a solution-pool view rather than a true B&B trajectory. The original
certified Stage I-b incumbent rows from `all_evaluated_stage_h_points.csv` must
remain in the augmented table; solution-pool rows are supplemental and should
not replace the certified incumbent. This matters because `PoolSearchMode=2`
can return a useful pool whose first listed solution does not match the
separately certified incumbent for this MIQP/QP setting.

The refreshed plots now include the MIQP pool in both the with-GridFM and
no-GridFM variants:

```text
plots/per_rho/rho*/S*/pareto_frontier_scatter.png
plots/per_rho/rho*/S*/traditional_lambda_objective_convergence.png
plots/per_rho/rho*/S*/pareto_frontier_scatter_no_gridfm.png
plots/per_rho/rho*/S*/traditional_lambda_objective_convergence_no_gridfm.png
```

Additional MIQP-pool tables:

```text
tables/stage_i_b_solution_pool_raw.csv
tables/stage_i_b_solution_pool.csv
tables/all_evaluated_stage_h_points_with_miqp_pool.csv
tables/traditional_lambda_objective_convergence_with_miqp_pool.csv
tables/stage_i_b_solution_pool_refresh_summary.json
```

MIQP-pool refresh audit:

```text
main_results/r11:
  pool rows after rho/topology deduplication: 1082
  unique Stage I-b pool topologies: 142
  augmented evaluated points: 10364
  refreshed plots: 10 each for pareto, convergence, pareto_no_gridfm, convergence_no_gridfm

MLD/r5:
  pool rows after rho/topology deduplication: 500
  unique Stage I-b pool topologies: 50
  augmented evaluated points: 2326
  refreshed plots: 10 each for pareto, convergence, pareto_no_gridfm, convergence_no_gridfm
```

Additional per-lambda no-Stage-E Pareto plots were generated for the main
results so each scenario/rho folder has a direct scatter view for every
`lambda_R` case without the Stage E GridFM K2 family on the axes. Each folder
contains the full set:

```text
plots/per_rho/rho*/S*/pareto_frontier_scatter_no_stage_e_by_lambda/
  pareto_frontier_scatter_no_stage_e_lambda_R_0.png
  pareto_frontier_scatter_no_stage_e_lambda_R_0p2.png
  pareto_frontier_scatter_no_stage_e_lambda_R_0p5.png
  pareto_frontier_scatter_no_stage_e_lambda_R_0p8.png
  pareto_frontier_scatter_no_stage_e_lambda_R_1.png
```

The `lambda_R=0` case is included in the same subfolder as requested. The
generation manifest is:

```text
tables/per_lambda_no_stage_e_pareto_manifest.csv
```

Per-lambda no-Stage-E Pareto audit:

```text
main_results/r11: 50 plots total
  = 2 rho values x 5 scenarios x 5 lambda_R cases
```

DC runtime comparison tables were added for Stage I-a versus Stage I-b:

```text
tables/dc_runtime_comparison_stage_i_a_vs_i_b.csv
```

Runtime comparison semantics:

```text
Stage I-a runtime = sum of saved successful fixed-topology DC recourse solve
                    runtimes per scenario/lambda/proxy case, using rho=0 rows
                    to avoid duplicate rho accounting.

Stage I-b runtime = saved direct MIQP solve runtime per scenario/lambda/proxy
                    case, using rho=0 rows to avoid duplicate rho accounting.
```

This comparison does not include any unrecorded Python overhead or Stage I-a
topology proposal overhead beyond what is saved in `runtime_seconds`, so it is
best interpreted as a saved-solve-runtime comparison.

Runtime audit:

```text
main_results/r11:
  Stage I-a total saved recourse runtime: 10.9501 s across 2066 recourse solves
  Stage I-b total saved MIQP runtime:      1.7108 s across 25 MIQP solves
  Total saved-runtime ratio I-a / I-b:     6.4005x

MLD/r5:
  Stage I-a total saved recourse runtime: 1.5883 s across 398 recourse solves
  Stage I-b total saved MIQP runtime:     0.0602 s across 5 MIQP solves
  Total saved-runtime ratio I-a / I-b:    26.3650x
```

The intentionally omitted figure remains omitted:

```text
num_shutoffs_vs_objective_by_method.png
```

## Audit notes

- Full main and MLD use `rho_phys = [0.0, 2.0]`.
- Full main uses `lambda_R = [0.0, 0.2, 0.5, 0.8, 1.0]`.
- Corrected MLD uses `lambda_R = [0.0]` and `lambda_R_proxy = [1.0]`.
- Proxy-inner lambda study uses `rho_phys = [0.0]` and only Stage E K2 plus
  Stage I-a DC guided K2.
- All selected rows satisfy `num_shutoff_lines <= 2`.
- Stage D exhaustive and Stage E unconstrained are absent from Stage H outputs.
- Branch tap/shift audit found trivial taps/shifts for this MATPOWER case.
- Stage I-b rows include solver metadata, including `mip_gap`,
  `runtime_seconds`, `node_count`, `solution_count`, `time_limit_reached`, and
  `optimality_certified`.
- DC residual fidelity passes with `tol = 5e-4`; the largest full-main residual
  was `max_balance = 3.52244e-4` and `max_angle_flow = 1.09275e-4`, both from
  Stage I-b MIQP numerical tolerances.

## Current stopping point for next session

Last updated after generating the Stage I-a proxy-inner summary figures.

The current completed result sets to preserve are:

```text
main_results/r11
MLD/r5
proxy_inner_lambda_sweep/r2
```

The newest completed task generated compact presentation-oriented Stage I-a
summary plots comparing:

```text
main coupled Stage I-a:
  lambda_R_proxy = lambda_R

best proxy-inner Stage I-a setting:
  lambda_R_proxy selected per scenario
  inner lambda_R swept over [0, 0.2, 0.5, 0.8, 1.0]
```

The figures were written to:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S*/
  stage_i_a_main_vs_best_proxy/
    stage_i_a_pareto_main_vs_best_proxy.png
    stage_i_a_traditional_objective_main_vs_best_proxy.png
```

The generated manifest is:

```text
proxy_inner_lambda_sweep/r2/tables/stage_i_a_summary_figure_manifest.csv
```

Best Stage I-a proxy settings selected by the rule
`min mean J over lambda_R_inner > 0; tie by mean R_norm then mean L_shed`:

```text
S1: lambda_R_proxy = 1.0
S2: lambda_R_proxy = 0.5
S3: lambda_R_proxy = 0.5
S4: lambda_R_proxy = 0.0
S5: lambda_R_proxy = 1.0
```

The focused handoff for these plots and findings is:

```text
experiments/test/wildfire_tests/stage_i_dc_comparison/
  STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md
```

The refresh script is:

```text
experiments/test/wildfire_tests/stage_i_dc_comparison/
  run_proxy_inner_stage_ia_summary_figures.py
```

Primary conclusion from the new summary figures:

- Separating `lambda_R_proxy` from inner `lambda_R` is useful for Stage I-a
  because it can expand the discovered DC Pareto frontier and improve selected
  scalarized objective slices.
- The best proxy value is scenario-dependent, so the coupled convention
  `lambda_R_proxy = lambda_R` should be treated as one baseline, not the only
  defensible search design.
- S1 and S5 support the intuitive risk-heavy-proxy story.
- S4 is the important counterexample where `lambda_R_proxy = 0` gives the best
  mean nonzero objective, showing that load-delivery-oriented topology search
  can still reveal topologies that become useful under risk-aware recourse.

Important caveat:

```text
proxy_inner_lambda_sweep/r2 does not yet have AC projection distances.
```

Therefore, the proxy-inner sweep currently supports statements about DC
tradeoff-front quality and traditional objective convergence, but it does not
yet support a claim about improved AC projection distance or AC-feasible
realizability. The next clean follow-up is to run AC projection on selected
proxy-inner finalists and compare them against the `main_results/r11` Stage I-a
finalists.

Practical Windows/OneDrive path note:

- Some direct PowerShell `Get-ChildItem` / `Get-Content` calls can fail on the
  long Stage H result path even when files exist.
- `rg --files` and `rg -n` were reliable for checking the proxy-inner result
  files.
- If a laptop session has trouble reading the long final result folder, work
  from a deeper `workdir`, use `rg`, or copy a target result subfolder to a
  short temporary path for inspection.

## Reproduction commands

Use the Stage E reference run to avoid recomputing the 5,000 GridFM topology
continuous evaluations:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py --stage-e-reference-run tmp/stage_h_dc_stage_e_reference/full/run_20260714_015005
```

MLD:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py --mld
```

Proxy-inner lambda dimensionality study:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_proxy_inner_lambda_sweep.py
```

Focused Stage I-a proxy-inner summary figures and interpretation:

```text
experiments/test/wildfire_tests/stage_i_dc_comparison/
  STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md
```

Refresh those summary figures with:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_proxy_inner_stage_ia_summary_figures.py
```

Refresh Stage I-b MIQP solution-pool overlays for both `main_results/r11` and
`MLD/r5`:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_miqp_pool_refresh.py
```

Regenerate the per-lambda no-Stage-E Pareto plots for `main_results/r11`:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_per_lambda_no_stage_e_pareto.py
```

Do not reuse `tmp/stage_h_dc_stage_e_reference/full/run_20260714_015005` for
corrected MLD, because that reference was generated with coupled
`lambda_R_proxy = lambda_R`. Corrected MLD needs a Stage E reference whose tables
contain `lambda_R_proxy = 1` and `lambda_R = 0`; current corrected MLD `r5`
already satisfies this.

The local Gurobi license must be available under the licensed OS user. If the
sandbox reports a Gurobi user mismatch, rerun outside the sandbox.
