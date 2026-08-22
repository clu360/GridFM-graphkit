# J8 Budgeted Topology Smoke Status

Created: 2026-08-17

## Status

`PASS_WITH_LIMITATIONS`

J8 now has a first budgeted outer-topology smoke for `J-S1` using the locked
Stage J shallow continuous-alpha methodology. This is not the full Stage J
experiment; it is the first topology-loop implementation check after the J7.5
single-topology alpha smokes.

## Settings Used

```text
scenario_id: J-S1
target_branch_ids: 473
lambda_R: 0.8
lambda_R_proxy: 0.8
K: <= 2
topology_budget: 100
continuous_eval_budget: 20 per topology
q: 5 selected load buses per topology
full_coordinate_screen: false
alpha_optimizer: scipy.optimize.minimize, Powell, bounded [0, 1]
alpha_selection_policy: topology_endpoint_neighborhood_then_largest_pd
topology_pool: shared between Guided-DC and Guided-GridSFM
```

The J-S1 target line remains:

```text
branch_id: 473
from_bus: 377
to_bus: 337
edge_family: ac_line
p_env: 1.0
baseline_loading: 1.0000000049826396
rateA: 2.7849
c_l: 0.0
alternate_path_after_single_outage: true
R_base: 1.673539491356386
```

The current outer proxy still uses the Stage J cheap connectivity score:

```text
w_l = p_env_l * baseline_loading_l^2
R_proxy(y) = sum_l w_l (1 - y_l) / sum_l w_l
L_proxy(y) = sum_l c_l y_l
min lambda_R_proxy R_proxy + (1 - lambda_R_proxy) L_proxy
subject to sum_l y_l <= 2
```

The MLD/served-load one-off impact proxy is not active in this J8 smoke. It
remains a later proxy/audit upgrade.

## Result Artifacts

Due to Windows/OneDrive path-length behavior, the successful result folders use
a short output root:

```text
experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08/gdc/
experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08/gsfm/
```

The shared topology pool used by both methods is:

```text
experiments/test/wildfire_tests/goc_500_results/j8_smoke/
  J-S1_lambda0p8_k2_top100_eval20_q5/guided_dc/j8_topology_pool.csv
```

The longer descriptive directory should be treated as a pool/provenance folder,
not the final result folder for this smoke.

One intermediary GridSFM run produced only five eligible topology summaries
because the sandbox context could read existing GridSFM cycle-basis cache files
but could not create missing files under `C:\Users\Caleb Lu\.cache`. The final
GridSFM run was rerun in Caleb's normal user context with:

```text
XDG_CACHE_HOME / --xdg-cache-home:
  C:\Users\Caleb Lu\.gridfm_stage_j\cache\xdg
```

The final artifacts now explicitly report topology coverage:

```text
Guided-DC:      eligible_topology_count = 100, failed_topology_count = 0
Guided-GridSFM: eligible_topology_count = 100, failed_topology_count = 0
```

## Guided-DC Result

```text
status: PASS
runtime_seconds: 109.9085
num_topologies: 100
num_candidate_evaluations: 2000
trace_status_counts: ok = 2000
best_topology_rank: 4
best_topology_id: 276;473
contains_target_branch_473: true
search_objective = J_trade: 0.2801653178
R_norm: 0.3478827992
L_shed_total: 0.0092953921
L_shed_control: 0.0092953921
L_shed_island: 0.0
max_loading: 0.9999999999986418
num_loading_gt_1: 0
evaluation_status: ok
selected_load_ids: 156;157;158;262;276
best_selected_alpha:
  156: 1.0
  157: 0.00010696
  158: 1.0
  262: 1.0
  276: 1.0
```

Top five Guided-DC topology summaries by search objective:

```text
rank  topology  J_trade      R_norm      L_shed_total  target_hit
4     276;473   0.28016532   0.34788280  0.00929539    true
8     465;473   0.28122626   0.35153282  0.00000000    true
42    473;520   0.28140471   0.35175588  0.00000000    true
7     473;493   0.28447089   0.35558861  0.00000000    true
46    151;473   0.28520092   0.35497724  0.00609564    true
```

## Guided-GridSFM Result

```text
status: PASS
runtime_seconds: 826.3133
num_topologies: 100
num_candidate_evaluations: 2000
trace_status_counts: model_output_penalized = 2000
best_topology_rank: 4
best_topology_id: 276;473
contains_target_branch_473: true
search_objective = J_total: 0.3653721057
J_trade: 0.3648471803
R_norm: 0.4537537752
L_shed_total: 0.0092208011
L_shed_control: 0.0092208011
L_shed_island: 0.0
PAC_operational: 0.0001584854
PAC_AC: 0.0001039773
PAC_model: 0.0
PAC_total: 0.0002624627
max_loading: 1.3091901313
num_loading_gt_1: 5
feasibility_head: 0.9898030758
evaluation_status: model_output_penalized
selected_load_ids: 156;157;158;262;276
best_selected_alpha:
  156: 1.0
  157: 0.00813062
  158: 1.0
  262: 1.0
  276: 1.0
```

Top five Guided-GridSFM topology summaries by search objective:

```text
rank  topology  J_total     R_norm      L_shed_total  target_hit  max_loading
4     276;473   0.36537211  0.45375378  0.00922080    true        1.30919013
2     290;473   0.37343570  0.46390443  0.00914631    true        1.26643133
5     473;505   0.38025425  0.47365566  0.00304645    true        1.31367625
1     285;473   0.39296195  0.48915826  0.00000000    true        1.40893582
3     345;473   0.47056071  0.52252515  0.00068573    true        1.82096671
```

## Figures

Figure root:

```text
experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08/figures/
```

### Selection Objective

This plot compares the actual method-specific selection objective over the
shared 100-topology pool. For Guided-DC, the selection objective is `J_trade`.
For Guided-GridSFM, the selection objective is `J_total = J_trade + rho PAC`.

![J8 selection objective by topology rank](../../../goc_500_results/j8s/s1_l08/figures/j8_objective_by_topology_rank.png)

### Risk And Load-Shed Components

This plot separates the objective components, showing that topology decisions
drive most of the variation in `R_norm`, while the shallow `q=5` alpha search
only produces small changes in `L_shed_total`.

![J8 risk and load shed by topology rank](../../../goc_500_results/j8s/s1_l08/figures/j8_risk_load_by_topology_rank.png)

### Pareto Scatter

This plot shows all 100 evaluated topology-finalist points per method in
`L_shed_total` versus `R_norm` space, with the selected best point highlighted
for each method.

![J8 Pareto scatter](../../../goc_500_results/j8s/s1_l08/figures/j8_pareto_scatter.png)

### Target-Hit Behavior

This plot records whether each topology contains the J-S1 target branch `473`.
In this smoke, all 100 generated topologies contain the target branch, which is
consistent with `lambda_R_proxy = 0.8`, `p_env_473 = 1.0`, and `c_473 = 0.0`.

![J8 target hit by topology rank](../../../goc_500_results/j8s/s1_l08/figures/j8_target_hit_by_rank.png)

### GridSFM PAC And Loading

This plot isolates the GridSFM physics-aware penalty and predicted maximum
loading by topology rank. The large spikes show specific topologies where
GridSFM predicts substantially worse operational behavior.

![J8 GridSFM PAC and loading by topology rank](../../../goc_500_results/j8s/s1_l08/figures/j8_gridsfm_pac_loading_by_rank.png)

### Runtime

This plot summarizes total runtime and per-candidate runtime for the 2,000
candidate evaluations per method.

![J8 runtime summary](../../../goc_500_results/j8s/s1_l08/figures/j8_runtime_summary.png)

The figure manifest is:

```text
experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08/figures/
  j8_figure_manifest.csv
```

## Interpretation For This Smoke

The first topology-loop smoke supports the running hypothesis for J-S1:

- the outer proxy repeatedly prioritizes the high-risk, low-impact target line
  `473`;
- both Guided-DC and Guided-GridSFM choose the same best topology `276;473`;
- load shedding remains small, and the alpha optimizer behaves as shallow
  corrective recourse rather than the main risk-reduction mechanism;
- DC enforces its represented physics directly and reports no thermal overloads;
- GridSFM still reports nonzero overload behavior, but the calibrated
  `PAC_total` is small in this smoke and the selected candidate is retained as
  `model_output_penalized` rather than rejected.

## Limitations

- This is one scenario and one lambda setting only.
- The alpha search uses only `q=5` selected loads and `20` candidate evaluations
  per topology.
- The selected-load policy is a cheap deterministic heuristic, not a full
  per-load alpha optimizer.
- Reference A fixed-`z,alpha` economic AC-OPF, Reference B MLD, and warm-start
  audits were not run in this J8 smoke.
- The MLD/served-load one-off impact proxy has not yet replaced or augmented
  the current `c_l` connectivity proxy.
- GridSFM `PAC_model` demand-command consistency remains unavailable/off for
  the current OPF-surrogate API because `predict()` does not output `Pd/Qd`.

## Tests And Checks Run

```text
pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q
22 passed, 2 skipped, 3 warnings
```

Both method runs wrote summary JSON, topology summary CSV, alpha trace CSV, and
best selected-alpha CSV artifacts under the short `j8s/s1_l08` result root.

The runner now marks the top-level JSON status as `PASS` only when every
topology has an eligible candidate. Partial backend coverage is reported as
`PARTIAL_FAIL`.

## Implementation Auditor Check

The workflow implementation auditor was run after the corrected GridSFM rerun.
Bounded audit result:

```text
PASS
```

Auditor-confirmed evidence:

```text
Guided-DC:
  status = PASS
  eligible_topology_count = 100
  failed_topology_count = 0
  trace_status_counts = {"ok": 2000}

Guided-GridSFM:
  status = PASS
  eligible_topology_count = 100
  failed_topology_count = 0
  trace_status_counts = {"model_output_penalized": 2000}
  xdg_cache_home = C:\Users\Caleb Lu\.gridfm_stage_j\cache\xdg

Shared topology pool:
  100 unique guided topologies
  K = 2
  sequence mismatches between DC and GridSFM summaries = 0
```

## Next Steps

1. Decide whether to keep the current `q=5`, `20`-evaluation alpha budget for
   the next smoke or test a smaller budget such as `10` evaluations per topology.
2. Extend from one lambda/scenario to the coupled sweep for S1-S3 only after the
   J8 smoke plots and auditor pass are reviewed.
3. Add Reference A/B exact AC audits for selected finalists once the topology
   smoke behavior is accepted.
