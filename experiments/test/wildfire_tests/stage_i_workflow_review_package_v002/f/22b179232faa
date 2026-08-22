# Wildfire Current State Summary

## Next Session Pickup - July 13, 2026

The next work session should start from the methodology plan devised in the
separate agent: implement a DC approximation comparison, add an AC projection
step to check solution gaps, and use a common operational diagnostic across
GridFM, DC, AC-projected, and heuristic methods.

Open logistics:

- implement DC approximation plus AC projection/evaluation;
- align all methods on objective accounting and operational diagnostics;
- run the comparison experiments and regenerate plots/tables.

Scenario logistics for the first Stage I implementation are now locked: use the
existing five scenarios `S1` through `S5` exactly as before. Do not switch to
fairer `p_env` construction in this experiment; fair-scenario construction is a
later extension.

Research framing for the next professor meeting: compare the current
GridFM-based methodology against heuristic methods to identify limitations and
strengths, then use robust optimization as the next layer for understanding and
improving decision reliability.

## Stage I DC Approximation / AC Projection Locks - July 13, 2026

The Stage I methodology has been clarified before implementation:

```text
Stage I-a:
  its own Stage-E-style guided K2 topology proposal loop, with fixed-topology
  DC recourse as the evaluator.

Stage I-b:
  joint topology + continuous DC optimization, solved as DC MIQP when squared
  flow-risk is retained, with sum_{l in C} y_l <= 2.
```

The current Stage I comparison is locked to `K <= 2`. The main comparison
includes Stage E `K2`, Stage I-a guided-search DC `K <= 2`, Stage I-b DC MIQP
with `sum_{l in C} y_l <= 2`, and budget-compatible TH/AH `K <= 2` heuristic
points.

Do not include Stage E unconstrained or Stage I-b unconstrained/higher-K in the
main Pareto/frontier comparison. Those are later topology-flexibility
extensions.

Do not include Stage D exhaustive in the main Stage I comparison. It remains
historical context from prior work, but this experiment emphasizes guided
search and DC approximation methods rather than exhaustive enumeration.

Stage I-b solver settings are locked:

```text
MIPGap = 1e-4
TimeLimit = 600 seconds initially, adjustable after smoke tests
save incumbent if TimeLimit is reached
label time-limited / not certified optimal if final MIPGap > 1e-4
```

Stage I-b must save `solver_status`, `objective_value`, `best_bound`,
`mip_gap`, `runtime_seconds`, `node_count`, `solution_count`,
`time_limit_reached`, and `optimality_certified`. The MIP gap lock applies only
to Stage I-b MIQP, not Stage I-a LP/QP fixed-topology recourse.

The primary cross-method risk denominator is the stored-scenario baseline:

```text
R_baseline_shared =
  sum_{l in L_phys} p_env_l * (baseline_loading_l_stored)^2
```

Branch modeling must explicitly check MATPOWER taps and phase shifts. Use
`B_l = baseMVA / x_l` only when taps/shifts are trivial; otherwise use
`B_l = baseMVA / (x_l * tau_l)` and
`f_l = B_l * (theta_i - theta_j - phi_l)`. Do not silently ignore nonzero taps
or phase shifts.

Stage I-b must include the apples-to-apples `sum_{l in C} y_l <= 2` budgeted
variant. No `K <= 5`, unconstrained-over-`C`, or Stage E unconstrained variant
belongs in this current main experiment.

Runtime expectation before smoke calibration:

```text
GridFM Stage E K2:
  up to 5,000 GridFM continuous topology evaluations
  = 5 scenarios x 5 lambdas x 2 rho x 100 topology candidates

Stage I-a DC guided search:
  up to 2,500 fixed-topology DC recourse solves
  = 5 scenarios x 5 lambdas x 100 topology candidates

Stage I-b DC MIQP:
  25 MIQP solves before rho duplication
  = 5 scenarios x 5 lambdas
  TimeLimit=600 seconds gives about 4.2 hours hard worst-case MIQP time

AC projection:
  150 attempted finalist rows before caching
  expected unique solves closer to 100 because DC methods are rho-invariant
```

Planning estimate:

```text
optimistic:    2-4 hours
conservative: 6-10 hours
worst case:   10-20 hours if MIQP/projection repeatedly hit limits
```

The smoke run should update this estimate using observed per-job runtimes
before launching the full run.

AC projection remains required for selected finalists only. The default
required finalist set is `5 scenarios x 5 lambdas x 2 rho panels x 3 method
families = 150` attempted rows before caching, where the families are GridFM Stage E K2,
Stage I-a DC `K <= 2`, and Stage I-b DC `K <= 2`. If Stage D exhaustive AC
projection is intentionally added, it becomes a fourth family and raises the
uncached upper bound, but Stage D projection is not part of the default main
experiment.

Projection solves should be cached by solution identity so duplicate DC
solutions across `rho=0` and `rho=2` are not solved twice. Projection distance
means the minimum distance to the nearest AC-feasible operating point under the
same topology, not a residual.

GridFM projection is control-faithful:

```text
D_proj_GridFM* =
  min_{x_AC in F_AC(z_GridFM*)}
    D_Pg_cmd + D_Qg_cmd + D_s_cmd
```

Non-selected buses are not forced to match raw GridFM predictions. DC
projection is active-power-oriented:

```text
D_proj_DC* =
  min_{x_AC in F_AC(z_DC*)}
    D_Pg_DC + D_s_DC + D_f_DC
```

where `D_f_DC` compares AC from-end active branch flow against signed DC
`f_l` in the same canonical orientation. Angle distance is disabled by default
with `w_theta = 0`.

Pareto plots should use `x = L_shed` and `y = R_norm`, connecting only points
within the same method family, scenario, budget, and lambda sweep convention.
Duplicate topology points across methods should be made visible with jitter,
annotations, or z-ordering.

Locked visualization scope for this `K <= 2` experiment:

```text
cross_rho:
  effective load shedding by method
  physics feasibility sensitivity by method

per_rho:
  expected vs selected shutoffs
  target recall / precision / recall-precision metrics
  Pareto frontier scatter by scenario
  traditional lambda objective, clearest for rho=0
  common operational diagnostic by lambda
  AC projection distance for finalists
  Stage I-b solver diagnostics
```

For this experiment, omit `num_shutoffs_vs_objective` from the main plot set
because every topology-producing method is restricted to at most two line
shutoffs.

The key Pareto plot should be per scenario and rho panel, with:

```text
x = L_shed
y = R_norm
lambda_R sweep = [0.0, 0.2, 0.5, 0.8, 1.0]
```

Expected primary curves:

```text
Stage E K2
Stage I-a guided-search DC K <= 2
Stage I-b DC K <= 2
```

Also add sparse heuristic frontiers where useful:

```text
TH K <= 2, combining TH top-1 and TH top-2 points
AH K <= 2
```

Physics-infeasibility summary tables using the existing GridFM `PAC_total`
should be restricted to methods evaluated by the existing GridFM formulation:

```text
TH top-1
TH top-2
Stage E K2
AH
```

Stage I-a and Stage I-b should instead report DC residuals,
`PAC_common_op_overlap`, and AC projection distance.

Also run a smaller MLD literature-alignment sub-study under the same Stage H
comparison family. This is separate from the main lambda sweep and is meant to
align with maximum-load-delivery style wildfire outage comparisons.

For Stage E K2 and Stage I-a MLD:

```text
lambda_R_proxy = 1
lambda_R_inner = 0
lambda_L_inner = 1
topology_budget = K <= 2
rho_panels = [0, 2]
```

Interpretation:

```text
rho = 0:
  pure MLD-style load-delivery recourse

rho = 2:
  load-delivery recourse plus GridFM soft PAC penalty for GridFM-evaluated rows
```

Include TH/AH budget-compatible `K <= 2` MLD-style heuristic evaluations by
using their shutoff construction and setting the inner/evaluation
`lambda_R = 0`. Useful MLD visuals are expected-vs-selected shutoffs,
target recall/precision, a small `L_shed` vs `R_norm` scatter, effective load
shedding by method, GridFM-only physics/PAC sensitivity, common operational
diagnostic, and finalist AC projection distance where available.

Recommended Stage H result layout:

```text
results/leq/stage_h/
  dc_approximation_th_ah_heuristics_comparison/
    main_k2_lambda_sweep/
    mld_literature_alignment/
```

## Current Implemented Checkpoint - Revised GridFM Load/PAC Stage H

As of July 11, 2026, the GridFM-side load-shedding and PAC revision has been
implemented, tested, and run through the Stage H comparison.

Primary result folder:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

The run was first executed to a short temp path because the intended OneDrive
path exceeds normal Windows path limits, then copied back and finalized with
long-path handling. Finalized run summary:

```text
topology_rows_before_rho: 90
continuous_evaluations: 180
hard_methodology_failures: 0
```

Comparison coverage is preserved:

```text
Stage D exhaustive:       30 reference rows
Stage E k2:               30 reference rows
Stage E unconstrained:    30 reference rows
TH:                      150 best rows
AH:                       30 best rows
```

Validation:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py tests/test_wildfire_stage_h_heuristic_comparison.py -q
18 passed
```

The primary objective now uses `L_shed_hybrid`, while saving
`L_shed_cmd`, `L_shed_gridfm_raw`, `L_shed_gridfm_effective`, and
`L_shed_hybrid`. The corrected hybrid rule is:

```text
alpha_hybrid_i = 0.0                       if source-less islanded
alpha_hybrid_i = alpha_cmd_i               if selected and connected
alpha_hybrid_i = alpha_gridfm_effective_i  if non-selected and connected
```

where `alpha_gridfm_raw_i = Pd_raw_i / Pd_base_i` and
`alpha_gridfm_effective_i = clip(alpha_gridfm_raw_i, 0, 1)`.

The PAC score is now decomposed as:

```text
PAC_total =
    pac_operational_weight       * PAC_operational
  + pac_ac_weight                * PAC_AC
  + pac_model_consistency_weight * PAC_model_consistency
```

All group weights default to `1.0`. Branch-flow consistency is explicitly
unavailable/zero-weight because GridFM has no independent branch-flow output.
AC P/Q balance was available for all 180 revised Stage H rows.

Key finding: the old commanded-only connected-load accounting materially
understated GridFM-implied non-selected load degradation. Across the full
revised Stage H run:

```text
avg(L_shed_hybrid - L_shed_cmd):          0.2390
max(L_shed_hybrid - L_shed_cmd):          0.4943
avg(L_shed_gridfm_effective - L_shed_cmd): 0.2777
max(L_shed_gridfm_effective - L_shed_cmd): 0.6136
```

Average PAC decomposition:

```text
PAC_operational:        57.9186
PAC_AC:                 10.8316
PAC_model_consistency:  92.2338
PAC_total:             160.9840
```

Next methodological step: use this revised GridFM evaluator as the baseline for
the DC MILP construction and fairer scenario comparison. The DC MILP should not
be based on the earlier July 8 source-less-island correction alone.

## Historical Pre-Implementation Note - GridFM Load And PAC Revision

As of July 11, 2026, the next active methodological step is no longer the DC
MILP itself. Before finalizing the DC comparison, the GridFM-side Stage G/H
formulation should be revised so the load-shedding and physics-infeasibility
terms are not overstated relative to what is actually implemented, calculated,
and displayed.

The motivation is a remaining limitation in the corrected effective-alpha
accounting. The July 8 correction fixed source-less island undercounting, but
the current `L_shed` still treats every connected non-selected load bus as
fully served:

```text
alpha_cmd_eff_i = 0          if source-less islanded
alpha_cmd_eff_i = alpha_i    if selected/commandable and connected
alpha_cmd_eff_i = 1          if non-selected and connected
```

That remains useful as commanded/effective service accounting, but it can
understate service degradation implied by GridFM's full predicted post-topology
state. The planned update therefore keeps the current metric as `L_shed_cmd`
and adds GridFM-implied and hybrid service metrics:

```text
alpha_gridfm_raw_i     = Pd_raw_i / Pd_base_i
alpha_gridfm_i         = clip(alpha_gridfm_raw_i, 0, 1)

alpha_hybrid_eff_i = 0                if source-less islanded
alpha_hybrid_eff_i = alpha_i          if selected/commandable and connected
alpha_hybrid_eff_i = alpha_gridfm_i   if non-selected and connected

L_shed_hybrid =
  sum_i Pd_base_i * (1 - alpha_hybrid_eff_i) / sum_i Pd_base_i
```

The planned Stage G objective should support paired modes:

```text
load_shed_mode = cmd      -> objective uses L_shed_cmd
load_shed_mode = hybrid   -> objective uses L_shed_hybrid
```

All runs should report `L_shed_cmd`, `L_shed_gridfm`, `L_shed_hybrid`,
`Delta_L_hybrid_minus_cmd`, and `Delta_L_gridfm_minus_cmd` regardless of the
objective mode. `L_shed_hybrid` should be interpreted as GridFM-implied service
accounting, not verified physical delivered load.

The same update should split the current partial `PAC_total` into three
explicit groups:

```text
PAC_total =
  w_op    * PAC_operational
+ w_ac    * PAC_AC
+ w_model * PAC_model_consistency
```

Planned operational terms:

```text
voltage limits
thermal overloads using per-line MATPOWER rateA-normalized loading
evaluated generator bound violations over all generator buses
source-less island and topology consistency audits
```

Planned AC physics terms:

```text
active-power balance residual using evaluated/clamped injections and Ybus(z)
reactive-power balance residual using evaluated/clamped injections and Ybus(z)
```

Planned model-consistency terms:

```text
raw GridFM Pd vs commanded alpha at selected load buses
raw GridFM Pg/Qg vs commanded Pg/Qg at selected generator buses
raw generator bound violations over all generator buses
```

The current GridFM wildfire output is still `[Pd, Qd, Pg, Qg, Vm, Va]`; it
does not independently output branch flows or branch loadings. Branch-flow
consistency should therefore remain unavailable or zero-weight. Current branch
loading is reconstructed from predicted `Vm/Va` and MATPOWER metadata, so a
separate branch-flow consistency residual would be tautological or would
double-count thermal behavior.

Implementation should preserve the existing raw/evaluated state split:

```text
raw_state:
  direct GridFM output before clamping, used for model-reliability diagnostics

eval_state:
  selected Pg/Qg/Pd/Qd clamped to commanded values, non-controlled quantities
  left GridFM-predicted, used for objective and AC/operational evaluation
```

Existing tests already verify that selected controlled values are made visible
to GridFM input, raw predictions remain separately auditable, and objective
evaluation clamps selected controlled values. The latest focused check remains:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py -q
10 passed
```

After this GridFM update is implemented and smoke-tested, Stage H should be
regenerated under the revised evaluator to compare Stage D, Stage E k2, Stage E
unconstrained, TH, and AH under the same updated load-shedding and PAC
decomposition. The intended comparison should explicitly show operational,
model-alignment, and AC-physics violation groups and should answer whether the
hybrid load metric or expanded PAC decomposition materially changes objective
values, topology rankings, or heuristic conclusions. The DC MILP comparison is
deferred until this GridFM-side revision is complete and documented.

## Latest Implementation Update - Effective Load Shedding

As of July 8, 2026, Stage G revised continuous load-shedding accounting uses
effective service after source-less island detection. For each fixed topology,
the runner uses graph traversal to identify buses disconnected from all
source/generator/reference buses. Selected controllable load buses in those
components are still bound to `alpha_i = 0`, and non-selected islanded load
buses are now counted as effectively unserved:

```text
alpha_effective_i = 0        if bus i is source-less islanded
alpha_effective_i = alpha_i  if bus i is selected/commandable and connected
alpha_effective_i = 1        if bus i is non-selected and connected

L_shed = sum_i Pd_base_i * (1 - alpha_effective_i) / sum_i Pd_base_i
```

This removes the discrepancy where a non-commandable islanded load could retain
implicit alpha 1.0 and appear served in `L_shed`. The source-less island term
remains in `PAC_total`, but should now be interpreted as an additional
operational topology-validity / island-severity penalty, not as the mechanism
that corrects load-shed accounting.

Focused test status:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py -q
10 passed
```

Operational fallback:

If future runs fail while writing checkpoints, plots, or result folders under
OneDrive, run the experiment to a local non-OneDrive output directory first
such as `$env:TEMP` or another short local path. After the run completes and
passes validation, copy the entire completed `run_*` folder back into the
intended `experiments/test/wildfire_tests/results/...` subfolder. This is the
preferred fallback because it avoids transient OneDrive sync/materialization
issues without changing the experiment methodology or final result layout.

Next methodological step:

Use the corrected effective-load-shedding Stage G/Stage H framework as the
current GridFM baseline, then build a DC-approximation/MILP comparison using
the same wildfire topology-control formulation. The DC MILP direction should
serve as a fairer benchmark than the current designed-scenario TH/AH heuristic
comparison, especially under less skewed environmental-risk scenarios. The
planned DC baseline should enforce DC power-flow balance, component/topology
relationships, branch/thermal constraints, generator limits, and load-service
variables, then be evaluated through the same common post-solution metrics used
for GridFM: wildfire exposure, effective load shedding, physics/constraint
violations, and objective accounting. After the DC MILP run is implemented,
saved, and interpreted, update the external GridFM wildfire formulation PDF
with the completed methodology and results through this checkpoint.

## Latest Checkpoint - Revised Continuous Decision Quality Complete

The Stage G revised continuous implementation checkpoint is complete. The
canonical completed run is:

```text
results/leq/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_233908
```

This checkpoint closes the implementation-revision arc that began with Stage G
state/loading corrections and continued through scenario-baseline revision,
baseline-margin revision, and continuous-recourse masking. The current
formulation now separates topology decision variables, baseline state, GridFM
predictions, and objective-evaluation clamps:

```text
controlled Pg/Qg/Pd/Qd values are made visible to GridFM input;
controlled values are then clamped in objective evaluation;
non-controlled state remains GridFM-predicted;
wildfire exposure, load shedding, and physics infeasibility are scored from
the resulting post-topology/post-recourse state.
```

The completed reduced full experiment recorded all expected result tables,
progress/checkpoint artifacts, and regenerated plot families for `cross_rho`,
`per_rho`, and `summary`. Focused methodology tests passed after the masking
and objective-audit changes.

Current interpretation:

- The revised formulation usually avoids source-less island decisions in the
  tested scenarios.
- The selected topologies often align with intended wildfire-risk targets, but
  accuracy is scenario-dependent and does not yet have a theoretical guarantee.
- Continuous recourse substantially reduces commanded load shedding and
  physics infeasibility relative to the fixed-control margin revision.
- The evaluated best topology is not always identical to the margin-revision
  topology. With the margin revision fairly capped at the same 50-topology
  budget, `94 / 150` shared selected settings matched the revised continuous
  topology and `56 / 150` differed.
- The formulation should be described as a tentative but functioning small-grid
  decision-quality methodology: it selects line shutoffs to reduce wildfire
  exposure, applies post-topology continuous controls to recover service, and
  improves the physics-infeasibility score.

Important caveat:

The current evidence is empirical on the small MATPOWER-30/GridFM setting. It
does not yet establish reliability, optimality, or generalization. The Stage H
heuristic comparison is now the transparent heuristic checkpoint, but it may be
inflated by the designed scenario construction because the target-margin
environmental scores deliberately emphasize the intended lines. The next large
step is therefore a fairer DC-approximation/MILP comparison under less skewed
environmental-risk scenarios, followed by robustness checks before moving
toward larger grids.

Next experiment family:

```text
1. DC-approximation/MILP comparison:
   implement a DC-style optimization baseline using the current wildfire
   formulation structure, hard DC physics/topology constraints, and common
   post-solution evaluation metrics.

2. Robustness analysis:
   perturb wildfire probabilities, loading assumptions, scenario targets,
   rho/lambda settings, and topology budgets to see whether decisions are
   stable or brittle.

3. Formulation write-up:
   after the DC MILP results are run, saved, and interpreted, update the
   external GridFM wildfire formulation PDF with the effective-alpha correction,
   Stage H heuristic comparison, DC comparison, and current limitations.

4. Scale-up readiness:
   only after the heuristic checkpoint, DC comparison, and robustness
   comparisons, consider larger-grid tests and eventual real-world-scale
   extensions.

5. Larger-grid economic-impact extension:
   revisit source-less islanding as a possible economic-impact layer. The
   current island term is a uniform topology-validity / island-severity
   penalty, but larger grids could assign differentiated costs to islanded
   loads or areas, including critical facilities such as hospitals,
   emergency-service zones, dense load pockets, or other priority customers.
   Keep this separate from effective-load-shedding accounting, which already
   treats source-less islanded load as unserved.
```

## Active Status - Stage G Implementation Revision

Stage G is now the active correction period for implementation fidelity. It is
not a wildfire optimization methodology change. The Stage C/D/E/F objective
structure, lambda logic, and topology-comparison methodology are preserved
while the implementation is corrected to use physically meaningful state and
line-loading quantities.

Stage G G1-G5 corrections:

```text
G1. GridFM homogeneous outputs are [Pd, Qd, Pg, Qg, Vm, Va].
G2. Denormalized Va is in degrees and is converted to radians for phasors.
G3. Loading is apparent MVA flow / MATPOWER rateA, not synthetic current / 100.
G4. Static MATPOWER IEEE-30 branch metadata supplies reproducible rateA values.
G5. Wildfire risk is evaluated over canonical off-diagonal physical branches;
    self-loops are excluded and directed edge pairs share one physical loading.
```

The prior Stage F physics/topology results should be treated as provisional
until the Stage G corrected loading-ranking audit and follow-on reruns are
explicitly executed. Stage G adds an audit command that writes
`stage_g_loading_ranking_audit.csv` only when intentionally run; implementation
and tests should not generate research result artifacts by default.

The next Stage G subproblem after G1-G5 is continuous-control masking and exact
decision enforcement: GridFM should see the selected decision variables, and
objective evaluation should clamp those decision values while leaving
non-decision state to GridFM prediction. Later stages are expected to move
toward AH/TH heuristics and SC-OPS.

Stage G result generation now has a MATPOWER-30 decision-quality runner that
duplicates the Stage F physics-aware plots for both variants:

```text
results/leq/stage_g/matpower_30/with_physics_infeasibility/rho100
results/leq/stage_g/matpower_30/without_physics_infeasibility/rho0
```

These reruns still use GridFM-predicted state for initialization and topology
evaluation. The remaining high line loadings are treated as a disclosed
GridFM-state limitation, not replaced by manual loading values or a separate
solver. For scenario design, known GridFM-heavy-loading physical branches can
receive a temporary lower wildfire probability, currently defaulting to
`0.005`, unless that line is an explicit high-risk target for the scenario.
This keeps intended wildfire-risk targets emphasized while preventing the
loading-squared term from making the same model-state outliers dominate every
case.

Scenario target IDs are now canonicalized to physical branch IDs before p_env
construction and expected-hit scoring. The completed Stage G MATPOWER-30 runs
were repaired in place by recomputing the expected-vs-observed CSV and
per-scenario match plots from the saved chosen topologies; the optimization
itself was not rerun for that accounting-only repair.

Current Stage G interpretation issue: `rho_phys=100` appears large enough to
dominate topology selection. At `lambda_R=0`, the no-physics Stage G run
selects the empty topology as expected, while the `rho_phys=100` run still
de-energizes lines because the objective is driven by the physics residual.
This should be interpreted as a physics-penalty sensitivity question, not as a
wildfire-risk preference.

Near-term diagnostics to run before a methodology shift:

```text
1. Calibrate p_env suppression by rank for GridFM-heavy-loading outlier lines:
   in scenarios where an outlier is not an explicit target, choose p_env low
   enough that it falls below the intended wildfire-priority set.
2. Add a Stage G rho sensitivity, especially rho_phys=10, to test whether the
   physics residual can regularize infeasibility without overwhelming the
   lambda_R/lambda_L tradeoff.
```

Implementation note: this rho sensitivity is being added as a Stage G
physics-infeasibility case study. It keeps the fixed-control bi-level topology
setup: calibrated p_env and baseline loading-aware proxy coefficients are fixed
before Gurobi topology proposals, GridFM evaluates proposed topologies at fixed
controls, and rho only changes final scoring/selection through
`rho_phys * PAC_total`. There is no continuous optimization in this case study.
The p_env calibration is an explicit temporary diagnostic choice to reduce
confounding from GridFM-heavy-loading prediction artifacts in scenarios where
those outlier lines are not intended wildfire targets.

Completed Stage G physics-infeasibility sensitivity run:

```text
results/leq/stage_g/physics_infeasibility_sensitivity/run_20260630_020627/
```

The run evaluated one fixed-control topology metric pool and rescored it across
`rho_phys = [0, 10, 20, 50, 100]`. It produced 50,085 topology metric rows,
250,425 rho-rescored rows, 1,575 best-by-rho/scenario/lambda/stage rows, and
154 plots. All methodology fidelity checks passed.

High-level finding: `rho_phys=10` reduces average PAC substantially versus
`rho_phys=0`, but it also makes line 23 frequent again in most scenarios. This
suggests that even moderate physics weighting can dominate the intended
scenario target behavior because the PAC term is largely thermal-residual
driven under GridFM-predicted voltages.

Average line 23 selection frequency across stages and the full lambda sweep:

```text
S1: rho0 0.016, rho10 0.571, rho100 0.921
S2: rho0 0.810, rho10 1.000, rho100 1.000
S3: rho0 0.048, rho10 1.000, rho100 1.000
S4: rho0 0.206, rho10 1.000, rho100 1.000
S5: rho0 0.016, rho10 1.000, rho100 1.000
```

Average target overlap:

```text
S1: rho0 0.378, rho10 0.133, rho100 0.133
S2: rho0 0.800, rho10 1.000, rho100 1.000
S3: rho0 0.183, rho10 0.117, rho100 0.117
S4: rho0 0.800, rho10 0.400, rho100 0.267
S5: rho0 0.229, rho10 0.162, rho100 0.133
```

The next methodological step remains the continuous decision-variable
selection change: try to improve infeasibility and load shedding while
preserving wildfire-risk reduction, then assess how that interacts with these
physics weighting cases before moving to AH/TH heuristics and SC-OPS.

The same run was later rescored in place with a low-rho refinement using the
saved topology metric pool only; no new GridFM or Gurobi topology evaluations
were performed. The active rho grid is now:

```text
[0, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 50, 100]
```

Aggregate low-rho finding:

```text
rho    target_overlap  line23_freq  avg_PAC  avg_nonphysics_T  avg_cost_vs_rho0
0      0.537           0.248        2.010    0.233             0.000
0.1    0.503           0.276        1.824    0.239             0.004
0.25   0.516           0.533        1.591    0.279             0.040
0.5    0.466           0.584        1.484    0.316             0.065
0.75   0.413           0.657        1.418    0.357             0.134
1      0.384           0.721        1.368    0.399             0.152
2      0.376           0.835        1.296    0.499             0.262
5      0.367           0.895        1.273    0.561             0.302
10     0.362           0.914        1.270    0.584             0.380
20     0.332           0.949        1.266    0.643             0.433
50     0.330           0.984        1.262    0.724             0.475
100    0.330           0.984        1.262    0.734             0.545
```

Current interpretation: `rho_phys=0` remains best for pure decision-quality
behavior. Among positive values tested, `rho_phys=0.1` is the only candidate
that reduces PAC without a large line-23 resurgence. By `rho_phys=0.25`, line
23 frequency jumps strongly, so the current physics residual is already
steering topology selection.

## Active Update - Scenario-Baseline Revision

A new Stage G scenario-baseline revision has now been implemented and run:

```text
results/leq/stage_g/physics_infeasibility_sensitivity_scenario_baseline_revision/run_20260701_003629/
```

This run corrects the baseline/proxy side of the fixed-control physics
sensitivity methodology:

```text
baseline loading / R_base / Gurobi proxy:
  stored GridFM scenario Vm_base/Va_base + MATPOWER rateA

post-topology evaluation:
  GridFM inference at fixed baseline controls

p_env:
  scenario targets = 1.0
  non-targets = 0.05
  no rank-calibrated ultra-low suppression

continuous recourse:
  disabled
```

Grounded verification passed:

```text
scenario-baseline max loading:       0.877465015024
branches above 100% loading:         0
canonical line 23 baseline loading:  0.524397336396
methodology checks:                  19/19 passed
plot checks:                         259/259 present
```

Output sizes:

```text
topology_metric_pool rows:             50,085
rho_rescored_objectives rows:         601,020
best rows:                              3,780
expected-vs-selected rows:                900
p_env rows:                               200
rho summary rows:                          60
```

Aggregate finding:

```text
rho    target_overlap  line23_freq  avg_PAC  avg_R_norm  avg_L_shed
0      0.408           0.397        1.554    18.525      0.224
0.1    0.408           0.397        1.554    18.525      0.224
0.25   0.421           0.416        1.528    17.991      0.229
0.5    0.477           0.470        1.486    16.564      0.241
1      0.477           0.486        1.479    16.618      0.242
2      0.480           0.517        1.458    16.714      0.246
10     0.476           0.702        1.354    18.375      0.262
100    0.341           0.978        1.206    28.029      0.288
```

Interpretation: the scenario-baseline revision successfully prevents the
GridFM-inferred no-action baseline from contaminating topology proxy ranking
and `R_base`. However, line 23 remains frequent because GridFM post-topology
inference is still the true evaluator and the physics residual still rewards
topologies that reduce GridFM-predicted thermal infeasibility. Moderate rho
values around `0.5` to `2` improve average target overlap in this run, but at
the cost of higher line-23 frequency and more load shed. Large rho values are
again dominated by PAC behavior.

A scenario-design audit has been added to the same run:

```text
tables/scenario_design_audit.csv
tables/scenario_target_rank_audit.csv
```

Current interpretation from that audit:

```text
S1: scenario-design weakness; targets 18 and 32 are weak under p_env*loading^2.
S2: mixed/acceptable; target 23 ranks first and is frequently selected.
S3: scenario-design weakness; targets 18 and 22 are weaker than the strongest alternatives.
S4: mixed/acceptable; target 77 ranks first, but non-targets still compete through the full objective.
S5: scenario-design weakness; several intended targets are low-loading and weakly separable.
```

This suggests the lower hit rate in the scenario-baseline revision is partly a
scenario-design issue, not purely a methodology failure. Some intended target
lines do not remain high-priority after replacing the GridFM-inferred baseline
with physically reasonable stored-scenario loading.

## Active Update - Scenario-Baseline Margin Revision

The scenario-baseline revision has now been regenerated with the target-margin
scenario design:

```text
results/leq/stage_g/physics_infeasibility_sensitivity_scenario_baseline_margin_revision/run_20260701_013324/
```

This margin run keeps the corrected scenario-baseline loading implementation,
removes canonical line `32` from the target/expected sets because its baseline
loading is only about `0.0279`, and calibrates non-target `p_env` so each
non-target baseline proxy score is at most `0.25 * weakest remaining target
score`. Explicit targets remain at `p_env=1.0`.

Verification summary:

```text
topology_metric_pool rows:                 50,085
rho_rescored_objectives rows:             601,020
best_by_rho_scenario_lambda_stage rows:     3,780
expected_vs_selected_by_rho rows:             900
p_env_by_scenario rows:                       200
methodology fidelity checks:              19/19 passed
plot output checks:                      259/259 present
```

Calibrated non-target `p_env` ranges:

```text
S1 min 0.00765, median 0.05
S2 min 0.05,    median 0.05
S3 min 0.00765, median 0.05
S4 min 0.05,    median 0.05
S5 min 0.00289, median 0.04996
```

Aggregate margin sensitivity:

```text
rho    target_overlap  line23_freq  avg_PAC  avg_R_norm  avg_L_shed
0      0.424           0.419        1.553    17.273      0.231
0.1    0.424           0.435        1.540    17.261      0.233
0.25   0.438           0.460        1.509    16.704      0.239
0.5    0.496           0.492        1.487    15.105      0.244
0.75   0.496           0.492        1.481    15.113      0.244
1      0.496           0.502        1.477    15.126      0.244
2      0.491           0.521        1.464    15.181      0.247
5      0.491           0.552        1.449    15.644      0.247
10     0.491           0.603        1.431    16.308      0.251
20     0.447           0.692        1.405    17.971      0.259
50     0.418           0.768        1.335    22.603      0.273
100    0.350           0.848        1.295    25.410      0.280
```

The margin scenario-design audit now classifies `S1`, `S3`, and `S5` as
methodology/evaluation failures rather than scenario-design weaknesses. Their
targets are now top-ranked with a target-vs-non-target margin near 4x, but the
selected topologies still frequently include non-target lines such as `14`,
`23`, `72`, and `77`. This strengthens the current conclusion: after the
scenario-design margin correction, the remaining misses are tied to the
fixed-control GridFM evaluation/objective behavior rather than weak scenario
construction alone.

## Active Update - Revised Continuous Implementation

Stage G now has a revised continuous-recourse implementation path:

```text
experiments/test/wildfire_tests/stage_g_implementation_revision/run_stage_g_revised_continuous_implementation.py

results/leq/stage_g/physics_infeasibility_revised_continuous_implementation/
```

This runner extends the scenario-baseline margin revision with continuous
controls:

```text
u = [Delta_Pg, Delta_Qg, alpha]
Pg = Pg_base + Delta_Pg
Qg = Qg_base + Delta_Qg
Pd_served = alpha * Pd_base
Qd_served = alpha * Qd_base
```

Controlled selected-generator `Pg/Qg` and selected-load `Pd/Qd` are made
visible to GridFM by an intervention-aware effective mask, then clamped back
into the post-inference evaluation state. Load shedding is computed from
commanded/clamped `alpha`, not from GridFM-predicted demand. Wildfire risk is
computed from loading reconstructed from the combined controlled + predicted +
baseline state.

The true continuous wildfire-risk term intentionally excludes `impact_l`:

```text
R_raw = sum_l z_l * p_env_l * loading_l^2
```

Impact/consequence remains part of topology proxy construction and scenario
design, but not the GridFM-evaluated true wildfire risk.

First reduced continuous-run settings:

```text
scenarios: S1-S5
lambda_R: [0, 0.2, 0.5, 0.8, 1.0]
rho_phys: [0, 2]
Stage D: exhaustive k<=2
Stage E k2 budget: 50
Stage E unconstrained budget: 50
Delta_Qg bound: +/- 5 MVAr
expected continuous topology/rho optimizations: 18,850
```

Audit emphasis:

```text
masking_clamping_audit.csv
controlled_state_consistency.csv
load_shedding_provenance.csv
wildfire_risk_provenance.csv
```

Hard checks invalidate the run if controlled features are masked, clamped
values differ from commanded values, provenance sums do not match objective
components, or objective formulas drift.

Verification completed before the full reduced run:

```text
py_compile: passed
revised continuous unit tests: 5 passed
Stage G implementation + revised continuous tests: 18 passed
```

Partial Stage-D-only smoke validation completed:

```text
results/leq/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_023730/

scenario_ids: S1
lambda_R: [0, 0.5]
rho_phys: [0, 2]
stage_d_limit: 1
stage_e_budget: 0
call_budget: 3
continuous results: 4
methodology hard-check failures: 0
```

This validates continuous recourse, intervention-aware masking, post-inference
clamping, provenance tables, and plot/table writing for the Stage D path. The
Gurobi-backed Stage E smoke/full reduced run still needs to be launched outside
the sandbox or after escalation is available. A failed sandbox attempt left an
incomplete folder at `run_20260701_023702/`.

Latest rho=0 full reduced launch status:

```text
results/leq/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_034434/
```

This run used the intended `rho_phys=0` full reduced settings:

```text
scenarios: S1-S5
lambda_R: [0, 0.2, 0.5, 0.8, 1]
Stage D: exhaustive k<=2
Stage E k2 budget: 50
Stage E unconstrained budget: 50
```

It exceeded the 2.5-hour checkpoint, so `rho_phys=2` was not started. The
process later stopped before table-writing completed. The folder contains only
`inputs/selected_decision_buses.json`; there are no `tables/`, `plots/`, or
`inputs/metadata.json`. Treat `run_20260701_034434` as incomplete and not
analyzable.

## Current Finding - Physics Penalty And Continuous Corrective Control

The June 23-24, 2026 physics case study establishes two separate findings for
the current IEEE-30 `auto_env` experiment.

First, adding the physics-infeasibility penalty changes the preferred topology.
For example, the exhaustive Stage D topology changes across the traditional
lambda cases as follows:

```text
rho_phys = 0:
  lambda_R=0.8: [18,23]
  lambda_R=0.5: [14,23]
  lambda_R=0.2: [23,30]

rho_phys = 100:
  lambda_R=0.8: [23,27]
  lambda_R=0.5: [23,27]
  lambda_R=0.2: [23,27]
```

The constrained and unconstrained Stage E searches show the same broad
physics-sensitive topology shift, although the 100-proposal proxy search does
not always recover the exhaustive Stage D best topology.

Second, enabling the current inner continuous optimization does not change the
final selected topology relative to the corresponding fixed-control result in
any of the tested combinations:

```text
4 stages
x 3 traditional lambda cases
x 2 rho_phys settings
= 24 matched stage/lambda/physics comparisons
```

The final topology is identical with and without continuous optimization in all
24 comparisons. The continuous solve can change the retained objective and
alpha values, particularly when `rho_phys=100`, but topology selection remains
driven by the topology search and physics penalty rather than by the current
continuous-control design.

### Quantified Continuous-Optimization Improvement

Matched comparison of each fixed-control best solution against the
continuous-control best solution for the same stage, lambda, `rho_phys`, and
final topology gives:

```text
rho_phys = 0:
  mean objective decrease across 12 comparisons: 3.78e-6
  maximum objective decrease:                   1.35e-5
  maximum relative decrease:                    0.00594%
  topology changes:                             0 of 12

rho_phys = 100:
  mean objective decrease across 12 comparisons: 3.873
  maximum objective decrease:                    11.478
  maximum relative decrease:                     0.2879%
  topology changes:                              0 of 12
```

The largest `rho_phys=100` improvements occur because the continuous solve
slightly reduces a physics-penalty term that is multiplied by 100. For example:

```text
Stage E constrained/unconstrained, lambda_R=0.2, topology [23,38]:
  fixed J_true:      3987.392908
  continuous J_true: 3975.914452
  decrease:          11.478456 = 0.287869%

  fixed PAC_total:      39.870872
  continuous PAC_total: 39.756084
```

For the exhaustive Stage D reference with `rho_phys=100`, continuous recourse
decreases the objective by approximately `2.71-2.80`, or only
`0.083-0.086%`, across the three traditional lambda cases. With
`rho_phys=0`, improvements are numerically negligible.

These modest improvements required:

```text
rho_phys=0:
  199,855 GridFM calls
  32.53 minutes

rho_phys=100:
  227,418 GridFM calls
  38.40 minutes

combined:
  427,273 GridFM calls
  approximately 70.9 minutes
```

The present continuous methodology is therefore not justified as a default
component of every decision-quality scenario. Its computational cost is large,
it changes no final topology in the current study, and its objective
improvements are below `0.3%`.

The current continuous variables are selected generically:

```text
Delta_Pg: three largest baseline PV generators [1,10,12]
alpha:    five largest baseline-demand PQ buses [6,20,11,29,18]
```

The best solutions show essentially no generator redispatch:

```text
rho_phys=0:   max_abs_delta_pg = 0 for all reported best solutions
rho_phys=100: max_abs_delta_pg remains below 0.0034 MW
```

This indicates that the present top-three-PV/top-five-PQ selection is not yet a
convincing post-topology corrective-action methodology. It is an initial
reduced control parameterization, not evidence that continuous recourse is
unimportant.

Canonical evidence:

```text
results/leq/stage_e/physics_infeasibility_case_study/
  without_continuous_optimization/
    rho0_no_physics/run_20260623_220854/
    rho100_with_physics/run_20260623_221317/
  with_continuous_optimization/
    rho0_no_physics/run_20260624_000543/
    rho100_with_physics/run_20260624_003815/
```

Key comparison artifacts:

```text
without_continuous_optimization/.../traditional_lambda_summary.csv
with_continuous_optimization/.../best_by_stage_lambda.csv
with_continuous_optimization/.../final_topologies_by_lambda.csv
with_continuous_optimization/.../alpha_consistency_diagnostics.csv
with_continuous_optimization/.../plots/traditional_lambda_objective_comparison.png
```

Current interpretation:

```text
topology control:
  select line de-energization decisions to reduce wildfire exposure while
  accounting for load service and physics infeasibility

post-topology corrective control:
  after fixing the topology, use targeted generator redispatch and explicit
  load-shedding controls to reduce remaining physics infeasibility and load
  shedding without undoing the topology's wildfire-risk benefit
```

The immediate research task is therefore to redesign how generator and load
control buses are selected and how their corrective actions are evaluated.
The revised continuous layer should be judged by whether it improves a fixed
topology's feasibility and service outcome, not primarily by whether it changes
the topology decision.

### Physics-First Decision-Quality Hypothesis

The next decision-quality experiment should isolate the value of physics-aware
topology evaluation before reintroducing expensive continuous recourse.

Proposed immediate Stage F extension:

```text
topology generation:
  keep the current Stage D exhaustive and Stage E Gurobi-proxy candidate sets

continuous controls:
  disabled; evaluate at fixed u_base

final true evaluation:
  J_true =
      lambda_R * R_norm
      + lambda_L * L_shed
      + rho_phys * PAC_total

comparison:
  run matched rho_phys=0 and rho_phys=100 decision-quality scenarios
```

This directly tests the current hypothesis:

```text
physics infeasibility adds operational explainability and pushes selected
topologies toward behavior that better matches physical expectations
```

The physics penalty still does not enter the current Gurobi proxy master. It
enters the GridFM true evaluation and final ranking of evaluated topologies.
Stage D remains the exhaustive `k<=2` reference for determining whether the
physics-unaware 100-candidate Stage E proxy search misses desirable
physics-aware solutions.

Only after this physics-only decision-quality comparison should the targeted
post-topology corrective layer be added. That later experiment should test
whether better generator/load-control selection can:

```text
1. reduce PAC_total materially,
2. reduce required explicit load shedding,
3. preserve the topology's wildfire-risk reduction,
4. lower J_true enough to justify its computational cost,
5. and possibly change the preferred topology when corrective capability is
   evaluated consistently across candidates.
```

### Current Decision-Variable Injection Audit

The current homogeneous IEEE-30 wildfire path does not create or attach
artificial, virtual, or auxiliary buses for decision variables. It directly
modifies features on the existing selected buses:

```text
selected generator bus:
  Pg_input = Pg_base + Delta_Pg

selected load bus:
  Pd_input = alpha * Pd_base
  Qd_input = alpha * Qd_base
```

These edits occur before normalization and GridFM inference. Line
de-energization separately removes entries from `edge_index`, `G`, `B`, and
`rate_a`.

However, the current experiment loads processed tensors generated with:

```text
mask_type = rnd
mask_ratio = 0.5
mask_dim = 6
```

The saved random mask is then applied after the decision variables modify the
node features. Any controlled feature whose mask entry is true is replaced by
the model mask value before GridFM sees it. The actual GNN scenario used in the
June 24 continuous study has:

```text
selected generators:
  bus 1:  Pg masked
  bus 10: Pg visible
  bus 12: Pg visible

selected loads:
  bus 6:  Pd masked, Qd visible
  bus 20: Pd masked, Qd visible
  bus 11: Pd masked, Qd visible
  bus 29: Pd masked, Qd visible
  bus 18: Pd visible, Qd masked
```

Therefore every selected load has at least one alpha-controlled channel hidden
from GridFM, and one of the three selected generator redispatch channels is
hidden. This materially weakens the interpretation of the completed continuous
optimization study and plausibly contributes to its negligible generator
movement and commanded-versus-predicted load-service mismatch.

The wrapper also currently returns GridFM predictions for all six channels:

```text
Pd, Qd, Pg, Qg, Vm, Va
```

It does not restore manually controlled `Pd`, `Qd`, or `Pg` values after
inference. The revised corrective-control methodology should split quantities
into two categories:

```text
manually controlled/clamped:
  selected-load Pd and Qd from alpha
  selected-generator Pg from Delta_Pg
  any other future explicit operating decision

GridFM-predicted:
  only unresolved state quantities and uncontrolled channels
```

Minimum implementation requirement:

```text
1. Build an intervention-aware inference mask.
2. Never mask a feature that is explicitly set by a decision variable.
3. Preserve/clamp controlled values in the evaluated post-inference state.
4. Use GridFM predictions only for features not fixed by the decisions.
5. Recompute wildfire exposure and PAC residuals from this combined
   controlled-plus-predicted state.
```

Whether artificial control nodes would improve the representation remains an
open design option, but they are not part of the current code and should not be
assumed necessary. The first revised implementation should test explicit
feature clamping on the existing buses because it matches the physical meaning
of load and generator setpoints and requires less architectural change.

## Refactor Status - June 10, 2026

The active wildfire research harness now lives under:

```text
experiments/test/wildfire_tests/
```

The old `experiments/test/wildfire_initial_tests/` package remains as a
temporary compatibility shim while we decide whether to remove it. Legacy
methodology-test outputs are archived under
`wildfire_tests/methodology_testing_results/old/`.

Current folder roles:

```text
gridfm_support/          GridFM wrappers and IEEE-30 scenario/model helpers
shared/                  reusable config, paths, risk, objective, plotting
stage_a_first_pass/      fixed-topology first-pass and connected-corridor runs
stage_b_multigroup/      multi-group threshold and multistart runs
stage_c_psps_baseline/   deterministic PSPS baseline rescoring
stage_d_deenergization/  limited enumerated de-energization
analysis/                objective analysis, AC-OPF, parity comparison
results/                 new refactored outputs
methodology_testing_results/  archived old methodology-test outputs retained for reference
```

The canonical lambda convention for new runs is:

```text
risk_leaning:    lambda_R = 0.9, lambda_L = 0.1
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.1, lambda_L = 0.9
```

Legacy lambda definitions are still available in code for parity checks, but
canonical output generation now uses the standardized convention above.

## Immediate Next Steps

The immediate checkpoint is a targeted post-topology corrective-control
methodology. Preserve the selected wildfire-risk topology, then investigate
which generators and load buses should participate in corrective recourse.
Candidate selection should consider electrical and topology relevance rather
than only baseline `Pg` and `Pd` magnitude.

After defining and validating the revised continuous controls:

```text
1. Rerun the physics-infeasibility case study with the revised post-topology
   corrective-control methodology.
2. Rerun decision-quality scenario analysis using the revised recourse layer.
3. Compare the revised Stage E solutions with TH and AH heuristic methods.
4. Evaluate the resulting decisions under robust/security-constrained
   contingency methodology, including the planned SC-OPS-style analysis.
5. After the small-grid decision-quality and robustness checks, expand to
   larger grids for computational-efficiency and economic-impact analysis.
```

## Stage E Gurobi Proxy Master - June 13, 2026

Stage E has been added as an additive experiment, not as a replacement for
Stage D enumeration:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/
```

Outputs are written under:

```text
results/leq/stage_e/gurobi_gridfm/
```

Stage E uses Gurobi as a proxy master that proposes binary line de-energization
topologies. GridFM remains the true evaluator. The first implementation
matches current Stage D fixed-control behavior and evaluates each topology at
`decision_vector.u_base`; it does not add inner SciPy continuous optimization.

The true wildfire exposure used in the Stage E objective is:

```text
R_true = sum_{l in C} z_l * p_env_l * loading_l^2
R_norm = R_true / R_base_true
```

where `C` is the selected high-risk/candidate line scope, matching Stage D's
grouped-risk scope. The historical `I_l` impact score is now interpreted as
`c_l`, a single-line load-service consequence score, and is used only in the
Gurobi proxy load term `L_hat(y) = sum_l c_l y_l`. It is not used in the true
GridFM wildfire exposure.

Stage E v1 lambda cases:

```text
risk_priority = (0.8, 0.2)
balanced      = (0.5, 0.5)
load_priority = (0.2, 0.8)
```

Master and true-evaluation lambdas are equal for v1 and are saved explicitly.
`R_norm` remains baseline-relative and is not bounded to `[0, 1]`; `L_shed`
remains demand-weighted and bounded to `[0, 1]`. Pareto-front computation is
deferred, but all evaluated candidate points are saved.

Stage E implementation verification:

```text
Compile check passed.
Stage C/D/E focused tests: 23 passed, 1 skipped, 3 external warnings.
Licensed Gurobi/GridFM smoke: ok.
```

An additive unconstrained Stage E study also exists:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_stage_e_gurobi_gridfm_unconstrained.py
results/leq/stage_e/unconstrained/
```

This removes the `sum_l y_l <= K` constraint and evaluates 100 unique
proxy-ranked topologies per case/lambda using no-good cuts. It is not compared
against Stage D because there is no fixed `K` enumeration counterpart. Each run
writes `figures/unconstrained_objective_trace.png`, whose title includes the
lambda objective and final best topology tradeoff.

Latest unconstrained 100-topology results:

```text
auto_env risk_priority: [18, 23, 27, 32], R_norm=0.060020, L_shed=0.448741, J=0.137764
auto_env balanced:      [18, 23, 27],     R_norm=0.063963, L_shed=0.466447, J=0.265205
auto_env load_priority: [],               R_norm=1.000000, L_shed=0.000000, J=0.200000
lgh risk_priority:      [18, 23, 27, 32], R_norm=0.056230, L_shed=0.448741, J=0.134732
lgh balanced:           [18, 23, 27],     R_norm=0.060074, L_shed=0.466447, J=0.263260
lgh load_priority:      [],               R_norm=1.000000, L_shed=0.000000, J=0.200000
```

The unconstrained frontier sweep is stored under:

```text
results/leq/stage_e/unconstrained_frontier/
```

It sweeps `lambda_R = 0.00, 0.05, ..., 1.00`, sets
`lambda_L = 1 - lambda_R`, evaluates 100 unconstrained Stage E topologies per
lambda/case, and constructs the nondominated frontier in
`(true_L_shed, true_R_norm)` space. The run completed with 42/42 successful
lambda-case runs, 4,200 total evaluated candidate rows, 1,009 unique
topologies, and 59 nondominated frontier points. The requested scatterplot is:

```text
results/leq/stage_e/unconstrained_frontier/figures/unconstrained_pareto_frontier_scatter.png
```

Latest Stage E smoke output:

```text
results/leq/stage_e/gurobi_gridfm/t0p30/k1/gps/auto/bal/run_20260613_191953/
```

Latest real Stage E experiment:

```text
results/leq/stage_e/gurobi_gridfm/stage_e_gurobi_gridfm_summary.csv
results/leq/stage_e/gurobi_gridfm/experiment_summaries/
```

Configuration:

```text
model: gps
cases: auto_env, largest_group_high
lambda cases: risk_priority, balanced, load_priority
K: 1, 2
proxy_type: env_loading_base
evaluation_budget: 20 per lambda case
```

The revised Stage D comparison helper recomputed enumeration with the Stage E
exposure-only objective rather than comparing against old impact-weighted Stage
D scores. Stage E matched revised Stage D's best topology in all 12 matched
settings. For K=2, Stage E used 20 candidate evaluations versus 562 revised
Stage D enumeration candidates. Proxy-best equaled true-best in all K=1 runs
and in 2 of 6 K=2 runs, so proxy ranking is imperfect but the no-good loop
still found the revised enumeration best within the 20-evaluation budget.

Verification completed during the refactor:

```powershell
python -m compileall experiments/test/wildfire_tests tests/test_wildfire_refactor_structure.py
$files = Get-ChildItem tests -Filter 'test_wildfire*.py' | ForEach-Object { $_.FullName }; pytest $files -q
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --help
python -m experiments.test.wildfire_tests.stage_d_deenergization.run_stage_d_deenergization --help
```

Results:

```text
42 wildfire tests passed.
Compile check passed.
New module entry points resolved.
```

Full legacy parity runs and full canonical standardized-weight regeneration
have not yet been executed. Old results therefore remain the comparison source,
not deleted reference outputs.

## Current Optimization Problem

The active experiment path is `experiments/test/wildfire_tests`.

The first-pass optimizer solves a reduced, fixed-topology predict-then-optimize
problem over selected generator redispatch and selected load-service fractions:

```text
minimize_u  J(u)

u = [Delta_Pg_selected, alpha_selected]
```

The current scalar objective is:

```text
J(u) = lambda_R * R_norm(u) + lambda_L * L_norm(u)
```

with:

```text
R_norm(u) = R_group(u) / R_group(u_base)
w_n       = P_D,n / sum_m P_D,m
L_shed(u) = sum_n w_n * (1 - alpha_n)
L_norm(u) = L_shed(u)
```

So the implemented objective is:

```text
J(u) = lambda_R * (R_group / R_baseline)
     + lambda_L * sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)
```

`R_baseline` is grouped wildfire risk at
`u_base = [0, ..., 0, 1, ..., 1]`. The demand-weighted load-shedding term is
already normalized because the weights sum to 1 across total positive baseline
demand, so `load_shedding_normalizer = 1.0` in current generated runs.

The active canonical tradeoff settings all use convex weights:

```text
risk_leaning:    lambda_R = 0.9, lambda_L = 0.1
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.1, lambda_L = 0.9
```

Generator movement is recorded as a diagnostic:

```text
sum_i (Delta_Pg_i / 5 MW)^2
```

but its objective weight is `0.0`, so it does not affect `J(u)`.

## Decision Variables And Constraints

The decision vector is:

```text
u = [Delta_Pg_1, ..., Delta_Pg_m, alpha_1, ..., alpha_q]
```

Current auto-selected control dimensions:

```text
m = 3 selected generator buses
q = 5 selected load buses
```

Latest GPS run selected:

```text
selected_generator_buses = [1, 10, 12]
selected_load_buses = [6, 20, 11, 29, 18]
```

The explicit optimizer bounds are:

```text
-5 MW <= Delta_Pg_i <= +5 MW
0.0   <= alpha_j    <= 1.0
```

Fixed assumptions:

```text
y_n = 1 for all buses
z_l = 1 for all lines
Qg fixed at baseline
no bus shutoff
no line shutoff as an optimizer decision
no mixed-integer variables
no hard AC power-flow constraints inside the optimizer
```

Only selected load buses can shed. Non-selected buses always have
`alpha_n = 1`. `Pd` and `Qd` are scaled by the same selected `alpha_j`.

## Wildfire Risk Term

The risk term follows the probability-times-consequence structure:

```text
R_group(u) = sum_k group_weight_k * sum_{l in G_k}
             z_l * p_env_l * loading_l(u)^2 * I_l(u)
```

Line energization is not optimized yet, so `z_l = 1` for all lines. The
relative-flow term `loading_l` is reconstructed from predicted `Vm` and `Va`
and normalized by the line rating/proxy rating before squaring.

The consequence term `I_l(u)` is computed by a counterfactual GridFM line
outage:

```text
I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
```

`S` is an equal-weight bus service estimate inferred from predicted `Pd` as
clipped `Pd_pred / Pd_base`, with zero-load buses counted as fully served.

## Current Simplifications

- No de-energization decision variables: line removal only happens inside the
  counterfactual `I_l(u)` calculation, not as an optimizer action.
- Synthetic environmental ignition probability: `p_env_l = 1.0` on selected
  corridor lines and `0.1` elsewhere.
- Single wildfire group: `high_risk_corridor`, `G_1 = [23, 32, 26]`.
- Reduced decision vector: only selected `Delta_Pg` and selected `alpha` are
  optimized.
- Load shedding is demand-weighted by baseline MW demand in the scalar
  objective. The old equal-bus shedding sum and unserved MW are retained as
  diagnostics in objective components/traces.
- Consequence score is still equal-bus service loss, not MW-weighted
  consequence.
- Counterfactual outage is surrogate-based through GridFM, not a validated AC
  power-flow or contingency solver.
- No hard AC feasibility constraints are enforced in the optimizer.
- Static single scenario: one IEEE-30 operating scenario.
- Risk normalization is baseline-relative.

## Latest Demand-Weighted Runs

On May 29, 2026, the optimized load-shedding cost was changed from the
equal-bus sum divided by `N_bus` to the demand-weighted fraction:

```text
L_shed_weighted = sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)
```

This keeps the load term normalized on a 0-to-1 scale and ensures that a 50%
curtailment at a 100 MW bus costs more than a 50% curtailment at a 5 MW bus.
The current generated `objective_normalizers.json` files therefore record:

```text
load_shedding_normalizer = 1.0
load_shedding_metric = demand_weighted_fraction
```

Baseline-started connected-corridor runs:

```text
results/demand_weighted/connected_corridor/risk/gnn/risk_gnn_20260529_134532/
results/demand_weighted/connected_corridor/risk/gps/risk_gps_20260529_134541/
results/demand_weighted/connected_corridor/balanced/gnn/balanced_gnn_20260529_134543/
results/demand_weighted/connected_corridor/balanced/gps/balanced_gps_20260529_134550/
results/demand_weighted/connected_corridor/shed/gnn/shed_gnn_20260529_134552/
results/demand_weighted/connected_corridor/shed/gps/shed_gps_20260529_134553/
```

Baseline-started summary:

```text
risk/gnn:
  objective:      0.999001 -> 0.9989932074556757
  group risk:     9.276730045998226 -> 9.276657378702414
  mean alpha:     0.9999863345024738
  min alpha:      0.9995908031440097

risk/gps:
  objective:      0.999001 -> 0.999001
  group risk:     2.6937669151682497 -> 2.6937669151682497
  mean alpha:     1.0

balanced/gnn:
  objective:      0.5 -> 0.5
  group risk:     9.276730045998226 -> 9.276730045998226
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:      0.5 -> 0.5
  group risk:     2.6937669151682497 -> 2.6937669151682497

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730045998226 -> 9.276730045998226

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.6937669151682497 -> 2.6937669151682497
```

Grid-seeded multistart runs:

```text
results/demand_weighted/multistart/risk/gnn/multistart_gnn_20260529_134608/
results/demand_weighted/multistart/risk/gps/multistart_gps_20260529_134657/
results/demand_weighted/multistart/balanced/gnn/multistart_gnn_20260529_134755/
results/demand_weighted/multistart/balanced/gps/multistart_gps_20260529_134825/
results/demand_weighted/multistart/shed/gnn/multistart_gnn_20260529_134854/
results/demand_weighted/multistart/shed/gps/multistart_gps_20260529_134901/
```

Multistart summary:

```text
risk/gnn:
  objective:                  0.999001 -> 0.9953593075274626
  group risk:                 9.24251512077115
  demand-weighted shedding:   0.042918644954574224
  equal-bus shedding:         1.0817346814920368
  unserved demand:            6.847095500765371 MW

risk/gps:
  objective:                  0.999001 -> 0.8961469098945768
  group risk:                 2.4163346027562054
  demand-weighted shedding:   0.0335228777710078
  equal-bus shedding:         1.0000237157820664
  unserved demand:            5.348126572996836 MW

balanced/gnn:
  objective:                  0.5 -> 0.5
  group risk:                 9.276730045998226
  demand-weighted shedding:   0.0
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:                  0.5 -> 0.4652724237474569
  group risk:                 2.4163717542193073
  demand-weighted shedding:   0.03352152279186194
  equal-bus shedding:         1.0
  unserved demand:            5.347910404205322 MW

shed/gnn:
  objective:                  0.000999 -> 0.000999
  group risk:                 9.276730045998226
  demand-weighted shedding:   0.0

shed/gps:
  objective:                  0.000999 -> 0.000999
  group risk:                 2.6937669151682497
  demand-weighted shedding:   0.0
```

Machine-readable summaries:

```text
results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.csv
results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.json
results/demand_weighted/multistart/multistart_tradeoff_summary.csv
results/demand_weighted/multistart/multistart_tradeoff_summary.json
```

## Automatic Figures

Every child run writes:

```text
figures/optimization_behavior.png
figures/ieee30_network_changes.png
visualization_summary.json
```

`optimization_behavior.png` shows objective, grouped wildfire risk, the
demand-weighted load-shedding metric, and configured objective weights.
`ieee30_network_changes.png` shows the topology-connected corridor and
selected control buses. The network layout is topology-accurate, not
geographic.

## Stage B.1/B.2 Automatic Multi-Group Sensitivity

Stage B.1/B.2 has been added without replacing the manual connected-corridor
workflow. The new selection mode is:

```yaml
wildfire:
  selection_method: automatic_risk_components
```

Automatic ranking uses a uniform synthetic environmental probability for every
candidate line before selection. Current automatic configs set
`risk_score.candidate_p_env = 1.0`, avoiding the circular manual-corridor
convention where selected lines had hazard `1.0` and non-selected lines had
hazard `0.1`.

The automatic baseline score is:

```text
score_l = p_env * loading_l(base)^2 * I_l(base)
```

Lines are selected by top fraction using `ceil(num_lines * top_fraction)`, so
outputs include both `requested_top_fraction` and
`realized_selected_fraction = num_selected_lines / num_lines`. Selected lines
are grouped by connected components and written as `G_1`, `G_2`, ...

The Stage B threshold runner is:

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

It defaults to grid-seeded multistart for `gps` and `gnn` over `risk`,
`balanced`, and `shed`, using near-0/near-1 lambda cases:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

Outputs are under:

```text
results/multi_group/threshold_0p10/
results/multi_group/threshold_0p125/
results/multi_group/threshold_0p15/
results/multi_group/threshold_0p175/
results/multi_group/threshold_0p20/
```

Each automatic run writes `automatic_line_risk_scores.csv`,
`automatic_wildfire_groups.json`, `automatic_group_summary.csv`, the standard
multistart artifacts, and `figures/ieee30_network_changes.png` with automatic
group colors/legend entries. Aggregate outputs are:

```text
results/multi_group/multi_group_threshold_sensitivity_summary.csv
results/multi_group/multi_group_threshold_sensitivity_summary.json
```

Verified smoke run:

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 --models gps --tradeoff-cases risk --num-seed-points 3 --max-seeds 2
```

Latest smoke output:

```text
results/multi_group/threshold_0p10/risk/gps/multistart_gps_20260531_191421/
```

The smoke selected 11 of 110 directed scenario edges
(`realized_selected_fraction = 0.10`), produced 5 automatic groups, did not
collapse to a single group, and wrote both automatic audit artifacts and both
figures. The full 30-run default sweep has not yet been run in full because it
is materially heavier than the smoke check.

Stage B.3/B.4 deferred: distinct seeded grouping and manual multi-region stress
testing are intentionally not implemented yet. They should only be considered
after manual review of Stage B.1/B.2 results under `results/multi_group/`.

## Stage C PSPS Threshold Baseline

Stage C has been added as a deterministic PSPS-only comparator. It does not
optimize `z_l`, `y_n`, or continuous controls after PSPS. Stage D optimized
de-energization remains deferred.

Stage C uses two thresholds:

```text
grouping_top_fraction = 0.30
psps_top_fraction = 0.10
```

The 30% threshold builds the automatic multi-group scenario. Only those
candidate lines are eligible for PSPS de-energization. The 10% PSPS threshold
then de-energizes:

```text
num_psps_lines = max(1, ceil(psps_top_fraction * num_candidate_lines))
```

Stage C replaces the dynamic first-pass consequence `I_l(u)` with a fixed
baseline demand-weighted consequence:

```text
I_l = max(0, S_D_base - S_D_outage_l) / S_D_base
S_D_base = sum_n P_D,n
```

For the post-PSPS diagnostic, demand-weighted load shed is inferred from the
PSPS GridFM prediction:

```text
service_fraction_n = clipped(Pd_pred,n / Pd_base,n, 0, 1)
L_shed = sum_n P_D,n * (1 - service_fraction_n) / sum_n P_D,n
```

Each environmental case assigns `p_env_l` before PSPS ranking, so different
cases can de-energize different lines:

```text
baseline_psps_risk_l = p_env_l * loading_l(base)^2 * I_l
```

Implemented cases:

```text
auto_env
largest_group_high
```

Stage C uses the near-boundary risk-emphasized weights, matching the
optimization-behavior convention used elsewhere:

```text
lambda_R = 0.999001
lambda_L = 0.000999
```

Runner:

```powershell
python experiments/test/wildfire_tests/run_stage_c_psps_baseline.py --grouping-top-fraction 0.30 --psps-top-fraction 0.10 --models gps --cases auto_env largest_group_high
```

The planned long folder names were shortened internally to avoid Windows
OneDrive path-length failures. Outputs are under:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

Aggregate outputs:

```text
results/stage_c_psps/stage_c_psps_summary.csv
results/stage_c_psps/stage_c_psps_summary.json
```

GPS results:

```text
auto_env:
  candidate lines:              33
  PSPS lines:                   4
  realized PSPS fraction:       0.12121212121212122
  de-energized line IDs:        [23, 18, 27, 101]
  baseline all-energized risk:  48.347018605732885
  post-PSPS risk:               2.2689103960556243
  risk reduction fraction:      0.9530703141271553
  demand-weighted load shed:    0.5495700231022472
  objective:                    0.999001 -> 0.047431823569736915

largest_group_high:
  candidate lines:              33
  PSPS lines:                   4
  realized PSPS fraction:       0.12121212121212122
  largest/manual group:         G_1
  de-energized line IDs:        [23, 18, 27, 101]
  baseline all-energized risk:  48.105301041506934
  post-PSPS risk:               2.172713688167144
  risk reduction fraction:      0.9548342149175525
  demand-weighted load shed:    0.5495700231022472
  objective:                    0.999001 -> 0.045669684916229365
```

Each run writes fixed consequence scores, PSPS ranking/decision CSVs, a
two-row `objective_trace.csv`, automatic group artifacts, standard summaries,
and figures. `ieee30_network_changes.png` now overlays PSPS de-energized lines
when the Stage C artifacts are present.

## Stage D Limited Enumerated De-Energization

Stage D v1 adds an interpretable enumerated `z_l` study. It is not a
mixed-integer optimizer and does not optimize continuous controls after
choosing topology. For each model/environmental case it builds the same
automatic `grouping_top_fraction = 0.30` candidate set, computes:

```text
num_candidate_lines = len(candidate_line_ids)
num_evaluated_subsets = 1 + num_candidate_lines + C(num_candidate_lines, 2)
```

and evaluates all candidate de-energization subsets with 0, 1, or 2 lines.
The empty subset is included as the all-energized baseline candidate.

Stage D uses the Stage C fixed demand-weighted consequence score:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l
R_norm = R_group / R_group_baseline
L_norm = demand-weighted load shed from the post-topology GridFM prediction
J = lambda_R * R_norm + lambda_L * L_norm
```

`R_group_baseline` is the all-energized baseline under the same model,
environmental case, grouping threshold, fixed `I_l`, and candidate groups. It
is computed once per model/environmental case and reused across lambda cases.
The topology evaluations are also shared across lambda cases; only the scalar
objective and chosen best subset change with lambda.

Lambda cases:

```text
risk_leaning:    lambda_R = 0.8, lambda_L = 0.2
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.2, lambda_L = 0.8
```

Runner:

```powershell
python experiments/test/wildfire_tests/run_stage_d_deenergization.py --grouping-top-fraction 0.30 --models gps --cases auto_env largest_group_high --evaluation-mode limited_enumerated_z_only --max-deenergized-lines 2
```

Outputs are under:

```text
results/stage_d_deenergization/t0p30/gps/auto/<risk|bal|svc>/run_20260531_2121xx/
results/stage_d_deenergization/t0p30/gps/lgh/<risk|bal|svc>/run_20260531_2121xx/
results/stage_d_deenergization/stage_d_deenergization_summary.csv
results/stage_d_deenergization/stage_d_deenergization_summary.json
```

GPS result highlights:

```text
candidate lines:         33
evaluated subsets:       562 per environmental case

auto_env:
  risk_leaning best:      [23, 27], objective 0.1740339143005827
  balanced best:          [23, 27], objective 0.27244685090395315
  service_leaning best:   [],       objective 0.2

largest_group_high:
  risk_leaning best:      [23, 27], objective 0.1714800049354558
  balanced best:          [23, 27], objective 0.27085065755074883
  service_leaning best:   [],       objective 0.2
```

When Stage C outputs are available, Stage D loads the fixed Stage C PSPS
topology for the matching model/environmental case and re-scores its scalar
objective under each Stage D lambda for apples-to-apples comparison. The Stage
C topology itself remains independent of Stage D lambda.

Each Stage D run writes `candidate_line_risk_scores.csv`,
`deenergization_candidate_evaluations.csv`,
`optimized_deenergization_decisions.csv`, automatic group artifacts,
`objective_trace.csv`, `optimization_summary.json`, `visualization_summary.json`,
and figures. `ieee30_network_changes.png` now labels Stage D optimized
de-energized lines separately from PSPS de-energized lines.

Deferred: full mixed-integer topology control, relaxed continuous `z_l`,
optimized `y_n`, AC feasibility enforcement, and continuous-control
optimization after selecting `z_l` remain out of scope.

## Separate AC-OPF Hard-Constraint Experiment

`ac_opf_experiment.py` is an isolated exploratory path for testing hard AC-OPF
constraints separately from the main GridFM connected-corridor workflow. The
latest run on May 28, 2026 did not satisfy hard AC constraints, so those
outputs should be treated as evidence that the local `ScenarioData`
approximation is not yet a reliable hard AC-OPF model.

## Tests And Commands Run

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py tests/test_wildfire_first_pass_multistart.py -q
python experiments/test/wildfire_tests/run_connected_corridor_tradeoffs.py
python experiments/test/wildfire_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

Latest focused test result:

```text
16 passed, 3 external deprecation warnings
```

## Limitations And Next Steps

This is not a full wildfire-resilience-aware OPF. The demand-weighted
load-shedding term fixes the previous equal-bus cost issue, but the first pass
still uses a GridFM surrogate consequence score and lacks hard AC feasibility
constraints. Stage D now evaluates limited enumerated line de-energization, but
does not yet introduce a full topology-control optimizer.

## Latest Checkpoint: Stage H Heuristic Comparison

As of July 3, 2026, the active Stage H checkpoint compares paper-inspired
heuristic topology rules against the completed Stage G revised continuous
formulation on the five decision-quality scenarios.

Completed top-k run:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
```

Methodology:

- TH now uses rank-based top-k shutoff over
  `score_l = p_env_l * baseline_loading_l^2`, with `k = 5, 4, 3, 2, 1`.
- AH remains the connected-network analogue: top 30 percent scored candidate
  lines, connected components by network topology, choose the component with the
  highest average score.
- Both heuristics use the same Stage G revised continuous recourse evaluator and
  objective accounting:

```text
J_no_phys = lambda_R * R_norm + lambda_L * L_shed
J_true    = J_no_phys + rho_phys * PAC_total
```

Validation:

```text
topology rows before rho expansion: 90
continuous evaluations:             180
TH audit rows:                       25
Stage G reference rows:              90
hard methodology failures:           0
focused Stage H tests:               6 passed
```

Comparison references included:

```text
stage_d_k2_exhaustive
stage_e_k2
stage_e_unconstrained
```

Key interpretation:

- Stage H is a sparse heuristic-policy comparison, not a dense Pareto-frontier
  topology search. This is why the Stage H Pareto scatter is less frontier-like
  than the Stage G continuous plots.
- Stage E unconstrained is the most consistently strong formulation baseline.
- Stage D exhaustive remains strong as a `k<=2` reference, especially in S1 and
  S3.
- TH top 4 performs best among the heuristics and even ranks first in S2 and S4
  under both rho settings.
- AH is generally weaker, except S3 at `rho=0`, where its connected component
  aligns well with the scenario structure.

Next step:

Use the Stage H top-k comparison as the heuristic baseline checkpoint before
moving into robustness/OPS-style experiments. The heuristic results motivate the
Stage G formulation by showing that simple ranked-risk rules can be useful in
some structured cases but are less consistent across scenarios than the
formulation-based Stage E unconstrained decisions.

## Current Checkpoint: Revised Load/PAC Stage H

As of July 11, 2026, the active checkpoint is the revised Stage H run with
hybrid load shedding and decomposed PAC accounting:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

The old top-k Stage H run remains useful only as the prior baseline comparison:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
```

Current evaluator semantics:

```text
raw_prediction:
  GridFM output for [Pd, Qd, Pg, Qg, Vm, Va]

combined/eval state:
  raw prediction with selected controlled Pd/Qd/Pg/Qg clamped to commanded
  values

objective load term:
  L_shed = L_shed_hybrid
```

Saved load-shedding metrics:

```text
L_shed_cmd
L_shed_gridfm_raw
L_shed_gridfm_effective
L_shed_hybrid
L_shed
load_shed_mode = hybrid
```

The hybrid load rule is:

```text
source-less islanded bus:          alpha_hybrid = 0
selected connected load bus:       alpha_hybrid = alpha_cmd
non-selected connected load bus:   alpha_hybrid = alpha_gridfm_effective
```

Revised PAC groups:

```text
PAC_total =
    PAC_operational
  + PAC_AC
  + PAC_model_consistency
```

with default group weights all equal to 1. Branch-flow consistency remains
unavailable because GridFM has no independent branch-flow/loading channel. AC
P/Q balance diagnostics are available in the retained revised run.

Validation status:

```text
focused revised tests:              18 passed
topology rows before rho expansion: 90
continuous evaluations:             180
hard methodology failures:           0
```

Key result interpretation:

The prior Stage H baseline understated both load shedding and physics
infeasibility. For heuristic rows, the old baseline had approximately:

```text
rho=0: L_shed 0.0388, PAC_total 2.1621
rho=2: L_shed 0.0386, PAC_total 2.1621
```

The revised run shows:

```text
rho=0: L_shed 0.2790, PAC_total 169.5884
rho=2: L_shed 0.3663, PAC_total 152.3797
```

This means the previous commanded/effective load metric was too optimistic for
non-selected connected load buses, and the previous PAC was only a partial
physics score. The revised PAC is dominated by generator/model-consistency
terms, with AC P/Q balance present but not the largest contributor.

Current methodological caveat:

The revised TH/AH heuristic rows use the new hybrid load/PAC accounting. The
Stage D/E reference rows included by Stage H still come from the existing Stage
G reference mechanism and may retain older PAC/load accounting unless the Stage
G reference set is regenerated under the revised evaluator. A fully fair final
comparison before publication-quality claims should regenerate Stage G reference
rows with the revised evaluator.

Next work direction:

Use this revised evaluator as the GridFM basis for DC MILP construction. The DC
MILP comparison should preserve hybrid load accounting, source-less island
correction, common post-evaluation metrics, and revised PAC diagnostics. The
scenario redesign should make environmental probabilities less obvious so the
heuristic, GridFM, and DC MILP comparison is not biased by target-margin
construction.
