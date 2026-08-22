# DC MILP Formulation Handoff From Current GridFM Wildfire Work

## Purpose

This document is a comprehensive handoff for a separate implementation agent.
The current research checkpoint is complete through the revised Stage G/H
GridFM formulation with hybrid load-shedding, decomposed PAC accounting, and a
regenerated Stage H comparison. The next task is now to design and implement a
fair DC-approximation/MILP baseline that stems from the same wildfire-aware
topology-control formulation and compare:

```text
1. Updated GridFM formulation with topology search plus continuous recourse
2. New DC MILP formulation
3. Existing Stage D/Stage E formulation baselines and TH/AH heuristic evaluators
```

The current Stage I experiment should use the existing five decision-quality
scenarios exactly as before:

```text
S1, S2, S3, S4, S5
```

Do not switch to fairer `p_env` construction in this first Stage I
implementation. Fairer scenario construction remains a later extension after
the locked `K <= 2` comparison is implemented and understood.

The most recent implemented methodological correction established that the
objective should use hybrid effective load shedding rather than only the
commanded load-service variables. Source-less islanded load buses are counted
as unserved, including non-commandable buses that previously could retain
implicit alpha 1.0 and understate load shedding. Connected non-selected load
buses are now assigned GridFM-implied service through the raw predicted `Pd`
channel.

Implemented July 11, 2026 clarification:

```text
The source-less island correction was necessary but not sufficient for the
final GridFM baseline. The implemented evaluator now adds hybrid GridFM-implied
service accounting for non-selected connected loads, explicit raw-vs-evaluated
state diagnostics, and decomposed PAC with operational, AC-balance, and
model-consistency groups.
```

The DC MILP comparison should therefore be built downstream of the revised
GridFM evaluator described here, not from the earlier July 8 islanding-only
correction.

Revised Stage H result anchor:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

## Stage I Locked Methodology And Solver Clarifications

As of the latest Stage I planning pass, the current experiment is locked to a
budget-compatible `K <= 2` topology comparison. Every topology-producing method
in the main comparison must satisfy:

```text
|S_off| <= 2
```

or equivalently:

```text
sum_{l in C} y_l <= 2
```

The included main-comparison methods are:

```text
Stage E K2
Stage I-a K2 guided-search DC recourse
Stage I-b direct DC MIQP with sum_{l in C} y_l <= 2
TH/AH budget-compatible K <= 2 heuristic points
```

For this experiment, do not include Stage E unconstrained in the main
Pareto/frontier comparison. Do not include Stage I-b unconstrained or higher-K
variants in the main comparison. Stage E unconstrained and Stage I-b
higher-K/unconstrained are later "topology flexibility" extensions, not part of
the locked `K <= 2` experiment.

Do not include Stage D exhaustive in the main Stage I comparison. Stage D was
studied previously and may remain useful as historical context, but this first
Stage I build should emphasize guided-search methodologies and DC
approximations rather than exhaustive enumeration.

The DC comparison is split into two related tracks:

```text
Stage I-a:
  Its own K2 guided topology proposal/evaluation loop using Stage-E-style
  proxy/no-good/no-revisit mechanics, with fixed-topology DC recourse as the
  evaluator.

Stage I-b:
  Joint topology plus continuous DC optimization, solved as a DC MIQP when the
  squared wildfire-risk objective is retained.
```

Stage I-b is not an explicit topology enumerator. Gurobi searches the topology
space through branch-and-bound, returns an incumbent, and either certifies
optimality or reports the remaining optimality gap. The locked Stage I-b solver
settings are:

```text
MIPGap = 1e-4
TimeLimit = 600 seconds initially, adjustable after smoke tests
save incumbent if TimeLimit is reached
if MIPGap > 1e-4 at termination, label result time-limited / not certified
```

The following solver diagnostics are required for every Stage I-b row:

```text
solver_status
objective_value
best_bound
mip_gap
runtime_seconds
node_count
solution_count
time_limit_reached
optimality_certified =
  (mip_gap <= 1e-4 and status is optimal or gap-satisfied)
```

The `MIPGap = 1e-4` lock applies only to Stage I-b MIQP runs. It does not apply
to Stage I-a LP/QP recourse, because Stage I-a fixes topology and has no
integer variables in the inner solve.

Stage I uses shared data plumbing before either solve:

```text
L_phys = canonical physical branch universe
C      = candidate physical branches eligible for shutoff
D      = load buses with Pd_base_i > tau_load
G      = generator rows or generator-capable buses, depending on available data
Fmax_l = rateA_l
p_env_l
Pd_base_i
Pg_min/max
```

The primary cross-method risk denominator is the shared stored-scenario
baseline:

```text
R_baseline_shared =
  sum_{l in L_phys} p_env_l * (baseline_loading_l_stored)^2
```

This is the denominator for primary GridFM/DC/heuristic Pareto plots. A
DC-native baseline denominator from a no-shutoff DC base solve may be saved as
a diagnostic, but it is not the primary cross-method normalization.

The Stage I branch model must inspect the MATPOWER branch data for transformer
taps and phase shifts. If all taps and shifts are trivial:

```text
B_l = baseMVA / x_l
f_l = B_l * (theta_i - theta_j)
```

If any tap or phase shift is nontrivial, use the MATPOWER-consistent DC form:

```text
tau_l =
  1       if MATPOWER tap is 0
  tap_l   otherwise

phi_l = shift_l * pi / 180
B_l   = baseMVA / (x_l * tau_l)
f_l   = B_l * (theta_i - theta_j - phi_l)
```

The implementation must not silently ignore nonzero taps or phase shifts. Save
`num_nonunity_taps`, `num_nonzero_phase_shifts`, `dc_branch_model_used`, and
`branch_model_warning`.

Stage I-b must include the apples-to-apples budgeted variant:

```text
sum_{l in C} y_l <= 2
```

No `K <= 5`, unconstrained-over-`C`, or Stage E unconstrained comparison should
be included in the current main experiment. Those belong to a later extension.

AC projection remains required for selected finalists, not every candidate
topology. Projection solves should be cached by solution identity so duplicate
DC solutions across `rho=0` and `rho=2` are not solved twice.

For the locked `K <= 2` experiment, the required AC projection finalist set is:

```text
5 scenarios
x 5 lambda values
x 2 rho panels
x 3 method families
= 150 projection jobs before caching
```

The three required method families are:

```text
1. GridFM Stage E K2 finalist
2. Stage I-a DC K <= 2 finalist
3. Stage I-b DC K <= 2 finalist
```

The default required set is the three-family 150-job version. DC finalists are
rho-invariant, so caching should reduce the number of unique DC projection jobs
substantially.

GridFM AC projection is control-faithful:

```text
D_proj_GridFM = D_Pg_cmd + D_Qg_cmd + D_s_cmd
```

Non-selected buses should not be forced to match raw GridFM predictions in the
primary projection distance. DC AC projection is active-power-oriented:

```text
D_proj_DC = D_Pg_DC + D_s_DC + D_f_DC
```

Angle distance should be zero-weighted by default:

```text
w_theta = 0
```

Pareto/frontier plots should use:

```text
x = L_shed
y = R_norm
```

Connect only points from the same method family, scenario, budget, and lambda
sweep convention. TH/AH are sparse policy points and should be highlighted
without necessarily drawing frontier lines through them. Duplicate topologies
across Stage D/E/TH/AH/DC must remain visible using jitter, annotations, or
z-ordering so overlapping points are not hidden.

The projection distance means the minimum distance from the method's chosen
solution to the nearest AC-feasible operating point under the same topology. It
is not merely an equation residual.

For each scenario `s`, lambda `lambda_R`, and rho panel `rho`, choose the
GridFM Stage E K2 finalist as:

```text
x_GridFM*(s, lambda_R, rho) =
  argmin_x [
    lambda_R * R_norm(x)
  + (1 - lambda_R) * L_shed_hybrid(x)
  + rho * PAC_total(x)
  ]
```

For Stage I-a, run a separate Stage-E-style guided K2 topology proposal loop
with DC recourse as the evaluator. For each proposed fixed topology `z`, solve
DC recourse, evaluate DC risk/load, add a no-revisit/no-good cut, and continue
until the 100-topology budget is reached or no valid candidate remains. Then
choose:

```text
x_I-a*(s, lambda_R, rho) =
  argmin_{x in X_I-a,K2^100} [
    lambda_R * R_norm_DC(x_I-a(z))
  + (1 - lambda_R) * L_shed_DC(x_I-a(z))
  ]
```

This Stage I-a finalist is rho-invariant unless rho is used only for reporting,
so duplicate AC projection solves across `rho=0` and `rho=2` should be cached.

For Stage I-b, solve the budgeted MIQP:

```text
x_I-b*(s, lambda_R) =
  argmin_{z,y,theta,f,Pg,s} [
    lambda_R * R_norm_DC
  + (1 - lambda_R) * L_shed_DC
  ]

subject to sum_{l in C} y_l <= 2
```

This is also rho-invariant unless rho is only used for reporting, so duplicate
projection solves across rho panels should be cached.

GridFM projection is anchored to the selected commands used to condition
GridFM:

```text
D_proj_GridFM* =
  min_{x_AC in F_AC(z_GridFM*)}
    D_Pg_cmd + D_Qg_cmd + D_s_cmd
```

with:

```text
D_Pg_cmd =
  (1 / |G_cmd|)
  sum_{g in G_cmd}
    ((Pg_g_AC - Pg_g_cmd) / (Pg_g_max - Pg_g_min + epsilon))^2

D_Qg_cmd =
  (1 / |G_cmd|)
  sum_{g in G_cmd}
    ((Qg_g_AC - Qg_g_cmd) / (Qg_g_max - Qg_g_min + epsilon))^2

D_s_cmd =
  sum_{i in D_cmd} Pd_i_base * (s_i_AC - alpha_i_cmd)^2
  / (sum_{i in D_cmd} Pd_i_base + epsilon)
```

If `Qg` bounds are unavailable, set `D_Qg_cmd` unavailable or zero-weighted and
flag it. Non-selected buses are not forced to match raw GridFM predictions in
the primary projection objective.

DC projection is anchored to the full DC active-power/load/flow solution:

```text
D_proj_DC* =
  min_{x_AC in F_AC(z_DC*)}
    D_Pg_DC + D_s_DC + D_f_DC
```

with:

```text
D_Pg_DC =
  (1 / |G|)
  sum_{g in G}
    ((Pg_g_AC - Pg_g_DC) / (Pg_g_max - Pg_g_min + epsilon))^2

D_s_DC =
  sum_{i in D} Pd_i_base * (s_i_AC - s_i_DC)^2
  / (sum_{i in D} Pd_i_base + epsilon)

D_f_DC =
  (1 / |L_active|)
  sum_{l in L_active}
    ((P_l,from_AC - f_l_DC) / Fmax_l)^2
```

`P_l,from_AC` must use the same signed canonical branch orientation as
`f_l_DC`. Angle mismatch is disabled by default with `w_theta = 0`.

Finalized revised Stage H summary:

```text
topology_rows_before_rho: 90
continuous_evaluations: 180
hard_methodology_failures: 0
Stage D exhaustive reference rows: 30
Stage E k2 reference rows: 30
Stage E unconstrained reference rows: 30
TH best rows: 150
AH best rows: 30
```

Key revised-evaluator findings:

```text
avg(L_shed_hybrid - L_shed_cmd):           0.2390
max(L_shed_hybrid - L_shed_cmd):           0.4943
avg(L_shed_gridfm_effective - L_shed_cmd): 0.2777
max(L_shed_gridfm_effective - L_shed_cmd): 0.6136

avg PAC_operational:        57.9186
avg PAC_AC:                 10.8316
avg PAC_model_consistency:  92.2338
avg PAC_total:             160.9840
```

Interpretation: the earlier commanded-only connected-load accounting materially
understated GridFM-implied service degradation at non-selected connected load
buses. The revised PAC also shows a large model-consistency component, so the
GridFM comparison should be described as both an operational feasibility and
model-alignment assessment.

Latest metric clarification:

```text
target_overlap_fraction is now retained only as a backward-compatible alias for
target_recall.

target_recall =
  number of selected shutoff lines that are expected targets
  / number of expected target lines

target_precision =
  number of selected shutoff lines that are expected targets
  / total number of selected shutoff lines
```

This split is methodologically important for the DC MILP comparison because a
method that is allowed or inclined to de-energize more lines can look more
accurate under recall/overlap alone. Future comparison tables and plots should
report both recall and precision so broad shutoff policies do not receive
unearned credit for target agreement.

## Current Code And Result Anchors

Active research package:

```text
experiments/test/wildfire_tests/
```

Key current files:

```text
experiments/test/wildfire_tests/CURRENT_STATE_SUMMARY.md
experiments/test/wildfire_tests/HISTORY.md
experiments/test/wildfire_tests/README.md

experiments/test/wildfire_tests/stage_g_implementation_revision/
  run_stage_g_revised_continuous_implementation.py
  run_stage_g_scenario_baseline_physics_sensitivity.py
  run_stage_g_physics_sensitivity.py

experiments/test/wildfire_tests/stage_h_heuristic_comparison/
  run_stage_h_heuristic_baseline_comparison.py

experiments/test/wildfire_tests/stage_e_gurobi_implementation/
  gurobi_master.py
  physics_infeasibility_evaluator.py
  stage_e_gurobi.py

experiments/test/wildfire_tests/gridfm_support/
  branch_metadata.py

experiments/test/wildfire_tests/shared/
  wildfire_risk.py
```

Focused tests that ground the current checkpoint:

```text
tests/test_wildfire_stage_g_revised_continuous.py
tests/test_wildfire_stage_h_heuristic_comparison.py
tests/test_wildfire_stage_g_implementation.py
```

Current Stage G reference path used by Stage H:

```text
experiments/test/wildfire_tests/results/leq/stage_g/
  physics_infeasibility_revised_continuous_implementation/continuous_run/
```

Completed Stage H top-k heuristic comparison:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
```

Important operational note: Windows OneDrive sometimes allows nested result
directories to be listed but fails when individual deep CSV paths are opened.
For future heavy runs, write to a short local output root first, such as a temp
directory, validate the run, then copy the completed `run_*` folder back under
the intended result tree.

## Research Progress So Far

The staged wildfire harness has evolved as follows:

```text
Stage A: first-pass connected-corridor risk/load tradeoff.
Stage B: automatic multi-group selection and threshold sensitivity.
Stage C: deterministic PSPS baseline.
Stage D: limited enumerated de-energization, especially k <= 2.
Stage E: Gurobi proxy-master topology candidates with GridFM true evaluation.
Stage F: decision-quality scenario studies.
Stage G: corrected physical/state implementation and revised continuous recourse.
Stage H: TH/AH heuristic comparison against Stage G revised continuous.
```

The current baseline is the Stage G revised continuous formulation. It selects
line shutoffs, applies continuous post-topology recourse, evaluates the resulting
state through GridFM, clamps controlled values for objective evaluation, and
scores wildfire exposure, effective load shedding, and partial physics
infeasibility.

The current evidence is empirical on the small MATPOWER/IEEE-30 setting. It
does not establish global optimality, reliability, or generalization. The new
DC MILP is intended as a fairer optimization baseline than the current sparse
heuristic comparison.

## Grid And Line Semantics

The active study uses the IEEE-30/MATPOWER case structure as represented in the
GridFM test data. Physical branch metadata is attached in:

```text
experiments/test/wildfire_tests/gridfm_support/branch_metadata.py
```

Important semantics:

```text
1. MATPOWER branch ratings use rateA.
2. Directed GridFM scenario edges are mapped to physical branches.
3. Canonical physical line IDs are used for wildfire/topology decisions.
4. Self-loops are excluded from physical wildfire-risk decisions.
5. Directed edge pairs share one physical branch loading.
6. Removing a physical line expands to all directed line IDs in that branch.
```

Stage G corrected earlier implementation artifacts:

```text
G1. GridFM homogeneous outputs are [Pd, Qd, Pg, Qg, Vm, Va].
G2. Denormalized Va may be in degrees and is converted to radians for phasors.
G3. Loading is apparent MVA flow / MATPOWER rateA.
G4. Static MATPOWER IEEE-30 branch metadata supplies reproducible rateA values.
G5. Wildfire risk is evaluated over canonical off-diagonal physical branches.
```

The DC MILP should use the same physical branch universe and canonical line
semantics. It should not optimize over duplicate directed edges as independent
topology decisions.

## Current Decision Variables In The GridFM Formulation

The current GridFM formulation separates topology search from continuous
recourse.

Topology variables/concepts:

```text
y_l = 1 if candidate physical line l is de-energized
z_l = 1 - y_l if line l is energized
```

Continuous recourse variables in the GridFM objective evaluation:

```text
Delta Pg_i for selected generator buses
Delta Qg_i for selected generator buses in the revised Stage G runner
alpha_i for selected controllable load buses
```

Current selected continuous controls are not all buses. The decision vector is
small and scenario-specific. Non-selected controls remain at baseline unless
they are inferred by GridFM as non-controlled predicted state.

The Stage G revised continuous runner uses SciPy continuous optimization within
each fixed topology, with a GridFM call budget. The default Stage H comparison
metadata used:

```text
lambda_R in [0.8, 0.5, 0.2]
rho_phys in [0.0, 2.0]
call_budget = 100
```

Within each fixed topology, the runner keeps the best valid objective call,
even if SciPy terminates because the call budget is reached.

## Current GridFM Objective

For a topology and continuous-control vector `u`, the current Stage G revised
continuous objective is:

```text
J_true(z, u)
  = lambda_R * R_norm(z, u)
  + lambda_L * L_shed(z, u)
  + rho_phys * PAC_total(z, u)

lambda_L = 1 - lambda_R
```

The non-physics objective is:

```text
J_no_phys(z, u)
  = lambda_R * R_norm(z, u)
  + lambda_L * L_shed(z, u)
```

Where:

```text
R_norm    = normalized wildfire exposure
L_shed    = effective demand-weighted load shedding
PAC_total = partial soft physics/operational infeasibility score
```

The DC MILP should be designed from this same decomposition. It does not need
to use the exact same soft `PAC_total` internally if it enforces hard DC
constraints. However, its final solution should still be passed through a
common post-solution evaluator that reports wildfire exposure, effective load
shedding, DC/physics violations, source-less island behavior, and a comparable
objective accounting surface.

## Wildfire Exposure Term

Current true GridFM-evaluated wildfire exposure is:

```text
R_raw(z, u)
  = sum_{l in candidate physical lines}
      z_l * p_env_l * loading_l(z, u)^2
```

It intentionally excludes the older single-line consequence score `c_l` from
the true risk term. That consequence score remains part of the proxy master
candidate generation, not the true evaluation term.

Normalization:

```text
R_norm(z, u) = R_raw(z, u) / R_baseline
```

Where `R_baseline` is computed from the same scenario environmental
probabilities and baseline loading:

```text
R_baseline
  = sum_l p_env_l * baseline_loading_l^2
```

Current code anchors:

```text
shared/wildfire_risk.py:
  compute_operational_wildfire_exposure

stage_e_gurobi_implementation/stage_e_gurobi.py:
  normalize_true_exposure
```

DC MILP implication:

The DC formulation needs line flow variables and energized-state variables.
The most natural DC analogue is:

```text
R_raw_DC
  = sum_l z_l * p_env_l * (|f_l| / Fmax_l)^2
```

Because this is quadratic in flow, a pure MILP needs a linearization or
piecewise-linear approximation. Candidate options:

```text
1. MIQP/QP if allowed by the solver and acceptable for the comparison.
2. Piecewise-linear convex approximation of normalized absolute flow squared.
3. Linear proxy risk using p_env_l * |f_l| / Fmax_l.
4. Two-level comparison: MILP optimizes a linearized/proxy objective, then
   common evaluator computes exact squared exposure from the solved flow.
```

For a fair baseline, prefer a MILP-compatible piecewise-linear approximation
or clearly name the run as DC MIQP if quadratic objective is retained.

## Effective Load-Shedding Term

The latest implemented correction is central. The revised GridFM evaluator now
saves four load-shedding metrics:

```text
L_shed_cmd
L_shed_gridfm_raw
L_shed_gridfm_effective
L_shed_hybrid
```

Definitions:

```text
alpha_cmd_i =
  commanded alpha for selected load buses
  1.0 for non-selected load buses

alpha_gridfm_raw_i =
  Pd_raw_i / Pd_base_i, if Pd_base_i > 0
  1.0, if Pd_base_i <= 0

alpha_gridfm_effective_i = clip(alpha_gridfm_raw_i, 0, 1)

alpha_hybrid_i =
  0.0                       if source-less islanded
  alpha_cmd_i               if selected and connected
  alpha_gridfm_effective_i  if non-selected and connected

L_shed
  = L_shed_hybrid
  = sum_i Pd_base_i * (1 - alpha_hybrid_i)
    / sum_i Pd_base_i
```

This was added because the source-less islanding correction fixed only one
underrepresentation. A non-commandable islanded load could previously remain at
implicit alpha 1.0 and appear fully served; after that fix, a connected
non-selected load bus could still appear fully served even when GridFM's raw
state implied reduced served demand. The revised hybrid objective now counts
source-less load as unserved and uses GridFM-implied service for connected
non-selected load buses.

Current code anchor:

```text
stage_g_implementation_revision/run_stage_g_revised_continuous_implementation.py:
  _load_shedding_provenance
```

Focused tests verify:

```text
1. load_shed_weighted sums to the objective L_shed.
2. L_shed_cmd, L_shed_gridfm_raw, L_shed_gridfm_effective, and L_shed_hybrid
   each sum from bus-level provenance.
3. selected source-less islanded buses have alpha_effective = 0.
4. non-selected source-less islanded buses also have alpha_effective = 0.
5. non-selected connected buses use GridFM-effective alpha in hybrid mode.
6. selected connected buses use commanded alpha in hybrid mode.
```

DC MILP implication:

The DC model should use explicit load-service variables for every load bus, not
only the small GridFM selected-load subset. Suggested variable:

```text
s_i in [0, 1] = served fraction of load at bus i
```

Then:

```text
L_shed_DC
  = sum_i Pd_base_i * (1 - s_i)
    / sum_i Pd_base_i
```

If topology/component constraints are included, any load bus disconnected from
all source/generator/reference buses should be forced to:

```text
s_i = 0
```

At minimum, the post-solution evaluator must identify source-less components
and report effective load shedding with the same correction.

## Source-Less Islanding Operation

Current source-less island detection:

```text
1. Build an undirected graph over buses.
2. Remove all directed IDs corresponding to de-energized physical lines.
3. Source buses are buses with Pg_base > 1e-9, PV buses, and the reference bus.
4. Any connected component with no source bus is source-less.
5. All buses in source-less components are reported as source_less_island_buses.
```

Current code anchor:

```text
stage_e_gurobi_implementation/physics_infeasibility_evaluator.py:
  source_less_island_buses
```

Current Stage G recourse behavior:

```text
1. If a selected controllable load bus is source-less, its alpha bound is
   forced to [0, 0].
2. Non-selected source-less load buses are not continuous variables, but the
   corrected effective-load-shed accounting assigns alpha_effective = 0.
3. The island term remains in PAC_total as an additional topology-validity /
   island-severity penalty.
```

DC MILP implication:

The DC model should not allow physically impossible service in source-less
components. There are two possible levels:

```text
Strict formulation:
  Add connectivity/source-reachability constraints so served load must be
  connected to at least one source/generator/reference bus through energized
  lines.

Simpler formulation:
  Enforce DC nodal power balance with generator limits and load-service
  variables. This naturally makes source-less load service impossible if there
  is no generation in that component, provided the model has no artificial
  slack injection at load buses. Still run the same graph-based post-solution
  source-less island checker for audit.
```

The strict formulation is preferable if the implementation agent can add
component/reachability constraints cleanly. The simpler formulation may be
acceptable for the first baseline if the post-solution evaluator confirms that
served source-less load is zero.

## Current Physics-Infeasibility And Model-Consistency Term

The revised `PAC_total` is a decomposed soft score, not a complete hard AC OPF
feasibility certificate. Implemented group weights default to:

```text
pac_operational_weight        1.0
pac_ac_weight                 1.0
pac_model_consistency_weight  1.0
```

The implemented formula is:

```text
PAC_total =
    pac_operational_weight       * PAC_operational
  + pac_ac_weight                * PAC_AC
  + pac_model_consistency_weight * PAC_model_consistency
```

Current implemented Stage G/Stage H components:

```text
voltage_limits:
  mean_i [max(0, 0.95 - Vm_i)^2 + max(0, Vm_i - 1.05)^2]

thermal_limits:
  mean_l_active [max(0, loading_l - 1)^2]

generator_limits_eval:
  normalized squared Pg bound violation over generator-capable buses using the
  clamped/evaluated state.

generator_limits_raw:
  normalized squared Pg bound violation over generator-capable buses using raw
  GridFM predictions before clamping.

island_source_feasibility:
  sum_{i in source-less buses} Pd_base_i * alpha_i^2 / total_demand

p_balance, q_balance:
  normalized squared AC active/reactive nodal balance residuals from the
  outage-adjusted admittance data when available. The revised Stage H run had
  both available for all 180 evaluated rows.

branch_flow_consistency:
  unavailable in the current GridFM channel set because branch loading is
  reconstructed from Vm/Va, not independently predicted. It is reported with
  availability False and zero weight to avoid double-counting thermal behavior.

cmd_load, cmd_Pg, cmd_Qg:
  normalized raw GridFM prediction deviations from commanded selected-control
  values. These are model-consistency/reliability diagnostics, not operational
  dispatch variables.
```

Current code anchors:

```text
stage_g_implementation_revision/run_stage_g_revised_continuous_implementation.py:
  _physics_components
  _command_consistency_components
  _load_shedding_provenance
```

DC MILP implication:

The DC baseline should be stronger than this soft PAC term for the DC physics
it represents. It should enforce:

```text
1. DC nodal active-power balance.
2. Branch flow-angle equations when a line is energized.
3. Branch thermal limits.
4. Generator active-power limits.
5. Load-service bounds.
6. Topology/offline-line constraints.
7. A no-service rule or audited consequence for source-less islands.
```

The DC model does not represent reactive power, voltage magnitudes, or AC
branch-flow consistency. These should be named as limitations and, if useful,
reported as "not modeled by DC MILP" rather than silently compared to AC PAC
components.

## Current Topology Search And Continuous Evaluation

The current formulation is not a single monolithic optimization. It is:

```text
Outer topology generation/search:
  Stage D exhaustive k <= 2, or
  Stage E Gurobi proxy-master candidate generation, or
  Stage E unconstrained proxy candidates, or
  Stage H heuristic candidates.

Inner continuous recourse per fixed topology:
  SciPy minimizes J_true over selected Delta Pg, Delta Qg, and alpha variables.
  GridFM is called for each objective evaluation.
  Controlled variables are visible to GridFM input.
  Controlled variables are clamped in objective evaluation.
  Non-controlled state remains GridFM-predicted.
```

The current per-topology objective call keeps the best valid observed objective
within the call budget. This matters for comparing to DC MILP: the GridFM
baseline is a topology-search plus local continuous recourse procedure, not a
guaranteed global optimizer.

DC MILP implication:

The DC MILP should be formulated as a direct optimization over topology,
generation dispatch, branch flows, angles, and load service. This gives it a
different computational structure from GridFM, so the comparison must be clear:

```text
GridFM method:
  candidate topology search + continuous local recourse + GridFM state
  evaluation.

DC MILP:
  direct optimization under linearized DC physics and mixed-integer topology.

Heuristics:
  sparse policy topology proposals + same/common post-solution evaluation.
```

## Existing Stage E Proxy Master

The current Gurobi proxy master is not the desired DC MILP. It only proposes
topologies using a simple proxy objective:

```text
proxy_R = sum_l risk_coeff_l * (1 - y_l) / denominator
proxy_L = sum_l c_l * y_l
min lambda_R * proxy_R + lambda_L * proxy_L
```

Where, for the default proxy:

```text
risk_coeff_l = p_env_l * baseline_loading_l^2
```

It may impose:

```text
sum_l y_l <= K
```

It also uses no-good cuts to ask for unique candidate topologies.

Current code anchor:

```text
stage_e_gurobi_implementation/gurobi_master.py
```

The new DC MILP should not stop at this proxy. It should include network
physics and load-service decisions directly.

## Existing Heuristic Evaluators

Stage H compares two paper-inspired heuristic families against the current
Stage G revised continuous baseline:

```text
TH:
  Rank-based top-k shutoff over score_l = p_env_l * baseline_loading_l^2.
  Current top-k settings: k = 5, 4, 3, 2, 1.

AH:
  Select top 30 percent scored candidate lines.
  Group selected lines into connected network components.
  Shut off the component with the highest average score.
```

Current code anchor:

```text
stage_h_heuristic_comparison/run_stage_h_heuristic_baseline_comparison.py
```

Stage H is a sparse policy comparison, not a dense frontier search. The current
interpretation is:

```text
1. Stage E unconstrained is the strongest and most consistent formulation
   baseline in the current empirical runs.
2. Stage D exhaustive remains strong as a k <= 2 reference.
3. TH top 4 can win in structured scenarios such as S2 and S4.
4. AH is usually weaker except when its connected component aligns with the
   scenario structure.
```

The DC MILP comparison should include the same heuristic outputs or rerun the
heuristics under the modified fairer scenarios.

## Current Scenarios

The current five decision-quality scenarios are defined in:

```text
stage_f_decision_quality/scenario_definitions.py
```

They are:

```text
S1_low_impact_high_risk:
  target_high_risk_line_ids = (27, 32, 101, 36)

S2_high_consequence_non_bridge:
  target_high_risk_line_ids = (23,)

S3_redundant_g1_corridor:
  target_high_risk_line_ids = (18, 27, 16, 19, 22)
  suppressed_line_ids = (23,)

S4_source_less_island_trap:
  target_high_risk_line_ids = (77, 79)

S5_local_vs_distributed:
  target_high_risk_line_ids = (27, 32, 36, 101, 47, 51, 77, 79, 88, 91)
  suppressed_line_ids = (23,)
```

The current Stage G/H target-margin p_env construction is intentionally
designed:

```text
1. Targets receive p_env = 1.0.
2. Non-targets start around p_env = 0.05.
3. Non-target p_env may be capped so non-target baseline score is at most
   0.25 * weakest target baseline score.
4. Line 32 is excluded from the target-margin expected set because its
   baseline loading is extremely low in the corrected scenario baseline.
```

This helped diagnose decision quality but may make the scenarios too obvious.
The next fair benchmark should preserve the scenario concepts while reducing
the artificial separability of the environmental-probability design.

## Deferred Modified Scenarios For Later Fair DC Comparison

Do not implement modified/fair `p_env` scenarios in the current locked `K <= 2`
Stage I experiment. Use the existing five scenarios exactly as before for the
first implementation. The modified scenario family below is deferred for a
later fair-scenario extension.

For that later extension, the implementation agent may create a modified
scenario family that keeps the same conceptual cases but uses less obvious
`p_env` distributions. The goal would be to avoid a benchmark where target
lines are trivially discoverable from `p_env_l * baseline_loading_l^2`.

Recommended scenario principles:

```text
1. Keep the same S1-S5 conceptual structure and target sets for interpretive
   continuity.
2. Use moderate p_env contrast rather than binary high/low values.
3. Avoid target-margin capping that guarantees all target scores dominate all
   non-target scores.
4. Include nearby decoy lines with comparable p_env and/or baseline score.
5. Preserve suppressed-line intent only where it is part of the scenario
   concept, but do not over-suppress all inconvenient alternatives.
6. Store p_env provenance for every physical line and scenario.
7. Report target-overlap metrics, but do not optimize the methods directly for
   those target labels.
```

Concrete candidate p_env families to implement and compare:

```text
fair_moderate_contrast:
  targets:      p_env in roughly [0.45, 0.70]
  near decoys:  p_env in roughly [0.30, 0.60]
  background:   p_env in roughly [0.05, 0.25]

fair_rank_noisy:
  start from current conceptual target p_env values, then add deterministic
  seeded noise and clamp to [0.02, 0.80].

fair_score_overlap:
  choose p_env so that several non-target p_env_l * baseline_loading_l^2
  values overlap with the lower target scores.
```

For reproducibility, use deterministic seeds and write a table:

```text
scenario_id
line_id
is_target
is_decoy
is_suppressed
p_env
p_env_mode
baseline_loading
score = p_env * baseline_loading^2
rank_by_score
```

## Proposed DC MILP Formulation

This section is a recommended formulation for the next implementation. It is
not yet implemented in the repository.

Sets:

```text
N = buses
G = generators
D = load buses
L = canonical physical branches
C subset L = candidate lines allowed for wildfire topology control
S = source buses, including generator/PV/reference buses
```

Parameters:

```text
Pd_i          baseline active demand
Pg_min_g      generator lower bound
Pg_max_g      generator upper bound
B_l           DC susceptance, approximately 1 / x_l
Fmax_l        branch active-flow limit, preferably rateA or a consistent MW limit
p_env_l       environmental wildfire probability
lambda_R      wildfire-risk weight
lambda_L      load-shedding weight = 1 - lambda_R
rho_phys      physics/infeasibility weight for post-solution accounting, not
              necessarily needed inside hard-constrained DC MILP
M_l           big-M for branch angle relation when line is off
```

Decision variables:

```text
z_l in {0,1}       line energized status
y_l in {0,1}       line de-energized status, y_l = 1 - z_l for candidate lines
theta_i continuous bus voltage angle
f_l continuous branch active flow
Pg_g continuous generator active output
s_i in [0,1]       served load fraction
possibly r_l >= 0  absolute normalized flow approximation variable
possibly q_l >= 0  piecewise-linear approximation of r_l^2
```

Topology constraints:

```text
z_l = 1 for l not in C
z_l + y_l = 1 for l in C
optional: sum_{l in C} y_l <= K for budgeted variants
optional: no-good cuts when enumerating multiple DC MILP solutions
```

DC flow constraints for each branch `l = (i,j)`:

```text
f_l - B_l * (theta_i - theta_j) <= M_l * (1 - z_l)
f_l - B_l * (theta_i - theta_j) >= -M_l * (1 - z_l)

-Fmax_l * z_l <= f_l <= Fmax_l * z_l
```

Reference angle:

```text
theta_ref = 0
```

Nodal active-power balance:

```text
sum_{g at i} Pg_g
- s_i * Pd_i
- sum_{l outgoing from i} f_l
+ sum_{l incoming to i} f_l
= 0
```

If multiple loads/generators share a bus, aggregate or index them cleanly.

Generator limits:

```text
Pg_min_g <= Pg_g <= Pg_max_g
```

Load-service bounds:

```text
0 <= s_i <= 1 for load buses
s_i = 1 for non-load buses or omit non-load service variables
```

Island/source feasibility:

Preferred strict version:

```text
If bus i is served, it must be connected to at least one source through
energized lines.
```

This can be implemented with single-commodity flow reachability, multi-commodity
flow, or cut constraints. A simple single-commodity source-reachability model:

```text
h_ij <= (|N|-1) * z_l for each directed arc of physical line l
h_ij >= 0
sum incoming h - sum outgoing h >= s_i for load buses, with source injection
capacity from source buses
```

Implementation details can vary. The key requirement is that served load in a
source-less component is impossible or audited as zero.

Wildfire exposure objective:

MILP-compatible piecewise-linear option:

```text
r_l >= f_l / Fmax_l
r_l >= -f_l / Fmax_l
q_l approximates r_l^2 by convex piecewise-linear segments

R_raw_DC = sum_{l in C or L_eval} z_l * p_env_l * q_l
```

The product `z_l * q_l` can often be avoided because `f_l = 0` when `z_l = 0`,
so `r_l` and `q_l` should also be constrained to zero when `z_l = 0`. If that
is not clean, add standard linearization for the product.

Risk normalization:

```text
R_norm_DC = R_raw_DC / R_baseline_DC
```

Load shedding:

```text
L_shed_DC
  = sum_i Pd_i * (1 - s_i) / sum_i Pd_i
```

Candidate objective:

```text
minimize
  lambda_R * R_norm_DC
  + lambda_L * L_shed_DC
```

If adding soft penalties for violated constraints in a relaxed diagnostic
variant:

```text
minimize
  lambda_R * R_norm_DC
  + lambda_L * L_shed_DC
  + rho_DC * PAC_DC
```

But the primary DC MILP baseline should preferably enforce DC feasibility hard
and reserve physics/infeasibility reporting for post-solution diagnostics.

## Common Post-Solution Evaluator Required

To compare GridFM, DC MILP, and heuristics fairly, write a common evaluator
that accepts at least:

```text
scenario_id
method
lambda_R
lambda_L
rho_phys
shutoff_line_ids
z_by_line
served load fractions or alpha_effective
branch loadings/flows
generator dispatch if available
p_env_by_line
baseline_R
```

It should output:

```text
R_raw
R_norm
L_shed_effective
PAC_total or method-specific feasibility diagnostics
source_less_bus_ids
served_load_in_source_less_islands
num_shutoff_lines
expected_target_line_ids
observed_target_subset
observed_non_target_lines
expected_targets_not_selected
num_expected_target_lines
num_observed_shutoff_lines
num_observed_target_lines
num_observed_non_target_lines
target_recall
target_precision
target_overlap_fraction, retained as a compatibility alias for target_recall
line-level wildfire exposure provenance
load-shedding provenance
objective accounting:
  J_no_phys = lambda_R * R_norm + lambda_L * L_shed_effective
  J_true    = J_no_phys + rho_phys * PAC_total, where applicable
```

For DC MILP, include additional DC diagnostics:

```text
max_abs_nodal_balance_residual
max_branch_flow_limit_violation
max_angle_flow_equation_violation
generation_limit_violation
load_service_bound_violation
source_less_served_load
```

For GridFM, keep current PAC components:

```text
PAC_voltage_limits
PAC_thermal_limits
PAC_generator_limits
PAC_island_source_feasibility
PAC_total
```

Do not pretend DC voltage/reactive terms exist. Report them as not modeled by
the DC MILP if the comparison table needs placeholders.

## Locked K <= 2 Comparison Matrix

Required first comparison:

```text
models/methods:
  GridFM Stage G revised continuous:
    stage_e_k2

  DC MILP:
    Stage I-a K2 guided-search DC recourse
    Stage I-b direct DC MIQP with sum_{l in C} y_l <= 2

  Heuristics:
    budget-compatible TH/AH K <= 2 points

scenarios:
  existing S1-S5 scenario definitions, exactly as before

lambda_R:
  [0.8, 0.5, 0.2] for direct Stage H continuity
  optional full sweep [0.0, 0.05, ..., 1.0] after smoke validation

rho_phys:
  [0.0, 2.0] for continuity with Stage H
```

Stage E unconstrained and Stage I-b higher-K/unconstrained are explicitly out
of scope for this first comparison. They can be added later as a separate
topology-flexibility experiment.

Stage D exhaustive K<=2 is also omitted from the main comparison plots/tables
for this first Stage I build. It remains historical context, not a central
method in the guided-search comparison.

## Runtime Estimate And Execution Goal

The first implementation should run in a goal-oriented, restartable mode and
should not be considered complete until all required main and MLD tables,
figures, smoke checks, full-result checks, and final summaries are generated or
a true blocker is recorded.

Budget-derived workload estimate for the main run:

```text
GridFM Stage E K2:
  5 scenarios x 5 lambdas x 2 rho panels x up to 100 topologies
  = up to 5,000 GridFM continuous topology evaluations.

Stage I-a DC guided search:
  5 scenarios x 5 lambdas x up to 100 topology candidates
  = up to 2,500 fixed-topology DC recourse solves, duplicated across rho only
    for reporting.

Stage I-b DC MIQP:
  5 scenarios x 5 lambdas = 25 MIQP solves, duplicated across rho only for
  reporting.
  With TimeLimit=600 seconds, hard worst-case solver time is about 4.2 hours.

AC projection:
  5 scenarios x 5 lambdas x 2 rho x 3 families = 150 attempted finalist
  projection rows before caching.
  Since Stage I-a and Stage I-b are rho-invariant, expected unique projection
  solves are closer to 100 before any additional topology/solution duplicates.
```

Practical runtime expectation before smoke calibration:

```text
optimistic:    2-4 hours if GridFM/MIQP/projection solves are quick
conservative: 6-10 hours for full main + MLD + projection + plots
worst case:   10-20 hours if many MIQP solves reach TimeLimit or SciPy AC
              projection struggles
```

The smoke run must record observed per-job runtimes and update the projected
full-run estimate before launching the full experiment.

## Locked Stage I Visualization Scope

For the current `K <= 2` Stage I experiment, the main visual goal is to compare
guided GridFM search, DC approximation methods, and budget-compatible
heuristics without mixing in unconstrained topology flexibility. Stage D
exhaustive remains useful as an internal/reference table, but it should not be
the central guided-search curve in the main Pareto plots because it is an
enumeration methodology rather than the outer-topology/inner-recourse workflow
being studied.

Required cross-rho summary plots:

```text
1. effective_load_shedding_by_method
2. physics_feasibility_sensitivity_by_method
```

These cross-rho views should include the relevant `K <= 2` methods:

```text
Stage E K2
Stage I-a DC K <= 2
Stage I-b DC K <= 2
TH budget-compatible K <= 2, especially TH top-1 and TH top-2
AH budget-compatible K <= 2
```

Interpretation note: Stage I-a and Stage I-b enforce DC feasibility and do not
have the same GridFM PAC/physics-infeasibility semantics as GridFM-evaluated
methods. Their rows should therefore carry DC residual diagnostics,
`PAC_common_op_overlap`, and AC projection distance rather than pretending they
have the same GridFM `PAC_total` meaning. Physics-infeasibility/PAC summary
tables that use the existing GridFM PAC should be restricted to methods
evaluated with the existing GridFM formulation:

```text
TH top-1
TH top-2
Stage E K2
AH
```

Required per-rho plots:

```text
1. expected_vs_selected_shutoff_lines
2. target_recall_by_method
3. target_precision_by_method
4. target_recall_precision_by_method
```

`target_overlap_fraction` may remain as a backward-compatible alias for
`target_recall`, but the main interpretation should use both recall and
precision so methods are not rewarded merely for turning off more lines.

For this `K <= 2` experiment, omit `num_shutoffs_vs_objective` from the main
plot set because shutoff count has little variation once all methods are
restricted to at most two line shutoffs.

The Pareto/frontier scatter is a primary figure. For each rho panel and each
scenario, create a graph with:

```text
x-axis: L_shed
y-axis: R_norm
lambda sweep: lambda_R in [0.0, 0.2, 0.5, 0.8, 1.0]
```

The main Pareto figure should show non-dominated solutions across the evaluated
lambda cases for each method family. The expected primary curves are:

```text
1. Stage E K2 GridFM curve
2. Stage I-a K2 guided-search DC curve
3. Stage I-b DC K <= 2 curve
```

Add heuristic frontier overlays where useful:

```text
4. TH K <= 2 sparse frontier, combining TH top-1 and TH top-2 points
5. AH K <= 2 sparse frontier
```

These heuristic curves may be much sparser than the Stage E / Stage I curves.
Only connect points within the same method family, scenario, rho panel, budget,
and lambda sweep convention. Duplicate or overlapping topology points should
remain visible through jitter, annotations, or z-ordering.

Traditional lambda objective plots should be interpreted most cleanly under
`rho=0`. Under `rho=2`, if the plot is retained, also show or report the
objective with the physics-infeasibility component subtracted so the traditional
lambda tradeoff can still be compared:

```text
J_no_phys = J_true - rho * PAC_total
```

Required common operational diagnostic plots:

```text
x-axis: lambda_R
y-axis: PAC_common_op_overlap
line grouping: method family, scenario, rho panel as appropriate
```

This plot is the shared operational comparison layer for GridFM, DC, and
heuristic methods, unlike the GridFM-only `PAC_total` plot.

Required AC projection distance plots:

```text
1. GridFM Stage E K2 projection distance by scenario and lambda_R
2. Comparative Stage I-a vs Stage I-b DC projection distance by scenario and lambda_R
```

Projection distance is run only on finalists and means the minimum distance to
the nearest AC-feasible operating point under the same topology. GridFM
projection is command-faithful; DC projection is anchored to the DC
active-power/load/flow solution.

Useful additional plots to include if time permits:

```text
1. AC projection component breakdown:
   GridFM: D_Pg_cmd, D_Qg_cmd, D_s_cmd
   DC:     D_Pg_DC, D_s_DC, D_f_DC

2. solver diagnostics for Stage I-b:
   runtime_seconds, mip_gap, node_count, solution_count, optimality_certified

3. DC residual diagnostics:
   max nodal balance residual, flow-angle residual, branch limit violation,
   generator bound violation, load-service bound violation

4. method summary heatmaps:
   best method by scenario/lambda/rho for L_shed, R_norm,
   PAC_common_op_overlap, and AC projection distance
```

## Special MLD Literature-Alignment Study

In addition to the main `K <= 2` lambda-sweep results, run a smaller special
case aligned with maximum-load-delivery style comparisons from prior wildfire
outage literature, including the Rhodes balancing wildfire risk and power
outages framing. This is a separate result series, not a replacement for the
main generalized wildfire-risk/load-shedding tradeoff.

The conceptual purpose is to show how the current generalized formulation
behaves when it is reduced to an MLD-style setting:

```text
outer/proxy topology proposal: wildfire-risk-driven
inner continuous recourse: load-delivery-driven
```

For Stage E K2 and Stage I-a, use:

```text
lambda_R_proxy = 1
lambda_R_inner = 0
lambda_L_inner = 1
```

For Stage E K2 under GridFM:

```text
1. Use proxy_lambda_R = 1 to propose K <= 2 topology decisions.
2. For each proposed topology, run the existing inner continuous GridFM
   recourse with lambda_R = 0 and lambda_L = 1.
3. Run both rho panels:
   rho = 0 gives pure GridFM MLD-style load-delivery recourse.
   rho = 2 gives load-delivery recourse plus GridFM soft PAC penalty and must
   be labeled that way.
```

For Stage I-a:

```text
1. Use the same K <= 2 MLD topology pool/proposal convention.
2. For each fixed topology, solve DC recourse with lambda_R_inner = 0.
3. The inner objective is min L_shed_DC.
4. The resulting DC point is rho-invariant except for reporting/projection
   grouping.
```

For TH/AH heuristic comparisons in this MLD sub-study:

```text
1. Use the budget-compatible TH/AH K <= 2 construction to choose shutoff lines.
2. Run maximum-load-delivery evaluation by setting inner lambda_R = 0.
3. Include both rho panels for GridFM-evaluated heuristic rows, with rho=2
   clearly interpreted as adding GridFM soft PAC to the load-delivery recourse.
```

Expected MLD sub-study visuals:

```text
1. expected_vs_selected_shutoff_lines comparing Stage E K2 MLD,
   Stage I-a MLD, TH K <= 2 MLD, and AH K <= 2 MLD.

2. target recall / precision metrics for the same methods.

3. small Pareto/scatter view using x = L_shed and y = R_norm.
   Because this is not a lambda sweep, it may contain only a small set of
   non-dominated points. That is acceptable.

4. effective load shedding by method for rho=0 and rho=2.

5. physics/PAC sensitivity only for GridFM-evaluated rows:
   Stage E K2 MLD, TH K <= 2 MLD, and AH K <= 2 MLD.

6. common operational diagnostic and AC projection distance for finalist
   Stage E K2 MLD and Stage I-a MLD points where available.
```

Recommended result layout:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  dc_approximation_th_ah_heuristics_comparison/
    main_k2_lambda_sweep/
    mld_literature_alignment/
```

The MLD subfolder should carry its own metadata recording:

```text
proxy_lambda_R = 1
inner_lambda_R = 0
inner_lambda_L = 1
rho_panels = [0, 2]
topology_budget = K <= 2
```

For DC MILP, if feasibility is hard and no comparable PAC is in the objective,
still evaluate and report:

```text
J_no_phys
J_true_with_common_PAC_if_defined
DC objective value
DC feasibility status
```

## Expected Artifacts

The DC MILP implementation should write a result folder with the same spirit as
Stage G/H:

```text
run_metadata.json
progress.json
tables/
  dc_milp_solution_summary.csv
  dc_milp_variable_values.csv
  dc_milp_solver_diagnostics.csv
  dc_milp_constraint_diagnostics.csv
  p_env_by_scenario.csv
  scenario_baseline_loading_ranking.csv
  common_evaluator_results.csv
  wildfire_risk_provenance.csv
  load_shedding_provenance.csv
  source_less_island_audit.csv
  expected_vs_selected_by_method.csv
  method_comparison_summary.csv
plots/
  pareto_frontier_scatter.png
  objective_by_lambda.png
  target_recall_by_method.png
  target_precision_by_method.png
  target_recall_precision_by_method.png
  target_overlap_by_method.png, optional compatibility plot equivalent to recall
  effective_load_shed_by_method.png
  wildfire_exposure_by_method.png
```

The metadata should record:

```text
git branch and commit
scenario IDs
p_env mode
candidate physical line IDs
physical branch IDs
lambda values
rho values
DC formulation variant
solver name and version
time limit / MIP gap / final status
whether risk was MILP-linearized, MIQP, or post-evaluation only
```

## Validation Tests To Add

Suggested focused tests:

```text
1. DC topology variables:
   z_l + y_l = 1 for candidates and z_l = 1 for non-candidates.

2. Line-off physics:
   if z_l = 0, then f_l = 0.

3. DC nodal balance:
   saved solution satisfies balance within tolerance.

4. Load-shedding objective:
   saved L_shed equals sum Pd_i * (1 - s_i) / total Pd.

5. Source-less island correction:
   any bus in a source-less component has effective service zero in the
   common evaluator.

6. Wildfire exposure:
   saved R_raw equals sum z_l * p_env_l * loading_l^2 or the documented DC
   approximation.

7. Existing scenario reuse:
   current Stage I uses the existing S1-S5 scenarios exactly as before.
   Modified/fair p_env scenarios are deferred to a later extension.

8. Method comparison schema:
   GridFM, DC MILP, and heuristic rows share the same comparison columns.

9. Target agreement metrics:
   target_recall equals selected target lines divided by expected target lines;
   target_precision equals selected target lines divided by total selected
   shutoff lines. A method that selects one target and one non-target from a
   two-line expected target set should have recall 0.5 and precision 0.5.
```

## Interpretation Boundaries

Be explicit in the writeup and code comments:

```text
1. The current GridFM formulation is a surrogate predict-to-optimize workflow,
   not a hard AC OPF.

2. The revised GridFM evaluator decomposes PAC into operational, AC, and
   model-consistency groups. AC P/Q balance diagnostics are available in the
   retained revised Stage H run. Branch-flow consistency remains unavailable
   because GridFM does not expose an independent branch-flow/loading prediction
   channel; thermal loading is reconstructed from Vm/Va and should not be
   double-counted as independent branch-flow consistency.

3. The DC MILP is a hard-constrained DC approximation. It is a fairer
   optimization baseline, not a replacement for AC feasibility analysis.

4. The current designed target-margin scenarios helped debug methodology but
   may overstate heuristic and formulation performance. The DC comparison
   should use less obvious environmental probabilities.

5. Source-less islanding must affect effective load shedding. The current
   GridFM objective uses `L_shed_hybrid`, not the old commanded-only/effective
   load metric.
```

## Revised GridFM Checkpoint For DC MILP Construction

The current GridFM checkpoint before DC MILP construction is:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

The prior baseline used for before/after interpretation is:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
```

The revised evaluator distinguishes:

```text
L_shed_cmd
L_shed_gridfm_raw
L_shed_gridfm_effective
L_shed_hybrid
L_shed
```

with:

```text
L_shed = L_shed_hybrid
load_shed_mode = hybrid
```

The hybrid alpha semantics are:

```text
alpha_hybrid_i = 0.0
  if bus i is in a source-less island

alpha_hybrid_i = alpha_cmd_i
  if bus i is selected, controlled, and connected

alpha_hybrid_i = alpha_gridfm_effective_i
  if bus i is non-selected and connected
```

where:

```text
alpha_gridfm_raw_i =
  Pd_raw_i / Pd_base_i, if Pd_base_i > 0
  1.0, otherwise

alpha_gridfm_effective_i = clip(alpha_gridfm_raw_i, 0, 1)
```

The revised heuristic results show that the old load metric materially
understated service loss:

```text
prior heuristic baseline:
  rho=0: L_shed 0.0388, PAC_total 2.1621
  rho=2: L_shed 0.0386, PAC_total 2.1621

revised heuristic accounting:
  rho=0: L_shed 0.2790, PAC_total 169.5884
  rho=2: L_shed 0.3663, PAC_total 152.3797
```

The revised load decomposition was:

```text
rho=0:
  L_shed_cmd              0.0399
  L_shed_gridfm_raw       0.0915
  L_shed_gridfm_effective 0.3613
  L_shed_hybrid           0.2790

rho=2:
  L_shed_cmd              0.1273
  L_shed_gridfm_raw       0.0914
  L_shed_gridfm_effective 0.3613
  L_shed_hybrid           0.3663
```

The revised PAC decomposition was:

```text
rho=0:
  PAC_operational         55.4537
  PAC_AC                  10.8458
  PAC_model_consistency  103.2890
  PAC_total              169.5884

rho=2:
  PAC_operational         60.3835
  PAC_AC                  10.8175
  PAC_model_consistency   81.1787
  PAC_total              152.3797
```

Interpretation for the DC MILP agent:

- The DC MILP should not inherit the old commanded-only load accounting.
- Common post-evaluation should report both commanded and hybrid/effective load
  shedding.
- If the DC MILP has hard DC feasibility, still report the common GridFM-style
  post-evaluation metrics separately for comparison.
- Do not add an independent branch-flow consistency penalty unless a method
  exposes an independent branch-flow/loading prediction channel.
- Generator redispatch should be treated as implemented for controlled
  variables in the eval state, while raw-vs-commanded deviations remain
  model-consistency diagnostics.
- The next fair comparison should regenerate or otherwise align Stage G
  reference rows under the revised load/PAC evaluator before making final
  comparative claims against DC MILP.

## Immediate Next Task

We are currently at the step of devising the DC MILP based on the existing
GridFM wildfire formulation. The next implementation should:

```text
1. Add a Stage I or similarly named DC MILP comparison module.
2. Reuse the existing S1-S5 scenarios exactly as before for the first locked
   K <= 2 experiment.
3. Implement Stage I-a as its own Stage-E-style guided K2 topology loop with
   fixed-topology DC recourse as evaluator and no-revisit/no-good cuts.
4. Implement Stage I-b DC MIQP with topology, dispatch, DC flow, thermal limits, and
   load-service variables.
5. Add common post-solution evaluation that preserves effective load shedding
   and source-less island accounting.
6. Run a small smoke case first, preferably to a short local output path.
7. Compare GridFM Stage E K2, Stage I-a guided-search DC K <= 2, Stage I-b DC
   K <= 2, and budget-compatible TH/AH under the same existing scenarios and
   objective accounting.
8. Run AC projection only on selected finalists, with the default three-family
   finalist set capped at 150 attempted rows before caching.
9. Update the formulation/writeup after the Stage I results are saved and
   interpreted.
```
