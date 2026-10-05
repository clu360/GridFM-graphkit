# Wildfire First Pass History And Future Context

## Session Guard For Hallucination Checks

When the user sends a message in this work session or in any future session
that has read this file, the assistant must begin its next response with the
exact text `Message Confirmed` before continuing with the rest of the normal
response. This is an intentional user-requested consistency check to help
detect hallucinated results, stale methodology, or failure to read this
handoff.

This file is the durable handoff for future Codex sessions. It should be read before changing the wildfire first-pass experiment.

## August 22, 2026 - FT3/FT4 Cross-Model Figure Correction

Corrected all 13 fine-tuned Stage J figures to show the controlled comparison
requested by the experiment design. The original FT3/FT4 figures were generated
from the FT-only package, so the nine search figures could only show fine-tuned
Guided-GridSFM and fine-tuned TH, while the exact-reference figures could only
show methods available in that package. This was a visualization assembly
problem, not an execution or model-result problem.

No model was rerun. The correction joins the existing completed frozen/DC
package with the completed fine-tuned package and plots:

```text
Guided-DC
Guided-GridSFM (frozen)
Guided-GridSFM (fine-tuned)
TH-GridSFM-top1
TH-GridSFM-top2
```

The TH rows deliberately come from the established frozen run; only the guided
GridSFM rows come from FT3. Six merged `compare_*.csv` tables encode provenance
with `comparison_variant={dc,frozen,ft,th_frozen}`. FT4 validation now fails if
any comparison table omits a method/variant or introduces the FT-TH rerun into
the retained TH baseline. Both the external authoritative package and the
repository duplicate pass the updated validation.

## August 21, 2026 - Stage J GridSFM FT0 Smoke Validated

Implemented and completed FT0 for the separate Stage J GridSFM fine-tuning
extension. This is a controlled model-weight intervention and does not alter
the completed Stage J methodology. The official released v1.1 checkpoint was
fine-tuned through Microsoft/GridSFM's OPFData pipeline on ten GOC-500 FullTop
train graphs, then saved externally, reloaded in a fresh process, evaluated on
ten FullTop test graphs, and applied to one identical saved Stage J two-outage
candidate.

Final status is `FT0_VALIDATED_READY_FOR_FT1`. All parameter, finite-loss,
fresh-reload, output-schema, prediction-difference, Stage J evaluation, and
`D_input=0` gates passed. The FT0 checkpoint remains outside Git at:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\gridsfm_goc500_fulltop_ft0_n10.pt
SHA-256: CF72E0F6036D1E37FB5A298EB3DB5E69A6524F48BC7B0B8B5CD34E758DDFE4A2
```

The first OPFData use downloaded and processed a complete 15,000-graph shard
before applying the ten-graph cap. Approximately 13.05 GB of raw data and 1.35
GB of processed data remain in the external cache. Do not place these files or
future checkpoints in Git.

FT0 held-out metrics moved in mixed directions and must not be interpreted as
model-quality evidence. FT1 through FT5 remain unexecuted until FT0 review.
See `FT0_GRIDSFM_FINETUNE_SMOKE_STATUS.md` for the complete implementation and
methodology audit.

## August 19, 2026 - Stage J Complete Run Findings And Research Meeting Hold

Stage J is complete through the GOC-500 primary comparison and exact-AC
Reference A/B audits. The retained result package is:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run/
```

The completed experiment covers `J-S1` through `J-S3`,
`lambda_R=[0, 0.2, 0.5, 0.8, 1]`, `K<=2`, topology budget `100`, continuous
budget `20` per topology, and the approved `q=5` selected-load alpha
correction. It includes Guided-DC, Guided-GridSFM, TH-GridSFM-top1, and
TH-GridSFM-top2, with all finalist Reference A/B and warm-start records
present.

The retained evidence comprises:

```text
3,030 topology objective rows
60,600 candidate alpha evaluations
60 method finalists
60 Reference A fixed-z/fixed-alpha economic AC-OPF rows
120 Reference B maximum-load-delivery / economic tie-break rows
240 warm-start rows
480 native-to-Reference-A state-fidelity rows
```

The central guided-method finding is that Guided-DC is stronger than
Guided-GridSFM on the common wildfire-service decision surface in this
completed `q=5`, `K<=2`, 100-topology-budget study:

```text
Guided-DC average native J_trade:      0.260473
Guided-GridSFM average native J_trade: 0.333630

Guided-DC average native R_norm:       0.603736
Guided-GridSFM average native R_norm:  0.754192

Guided-DC average native L_shed_total:      0.002963
Guided-GridSFM average native L_shed_total: 0.007015
```

Reference A fixes each finalist's selected topology `z*` and load-service
command `alpha*`, then solves exact economic AC-OPF. Reference A confirms that
Guided-DC finalist instances solve slightly faster and have lower average AC
`R_norm`, while Guided-GridSFM finalist instances have lower average economic
AC objective:

```text
Guided-DC average Reference A solver time:      6.6661 s
Guided-GridSFM average Reference A solver time: 6.7369 s

Guided-DC average Reference A R_norm:      0.674001
Guided-GridSFM average Reference A R_norm: 0.680575

Guided-DC average Reference A objective:      460634.43
Guided-GridSFM average Reference A objective: 453440.84
```

Reference B fixes only the selected topology and maximizes load delivery under
exact AC redispatch. Guided-DC selected topologies recovered full load in all
15 guided finalist cases. Guided-GridSFM selected topologies recovered nearly
all load but had small residual MLD shedding in some cases:

```text
Guided-DC average Reference B served-load fraction:      1.000000
Guided-GridSFM average Reference B served-load fraction: 0.998950

Guided-DC minimum Reference B served-load fraction:      1.000000
Guided-GridSFM minimum Reference B served-load fraction: 0.994752
```

The state-fidelity comparison also favors Guided-DC in the aggregated
native-to-Reference-A normalized distance metric:

```text
Guided-DC average native-state distance to Reference A:      0.022693
Guided-GridSFM average native-state distance to Reference A: 0.207259
```

Warm-start solves modestly reduce Reference A solver time for both guided
finalist families. The relative savings are slightly larger for some
Guided-GridSFM finalist instances, but raw exact-AC Reference A solve times
remain slightly lower for Guided-DC finalists. This supports the interpretation
that Guided-DC selected better wildfire-service decisions and slightly easier
exact-AC instances in this run.

The supporting figure set added to the complete-run package is:

```text
figures/stage_j_guided_reference_a_discrepancy_distance.png
figures/stage_j_guided_reference_b_mld.png
figures/stage_j_guided_state_distance_heatmap.png
figures/stage_j_guided_warm_start_study.png
```

Interpretation boundary:

```text
Do not frame this as a blanket failure of GridSFM.
```

Guided-GridSFM did not outperform Guided-DC as the wildfire-service guided
selector under the current `q=5`, `K<=2`, 100-topology-budget design. However,
GridSFM remains useful as a frozen AC-OPF-surrogate state channel and gives a
distinct economic-dispatch perspective, including lower average Reference A
economic objective in this run.

No next implementation stage is currently authorized. The next step is the
research meeting. Bring the Stage J complete-run package, figures, Reference
A/B interpretation, warm-start results, and Guided-DC versus Guided-GridSFM
finding to that meeting before deciding any future work.

## August 21, 2026 - Large Experiment Artifact Handling

Future wildfire experiment runs should not commit large generated CSV traces
or temporary checkpoint tables directly to Git. This includes objective-call
traces, provenance tables, masking/clamping audits, rescored objective tables,
and any `tmp/` run artifacts.

Use compressed Parquet for retained large tabular artifacts instead of CSV.
Parquet stores typed columnar data and compresses repeated numeric/categorical
columns much more efficiently than row-oriented text CSV. The local conversion
utility is:

```powershell
python experiments/test/wildfire_tests/convert_large_csvs_to_parquet.py --remove-csv
```

Before committing a completed experiment package, run the converter or an
equivalent export path, keep compact summaries/configs/figures/manifests in
Git, and keep full oversized traces in Parquet or external artifact storage
with checksums. The repository ignore rules now exclude `tmp/` and the known
large wildfire trace-table CSV names across the wildfire experiment tree.

## August 17, 2026 - Stage J J8 Budgeted Topology Smoke

The first J8 outer topology-loop smoke has completed for `J-S1` with:

```text
lambda_R = 0.8
lambda_R_proxy = 0.8
K <= 2
topology_budget = 100
continuous_eval_budget = 20 per topology
q = 5 selected load buses per topology
full_coordinate_screen = false
```

Both methods used the same 100-topology pool generated by the outer proxy. The
current proxy is still the cheap Stage J connectivity/source-less score:

```text
w_l = p_env_l * baseline_loading_l^2
R_proxy(y) = sum_l w_l (1 - y_l) / sum_l w_l
L_proxy(y) = sum_l c_l y_l
```

The MLD/served-load one-off impact proxy is not active in this J8 smoke.

Both Guided-DC and Guided-GridSFM selected topology `276;473`, containing the
intended J-S1 high-risk target line `473`. Guided-DC evaluated 2,000 alpha
candidates in about `109.91` seconds and returned:

```text
J_trade = 0.2801653
R_norm = 0.3478828
L_shed_total = 0.0092954
max_loading ~= 1.0
num_loading_gt_1 = 0
```

Guided-GridSFM evaluated the same 2,000 alpha candidates in about `826.31`
seconds and returned:

```text
J_total = 0.3653721
J_trade = 0.3648472
R_norm = 0.4537538
L_shed_total = 0.0092208
PAC_total = 0.00026246
max_loading = 1.3091901
num_loading_gt_1 = 5
evaluation_status = model_output_penalized
```

The final J8 run has full coverage:

```text
Guided-DC:      100 eligible topologies, 0 failed, 2000/2000 ok evaluations
Guided-GridSFM: 100 eligible topologies, 0 failed,
                2000/2000 model_output_penalized evaluations
```

An intermediary GridSFM run was partial because the sandbox execution context
could not create missing official GridSFM cycle-basis cache files. The final
run was executed in Caleb's normal user context and redirected GridSFM runtime
cache outside git:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\cache\xdg
```

The successful result root is intentionally short to avoid Windows/OneDrive
path-length failures:

```text
experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08/
```

The detailed status handoff is:

```text
experiments/test/wildfire_tests/workflow/cases/
  CASE-003-stage-j-gridsfm-goc500-implementation/
  J8_BUDGETED_TOPOLOGY_SMOKE_STATUS.md
```

## August 17, 2026 - Stage J GridSFM OPF-Surrogate Model-Consistency Guardrail

Stage J is the current next-stage implementation focused on `GridSFM GOC-500`.
The active workflow case is:

```text
experiments/test/wildfire_tests/workflow/cases/
  CASE-003-stage-j-gridsfm-goc500-implementation/
```

Stage J has completed environment/bootstrap, adapter, baseline/DC, GridSFM
smoke, J7.5 alpha-search smoke gates, and the first J8 budgeted topology-loop
smoke.

Important methodological distinction:

The older Stage G/I GridFM workflow used a power-flow-style model that could
predict load/generator channels inconsistent with commanded control values.
That is why `PAC_model_consistency` included command-vs-predicted terms such as
raw predicted `Pd/Qd` compared with commanded load service.

The official GridSFM API currently used for Stage J is an OPF-surrogate style
interface. It receives the commanded demand and topology and returns OPF-like
electrical recourse outputs:

```text
inputs:  z, Pd_cmd, Qd_cmd, generator limits/costs, voltage/branch limits
outputs: theta, V, Pg, Qg, Pij, Qij, Pji, Qji, feasibility score
```

It does not output `Pd_pred` or `Qd_pred`. Therefore, for current GridSFM OPF
runs, the Stage I-style demand-command model-consistency check is retained in
the code path but marked unavailable/off:

```text
PAC_model demand-command consistency = unavailable / zero
```

This is intentional for OPF-surrogate models, not a missing penalty. The
guardrail remains in place for future PF-mode or alternate model comparisons.
If a future GridFM/GridSFM mode exposes predicted `Pd/Qd`, then re-enable:

```text
PAC_model_load =
mean(
  normalized_mse(Pd_pred, alpha_effective Pd_pre),
  normalized_mse(Qd_pred, alpha_effective Qd_pre)
)
```

Stage J still hard-enforces the islanding clamp before any DC/GridSFM
evaluation:

```text
alpha_effective_i = 0        if load i is source-less under topology z
alpha_effective_i = alpha_i  otherwise

Pd_cmd_i = alpha_effective_i Pd_pre_i
Qd_cmd_i = alpha_effective_i Qd_pre_i
```

Save both `alpha_requested` and `alpha_effective`. A difference between them
caused by source-less islanding is topology-forced island shedding, not a
GridSFM model inconsistency. For current J8 GridSFM candidate selection, the
active physics-aware term is:

```text
J_total^SFM =
  J_trade^SFM
  + rho_phys * (w_op PAC_operational + w_AC PAC_AC)
```

with `PAC_model` retained in schemas/artifacts for future PF-vs-OPF testing but
off/unavailable for the official GridSFM OPF-surrogate API unless a valid
independent consistency channel is added.

Workflow files updated for this decision:

```text
STAGE_J_PRIMARY_DESIGN.md
J7_5_V2_SCREENED_SCIPY_ALPHA_STATUS.md
CURRENT_STATE_SUMMARY.md
HISTORY.md
```

## July 31, 2026 - Stage I Result Root Naming Correction

The DC approximation + baseline heuristic + GridFM comparison is a Stage I
experiment. Its first generated result folders were placed under
`ieee_30_stage_a_to_i_results/stage_h/` because of naming drift from the Stage H heuristic
comparison lineage. The canonical result root has been corrected to:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/
  DC Approximation + Baseline Heuristic Comparison/
```

The older Stage H result root remains the home for the true Stage H heuristic
comparison runs, including `heuristic_baseline_comparison_topk` and
`heuristic_baseline_comparison_revised_load_pac`.

## July 13, 2026 - Session Closeout: Next Methodology Plan

The next work session should pick up from the methodology plan devised in a
separate agent. The plan includes a DC approximation baseline and an AC
projection step to check solution gaps after candidate decisions are produced.
A common operational diagnostic should be used across GridFM, DC, AC-projected,
and heuristic comparisons so that method differences are interpreted under the
same reliability/physics lens.

Remaining logistics before running experiments:

- implement the DC approximation and AC projection/evaluation path;
- align all methods on common objective accounting and operational diagnostic
  outputs;
- run the comparison experiments and regenerate summary plots/tables.

The first Stage I implementation should use the existing five scenarios `S1`
through `S5` exactly as before. Fairer `p_env` scenario construction is deferred
to a later extension after the locked `K <= 2` comparison is implemented and
understood.

The immediate literature-comparison goal is to apply the current GridFM-based
methodology alongside other heuristic methods, then use the results to identify
strengths and limitations of the GridFM approach. The next major research step
is robust optimization: add another layer for understanding decision
reliability, whether robustness improves decisions, and how it changes the
comparison against deterministic GridFM/DC/heuristic methods. The intended
milestone is to explore these pieces before the next professor meeting, then
use that meeting to decide next research steps.

## July 13, 2026 - Stage I DC/AC Projection Clarifications Locked

The Stage I implementation plan was clarified after reviewing the separate
methodology and clarification documents. The current Stage I experiment is
locked to a `K <= 2` topology comparison:

```text
|S_off| <= 2
sum_{l in C} y_l <= 2
```

The included main-comparison methods are:

```text
Stage D exhaustive K <= 2
Stage E K2
Stage I-a DC recourse evaluated on K <= 2 topology pools
Stage I-b direct DC MIQP with sum_{l in C} y_l <= 2
TH/AH budget-compatible K <= 2 heuristic points
```

Do not include Stage E unconstrained or Stage I-b unconstrained/higher-K in the
main Pareto/frontier comparison. Those are later topology-flexibility
extensions.

The DC comparison should be built as:

```text
Stage I-a:
  fixed-topology DC recourse over K <= 2 topology pools generated by the
  existing Stage D / Stage E / Stage H / GridFM workflow.

Stage I-b:
  joint topology plus continuous DC optimization, solved as DC MIQP when the
  squared flow-risk objective is retained, with sum_{l in C} y_l <= 2.
```

Stage I-b has the following locked solver settings:

```text
MIPGap = 1e-4
TimeLimit = 600 seconds initially, adjustable after smoke tests
save incumbent if TimeLimit is reached
if MIPGap > 1e-4 at termination, label result time-limited / not certified
```

Required Stage I-b saved diagnostics:

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

The `MIPGap = 1e-4` setting applies only to Stage I-b MIQP. Stage I-a has fixed
topology and no integer variables in the inner LP/QP recourse solve, so this
MIP gap setting does not apply there.

The primary risk denominator for cross-method Pareto comparison is:

```text
R_baseline_shared =
  sum_{l in L_phys} p_env_l * (baseline_loading_l_stored)^2
```

Stage I must inspect MATPOWER branch data for nontrivial transformer taps and
phase shifts. Use `B_l = baseMVA / x_l` only when taps/shifts are trivial. If
taps or shifts are nontrivial, use:

```text
tau_l = 1 if MATPOWER tap is 0, otherwise tap_l
phi_l = shift_l * pi / 180
B_l   = baseMVA / (x_l * tau_l)
f_l   = B_l * (theta_i - theta_j - phi_l)
```

Do not silently ignore nonzero taps or shifts.

Stage I-b must include the apples-to-apples budgeted variant:

```text
sum_{l in C} y_l <= 2
```

No `K <= 5`, unconstrained-over-`C`, or Stage E unconstrained variant belongs
in this current main experiment.

AC projection remains required for selected finalists only, with projection
solves cached by solution identity so duplicate solutions across `rho=0` and
`rho=2` are not solved twice. The default required finalist set is:

```text
5 scenarios
x 3 lambda values
x 2 rho panels
x 3 method families
= 90 projection jobs before caching
```

The three required families are GridFM Stage E K2, Stage I-a DC `K <= 2`, and
Stage I-b DC `K <= 2`. If Stage D exhaustive AC projection is intentionally
added, it becomes a fourth family and raises the uncached upper bound to 120
jobs.

Projection distance means the minimum distance from the method's chosen
solution to the nearest AC-feasible operating point under the same topology,
not merely an equation residual. GridFM projection is control-faithful:

```text
D_proj_GridFM = D_Pg_cmd + D_Qg_cmd + D_s_cmd
```

DC projection is active-power-oriented:

```text
D_proj_DC = D_Pg_DC + D_s_DC + D_f_DC
```

For GridFM, non-selected buses are not forced to match raw GridFM predictions
in the primary projection objective. For DC, `D_f_DC` compares AC from-end
active branch flow against signed DC `f_l` in the same canonical orientation.
Angle distance is zero-weighted by default (`w_theta = 0`).

Pareto/frontier plots should use `x = L_shed`, `y = R_norm`, connect only
points from the same method family/scenario/budget/lambda sweep convention, and
make duplicate topologies visible with jitter, annotations, or z-ordering.

## July 11, 2026 - Implemented Hybrid Load-Shedding And PAC Decomposition; Regenerated Stage H

Implemented the GridFM-side revision that must precede DC MILP construction.
The revised Stage G/Stage H evaluator now distinguishes commanded load
accounting from GridFM-implied load accounting:

```text
L_shed_cmd
L_shed_gridfm_raw
L_shed_gridfm_effective
L_shed_hybrid
L_shed
load_shed_mode = hybrid
```

The primary objective now uses `L_shed_hybrid`. `L_shed_cmd` remains as the
prior corrected commanded/effective metric. `L_shed_gridfm_raw` uses
`Pd_raw / Pd_base` without clipping, while `L_shed_gridfm_effective` clips that
ratio to `[0, 1]` and still forces source-less islanded load to unserved.
`L_shed_hybrid` uses commanded alpha for selected connected load buses,
GridFM-effective alpha for non-selected connected load buses, and zero service
for all source-less islanded load buses.

The PAC term is now decomposed and reported as:

```text
PAC_total =
    pac_operational_weight       * PAC_operational
  + pac_ac_weight                * PAC_AC
  + pac_model_consistency_weight * PAC_model_consistency
```

with default group weights equal to `1.0`. The saved result rows now include
operational, AC, and model-consistency groups plus component columns for
voltage, thermal, eval/raw generator limits, island feasibility, P/Q balance,
branch-flow consistency availability, and command-vs-raw GridFM deviations.
Branch-flow consistency is explicitly unavailable because GridFM has no
independent branch-flow/loading output channel; it is saved with availability
`False` and zero weight rather than silently treated as an active zero residual.
AC P/Q balance is computed when the outage-adjusted admittance data are
available; the regenerated Stage H run had P and Q balance available for all
180 evaluated rows.

Validation:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py tests/test_wildfire_stage_h_heuristic_comparison.py -q
18 passed
```

Full revised Stage H was run to a short temp path to avoid Windows/OneDrive
path-length failures, then copied and finalized under:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

Finalized run summary:

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

All 28 hard methodology checks passed. The revised run generated the original
Stage H plot set plus new summary plots for operational/AC/model PAC, commanded
vs GridFM vs hybrid load shedding, model alignment, and AC balance.

Key implementation finding: the connected non-selected load issue is real under
the revised accounting. Across the 180 Stage H continuous evaluations,
`L_shed_hybrid - L_shed_cmd` averaged approximately `0.2390` with a maximum of
approximately `0.4943`. `L_shed_gridfm_effective - L_shed_cmd` averaged
approximately `0.2777` with a maximum of approximately `0.6136`. This means the
old commanded-only connected-load accounting materially understated
GridFM-implied service degradation at non-selected connected load buses.

Average PAC decomposition over the 180 evaluated rows:

```text
PAC_operational:        57.9186
PAC_AC:                 10.8316
PAC_model_consistency:  92.2338
PAC_total:             160.9840
```

This reinforces the methodological framing: the revised evaluator is no longer
only an operational infeasibility score. It also records model-alignment
reliability under commanded interventions. The DC MILP handoff should now use
this revised evaluator as the GridFM baseline and should not build from the
earlier July 8 source-less-island correction alone.

## July 11, 2026 - Planned GridFM Load-Shedding And PAC Revision Before DC MILP

The active next step is a GridFM-side formulation revision before finalizing the
DC MILP comparison. This supersedes the earlier plan to move directly from the
July 8 effective-load-shedding correction into DC MILP construction. The reason
is methodological: the current Stage G revised continuous workflow still has
load-service and PAC accounting conventions that should be made more explicit
and better aligned with the claim that GridFM is being used as a fast learned
post-topology state predictor.

The July 8 correction fixed one important undercounting issue: all source-less
islanded load buses are counted as unserved. However, the current corrected
metric still assumes every connected non-selected load bus is fully served:

```text
alpha_cmd_eff_i = 0          if source-less islanded
alpha_cmd_eff_i = alpha_i    if selected/commandable and connected
alpha_cmd_eff_i = 1          if non-selected and connected

L_shed_cmd =
  sum_i Pd_base_i * (1 - alpha_cmd_eff_i) / sum_i Pd_base_i
```

This is valid as commanded/effective service accounting, but it is incomplete
for a GridFM-centered workflow. If GridFM predicts reduced `Pd_raw` at a
connected non-selected load bus after topology changes and recourse inputs, the
current objective ignores that model-implied service degradation by forcing
that bus's effective service to 1.0.

The planned update keeps `L_shed_cmd` for continuity and adds two GridFM-based
metrics:

```text
alpha_gridfm_raw_i = Pd_raw_i / Pd_base_i
alpha_gridfm_i     = clip(alpha_gridfm_raw_i, 0, 1)

alpha_gridfm_eff_i = 0                if source-less islanded
alpha_gridfm_eff_i = alpha_gridfm_i   otherwise

L_shed_gridfm =
  sum_i Pd_base_i * (1 - alpha_gridfm_eff_i) / sum_i Pd_base_i
```

and the proposed primary updated GridFM service metric:

```text
alpha_hybrid_eff_i = 0                if source-less islanded
alpha_hybrid_eff_i = alpha_i          if selected/commandable and connected
alpha_hybrid_eff_i = alpha_gridfm_i   if non-selected and connected

L_shed_hybrid =
  sum_i Pd_base_i * (1 - alpha_hybrid_eff_i) / sum_i Pd_base_i
```

Interpretation boundary:

```text
L_shed_cmd is commanded/effective load shedding.
L_shed_gridfm is GridFM-implied service accounting.
L_shed_hybrid is the updated GridFM-centered evaluated service metric.
L_shed_hybrid is not verified physical delivered load; it is the service
implied by GridFM for non-controlled connected loads while preserving
commanded alpha at selected controlled loads.
```

The updated objective should support paired modes:

```text
load_shed_mode = cmd      -> objective uses L_shed_cmd
load_shed_mode = hybrid   -> objective uses L_shed_hybrid

J_true =
  lambda_R * R_norm
+ lambda_L * L_shed_objective
+ rho_phys * PAC_total
```

All result rows should save `L_shed_cmd`, `L_shed_gridfm`, `L_shed_hybrid`,
`Delta_L_hybrid_minus_cmd`, and `Delta_L_gridfm_minus_cmd`, regardless of the
active objective mode.

The PAC update should decompose infeasibility into three groups:

```text
PAC_total =
  w_op    * PAC_operational
+ w_ac    * PAC_AC
+ w_model * PAC_model_consistency
```

with initial group weights:

```text
w_op = 1.0
w_ac = 1.0
w_model = 1.0
```

`PAC_operational` should include operating-limit and topology-validity
diagnostics evaluated on the final implemented/evaluated state:

```text
voltage-limit violation
thermal-limit violation over active physical branches
evaluated generator bound violations over all generator buses
source-less island accounting / topology consistency audits
```

Thermal loading must remain the Stage G corrected per-line rating-normalized
quantity:

```text
loading_l = apparent_MVA_flow_l(Vm_eval, Va_eval, topology) / rateA_l
PAC_thermal = mean_active_l max(0, loading_l - 1)^2
```

It is not a global maximum-flow normalization.

`PAC_AC` should include AC nodal-balance residual diagnostics computed from the
topology-adjusted admittance matrix and evaluated/clamped injections:

```text
rP_i = Pg_eval_i - Pd_eval_i - P_flow_i(Vm_eval, Va_eval, z)
rQ_i = Qg_eval_i - Qd_eval_i - Q_flow_i(Vm_eval, Va_eval, z)

PAC_P_balance = mean_i (rP_i / S_base)^2
PAC_Q_balance = mean_i (rQ_i / S_base)^2
```

Branch-flow consistency should not be activated in the current homogeneous
GridFM wildfire path. GridFM outputs:

```text
[Pd, Qd, Pg, Qg, Vm, Va]
```

It does not independently output branch flows or branch loadings. Current
branch loading is reconstructed from `Vm/Va` and MATPOWER branch metadata, so a
branch-flow consistency residual would compare a voltage-derived flow to
another voltage-derived flow. It should remain unavailable/NaN or zero-weight,
not treated as perfect consistency. Thermal violation plus P/Q balance are the
meaningful current AC-style diagnostics.

`PAC_model_consistency` should use raw GridFM predictions before clamping. It
answers whether GridFM respects the controls it is given:

```text
PAC_cmd_load:
  raw Pd-implied alpha vs commanded alpha at selected load buses

PAC_cmd_Pg:
  raw Pg prediction vs commanded Pg at selected generator buses

PAC_cmd_Qg:
  raw Qg prediction vs commanded Qg at selected generator buses

PAC_generator_limits_raw:
  raw generator bound violations over all generator buses
```

The current GridFM plumbing supports this raw/evaluated split. The decision
vector constructs full controlled values:

```text
Pg_cmd = Pg_base + Delta_Pg
Qg_cmd = Qg_base + Delta_Qg
Pd_cmd = alpha * Pd_base
Qd_cmd = alpha * Qd_base
```

Those selected `Pg/Qg/Pd/Qd` features are unmasked before GridFM inference so
GridFM sees the controls. GridFM still predicts all six output channels. The
Stage G evaluator then clamps selected controlled values back to the commanded
values in `x_eval` for objective evaluation, while retaining `x_raw` for
model-consistency diagnostics.

This means selected generator redispatch and selected load shedding should be
evaluated as implemented, but raw GridFM deviations from those commands should
be recorded. Generator bound diagnostics should be reported for both raw
predictions and the evaluated/clamped state, and evaluated generator bounds
should cover all generator buses, not only selected decision variables.

Focused plumbing validation remains:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py -q
10 passed
```

That test file verifies that selected controlled features are updated as full
`Pg_base + Delta_Pg`, `Qg_base + Delta_Qg`, `alpha * Pd_base`, and
`alpha * Qd_base`; that the controlled feature mask marks only selected
`Pg/Qg/Pd/Qd`; that clamping overwrites only selected controlled values; and
that the raw GridFM prediction is kept distinct from objective-evaluation
values.

Implementation order for the next coding session:

```text
1. Add/clarify x_raw and x_eval state handling.
2. Add L_shed_cmd, L_shed_gridfm, L_shed_hybrid.
3. Add load_shed_mode in {cmd, hybrid}.
4. Add PAC_cmd_load, PAC_cmd_Pg, PAC_cmd_Qg.
5. Split generator bounds into raw and evaluated all-generator diagnostics.
6. Add AC P/Q balance residual diagnostics.
7. Keep branch_flow_consistency unavailable/zero-weight.
8. Report PAC_operational, PAC_AC, PAC_model_consistency, and PAC_total.
9. Add bus/line/generator provenance tables and focused tests.
10. Run small paired smoke comparisons in cmd and hybrid modes.
```

After smoke validation, regenerate Stage H under the updated evaluator. The
regeneration should compare Stage D exhaustive `k<=2`, Stage E constrained k2,
Stage E unconstrained, TH top-k, and AH connected-topology heuristics under
the same updated load-shedding and PAC decomposition. The key interpretation
questions are:

```text
1. Does L_shed_hybrid reveal hidden service degradation at non-selected
   connected load buses?
2. Does the updated PAC decomposition change objective values or selected
   topologies?
3. Are operational, AC, or model-consistency violations dominant?
4. Do TH/AH heuristics look stronger or weaker under hybrid service accounting?
5. Does Stage E unconstrained remain the strongest empirical GridFM baseline?
6. Do command-tracking errors motivate a future projection/restoration layer?
7. Which topologies should be carried forward into SC-OPS/contingency analysis?
```

The DC MILP comparison should be deferred until this GridFM-side revision has
been implemented, smoke-tested, and documented. The DC baseline should compare
against the updated GridFM formulation, not the older commanded-only
load-shedding formulation.

## July 8, 2026 - Effective Load-Shedding Island Correction

The Stage G revised continuous implementation now treats source-less islanded
loads as unserved in the load-shedding objective accounting. Previously,
`L_shed` was based only on commanded alpha values, so a non-commandable load bus
inside a source-less island could still appear as fully served because its
implicit alpha was 1. The updated accounting keeps commanded and effective
service separate:

```text
alpha_effective_i = 0        if bus i is source-less islanded
alpha_effective_i = alpha_i  if bus i is selected/commandable and connected
alpha_effective_i = 1        if bus i is non-selected and connected

L_shed = sum_i Pd_base_i * (1 - alpha_effective_i) / sum_i Pd_base_i
```

The existing graph-traversal source-less island detection remains the GridFM
implementation mechanism because topology is fixed before continuous recourse.
Selected controllable load buses in source-less islands are still bound to
`alpha_i = 0`; non-selected islanded buses are now also counted as fully shed
through `alpha_effective_i = 0`.

The island component remains in `PAC_total`, but its interpretation is now an
additional operational topology-validity / island-severity penalty rather than
a patch for understated load shedding. This is useful for future larger-grid
extensions where critical loads, such as hospitals or emergency-service areas,
may carry higher islanding cost.

Focused validation:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py -q
10 passed
```

A direct calculation check confirmed the desired behavior: a non-selected
source-less islanded bus with commanded alpha 1.0 is assigned effective alpha
0.0 and counted as 100 percent shed. A minimal Stage D smoke run completed the
continuous evaluation and wrote the updated load-shedding provenance/checkpoint
tables, but plot finalization hit a local OneDrive/PIL path-materialization
issue in the temporary smoke folder.

Operational fallback for OneDrive/sandbox path issues:

If future smoke or full experiment runs fail during checkpoint, plot, or result
finalization because OneDrive is still syncing or materializing long nested
paths, use a local non-OneDrive output directory first, then copy the completed
run folder back into the intended repository result folder. The successful
smoke fallback pattern was:

```powershell
$out = Join-Path $env:TEMP 'stage_g_effective_load_shed_smoke'
python experiments/test/wildfire_tests/stage_g_implementation_revision/run_stage_g_revised_continuous_implementation.py `
  --models gnn `
  --scenario-ids S4 `
  --lambda-values 0.5 `
  --rho-phys 2 `
  --stages stage_d_k2_exhaustive `
  --stage-d-limit 1 `
  --call-budget 3 `
  --output-root $out
```

For full research runs, the same idea should be applied with a descriptive
local output folder. After successful completion and validation, copy the
entire `run_*` folder back into the corresponding `experiments/test/.../results`
subfolder. This preserves reproducible result organization while avoiding
transient OneDrive write/materialization failures during the run itself.

Current methodological next step:

Use the corrected effective-load-shedding implementation as the current
GridFM formulation baseline, then develop a fairer DC-approximation/MILP
comparison using the existing wildfire topology-control formulation. The DC
MILP should represent an industry-style approximation baseline with hard DC
power-flow balance, branch/thermal constraints, generator limits, load-service
variables, and algebraic topology/component relationships where needed. It
does not need to optimize the exact same `PAC_total` soft residual internally,
but comparisons should use a common post-solution evaluation layer for
wildfire exposure, effective load shedding, physics/constraint violations, and
objective accounting. After running, saving, and interpreting those DC MILP
comparison results, update the external GridFM wildfire formulation PDF with
the complete methodology progression through Stage G/H, the effective-alpha
island correction, the heuristic comparison caveat, and the DC MILP comparison
findings.

Future scale-up reminder:

After the small-grid heuristic, reliability, robustness, and DC-comparison
work is complete, revisit source-less islanding as an economic-impact modeling
venue for larger grids. The current island term is a uniform topology-validity
penalty, but larger systems can assign differentiated costs to islanded loads
or areas. This would let the formulation represent higher consequence for
critical facilities or service territories, such as hospitals, emergency
services, dense load pockets, or other priority customers, without confusing
that economic-impact layer with the basic effective-load-shedding accounting.

## July 3, 2026 - Revised Continuous Decision Quality Checkpoint Complete

The Stage G revised continuous implementation checkpoint is complete. The
completed canonical result folder is:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_233908
```

This closes the current implementation-revision arc:

```text
Stage G loading/state correction
-> scenario-baseline revision
-> baseline-margin revision
-> continuous-control masking and exact controlled-value objective evaluation
-> revised continuous decision-quality run and plot regeneration
```

Key methodology state:

- Topology decisions, baseline features, GridFM-visible controlled inputs,
  raw GridFM predictions, and clamped objective-evaluation values are now
  explicitly separated.
- GridFM sees selected controlled `Pg/Qg/Pd/Qd` values through the effective
  input mask.
- Objective evaluation clamps controlled decision values while leaving
  non-controlled state to GridFM prediction.
- Wildfire exposure, load shedding, and physics-infeasibility terms are then
  evaluated from that post-topology/post-recourse state.
- Result generation includes progress/checkpoint artifacts so interrupted runs
  can be resumed.

Empirical conclusion at this checkpoint:

The revised continuous formulation is a tentative but functioning small-grid
methodology for wildfire-aware topology control with post-topology recourse.
Across many tested settings, it chooses plausible line shutoffs to mitigate
wildfire exposure, typically avoids source-less islanding behavior, and uses
continuous controls to reduce load shedding and physics infeasibility relative
to the fixed-control baseline-margin revision.

The evidence remains empirical. It is difficult to quantify accuracy or provide
theoretical reliability guarantees from the current small-grid study alone.
Topology agreement with the previous margin revision is only partial. When the
margin revision is capped at the same 50-topology budget as the revised
continuous run, `94 / 150` shared selected settings choose the same topology
and `56 / 150` choose a different topology. This means the continuous recourse
is not merely improving feasibility on fixed topologies; it also changes the
objective landscape and selected best topology in a meaningful subset of
settings.

Next major step:

Compare the revised continuous formulation against transparent TH/AH heuristic
baselines and then run robustness checks. The heuristic comparison should use
the same scenario definitions, objective accounting, lambda settings,
rho settings, and post-topology evaluation conventions so that decision quality
can be interpreted directly. Robustness analysis should then perturb wildfire
probabilities, loading assumptions, scenario targets, rho/lambda settings, and
topology budgets. Only after these comparisons should the work move toward
larger-grid and eventual real-world-scale experiments.

Physics-infeasibility interpretation caveat:

The current `PAC_total` physics-infeasibility term should not be described as
a complete normalized AC OPF residual. It is presently a partial soft
infeasibility score:

```text
PAC_total =
  voltage_limits
+ thermal_limits
+ generator_limits
+ island_source_feasibility
```

with default unit weights. The implemented active terms are:

- voltage-limit violation, measured as squared per-unit excursions outside
  `[0.95, 1.05]`;
- branch thermal violation, measured as squared excess over normalized active
  line loading ratio `S / Smax > 1`;
- generator/decision bound violation, currently represented as a bound-check
  penalty in the revised continuous runner;
- source-less island feasibility, measured as demand-weighted served load on
  buses disconnected from generation/source buses.

The AC OPF analogue is not fully implemented yet. Active/reactive nodal power
balance residuals, reactive generator-limit residuals beyond the available
decision-vector bounds, AC branch-flow consistency residuals, and explicit
topology-dependent on-line/off-line branch-equation residuals are currently
placeholders or implicit effects rather than active normalized constraints.
This matters because the current PAC can reduce when many lines are
de-energized, especially when thermal violations on remaining active lines
drop. Future methodology work should revisit whether to add feasibility
restoration layers, hard feasibility screens, improved residual normalization,
or AC power-flow/OPF-based repair. However, the immediate plan is to first
evaluate the current methodology against TH/AH heuristics and robustness /
SC-OPS-style checks, then use those comparisons to decide how to refine the
small-grid physics methodology.

## June 26, 2026 - Stage G Implementation Revision Period

Stage G is an implementation-correction period, not a change to the wildfire
optimization methodology. The active goal is to make the existing Stage C/D/E/F
objective and topology-selection machinery operate on correctly interpreted
GridFM state, physical branch loading, and MATPOWER IEEE-30 branch ratings.

Corrections being introduced under Stage G:

```text
G1. Interpret homogeneous GridFM outputs as [Pd, Qd, Pg, Qg, Vm, Va].
G2. Convert denormalized voltage angles from degrees to radians for phasors.
G3. Compute loading as apparent MVA flow divided by physical rateA.
G4. Attach static MATPOWER IEEE-30 branch metadata without a pandapower runtime dependency.
G5. Treat wildfire line risk over canonical off-diagonal physical branches:
    self-loops are excluded, directed edge pairs share one physical loading,
    and current line IDs remain available for compatibility outputs.
```

The earlier Stage F physics/topology conclusions are provisional until rerun
after these corrections. In particular, the previously observed line 23/30
thermal outlier was traced to implementation artifacts: output-column
misinterpretation, degree/radian handling, and missing physical rateA metadata.

Stage G also adds an explicit loading-ranking audit command. When intentionally
run, it writes `stage_g_loading_ranking_audit.csv` and prints a compact CSV
view for chat/reporting. No Stage G research results should be generated
implicitly during implementation.

After G1-G5 are corrected and audited, the next Stage G subproblem is the
continuous-recourse masking design: decision variables must be visible to
GridFM and then enforced as exact values in objective evaluation while
non-decision buses/lines remain GridFM-predicted. Later stages are expected to
focus on AH/TH heuristics and SC-OPS.

Stage G MATPOWER-30 decision-quality result generation now intentionally keeps
GridFM as the state-prediction component for both initialization and topology
evaluation. The MATPOWER branch-flow comparison showed that the remaining
extreme line loadings are driven primarily by the GridFM-predicted voltage
state, not by the simplified loading equation. This is recorded as a current
model-state limitation rather than corrected by manually replacing loading
values or solving a separate AC power flow.

For the Stage G MATPOWER-30 decision-quality reruns, the Stage F
physics-aware result methodology and plot suite are duplicated under
`ieee_30_stage_a_to_i_results/stage_g/matpower_30/` with two families:

```text
with_physics_infeasibility/rho100
without_physics_infeasibility/rho0
```

Both families use the same physical-branch corrections and canonical line
semantics. GridFM-heavy-loading physical lines identified in the Stage G audit
are assigned a lower temporary wildfire probability when they are not explicit
scenario targets, so they do not dominate every scenario only because of the
current GridFM voltage-state limitation. Explicit scenario targets still take
priority and remain high probability.

Follow-up correction: Stage G decision-quality scenario target IDs, suppressed
IDs, and expected-match IDs are canonicalized to physical branch IDs before
constructing `p_env` and before computing expected-vs-observed hit metrics.
The completed MATPOWER-30 result folders were repaired in place by recomputing
`expected_vs_observed_line_subsets.csv`, `scenario_definitions.json`, and the
per-scenario match plots from the already saved best-topology CSVs. No GridFM
or Gurobi rerun was required for this accounting repair.

Open Stage G interpretation issue: after the implementation corrections, the
`rho_phys=100` physics-infeasibility term can dominate topology selection. In
the Stage G MATPOWER-30 edge cases, `lambda_R=0` with `rho_phys=0` selects the
empty topology as expected, but `lambda_R=0` with `rho_phys=100` still shuts
off lines because the objective is effectively minimizing `L_shed + 100 *
PAC_total`. This is not a wildfire-risk signal; it is the physics residual
driving topology selection.

Two immediate diagnostics are proposed before changing the broader
optimization methodology:

1. Calibrate temporary `p_env` suppression for GridFM-heavy-loading outlier
   branches by rank. For scenarios where an outlier line is not an explicit
   target, solve for a low enough `p_env` that the line falls below the top
   wildfire-priority set rather than simply picking a fixed value such as
   `0.005`.
2. Run a Stage G physics sensitivity with a smaller residual weight such as
   `rho_phys=10` and compare against `rho_phys=0` and `rho_phys=100`. This
   tests whether the physics residual is useful as a feasibility regularizer
   without overwhelming the intended wildfire/load-shed tradeoff.

Longer-term possibilities for handling GridFM prediction residuals include
better residual normalization, scenario-specific residual calibration, and
distance-to-feasible-set ideas such as projection-style penalties. These are
not being implemented yet; they are recorded as potential methodological
directions after the current Stage G implementation audit.

Stage G now proceeds to a physics-infeasibility sensitivity case study under
the corrected MATPOWER-30 implementation. The case study keeps the fixed-control
bi-level topology setup: p_env and baseline loading-aware proxy coefficients
are fixed before Gurobi topology proposals, Gurobi proposes candidate
topologies, and GridFM evaluates each topology at fixed controls. There is no
continuous-control optimization in this case study.

Because the remaining extreme loadings are driven by GridFM voltage-state
predictions, the case study manually calibrates p_env for non-target
GridFM-heavy-loading outlier branches. Explicit scenario targets remain high
probability; ordinary non-targets remain low probability; non-target outlier
branches are lowered enough to fall below the weakest intended target in that
scenario. This is a temporary diagnostic control to reduce confounding from
GridFM loading artifacts, not a replacement for GridFM state prediction.

The rho sensitivity will evaluate and rescore the same fixed-control topology
metric pool across `rho_phys = [0, 10, 20, 50, 100]` and lambda values from
0.0 to 1.0 in steps of 0.05. Methodology fidelity checks must confirm that rho
only changes final scoring/selection, not p_env, GridFM predictions, candidate
physical line IDs, or intrinsic topology metrics.

Stage G physics-infeasibility sensitivity run completed:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_sensitivity/run_20260630_020627/
```

Verification summary:

```text
topology_metric_pool.csv rows:              50,085
rho_rescored_objectives.csv rows:          250,425
best_by_rho_scenario_lambda_stage rows:      1,575
expected_vs_selected_by_rho rows:               375
physics_sensitivity rows:                       225
cost_of_feasibility rows:                       225
line23_frequency_by_rho rows:                    75
rho_tradeoff_summary rows:                       25
plot PNG count:                                 154
methodology fidelity checks: all hard checks passed
```

Main sensitivity finding: `rho_phys=10` captures much of the PAC reduction
relative to `rho_phys=0`, but it also reintroduces frequent selection of
canonical line 23 in non-S2 scenarios. Averaged over stages and the full
lambda sweep, line 23 frequency changes as follows:

```text
S1: rho0 0.016 -> rho10 0.571 -> rho100 0.921
S2: rho0 0.810 -> rho10 1.000 -> rho100 1.000
S3: rho0 0.048 -> rho10 1.000 -> rho100 1.000
S4: rho0 0.206 -> rho10 1.000 -> rho100 1.000
S5: rho0 0.016 -> rho10 1.000 -> rho100 1.000
```

Average target-overlap also generally decreases outside S2 as rho increases:

```text
S1: rho0 0.378 -> rho10 0.133
S2: rho0 0.800 -> rho10 1.000
S3: rho0 0.183 -> rho10 0.117
S4: rho0 0.800 -> rho10 0.400 -> rho100 0.267
S5: rho0 0.229 -> rho10 0.162 -> rho100 0.133
```

Interpretation: the p_env calibration successfully de-prioritizes non-target
GridFM-heavy-loading outliers for the rho0 wildfire/load objective, but even a
moderate physics penalty can pull decisions back toward line 23 because the
PAC term is dominated by GridFM-predicted thermal residuals. This supports
treating physics-infeasibility weighting as an active decision-quality design
question before moving to the continuous decision-variable methodology shift.

Low-rho refinement was appended to the same run by reusing the saved topology
metric pool, with no new GridFM or Gurobi topology evaluations:

```text
rho_phys = [0, 0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 50, 100]
```

Updated aggregate sensitivity:

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

Interpretation of the refinement: `rho_phys=0.1` appears to be the only
positive rho in the tested grid that meaningfully reduces PAC while mostly
preserving target behavior and avoiding a large line-23 resurgence. By
`rho_phys=0.25`, line 23 frequency more than doubles relative to rho0, so the
physics residual is already exerting a strong topology-selection effect.

## July 1, 2026 - Stage G Scenario-Baseline Physics Sensitivity Revision

Stage G now includes a scenario-baseline revision of the physics-infeasibility
sensitivity study. This is still an implementation-revision experiment, not a
new wildfire optimization methodology.

The correction separates the roles of each data source:

```text
stored GridFM scenario baseline:
  baseline Pd/Qd/Pg/Qg/Vm/Va, baseline loading, R_base, proxy ranking

MATPOWER IEEE-30:
  physical branch identity, rateA, and canonical branch metadata

GridFM inference:
  post-topology fixed-control state evaluation only
```

The stored scenario baseline was verified as physically reasonable under the
Stage G MATPOWER-rate loading calculation:

```text
max scenario-baseline loading:       0.877465015024
branches above 100% loading:         0
canonical line 23 baseline loading:  0.524397336396
```

Because this removes the GridFM-inferred baseline loading artifact from proxy
construction and `R_base`, the default scenario probabilities are restored:
explicit scenario targets use `p_env=1.0`, and non-target lines use
`p_env=0.05`. The rank-calibrated ultra-low outlier suppression remains a
diagnostic artifact-control sensitivity, not the corrected default.

New result folder:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_sensitivity_scenario_baseline_revision/run_20260701_003629/
```

Verification summary:

```text
scenario_baseline_loading_ranking rows:        40
topology_metric_pool rows:                 50,085
rho_rescored_objectives rows:             601,020
best_by_rho_scenario_lambda_stage rows:     3,780
expected_vs_selected_by_rho rows:             900
p_env_by_scenario rows:                       200
methodology fidelity checks:              19/19 passed
plot output checks:                      259/259 present
```

Aggregate scenario-baseline revision sensitivity:

```text
rho    target_overlap  line23_freq  avg_PAC  avg_R_norm  avg_L_shed
0      0.408           0.397        1.554    18.525      0.224
0.1    0.408           0.397        1.554    18.525      0.224
0.25   0.421           0.416        1.528    17.991      0.229
0.5    0.477           0.470        1.486    16.564      0.241
0.75   0.477           0.473        1.483    16.585      0.241
1      0.477           0.486        1.479    16.618      0.242
2      0.480           0.517        1.458    16.714      0.246
5      0.476           0.613        1.402    17.395      0.257
10     0.476           0.702        1.354    18.375      0.262
20     0.430           0.800        1.327    20.092      0.268
50     0.398           0.914        1.244    25.236      0.281
100    0.341           0.978        1.206    28.029      0.288
```

Interpretation: using stored scenario-baseline loading fixes the baseline/proxy
artifact and removes the need for default p_env suppression, but line 23 can
still be selected frequently because GridFM post-topology inference remains the
true evaluator and the PAC term still favors topologies that reduce its
GridFM-predicted thermal residuals. The `rho=0.5` to `rho=2` range gives the
best target-overlap averages in this revised fixed-control run, but it also
increases line 23 frequency relative to `rho=0`. Very large rho values reduce
PAC further but again degrade target behavior.

A post-run scenario-design audit was added to the same result folder:

```text
tables/scenario_design_audit.csv
tables/scenario_target_rank_audit.csv
```

The audit checks target membership, target rank under the corrected
scenario-baseline proxy, target-vs-non-target score margin, and selected-line
frequency. It classifies `S1`, `S3`, and `S5` as scenario-design weaknesses
because at least some expected targets are too weak under `p_env * loading^2`
after the scenario-baseline correction:

```text
S1 weak targets: 18,32   weakest target rank: 36   min margin: 0.020
S3 weak targets: 18,22   weakest target rank: 5    min margin: 0.612
S5 weak targets: 18,32,47,88,91  weakest target rank: 37  min margin: 0.020
```

`S2` and `S4` are classified as mixed/acceptable: their explicit target lines
rank first under the scenario-baseline proxy, but the broader objective still
selects non-target lines such as 14, 23, 72, and 77 depending on rho/lambda.
This means the low hit rates are not only a methodology failure; some of the
scenario targets are not strongly separable once the physically reasonable
baseline loading is used.

Follow-up margin scenario revision completed. Canonical line 32 was removed
from the expected/target sets because its stored scenario-baseline loading is
only about `0.0279`, making it an unsuitable target floor. Non-target p_env
values were then calibrated so each non-target baseline proxy score is at most
`0.25 * weakest remaining target score` for that scenario, with targets kept
at `p_env=1.0`.

New margin result folder:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_sensitivity_scenario_baseline_margin_revision/run_20260701_013324/
```

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

Calibrated non-target p_env ranges:

```text
S1 min p_env 0.00765, median 0.05
S2 min p_env 0.05,    median 0.05
S3 min p_env 0.00765, median 0.05
S4 min p_env 0.05,    median 0.05
S5 min p_env 0.00289, median 0.04996
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
methodology/evaluation failures rather than scenario-design weaknesses: their
targets are now top-ranked with a target-vs-non-target margin near 4x, but the
selected topologies still frequently include non-target lines such as 14, 23,
72, and 77. This strengthens the conclusion that the remaining misses are tied
to the fixed-control GridFM evaluation/objective behavior, not simply weak
scenario construction.

Stage G revised continuous implementation added. A new runner implements the
next continuous-recourse checkpoint under:

```text
experiments/test/wildfire_tests/stage_g_implementation_revision/run_stage_g_revised_continuous_implementation.py

ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_revised_continuous_implementation/
```

The continuous decision vector is now `[Delta_Pg, Delta_Qg, alpha]`.
Selected generator buses control both `Pg = Pg_base + Delta_Pg` and
`Qg = Qg_base + Delta_Qg`; selected load buses use one `alpha` to clamp both
`Pd = alpha * Pd_base` and `Qd = alpha * Qd_base`. Because `ScenarioData`
does not currently expose `Qg_min/Qg_max`, this first Stage G implementation
uses symmetric `Delta_Qg` bounds of `+/- 5 MVAr`.

The revised continuous path adds intervention-aware GridFM masking:
controlled `Pg/Qg/Pd/Qd` features are made visible to GridFM even if the saved
random scenario mask would otherwise hide them. After GridFM inference, those
controlled values are clamped back into the evaluation state. `Vm/Va` remain
GridFM-predicted, and non-selected controls remain baseline/unchanged unless
recorded otherwise by the provenance tables.

True continuous objective definitions:

```text
R_raw = sum_l z_l * p_env_l * loading_l(combined_state)^2
R_norm = R_raw / R_base_s
L_shed = sum_n Pd_base[n] * (1 - alpha_full[n]) / sum_n Pd_base[n]
J_no_phys = lambda_R * R_norm + (1-lambda_R) * L_shed
J_true = J_no_phys + rho_phys * PAC_total
```

Important clarification: `impact_l` is intentionally excluded from the true
wildfire-risk evaluation term. It remains part of the topology/proxy side and
scenario construction logic, but not the GridFM-evaluated true wildfire risk.

The first planned reduced run keeps Stage D exhaustive and caps only Stage E:

```text
scenarios: S1-S5
lambda_R: [0, 0.2, 0.5, 0.8, 1.0]
rho_phys: [0, 2]
Stage D: exhaustive k<=2
Stage E k2 budget: 50
Stage E unconstrained budget: 50
expected continuous topology/rho optimizations: 18,850
```

Audit tables include `masking_clamping_audit.csv`,
`controlled_state_consistency.csv`, `load_shedding_provenance.csv`, and
`wildfire_risk_provenance.csv`. Hard checks fail the run if controlled
features are hidden from GridFM, clamped values do not match commanded values,
load-shedding provenance does not sum to `L_shed`, wildfire-risk provenance
does not sum to `R_raw`, or the true objective formulas drift.

Verification so far:

```text
py_compile: passed
tests/test_wildfire_stage_g_revised_continuous.py: 5 passed
tests/test_wildfire_stage_g_implementation.py + revised continuous tests: 18 passed
```

Stage-D-only smoke validation completed because Gurobi Stage E proposal
generation cannot run under the sandbox username:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_023730/

scenario_ids: S1
lambda_R: [0, 0.5]
rho_phys: [0, 2]
stage_d_limit: 1
stage_e_budget: 0
call_budget: 3
continuous results: 4
methodology hard-check failures: 0
```

The Gurobi-backed smoke/full reduced run still needs to be launched outside
the sandbox or after escalation is available. A failed sandbox attempt left an
incomplete folder at `run_20260701_023702/`.

Follow-up rho=0 launch status. The full reduced `rho_phys=0` run was launched
with Stage D exhaustive, Stage E budgets `50/50`, all five scenarios, and
`lambda_R = [0, 0.2, 0.5, 0.8, 1]`. It exceeded the 2.5-hour checkpoint, so
`rho_phys=2` was not started. The process later stopped without reaching the
table-writing phase. The latest folder is:

```text
ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_revised_continuous_implementation/run_20260701_034434/
```

That folder currently contains only:

```text
inputs/selected_decision_buses.json
```

No `tables/`, `plots/`, or `inputs/metadata.json` were written, so this run
should be treated as incomplete and not analyzed as a result.

## June 25, 2026 - Current Investigation: Targeted Post-Topology Corrective Actions

The fixed-control and continuous traditional-lambda physics studies have now
been compared directly.

Current empirical finding:

```text
1. The physics-infeasibility penalty changes the final topology.
2. The current continuous optimization does not change the final topology
   relative to the matched fixed-control study.
```

The second statement was checked across:

```text
4 stages:
  Stage C
  Stage D exhaustive k<=2
  Stage E constrained K<=2
  Stage E unconstrained

3 traditional lambda cases:
  lambda_R = 0.8, 0.5, 0.2

2 physics settings:
  rho_phys = 0, 100

total matched comparisons = 24
```

All 24 matched comparisons selected the same topology with and without the
current continuous optimization. The physics setting itself does change the
selected topology. For the exhaustive Stage D reference:

```text
rho_phys=0:
  lambda_R=0.8 -> [18,23]
  lambda_R=0.5 -> [14,23]
  lambda_R=0.2 -> [23,30]

rho_phys=100:
  lambda_R=0.8 -> [23,27]
  lambda_R=0.5 -> [23,27]
  lambda_R=0.2 -> [23,27]
```

The current continuous parameterization is:

```text
Delta_Pg on the three largest baseline PV generators:
  [1,10,12]

alpha on the five largest baseline-demand PQ buses:
  [6,20,11,29,18]
```

This parameterization was intentionally simple, but it is not sufficiently
targeted for the role continuous optimization is now expected to play.
Reported best solutions have zero generator movement for `rho_phys=0` and less
than `0.0034 MW` maximum generator movement for `rho_phys=100`. Alpha can move
substantially under the physics penalty, but the selected buses are based only
on demand magnitude, and the commanded-versus-predicted service diagnostics
remain difficult to interpret as a physically enforced recourse solution.

The new methodological framing is:

```text
topology stage:
  choose de-energized lines to reduce wildfire exposure while considering
  service and physics quality

corrective stage:
  hold the selected topology fixed and use targeted generator redispatch and
  explicit load shedding as post-topology corrective actions

corrective objective:
  reduce remaining physics infeasibility and load shedding while preserving
  the wildfire-risk reduction delivered by the topology decision
```

The corrective layer is not presently expected to replace or reverse the
topology decision. Its purpose is to improve the operating state associated
with a converged topology. This role will be important later when evaluating:

```text
- converged solution quality,
- Stage F decision quality,
- AH and TH heuristic comparisons,
- robust/security-constrained operation and SC-OPS-style contingency tests.
```

Current working sequence, explicitly flexible and subject to change:

```text
1. First extend the existing fixed-control Stage F decision-quality analysis
   with matched rho_phys=0 and rho_phys=100 true evaluation. This isolates
   whether physics-aware ranking improves scenario behavior without the cost
   or confounding effects of continuous recourse.
2. Then define a targeted methodology for selecting redispatch generators.
3. Define a targeted methodology for selecting controllable load-shedding
   buses.
4. Replace the saved random inference mask with an intervention-aware mask so
   controlled Pd, Qd, and Pg values are never hidden from GridFM.
5. Treat alpha-commanded Pd and Qd and Delta_Pg-commanded Pg as clamped
   quantities in the combined post-inference state; use GridFM predictions
   only for unresolved or uncontrolled quantities.
6. Settle the final load-service/recourse evaluation definition.
7. Test whether the revised controls reduce PAC physics residuals and required
   load shedding for fixed candidate topologies.
8. Check whether wildfire exposure remains acceptably close to the topology
   stage's intended risk reduction.
9. Rerun the physics-infeasibility study with corrected continuous controls.
10. If promising, rerun Stage F decision quality with the corrected recourse
    layer.
11. Compare against AH and TH heuristics.
12. Evaluate robust/security-constrained and SC-OPS-style operation.
```

After the corrective methodology is finalized:

```text
1. Rerun the physics-infeasibility case study.
2. Rerun the Stage F decision-quality analysis.
3. Compare revised solutions against AH and TH heuristics.
4. Implement and evaluate the robust/security-constrained methodology,
   including SC-OPS-style contingency preservation.
5. Then evaluate larger-grid computational efficiency and economic impact.
```

Canonical evidence:

```text
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/
  without_continuous_optimization/
    rho0_no_physics/run_20260623_220854/
    rho100_with_physics/run_20260623_221317/
  with_continuous_optimization/
    rho0_no_physics/run_20260624_000543/
    rho100_with_physics/run_20260624_003815/
```

Key artifacts:

```text
without_continuous_optimization/.../traditional_lambda_summary.csv
with_continuous_optimization/.../best_by_stage_lambda.csv
with_continuous_optimization/.../final_topologies_by_lambda.csv
with_continuous_optimization/.../alpha_consistency_diagnostics.csv
with_continuous_optimization/.../plots/traditional_lambda_objective_comparison.png
```

Current implementation audit:

```text
there are no artificial or auxiliary decision-variable buses

decisions directly modify existing bus features:
  Pg = Pg_base + Delta_Pg
  Pd = alpha * Pd_base
  Qd = alpha * Qd_base

the loaded IEEE-30 tensor uses a saved 50% random six-feature mask
```

The mask is applied after decision-variable injection. In the actual GNN
scenario:

```text
Pg is masked at selected generator bus 1
Pd is masked at selected load buses 6, 20, 11, and 29
Qd is masked at selected load bus 18
```

Every selected alpha bus therefore has either its commanded `Pd` or commanded
`Qd` hidden before inference. The wrapper subsequently returns predictions for
all `Pd`, `Qd`, `Pg`, `Qg`, `Vm`, and `Va` channels without restoring the
controlled values. This is now considered a methodological defect in the
current continuous experiment, not merely a weak bus-selection heuristic.

Revised design principle:

```text
controlled quantities:
  persist in the graph input and remain clamped in the evaluated output state

uncontrolled/state quantities:
  supplied by GridFM prediction
```

Artificial control buses may be explored later, but none exist currently. The
first correction should use intervention-aware masking and explicit clamping on
the existing physical buses.

## June 22, 2026 - Repository-Local Virtual Environment Cleanup

Checked the desktop PowerShell environment after the laptop-side `.venv`
handoff. The repository-local `.venv` was not a portable experiment dependency:
its `pyvenv.cfg` pointed to the laptop-style interpreter path
`C:\Users\caleb\AppData\Local\Programs\Python\Python311\python.exe`, which does
not exist on this desktop checkout. The `.venv` directory was about 2.49 GB and
was removed, along with the local editable-install artifact
`gridfm_graphkit.egg-info`.

The desktop Anaconda Python is the active workflow interpreter:

```text
C:\Users\Caleb Lu\anaconda3\python.exe
Python 3.12.11
```

Verification with the active Python:

```text
torch import ok: 2.10.0+cpu
lightning import ok: 2.6.1
GridFM runtime imports ok
Stage E/F implementation compile check passed
Stage F and physics case-study --help commands resolved
gurobipy import ok: 12.0.0
```

`gridfm-datakit` is not installed in the desktop Anaconda environment, but the
current wildfire Stage E/F runner imports and GridFM runtime imports tested
above do not require it directly. It remains a declared package dependency in
`pyproject.toml` for the broader graphkit package.

Important Gurobi interpretation:

- The Codex tool process still fails a tiny Gurobi solve with a username
  mismatch because it runs as `codexsandboxoffline`.
- This is independent of `.venv`; a virtual environment does not fix the
  Gurobi license/user binding.
- Full Stage E/F Gurobi result generation should be launched from the licensed
  desktop or laptop user shell for that device.
- Canonical handoff commands should use `python -m ...` from the active
  machine environment, not `.\.venv\Scripts\python.exe`.

Added `.mplconfig/` to `.gitignore` so Matplotlib font-cache state can remain
local if future runs set `MPLCONFIGDIR` inside the workspace.

## June 22, 2026 - Stage F Results Regenerated After OneDrive Placeholder Failure

The prior Stage F result folders under:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f/
```

had OneDrive placeholder/stale-file failures. Files such as
`corrected_selection_by_alpha_objective.csv`,
`fixed_control_best_among_same_candidates.csv`, and the old
`optimizer_traces/candidate_*.csv` entries could be enumerated by PowerShell
but could not be opened with `Get-Content`, `Import-Csv`, or `Test-Path`.

The broken Stage F tree was isolated as:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f_broken_onedrive_20260621_213958/
```

Fresh Stage F decision-quality results were regenerated from the licensed
desktop PowerShell/Gurobi environment:

```powershell
python -m experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality --models gnn
```

Fresh output:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f/decision_quality_analysis/run_20260621_214014/
```

The continuous topology pilot initially failed when writing
`optimizer_traces/candidate_000_trace.csv` because the full OneDrive path was
too long for normal Windows file opening. The shared reporting writer was
updated to use Windows long-path form for CSV/JSON writes, and the continuous
pilot trace folder was shortened from `optimizer_traces/` to `traces/` so the
generated trace files can also be opened normally.

Fresh continuous pilot command:

```powershell
python -m experiments.test.wildfire_tests.stage_f_decision_quality.run_s1_risk_priority_continuous_topology_pilot --model gnn --evaluation-budget 100
```

Fresh output:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f/continuous_topology_pilot/s1_risk_priority_k2/run_20260621_215238/
```

Validation:

```text
decision_quality_analysis/run_20260621_214014/stage_e_best_by_lambda.csv readable
continuous_topology_pilot/.../run_20260621_215238/continuous_best_decision.csv readable
continuous_topology_pilot/.../run_20260621_215238/traces/candidate_000_trace.csv readable
continuous best topology remains [22, 23]
continuous optimized_J_true = 0.21945875520161004
```

Intermediate failed/long-trace reruns under the fresh `stage_f` tree were
removed. The isolated broken OneDrive folder still resisted full deletion from
PowerShell because of inaccessible placeholder entries; it should be removed
manually from Explorer/OneDrive after sync settles if it remains visible.

Portable machine setup guidance:

- This repository should not contain a committed or handoff-required virtual
  environment. A repo-local `.venv` may be created temporarily by a developer,
  but it is local machine state and can be deleted after use.
- On a laptop or any new machine, first run a one-time environment check from
  the repository root using that machine's intended Python:

```powershell
python --version
python -c "import torch, lightning, gurobipy; import gridfm_graphkit; print('core imports ok')"
python -c "import gurobipy as gp; m=gp.Model(); print('gurobi license ok')"
python -m experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality --help
```

- If these checks pass, do not create a new repo-local `.venv`; use that active
  Python environment for wildfire commands.
- If imports are missing, create or repair a machine-local environment outside
  the repository, for example a Conda environment:

```powershell
conda create -n gridfm-wildfire python=3.12
conda activate gridfm-wildfire
python -m pip install -e ".[test]"
python -m pip install gurobipy
```

- After installing, rerun the one-time checks above. If a temporary repo-local
  `.venv` was created only for setup/debugging, it can be deleted once a clean
  machine-local environment works.
- The research methodology begins after the environment and Gurobi license
  checks pass. Missing imports are an environment setup issue, not a change to
  the Stage E/F optimization methodology.
- On Windows/OneDrive paths, deeply nested result files can be valid but fail
  ordinary reads/writes because the absolute path is too long. Without moving
  files, map the repository to a short temporary drive before running or
  inspecting deep results:

```powershell
subst G: "C:\path\to\GridFM-graphkit"
G:
python -m experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality --help
```

Remove the mapping with:

```powershell
subst G: /D
```

## Next Research Roadmap - Immediate Continuation

Stop point for this work session: Stage E Gurobi proxy-master, K-constrained
comparison, unconstrained 100-topology study, and unconstrained lambda-sweep
Pareto frontier are implemented and documented for the IEEE-30/small-grid
setting.

The next immediate step is to use the Stage E methodology and Pareto-front
results to design hypotheses for 3-5 targeted scenarios. For each scenario,
test the optimal Stage E/Gurobi/GridFM solution against our expected physical
and operational understanding. This is the major checkpoint for evaluating
model decision quality, not just numerical objective value.

After the 3-5 scenario hypothesis tests, immediately revisit the
physics-infeasibility and within-topology continuous-optimization question.
This is the next checkpoint before comparing against external heuristic
benchmarks. The goal is to decide whether physics infeasibility should remain a
diagnostic, become part of the objective, or require a different recourse design
before broader comparisons.

After the physics-infeasibility / continuous-recourse checkpoint, continue
decision-quality evaluation by comparing Stage E decisions against the TH and AH
heuristic methods:

```text
TH: transmission-line / risk-threshold heuristic
AH: area / risk-threshold heuristic
```

After heuristic comparison, evaluate solution reliability using the SC-OPS /
security-constrained methodology: find solutions that preserve contingency
scenarios under a chosen threshold. This should test whether promising
wildfire/load-shed tradeoff solutions remain acceptable under
contingency-aware operation.

These steps are the major small-grid evaluation sequence:

```text
1. Stage E + Pareto-front scenario hypotheses, 3-5 scenarios
2. Test optimal solutions against expected model/physics behavior
3. Revisit physics infeasibility and within-topology continuous optimization
4. Compare decision quality with TH and AH heuristics
5. Evaluate reliability with SC-OPS / security-constrained methodology
6. Then expand the methodology to larger grids for computational efficiency
   and economic-impact analysis
```

Current stop point update, June 19, 2026: pause Stage F decision-quality
scenario work here. The next immediate work session should pick up with the
physics-infeasibility and continuous-recourse checkpoint, then re-evaluate
decision quality before moving on to additional Stage F tasks such as TH/AH
heuristic benchmarks and SC-OPS/security-constrained analysis.

## June 19, 2026 - Stage F Decision-Quality Scenario Suite Corrected

Added and corrected Stage F under:

```text
experiments/test/wildfire_tests/stage_f_decision_quality/
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f/decision_quality_analysis/
```

Stage F is the current decision-quality scenario phase. It compares Stage D
exhaustive `k<=2` against Stage E constrained Gurobi `K=2` for five synthetic
IEEE-30 environmental-risk scenarios and three lambda cases. It remains a
fixed-control `u_base` study: no SciPy recourse, no alpha optimization, and no
physics-infeasibility objective.

Important correction: the first full Stage F run
`run_20260619_000604` was deleted because it inherited candidate groups from
the auto artifacts. That allowed line IDs such as `14` and `30` to be selected,
which were not in the planned fixed `t0p30` scenario candidate set.

The corrected Stage F implementation now uses the explicit fixed candidate set:

```text
[2,5,6,8,10,12,16,18,19,20,22,23,25,26,27,32,33,35,36,37,38,40,47,50,51,72,74,77,79,88,91,97,101]
```

The true wildfire-risk evaluator now uses all IEEE-30 line IDs:

```text
R_true = sum_{l in L} z_l * p_env_l * loading_l(u_base, z)^2
R_base_s = no-shutoff R_true under each scenario-specific p_env map
R_norm = R_true / R_base_s
```

Only the fixed candidate set is available for shutoff decisions. Non-candidate
lines remain energized in `z` but are included in the all-line risk sum. Each
scenario assigns `p_env=1.0` to target high-risk lines and `p_env=0.05` to all
other line IDs.

Corrected full run:

```text
ieee_30_stage_a_to_i_results/stage_f/decision_quality_analysis/run_20260619_004038/
```

Validation for the corrected run:

```text
Stage D rows: 2810 = 5 scenarios * 562 k<=2 topologies
Stage E rows: 1500 = 5 scenarios * 3 lambda cases * 100 candidates
Stage E status: 1500/1500 ok
risk_scope: all_lines
num_risk_lines: 110
selected IDs subset of fixed candidate set: yes
deleted contaminated original run_20260619_000604
deleted smoke run_20260619_003915
```

Replacement decision-topology figures were generated under:

```text
run_20260619_004038/plots/decision_topology/
```

## June 19, 2026 - S1 Risk-Priority Continuous Topology Pilot

Added an isolated pilot runner:

```text
experiments/test/wildfire_tests/stage_f_decision_quality/run_s1_risk_priority_continuous_topology_pilot.py
```

This is not yet integrated into the main Stage F workflow. It tests one case:

```text
model = gnn
scenario = S1_low_impact_high_risk
lambda = risk_priority = (0.8, 0.2)
K = 2
Gurobi topology proposals = 100
continuous controls = [Delta_Pg, alpha]
```

The pilot keeps Gurobi as the topology-only proxy master. For each proposed
topology, it runs SciPy over the reduced continuous controls and evaluates the
corrected true objective:

```text
J_true(u,z) = 0.8 * R_norm(u,z) + 0.2 * L_shed(alpha)
R_norm(u,z) = [sum_{l in L} z_l * p_env_l * loading_l(u,z)^2] / R_base,S1
```

The true objective excludes `I_l`, `c_l`, impact, and consequence scores.
Those terms remain allowed only in the Gurobi proxy proposal step.

Full pilot run:

```text
ieee_30_stage_a_to_i_results/stage_f/continuous_topology_pilot/s1_risk_priority_k2/run_20260619_011347/
```

Primary pilot artifacts:

```text
continuous_candidate_evaluations.csv
continuous_best_decision.csv
continuous_best_successful_decision.csv
fixed_control_best_among_same_candidates.csv
optimizer_traces/
metadata.json
```

Result:

```text
best optimized shutoff = [22, 23]
optimized R_norm = 0.274324
optimized L_shed = 0.000000
optimized J_true = 0.219459
R_base,S1 = 47.297886
runtime_seconds = 1437.68
```

SciPy mostly found `u_base` to already be locally optimal under this pilot
objective: the best solution had `max_abs_delta_pg=0`, `mean_alpha=1`, and
`min_alpha=1`. The largest observed improvement from continuous recourse over
the fixed continuous objective was about `2.94e-6`. This suggests that, for this
S1 risk-priority pilot, the topology ranking change comes from using the
alpha-based continuous objective rather than from meaningful continuous-control
movement. This should be investigated before integrating continuous recourse
into the main Stage F results.

Follow-up trace after the pilot confirmed the SciPy inner objective did use the
alpha-based formulation:

```text
L_shed_alpha = decision_vector.load_shedding(u)
J_true = 0.8 * R_norm + 0.2 * L_shed_alpha
```

It did not use `demand_weighted_load_shed_from_prediction(...)` inside the
SciPy objective. The saved `optimized_J_true` equals the reconstructed alpha
objective to numerical precision. The pilot does not save full per-bus
`alpha_values_best` or a GridFM-predicted load-shed diagnostic for every
candidate. It does save enough to reconstruct the selected topology under the
alpha objective:

```text
corrected_selection_by_alpha_objective.csv
old_selected_topology = 22,23
new_selected_topology = 22,23
requires_new_inner_optimization = false
```

Before adding AH/TH benchmark comparisons or SC-OPS, revisit whether the main
Stage F fixed-control decision-quality conclusions should be re-evaluated with:

```text
1. physics infeasibility diagnostics/objective treatment,
2. source-less island diagnostics,
3. a finalized continuous-recourse load-service formulation,
4. apples-to-apples comparison between fixed-control and recourse objectives.
```

## June 13, 2026 - Stage E Unconstrained Topology-Budget Study

Added a separate Stage E variant runner:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_stage_e_gurobi_gridfm_unconstrained.py
```

This is a tentative study and is not a Stage D comparison because the `K`
cardinality constraint is removed. The Gurobi proxy master can de-energize any
number of candidate lines and uses exact no-good cuts to produce unique
proxy-ranked topologies. GridFM remains the true evaluator and the objective is
unchanged:

```text
J_true = lambda_R * R_norm + lambda_L * L_shed
R_true = sum_{l in C} z_l * p_env_l * loading_l^2
```

Outputs are isolated from the K-constrained Stage E runs:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/unconstrained/
```

Run command used:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm_unconstrained --models gps --cases auto_env largest_group_high --lambda-cases risk_priority balanced load_priority --evaluation-budget 100
```

The sandbox Gurobi user still fails with a license username mismatch, so the
full 100-topology study was rerun under the licensed Windows user. The first
longer output path under `stage_e/gurobi_gridfm/unconstrained_budget/` exceeded
Windows path limits for `gurobi_candidate_evaluations.csv`, so the final study
root was shortened to `stage_e/unconstrained/`.

Validation:

```text
6/6 runs ok
100 evaluated topologies per run
0 duplicate y_vector rows per run
figures/unconstrained_objective_trace.png written per run
figures/ieee30_network_changes.png written per run
```

Best results:

```text
auto_env, risk_priority: [18, 23, 27, 32], R_norm=0.060020, L_shed=0.448741, J=0.137764
auto_env, balanced:      [18, 23, 27],     R_norm=0.063963, L_shed=0.466447, J=0.265205
auto_env, load_priority: [],               R_norm=1.000000, L_shed=0.000000, J=0.200000

largest_group_high, risk_priority: [18, 23, 27, 32], R_norm=0.056230, L_shed=0.448741, J=0.134732
largest_group_high, balanced:      [18, 23, 27],     R_norm=0.060074, L_shed=0.466447, J=0.263260
largest_group_high, load_priority: [],               R_norm=1.000000, L_shed=0.000000, J=0.200000
```

The objective-trace figure title records the lambda-weighted objective and the
final best topology's wildfire/load-shed tradeoff.

## June 13, 2026 - Stage E Unconstrained Frontier Lambda Sweep

Added a frontier-specific runner:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_stage_e_unconstrained_frontier.py
```

This study keeps the unconstrained topology-budget methodology, sweeps
`lambda_R` from `0.00` to `1.00` in increments of `0.05`, sets
`lambda_L = 1 - lambda_R`, and evaluates 100 unique no-good-cut Gurobi proxy
topologies per lambda with GridFM. It constructs the nondominated frontier from
the union of evaluated topologies using:

```text
minimize true_R_norm
minimize true_L_shed
```

Run command:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_unconstrained_frontier --models gps --cases auto_env largest_group_high --lambda-step 0.05 --evaluation-budget 100
```

This was run outside the sandbox under the licensed Gurobi Windows user.

Outputs:

```text
ieee_30_stage_a_to_i_results/stage_e/unconstrained_frontier/
```

Key artifacts:

```text
unconstrained_frontier_summary.csv
unconstrained_frontier_summary.json
all_candidate_points.csv
unique_candidate_points.csv
pareto_frontier_points.csv
figures/unconstrained_pareto_frontier_scatter.png
```

Validation:

```text
42/42 lambda-case runs ok
4,200 total evaluated candidate rows
1,009 unique topologies across case/lambda runs
59 nondominated frontier points
0 duplicate y_vector rows within each run
```

## June 13, 2026 - Stage E Gurobi Master + GridFM Evaluator Added

Stage E was added as an additive experiment under:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/
```

Stage D limited enumeration remains intact and is still the comparison method.
Stage E writes to the regenerated result surface:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/
```

Stage E uses a Gurobi proxy master only as a topology candidate generator.
Gurobi does not optimize through GridFM. GridFM remains the true evaluator of
each proposed post-topology operating state.

The v1 Stage E evaluator preserves current Stage D behavior: fixed-control
GridFM evaluation at `decision_vector.u_base`, with no inner SciPy continuous
optimization.

Formulation update:

- The old `I_l` term is now interpreted as `c_l`, a single-line load-service
  consequence score.
- `c_l` is used only in the Gurobi proxy load-consequence term:

```text
L_hat(y) = sum_l c_l y_l
```

- The true GridFM-evaluated wildfire exposure no longer multiplies by `I_l` or
  `c_l`.
- The primary Stage E true risk uses the same selected candidate/group scope as
  Stage D:

```text
R_true = sum_{l in C} z_l * p_env_l * loading_l(u_base, z)^2
R_base_true = sum_{l in C} p_env_l * loading_base_l^2
R_norm = R_true / R_base_true
```

`R_norm` is baseline-relative and is not bounded to `[0, 1]`. It may exceed
`1.0` if a proposed topology increases loading-dependent wildfire exposure.
`L_shed` remains demand-weighted and bounded to `[0, 1]` under the clipped
service-fraction calculation.

Stage E v1 lambda cases:

```text
risk_priority = (0.8, 0.2)
balanced      = (0.5, 0.5)
load_priority = (0.2, 0.8)
```

For v1, master lambdas and true-evaluation lambdas are equal and saved
explicitly. `lambda_P = 0.0` because no active AC infeasibility penalty is
integrated into Stage D/Stage E topology evaluation yet. Pareto-front
computation is intentionally deferred; all evaluated candidate points are
saved for later analysis.

Important Stage E artifacts per run:

```text
baseline_metrics.json
line_consequence_scores.csv
candidate_line_risk_scores.csv
gurobi_candidate_evaluations.csv
gurobi_best_decision.csv
optimized_deenergization_decisions.csv
objective_trace.csv
optimization_summary.json
visualization_summary.json
```

Aggregate Stage E artifacts:

```text
stage_e_gurobi_gridfm_summary.csv
stage_e_gurobi_gridfm_summary.json
proxy_vs_true_ranking_summary.csv
runtime_summary.csv
```

Validation targets:

```powershell
python -m compileall experiments/test/wildfire_tests tests/test_wildfire_stage_e_gurobi.py
pytest tests/test_wildfire_stage_d_deenergization.py tests/test_wildfire_stage_e_gurobi.py -q
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm --models gps --cases auto_env --lambda-cases balanced --max-deenergized-lines 1 --evaluation-budget 3
```

Verification completed during implementation:

```powershell
python -m compileall experiments/test/wildfire_tests tests/test_wildfire_stage_e_gurobi.py
pytest tests/test_wildfire_stage_d_deenergization.py tests/test_wildfire_stage_e_gurobi.py -q
pytest tests/test_wildfire_stage_c_psps.py tests/test_wildfire_stage_d_deenergization.py tests/test_wildfire_stage_e_gurobi.py -q
```

Results:

```text
Compile check passed.
Stage D/E focused tests: 16 passed, 1 skipped, 3 external warnings.
Stage C/D/E focused tests: 23 passed, 1 skipped, 3 external warnings.
```

Stage E smoke command:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm --models gps --cases auto_env --lambda-cases balanced --max-deenergized-lines 1 --evaluation-budget 3
```

The sandboxed run reached Gurobi but failed with the expected academic-license
username mismatch for `codexsandboxoffline`. The same command was rerun outside
the sandbox under the licensed Windows user and completed successfully.

Smoke output:

```text
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/stage_e_gurobi_gridfm_summary.csv
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/t0p30/k1/gps/auto/bal/run_20260613_191953/
```

Smoke result summary:

```text
status: ok
model: gps
case: auto_env
lambda_case: balanced
K: 1
evaluation_budget: 3
num_candidate_lines: 33
num_evaluated_candidates: 3
num_gridfm_calls: 2
best_deenergized_line_ids: [23]
best_true_R_norm: 0.18972482142823735
best_true_L_shed: 0.42697785716434883
best_true_objective: 0.3083513392962931
baseline_R_raw_new: 117.17394541976515
```

## June 13, 2026 - Stage E Real Experiment And Revised Stage D Comparison

Added a non-destructive analysis helper:

```text
experiments/test/wildfire_tests/analysis/run_stage_e_real_experiment_analysis.py
```

The helper rebuilds the same Stage D context/candidate set, enumerates feasible
topology subsets, evaluates them with the same GridFM topology path, and
rescales them under the revised Stage E true objective:

```text
R_true = sum_{l in C} z_l * p_env_l * loading_l^2
R_norm = R_true / R_base_true
J_true = lambda_R * R_norm + lambda_L * L_shed
```

It does not modify or overwrite Stage D outputs. It writes comparison artifacts
under:

```text
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/
```

Stage E smoke command:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm --models gps --cases auto_env --lambda-cases balanced --max-deenergized-lines 1 --proxy-type env_loading_base --evaluation-budget 3
```

As before, the sandboxed smoke wrote a failed Gurobi summary because the
sandbox user does not match the academic license. The exact command was rerun
outside the sandbox under the licensed Windows user and completed successfully.

Full Stage E command:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm --models gps --cases auto_env largest_group_high --lambda-cases risk_priority balanced load_priority --max-deenergized-lines 1 2 --proxy-type env_loading_base --evaluation-budget 20
```

Full Stage E result:

```text
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/stage_e_gurobi_gridfm_summary.csv
num_runs: 12
num_successful_runs: 12
```

Revised Stage D comparison command:

```powershell
python -m experiments.test.wildfire_tests.analysis.run_stage_e_real_experiment_analysis --models gps --cases auto_env largest_group_high --max-deenergized-lines 1 2
```

Comparison outputs:

```text
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/stage_e_real_experiment_summary.csv
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/stage_e_real_experiment_summary.json
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/stage_e_vs_stage_d_comparison.csv
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/proxy_vs_true_alignment.csv
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/best_decisions_by_lambda.csv
ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/experiment_summaries/runtime_and_call_count_summary.csv
```

Key comparison result:

```text
Matched settings: 12
Stage E matched revised Stage D best topology: 12 / 12
Max absolute objective gap vs revised Stage D: 4.524047780506725e-16
```

Candidate evaluation and search-call comparison:

```text
K=1:
  Stage E candidate evaluations: 20
  revised Stage D candidate evaluations: 34
  Stage E search GridFM calls: 19
  revised Stage D search GridFM calls: 33

K=2:
  Stage E candidate evaluations: 20
  revised Stage D candidate evaluations: 562
  Stage E search GridFM calls: 19 or 20 depending on no-action reuse
  revised Stage D search GridFM calls: 561
```

The analysis helper reports `setup_gridfm_calls = 221` per rebuilt context,
estimated as one baseline call plus 110 automatic impact counterfactual calls
plus 110 fixed `c_l` consequence calls. Search calls count non-baseline
candidate topology evaluations only.

Best Stage E decisions:

```text
auto_env, K=1:
  risk_priority: [23]
  balanced: [23]
  load_priority: []

auto_env, K=2:
  risk_priority: [23, 27]
  balanced: [23, 27]
  load_priority: []

largest_group_high, K=1:
  risk_priority: [23]
  balanced: [23]
  load_priority: []

largest_group_high, K=2:
  risk_priority: [23, 27]
  balanced: [23, 27]
  load_priority: []
```

Proxy-vs-true alignment:

```text
K=1: proxy-best was true-best in 6 / 6 runs
K=2: proxy-best was true-best in 2 / 6 runs
Spearman proxy/true objective range across runs: 0.3533834586466165 to 0.8165413533834586
```

Interpretation:

- K=1 behavior validation is clean: Stage E recovers revised enumeration best.
- K=2 begins the efficiency story: Stage E recovers revised enumeration best
  using 20 candidate evaluations rather than 562.
- The proxy ranking is imperfect for K=2, but the no-good candidate loop still
  reaches the revised true best within the 20-evaluation budget.

Validation after the run:

```powershell
python -m compileall experiments/test/wildfire_tests/analysis/run_stage_e_real_experiment_analysis.py
pytest tests/test_wildfire_stage_e_gurobi.py -q
```

Results:

```text
Compile check passed.
Stage E tests: 9 passed, 1 skipped, 3 external warnings.
Additional artifact validation passed:
  12 Stage E rows, all ok
  no duplicate y_vector rows within each run
  baseline_R_norm = 1.0 in every baseline_metrics.json
  comparison rows = 12
  same_best_topology true for every comparison row
```

## June 12, 2026 - Refactor Complete And Stage E Gurobi Direction

The wildfire repository refactor is now complete enough to treat
`experiments/test/wildfire_tests/` as the active standalone research harness.
The old `experiments/test/wildfire_initial_tests/` folder was checked for active
dependencies and deleted after confirming the refactored workflow no longer
depends on it. Historical/older methodology artifacts were preserved under the
new harness rather than kept in the old folder.

Current result organization:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/
  stage_a/connected_corridor/
  stage_b/multistart/
  stage_b/multi_group/
  stage_c/psps_baseline/
  stage_d/deenergization/
  generated_configs/

experiments/test/wildfire_tests/methodology_testing_results/
```

Stage C and Stage D now have both the earlier/original lambda outputs and the
new GNN-only configured-lambda outputs. The configured lambda cases are:

```text
risk_leaning:    lambda_R = 0.9, lambda_L = 0.1
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.1, lambda_L = 0.9
```

Latest generated configured-lambda outputs:

```text
ieee_30_stage_a_to_i_results/stage_c/psps_baseline/gnn_lambdas_configured/
ieee_30_stage_a_to_i_results/stage_d/deenergization/gnn_lambdas_configured/
```

Verification status for these configured-lambda outputs:

- Stage C summary has 6 rows: 3 lambda cases for `auto_env` and
  `largest_group_high`, all `ok`.
- Stage D summary has 6 rows: 3 lambda cases for `auto_env` and
  `largest_group_high`, all `ok`.
- Lambda values in summaries/configs are correct.
- Stage D objective traces carry the correct lambda values.
- Stage C behavior plots read `lambda_R` and `lambda_L` from
  `optimization_summary.json`; those summaries were verified correct.
- Temporary `W:` path aliases used to avoid Windows path-length issues were
  removed from the generated metadata.
- JSON metadata in those configured-lambda result folders was rechecked for
  parse validity after cleanup.

Important interpretation from the latest Stage D configured-lambda discussion:

- `service_leaning` with `lambda_L = 0.9` often returns the no-action topology.
- In Stage D, the no-action baseline has `R_norm = 1.0`, `L_norm = 0.0`, and
  objective `0.1` for `lambda_R = 0.1`, `lambda_L = 0.9`.
- Candidate de-energization topologies can reduce wildfire risk, but if they
  shed enough load then `0.9 * L_norm` dominates and the objective rises above
  the no-action baseline. Those larger objective values are evaluated
  candidates, not accepted optimized solutions.

Stage A/B convergence interpretation:

- Stage A/B use continuous SciPy/L-BFGS-B inner optimization over the reduced
  continuous controls.
- Most Stage A/B runs terminate successfully with optimizer convergence
  messages, but convergence does not imply large objective improvement.
- Stage A often converges to the initial/no-change point.
- Stage B multistart generally converges to local surrogate-optimal points;
  the largest selected run-level Stage B objective reduction observed in the
  current results is approximately `0.0411256695`, from `0.9396712495` to
  `0.8985455800` in:

```text
ieee_30_stage_a_to_i_results/stage_b/multi_group/threshold_0p30/risk/gps/multistart_gps_20260611_220154/
```

Research direction after deliberation:

The next major formulation change is to add a new Stage E that uses Gurobi as
an outer topology-decision generator while keeping GridFM as the true evaluator
of each proposed topology. This should be framed as a Gurobi-guided GridFM
evaluation loop, not as Gurobi directly optimizing through the GridFM/PyTorch
model.

Intended Stage E loop:

```text
Gurobi master proposes line de-energization vector y
-> modify grid topology
-> run GridFM on modified topology
-> compute true wildfire risk, served load, and feasibility penalties
-> update incumbent best solution
-> add a no-good cut to prevent exact repeated topology evaluations
-> repeat under an evaluation budget
```

Reasoning:

- `scipy.minimize` is appropriate for the existing continuous inner controls,
  but not for binary topology variables.
- Directly embedding GridFM inside Gurobi is not feasible in the current
  method because Gurobi requires algebraic variables, constraints, and
  objective expressions, while GridFM is a black-box neural evaluator.
- The correct near-term role for Gurobi is therefore a master problem that
  proposes promising binary topologies using proxy terms. GridFM then evaluates
  the true objective for each proposed topology.

Formulation revision for Stage E:

Move the current GridFM counterfactual line-outage consequence score out of the
inner wildfire-risk term. The current consequence score measures grid service
loss from removing a line, so it is better interpreted as a service-consequence
proxy than as wildfire harm.

Use the true GridFM-evaluated wildfire exposure term:

```text
R_true(u, z) = sum_l p_env,l * z_l * loading_l(u, z)^2
```

where:

```text
p_env,l = environmental wildfire risk for line l
z_l = 1 if line l is energized, 0 if de-energized
loading_l(u, z)^2 = GridFM-predicted loading/stress effect
```

Keep demand-weighted load shedding as a separate objective term:

```text
L_true(u, z) = demand-weighted fraction of load not served
```

The true evaluated objective should be:

```text
J_true(u, z) = lambda_R * R_true(u, z)
             + lambda_L * L_true(u, z)
             + lambda_P * infeasibility_penalty(u, z)
```

This avoids double-counting grid service consequence inside both the wildfire
risk term and the load-shedding term. The research story becomes wildfire
exposure reduction versus load-service preservation.

Use the previous impact/consequence score only in the Gurobi master as a proxy
load-service consequence score:

```text
c_l = (S_D_base - S_D_outage,l) / S_D_base
```

where:

```text
S_D_base = total served demand in the all-energized baseline
S_D_outage,l = total served demand when only line l is removed
```

Interpretation: `c_l` estimates how damaging it is to remove line `l` from a
load-service perspective. It is a first-order proxy for Gurobi candidate
generation only; GridFM remains responsible for evaluating the actual multi-line
topology effect.

Candidate Gurobi master formulations:

Option 1, weighted proxy objective:

```text
min_y lambda_R * R_hat(y) + lambda_L * L_hat(y)
```

with:

```text
R_hat(y) = normalized environmental risk remaining after shutoffs
L_hat(y) = sum_l c_l y_l
```

Option 2, epsilon-constraint formulation:

```text
min_y R_hat(y)
subject to:
    L_hat(y) <= epsilon
    sum_l y_l <= K
    y_l in {0, 1}
```

Option 2 may be cleaner and more PSPS-aligned because it minimizes proxy
wildfire exposure while enforcing an estimated service-impact budget.

Immediate Stage E implementation plan:

1. Compute or load environmental wildfire risk values `p_env,l`.
2. Compute baseline GridFM state under the all-energized topology.
3. Compute single-line consequence scores `c_l` by removing one candidate line
   at a time and measuring normalized demand-weighted served-load reduction.
4. Build a Gurobi master problem over binary line de-energization variables
   `y_l`.
5. Add constraints:
   `sum_l y_l <= K`, `y_l in {0,1}`, candidate lines restricted to selected
   high-risk group, and no-good cuts for already evaluated topologies.
6. Use the master problem to propose candidate shutoff vectors.
7. For each proposed topology, modify topology, run GridFM, compute true
   loading-squared wildfire risk, compute true demand-weighted load shedding,
   compute infeasibility penalty, and compute true objective.
8. Track best topology found, best true objective value, `R_true`, `L_true`,
   infeasibility penalty, number of GridFM calls, and runtime.
9. Compare the Gurobi-guided method against limited enumeration on small cases
   to verify that it can recover good decisions with fewer candidate
   evaluations.

Near-term behavior-validation experiments should test:

- high wildfire risk / low load-consequence line
- high wildfire risk / high load-consequence line
- low wildfire risk / high load-consequence line
- multi-line rerouting case
- wildfire-priority case
- load-service-priority case
- balanced case

Baseline comparison plan after Stage E works:

```text
AH: area/risk-threshold heuristic
TH: transmission-line/risk-threshold heuristic
limited enumeration baseline
Gurobi-master + GridFM evaluator
```

Metrics:

```text
planned wildfire risk
planned load shedding
true GridFM objective
runtime
number of GridFM evaluations
objective improvement per GridFM call
```

Reliability extension after the base Stage E behavior works:

For each planned PSPS topology, evaluate the planned topology with GridFM, then
for each additional N-1 contingency line remove that line as an unexpected
outage, run GridFM again, and record worst-case/average additional load
shedding plus any contingency violations. This is a later SC-OPS-inspired
extension, not the first Stage E implementation target.

Central intended contribution:

```text
A GridFM-enabled predict-then-optimize framework for PSPS-style wildfire-risk
mitigation, where binary de-energization decisions are proposed by a tractable
Gurobi master problem and evaluated by a pretrained surrogate power-flow model
using loading-dependent wildfire exposure, demand-weighted load shedding, and
feasibility penalties.
```

## Working Protocol

After every meaningful implementation or experiment change, update this file so
a new session can recover the current research state without needing the prior
chat. Keep this file aligned with the code, configs, tests, latest verified
commands, and current next steps. If a run or test changes the evidence base,
record the command and outcome here after verification.

As of May 28, 2026, the active state is the connected-corridor first pass:

- target workflow: `configs/connected_corridor_gnn.yaml` and
  `configs/connected_corridor_gps.yaml`
- target result layout for demand-weighted runs:
  `results/demand_weighted/connected_corridor/...`
- high-risk corridor: the manually selected, connected line IDs `[23, 32, 26]`
- visualization: `figures/ieee30_network_changes.png` plots the IEEE-30
  topology and highlights the synthetic high-risk corridor
- topology controls are not active; buses and lines stay energized
- wildfire risk now uses probability-times-consequence form with `z_l = 1`,
  synthetic `p_env_l`, relative loading squared, and a GridFM counterfactual
  line-outage consequence score
- load shedding term: modeled as a demand-weighted shed fraction,
  `sum_n (P_D,n / sum_m P_D,m) * (1 - alpha_n)`
- selected-load alpha is relaxed to `[0.0, 1.0]`
- connected-corridor outputs are generated as three tradeoff sets:
  `risk`, `balanced`, and `shed`

Important wording: the current network visual is topology-accurate, not
geographic. The selected lines are adjacent/connected in the IEEE-30 graph, but
the plotted coordinates come from a deterministic spring layout rather than
real geographic bus coordinates.

As of May 29, 2026, the simplified fixed-energization implementation has been
polished into a coherent first-pass pipeline: the corrected probability-times-
consequence wildfire risk term is implemented, decision-variable and frozen
consequence objective analyses are separated, baseline-started and grid-seeded
multistart optimizer behavior can be compared, three lambda tradeoff cases are
generated for both GNN and GPS, figures and machine-readable summaries are
written consistently, and focused tests pass. The next research phase should
evaluate how the optimization behavior changes when de-energization is included
as an explicit decision/control pathway rather than only as a counterfactual
line-outage consequence calculation.

Later on May 29, 2026, the optimized load-shedding term was changed from an
equal-bus shed-fraction sum divided by `N_bus` to a demand-weighted shed
fraction:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

This term is already normalized because the weights sum to 1 across positive
baseline demand, so current generated runs use
`load_shedding_normalizer = 1.0`. The previous equal-bus shedding sum and
unserved MW are retained as diagnostics.

## Why This Folder Exists

The research goal is to test a limited GridFM predict-then-optimize loop before claiming a full wildfire-resilience-aware OPF result.

The first research claim is:

```text
In a fixed IEEE-30 operating scenario with a synthetic high-risk wildfire corridor,
a GridFM-based predict-then-optimize loop can produce stable and interpretable
operating changes that reduce aggregate wildfire exposure while preserving most
served demand.
```

This is intentionally not a full ACOPF, topology-control, or wildfire physics claim.

## Current Active Folder

The active implementation is:

```text
experiments/test/wildfire_tests/
```

The active support files kept at `experiments/test/` are:

```text
experiments/test/__init__.py
experiments/test/pipeline_utils.py
experiments/test/scenario_data.py
experiments/test/neural_solver.py
experiments/test/pv_dispatch.py
experiments/test/overload_penalty.py
```

These support files are still needed for scenario loading, checkpoint loading, in-memory GridFM inference, and branch-loading reconstruction. Do not remove them unless `wildfire_initial_tests` is refactored to replace their functionality.

## Cleanup Completed

The following were removed from `experiments/test` because they belonged to earlier workflows and were no longer required by the current first-pass formulation:

```text
experiments/test/improved_optimization/
experiments/test/example_optimization.py
experiments/test/example_optimization_with_shedding.py
experiments/test/extended_dispatch_spec.py
experiments/test/load_shedding_spec.py
experiments/test/optimization.py
experiments/test/validation.py
experiments/test/wildfire_penalty.py
experiments/test/test_pipeline.py
experiments/test/test_pipeline_ieee30.py
experiments/test/test_gnn_vs_gps_shedding.py
experiments/test/ieee30_optimization_validation.ipynb
experiments/test/pf_node.csv
experiments/test/pf_edge.csv
```

The first-pass code was also refactored so it no longer imports `experiments.test.improved_optimization.wildfire_metrics`. Branch loading now goes through `experiments.test.wildfire_tests.gridfm_support.overload_penalty.OverloadPenaltyEvaluator`.

## Current Reduced Formulation

The full formulation in the research notes uses:

```text
u = (Pg, Qg, alpha_n, y_n, z_l)
```

The first-pass implementation uses:

```text
u_first = [Delta_Pg_selected, alpha_selected]
```

Fixed assumptions:

- `y_n = 1` for all buses
- `z_l = 1` for all lines
- `Qg` fixed at baseline
- no line shutoff
- no bus shutoff
- no mixed-integer optimization
- no hard AC feasibility constraints
- no external wildfire physics solver

Bounds:

- `Delta_Pg` in `[-5 MW, +5 MW]`
- `alpha` in `[0.00, 1.00]`

Objective, aligned with the current reduced formulation:

```text
J(u) = lambda_R * (R_group / R_baseline)
     + lambda_L * L_shed_weighted
```

Generator redispatch remains available as a control variable. Its normalized
movement is recorded in objective components and traces for interpretability,
but it is not part of the optimized scalar objective. This keeps the first pass
focused on the intended tradeoff between grouped wildfire exposure and unmet
demand service.

The grouped wildfire risk now follows the full formulation structure while
keeping line energization fixed:

```text
R_group(u) = sum_k group_weight_k * sum_{l in G_k}
             z_l * p_env_l * loading_l(u)^2 * I_l(u)
```

Current simplifications:

- `z_l = 1` for all lines; line de-energization is not a decision variable
- `p_env_l` is the configured synthetic environmental ignition score
- `loading_l` is reconstructed relative line loading, normalized by rating or
  the configured proxy rating before squaring
- `I_l(u)` is a counterfactual consequence score from removing line `l` from
  the GridFM graph under the current decision vector

The current consequence score is equal-weight service loss, not MW-weighted
load loss:

```text
I_l(u) = max(0, S_current(u) - S_outage_l(u)) / max(S_current(u), eps)
```

`S` is inferred from predicted `Pd` as clipped `Pd_pred / Pd_base`, summed with
equal weight per bus, and zero-load buses counted as fully served. This replaced
the previous constant `impact_l = 1.0` behavior on May 28, 2026.

`L_shed_weighted` is currently the demand-weighted load-shedding fraction over
the full bus vector:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

Non-selected buses have `alpha_n = 1`, so they contribute zero. This replaced
the equal-bus shed-fraction objective on May 29, 2026 so that shedding 50% of a
large-demand bus is more expensive than shedding 50% of a small-demand bus. The
term is already normalized by total baseline demand, so current generated runs
write `load_shedding_normalizer = 1.0` and
`load_shedding_metric = demand_weighted_fraction`. The old equal-bus sum and
unserved MW remain diagnostic outputs.

Current connected-corridor work uses three convex normalized objective tradeoff
sets:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

The `risk` set is the 0-to-1 equivalent of the previous 1000:1
risk-prioritized preference. The normalizers are written per run in
`objective_normalizers.json`.

## Risk Scenario

The risk scenario is synthetic:

- one line group named `high_risk_corridor`
- line group is manually configured as one connected corridor
- current connected corridor line IDs are `[23, 32, 26]`
- those scenario edges correspond to bus 6--8, bus 8--28, and bus 6--28
- selected group lines get weather/environment ignition probability proxy `1.0`
- non-selected lines get hazard `0.1`
- impact is now computed per evaluated decision as counterfactual equal-weight
  service loss under a one-line GridFM outage
- group weight is currently `1.0`

The previous `top_loaded` method selected the top 3 baseline-loaded line IDs,
which could produce a high-risk line group that was not physically connected in
the topology. The current default configs use `selection_method:
manual_connected` and validate that the selected line IDs form one connected
component before writing results.

Line risk:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l(u)
```

`z_l` is fixed to `1` in the current first pass.

Grouped risk:

```text
R_group = sum_k group_weight_k * sum_{l in G_k} risk_l
```

## Current Files In wildfire_initial_tests

Core modules:

```text
config.py
scenario.py
decision_vector.py
gridfm_runner.py
state_extraction.py
wildfire_scenario.py
wildfire_risk.py
objective.py
optimization_problem.py
validation.py
reporting.py
objective_analysis.py
ac_opf_experiment.py
```

Entry points:

```text
run_basic_case.py
run_stability_sweep.py
run_connected_corridor_tradeoffs.py
plot_optimization_behavior.py
plot_network_changes.py
objective_analysis.py
ac_opf_experiment.py
run_multistart_optimization.py
```

Configs:

```text
configs/basic_gps.yaml
configs/basic_gnn.yaml
configs/connected_corridor_gps.yaml
configs/connected_corridor_gnn.yaml
configs/stability_sweep.yaml
```

Documentation and artifacts:

```text
FIRST_PASS_SUMMARY.md
HISTORY.md
results/
```

## Tests

Focused tests live outside `experiments/test`:

```text
tests/test_wildfire_first_pass_scenario.py
tests/test_wildfire_first_pass_risk.py
tests/test_wildfire_first_pass_decision_vector.py
tests/test_wildfire_first_pass_objective.py
tests/test_wildfire_first_pass_basic_run.py
tests/test_wildfire_first_pass_objective_analysis.py
tests/test_wildfire_first_pass_multistart.py
```

Latest post-cleanup result:

```text
16 passed
```

Latest verification on May 29, 2026:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py tests/test_wildfire_first_pass_multistart.py -q
```

Result:

```text
16 passed, 3 external deprecation warnings
```

This check includes the counterfactual line-impact unit test, the frozen
objective-analysis sweep test, and the grid-seeded multistart seed-selection
test.

## Latest Connected-Corridor Runs

Fresh connected-corridor runs were produced on May 28, 2026 after changing the
wildfire consequence term from constant `impact_l = 1.0` to a counterfactual
equal-weight served-load-loss score from one-line GridFM outage predictions.

An attempted `--clear` regeneration could not remove one older OneDrive-locked
figure directory, so the successful May 28 regeneration was run without
`--clear`. The aggregate CSV/JSON summary points to the new May 28 run
directories, while at least one older locked result folder may remain on disk.

The three result sets are under:

```text
results/connected_corridor/risk/
results/connected_corridor/balanced/
results/connected_corridor/shed/
```

Aggregate machine-readable outputs:

```text
results/connected_corridor/connected_corridor_tradeoff_summary.csv
results/connected_corridor/connected_corridor_tradeoff_summary.json
```

Latest run summary after deleting and regenerating `results/connected_corridor`
with selected-corridor `p_env_l = 1.0`:

```text
risk/gnn:
  objective:      0.999001 -> 0.999001
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000
  mean alpha:     1.000000

risk/gps:
  objective:      0.999001 -> 0.999001
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000
  mean alpha:     1.000000

balanced/gnn:
  objective:      0.500000 -> 0.500000
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000
  note: optimizer reported ABNORMAL but returned the baseline decision

balanced/gps:
  objective:      0.500000 -> 0.500000
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730 -> 9.276730
  load shedding:  0.000000

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.693767 -> 2.693767
  load shedding:  0.000000
```

On May 15, 2026, the implementation was corrected to remove the previous
generator movement cost from the optimized scalar objective. On May 18, 2026,
the load penalty was changed from MW-weighted unserved demand to unweighted
load-shedding fraction sum, the objective weights were converted from
`1000.0`/`1.0` to convex tradeoff weights, and `alpha_min` was relaxed to
`0.0`. Generator movement is still recorded as a diagnostic, but the scalar
objective is now exactly the two-term wildfire-risk versus load-shedding
tradeoff.

On May 19, 2026, the optimization behavior plot methodology was clarified:
the fourth panel now plots the configured weights `lambda_R` and `lambda_L`,
not the realized weighted normalized terms. The realized terms remain in
`objective_trace.csv` for auditability.

On May 28, 2026, `GridFMRunner.predict_with_line_outage` and
`compute_counterfactual_line_impacts` were added so `impact` in the risk CSVs
is a decision-dependent counterfactual consequence score. The May 28 runs
returned the baseline decision for all tradeoff sets, which should be treated
as evidence that the current corrected risk term and selected controls do not
yet produce an accepted improvement, not as a final physical conclusion.

Also on May 28, 2026, `FIRST_PASS_SUMMARY.md` was updated to include the
current optimization problem, objective value definition, latest objective
values, decision variables, explicit bounds, fixed topology assumptions,
wildfire scenario constraints, prediction validity rules, and optimizer
settings.

Later on May 28, 2026, the selected-corridor weather/environment ignition
factor was changed from `5.0` to `1.0` in the first-pass configs so the
selected-line probability factor is probability-like rather than a large hazard
multiplier. The existing `results/connected_corridor` directory was deleted
successfully, and `run_connected_corridor_tradeoffs.py` regenerated the three
tradeoff sets. Group risks dropped by the expected factor, but all six runs
still returned the baseline decision.

Also on May 28, 2026, `objective_analysis.py` was added for frozen objective
sensitivity analysis. It performs one GridFM evaluation at a fixed decision,
freezes the prediction, decision vector, reconstructed loading, normalizers,
and all line impacts, then manually varies exactly one line consequence value
without rerunning GridFM or invoking the optimizer. The first analysis swept
`I_23(u)` from `0.0` to `1.0` using
`configs/connected_corridor_gps.yaml` and wrote:

```text
results/objective_analysis/line_23_impact_sweep_20260528_220439/
```

The frozen baseline values were:

```text
frozen grouped risk: 2.6937669151682497
risk normalizer:     2.6937669151682497
load normalizer:     30.0
lambda_R:            0.999001
lambda_L:            0.000999
base I_23(u):        0.03000268142041778
```

`FIRST_PASS_SUMMARY.md` was also expanded with a dedicated simplifications
section covering fixed energization, synthetic environmental probability,
single wildfire group, reduced decision vector, selected-load-only shedding,
equal-bus shedding/consequence metrics, surrogate line-outage impact,
reconstructed/proxy loading, missing hard AC feasibility constraints, absence
of full wildfire physics, static single-scenario scope, local optimizer
limitations, baseline-relative normalization, and limited current result
interpretation.

Later on May 28, 2026, `objective_analysis.py` was extended with a second,
separate decision-variable sweep mode. This is different from the frozen
consequence sweep above:

- the frozen consequence sweep freezes one GridFM evaluation and manually
  changes one line impact value, such as `I_23`, without rerunning GridFM
- the decision-variable sweep varies one actual optimizer variable at a time,
  reruns GridFM and the counterfactual line-impact calculations at each point,
  and still does not invoke the optimizer

The latest decision-variable sweep commands were:

```powershell
python experiments/test/wildfire_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_tests/configs/connected_corridor_gnn.yaml --num-points 21
```

They wrote:

```text
results/objective_analysis/decision_sweeps/decision_sweeps_gps_20260528_235031/
results/objective_analysis/decision_sweeps/decision_sweeps_gnn_20260528_235108/
```

In the decision-sweep figure, each subplot corresponds to one actual decision
variable while all other decision variables remain fixed at baseline. The
x-axis is the decision value: `Delta_Pg` sweeps from `-5 MW` to `+5 MW`, and
`alpha` sweeps from `0.0` to `1.0`. The left y-axis is scalar objective `J(u)`
as the black curve. The right y-axis is raw grouped wildfire risk `R_group(u)`
as the red curve. The CSV carries the audit columns needed to reconstruct the
objective: normalized risk, normalized load shedding, weighted risk term,
weighted load-shedding term, and corridor line impacts.

GPS decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_7_bus18 = 0.0
best objective: 0.8961604984016736
best grouped risk: 2.4163717542193073
best load shedding: 1.0
```

For this best GPS point, the objective change decomposes as:

```text
alpha_7_bus18:                  1.0 -> 0.0
R_group:                        2.6937669151682497 -> 2.4163717542193073
normalized R_group:             1.0 -> 0.8970233247030519
L_shed:                         0.0 -> 1.0
normalized L_shed:              0.0 -> 0.03333333333333333
risk objective term:            0.999001 -> 0.8961271984016735
load-shedding objective term:   0.0 -> 0.0000333
total objective:                0.999001 -> 0.8961604984016736
```

The objective changes only modestly in many plots because the current GPS
risk-prioritized config heavily favors risk:
`lambda_R = 0.999001`, `lambda_L = 0.000999`. Completely shedding one selected
bus adds only `0.000999 * (1/30) = 0.0000333` to the scalar objective. Thus,
unless the candidate decision meaningfully changes normalized grouped risk,
the total objective will stay visually close to baseline.

GNN decision sweep result:

```text
baseline objective: 0.999001
best one-variable point: alpha_5_bus11 = 0.0
best objective: 0.9953966468056175
best grouped risk: 9.24295077382007
best load shedding: 1.0
```

Interpretation: the decision sweeps show objective-improving one-variable
points, especially for GPS, even though the local L-BFGS-B tradeoff runs
returned the baseline decision. This suggests the no-movement result is more
likely due to local-search, finite-difference, or nonsmooth surrogate behavior
than due to a total absence of objective signal. The GPS `alpha_7_bus18` sweep
is also nonmonotonic: small reductions from `alpha = 1.0` initially make the
objective worse, while larger reductions eventually improve it, so a local
optimizer initialized at baseline can rationally stay near baseline even though
a farther one-variable point is better.

On May 29, 2026, `run_multistart_optimization.py` was added as a separate
grid-seeded multistart diagnostic under `results/multistart/`. It does not
change the objective or constraints. It evaluates a one-variable grid around
the baseline, selects the best unique seed decisions, runs the same L-BFGS-B
optimizer from each seed, and keeps the best final result. The optimizer class now has
`optimize_multistart(starts)` while preserving the original single-start
`optimize(...)` path.

Command:

```powershell
python experiments/test/wildfire_tests/run_multistart_optimization.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml --num-seed-points 11 --max-seeds 5
```

Output:

```text
results/multistart/multistart_gps_20260529_000941/
```

Selected starts:

```text
alpha_7_bus18 = 0.0, 0.1, 0.2, 0.3, 0.4
```

Result:

```text
baseline objective:          0.999001
best seed objective:         0.9092841653972372
best final objective:        0.8961482348645832
best final R_group:          2.416338685293566
best final L_shed:           1.0000086888091182
best start index:            1
optimizer success:           True
optimizer message:           CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH
```

All five starts converged successfully to nearly the same objective region.
The final decision keeps all generator redispatch at `0.0`, sets
`alpha_7_bus18 = 0.0`, and makes only tiny numerical changes to a few other
selected alpha values. This confirms that the risk-preferred GPS formulation
has a lower-objective point that the baseline-started optimizer was not
finding. The objective moves in two stages: the grid-seeded start moves the
candidate from the baseline objective `0.999001` to `0.9092841653972372`, and
then L-BFGS-B improves that modified start to `0.8961482348645832`.

The multistart runner now writes both multistart-specific artifacts and the
standard optimization-behavior plot for the best start:

```text
results/multistart/multistart_gps_20260529_000941/selected_starts.csv
results/multistart/multistart_gps_20260529_000941/multistart_results.csv
results/multistart/multistart_gps_20260529_000941/objective_trace.csv
results/multistart/multistart_gps_20260529_000941/figures/optimization_behavior.png
```

Later on May 29, 2026, the older nested multistart output under
`results/objective_analysis/multistart/` was deleted so multistart is no longer
mixed with the objective-analysis sensitivity folders. The multistart runner
was extended with `--tradeoff-sets`, which generates the three standard lambda
cases (`risk`, `balanced`, `shed`) for both `gnn` and `gps` under:

```text
results/multistart/<tradeoff_set>/<model_type>/<timestamped_run>/
```

Command:

```powershell
python experiments/test/wildfire_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

An attempted `--clear` run could not delete one older direct
`results/multistart/multistart_gps_20260529_000941/figures` directory because
OneDrive/Windows reported an access-denied lock. This did not affect the
requested cleanup of the nested objective-analysis folder, and the six fresh
tradeoff runs were generated without `--clear`.

Latest multistart tradeoff outputs:

```text
results/multistart/risk/gnn/multistart_gnn_20260529_001357/
results/multistart/risk/gps/multistart_gps_20260529_001442/
results/multistart/balanced/gnn/multistart_gnn_20260529_001543/
results/multistart/balanced/gps/multistart_gps_20260529_001624/
results/multistart/shed/gnn/multistart_gnn_20260529_001700/
results/multistart/shed/gps/multistart_gps_20260529_001708/
```

Aggregate summary:

```text
results/multistart/multistart_tradeoff_summary.csv
results/multistart/multistart_tradeoff_summary.json
```

Each run includes `figures/optimization_behavior.png`, `selected_starts.csv`,
`multistart_results.csv`, `objective_trace.csv`, `best_objective_trace.csv`,
`best_decision_vector.csv`, `analysis_summary.json`, and
`optimization_summary.json`.

Latest multistart tradeoff result summary:

```text
risk/gnn:
  objective:      0.999001 -> 0.9950359605419952
  group risk:     9.239211240283675
  load shedding:  2.2618633086571864

risk/gps:
  objective:      0.999001 -> 0.8961482348645832
  group risk:     2.416338685293566
  load shedding:  1.0000086888091182

balanced/gnn:
  objective:      0.5 -> 0.5
  group risk:     9.276730045998226
  load shedding:  0.0
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:      0.5 -> 0.4651783290181926
  group risk:     2.4163717542193073
  load shedding:  1.0

shed/gnn:
  objective:      0.000999 -> 0.000999
  group risk:     9.276730045998226
  load shedding:  0.0

shed/gps:
  objective:      0.000999 -> 0.000999
  group risk:     2.6937669151682497
  load shedding:  0.0
```

Interpretation: risk-preferred GPS remains the clearest improvement, balanced
GPS also improves, risk-preferred GNN improves slightly, and both
shed-preferred cases correctly stay at baseline because the objective strongly
penalizes any load shedding.

Later on May 29, 2026, after switching the optimized load-shedding cost to the
demand-weighted fraction, the three connected-corridor tradeoff cases were
regenerated for both GNN and GPS without multistart:

```powershell
python experiments/test/wildfire_tests/run_connected_corridor_tradeoffs.py
```

Latest demand-weighted baseline-started outputs:

```text
results/demand_weighted/connected_corridor/risk/gnn/risk_gnn_20260529_134532/
results/demand_weighted/connected_corridor/risk/gps/risk_gps_20260529_134541/
results/demand_weighted/connected_corridor/balanced/gnn/balanced_gnn_20260529_134543/
results/demand_weighted/connected_corridor/balanced/gps/balanced_gps_20260529_134550/
results/demand_weighted/connected_corridor/shed/gnn/shed_gnn_20260529_134552/
results/demand_weighted/connected_corridor/shed/gps/shed_gps_20260529_134553/
```

Demand-weighted baseline-started summary:

```text
risk/gnn:
  objective:  0.999001 -> 0.9989932074556757
  group risk: 9.276730045998226 -> 9.276657378702414
  min alpha:  0.9995908031440097

risk/gps:
  objective:  0.999001 -> 0.999001
  group risk: 2.6937669151682497 -> 2.6937669151682497

balanced/gnn:
  objective:  0.5 -> 0.5
  group risk: 9.276730045998226 -> 9.276730045998226
  note: optimizer reported ABNORMAL and returned baseline

balanced/gps:
  objective:  0.5 -> 0.5
  group risk: 2.6937669151682497 -> 2.6937669151682497

shed/gnn:
  objective:  0.000999 -> 0.000999
  group risk: 9.276730045998226 -> 9.276730045998226

shed/gps:
  objective:  0.000999 -> 0.000999
  group risk: 2.6937669151682497 -> 2.6937669151682497
```

The matching demand-weighted multistart set was also regenerated:

```powershell
python experiments/test/wildfire_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
```

Latest demand-weighted multistart outputs:

```text
results/demand_weighted/multistart/risk/gnn/multistart_gnn_20260529_134608/
results/demand_weighted/multistart/risk/gps/multistart_gps_20260529_134657/
results/demand_weighted/multistart/balanced/gnn/multistart_gnn_20260529_134755/
results/demand_weighted/multistart/balanced/gps/multistart_gps_20260529_134825/
results/demand_weighted/multistart/shed/gnn/multistart_gnn_20260529_134854/
results/demand_weighted/multistart/shed/gps/multistart_gps_20260529_134901/
```

Demand-weighted multistart summary:

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

Interpretation: demand weighting makes the load penalty larger than the old
`1/N_bus` equal-bus normalization for the same complete selected-bus shed, but
the risk-preferred GPS and balanced GPS multistart cases still improve because
the risk reduction is large enough to offset about 3.35% demand-weighted
shedding. Shed-preferred cases still stay at baseline.

Later, `ac_opf_experiment.py` was added as a deliberately isolated hard
AC-OPF experiment. It follows the standard AC-OPF constraint categories from
MATPOWER/PowerModels-style formulations: AC P/Q nodal balance equalities,
branch apparent-flow thermal inequalities, generator P/Q bounds, voltage
magnitude bounds, reference-angle constraint, and selected alpha bounds. It is
not imported by the main GridFM workflow and can be removed by deleting the
module plus `results/ac_opf_connected_corridor/`.

The first run wrote:

```text
results/ac_opf_connected_corridor/ac_opf_20260528_223144/
```

Command:

```powershell
python experiments/test/wildfire_tests/ac_opf_experiment.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml
```

Result: all three tradeoff solves failed to satisfy hard AC constraints. SLSQP
reported `Positive directional derivative for linesearch`; final max absolute
P/Q balance residuals stayed near `7767.5 MVA`, and final thermal margins were
about `-2884.6 MVA`. Treat these outputs as evidence that the local
`ScenarioData` approximation is not yet a reliable hard AC-OPF model, not as a
valid AC-OPF comparison result.

## Stage B.1/B.2 Multi-Group Wildfire Risk Sensitivity

On May 31, 2026, Stage B.1/B.2 was implemented as an extension of the fixed
topology first-pass workflow. The manual connected corridor remains available
through:

```yaml
wildfire:
  selection_method: manual_connected
```

The new automatic mode is:

```yaml
wildfire:
  selection_method: automatic_risk_components
```

Automatic ranking computes a baseline score for every candidate line:

```text
score_l = p_env * loading_l(base)^2 * I_l(base)
```

For Stage B.1/B.2, `p_env` is uniform before selection. The checked-in
automatic configs set `risk_score.candidate_p_env = 1.0`, so the ranking does
not reuse the old manual-corridor convention of `1.0` on selected lines and
`0.1` elsewhere. This avoids circular selection because selected lines do not
exist until after ranking.

Top-fraction selection uses:

```text
num_selected_lines = ceil(num_lines * requested_top_fraction)
realized_selected_fraction = num_selected_lines / num_lines
```

Selected lines are grouped by connected components into `G_1`, `G_2`, ... with
equal group weights. The automatic code allows a single connected component and
records whether the result collapsed to one group.

New configs:

```text
configs/automatic_multigroup_gps.yaml
configs/automatic_multigroup_gnn.yaml
```

New runner:

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

The runner defaults to grid-seeded multistart for `gps` and `gnn`, over
`risk`, `balanced`, and `shed`, with near-0/near-1 lambda cases:

```text
risk:     lambda_R = 0.999001, lambda_L = 0.000999
balanced: lambda_R = 0.5,      lambda_L = 0.5
shed:     lambda_R = 0.000999, lambda_L = 0.999001
```

New result root:

```text
results/multi_group/
```

Threshold folder names use:

```text
threshold_0p10
threshold_0p125
threshold_0p15
threshold_0p175
threshold_0p20
```

Each automatic run writes:

```text
automatic_line_risk_scores.csv
automatic_wildfire_groups.json
automatic_group_summary.csv
```

Multistart runs now also write the standard run artifacts needed for
comparison and visualization: baseline/final objective components,
baseline/risk-before-after CSVs, decision vectors, traces, normalizers,
`optimization_summary.json`, `analysis_summary.json`,
`figures/optimization_behavior.png`, and
`figures/ieee30_network_changes.png`.

The topology visualization now colors automatic high-risk lines by group and
adds group legend entries while preserving selected generator/load bus
highlighting. `visualization_summary.json` includes selection method,
requested/realized top fraction, selected line IDs, group IDs, group line and
bus IDs, collapsed-to-single-group status, and largest-group diagnostics.

Verified smoke command:

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 --models gps --tradeoff-cases risk --num-seed-points 3 --max-seeds 2
```

Latest smoke result:

```text
results/multi_group/threshold_0p10/risk/gps/multistart_gps_20260531_191421/
```

Smoke summary:

```text
requested_top_fraction: 0.10
num_lines: 110
num_selected_lines: 11
realized_selected_fraction: 0.10
num_groups: 5
largest_group_num_lines: 5
largest_group_fraction_of_selected_lines: 0.45454545454545453
collapsed_to_single_group: false
baseline_objective: 0.999001
best_seed_objective: 0.8983104240958543
best_objective: 0.898298444155839
best_grouped_risk: 2.5433063515171694
best_load_shedding: 0.03352434654418237
optimizer_success: true
```

The full 30-run default Stage B.2 sweep has not yet been run in full because
the verified smoke run took about 100 seconds for one reduced GPS/risk case
with `--num-seed-points 3 --max-seeds 2`; the default
`--num-seed-points 11 --max-seeds 5` sweep is materially heavier.

Focused verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_first_pass_multi_group_runner.py -q
```

Result:

```text
18 passed, 3 external deprecation warnings
```

Stage B.3/B.4 remain deferred. Distinct seeded grouping and manual
multi-region stress testing were intentionally not implemented and should only
be considered after manual review of Stage B.1/B.2 results under
`results/multi_group/`.

Later on May 31, 2026, GPS-only Stage B.2 results were generated for
20% and 30% thresholds using the default demand-weighted, automatic
multi-group, grid-seeded multistart methodology:

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.20 0.30 --models gps --tradeoff-cases risk balanced shed
```

The objective normalizers in these runs record:

```text
load_shedding_normalizer = 1.0
load_shedding_metric = demand_weighted_fraction
```

Output directories:

```text
results/multi_group/threshold_0p20/risk/gps/multistart_gps_20260531_192631/
results/multi_group/threshold_0p20/balanced/gps/multistart_gps_20260531_193233/
results/multi_group/threshold_0p20/shed/gps/multistart_gps_20260531_193552/
results/multi_group/threshold_0p30/risk/gps/multistart_gps_20260531_193631/
results/multi_group/threshold_0p30/balanced/gps/multistart_gps_20260531_194657/
results/multi_group/threshold_0p30/shed/gps/multistart_gps_20260531_195041/
```

Aggregate summary:

```text
results/multi_group/multi_group_threshold_sensitivity_summary.csv
results/multi_group/multi_group_threshold_sensitivity_summary.json
```

GPS 20% summary:

```text
num_selected_lines: 22 / 110
realized_selected_fraction: 0.20
num_groups: 6
largest_group_num_lines: 10
largest_group_fraction_of_selected_lines: 0.45454545454545453
collapsed_to_single_group: false
risk objective:     0.999001 -> 0.8985520880723752
balanced objective: 0.5      -> 0.4664756256271531
shed objective:     0.000999 -> 0.000999
```

GPS 30% summary:

```text
num_selected_lines: 33 / 110
realized_selected_fraction: 0.30
num_groups: 5
largest_group_num_lines: 24
largest_group_fraction_of_selected_lines: 0.7272727272727273
collapsed_to_single_group: false
risk objective:     0.999001 -> 0.8985455799965569
balanced objective: 0.5      -> 0.4664754236987482
shed objective:     0.000999 -> 0.000999
```

## Stage C PSPS Threshold Baseline

On May 31, 2026, Stage C was implemented as a deterministic PSPS-only
baseline/comparator. It does not optimize `z_l` or `y_n`, does not add
mixed-integer topology controls, and does not optimize continuous controls
after PSPS. Stage D optimized de-energization remains deferred.

Methodology:

```text
grouping_top_fraction = 0.30
psps_top_fraction = 0.10
lambda_R = 0.999001
lambda_L = 0.000999
evaluation_mode = psps_only
```

The grouping threshold builds the Stage B automatic multi-group scenario.
Only those candidate high-risk lines are eligible for Stage C PSPS
de-energization; all non-candidate lines keep `z_l = 1`.

Stage C uses a fixed demand-weighted line consequence score:

```text
I_l = max(0, S_D_base - S_D_outage_l) / S_D_base
S_D_base = sum_n P_D,n
```

This is computed once per line from the baseline one-line outage prediction.
Stage C does not use the old dynamic equal-bus `I_l(u)` consequence in risk
calculations.

For each environmental case, case-specific `p_env_l` is assigned before PSPS
ranking. The PSPS ranking score is:

```text
baseline_psps_risk_l = p_env_l * loading_l(base)^2 * I_l
```

The de-energization rule is:

```text
num_psps_lines = max(1, ceil(psps_top_fraction * num_candidate_lines))
realized_psps_fraction = num_psps_lines / num_candidate_lines
```

Then candidate lines are sorted by descending `baseline_psps_risk_l` and
ascending `line_id`; the top `num_psps_lines` get `z_l = 0`. A line with
`z_l = 0` contributes zero wildfire risk.

Implemented environmental cases:

```text
auto_env
largest_group_high
```

`largest_group_high` identifies the largest automatic connected group by line
count, then baseline group risk, then lower numeric group ID. It sets that
group to `p_env = 1.0`; other groups receive seeded random group-level
`p_env` values in `[0.6, 0.9]` with seed `30`.

New files:

```text
experiments/test/wildfire_tests/stage_c_psps.py
experiments/test/wildfire_tests/run_stage_c_psps_baseline.py
tests/test_wildfire_stage_c_psps.py
```

The GridFM runner now supports `predict_with_line_outages(...)` for one or
more removed scenario lines. For post-PSPS line loading, Stage C uses the
PSPS topology GridFM bus prediction on the original line indexing and
explicitly sets de-energized line loading to zero. This avoids a reduced-edge
loading shape mismatch in the existing overload evaluator while preserving
original line-ID auditability.

The planned long Stage C folder names were shortened because Windows/OneDrive
path length limits prevented writing required artifact filenames. Outputs are
still under `results/stage_c_psps/`:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

Aggregate outputs:

```text
results/stage_c_psps/stage_c_psps_summary.csv
results/stage_c_psps/stage_c_psps_summary.json
```

Command run:

```powershell
python experiments/test/wildfire_tests/run_stage_c_psps_baseline.py --grouping-top-fraction 0.30 --psps-top-fraction 0.10 --models gps --cases auto_env largest_group_high
```

GPS Stage C results:

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
  disconnected components:      1
  affected bus IDs:             [4, 5, 6, 7, 27]

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
  disconnected components:      1
  affected bus IDs:             [4, 5, 6, 7, 27]
```

Each run writes:

```text
fixed_line_consequence_scores.csv
psps_line_risk_scores.csv
psps_deenergization_decisions.csv
automatic_wildfire_groups.json
automatic_group_summary.csv
objective_trace.csv
optimization_summary.json
visualization_summary.json
figures/optimization_behavior.png
figures/ieee30_network_changes.png
```

`objective_trace.csv` has exactly two rows: baseline all-energized and
post-PSPS threshold topology. The network visualization now overlays PSPS
de-energized lines when the Stage C decision artifact is present.

Focused verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py -q
```

Result:

```text
23 passed, 3 external deprecation warnings
```

Later on May 31, 2026, the Stage C network visualization was cleaned up so
PSPS de-energized lines are visually unambiguous. De-energized lines are now
drawn last with a red dashed stroke, red X markers, and `OFF <line_id>` edge
labels. The legend is placed below the lower-right side of the plot in two
columns so it no longer blocks the lower-left network area. The current Stage C
GPS figures were regenerated for:

```text
results/stage_c_psps/t0p30/p0p10/r/gps/auto/run_20260531_205156/
results/stage_c_psps/t0p30/p0p10/r/gps/lgh/run_20260531_205158/
```

The `results/stage_c_psps/` directory was also cleaned to retain only the
aggregate summary files and the two current near-lambda GPS runs with figures.
Older failed path-length attempts and pre-near-lambda runs were removed.

Verification:

```powershell
pytest tests/test_wildfire_stage_c_psps.py -q
```

Result:

```text
7 passed, 3 external deprecation warnings
```

Each `run_basic_case.py` execution creates a timestamped run directory:

```text
results/<run_name>_<YYYYMMDD_HHMMSS>/
```

For the current demand-weighted connected-corridor workflow, model runs are
organized as:

```text
results/demand_weighted/connected_corridor/<tradeoff_set>/gnn/<run_name>_<timestamp>/
results/demand_weighted/connected_corridor/<tradeoff_set>/gps/<run_name>_<timestamp>/
results/demand_weighted/connected_corridor/sweeps/<sweep_name>_<timestamp>/
```

Core machine-readable outputs stay at the run-directory root as JSON and CSV
files. Human-facing plots are grouped under:

```text
results/<run_name>_<timestamp>/figures/
```

Automatic figures from `run_basic_case.py`:

```text
figures/optimization_behavior.png
figures/ieee30_network_changes.png
visualization_summary.json
```

`optimization_behavior.png` shows objective, grouped wildfire risk, load
shedding over objective evaluations, and the configured objective weights
`lambda_R` and `lambda_L`. The weight panel is visual-only; realized weighted
normalized terms remain in `objective_trace.csv`. `ieee30_network_changes.png`
is generated from the actual IEEE-30 `edge_index` used by the scenario. The
layout is topology-accurate but not geographic: buses connected in the scenario
graph are connected in the figure, but coordinates come from a deterministic
spring layout. Bus labels are displayed as 1-based IEEE-style labels; high-risk
line labels are the internal scenario edge IDs used by the wildfire scenario
JSON and risk CSV files.

For stability sweeps, `run_stability_sweep.py` creates one timestamped sweep
directory and places the child optimization runs under:

```text
results/<sweep_name>_<timestamp>/runs/
```

This keeps repeated sweeps and their per-model/per-seed figures grouped
together instead of scattering child run folders at the top level.

## Commands To Reproduce

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py -q
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_decision_vector.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_objective_analysis.py -q
python experiments/test/wildfire_tests/run_connected_corridor_tradeoffs.py --clear
python experiments/test/wildfire_tests/run_connected_corridor_tradeoffs.py
python experiments/test/wildfire_tests/objective_analysis.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml --line-id 23 --num-points 101
python experiments/test/wildfire_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml --num-points 21
python experiments/test/wildfire_tests/objective_analysis.py --mode decision --config experiments/test/wildfire_tests/configs/connected_corridor_gnn.yaml --num-points 21
python experiments/test/wildfire_tests/run_multistart_optimization.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_tests/run_multistart_optimization.py --tradeoff-sets --num-seed-points 11 --max-seeds 5
python experiments/test/wildfire_tests/ac_opf_experiment.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml
python experiments/test/wildfire_tests/run_basic_case.py --config experiments/test/wildfire_tests/configs/basic_gps.yaml
python experiments/test/wildfire_tests/run_basic_case.py --config experiments/test/wildfire_tests/configs/basic_gnn.yaml
python experiments/test/wildfire_tests/run_basic_case.py --config experiments/test/wildfire_tests/configs/connected_corridor_gps.yaml
python experiments/test/wildfire_tests/run_basic_case.py --config experiments/test/wildfire_tests/configs/connected_corridor_gnn.yaml
python experiments/test/wildfire_tests/run_stability_sweep.py --config experiments/test/wildfire_tests/configs/stability_sweep.yaml
python experiments/test/wildfire_tests/plot_optimization_behavior.py --run-dir experiments/test/wildfire_tests/results/<run_name>
python experiments/test/wildfire_tests/plot_network_changes.py --run-dir experiments/test/wildfire_tests/results/<run_name>
```

## Important Implementation Notes

- Do not call GridFM through the CLI inside the objective loop.
- Use in-memory model loading through `pipeline_utils.py`.
- Use `NeuralSolverWrapper` for prediction.
- Counterfactual line-outage impact currently removes one scenario edge inside
  `GridFMRunner.predict_with_line_outage`; this is a surrogate consequence
  calculation, not an optimizer decision to de-energize the line.
- Keep generated objective components separate from scalar objective values.
- Invalid predictions should produce explicit penalties or validation failures, not silent crashes.
- Generated result directories are currently under `experiments/test/wildfire_tests/results`.

## Next Session Launch Plan

When a new Codex session starts, read this section first and then inspect the
current code/results before editing. Preserve the current methodology:

- update code, configs, tests, generated outputs, `FIRST_PASS_SUMMARY.md`, and
  this `HISTORY.md` together when behavior changes
- run focused tests after implementation changes
- rerun the relevant model workflows when objective, decision, risk, or result
  semantics change
- record commands and outcomes here so this file remains a durable handoff

Current research transition: the simplified fixed-topology/fixed-energization
case is now polished enough for first-pass interpretation. The next major
methodology step is to evaluate behavior with de-energization included, while
keeping the existing simplified results as the baseline comparison.

Stage D update, May 31, 2026:

- Added `run_stage_d_deenergization.py` and Stage D helper logic for
  `evaluation_mode = limited_enumerated_z_only`.
- Stage D uses `grouping_top_fraction = 0.30`, computes the generated
  candidate count dynamically, and enumerates all `z_l` subsets with 0, 1, or
  2 de-energized candidate lines.
- The current GPS run generated 33 candidate lines and 562 evaluated subsets
  per environmental case.
- Subset evaluations are shared across the three Stage D lambda cases:

```text
risk_leaning:    lambda_R = 0.8, lambda_L = 0.2
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.2, lambda_L = 0.8
```

- Stage D uses the fixed demand-weighted `I_l` from Stage C:

```text
risk_l = z_l * p_env_l * loading_l^2 * I_l
```

- `R_group_baseline` is the all-energized baseline for the same model,
  environmental case, grouping threshold, fixed `I_l`, and candidate groups;
  it is not lambda-dependent.
- Stage C comparison is available for the GPS runs. The Stage C PSPS topology
  is fixed, while its scalar objective is re-scored under each Stage D lambda
  for apples-to-apples comparison.
- Generated outputs:

```text
results/stage_d_deenergization/t0p30/gps/auto/risk/run_20260531_212125/
results/stage_d_deenergization/t0p30/gps/auto/bal/run_20260531_212126/
results/stage_d_deenergization/t0p30/gps/auto/svc/run_20260531_212128/
results/stage_d_deenergization/t0p30/gps/lgh/risk/run_20260531_212136/
results/stage_d_deenergization/t0p30/gps/lgh/bal/run_20260531_212138/
results/stage_d_deenergization/t0p30/gps/lgh/svc/run_20260531_212139/
results/stage_d_deenergization/stage_d_deenergization_summary.csv
```

- GPS best subsets:

```text
auto_env/risk_leaning:          [23, 27]
auto_env/balanced:              [23, 27]
auto_env/service_leaning:       []
largest_group_high/risk_leaning:[23, 27]
largest_group_high/balanced:    [23, 27]
largest_group_high/service:     []
```

- `ieee30_network_changes.png` now recognizes
  `optimized_deenergization_decisions.csv` and labels those lines as
  Stage D optimized de-energized lines.
- Deferred: full mixed-integer topology optimization, relaxed continuous
  `z_l`, optimized `y_n`, AC feasibility enforcement, and continuous-control
  optimization after selecting `z_l`.

Latest Stage D verification:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py tests/test_wildfire_stage_d_deenergization.py -q
python experiments/test/wildfire_tests/run_stage_d_deenergization.py --grouping-top-fraction 0.30 --models gps --cases auto_env largest_group_high --evaluation-mode limited_enumerated_z_only --max-deenergized-lines 2
```

Test result: 30 passed, with 3 external deprecation warnings.

Gurobi transition checkpoint, June 4, 2026:

- Current session began with an orientation-only pass: read this `HISTORY.md`
  and `FIRST_PASS_SUMMARY.md`, inspect Stage C/Stage D optimizer touchpoints,
  and avoid code revisions until the next manual prompt.
- The desktop has `gurobipy` installed in the active Anaconda environment.
  A sandboxed license check failed because the sandbox username did not match
  the academic license user, but the same tiny Gurobi LP run outside the
  sandbox solved successfully under the real Windows user. The detected
  academic license is non-commercial and expires on June 5, 2027.
- The main work for the next prompts is to integrate Gurobi as the optimizer
  path for future work, while keeping Stage C and Stage D methodology
  traceable and preserving generated result artifacts.
- As Gurobi changes and regenerated results are produced, continue the
  existing methodology: update code/configs/tests/results together, update
  `FIRST_PASS_SUMMARY.md` when behavior or interpretation changes, and record
  commands and outcomes in this file.
- Be careful that future evaluations preserve the automatic multi-group Stage
  B/C/D setup where intended; do not accidentally revert to the earlier
  single manual connected-corridor group for multi-group evaluations.
- After this session's Gurobi integration/cleanup work, the planned repository
  maintenance step is to preserve the current research files while aligning
  this fork with the original upstream repository, then continue the wildfire
  research workflow.

Gurobi discussion follow-up, June 5, 2026:

- Put Gurobi integration on hold for the next session. The current
  predict-then-optimize workflow repeatedly calls GridFM as a Python black-box
  evaluator: choose `u`, run GridFM, extract predicted loading/service state,
  compute wildfire risk and demand-weighted load shedding, then return a
  scalar objective.
- This black-box loop is compatible with SciPy-style objective-function
  minimization, grid-seeded multistart, manual sweeps, deterministic PSPS
  thresholding, and bounded enumeration, but it is not directly compatible
  with a simple `gurobipy.minimize` replacement. Gurobi needs algebraic
  variables, constraints, and objective expressions; it cannot natively
  optimize through a PyTorch/GridFM call.
- Current Stage C remains a deterministic PSPS comparator, and current Stage D
  remains limited enumeration over candidate `z_l` subsets with GridFM
  evaluation. A Gurobi-backed Stage D could select from precomputed candidate
  evaluations, but that would be a candidate-selection model rather than true
  optimization through GridFM.
- Plausible future paths are: Gurobi over precomputed GridFM-evaluated
  candidates, Gurobi over an algebraic response-surface surrogate fitted from
  GridFM samples, or a larger reformulation that approximates/embeds GridFM.
  Treat this as a methodology design task before implementation.

Methodology refocus, June 10, 2026:

- After discussion with the professor, Gurobi is no longer part of the active
  wildfire first-pass research path. Treat it as a separate future research
  direction, possibly for a later paper or a distinct methodology thread.
- The active claim to develop is decision quality, not solver superiority:
  within small curated grid topologies and toy wildfire scenarios, evaluate
  whether the predict-to-optimize framework makes sensible decisions as the
  scenario changes.
- This refocus also helps justify continuing with SciPy-style black-box
  optimization for now. The current objective repeatedly evaluates GridFM, so
  SciPy/multistart/sweeps/enumeration match the present experimental method
  better than forcing an algebraic optimizer into a non-algebraic loop.
- Near-term experiments should be designed around scenario hypotheses. For
  each curated wildfire risk case, record what decision changes are expected,
  what the optimizer actually changes, and whether those decisions are
  interpretable under the modeled risk/service tradeoff.
- Keep the existing discipline: update code, configs, tests, generated
  outputs, `FIRST_PASS_SUMMARY.md`, and this file together whenever behavior
  or interpretation changes.

Current near-term plan:

1. Align this wildfire experiment branch with the latest commits from the
   main branch and original upstream repository while preserving the research
   artifacts and notes.
2. Simplify the formulation to make it less constrained and test whether that
   improves optimization behavior or interpretability.
3. Curate small-grid wildfire risk scenarios with explicit hypotheses about
   expected optimizer behavior, then rerun the updated methodology.
4. If decision quality looks promising, benchmark against heuristic baselines,
   including transmission and area heuristics inspired by the
   "Balancing Wildfire Risk and Power Outages" paper.
5. Scale to larger grid scenarios after the toy cases are understood, with
   emphasis on computational efficiency and economic impact. Expect the
   formulation and heuristic comparisons to need adjustment before that scale
   up is meaningful.

Integration audit, June 10, 2026:

- The fork's `main`, `origin/main`, and `upstream/main` were aligned at
  commit `2cdd791`.
- Before merging updated `main` into `wildfire-experiments`, a non-mutating
  integration audit was performed and documented in
  `INTEGRATION_AUDIT_20260610.md`.
- No merge was performed during the audit.
- The actual upstream delta from the merge base to updated `main` touches only
  six files: two GitHub workflow files, `gridfm_graphkit/__main__.py`,
  `gridfm_graphkit/cli.py`,
  `gridfm_graphkit/datasets/hetero_powergrid_datamodule.py`, and
  `gridfm_graphkit/models/gnn_heterogeneous_gns.py`.
- `git merge-tree` reported no textual conflicts. The audit notes that
  `git diff HEAD..main` is misleading here because it shows branch-local
  wildfire files as deletions, which is not the same as merging `main` into
  `wildfire-experiments`.
- Preserved research dependencies remain important: the GNN/GPS checkpoints
  under `examples/models/`, the wildfire code under `experiments/test/`, and
  the IEEE-30 processed tensors under `tests/data/case30_ieee/processed/`.
  The IEEE-30 tensors were initially ignored by Git because of `*.pt` and
  `*.done`; because no regeneration process is currently available, the 26
  small processed files were intentionally force-added with `git add -f` for
  reproducibility.
- Baseline focused wildfire verification before merge:

```powershell
pytest tests/test_wildfire_first_pass_scenario.py tests/test_wildfire_first_pass_risk.py tests/test_wildfire_first_pass_objective.py tests/test_wildfire_first_pass_basic_run.py tests/test_wildfire_first_pass_multistart.py tests/test_wildfire_stage_c_psps.py tests/test_wildfire_stage_d_deenergization.py -q
```

Result: 30 passed, with 3 external deprecation warnings.

Immediate Stage B follow-up: run the full default multi-group threshold sweep
when compute time is available, then manually inspect
`results/multi_group/multi_group_threshold_sensitivity_summary.csv` and the
child automatic group artifacts before deciding whether Stage B.3/B.4 are
needed.

```powershell
python experiments/test/wildfire_tests/run_multi_group_threshold_sensitivity.py --top-fractions 0.10 0.125 0.15 0.175 0.20
```

Priority 1: add a PSPS-style baseline with threshold de-energization.

The previous visual-clarity task for `optimization_behavior.png` was completed
on May 19, 2026: the plot now shows configured objective weights instead of
realized weighted term magnitudes.

Remaining methodology note: if future work again compares realized objective
terms across tradeoff sets, start from
`results/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.csv`
and the child `objective_trace.csv` files, and remember that the current
normalizers are `R_group / R_baseline` and the already normalized
demand-weighted `L_shed_weighted`.

- Implement a comparison baseline that de-energizes all lines above a wildfire
  risk threshold, representing a simple PSPS rule.
- Keep this separate from the current GridFM continuous optimization at first:
  it should be a baseline/comparator, not yet a mixed-integer optimizer.
- Decide how to represent de-energized lines in the current data path:
  likely a new result-generation module and reporting artifacts before
  introducing `z_l` into the optimizer.
- Compare at least:
  grouped wildfire risk, load shedding/service loss, affected buses/loads,
  disconnected components if any, and visualization of de-energized corridor
  lines.
- Add tests for threshold selection and reporting behavior before trusting the
  baseline results.

Priority 2 status: demand-weighted load-shedding cost has been implemented.

- Completed: the optimized load-shedding term now uses demand weights:

```text
w_n = P_D,n / sum_m P_D,m
L_shed_weighted = sum_n w_n * (1 - alpha_n)
```

- Completed: current generated runs use `load_shedding_normalizer = 1.0`
  because the demand weights sum to 1.
- Completed: tests were updated so a large-load bus contributes more to the
  objective than an equal fractional shed at a small-load bus.
- Completed: traces and multistart summaries now include diagnostic
  `equal_bus_load_shedding` and `unserved_demand_mw` where available.
- Remaining: if a future comparison needs the old equal-bus objective, add it
  as a named objective option rather than silently changing this metric back.

Deferred but still relevant:

1. Decide whether relaxed `[0, 1]` alpha should be the main setting or an
   emergency-style sensitivity, since `risk/gps` currently reaches
   `alpha = 0.0` on selected buses.
2. Run a reduced stability sweep and summarize variability.
3. Inspect sensitivity of the new counterfactual `I_l(u)` term to `alpha` and
   `Delta_Pg`, and decide whether to back it with a physical solver.
4. Add corridor-specific finite-difference control selection.
5. Add voltage and thermal violation diagnostics to CSV/JSON summaries.
6. Decide whether generated `results/` should be tracked or moved to ignored
   local output.
7. Later, revisit topology controls `y_n` and `z_l`, but do not add
   mixed-integer logic until the PSPS baseline is understood.

## June 10, 2026 - Wildfire Harness Refactor Implemented

Refactored the active wildfire experiment workflow from the old flat
`experiments/test/wildfire_initial_tests` layout into the new staged harness:

```text
experiments/test/wildfire_tests/
  HISTORY.md
  CURRENT_STATE_SUMMARY.md
  INTEGRATION_AUDIT_20260610.md
  README.md
  gridfm_support/
  shared/
  stage_a_first_pass/
  stage_b_multigroup/
  stage_c_psps_baseline/
  stage_d_deenergization/
  analysis/
  results/
  methodology_testing_results/
```

Key implementation details:

- Renamed `FIRST_PASS_SUMMARY.md` to `CURRENT_STATE_SUMMARY.md`.
- Moved GridFM helper wrappers into `gridfm_support/`.
- Moved reusable wildfire config, objective, path, plotting, reporting, and
  risk logic into `shared/`.
- Moved Stage A-D runners into stage-specific folders.
- Moved objective-analysis and AC-OPF tooling into `analysis/`.
- Left `wildfire_initial_tests/` in place as a compatibility shim for old
  imports and runner paths.
- Centralized refactored paths in `shared/paths.py`.
- Centralized canonical and legacy lambda definitions in
  `shared/lambda_cases.py`.
- Updated active base configs to use the canonical standardized lambda cases.
- Updated Stage C and Stage D to run all canonical lambda cases by default.
- Added parity comparison tooling in `analysis/parity_compare.py`.
- Added refactor-structure tests covering path helpers, compatibility imports,
  lambda sums, legacy lambda availability, and parity output schema.

Canonical lambda convention now used for new outputs:

```text
risk_leaning:    lambda_R = 0.9, lambda_L = 0.1
balanced:        lambda_R = 0.5, lambda_L = 0.5
service_leaning: lambda_R = 0.1, lambda_L = 0.9
```

Verification commands run:

```powershell
python -m compileall experiments/test/wildfire_tests experiments/test/wildfire_initial_tests tests/test_wildfire_refactor_structure.py
$files = Get-ChildItem tests -Filter 'test_wildfire*.py' | ForEach-Object { $_.FullName }; pytest $files -q
python -m experiments.test.wildfire_tests.stage_a_first_pass.run_basic_case --help
python -m experiments.test.wildfire_tests.stage_d_deenergization.run_stage_d_deenergization --help
python -c "from experiments.test.wildfire_initial_tests.config import FirstPassConfig; from experiments.test.wildfire_initial_tests.run_basic_case import run_basic_case; from experiments.test import pipeline_utils; print(FirstPassConfig.__name__, callable(run_basic_case), pipeline_utils.__name__)"
```

Verification result:

```text
42 wildfire tests passed.
Compile check passed.
New module entry points resolved.
Old compatibility imports resolved.
```

Cleanup decision:

- Old `wildfire_initial_tests/results/` folders were kept in place.
- The untracked Stage D `run_20260531_2118xx` folders were left untouched.
- No old results were deleted or moved because full legacy parity comparisons
  and full canonical standardized-weight regeneration have not yet been run.
- IEEE-30 processed tensors remain intentionally force-added because no
  regeneration process is currently available.

Next execution gate:

1. Run legacy-equivalent outputs using the new harness.
2. Compare those outputs against old `wildfire_initial_tests/results/` artifacts
   with `analysis/parity_compare.py`.
3. Only if parity passes, run the canonical standardized-weight Stage A-D and
   AC-OPF outputs under `wildfire_tests/results/`.
4. Update `CURRENT_STATE_SUMMARY.md` with the actual parity/canonical result
   directories before archiving or deleting old results.

## June 10, 2026 - Current Pickup Point Before Research Runs

Current state:

- The wildfire harness refactor is implemented and smoke-tested.
- `experiments/test/wildfire_tests/results/` exists as the new canonical output
  root, but no new refactored wildfire result files have been generated there
  yet.
- Existing historical outputs remain under
  `experiments/test/wildfire_initial_tests/results/`.
- Those old outputs are still the comparison source for parity checking.
- No old result folders have been deleted, moved, or treated as superseded.
- The untracked Stage D `run_20260531_2118xx` folders remain untouched.

Immediate next step:

1. Generate legacy-equivalent runs with the refactored `wildfire_tests` harness.
2. Compare those new legacy-equivalent outputs against the old
   `wildfire_initial_tests/results/` outputs.
3. If parity is acceptable, generate canonical standardized-weight outputs under
   `experiments/test/wildfire_tests/results/`.
4. Record the actual generated run directories and comparison status in
   `CURRENT_STATE_SUMMARY.md`.
5. After the results pipeline is verified, continue research work by simplifying
   the formulation and testing whether the optimization behavior improves.

Do not begin old-result cleanup until parity and canonical regeneration are
complete.

## June 18, 2026 - Physics-Aware Stage C/D/E Case-Study Implementation

Implemented an additive physics-aware case-study workflow under the Stage E
Gurobi implementation folder. The new output root is:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/
```

New implementation files:

- `stage_e_gurobi_implementation/physics_infeasibility_evaluator.py`
- `stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py`
- `tests/test_wildfire_physics_case_study.py`

Current behavior:

- Compares Stage C risk-only heuristic, Stage D exhaustive `k<=2`,
  constrained Stage E `K=2`, and unconstrained Stage E proposals.
- Uses `gnn` only, `t0p30` candidates, user-facing cases `auto_env` and `lfg`
  where `lfg` maps to the existing `largest_group_high` environmental case.
- Sends all final rows through one shared evaluator.
- Removes shutoff lines before GridFM input and preserves original line IDs in
  result rows.
- Runs SciPy recourse over the existing reduced `[Delta_Pg, alpha]` controls.
- Sets selected-load `alpha` upper bounds to zero for source-less islands.
- Computes final wildfire exposure as
  `sum_l z_l * p_env_l * loading_l^2 / R_base`.
- Does not use `I_l`, `c_l`, impact, or consequence in final Stage C/D/E
  wildfire scoring.
- Adds default-on dimensionless physics terms for voltage limits, thermal
  limits, generator limits, and source-less island feasibility.
- Keeps power-balance and branch-flow-consistency diagnostics at zero weight.
- Writes combined results, best-by-stage, Stage E-vs-Stage D gaps, physics
  sensitivity, Pareto/nondominated tables, recourse traces, metadata, and plots.

Suggested smoke command:

```powershell
python experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py --smoke
```

Verification completed in the current environment:

```powershell
python -m py_compile experiments/test/wildfire_tests/stage_e_gurobi_implementation/physics_infeasibility_evaluator.py experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py
python -m py_compile tests/test_wildfire_physics_case_study.py
```

Verification blocked:

- `pytest tests/test_wildfire_physics_case_study.py -q` could not run because
  the local environment is missing `torch`, and the repo-wide pytest conftest
  imports wildfire package modules that require it.

## June 18, 2026 - Physics Case Study Environment Setup Pause

Follow-up status before pausing result generation:

- Attempted to run the physics-aware smoke case with the base Conda Python.
- Base Conda Python failed immediately because `torch` was missing.
- Found the workspace `.venv`, but it also lacked the GridFM runtime stack.
- Installed missing runtime dependencies into `.venv`, then installed the local
  `gridfm-graphkit` package in editable mode with `pip install -e .`.
- The editable install brought in the remaining declared dependencies including
  `lightning`, `gridfm-datakit`, and related runtime packages.
- Import check passed inside `.venv`:

```text
torch 2.8.0+cpu
lightning 2.6.5
imports ok
```

- Created `.mplconfig/` in the repo so future Matplotlib runs can write font
  cache files inside the workspace instead of trying to write under the user
  home directory.
- Started the smoke command with `MPLCONFIGDIR` pointed at `.mplconfig`, then
  intentionally stopped before completion at the user's request.
- Checked for leftover Python processes after interruption; none were running.

Next continuation command:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .\.mplconfig).Path
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py --smoke
```

If the smoke run succeeds, continue with the full case-study run:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .\.mplconfig).Path
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py
```

No physics case-study result folder has been confirmed yet because result
generation was paused before completion.

## June 18, 2026 - Physics Case Study Smoke Run Completed

Deleted incomplete partial run directories under:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/
```

Implementation fixes made before the successful smoke:

- Added clean handling for any Gurobi proposal exception so the case-study
  runner records `proposal_failed` Stage E rows instead of crashing.
- Fixed active-topology evaluation after line removal by temporarily clearing
  cached `Yf/Yt` admittance matrices while the reduced topology is active, then
  restoring the original matrices after state extraction. This resolves the
  previous 110-vs-108 line-count mismatch for outage topologies.
- Added a smoke-only Stage D limit. Full runs still default to exhaustive
  Stage D `k<=2`, but `--smoke` now caps Stage D evaluations so the evaluator
  and output plumbing can be tested quickly.

Successful smoke command:

```powershell
$env:MPLCONFIGDIR=(Resolve-Path .\.mplconfig).Path
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_e_gurobi_implementation/run_physics_infeasibility_case_study.py --smoke
```

Smoke output directory:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/run_20260618_213145/
```

Smoke result status:

```text
combined_stage_cde_physics_results.csv shape: 22 rows x 59 columns

stage_c_risk_only      ok                  2
stage_d_k2_exhaustive  ok                 16
stage_e_k2             proposal_failed     2
stage_e_unconstrained  proposal_failed     2
```

Stage C/D smoke outputs were generated successfully, including:

```text
combined_stage_cde_physics_results.csv
combined_stage_cde_physics_results.json
best_by_stage.csv
physics_sensitivity_summary.csv
pareto_points.csv
nondominated_points.csv
physics_recourse_trace.csv
plots/
```

Stage E limitation in this environment:

- Stage E Gurobi proposal generation is still blocked from this Codex tool
  process because the active Gurobi license is tied to Windows user `caleb`,
  while this process runs as `codexsandboxoffline`.
- The smoke records this as `proposal_failed` rows with:

```text
User name mismatch (licensed to 'caleb', current user is 'codexsandboxoffline')
```

Focused test status:

- `py_compile` passed for the new implementation files.
- `.venv` now has the GridFM runtime dependencies, but does not currently have
  `pytest`, so `.\.venv\Scripts\python.exe -m pytest tests/test_wildfire_physics_case_study.py -q`
  stops with `No module named pytest`.

Next continuation options:

1. Run the full case study from a shell/user context where Gurobi is licensed
   as `caleb`.
2. Or install `pytest` into `.venv` and run the focused tests first.
3. Or add a non-Gurobi Stage E proposal fallback only for smoke/debugging, while
   keeping the full benchmark labeled as Gurobi-required.

## June 18, 2026 - Decision-Quality Testing Takes Priority

Current research direction after today's discussion:

- Move next to testing optimizer decision quality using the existing Stage C,
  Stage D, constrained Stage E, unconstrained Stage E, and unconstrained
  frontier result methodology.
- The existing Stage D/Stage E result methodology should be interpreted as a
  topology-candidate search with fixed-control GridFM evaluation at
  `decision_vector.u_base`.
- The previous `optimization_behavior` and objective-trace plots are
  best-so-far traces over evaluated topology candidates. They are not inner
  SciPy continuous-control optimization traces.
- Existing constrained Stage E, unconstrained Stage E, and unconstrained
  frontier results do not optimize `[Delta_Pg, alpha]` inside each topology.

Important pinned point:

- Continuous optimization within each topology is not part of the current
  validated Stage C/D/E decision-quality result surface.
- The physics-infeasibility case-study code includes a prototype shared
  evaluator intended to run SciPy recourse over `[Delta_Pg, alpha]`, but this
  workflow has not been validated as a proper research result and should not be
  used for interpretation yet.
- Today's smoke run only confirmed partial plumbing: Stage C and capped Stage D
  rows can evaluate under the prototype physics evaluator. Stage E rows were
  blocked by the Gurobi license user mismatch in the Codex tool process.
- The runtime impact of adding continuous recourse appears large because each
  topology can require many GridFM calls rather than the previous one fixed-u
  GridFM call per topology.

Next immediate work:

1. Continue with targeted decision-quality scenario tests using the current
   fixed-control Stage E and Pareto-front artifacts.
2. Immediately after the scenario decision-quality phase, return to the
   physics-infeasibility/continuous-recourse question.
3. Before treating physics-infeasibility results as research evidence, decide
   whether the continuous-recourse design is computationally acceptable, whether
   it needs caching or reduced budgets, and whether it should run on GPU or a
   licensed user shell.
4. Then compare against the other paper's AH and TH heuristics.
5. Finally, evaluate security-constrained / SC-OPS reliability.

## June 19, 2026 - Stage F Decision-Quality Suite Implemented

Implemented an additive Stage F decision-quality scenario suite for the current
fixed-control K<=2 methodology.

New files:

- `experiments/test/wildfire_tests/stage_f_decision_quality/__init__.py`
- `experiments/test/wildfire_tests/stage_f_decision_quality/scenario_definitions.py`
- `experiments/test/wildfire_tests/stage_f_decision_quality/run_stage_f_decision_quality.py`
- `tests/test_wildfire_stage_f_decision_quality.py`

Result root:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_f/decision_quality_analysis/
```

Stage F behavior:

- Uses the existing auto/t0p30 construction only to define the fixed candidate
  set and connected groups.
- Replaces the old `auto/lfg` case dimension with five controlled synthetic
  scenario p_env maps.
- Uses fixed-control GridFM evaluation at `decision_vector.u_base`.
- Does not add SciPy recourse, alpha optimization, or physics-infeasibility
  objective terms.
- Preserves the current fixed-control load-service proxy:
  `demand_weighted_load_shed_from_prediction(...)`.
- Saves source-less island diagnostics separately for interpretation.
- Runs Stage D exhaustive `k<=2` and Stage E constrained `K=2`.
- Computes Stage E best topology by true evaluated objective, not the Gurobi
  proxy objective.
- Writes Gurobi proxy objective/components separately from true evaluated
  objective/components.
- Computes absolute and relative Stage E-vs-Stage D objective gaps.
- Generates diagnostic Pareto plots from saved CSVs only.

Scenario definitions:

```text
S1: targets [27, 32, 101, 36]
S2: targets [23]
S3: targets [18, 27, 16, 19, 22], suppress [23]
S4: targets [77, 79]
S5: targets [27, 32, 36, 101, 47, 51, 77, 79, 88, 91], suppress [23]
```

S3 note:

- The initial S3 proposal overlapped S1 too strongly.
- The implemented S3 uses a more connected G1 corridor set around buses
  1/4/5/6: `[18, 27, 16, 19, 22]`.

Verification completed:

```powershell
# syntax parse without writing __pycache__
.\.venv\Scripts\python.exe - <source-parse-check>

# no-bytecode CLI import/help check
$env:PYTHONDONTWRITEBYTECODE='1'
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_f_decision_quality/run_stage_f_decision_quality.py --help

# manual invariant checks
# - exactly five scenarios
# - p_env targets high and non-targets low
# - S3 is not S1 repeated
# - K<=2 subset count is 562 for the 33-line candidate set
# - absolute and relative gap math works
```

Verification limitations:

- `.venv` does not currently have `pytest`, so the focused Stage F pytest file
  was added but not run through pytest.
- `py_compile` was blocked by Windows access to the new package `__pycache__`,
  so syntax was checked by parsing source without writing bytecode.
- Full Stage E execution still needs to run from a Gurobi-licensed `caleb`
  shell if the Codex tool process hits the license user mismatch.

Smoke command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR=(Resolve-Path .\.mplconfig).Path
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_f_decision_quality/run_stage_f_decision_quality.py --models gnn --scenarios S1 --lambda-cases balanced --stage-e-budget 5
```

Full run command:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:MPLCONFIGDIR=(Resolve-Path .\.mplconfig).Path
.\.venv\Scripts\python.exe experiments/test/wildfire_tests/stage_f_decision_quality/run_stage_f_decision_quality.py --models gnn
```

## June 23, 2026 - Physics Case-Study Fixed-Control Load-Shedding Correction

The following fixed-alpha physics-infeasibility result families were generated
before the load-shedding correction and are now superseded:

```text
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/rho0_no_physics/run_20260623_114541/
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/rho100_with_physics/run_20260623_115025/
```

These runs use `auto_env`, `gnn`, the `t0p30` candidate set, and a full lambda
sweep:

```text
lambda_R = 0.00, 0.05, ..., 1.00
lambda_L = 1 - lambda_R
```

For each lambda and `rho_phys` setting, the runs contain one Stage C heuristic
topology, all 562 Stage D `k<=2` topologies, 100 constrained Stage E `K<=2`
proposals, and 100 unconstrained Stage E proposals. All saved rows completed
successfully.

Important interpretation correction:

- These completed results do not run SciPy continuous recourse over
  `[Delta_Pg, alpha]` within each topology.
- They evaluate each topology at the clipped fixed control
  `decision_vector.u_base`.
- The saved `L_shed` was computed from the fixed control vector using
  `decision_vector.demand_weighted_load_shedding(u_base)`.
- Since the baseline load-service variables normally remain `alpha=1`, this
  produces `L_shed=0` for Stage C, Stage D, and constrained Stage E even when
  transmission lines were de-energized.
- Nonzero values in the unconstrained study occur only when source-less-island
  handling forces selected `alpha` values to zero.

Therefore, zero saved load shedding does not mean that no de-energization
topologies were evaluated. It means that the current fixed-control load-shedding
metric does not measure the service degradation predicted by GridFM after the
topology change.

The current result files remain useful for topology-search behavior, wildfire
exposure, physics residuals, and physics-penalized objective diagnostics.
However, they should not be used as the final wildfire-risk-versus-load-shedding
Pareto result. Stage-wise Pareto fronts based on these saved `L_shed` values are
largely degenerate.

These superseded directories were subsequently removed by the corrected full
regeneration documented in the next section.

The corrected fixed-control methodology should avoid expensive inner continuous
optimization while measuring topology-induced service loss:

```text
1. De-energize the proposed transmission lines.
2. Run GridFM once on the modified topology at fixed u_base.
3. Read the predicted served-load/service fraction at each demand bus.
4. Compute:

   L_shed_prediction =
       sum_i Pd_base_i * (1 - service_fraction_i)
       ------------------------------------------------
                       sum_i Pd_base_i

5. Evaluate:

   J_true =
       lambda_R * R_norm
       + lambda_L * L_shed_prediction
       + rho_phys * PAC_total
```

The service fractions must be clipped to `[0,1]`. This quantity should be
described as GridFM-predicted load shedding after de-energization, not as
optimal load shedding from redispatch or continuous recourse.

Before publishing the intended stage-wise Pareto and traditional-lambda
convergence plots, regenerate or rescore the physics case study using this
prediction-based load-shedding metric. The plots should then be reconstructed
as follows:

- Compute a separate nondominated `(R_norm, L_shed_prediction)` frontier for
  Stage C, Stage D, constrained Stage E, and unconstrained Stage E within each
  fixed `rho_phys` family.
- Show evaluated topologies in gray and connect each stage's nondominated
  points with one red frontier line. Stage C is a one-time heuristic and will
  normally appear as one frontier marker.
- For `lambda_R = 0.8, 0.5, 0.2`, show cumulative best `J_true` by topology
  evaluation for each stage, with the final best value and topology marked.
- Compare objectives only when `lambda_R`, `lambda_L`, `rho_phys`, and the
  load-shedding definition are fixed and identical.

## June 23, 2026 - Corrected Physics Case Study Regenerated

The physics-infeasibility case study was regenerated from scratch using the
corrected fixed-control load-shedding methodology. The prior fixed-alpha result
families and the validation smoke outputs were cleared before the final run.

Final result families:

```text
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/rho0_no_physics/run_20260623_220854/
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/rho100_with_physics/run_20260623_221317/
```

Final command:

```powershell
python -m experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_physics_infeasibility_case_study --cases auto_env --rho-phys 0 100 --lambda-step 0.05 --stage-e-k2-budget 100 --stage-e-unconstrained-budget 100 --no-continuous-recourse --clear
```

The run used the licensed Gurobi `caleb` user and completed without proposal or
license failures.

Final fixed-control evaluation methodology:

```text
1. Apply each proposed line-de-energization topology.
2. Run GridFM once at clipped decision_vector.u_base.
3. Compute R_norm from post-topology loading.
4. Compute L_shed from GridFM-predicted served demand:

   sum_i Pd_base_i * (1 - clip(Pd_pred_i / Pd_base_i, 0, 1))
   ---------------------------------------------------------
                        sum_i Pd_base_i

5. Compute PAC_total from voltage, thermal, generator, and source-less-island
   residual components.
6. Evaluate:

   J_true = lambda_R * R_norm + lambda_L * L_shed + rho_phys * PAC_total
```

Continuous recourse over `[Delta_Pg, alpha]` remained disabled. These results
measure GridFM-predicted load shedding after de-energization under fixed
controls; they do not claim optimal redispatch or optimal load shedding.

Per result family:

```text
total rows:                    16,023
Stage C rows:                      21
Stage D exhaustive rows:       11,802 = 21 * 562
Stage E constrained rows:       2,100 = 21 * 100
Stage E unconstrained rows:     2,100 = 21 * 100
successful rows:               16,023
proposal_failed rows:               0
lambda_R values:                   21 = 0.00..1.00 step 0.05
continuous_recourse_optimized:  false
```

The corrected load-shedding metric now varies across topology evaluations:

```text
Stage C distinct L_shed values:                  1
Stage D distinct L_shed values:                561
Stage E constrained distinct L_shed values:    195
Stage E unconstrained distinct L_shed values:  545
```

Stage-wise nondominated frontier sizes in each rho family:

```text
Stage C:                  1
Stage D:                  9
Stage E constrained:     12
Stage E unconstrained:   41
```

Pareto nondominance is now computed independently for each stage from the union
of that stage's unique evaluated topologies across the complete lambda sweep.
The stage-wise Pareto figure uses four panels, gray evaluated topologies, and
one red nondominated frontier per stage.

The traditional-lambda convergence figure contains separate panels for:

```text
lambda_R = 0.8
lambda_R = 0.5
lambda_R = 0.2
```

Each panel shows Stage C as a one-time horizontal reference and cumulative-best
objective traces for Stage D, constrained Stage E, and unconstrained Stage E.
Red markers identify each method's final best value, and the corresponding
topologies are listed inside the panel.

Validation:

```text
focused physics case-study tests: 9 passed
objective formula maximum absolute saved-row error:
  rho_phys=0:   6.66e-16
  rho_phys=100: 3.64e-12
traditional lambda summary rows per family: 12
Stage E versus Stage D gap rows per family: 21
all expected plots present
```

## June 23, 2026 - Stage E Topology Plus Continuous-Optimization Runtime Profile

Added a separate internal analysis runner for the currently proposed Stage E
topology-control plus continuous-optimization methodology:

```text
experiments/test/wildfire_tests/analysis/profile_stage_e_topology_continuous_optimization.py
```

Analysis result root:

```text
ieee_30_stage_a_to_i_results/stage_e/analysis/topology_continuous_optimization_profile/
```

The profiler fixes one constrained `K<=2` topology and records:

- SciPy objective evaluations and GridFM calls,
- pure `runner.predict` time,
- topology mutation plus state-extraction overhead,
- remaining SciPy/Python overhead,
- the objective-call trace,
- the optimized control movement and final objective.

Profile command:

```powershell
python -m experiments.test.wildfire_tests.analysis.profile_stage_e_topology_continuous_optimization --shutoff-line-ids 18 23 --lambda-r 0.8 --rho-phys 0 --optimizer-maxiter 10
```

Profile output:

```text
ieee_30_stage_a_to_i_results/stage_e/analysis/topology_continuous_optimization_profile/run_20260623_233144/
```

Measured result:

```text
topology: [18,23]
continuous optimization runtime: 2.053 seconds
GridFM calls: 235
mean GridFM inference: 0.00779 seconds
total pure GridFM inference time: 1.831 seconds
total topology prediction/state pipeline time: 1.984 seconds
topology/state overhead outside inference: 0.153 seconds
remaining SciPy/Python overhead: 0.069 seconds
SciPy success: true
```

Objective/control result:

```text
fixed-control J_true at first objective call: 0.1664025095102349
optimized final J_true:                     0.16640228488197123
minimum traced J_true:                      0.1664020636666643
max_abs_delta_pg:                           0.0
mean_alpha:                                 0.9997745668309387
min_alpha:                                  0.9989667049718864
```

Interpretation:

- A single GridFM inference is fast.
- Continuous optimization is expensive because finite-difference SciPy
  optimization called GridFM 235 times for one topology.
- For this topology, the continuous controls moved only slightly and changed
  the objective by less than `5e-7`.
- Projecting this profile directly gives about 205 seconds, or 3.4 minutes, for
  100 topologies at the same optimizer settings, excluding model setup and
  Gurobi proposal time. Runtime can vary substantially with convergence.

Same-topology evaluator parity was also checked across the corrected physics
results. For every topology evaluated by both Stage D and Stage E under the
same lambda and `rho_phys`, the maximum absolute differences in `J_true`,
`J_no_phys`, `R_norm`, `L_shed`, and `PAC_total` were all exactly zero.

Current Stage D and Stage E physics results therefore differ only in topology
search:

```text
Stage D: exhaustive enumeration of all k<=2 topologies
Stage E constrained: Gurobi-proposed 100-topology K<=2 search
Stage E unconstrained: Gurobi-proposed 100-topology unrestricted search
```

All three currently evaluate each topology once at fixed `u_base`; none runs
continuous optimization within each topology. GridFM supplies the predicted
post-topology state and load service, but GridFM itself is not performing an
inner optimization in these fixed-control results.

## June 24, 2026 - Traditional-Lambda Continuous Physics Study Completed

Implemented and completed the final Stage E topology-control plus continuous-
optimization study for the current work session.

Implementation:

```text
experiments/test/wildfire_tests/stage_e_gurobi_implementation/
  run_physics_continuous_traditional_lambdas.py

tests/test_wildfire_physics_continuous_traditional.py
```

Canonical continuous result families:

```text
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/
  with_continuous_optimization/
    rho0_no_physics/run_20260624_000543/
    rho100_with_physics/run_20260624_003815/
```

The fixed-control result families remain the comparison surface at:

```text
ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/
  without_continuous_optimization/
    rho0_no_physics/run_20260623_220854/
    rho100_with_physics/run_20260623_221317/
```

The intended final conceptual organization is:

```text
physics_infeasibility_case_study/
  without_continuous_optimization/
    rho0_no_physics/
    rho100_with_physics/
  with_continuous_optimization/
    rho0_no_physics/
    rho100_with_physics/
```

The fixed-control folders are now organized under
`without_continuous_optimization/`. Their contents and canonical run timestamps
remain unchanged.

Study matrix:

```text
case: auto_env
model: gnn
candidate set: t0p30

lambda_R = 0.8, 0.5, 0.2
lambda_L = 1 - lambda_R

rho_phys = 0
rho_phys = 100

Stage C:                  1 topology per lambda/rho
Stage D:                562 exhaustive k<=2 topologies per lambda/rho
Stage E constrained:    100 K<=2 topologies per lambda/rho
Stage E unconstrained:  100 unrestricted topologies per lambda/rho
```

Continuous decision:

```text
u = [Delta_Pg, alpha]
3 selected generator buses: [1, 10, 12]
5 selected load buses:      [6, 20, 11, 29, 18]
```

Each topology receives a hard budget of 100 GridFM objective calls. The method
explicitly retains the lowest valid objective call observed within the
topology, even when SciPy's final point is worse or the call budget is reached
before formal convergence.

The true continuous objective is:

```text
J_true(z,u) =
    lambda_R * R_norm(z,u)
    + lambda_L * L_shed_prediction_system_wide(z,u)
    + rho_phys * PAC_total(z,u)
```

`L_shed_prediction_system_wide` is computed from GridFM-predicted served demand
over all load buses. Alpha-commanded shedding is retained as a diagnostic, not
used as the primary load-shedding objective:

```text
L_shed_prediction =
    sum_i Pd_base_i * (1 - clip(Pd_pred_i / Pd_base_i, 0, 1))
    ---------------------------------------------------------
                         sum_i Pd_base_i
```

Nested selection:

```text
within topology:
  retain the best valid J_true among at most 100 continuous-control calls

within stage/lambda/rho:
  compare retained topology objectives and select the best topology
```

Per continuous result family:

```text
topology optimizations:            2,289
Stage C rows:                          3
Stage D rows:                      1,686 = 3 * 562
Stage E constrained rows:            300 = 3 * 100
Stage E unconstrained rows:          300 = 3 * 100
best-by-stage/lambda rows:             12
Stage E versus Stage D gap rows:        6
alpha diagnostic rows:             11,445 = 2,289 * 5 selected load buses
invalid GridFM calls:                   0
```

Call and runtime totals:

```text
rho_phys=0:
  objective/GridFM calls: 199,855
  wall runtime:           1,951.85 seconds = 32.53 minutes
  budget-limited rows:   1,954
  SciPy-converged rows:    335

rho_phys=100:
  objective/GridFM calls: 227,418
  wall runtime:           2,304.20 seconds = 38.40 minutes
  budget-limited rows:   2,247
  SciPy-converged rows:     42

combined wall runtime: approximately 70.9 minutes
```

Validation:

```text
focused physics tests: 13 passed
continuous-study focused tests after plot revision: 4 passed
all saved J_true values finite
invalid continuous objective calls: 0
maximum GridFM calls per topology: 100
best saved topology objective equals minimum valid call-trace objective:
  maximum absolute difference = 0.0
objective arithmetic maximum absolute error:
  rho_phys=0:   3.33e-16
  rho_phys=100: 3.64e-12
call-trace row count equals summed topology call count in both families
all expected alpha, runtime, gap, metadata, config, and plot outputs present
```

Principal outputs per family:

```text
combined_continuous_results.csv
best_within_topology.csv
best_by_stage_lambda.csv
continuous_objective_call_trace.csv
alpha_consistency_diagnostics.csv
stage_e_vs_stage_d_gap.csv
runtime_summary.csv
metadata.json
config.yaml
plots/traditional_lambda_objective_comparison.png
plots/alpha_consistency_comparison.png
plots/runtime_and_call_usage.png
```

The traditional-lambda objective figure is a topology-search convergence plot.
For each of the three lambda settings it shows:

```text
Stage C: one-topology horizontal reference
Stage D: cumulative best retained topology objective across 562 iterations
Stage E constrained: cumulative best across 100 topology iterations
Stage E unconstrained: cumulative best across 100 topology iterations
```

It does not show the internal 100-call continuous-control trace. Those traces
remain available in `continuous_objective_call_trace.csv` for audit.

Best topology results for `rho_phys=0`:

```text
lambda_R=0.8:
  Stage D / constrained Stage E: [18,23], J=0.166402
  unconstrained Stage E: [18,23,26,30,100], J=0.124769

lambda_R=0.5:
  Stage D / both Stage E methods: [14,23], J=0.227320

lambda_R=0.2:
  Stage D / both Stage E methods: [23,30], J=0.265656
```

Best topology results for `rho_phys=100`:

```text
lambda_R=0.8:
  Stage D / constrained Stage E: [23,27], J=3258.319712
  unconstrained Stage E: [23,27,30], J=3394.441712

lambda_R=0.5:
  Stage D / constrained Stage E: [23,27], J=3258.530875
  unconstrained Stage E: [23,26], J=3938.860182

lambda_R=0.2:
  Stage D: [23,27], J=3258.572165
  constrained and unconstrained Stage E: [23,38], J=3975.914452
```

Interpretation boundary:

- Stage D remains the exhaustive `k<=2` reference within the GridFM surrogate
  and the selected candidate topology space.
- Constrained Stage E exactly matched Stage D for all three `rho_phys=0`
  lambdas and for `rho_phys=100` at lambda_R 0.8 and 0.5.
- Under `rho_phys=100`, neither 100-topology Stage E search recovered the
  Stage D best at lambda_R 0.2.
- These are 100-call-budget continuous-optimization results, not guaranteed
  globally converged continuous solutions.
- The alpha consistency diagnostics show substantial differences between
  commanded selected-bus service and GridFM-predicted service. The final
  load-service definition is now an active methodology question: alpha-commanded
  `Pd` and `Qd` are explicit input interventions, while predicted `Pd` should
  not be treated as physically enforced recourse without further validation.

## Stage H Heuristic Baseline Comparison - Top-K TH Follow-Up

Session note, July 3, 2026:

We implemented and completed a Stage H heuristic comparison against the Stage G
revised continuous formulation. The comparison uses the same five
decision-quality scenarios, the same revised continuous recourse evaluator, and
the same objective accounting:

```text
J_no_phys = lambda_R * R_norm + lambda_L * L_shed
J_true    = J_no_phys + rho_phys * PAC_total
```

The initial paper-inspired transmission heuristic used percentile thresholds
over scenario-specific baseline wildfire scores:

```text
score_l = p_env_l * baseline_loading_l^2
```

That percentile rule exposed a tie issue: for S1, thresholds from 60 percent
through 85 percent landed on the same calibrated score block and selected the
same topology. We therefore switched TH to a rank-based top-k sweep:

```text
TH settings: top 5, top 4, top 3, top 2, top 1 scored candidate lines
```

The area heuristic remains a connected-network analogue of the paper's area
idea: select the top 30 percent scored candidate lines per scenario, form
connected components in network topology, choose the connected component with
the highest average score, and shut off all lines in that component. This is not
a geographic area heuristic because the small-grid setup does not define
geographic regions.

Final top-k run:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
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

The Stage G comparison reference includes all three completed Stage G continuous
settings:

```text
stage_d_k2_exhaustive
stage_e_k2
stage_e_unconstrained
```

The per-rho Stage H plot folders were condensed to avoid duplicate graph
families:

```text
expected_vs_selected_shutoff_lines.png
pareto_frontier_scatter.png
traditional_lambda_objective_convergence.png
th_topk_sensitivity.png
num_shutoffs_vs_objective_by_method.png
```

The Pareto plot axis convention was corrected to match Stage G:

```text
x-axis: R_norm
y-axis: L_shed
```

Interpretation:

- Stage H is a sparse policy comparison, not a topology-frontier search. TH has
  five fixed top-k policies per scenario and AH has one connected-component
  policy per scenario, so it cannot produce the dense Pareto clouds seen in the
  Stage G continuous topology-search run.
- Stage E unconstrained is the most consistently strong formulation baseline
  across the five scenarios and both rho settings.
- Stage D exhaustive remains very strong, especially in S1 and S3, because it is
  an exhaustive `k<=2` reference within the candidate topology space.
- TH top 4 is competitive in S2 and S4, ranking first under both rho settings.
  This suggests that simple ranked-risk shutoff can work well when the
  high-scoring lines align with the useful outage topology.
- AH is generally weaker than Stage G and the better TH settings, except in S3
  at `rho=0`, where the connected high-risk component aligns relatively well
  with the scenario structure.
- TH top 1 and AH often coincide when the AH selected component is a single
  highest-risk line; in those cases AH behaves like a one-line TH rule.

Average-`J_true` ranking by scenario and rho:

```text
S1 rho=0:
  Stage D, Stage E unconstrained, Stage E k2, TH top 4, TH top 2,
  TH top 5, TH top 1, AH, TH top 3

S1 rho=2:
  Stage D, Stage E unconstrained, Stage E k2, TH top 2, TH top 4,
  TH top 1, AH, TH top 5, TH top 3

S2 rho=0:
  TH top 4, Stage E unconstrained, TH top 3, Stage E k2, Stage D,
  TH top 5, AH, TH top 1, TH top 2

S2 rho=2:
  TH top 4, Stage E unconstrained, TH top 5, TH top 3, Stage D,
  Stage E k2, AH, TH top 1, TH top 2

S3 rho=0:
  Stage D, AH, Stage E unconstrained, TH top 5, TH top 4, TH top 3,
  Stage E k2, TH top 1, TH top 2

S3 rho=2:
  Stage D, Stage E unconstrained, Stage E k2, AH, TH top 5,
  TH top 1, TH top 3, TH top 4, TH top 2

S4 rho=0:
  TH top 4, Stage E unconstrained, TH top 5, Stage D, Stage E k2,
  TH top 2, TH top 3, AH, TH top 1

S4 rho=2:
  TH top 4, Stage E unconstrained, TH top 5, Stage D, Stage E k2,
  TH top 2, TH top 3, AH, TH top 1

S5 rho=0:
  Stage E unconstrained, TH top 5, Stage D, Stage E k2, TH top 2,
  TH top 4, TH top 3, AH, TH top 1

S5 rho=2:
  Stage E unconstrained, Stage D, Stage E k2, TH top 2, TH top 4,
  TH top 3, TH top 1, AH, TH top 5
```

Checkpoint conclusion:

The Stage H comparison supports the value of the Stage G formulation. The
formulation-based Stage E unconstrained decisions are more consistent than the
paper-inspired heuristics, while rank-based TH is a useful simple comparator
that can win in specific scenarios. The comparison also clarifies that heuristic
sweeps should be interpreted as sparse policy points, not Pareto-frontier
searches.

## July 11, 2026: Revised Load/PAC Stage H Session Close

Implemented and regenerated the revised GridFM-side load-shedding and physics
infeasibility accounting before moving to DC MILP construction.

Final revised Stage H run retained for this checkpoint:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/
  heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/
```

Prior comparison baseline retained for before/after interpretation:

```text
experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/
  heuristic_baseline_comparison_topk/run_gnn_20260703_034242/
```

The revised run produced complete `tables/`, `plots/cross_rho/`,
`plots/per_rho/`, and `plots/summary/` outputs. The summary plot family now
includes the revised diagnostic views:

```text
average_operational_ac_model_pac_by_method.png
load_shed_cmd_gridfm_hybrid_by_method.png
model_alignment_violation_by_method.png
ac_balance_violation_by_method.png
```

Methodology updates represented in the revised run:

- The objective load term now uses hybrid effective load shedding:
  `L_shed = L_shed_hybrid`.
- `L_shed_cmd`, `L_shed_gridfm_raw`, `L_shed_gridfm_effective`, and
  `L_shed_hybrid` are all saved for comparison.
- Source-less islanded buses are forced unserved.
- Selected connected load buses use commanded alpha.
- Non-selected connected load buses use GridFM effective alpha, clipped to
  `[0, 1]`.
- PAC is decomposed into operational, AC, and model-consistency groups:
  `PAC_total = PAC_operational + PAC_AC + PAC_model_consistency` under default
  group weights of 1.
- P/Q AC balance diagnostics are available in the revised run.
- Branch-flow consistency remains unavailable because GridFM does not expose an
  independent branch-flow/loading prediction channel; it is saved as unavailable
  rather than silently treated as a zero violation.

Focused validation:

```text
python -m pytest tests/test_wildfire_stage_g_revised_continuous.py tests/test_wildfire_stage_h_heuristic_comparison.py -q
18 passed
```

Finalization checks:

```text
topology rows before rho expansion: 90
continuous evaluations:             180
hard methodology failures:           0
Stage D/E reference rows included:   90
TH/AH heuristic rows evaluated:      180
```

Key numerical interpretation from the retained revised run:

```text
Prior heuristic baseline:
  rho=0: J_true 5.7299, L_shed 0.0388, PAC_total 2.1621
  rho=2: J_true 10.0541, L_shed 0.0386, PAC_total 2.1621

Revised heuristic accounting:
  rho=0: J_true 5.8503, L_shed 0.2790, PAC_total 169.5884
  rho=2: J_true 310.6523, L_shed 0.3663, PAC_total 152.3797
```

The revised load-shedding decomposition shows the previous commanded/effective
metric understated service loss at non-selected connected load buses:

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

The revised PAC decomposition shows that the old PAC was a partial score and
that the newly exposed infeasibility is dominated by generator/model-consistency
terms rather than voltage or thermal terms alone:

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

Important comparison caveat:

Stage H still loads the existing Stage G reference rows for
`Stage D exhaustive`, `Stage E k2`, and `Stage E unconstrained`. Those reference
rows preserve the historical Stage G reference accounting unless regenerated
under the revised evaluator. The revised TH/AH rows are valid for diagnosing the
new load/PAC treatment, but a fully apples-to-apples final comparison against
Stage D/E should regenerate the Stage G reference set with the same revised
hybrid load and PAC decomposition.

Session conclusion:

The revised run confirms the methodological issue that motivated this session.
The prior baseline understated effective load shedding and underrepresented
physics infeasibility. The retained revised run is the current GridFM checkpoint
to use before DC MILP construction and fairer scenario redesign.

## August 17, 2026: Stage J GOC-500 Complete Comparison

Completed the first full GOC-500 Stage J result package using the frozen
GridSFM-Open economic-AC-OPF surrogate, Guided-DC economic recourse, and the
transparent TH GridSFM heuristic.

```text
results root:
experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run/
```

Locked experiment settings:

```text
scenarios: J-S1, J-S2, J-S3
lambda_R = lambda_R_proxy: [0, 0.2, 0.5, 0.8, 1]
K <= 2
topology budget: 100
continuous budget: 20 per topology
q = 5 selected load buses for the bounded Powell correction
```

Every one of the 15 setting bundles completed Guided-DC, Guided-GridSFM,
TH-GridSFM top-1/top-2, fixed-z/fixed-alpha economic AC-OPF (Reference A),
fixed-z AC MLD plus economic tie-break (Reference B), and four warm starts.
The retained aggregate evidence comprises 3,030 topology rows, 60,600 alpha
evaluations, 60 finalists, 60 Reference A rows, 120 Reference B rows, and 240
warm-start rows.

Implementation refinements made during the full run:

- the complete-run executor is resumable and rebuilds aggregate tables from all
  completed cache settings rather than a partial rerun subset;
- final evidence copying uses portable extended Windows paths and keeps full
  requested/effective alpha vectors for every finalist;
- aggregate AC audit rows include scenario, lambda, and finalist-topology
  provenance;
- J8 treats a valid run with rejected DC-infeasible candidates as
  `PASS_WITH_INFEASIBLE_CANDIDATES` rather than a method-level failure;
- a J-S3 source-less-load index check and failure-message JSON escaping defect
  were repaired before rerunning only the affected exact AC reference bundles.

The implementation audit recorded `PASS_WITH_LIMITATIONS`. The core limitation
is that `q=5` is an approved selected-load approximation to the original full
per-load alpha formulation. GridSFM `model_output_penalized` rows are valid
surrogate evaluations with PAC, not AC-feasibility certification. The final
package retains 387 DC-infeasible rejections and 12 Gurobi no-incumbent events;
none was selected as a Guided-DC finalist.

## August 22, 2026: GridSFM FT1 Training And FT2 Held-Out Evaluation

Completed the next two controlled steps of the Stage J GridSFM fine-tuning
extension while leaving the completed Stage J experiment unchanged.

FT1 used the released GridSFM v1.1 checkpoint, FullTop GOC-500 train indices
0 through 999, `SyntheticMixedDataset(infeas_prob=0.3)`, batch size 8, ten
epochs, learning rate `1e-4`, weight decay `1e-4`, and seed 42. Every epoch
recorded official held-out metrics on all 750 FullTop validation graphs. No
training batch was skipped. The run took 9,082.9 seconds, changed all 1,221
floating parameter tensors, and passed a fresh-process reload.

```text
FT1 checkpoint:
C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\
  gridsfm_goc500_fulltop_ft_n1000.pt

SHA-256:
A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD
```

FT2 downloaded and processed one official 15,000-graph GOC-500 N-1 OPFData
group. It then used official `gridsfm.eval_pass` to compare released v1.1 and
FT1 on the same ordered FullTop test indices 0 through 749 and N-1 test indices
0 through 749. Both splits were inference-only and excluded from FT1.

Headline relative changes from released v1.1 to FT1:

```text
                 FullTop test   N-1 test
loss               -55.525%     -37.994%
cost MAPE          -16.093%     -24.726%
Pg MAE             -40.135%     -33.662%
Qg MAE             -61.270%     -52.820%
V MAE              -34.391%     -26.882%
theta MAE          -53.373%     -51.014%
branch P MAE       -45.580%     -43.025%
branch Q MAE       -56.494%     -50.182%
```

FT1 improved KCL and thermal metrics on both variants. FullTop feasibility
accuracy remained 1.0. N-1 feasibility accuracy declined from `0.998667` to
`0.993333`, or one miss versus five, and is retained as the immediate model
behavior divergence. FT2 therefore closed as
`FT2_EVALUATED_READY_FOR_REVIEW`, not as an unconditional replacement claim.

The pinned GridSFM commit, package environment, private API source hashes, and
direct checkpoint paths/hashes are retained in the FT1 and FT2 manifests. FT3
Stage J application remains a separate gate. A future FullTop+N-1 model may be
trained only with these two FT2 test splits kept sealed.

## August 22, 2026: GridSFM FT3/FT4 Stage J Application

Completed the linked FT3/FT4 experiment using only the FT1
`fulltop_ft_n1000` checkpoint. The frozen v1.1 and Guided-DC packages were
retained as existing comparison evidence and were not rerun. FT3 reused all 30
ordered Guided/TH topology pools byte-for-byte across 15 scenario/lambda
settings, then generated exact Reference A/B, state-fidelity, and four-way
warm-start evidence for 45 new FT finalists.

Validated aggregate counts were 1,530 topology rows, 30,600 candidate
evaluations, 45 finalists, 45 Reference A rows, 90 Reference B rows, 180
warm-start rows, and 360 state-fidelity rows. FT4 reproduced 13 nonempty figures
and four derived visual tables. All core rows identified the FT model variant
and checkpoint provenance, and all validation checks passed.

The authoritative CSV working package is retained outside OneDrive at
`C:\Users\Caleb Lu\.gridfm_stage_j\results\stage_j\
fulltop_ft_n1000_complete_run`. The Git checkpoint is
`stage_j_gridsfm_goc500/finetune/results/ft3_ft4_v002`. Its publication process
converts CSV to Parquet only after validation, retains all other evidence, and
records source/output paths, hashes, sizes, formats, and row counts in
`PUBLICATION_MANIFEST.json`. The verified publication contains 380 artifacts,
299 Parquet tables, 102,907 table rows, no CSV files, and no hash failures. It
reduced 94.4 MB of working artifacts to 14.0 MB.

Implementation observations retained for future runs:

- protect long experiment caches with an exclusive single-writer lock;
- keep authoritative working results outside OneDrive and publish validated
  compact artifacts into Git afterward;
- use short deterministic path IDs plus a manifest for deep Windows result
  trees that would exceed path limits;
- derive plot method panels from methods present so FT-only packages do not
  render blank historical-comparator panels;
- preserve the pinned `gurobipy==12.0.0` dependency required by J9's unchanged
  DC partial warm-start construction.

Cross-model scientific interpretation is deferred to FT5, which will join the
FT package to the existing released-v1.1 and Guided-DC packages using scenario,
lambda, opportunity-set, method-family, and provenance keys.

## August 22, 2026: FT5 Warm-Start Work Paused With Resume Contract

Added explicit audited cold initialization (`V=1`, `theta=0`, midpoint `Pg`,
zero `Qg`) and full-state GridSFM initialization (`Pg`, `Qg`, `V`, `theta`) to
the exact Reference A warm-start workflow. The fine-tuned package augmentation
completed for all 15 settings. The matching frozen augmentation was stopped at
the user's ten-minute boundary with four settings complete and no workers left
running.

The final study is explicitly a 5 finalist-family by 5 start-policy crossed
comparison. Frozen and fine-tuned GridSFM starts will be evaluated separately
against every retained finalist and displayed on one common runtime axis. The
complete resumable execution and validation checklist is retained in
`FT5_CROSSED_WARM_START_HANDOFF.md` in the Stage J workflow case.

## August 23, 2026: FT5 Crossed Warm-Start Comparison Complete

Resumed the frozen full-state augmentation offline, completed all 15 settings,
and added a checkpoint-specific full-warm append mode. The final crossed design
applied cold, DC partial, frozen GridSFM full, fine-tuned GridSFM full, and exact
starts to each of five retained finalist families without changing topology,
alpha, or Reference A formulation.

All 375 exact solves converged, every family/start cell contains 15 settings,
and maximum within-instance objective spread was `8.406234e-06` against the
`1e-3` gate. The final plot uses one common solver-runtime y-axis and separates
both GridSFM checkpoints. Fine-tuned initialization improved the fine-tuned
Guided finalist relative to frozen initialization, but frozen initialization
was faster for the other four families. DC partial initialization was the most
consistent simple improvement over cold.

The authoritative interpretation, paired values, provenance contract, and
limitations are recorded in `FT5_CROSSED_WARM_START_STATUS.md`. The raw working
package remains outside OneDrive and the Git publication continues to convert
validated CSV evidence to Parquet.

The post-FT5 publication contains 392 artifacts and 308 Parquet tables. It
retains 200,305 table rows in 22.7 MB versus 187.8 MB of source evidence. The
canonical 375-row warm-start Parquet passed read-back comparison against the
working CSV.

## August 24, 2026: FT5 Runtime Metric Corrected To Native IPOPT Solve Time

Reran all 375 crossed warm-start Reference A solves with unchanged fixed
topologies, alpha decisions, initialization payloads, formulation, and solver
options. The Julia reference wrapper now records InfrastructureModels'
`result["solve_time"]`, sourced from `JuMP.solve_time` / `MOI.SolveTimeSec`, in
addition to the broader elapsed PowerModels call. This isolates IPOPT after
model construction and initial-value loading. GridSFM inference and start
construction remain outside the primary timing boundary.

All solves succeeded. Every one of the 25 finalist/start cells contains 15
settings, maximum within-instance objective spread remained `8.406234e-06`,
and maximum objective delta from the prior run was `5.820766e-11`. The FT4 and
timing-specific validations passed.

The corrected aggregate median IPOPT times are cold `5.045` s, DC `4.740` s,
frozen GridSFM `4.872` s, fine-tuned GridSFM `4.882` s, and exact primal
`4.872` s. DC beat cold in 71/75 cases. Frozen and fine-tuned GridSFM each beat
cold in 69/75 cases. Fine-tuned beat frozen 40/75 times, with a marginal
`0.005` s median advantage. The prior conclusion that fine-tuned initialization
was broadly slower is superseded because it reflected PowerModels construction
and result overhead rather than native IPOPT solve time.

The existing common-axis warm-start figure now uses native IPOPT timing. A
second figure reports aggregate mean, median, interquartile spread, and paired
wins/losses. Exact remains a primal-only reference and does not restore IPOPT
dual variables or barrier state.

## August 24, 2026: FullTop FT Native Pareto Figures Expanded To All Candidates

Replaced the three risk/load plots based on one retained alpha per topology
with empirical Pareto frontiers derived from all eligible candidate-level
evidence. For each scenario and method, all five lambda-directed searches are
pooled because lambda controls discovery while the native `R_norm` and
`L_shed_total` coordinates remain candidate properties. Duplicate coordinates
are removed before minimizing both metrics and retaining nondominated points.

The source comparison contains 90,201 eligible finite evaluations. The final
derived table has 343 points over all 15 combinations of J-S1/J-S2/J-S3 and
the five methods. Guided fronts contain 18 to 50 points; TH fronts are often
singletons because their smaller evaluated pools are dominated. Those cases
are plotted as markers rather than artificial curves. The FT4 validator now
checks row count, complete group coverage, eligible statuses, metadata, and
per-group frontier-size consistency. No optimization, inference, or exact AC
solve was repeated.

## August 24, 2026: Sequential Fine-Tuning Ablation Split Into Two Gates

Frozen the concluding Stage J fine-tuning extension as FT6 and FT7. FT6 uses
the existing FullTop-1000 checkpoint as the byte-identical parent for two
sibling continuation models: 500 additional FullTop train cases and 500 N-1
train cases. Both branches use disjoint OPFData train and validation splits,
record FullTop and N-1 validation metrics every epoch, and join released v1.1
and FullTop-1000 in a four-model evaluation over one sealed test manifest with
375 FullTop and 375 N-1 cases.

FT6 terminates at a mandatory external review checkpoint. FT7 may not execute
until Caleb explicitly approves the two new checkpoint hashes and their FT6
results. The approved FT7 design reuses existing Guided-DC, released GridSFM,
and FullTop-1000 evidence, runs only the two new checkpoints, and produces
five-method OPS figures. TH remains intact in prior packages but is excluded
from all new follow-up figures and warm-start summaries. The revised crossed
warm-start contract contains five non-TH finalist families and seven starts for
525 fixed-instance Reference A rows. The complete manifests, internal
checkpoints, stop rules, runtime estimate, and two external gates are recorded
in `FT6_FT7_PLAN.md`.

## August 24, 2026: FT6 Sequential Fine-Tuning Ablation Complete

Completed the externally gated FT6 Part I study without changing the GridSFM
architecture, OPFData schema, AC-state generation, model loss, optimizer loop,
or official evaluation definitions. The preflight fingerprinted 2,500 selected
FullTop/N-1 training, validation, and test records and confirmed finite tensors,
within-stratum uniqueness, local-cache availability, and no graph-hash overlap
across train, validation, and sealed test uses.

M2 and M3 each branched directly from the existing FullTop-1000 checkpoint SHA
`A1378FDA...E119AD`. M2 trained on 500 additional FullTop cases and saved SHA
`08EDA702...A0C94FF6`; M3 trained on 500 N-1 cases and saved SHA
`4EC89D36...462F302D`. Both ten-epoch runs used a fresh official AdamW
optimizer, completed 630/630 batches with no skips, recorded separate 375-case
FullTop and N-1 validation metrics every epoch, changed all 1,221 floating
parameter tensors, and passed finite-weight, output-schema, checkpoint, and
fresh-process reload checks. The measured epoch-cycle runtimes were 76.2
minutes for M2 and 75.4 minutes for M3.

The sealed 375-FullTop plus 375-N-1 evaluation reran released v1.1, M1, M2, and
M3 through official `eval_pass` and passed in 19.0 minutes. M2 broadly improved
over M1 on every reported lower-is-better test metric. M3 produced the strongest
N-1 loss (`0.056797`), cost MAPE (`0.006347`), and feasibility accuracy
(`1.000`), but relative to M2 traded away several FullTop, reactive-flow,
Q-KCL, and thermal-overload measures. The retained interpretation is explicit
N-1 distribution adaptation with measurable tradeoffs, not unconditional
superiority.

N-1 training is supported by the pinned GridSFM `OPFDataAdapterDataset` and
official fine-tuning API. It is recorded as a direct supported extension, not
as a reproduction of the white paper's FullTop-only fine-tuning experiment.
The complete result packet is `FT6_SEQUENTIAL_FINETUNE_STATUS.md`; the terminal
state is `FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL`. No FT7 OPS work has begun.

## August 24, 2026: FT7 Refined Fine-Tuning Study Complete

Recorded Caleb's explicit approval and ran only M2/M3 through the unchanged
Stage J Guided-GridSFM OPS workflow. Both 15-setting packages passed counts,
checkpoint provenance, exact Reference A/B, and no-TH validation. Existing DC,
released M0, and FullTop-1000 M1 evidence was reused rather than rerun.

The five-method aggregate contains 150,000 candidate evaluations. Updated
risk/load figures pool all 149,601 eligible finite candidates across lambda per
scenario/model and retain 603 recomputed nondominated points. The 14 final
figures use explicit DC/M0/M1/M2/M3 labels and common axes where applicable.

Completed the five-finalist by seven-start crossed warm study with 525 rows.
All solves succeeded, all four GridSFM starts supplied full AC state, every
checkpoint SHA passed, and maximum objective spread was `8.406234e-06`. Native
IPOPT timing showed DC as the most consistent improvement over cold. M0/M1
were near cold in median; M2/M3 were slower despite stronger native/exact state
quality, so fine-tuning does not imply solver speedup.

M2 was strongest among GridSFM variants on the FullTop-oriented OPS diagnostic;
M3 improved over released M0 and generally M1, while retaining the FT6 N-1
specialization result. This is distribution-specific adaptation with tradeoffs.

Published `goc_500_results/stage_j/refined_finetune_study` with 18 Parquet
tables, 14 figures, zero CSVs, zero checkpoints, and a 12.97 MB largest file.
Parquet read-back, copy hashes, source hashes, row counts, approval, preflight,
disk, OPS, warm-start, and summary evidence are in the publication manifest.

## August 24, 2026: FT7 Controlled IPOPT Iteration Rerun

Superseded the mixed-batch FT7 warm-start timing aggregate with one controlled
rerun of all 525 fixed-instance solves. The seven starts were rotated across
execution positions; every start occupied each position 10 or 11 times. The
rerun completed in 32.6 minutes with four workers and passed all solve,
uniqueness, objective, timing-field, order-balance, and iteration-count gates.

Every initialization produced a positive IPOPT barrier-iteration count. Median
iterations were 33 for cold, 30 for DC, and 31 for M0, M1, M2, M3, and exact
primal. Mean IPOPT times were 5.221 seconds for cold, 5.151 for DC, 5.160 for
M0, 5.108 for M1, 5.104 for M2, 5.159 for M3, and 5.216 for exact. The prior
multi-second M2/M3 slowdown is superseded as a mixed-batch artifact. M1 and M2
are effectively tied by time, and all GridSFM checkpoints show modest median
iteration reductions without a strong checkpoint speed ordering.

The refreshed publication contains 48 files: 20 Parquet tables, 15 PNG
figures, 12 JSON records, and one Markdown summary. It includes a dedicated
iteration figure, iteration summary and paired tables, and the controlled-run
validation record. No CSVs or checkpoints are published.

## August 25, 2026: Stage J Integrated Research Summary Completed

Created `STAGE_J_FINAL_RESEARCH_SUMMARY.md` in the CASE-003 workflow directory
as the preferred paper-facing and implementation-facing entry point for Stage
J. The document consolidates the research questions, frozen OPS contract,
FT0-FT7 chronology, official GridSFM/OPFData compatibility boundary, data and
checkpoint provenance, sealed FullTop/N-1 evaluation, five-method OPS results,
all-candidate Pareto method, controlled IPOPT warm-start result, limitations,
code ownership map, artifact index, and rerun commands.

The summary explicitly distinguishes final evidence from superseded
intermediate timing conclusions. It records the 48-file publication as 38
primary manifest artifacts plus ten supporting status/provenance records and
retains Parquet as the required publication practice for large experiment
tables. No optimization, fine-tuning, inference, or exact AC solve was rerun.

## August 26, 2026: RQ1 Frozen M0 Evaluator Runtime Addendum

Added a paired evaluator-runtime benchmark to the Stage J RQ1 evidence. The
preflight preserved 75 provenance records while identifying 54 unique fixed
decisions with a canonical SHA-256 identity over the complete branch-status
vector, effective load-service vector, scaled active demand, and scaled
reactive demand. It also pinned the released M0 checkpoint, GridSFM commit,
raw GOC-500 case, and sealed FT7 source table.

Frozen M0 was measured with seven total-evaluator repetitions per decision and
adaptive core repetitions; Reference A used a persistent Julia process, three
warmups, and three fresh fixed-decision solves per decision. The primary timing
boundary starts when the initialized evaluator receives a fixed candidate and
ends when a usable electrical state is returned. Downstream wildfire scoring,
publication I/O, and one-time startup are excluded. Reference A retained its
existing `V=1`, `theta=0`, midpoint-`Pg`, zero-`Qg` initialization.

All 162 Reference A measurements solved locally with finite state and agreed
with the sealed exact objectives to within `1.56e-08`. Across the 54 paired
decisions, median total-evaluator time was `0.2973 s` for frozen M0 and
`1.0919 s` for Reference A, giving a `3.59x` median paired speedup (95% paired
bootstrap interval `3.48-3.68x`) and `3.61x` geometric-mean speedup. The core
forward-versus-IPOPT comparison gave `3.35x` median speedup. The result supports
M0 screening on this implementation, not an end-to-end OPS speedup claim.

The publication illustration was refined to use the M0-selected
`m0:s1_l0p8` finalist (`RQ1U031`), while all 54 unique decisions support the
timing inference. This case opens lines 276 and wildfire target 473, actively
curtails load 157, has no source-less loads, and returns predicted voltage
range `0.976-1.100 p.u.` and maximum branch loading `1.309 p.u.`. The shared-
layout before/after figure keeps decision overlays distinct from predicted
branch-loading and node-voltage encodings. The compact publication
under `goc_500_results/stage_j/rq1_frozen_m0_evaluator_study` uses Parquet for
tables and includes figures, summary/status records, hashes, and all three
phase-validation JSON files.

## September 21, 2026: Stage K Gate 2 Texas2k Package Implemented

Implemented the approved Stage K modified-Texas2k smoke package without
submitting PACE jobs. The frozen case uses Scenario 16, the June 23 16:00 CDT
`p_cumulative` snapshot, stored solved loading as the authoritative baseline,
and `R_base=32.97542569928573` over 3,993 controllable transmission lines.
Canonical preparation found 2,751 buses, 1,125 loads, 1,099 generators, 5,344
physical branches, and 1,351 fixed transformers.

Stage K directly reuses the Stage J connectivity/source-less `c_l`
implementation. All Texas2k single-line `c_l` values are zero. Exact-K proxy
separability was proved and tested, allowing deterministic K1 and parent-fixed
K2 ranking without Gurobi. Smoke/full configuration, GridSFM external alpha
search, native PowerModels DC/AC recourse, all-energized-line AC risk
epigraphs, sealed Reference A/B, exact discrepancy metrics, checkpoint/resume,
derived reporting, and four PACE job scripts were added.

Local canonical validation and 21 Stage K tests passed. Official GridSFM
preprocessing produced the expected Texas2k graph shapes; the released v1.1
checkpoint completed intact inference and a full five-alpha K1 candidate probe
on CPU. Julia/PowerModels/Ipopt execution, the intact AC consistency audit,
A100 inference, and live Phoenix resource syntax remain Gate 3 validations.
No smoke or production job was launched.

## September 21, 2026: Stage K GridSFM Environment Amendment

Added the missing reproducible Python/A100 contract before Gate 3. The package
now pins the official Microsoft GridSFM repository at commit
`1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`, GridSFM 1.1.0, Python 3.11.9,
torch 2.7.1+cu126, torch-geometric 2.6.1, NumPy 1.26.4, SciPy 1.15.3, and
huggingface-hub 0.34.4. It binds `microsoft/GridSFM_Open` checkpoint
`gridsfm_open_v1.1.pt` at repository revision
`1b41299b80252adf1869d5c3b479a4a402c52591` to SHA-256
`f8a4396122e603e8303afdebe3b093819c0f64dac0878394aed0bd63205fd831`.

The official package setup performs a one-time source checkout and checkpoint
cache, while inference jobs run with Hugging Face offline. GPU preflight now
requires exactly one visible A100 on `cuda:0`, CUDA runtime 12.6, real
environment-installed GridSFM and huggingface-hub packages, exact source and
checkpoint identities, and successful intact Texas2k inference. It records all
installed distributions and device metadata in an observed environment
manifest. The GridSFM batch job revalidates the contract before screening.
The temporary local Gate 2 import stub has no role in the PACE environment.
No PACE job was submitted.

## September 21, 2026: Stage K Gate 3 Deployment-Audit Amendment

Aligned CPU preflight with the frozen stored-baseline methodology after the
first Phoenix deployment probe. The newly solved intact economic AC-OPF is now
gated as a Julia/PowerModels/Ipopt and branch-export sanity check: accepted
termination status plus complete, unique, finite output for all 5,344 intact
physical branches. Its maximum loading discrepancy, weighted risk difference,
and largest discrepant branch IDs remain retained diagnostics but do not gate
PASS because economic redispatch is not required to reproduce the stored
Scenario-16 flow vector. The stored baseline, proxy weights, and `R_base` are
unchanged. The discovered deployment-only dependency `lightning==2.6.6` was
also added to the frozen GridSFM environment contract.

## September 21, 2026: Stage K Phoenix Resource Contract Verified

Aligned Gate 3 resource requests with Phoenix's resource-driven partition
assignment policy. A live request using `--constraint=graniterapids` reached
`cpu-gnr` and reported two Intel Xeon 6972P sockets, 192 cores, and about
1.5 TiB RAM. A live one-GPU request using `--gres=gpu:a100:1` was assigned to
`gpu-a100`; it was cancelled while pending for scheduler priority, so physical
A100 identity remains a mandatory check inside the eventual allocation.

The smoke runbook now omits hard-coded partition flags, records the verified
GNR and A100 resource arguments, and requires in-allocation hardware checks.
These probes did not run evaluator workloads or submit the Stage K smoke test.

The corrected Gate 3 CPU preflight was subsequently run on `cpu-gnr` and
passed. The intact deployment AC-OPF returned `LOCALLY_SOLVED` with all 5,344
physical branch outputs complete and finite; the stored Scenario-16 baseline
remained authoritative. The A100 GPU preflight request was routed to
`gpu-a100` but did not receive an allocation during the interactive queue
window and was cancelled. GPU identity, CUDA checks, checkpoint loading, and
intact Texas2k GridSFM inference therefore remain outstanding. No smoke job
was submitted, and no probe job was left queued or running.

## September 21, 2026: Stage K Gate 4 Phoenix Smoke Completed

Completed the Stage K smoke workflow on Phoenix after an explicitly approved
smoke-only move from queued A100 resources to an available V100. The frozen
GridSFM source, v1.1 checkpoint, Python environment, CUDA/device contract, and
intact Texas2k inference probe passed on the V100. GridSFM, native DC-OPF, and
native AC-OPF each evaluated the intact state, ten shared K1 states, and ten
evaluator-specific K2 states at `lambda_R=0.8`. All 63 candidate rows were
eligible; all DC/AC solves were locally solved, while all 21 released-GridSFM
states carried the frozen model-output penalty.

The selected topologies were GridSFM line 4872, DC line 2245, and AC lines
281/1950. Exact AC Reference A and maximum-delivery Reference B1 completed for
all three. GridSFM Reference B2 returned `OTHER_ERROR`, so its economic
tie-break is uncertified; its valid B1 result still establishes 100% maximum
service. Reporting now falls back to B1 physical diagnostics and emits
`PASS_WITH_WARNINGS` when B2 is not solver-eligible.

The smoke deployment exposed and fixed DC loading export through active power,
empty generation-cost diagnostics, stale-checkpoint handling, and GridSFM
runtime extrapolation. The corrected full serial screening estimates are 4.34
hours for GridSFM on V100, 4.01 hours for DC, and 51.47 hours for AC. Five-
lambda serial references are estimated at 7.97 hours. The three evaluator
families may run concurrently, but production AC should be split into
deterministic chunks, with references dependent on every successful chunk.

The combined reference/report job timed out only after all six reference solves
were written: postprocessing blocked in Phoenix `cl_sync_io_wait`. Immutable
Parquet outputs were retrieved and aggregation completed locally in under ten
seconds. The audited package is under `stage_k_smoke_audited`, and the detailed
Gate 4 interpretation is recorded in `stage_k_case_study/GATE4_SMOKE_TEST_REPORT.md`.
Thirty-one Stage K contract tests pass. Production remains a separate approval gate.

## September 22, 2026: Stage K Texas2k Production And Post-Run Analysis Completed

Completed the frozen five-lambda Texas2k production study on Phoenix using a
V100 for GridSFM and GNR CPUs for DC-OPF, AC-OPF, exact references, and
aggregation. Each evaluator attempted 1,505 states: intact plus 50 shared K1
and 250 evaluator-specific K2 candidates at each of five risk weights. GridSFM
retained all 1,505 candidates, DC retained 1,504, and AC retained 1,498. One AC
array task recovered from an 8 GB Slurm-memory failure by resuming its 301
deterministic checkpoints at 16 GB; no scientific configuration changed.

All 15 sealed finalists completed AC Reference A and Reference B1/B2. Every B1
and B2 solve was solver-eligible, every B1 certified essentially 100% maximum
service, and final validation passed with no failed checks or warnings. A final
validator batch initially exited after writing valid artifacts because two
Pandas reductions returned non-JSON-serializable NumPy booleans. The validator
now normalizes every check to a native Boolean; audit-only recovery job
13433845 completed with exit 0:0. The aggregate script now refreshes production
status after validation.

Post-run analysis is under `texas_2k_results/full_run`. It retains the primary
PACE tables, 14 supporting analysis tables, and ten figure families in PNG and
PDF. GridSFM selected topology 841/3268 for every positive risk weight and
preserved at least 99.80% selected service, but produced almost no exact AC risk
reduction. DC and AC found large exact AC risk reductions at higher weights but
selected substantial avoidable load shedding. Reference B1 showed every
selected topology could serve essentially all demand, separating topology
capability from continuous-recourse quality. Native-versus-Reference-A risk
discrepancies were material for all methods, confirming that exact AC auditing
must remain distinct from native screening metrics. Thirty-three Stage K tests
pass after the reporting additions.
