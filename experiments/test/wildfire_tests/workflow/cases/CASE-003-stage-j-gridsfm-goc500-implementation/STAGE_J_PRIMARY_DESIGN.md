# Stage J Primary Design v003: GridSFM GOC-500 Implementation

```yaml
artifact_id: CASE-003-STAGE-J-PRIMARY-DESIGN
artifact_version: v003
created_utc: 2026-08-12T00:00:00+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: APPROVED_TO_BEGIN_GATED_IMPLEMENTATION
sha256_when_frozen: null
```

## 0. Final Locked Implementation Corrections

This v003 design supersedes any earlier v001/v002 statement that deferred full per-load alpha, removed physics penalties from GridSFM candidate selection, or treated Reference A as a generic AC feasibility check.

Locked Stage J decisions:

```text
z_l in {0,1}
alpha_i in [0,1] for every load i in D
```

The wildfire-side search supplies `(z, alpha)` externally to each electrical backend. Guided-DC and Guided-GridSFM must therefore receive the same candidate topology, alpha vector, candidate set, topology budget, proxy scores, no-good-cut logic, and lambda sweep. If full per-load alpha search is not tractable, the implementation must report `APPROXIMATION REQUIRED` and pause for Caleb approval before replacing it with global, zonal, grouped, or selected-load alpha.

For reporting, the weighted wildfire/service tradeoff is:

```text
J_trade^m =
lambda_R R_norm^m
+ (1 - lambda_R) L_shed_total
```

For GridSFM candidate selection, the actual merit function is:

```text
J_total^SFM =
J_trade^SFM
+ rho_phys PAC_total^SFM
```

```text
PAC_total^SFM =
w_op PAC_operational
+ w_AC PAC_AC
+ w_model PAC_model
```

`J_trade` is the Pareto/scientific wildfire-service quantity. `J_total` is the physics-aware GridSFM search merit. `D_input`, `D_state_to_AC`, and warm-start metrics are separate diagnostics and are not optimization penalties.

Stage J uses separate electrical and wildfire scenarios:

```text
e in E = electrical scenario
w in W = wildfire scenario

Pd_pre[i,e], Qd_pre[i,e]
baseline_loading[l,e]
c_l[e]
p_env[l,w]
```

For the initial S1-S3 study, a shared electrical condition `e=e0` may be paired with different wildfire vectors `w`.

## 1. Purpose And Central Framing

Stage J is a minimal-change extension of the existing wildfire topology-search methodology to GridSFM-Open on GOC-500.

The primary controlled comparison is:

```text
guided outer topology search + DC recourse
versus
guided outer topology search + GridSFM recourse
```

Secondary/reference comparisons are:

```text
transparent wildfire-risk heuristic topology policies
direct DC MIQP after the guided comparison is stable
exact AC finalist audits / restoration
warm-start computational comparison
```

GridSFM remains frozen for the initial study. The correct framing is:

```text
frozen / zero-adaptation wildfire transfer on GOC-500
```

Do not call this unseen-grid zero-shot unless the GridSFM training-data relationship is later established and supports that claim.

The main scientific question is:

```text
Does replacing the DC / older learned physical evaluator with a modern AC-OPF
foundation-model surrogate change the quality of the selected wildfire
topology, the predicted electrical state, the required exact AC recourse, and
computational performance?
```

Stage J does not claim that GridSFM solves optimal power shutoff end to end. The surrounding wildfire optimization/search logic still chooses topology and load-service commands.

## 2. Relationship To Existing Stages

Stage J inherits these lessons from Stages E-I:

- Keep topology search and continuous recourse/evaluation distinct.
- Keep wildfire/load objective values distinct from diagnostics.
- Keep surrogate prediction quality distinct from final topology quality.
- Use canonical physical branch IDs for all topology, risk, flow, and rating quantities.
- Use source-less island service handling as a hard methodological invariant.
- Use common exact AC references to audit final approximate decisions.
- Do not interpret a learned model's raw prediction as final physical truth.

The main change from the old GridFM-PF formulation is generator redispatch.

Previous GridFM-style formulation approximately controlled:

```text
z
alpha
Delta Pg
Delta Qg
```

Stage J controls only:

```text
z_l       binary topology / line status
alpha_i   load-service command
```

GridSFM supplies AC-OPF recourse outputs:

```text
Pg
Qg
V
theta
Pij
Qij
Pji
Qji
feasibility score/logit
```

Conceptual pipeline:

```text
(z, alpha)
    ->
GridSFM AC-OPF surrogate
    ->
(Pg, Qg, V, theta, branch flows)
    ->
wildfire objective + diagnostics
```

Do not treat `Pg` or `Qg` as externally commanded GridSFM controls. GridSFM does not choose `z`, does not choose `alpha`, and does not optimize the wildfire objective internally.

## 3. Sets, Indices, And Core Parameters

Use canonical physical branch IDs throughout.

```text
N              set of buses
L_phys         set of canonical physical branches
L_ac           set of AC transmission line branches
L_tr           set of transformer branches
L_risk         wildfire-risk branches, locked initially to L_ac
L_eligible     branches eligible for topology control
C              screened candidate set, C subset of L_eligible
D              load buses or load records with positive requested demand
G              generator records
E              electrical scenario set
W              wildfire scenario set
K              topology shutoff budget, locked initially to K <= 2
Lambda         objective tradeoff weights
```

Branch parameters:

```text
from_l          canonical from bus for branch l
to_l            canonical to bus for branch l
orientation_l   orientation sign used for signed comparisons
rateA_l         thermal rating
Smax_l          apparent-power limit, initially rateA_l
p_env[l,w]      wildfire exposure score/probability for branch l in wildfire scenario w
x_l             reactance
r_l             resistance
tap_l           transformer tap ratio
shift_l         transformer phase shift
```

Scenario demand parameters:

```text
Pd_pre[i,e]     requested active demand at load i before wildfire-control actions
Qd_pre[i,e]     requested reactive demand at load i before wildfire-control actions
```

`Pd_pre` and `Qd_pre` replace the old conceptual use of `Pd_base` / `Qd_base`. The old field names may remain in code for compatibility, but the Stage J methodology must interpret them as scenario-specific pre-intervention requested demand, not as one permanent nominal MATPOWER load.

For the first S1-S3 diagnostic suite, these values may be numerically identical across scenarios if only `p_env` changes. The notation still preserves the ability to later perturb the electrical operating scenario.

Generator and voltage parameters:

```text
Pg_min/max      active generator bounds
Qg_min/max      reactive generator bounds
V_min/max       voltage magnitude bounds
gen_cost        generator-cost data, if economic AC-OPF references are run
epsilon         numerical floor, default 1e-6
theta_max       DC angle bound, default pi unless changed by audit
```

## 4. Load-Service Accounting

Stage J should not recreate the previous GridFM hybrid-load rule.

The old GridFM model predicted load channels, which forced us to compare commanded load against model-implied load and construct hybrid load accounting. GridSFM instead receives demand as an input and predicts the electrical OPF response.

Define effective served fraction:

```text
alpha_eff[i,z] =
0          if load i belongs to a source-less component under topology z
alpha_i    otherwise
```

Commanded served demand:

```text
Pd_cmd[i,e,z,alpha] = alpha_eff[i,z] Pd_pre[i,e]
Qd_cmd[i,e,z,alpha] = alpha_eff[i,z] Qd_pre[i,e]
```

Stage J v1 locks proportional active/reactive shedding:

```text
Qd_cmd = alpha_eff Qd_pre
```

unless a later sensitivity explicitly changes this.

Native load shedding is model-independent:

```text
L_shed_total[e](z,alpha) =
L_shed_control[e](z,alpha)
+ L_shed_island[e](z)
```

```text
L_shed_island[e](z) =
sum_{i in ISL(z)} Pd_pre[i,e]
/
sum_{i in D} Pd_pre[i,e]
```

```text
L_shed_control[e](z,alpha) =
sum_{i notin ISL(z)} Pd_pre[i,e](1 - alpha_i)
/
sum_{i in D} Pd_pre[i,e]
```

Use exactly the same requested-demand accounting for DC and GridSFM. Do not infer native load shedding from GridSFM output. Save both `alpha_requested` and `alpha_effective`; the former is the wildfire-side command and the latter includes topology-forced source-less island shedding. The objective uses `L_shed_total`.

## 5. Source-Less Island Handling

For every candidate topology `z`, compute connected components before electrical evaluation.

A component is source-less only if it contains no available generator or valid supply source. Do not define source-less as simply disconnected from the original slack bus; an island may contain its own generation.

For every load in a source-less component:

```text
alpha_eff[i,z] = 0
Pd_cmd[i,s,z,alpha] = 0
Qd_cmd[i,s,z,alpha] = 0
```

This applies to all load buses in the component, not merely selected alpha-controlled loads.

This must be implemented twice:

```text
1. hard preprocessing/enforcement before DC or GridSFM evaluation
2. independent post-evaluation assertion/check
```

Required validation condition:

```text
source_less_served_load =
sum_{i in source-less} Pd_pre[i,s] alpha_eff[i,z]
= 0
```

within tolerance. A nonzero value flags the row as an implementation/methodology failure, not as a soft penalty. A source-less/islanding severity diagnostic may still be reported separately.

## 6. Exact Intact AC Baseline And Normalization

For each distinct electrical operating scenario `s`, solve the intact all-energized exact AC-OPF or trusted exact AC baseline once.

Use this exact intact baseline to obtain:

```text
baseline_loading[l,s]
```

for every branch. This does not require one AC solve per line.

Scenario-specific wildfire denominator:

```text
R_base[e,w] =
sum_{l in L_risk} p_env[l,w] baseline_loading[l,e]^2
```

Normalized risk:

```text
R_norm = R_raw / R_base[e,w]
```

The baseline is shared across methods. Do not use GridSFM or DC native predictions to define the common baseline.

If S1-S3 share the same electrical operating point, they can reuse the same intact AC state. Their `R_base[e,w]` values may still differ because their `p_env[:,w]` vectors differ.

Required baseline audit:

```text
baseline_state_id
electrical_scenario_id
baseline_backend
max baseline_loading
number of branches with baseline_loading > 1
R_base[e,w]
baseline AC feasibility status
```

The intact baseline must be a successful economic AC-OPF:

```text
x_base^AC(e) =
argmin C_gen(Pg)
```

subject to the intact exact AC network and pre-intervention demand. If the intact exact AC-OPF is infeasible, the electrical scenario is not admitted to the primary experiment.

## 7. Wildfire Risk Metric

Stage J v1 locks:

```text
L_risk = AC transmission lines
```

Transformers must still be represented correctly in the electrical model, but they do not initially carry `p_env` or enter the wildfire objective unless a scenario explicitly motivates transformer wildfire exposure.

For AC-capable outputs:

```text
|S_ij| = sqrt(P_ij^2 + Q_ij^2)
|S_ji| = sqrt(P_ji^2 + Q_ji^2)
loading_l = max(|S_ij|, |S_ji|) / rateA_l
```

Every branch in `L_risk` must have finite positive `rateA_l`, and unit compatibility among `baseMVA`, GridSFM outputs, MATPOWER/PowerModels ratings, and conversion functions must be verified before wildfire metrics are computed.

For GridSFM:

```text
R_raw^SFM(z,alpha,s) =
sum_{l in L_risk}
z_l p_env[l,s]
(
    max(|S_ij^SFM|, |S_ji^SFM|) / rateA_l
)^2
```

```text
R_norm^SFM = R_raw^SFM / R_base[e,w]
```

For DC:

```text
R_raw^DC(z,alpha,s) =
sum_{l in L_risk}
z_l p_env[l,s]
(|f_l^DC| / rateA_l)^2
```

```text
R_norm^DC = R_raw^DC / R_base[e,w]
```

DC risk is an active-power approximation. It is comparable as a transparent optimization benchmark, but not identical to AC apparent-power risk.

For exact AC references:

```text
R_raw^AC(z,alpha,s) =
sum_{l in L_risk}
z_l p_env[l,s]
(
    max(|S_ij^AC|, |S_ji^AC|) / rateA_l
)^2
```

```text
R_norm^AC = R_raw^AC / R_base[e,w]
```

## 8. Tradeoff And Candidate-Selection Objectives

The Stage J wildfire/load tradeoff objective is:

```text
J_trade^m(z,alpha,e,w) =
lambda_R R_norm^m(z,alpha,e,w)
+ (1 - lambda_R) L_shed_total(z,alpha,e)
```

where `m` is the recourse backend:

```text
m in {Guided-DC, Guided-GridSFM, heuristic-evaluated-backend, optional DC-MIQP}
```

Both terms are minimized:

```text
removing a hazardous line can reduce R_norm
but can increase L_shed
```

This is the intended wildfire-resilience tradeoff.

For GridSFM candidate selection, use:

```text
J_total^SFM =
J_trade^SFM
+ rho_phys PAC_total^SFM
```

For DC, represented physics are hard constraints in the fixed-topology DC-OPF recourse, so candidate comparison uses `J_trade^DC`.

If old data schemas require the field name `J_true`, it may remain as a legacy compatibility field. Documentation must clarify:

```text
J_true legacy field = J_trade wildfire/load objective
not physical ground truth
```

Preferred new field:

```text
J_trade
```

Avoid adding new `J_native` fields in Stage J outputs except where a legacy reader strictly requires them.

## 9. Outer Wildfire Proxy

The Gurobi outer master continues to use a cheap proxy rather than optimizing through GridSFM.

Topology variables:

```text
y_l = 1 means shut line l off
z_l = 1 - y_l
```

Baseline wildfire score:

```text
w_l[s] =
p_env[l,s] baseline_loading[l,s]^2
```

Remaining-risk proxy:

```text
R_proxy(y,s) =
sum_{l in C} w_l[s] (1 - y_l)
/
sum_{l in C} w_l[s]
```

Single-line service-consequence proxy:

```text
c_l =
max(0, D_served_pre - D_served_after_single_outage(l))
/
D_requested_pre
```

where:

```text
D_requested_pre = sum_{i in D} Pd_pre[i,s]
D_served_pre    = served demand in the intact pre-control state
D_served_after_single_outage(l)
                 = estimated served demand after removing only line l
```

The load-consequence proxy is:

```text
L_proxy(y) =
sum_{l in C} c_l y_l
```

The master objective is:

```text
min_y
lambda_R^M R_proxy(y,s)
+ lambda_L^M L_proxy(y)
```

with:

```text
lambda_L^M = 1 - lambda_R^M
```

A high `c_l` is a reason not to remove the line. No inversion of `c_l` is needed because it enters as a positive penalty multiplied by the shutoff variable.

The role of `c_l` is topology-search guidance only. It is not:

```text
a wildfire-harm multiplier
part of R_raw / R_norm
part of the simple TH heuristic
a physical ground-truth quantity
```

Before changing the computational backend for `c_l`, inspect how the current repository computes the existing proxy. If reusable on GOC-500, preserve it first for methodological continuity.

If a new backend is required, the same `c_l` values must be supplied to both Guided-DC and Guided-GridSFM. Do not use GridSFM-derived `c_l` for GridSFM and DC-derived `c_l` for DC in the primary comparison. That would change the topology proposal mechanism as well as the inner recourse model.

If DC-MLD is used to precompute `c_l`, explicitly label the outer master as using:

```text
shared DC-informed service-impact proxy
```

and use that identical proxy for both guided methods.

Recommended later sensitivity, after the primary comparison works:

```text
c_l = existing service-impact proxy
c_l = connectivity/source-less-only proxy
c_l = 0
```

GridSFM-based fast single-outage impact estimation is a future engineered-screening study, not part of the primary DC-vs-GridSFM controlled experiment.

## 10. Candidate Set And Topology Budget

The conceptual wildfire shutoff problem can allow binary line decisions over all eligible lines.

Stage J v1 uses:

```text
C subset of L_eligible
sum_{l in C} y_l <= K
K <= 2
```

as computational/search restrictions, not as intrinsic optimal power shutoff physics and not as assumptions inherited from the Rhodes OPS formulation.

Likewise, `C` is a screening/scalability mechanism. Future studies may increase:

```text
|C|
K
```

or consider a more unrestricted Rhodes-like topology decision problem.

If a future search uses:

```text
stop when objective improvement < epsilon
```

that is only a stopping rule. It must not be described as global convergence, because beneficial multi-line interactions can exist even when no locally examined addition improves the objective.

No-good/no-revisit cuts remain required in the guided topology loop:

```text
sum_{l in C}
[
  (1 - y_l) if y_l^prev = 1
  y_l       if y_l^prev = 0
]
>= 1
```

for every previously evaluated topology vector.

## 11. Guided DC Versus Guided GridSFM

The primary fair comparison uses the same:

```text
candidate set C
K
outer proxy formula
c_l values
proxy lambda settings
no-good/revisit logic
topology evaluation budget
wildfire scenarios
exact intact AC baseline
load-service accounting
```

for:

```text
Guided-DC
Guided-GridSFM
```

The primary controlled experimental change is:

```text
inner electrical recourse/evaluator
```

Guided-DC branch:

```text
(z, alpha)
    ->
hard DC recourse
    ->
Pg, theta, active line flow
    ->
R_norm^DC, L_shed_total, J_trade^DC
```

Guided-GridSFM branch:

```text
(z, alpha)
    ->
frozen GridSFM
    ->
Pg, Qg, V, theta, P/Q branch flows
    ->
R_norm^SFM, L_shed_total, J_trade^SFM, PAC_total^SFM, J_total^SFM
```

The direct DC MIQP can remain a stronger secondary benchmark after the guided comparison is stable. Do not use direct DC MIQP as the only DC comparator because it changes both the recourse physics and the topology-search algorithm.

## 12. GridSFM Formulation

GridSFM is evaluated as a frozen surrogate map:

```text
F_SFM:
(GOC500_graph, z, Pd_cmd, Qd_cmd, scenario metadata)
->
x_SFM
```

where:

```text
x_SFM =
{
Pg^SFM,
Qg^SFM,
V^SFM,
theta^SFM,
Pij^SFM,
Qij^SFM,
Pji^SFM,
Qji^SFM,
feas^SFM
}
```

GridSFM evaluated objective:

```text
J_trade^SFM(z,alpha,e,w) =
lambda_R R_norm^SFM(z,alpha,e,w)
+ (1 - lambda_R) L_shed_total(z,alpha,e)
```

GridSFM candidate-selection objective:

```text
J_total^SFM(z,alpha,e,w) =
J_trade^SFM(z,alpha,e,w)
+ rho_phys PAC_total^SFM(z,alpha,e,w)
```

GridSFM is not algebraically embedded in Gurobi during Stage J v1. It is a black-box or batched evaluator queried by the external topology/alpha search.

The GridSFM search problem is:

```text
min over z, alpha
J_total^SFM(z,alpha,e,w)
```

subject to:

```text
z_l in {0,1}
y_l = 1 - z_l
sum_{l in C} y_l <= K
alpha_i in [0,1] for every load i in D
alpha_eff source-less rules
```

The implementation may use coordinate search, bounded derivative-free search, batched inference, or another full-vector search method. If it cannot tractably optimize the full per-load vector, it must report `APPROXIMATION REQUIRED` and pause for approval before switching to a reduced alpha parameterization. It must not claim differentiable or end-to-end optimization unless that is implemented and validated separately.

## 13. DC Formulation

The Guided-DC benchmark should use the same externally supplied topology and load-service semantics.

Variables:

```text
theta_i
f_l
Pg_g
```

For every externally supplied `(z, alpha)`, solve fixed-topology economic DC-OPF recourse:

```text
x_DC^*(z,alpha) =
argmin_{Pg,theta,f} C_gen(Pg)
```

subject to fixed commanded active demand:

```text
Pd_cmd[i,e,z,alpha] = alpha_eff[i,z] Pd_pre[i,e]
```

No binary topology variables appear in the Guided-DC inner recourse. `z` and `alpha` are fixed by the outer wildfire-side search. If the fixed-topology DC-OPF is infeasible, log `evaluation_status = dc_infeasible`, assign rejection merit, and do not allow the candidate to become the best candidate.

Fixed topology:

```text
z_l fixed to supplied topology
f_l = 0 for offline branches
```

Nodal balance:

```text
sum_{g in G_i} Pg_g
- alpha_eff[i,z] Pd_pre[i,e]
- sum_{l in delta+(i)} f_l
+ sum_{l in delta-(i)} f_l
= 0
```

Generator limits:

```text
Pg_g^min <= Pg_g <= Pg_g^max
```

DC branch equation for active branches:

```text
f_l = B_l(theta_i - theta_j - phi_l), if z_l = 1
```

Offline branch:

```text
f_l = 0, if z_l = 0
```

Thermal constraints:

```text
-rateA_l z_l <= f_l <= rateA_l z_l
```

Angle bounds:

```text
-theta_max <= theta_i <= theta_max
```

DC candidate comparison:

```text
J_trade^DC =
lambda_R R_norm^DC
+ (1 - lambda_R) L_shed_total
```

DC residuals after a successful solve should be near zero for represented DC equations. Reactive power, voltage magnitude, and AC branch-loss feasibility remain outside native DC physics.

## 14. Heuristic Reference

The main transmission heuristic remains intentionally simple:

```text
score_TH[l,s] =
p_env[l,s] baseline_loading[l,s]^2
```

Rank by this score and select the top-k lines under the tested budget.

Do not multiply the simple TH score by `c_l`. The heuristic is intended to represent visible wildfire-risk ranking using baseline conditions, not the complete topology optimizer.

If a later engineered heuristic incorporates impact, label it separately:

```text
impact-aware heuristic
```

and define the direction carefully.

The primary Stage J transparent heuristic is `TH-GridSFM`:

```text
score_TH = p_env * baseline_loading^2
    ->
select exact TH-1 and TH-2 topologies separately
    ->
run the same optimized-alpha continuous procedure on each fixed TH topology
    ->
GridSFM evaluation
    ->
exact AC finalist audits
```

In implementation this is treated as its own method family, not as a plotted
overlay of the guided pool. `TH-GridSFM-top1` fixes the single highest-scoring
line as the shutoff topology, and `TH-GridSFM-top2` fixes the two highest-scoring
lines as the shutoff topology. Each fixed topology then receives the same
budgeted selected-load alpha optimizer used by `Guided-GridSFM`; only the
topology proposal mechanism differs.

Optional secondary heuristic rows may evaluate the same TH topology through DC, but must be labeled separately as `TH-DC`. Do not store one ambiguous `TH` row if the electrical backend differs.

AH may remain as a prior-work secondary heuristic if convenient, but it should not drive the first Stage J implementation.

## 15. Alpha Search

Stage J locks one alpha decision per load:

```text
alpha_i in [0,1] for every load i in D
```

The implementation should smoke-test simple full-vector candidates before full search:

```text
alpha_i = 1 for all loads
uniform full-vector reduction, stored as a vector
targeted full-vector perturbations, stored as vectors
```

If a reduced parameterization appears necessary for runtime, classify it as `APPROXIMATION REQUIRED` and wait for explicit approval. Regardless of search implementation, all load-service accounting uses `Pd_pre` / `Qd_pre`, `alpha_requested`, and `alpha_effective`.

## 16. Initial GOC-500 Scenario Suite

The first GridSFM GOC-500 implementation constructs only:

```text
J-S1: high wildfire risk / low service impact
J-S2: high wildfire risk / high service impact
J-S3: high wildfire risk / electrically redundant alternate path
```

These are diagnostic scenarios, not statistical ground truth.

Each scenario must record:

```text
scenario_id
electrical_scenario_id
target / diagnostic branch IDs
candidate branch set C
p_env construction rule
exact intact baseline state reference
baseline_loading
c_l / impact information
expected qualitative behavior
connectivity/islanding metadata
```

Expected target branches are qualitative sanity checks. Do not treat target recall/precision as proof of optimality.

The purpose of S1-S3 is:

```text
does the new GOC-500 implementation make understandable decisions?
```

Defer J-S4/J-S5 until the GridSFM adapter and exact AC audit are stable.

## 17. Exact AC Reference Roles

After a method returns:

```text
z_m^*
alpha_m^*
x_m^native
```

run separate exact audits. Do not store all exact solves under one ambiguous "AC restoration" label.

### 17.1 Reference A: Fixed-z, Fixed-alpha Economic AC-OPF

Fix:

```text
z = z_m^*
alpha = alpha_m^*
```

Solve:

```text
minimize C_gen(Pg)
```

subject to exact AC physics, fixed topology `z = z_m^*`, fixed commanded demand from `alpha = alpha_m^*`, voltage bounds, generator limits, and apparent-power line limits.

This answers:

```text
Can the topology + requested served-load decision produced by the approximate
method actually be realized under AC physics?
```

For GridSFM, this also supplies the exact state against which the model state can be compared.

Use Reference A for `D_state_to_AC` and as the primary warm-start benchmark. If fixed-z/fixed-alpha economic AC-OPF is infeasible, record `evaluation_status = ac_reference_infeasible`, set `D_state_to_AC = N/A`, and do not invent a projection distance. Do not silently mix economic AC-OPF and minimum-load-shed restoration under one field name.

### 17.2 Reference B: Rhodes-Style Fixed-Topology AC Redispatch

Fix only:

```text
z = z_m^*
```

Release continuous operating decisions, including load-service fractions.

Solve:

```text
minimize
sum_{i in D} Pd_pre[i,s] (1 - alpha_i)
```

subject to:

```text
exact AC active balance
exact AC reactive balance
voltage bounds
branch thermal bounds
generator limits
0 <= alpha_i <= 1
source-less service rules
fixed selected topology z_m^*
```

This asks:

```text
Given the topology selected by the wildfire method, what is the maximum demand
that can actually be served under exact AC physics?
```

This is the primary literature-aligned AC redispatch / MLD-style final topology audit.

After the solve, compute rather than optimize away:

```text
R_norm^AC
L_shed^AC
J_native^AC_reported = lambda_R R_norm^AC + (1 - lambda_R) L_shed^AC
```

The outer wildfire method has already chosen the topology. Reference B tests the service capability and realized wildfire loading of that topology.

### 17.3 Reference C: Warm-Start Reference

For exact AC problem variants, compare:

```text
cold start
DC-informed start
GridSFM-informed start
exact/GT warm-start ceiling when available
```

Record:

```text
solver-only runtime
preprocessing time
inference time
end-to-end runtime
iterations
success/failure
final exact objective
```

Warm-start results remain separate from topology-quality claims.

## 18. Generator Flexibility In Exact AC Redispatch

Before treating Reference B as a ground-truth restoration layer, inspect GOC-500 generator limits and the PowerModels formulation.

Load shedding alone does not guarantee every disconnected topology is feasible. For example, an island may have minimum generation exceeding local load.

The exact restoration must define whether generation can:

```text
redispatch down to existing Pmin
curtail to zero
be deactivated
use another documented restoration convention
```

Do not silently modify generator constraints to make every topology feasible. If a topology remains infeasible under the locked exact restoration model, that is a legitimate result.

Implementation audit must also verify how the AC backend represents multiple disconnected components and angle references.

## 19. Diagnostic Families

Stage J retains the conceptual decomposition from earlier work:

```text
1. Operational diagnostics
2. AC-physics consistency
3. Model / interface consistency
```

For GridSFM candidate selection, normalized and validated members of these diagnostics enter `PAC_total^SFM` through `rho_phys PAC_total^SFM`. Store all subterms separately and preserve `J_trade` for Pareto interpretation. For DC, represented physics are hard constraints and AC-only diagnostics remain post-hoc or `N/A`.

### 19.1 Operational Diagnostics

For GridSFM include, where defined:

```text
voltage violations
thermal violations
Pg bound violations
Qg bound violations
offline-line flow
source-less service
load-service bounds
```

For DC, report only native quantities that DC represents. AC-only quantities must be `N/A` for raw DC output, not zero.

Recommended operational score:

```text
Phi_operational =
Phi_thermal
+ Phi_gen_P
+ Phi_gen_Q
+ Phi_voltage
+ Phi_load_service
+ Phi_source_less_service
+ Phi_offline_flow
```

but store subterms separately.

### 19.2 AC-Physics Consistency

For GridSFM compute raw AC active/reactive nodal residuals:

```text
rP_i =
sum_{g in G_i} Pg_g
- Pd_cmd[i,s]
- P_AC_i(V,theta,z)
```

```text
rQ_i =
sum_{g in G_i} Qg_g
- Qd_cmd[i,s]
- Q_AC_i(V,theta,z)
```

Report normalized mean, RMS, and maximum residuals.

Also audit branch-flow consistency where useful because GridSFM provides branch-flow outputs.

For DC:

```text
native DC equation residuals
```

should be near zero after a successful solve. Reactive/voltage AC consistency is not a native DC metric.

### 19.3 Model / Interface Consistency

Preserve this family because a major lesson from old GridFM work was the need to confirm that the model pipeline actually respected supplied values.

Split it into:

```text
D_input
PAC_model
D_state_to_AC
```

Input invariants define `D_input`:

```text
z_requested == z_GridSFM_graph
Pd_GridSFM_input == alpha_eff * Pd_pre
Qd_GridSFM_input == alpha_eff * Qd_pre
generator limits preserved
generator costs preserved
voltage limits preserved
rateA preserved
branch identities preserved
transformer metadata preserved
```

These should be zero-error invariants. An input-integrity failure is a hard methodology failure, not a soft objective penalty. Source-less island handling remains hard-clamped before inference:

```text
alpha_eff_i = 0        if i is source-less under z
alpha_eff_i = alpha_i  otherwise
```

Save both `alpha_requested` and `alpha_effective`; differences caused by source-less islands are topology-forced service loss, not a model error.

Retain the Stage I-style model-consistency guardrail where comparable output channels exist:

```text
PAC_model_load =
mean(
  normalized_mse(Pd_pred, alpha_eff Pd_pre),
  normalized_mse(Qd_pred, alpha_eff Qd_pre)
)
```

For the current official GridSFM `predict()` API, `Pd_pred` and `Qd_pred` are not returned; demand-output command consistency is therefore marked unavailable and contributes zero to `PAC_model` unless future outputs expose those channels. The hard `D_input` check still verifies that GridSFM received the clamped commanded demand.

`PAC_model` is reserved for native model self-consistency that can be computed from independent GridSFM output channels, such as direct branch-flow outputs versus flows reconstructed from predicted voltage/angle. Before using this term, inspect the actual GridSFM network heads. If flows are deterministically reconstructed internally rather than independently predicted, mark `PAC_model` as unavailable or zero-weighted with an audit note.

For fixed `z` and `alpha`, compare the native predicted state to Reference A as `D_state_to_AC`:

GridSFM `D_state_to_AC` can include:

```text
D_Pg
D_Qg
D_V
D_flow_P
D_flow_Q
optional aligned D_theta
```

DC `D_state_to_AC` can include:

```text
D_Pg
D_flow_P
optional aligned D_theta
```

and must mark:

```text
D_Qg
D_V
D_Qflow
```

as `N/A`. Do not penalize DC for quantities it never models. Do not compare GridSFM `Pg` against an externally commanded `Pg`, because `Pg` is now an OPF recourse output.

## 20. Recourse-Change Metrics

Keep surrogate/state fidelity separate from the amount of correction required after selecting a topology.

For each method `m`, record native:

```text
alpha_m^*
L_shed_native
Pg_native
Qg_native, V_native, flows_native where available
```

and Rhodes-style exact restored:

```text
alpha_AC
L_shed_AC
Pg_AC
Qg_AC
V_AC
flows_AC
```

Minimum recourse-change metrics:

```text
Delta_L_shed_m =
L_shed_AC - L_shed_native
```

```text
Delta_alpha_m =
sum_{i in D} Pd_pre[i,s] |alpha_AC[i] - alpha_native[i]|
/
sum_{i in D} Pd_pre[i,s]
```

Also record where meaningful:

```text
Delta_Pg
Delta_Qg
Delta_V
Delta_flow
```

Interpretation:

```text
How much did exact AC operation have to change after fixing the topology
selected by the approximate method?
```

This is distinct from `PAC_model` and from `D_state_to_AC`.

## 21. Final Metric Families

Stage J's main analysis must retain these separate metric families:

```text
Native wildfire/service tradeoff:
  R_norm
  L_shed_total
  L_shed_control
  L_shed_island
  J_trade

GridSFM search merit:
  PAC_operational
  PAC_AC
  PAC_model
  PAC_total
  J_total

Operational validity:
  Phi_operational

AC equation consistency:
  Phi_AC / raw residuals

Model/interface validation:
  D_input
  D_state_to_AC

Exact recourse correction:
  Delta_recourse
  Delta_L_shed
  Delta_alpha

Exact topology outcome:
  R_norm_AC
  L_shed_AC

Computation:
  topology search time
  alpha-search time
  GridSFM preprocessing
  GridSFM inference
  DC solve time
  AC solve time
  total wall time
  iterations
  success rate
  warm-start headroom
```

Do not collapse these into one unqualified "infeasibility" score.

## 22. Warm-Start Headroom

Where exact/GT warm-start reference is available:

```text
eta_m =
(T_cold - T_m)
/
(T_cold - T_GT)
```

Retain negative values if a warm start is slower than cold. Do not clip to `[0,1]`.

Flag cases where:

```text
T_cold - T_GT
```

is very small because normalized headroom becomes unstable.

Always report raw seconds and iteration counts alongside `eta_m`.

Distinguish:

```text
solver-only runtime
```

from:

```text
end-to-end runtime =
preprocessing
+ model/DC evaluation
+ exact AC solve
```

## 23. Branch Identity Gate

Stage J must preserve a canonical branch table:

```text
canonical_physical_branch_id
original_case_branch_id
GridSFM_edge_family
GridSFM_edge_index
from_bus
to_bus
orientation_sign
rateA
p_env
candidate_flag
wildfire_risk_flag
```

A topology mutation must never silently shift the meaning of an existing `p_env` or `rateA` entry.

Before any wildfire result is accepted, test:

```text
branch identity
edge orientation
transformer/AC-line family
line removal/deactivation
Pij/Pji mapping
Qij/Qji mapping
risk attachment
```

Use immutable canonical physical IDs even if GridSFM local edge arrays change after topology mutation.

## 24. Environment And Path Design

The current work is performed inside the cloned GridFM GraphKit repository.

Retain:

```text
GRIDFM_GRAPHKIT_ROOT
```

as the main research repository root:

```text
GRIDFM_GRAPHKIT_ROOT =
existing Stage A-I / Stage J research code
```

Add:

```text
GRIDSFM_ROOT
```

as the external Microsoft GridSFM checkout:

```text
GRIDSFM_ROOT =
Microsoft GridSFM repository checkout
```

Do not conflate the two.

Every runner should accept:

```text
--repo-root
--gridsfm-root
--model-cache-dir
--work-dir
--results-dir
--config
```

Default path resolution:

```text
1. explicit CLI argument
2. environment variable GRIDFM_GRAPHKIT_ROOT
3. git rev-parse --show-toplevel
4. parent search for known repository markers
```

Recommended environment split:

```text
existing_repo_env:
  Stage A-I code, Stage J orchestration, plotting, workflow, result aggregation

gridsfm_env:
  Hugging Face model, PyTorch, PyG, GridSFM inference, OPFData access

ac_solver_env:
  Julia, PowerModels, JuMP, IPOPT, or Docker equivalent
```

A subprocess/file-based bridge is acceptable for the first implementation if direct dependency integration is fragile:

```text
repo runner writes request JSON
GridSFM env runner reads request JSON
GridSFM env runner writes response JSON/Parquet
repo runner reads response and computes objectives/plots
```

Large GridSFM checkpoints, OPFData caches, and Julia solver artifacts should live outside git and be hash-indexed in environment/model manifests.

## 25. Data Exchange Schemas

### 25.1 Evaluation Request

```text
request_id
electrical_scenario_id
wildfire_scenario_id
method_family
lambda_R
lambda_R_proxy
topology_vector_z
topology_hash
alpha_vector
alpha_hash
candidate_branch_ids
p_env_hash
baseline_state_id
Pd_pre_reference
Qd_pre_reference
branch_identity_table_reference
impact_proxy_backend
impact_proxy_version
case_data_reference
model_backend
cache_policy
```

### 25.2 Recourse Response

```text
request_id
evaluation_status
backend
model_name
model_revision
runtime_preprocess_seconds
runtime_inference_seconds
runtime_total_seconds
Pg
Qg
V
theta
Pij
Qij
Pji
Qji
flow_edge_types
flow_edge_counts
feasibility_score
feasibility_label
D_input
warnings
```

Allowed `evaluation_status` values:

```text
ok
dc_infeasible
gridsfm_preprocessing_failure
gridsfm_inference_failure
model_output_penalized
input_integrity_failure
ac_reference_infeasible
methodology_failure
```

### 25.3 Native Objective Row

```text
electrical_scenario_id
wildfire_scenario_id
method_family
lambda_R
lambda_R_proxy
topology_id
topology_hash
alpha_id
alpha_hash
num_shutoffs
selected_branch_ids
R_raw
R_norm
L_shed_total
L_shed_control
L_shed_island
alpha_requested
alpha_effective
J_trade
PAC_operational
PAC_AC
PAC_model
PAC_total
J_total
J_wildfire_native
J_true_legacy
D_input
D_state_to_AC
runtime_seconds
evaluation_status
```

### 25.4 Exact AC Reference Row

```text
electrical_scenario_id
wildfire_scenario_id
method_family
lambda_R
topology_id
topology_hash
alpha_id
alpha_hash
ac_reference_type
evaluation_status
AC_objective
AC_R_norm
AC_L_shed
AC_Phi_operational
AC_runtime_solver_seconds
AC_runtime_end_to_end_seconds
AC_iterations
D_state_to_AC
Delta_alpha
Delta_L_shed
Delta_Pg
Delta_Qg
Delta_V
Delta_flow
restoration_gap
warnings
```

Allowed `ac_reference_type` values:

```text
fixed_z_alpha_economic_ac_opf_reference
fixed_z_rhodes_ac_redispatch
cold_start
dc_warm_start
gridsfm_warm_start
gt_warm_start
```

## 26. Required Plots

Native Pareto plot:

```text
x = L_shed_native
y = R_norm_native
families = Guided GridSFM, Guided DC, heuristic overlays, optional DC MIQP
```

Exact restored topology outcome:

```text
x = L_shed_AC_redispatch
y = R_norm_AC_redispatch
families = selected method topologies after fixed-z exact AC redispatch
```

Validation plots:

```text
D_state_to_AC vs lambda_R
AC residuals vs lambda_R
Delta_L_shed vs lambda_R
Delta_alpha vs lambda_R
Phi_operational vs lambda_R
runtime breakdown
warm-start runtime / iterations
headroom captured
```

Rename the current decision-quality target plot to:

```text
Diagnostic Target Agreement
```

because target recall, precision, and expected-vs-selected lines only indicate agreement with deliberately engineered S1-S3 expectations. The actual operational comparison should focus on native and exact-AC-restored risk/load outcomes.

Use the red/yellow/green convention from earlier stages where target-set diagnostic plots are produced.

## 27. Result Tables

Minimum required tables:

```text
stage_j_native_objective_rows.csv
stage_j_exact_ac_reference_rows.csv
stage_j_model_state_distance_rows.csv
stage_j_ac_residual_rows.csv
stage_j_recourse_change_rows.csv
stage_j_runtime_rows.csv
stage_j_warm_start_rows.csv
stage_j_branch_identity_table.csv
stage_j_scenario_register.csv
stage_j_diagnostic_target_agreement_rows.csv
stage_j_environment_manifest.json
stage_j_model_manifest.json
```

## 28. Implementation Phases

### J0: Revised Design Review

Produce:

```text
Stage J primary design v003
change log from v001/v002
scientific review
implementation audit
reconciliation
approval gate
```

### J1: GridSFM Environment Smoke

Goals:

```text
load released GridSFM checkpoint
reproduce official GOC-500 inference
record dependency and model versions
```

Pass criteria:

```text
model loads
GOC-500 example runs
outputs include V, theta, Pg, Qg, Pij, Qij, Pji, Qji
device and dependency versions recorded
```

### J2: GOC-500 Canonical Adapter

Goals:

```text
build canonical branch mapping
load/solve one intact economic AC-OPF baseline
verify rateA
verify topology
verify generator limits/costs
verify Pd_pre/Qd_pre
verify branch IDs
```

Pass criteria:

```text
all candidate/risk branches resolve to canonical physical IDs
R_base is nonzero for each scenario
topology vector z maps to GridSFM graph without identity loss
rateA is finite and positive for every risk branch
baseline economic AC-OPF status is successful before scenario admission
```

### J3: Unit, Preprocessing, And Model / Interface Validation

Before wildfire optimization, verify:

```text
baseMVA and GridSFM/PowerModels/MATPOWER units are compatible
raw topology/load is mutated before official GridSFM preparation
no stale prepared graph is mutated
Pd_cmd enters GridSFM unchanged
Qd_cmd enters GridSFM unchanged
topology mutation is correct
generator/network metadata is preserved
branch output mapping is correct
```

This is the Stage J equivalent of the earlier GridFM command/model consistency sanity check.

### J4: Topology Mutation Smoke

Evaluate:

```text
intact topology
one N-1 topology
several N-2 topologies
```

Test source-less-island enforcement and branch mapping. For representative fixed `z`, `alpha` examples, also attempt Reference A exact AC.

### J5: PAC Calibration And Alpha Integration

Before full S1-S3 comparisons, calibrate:

```text
rho_phys
w_op
w_AC
w_model
```

using only intact plus selected N-1/N-2 smoke states. Verify scales, freeze the weights, and do not tune them after seeing full comparative outcomes.

Test:

```text
alpha_i = 1 for every load
uniform full-vector reduction
targeted full-vector perturbation
```

Confirm `L_shed_total`, `L_shed_control`, and `L_shed_island` from `Pd_pre` exactly match the intended control and topology-forced shedding.

### J6: Construct S1-S3

Create:

```text
S1 high-risk / low-impact
S2 high-risk / high-impact
S3 high-risk / redundant path
```

Record qualitative expectations and target/diagnostic branches.

### J7: One-Scenario Method Smoke

Use:

```text
one scenario
lambda_R = 0.5
K <= 2
same outer proxy
same c_l
Guided DC
Guided GridSFM
simple heuristic
```

Run exact AC references for finalists.

### J8: Full S1-S3 Comparison

Run:

```text
lambda sweep
lambda_R_proxy = lambda_R
K <= 2
same candidate/evaluation budget
Guided DC
Guided GridSFM
heuristic reference
```

Add direct DC MIQP only after this primary comparison works.

### J9: Exact AC / Recourse Analysis

For finalists run:

```text
fixed-z-alpha exact reference
fixed-z Rhodes-style AC redispatch
D_state_to_AC / AC diagnostics
Delta_recourse
```

### J10: Warm-Start Study

Compare:

```text
cold
DC
GridSFM
GT ceiling
```

using raw and normalized timing metrics.

Only after J10 should Stage J consider:

```text
larger K
larger candidate sets
S4/S5
randomized scenario distributions
GridSFM fine-tuning
GridFM vs GridSFM
economic-conflict scenarios
MathOptAI / differentiable integration
security-constrained OPS
```

## 29. Locked Stage J v1 Decisions

```text
topology budget:
  K <= 2 initially

interpretation of K:
  experimental search restriction, not fundamental OPS assumption

wildfire-risk assets:
  AC transmission lines only initially

load reference:
  scenario pre-intervention requested demand

load shedding:
  model-independent alpha_requested / alpha_effective accounting

reactive shedding:
  proportional P/Q scaling initially

source-less islands:
  alpha_eff = 0 hard enforcement for all source-less loads

main objective:
  J_trade = lambda_R R_norm + (1 - lambda_R) L_shed_total

physics/model terms:
  GridSFM selection uses rho_phys * PAC_total; report subterms separately

GridSFM:
  frozen released checkpoint

primary methods:
  Guided GridSFM
  Guided DC

heuristic:
  baseline wildfire-risk score only

direct DC MIQP:
  secondary benchmark after guided adapter is stable

scenario suite:
  S1-S3 first

exact AC backend target:
  PowerModels / JuMP / IPOPT unless implementation audit finds a concrete
  reason to use an equivalent existing GridSFM pipeline component

AC endpoint:
  separate fixed-z-alpha state/feasibility reference and fixed-z Rhodes-style
  minimum-load-shed redispatch

environment:
  existing GridFM GraphKit repo plus separate GridSFM environment
```

## 30. Items To Audit During Implementation

These are investigation tasks, not reasons to reopen the Stage J scientific framing.

```text
1. What exact backend does the current repository use to generate c_l?
2. Can that same impact-proxy implementation be reused efficiently on GOC-500?
3. If c_l must be recomputed through DC-MLD or another approximation, what bias
   or sensitivity does that introduce into candidate proposal?
4. What generator-curtailment / minimum-output convention is required for
   disconnected exact AC redispatch cases?
5. How does PowerModels/IPOPT represent multiple electrically disconnected
   components and angle references?
6. How does GridSFM behave on disconnected/source-less/multi-island topologies?
7. What low-dimensional alpha parameterization gives acceptable runtime without
   materially changing the intended load-service control?
8. Can GridSFM outputs be passed cleanly as PowerModels/IPOPT warm-start values,
   and what variable/branch mapping is required?
9. What solver options are required for fair cold/DC/GridSFM warm-start
   comparisons?
```

## 31. Final Conceptual Architecture

```text
          exact intact scenario baseline
                     |
                     v
         baseline loading + p_env
                     |
       +-------------+-------------+
       |                           |
       v                           v
wildfire proxy w_l          service-impact c_l
       |                           |
       +-------------+-------------+
                     |
                     v
           Gurobi topology master
                     |
                     v
                candidate z
                     |
                     v
          external alpha search
                     |
          +----------+----------+
          |                     |
          v                     v
     Guided DC              GridSFM
      recourse            OPF recourse
          |                     |
          +----------+----------+
                     |
                     v
      native wildfire/load objective
      + diagnostic validation
                     |
                     v
               selected z*,alpha*
                     |
          +----------+-----------+
          |                      |
          v                      v
  fixed-z,alpha exact      fixed-z exact AC
    AC state/reference     minimum-load-shed
                           redispatch
          |                      |
          v                      v
     D_model/state       realized AC service
     AC consistency      and wildfire loading
          |
          v
     warm-start study
```

Core interpretation:

```text
topology and load service are wildfire-side decisions

GridSFM predicts the economically dispatched AC electrical state for those
supplied decisions

the wildfire objective trades predicted post-control line exposure against
known unserved scenario demand

exact AC references determine how physically credible/useful the approximate
decision was

DC and GridSFM are compared under the same guided topology-search structure
before stronger secondary baselines are added
```

## 32. Current Design Status

This design is ready for Caleb review but not ready for implementation approval.

Current status:

```text
STAGE_J_PRIMARY_DESIGN_V002_AWAITING_CALEB_REVIEW
```

Recommended next workflow step:

```text
scientific reviewer checks formulation and claims
implementation auditor checks environment, data, and code integration risks
reconciliation creates final implementation plan
Caleb approves or revises Stage J scope
```
