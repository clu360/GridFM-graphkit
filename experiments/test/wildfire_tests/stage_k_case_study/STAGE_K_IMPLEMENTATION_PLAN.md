# Stage K Texas2k Smoke-Test Implementation Plan

```yaml
artifact_id: STAGE-K-TEXAS2K-IMPLEMENTATION-PLAN
artifact_version: v001
created_utc: 2026-09-16
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: DRAFT_FOR_CALEB_REVIEW
sha256_when_frozen: null
```

## 1. Decision Requested

This plan defines the repository implementation and first PACE smoke run for
Stage K. Approval of this document authorizes implementation and local
validation only. It does not authorize submission of PACE jobs, a production
run, or interpretation of smoke results as scientific findings.

The smoke test answers one operational question: can the frozen Stage K
methodology run end to end on modified Texas2k, with auditable candidate
generation, three native evaluators, exact AC references, resumable PACE jobs,
and complete provenance?

## 2. Evidence And Reuse Boundary

The implementation will use the following repository evidence:

- `workflow/WORKFLOW_ARCHITECTURE.md` and
  `workflow/primary_agent/AGENT_INSTRUCTIONS.md` for authority and lifecycle;
- `workflow/shared/RESEARCH_INVARIANTS.md` for physical-branch, thermal-rating,
  islanding, and evaluator-interpretation constraints;
- `workflow/cases/CASE-003-stage-j-gridsfm-goc500-implementation/`
  `STAGE_J_FINAL_RESEARCH_SUMMARY.md` for the accepted Stage J methodology;
- Stage J `goc500_adapter.py`, `scenario_builder.py`, `outer_proxy.py`,
  `load_service.py`, `metrics.py`, and the J8-J10 runners for tested concepts;
- Stage J AC reference Julia files for Reference A and Reference B semantics;
- the frozen Stage J PAC calibration: `rho_phys=2.0`, `w_op=1.0`,
  `w_ac=1.0`, and `w_model=0.0`;
- the Stage K raw Texas2k files and the completed cumulative environmental
  snapshot under `../texas_2k_results/environment_snapshot_tau0p50/`.

Stage J is a methodological source, not a code path to mutate. Stage K will
reuse stable helpers where their contracts still hold and place Texas2k
adapters, native evaluators, orchestration, and outputs in the Stage K package.
No frozen Stage J result or workflow artifact will be modified.

## 3. Frozen Scientific Contract

### 3.1 Snapshot

```text
network: modified Texas2k
electrical state: load Scenario 16 (4 PM)
weather date: June 23, 2023
weather time: 16:00 CDT / 21:00 UTC
environmental hazard: p_env = p_cumulative
screening wildfire term: p_cumulative * loading^2
```

June 23 is frozen. The source `.pww` filename containing `2023-06-25` denotes
the weather case/export, not the selected timestamp. The manifest must bind the
source file hash and the exact June 23 timestamp to prevent that distinction
from being lost.

### 3.2 Common Objective

For every native evaluator:

```text
R_base = sum_l p_env,l * (loading_base,l)^2
R_norm = sum_l z_l * p_env,l * (loading_l)^2 / R_base
L_shed = total unserved active demand / total requested active demand
J_trade = lambda_R * R_norm + (1 - lambda_R) * L_shed
```

`z_l=1` means the physical transmission line is energized. AC and GridSFM use
two-ended apparent-power loading:

```text
loading_l = max(|S_from,l|, |S_to,l|) / rateA_l
```

DC uses active-power loading:

```text
loading_l = |P_l| / rateA_l
```

GridSFM alone uses:

```text
J_total = J_trade + rho_phys * PAC_total
```

with the frozen Stage J PAC weights. Native DC and native AC rank candidates by
`J_trade`. Economic generation cost is deliberately absent from native
screening and is evaluated downstream by Reference A/B.

### 3.3 Topology Scope

The controllable set `L_trans` contains canonical physical AC transmission
lines that are in service and have usable positive `rateA`. Transformers and
other branch families remain electrically represented and fixed. Duplicate
directed model edges map to one physical switching decision.

The implementation must derive and publish the actual Texas2k mapping and
counts. It must not remove a candidate merely because its outage islands load,
causes overload, produces poor service, or challenges a solver. Such behavior
is an evaluator outcome. A branch may be excluded only for a documented
identity/family/rating reason specified before evaluation.

For every topology, load in a connected component without an online source is
hard-set to unserved before evaluator scoring. This is common preprocessing,
not optional recourse.

## 4. Configuration Contract

Smoke and production use one runner and one schema. Only configuration values
change.

| Setting | Smoke | Full |
| --- | ---: | ---: |
| `lambda_r` | `[0.8]` | `[0.0, 0.2, 0.5, 0.8, 1.0]` |
| intact `K=0` | 1 | 1 |
| shared exact-`K=1` candidates | 10 | 50 |
| retained `K=1` parents/evaluator | 5 | 5 |
| exact-`K=2` children/parent | 2 | 50 |
| maximum unique `K=2`/evaluator | 10 | 250 |
| maximum modified states/evaluator/lambda | 20 | 300 |
| maximum states including intact | 21 | 301 |
| GridSFM selected loads `q` | 5 | 5 |
| GridSFM alpha evaluations/topology | 5 | 20 |

The word `maximum` is intentional. Duplicate K2 children, exhausted valid
children, or explicit solver-independent construction failures may produce
fewer unique candidates and must be reported. Evaluator failures do not reduce
the attempted candidate count and remain result rows.

All seeds, tolerances, time limits, solver versions, candidate ordering,
resource requests, and paths are resolved into a run-specific immutable config.

## 5. Implementation Package

The implementation will add this logical structure under
`stage_k_case_study/`:

```text
config/
  smoke.yaml
  full.yaml
src/
  identity.py
  prepare_inputs.py
  topology_candidates.py
  service.py
  objectives.py
  gridsfm_evaluator.py
  native_dc/
  native_ac/
  references/
  aggregate.py
  validate.py
scripts/
  preflight.py
  run_evaluator.py
  run_references.py
  aggregate_run.py
pace/
  stage_k_gridsfm_a100.sbatch
  stage_k_dc_gnr.sbatch
  stage_k_ac_gnr.sbatch
  stage_k_references_gnr.sbatch
  submit_smoke.sh
PACE_SMOKE_TEST_RUNBOOK.md
tests/
```

Names may be adjusted to established repository conventions during
implementation, but the ownership boundaries and outputs below remain fixed.

## 6. Phase K0: Canonical Inputs And Preflight

### K0.1 Parse And Bind Inputs

The preflight will:

1. parse `modifiedTexas2k.m`, including bus, generator, branch, and nonconstant
   `gencost` data;
2. load Scenario 16 demand from
   `modifiedTexas2k_loads_scenarios_0_23.parquet` and verify bus alignment;
3. load branch coordinates and the June 23 cumulative environmental snapshot;
4. verify one-to-one joins, units, finite values, and complete hazard coverage;
5. hash every source and write a resolved input manifest.

The scenario demand becomes the canonical requested demand for service and
shedding calculations. The raw MATPOWER demand must not silently substitute
for it.

### K0.2 Canonical Identity Tables

Preflight writes:

```text
canonical_bus.parquet
canonical_load.parquet
canonical_generator.parquet
canonical_branch.parquet
branch_model_mapping.parquet
```

The branch table includes canonical ID, MATPOWER row, from/to buses, branch
family, transformer flag/reason, status, `rateA`, switch eligibility,
environmental coverage, `p_cumulative`, baseline loading, and exclusion reason.
The model mapping records every GridSFM directed edge associated with each
physical branch.

### K0.3 Baseline State

The intact Scenario 16 state is solved with the same AC network conventions
used by Stage K AC evaluation. It supplies `loading_base` and `R_base`.
Preflight compares it against any existing baseline-loading artifact and
reports discrepancies; it does not silently inherit an unverified loading
column. Failure to obtain a finite, converged intact AC baseline is a hard stop.

### K0.4 Service-Consequence Proxy

Stage K will reuse the Stage J source-less single-outage logic from
`scenario_builder.connectivity_service_impact_proxy`, generalized only through
the Texas2k identity adapter:

```text
c_l = source-less active demand after outage l / total active demand
```

The graph includes every electrically represented in-service branch family,
while only `L_trans` is opened. Unit tests will compare the reused function with
hand-checkable islanding cases and selected Texas2k outages.

### K0.5 Environment Checks

Preflight validates:

- Python environment and package lock;
- Julia environment, PowerModels, and Ipopt;
- GridSFM source version, checkpoint path/hash, CUDA, and one A100 inference;
- Gurobi/license only if retained by the topology proxy implementation;
- writable project, scratch, cache, and result paths;
- adequate disk space and no output collision;
- CPU/GPU resource variables and Slurm command availability on PACE.

Preflight emits `PREFLIGHT_PASS.json`. No evaluator job may run without its
matching config/input hashes.

## 7. Phase K1: Shared And Evaluator-Specific Topologies

The Stage J proxy is retained:

```text
w_l = p_cumulative,l * (loading_base,l)^2
R_proxy = sum_l w_l * (1-y_l) / sum_l w_l
L_proxy = sum_l c_l * y_l
J_proxy = lambda_R * R_proxy + (1-lambda_R) * L_proxy
```

where `y_l=1` means line `l` is opened.

Candidate construction is deterministic:

1. Generate the shared exact-`K=1` pool with `sum(y)=1`, ordered by proxy
   objective and deterministic canonical-ID tie-breaks.
2. Evaluate intact and all K1 states with GridSFM, DC, and AC.
3. Within each evaluator, rank converged/eligible K1 outcomes by its native
   objective and retain the best five parents. Failed outcomes remain recorded
   but are ineligible as parents.
4. For each retained parent, fix its opened line and generate the configured
   number of exact-`K=2` children using the same proxy, `sum(y)=2`, and one
   additional line.
5. Deduplicate children by sorted canonical topology key while preserving
   parent provenance. Continue down the deterministic child ordering until the
   requested unique budget is met or the opportunity set is exhausted.
6. Evaluate each evaluator's own K2 pool only with that evaluator.

This produces shared K1 evidence and evaluator-specific K2 expansion without
allowing an evaluator to change the proxy definition. Candidate manifests will
record rank, parent, proxy components, topology key, and generation reason.

## 8. Phase K2: Native Evaluators

### K2.1 GridSFM

For each fixed topology:

1. hard-clamp source-less loads to zero service;
2. select `q=5` controllable loads by the Stage J topology-relative rule:
   graph distance from opened-line endpoints, then descending demand, then
   canonical load ID;
3. hold all other connected loads at full requested service;
4. search selected alpha values in `[0,1]` under the configured exact
   evaluation budget;
5. run the released GridSFM checkpoint with one resident model per job;
6. compute two-ended AC loading, `R_norm`, shedding, PAC components,
   `J_trade`, and `J_total`;
7. select the alpha candidate with minimum finite `J_total`.

Checkpoint path, SHA-256, source commit, model variant, device, inference time,
and full requested/effective alpha vectors are mandatory. GridSFM output is a
surrogate state and never receives an AC-feasible label.

### K2.2 Native DC

Each fixed topology is solved once with internal service variables over the
full connected load set. The formulation contains bus angles, active branch
flows, generator active powers, and bounded per-load service fractions.
Source-less service fractions are fixed to zero. It enforces nodal active-power
balance, generator limits, DC branch equations, branch statuses, and active
thermal limits.

The objective is `J_trade` using squared normalized active flow for wildfire
risk. Generator cost is recorded diagnostically if defined but is not part of
screening. The planned backend is PowerModels plus Ipopt for architecture
consistency with the requested PACE study; if implementation inspection shows
that a native convex QP backend is materially more reliable, changing the
backend requires a versioned amendment before results are compared.

### K2.3 Native AC

Each fixed topology is solved once using PowerModels `ACPPowerModel` and Ipopt
with internal service variables over the full connected load set. Service
scales both active and reactive demand at fixed power factor. The model enforces
the nonlinear AC equations, voltage limits, generator P/Q limits, branch
statuses, and two-ended thermal limits.

For each candidate line, an epigraph variable bounds both
`|S_from|^2/rateA^2` and `|S_to|^2/rateA^2`; the weighted sum forms the
numerator of `R_norm`. This implements the Stage J two-ended max convention
without a nonsmooth max in the objective. Native AC screening minimizes
`J_trade`, not fuel cost.

### K2.4 Common Failure Policy

Every attempted topology produces a row. Status distinguishes optimal,
locally solved, iteration/time limit, infeasible, numerical error, input error,
and evaluator exception. Nonfinite or unsuccessful outcomes cannot become
parents or finalists, but they are retained as scientific evidence. Retry
rules and solver options are fixed in configuration and cannot be changed after
seeing which topology failed.

For each evaluator and lambda, the finalist is the eligible minimum objective
across K0, K1, and K2 with deterministic tie-breaks: lower objective, lower K,
lexicographic topology key.

## 9. Phase K3: Exact AC Reference Studies

References run only after all three evaluator manifests pass and finalists are
sealed. Each evaluator contributes one finalist per lambda.

### Reference A: Fixed Decision Economic AC-OPF

Fix the selected topology and the finalist's complete effective service vector,
then solve economic nonlinear AC-OPF with the original Texas2k generator cost
curves. Report convergence, cost, exact AC risk/loading, selected-versus-exact
state discrepancy, and service consistency.

For GridSFM this audits the surrogate-selected alpha vector. For DC it audits
the DC-selected service vector in AC physics. For native AC it independently
re-solves the same topology/service command under economic dispatch, separating
screening dispatch from economic dispatch.

### Reference B: Fixed Topology Maximum Load Delivery

Fix only the selected topology. Stage B1 maximizes AC load delivery with
fixed-power-factor service variables and source-less clamping. Stage B2 applies
the existing Stage J economic-cost tie-break within the accepted B1 service
tolerance. Report maximum service, economic cost, exact risk/loading, and the
gap between finalist service and topology-available service.

References use cold starts for the primary result. No warm-start study is part
of Stage K.

## 10. Phase K4: PACE Execution Design

PACE contains four logical jobs:

| Job | Resource | Initial request | Responsibility |
| --- | --- | --- | --- |
| GridSFM | A100 GPU | 1 A100 plus normal host CPU/RAM | K0/K1/K2 surrogate evaluation |
| DC | CPU-GNR | 1 node, 8 CPUs, 32 GB | native DC evaluation |
| AC | CPU-GNR | 1 node, 8 CPUs, 32 GB | native AC evaluation |
| References | CPU-GNR | 1 node, 8 CPUs, 32 GB | Reference A/B after finalists |

Partition/constraint names, account, QoS, wall time, CPUs, memory, and paths are
configurable submission variables, not scientific constants. No personal
charge-account identifier is committed. CPU scripts use the same GNR class for
DC, AC, and references so their timings are comparable.

The initial smoke favors independent-job throughput rather than allocating a
whole 192-core node to one Ipopt solve. Every solve records elapsed time, solver
time, IPOPT iterations, termination status, peak process memory, requested
resources, allocated host, and CPU utilization where Slurm exposes it. Smoke
evidence determines whether an 8-versus-16 CPU benchmark is warranted before
production.

The dependency flow is:

```text
preflight + immutable K1 pool
             |
             +--> GridSFM smoke --+
             +--> DC smoke -------+--> seal finalists --> Reference A/B
             +--> AC smoke -------+                         |
                                                            v
                                                     aggregate + validate
```

The submission helper prints resolved commands and job IDs, uses `afterok`
dependencies, and requires an explicit submit flag so validation cannot launch
jobs accidentally. This implementation phase will not invoke `sbatch`.

## 11. First-Time PACE Runbook

`PACE_SMOKE_TEST_RUNBOOK.md` will provide a literal Windows-to-results path:

1. connect to GT VPN and open PowerShell;
2. SSH to Phoenix;
3. inspect identity, available accounts, partitions, and GNR/A100 access;
4. create project/scratch/cache locations using placeholders;
5. synchronize repository, Texas2k inputs, and checkpoint;
6. create/activate the Python environment;
7. verify CUDA, A100, GridSFM, and checkpoint hash;
8. create/activate the Julia environment;
9. verify PowerModels, Ipopt, and optional Gurobi license;
10. run preflight and inspect `PREFLIGHT_PASS.json`;
11. dry-run submission and inspect resolved resources;
12. submit evaluator jobs, monitor with Slurm, and inspect logs;
13. allow the dependency job to run references and aggregation;
14. inspect `smoke_summary.md`, failures, utilization, and timing estimates;
15. revise only resource configuration before production, or create a
    versioned methodology amendment if scientific settings must change.

Every step will include the exact command, purpose, expected success signal,
and a bounded troubleshooting path. Committed examples use `<GT_USERNAME>`,
`<PACE_ACCOUNT>`, `<PROJECT_DIR>`, and `<SCRATCH_DIR>` placeholders. 
 
## 12. Output And Checkpoint Contract 
 
Each run writes to a new run ID and never overwrites a completed run: 
 
```text 
results/<run_id>/ 
  resolved_config.yaml 
  provenance/
    input_manifest.json
    environment_manifest.json
    code_manifest.json
    checkpoint_manifest.json
  inputs/
    canonical_*.parquet
    branch_model_mapping.parquet
    baseline_state.parquet
    proxy_components.parquet
  candidates/
    shared_k1.parquet
    <evaluator>_k2.parquet
  evaluations/
    candidate_results.parquet
    alpha_evaluations.parquet
    state/
    logs/
  finalists/finalists.parquet
  references/reference_a.parquet
  references/reference_b.parquet
  runtime/resource_usage.parquet
  validation/validation_report.json
  smoke_summary.md
```

Primary keys include run ID, config hash, input hash, evaluator, lambda, K,
sorted topology key, and alpha/service identity where applicable. Checkpoints
are atomic and resumable only when those keys and hashes match. Partial,
failed, and quarantined outputs cannot satisfy completion checks.

`candidate_results` includes topology provenance, service vector pointer,
objective components, risk/loading, shedding decomposition, convergence,
solver diagnostics, and runtime. Full state vectors are stored separately to
avoid duplicating large records.

## 13. Verification Plan

Local tests before PACE submission will cover:

- MATPOWER and Scenario 16 parsing and bus alignment;
- canonical physical branch uniqueness and directed-edge mapping;
- transformer/fixed-element classification and `L_trans` count reconciliation;
- weather join, June 23 timestamp, source hashes, and complete branch coverage;
- source-less island clamping and Stage J `c_l` reuse;
- risk, service, `R_base`, `J_trade`, and PAC calculations;
- exact K1/K2 cardinality, parent fixation, deduplication, and deterministic
  ordering;
- smoke/full config expansion and expected maximum counts;
- tiny synthetic DC and AC cases with known service/risk behavior;
- Reference A fixed-service and Reference B released-service semantics;
- schema, resume, failure-row, and no-overwrite behavior;
- Slurm-script syntax and submission dry run without launching jobs.

PACE preflight then runs one intact and one K1 evaluator probe for each backend
before the full smoke arrays. These probes validate deployment only and use the
same code/config; they do not replace smoke rows.

Smoke acceptance requires:

```text
preflight status: PASS
shared K1 identity: identical across evaluators
attempted candidate accounting: complete
all successful rows: finite required metrics
all failed rows: explicit classified status and logs
finalists: exactly one eligible finalist/evaluator/lambda
Reference A/B: attempted for every sealed finalist
provenance/config/checkpoint hashes: complete
aggregation recomputation: exact within declared tolerances
resource/timing telemetry: present for all jobs
```

A solver failure on a difficult topology is retained evidence and does not by
itself fail the package. Missing accounting, silent fallback, objective drift,
or an untraceable result does fail it.

## 14. Implementation Gates And Stop Conditions

### Gate 1: Plan Approval

Caleb approves or amends this plan. Only then does implementation begin.

### Gate 2: Local Package Review

Implementation returns the files, resolved smoke/full configs, actual
Texas2k `L_trans` mapping/count, reused `c_l` evidence, evaluator equations,
test results, PACE scripts, runbook, and remaining PACE-only validations. No
jobs are submitted.

### Gate 3: PACE Preflight Review

The user executes the runbook through preflight and reviews resolved account,
resource names, environment, checkpoint, and input hashes before submission.

### Gate 4: Smoke Review

After smoke completion, review convergence, failures, objective accounting,
runtime, memory, CPU utilization, and queue behavior. Production is not
automatically authorized.

Hard stops include:

- June 23 or Scenario 16 provenance mismatch;
- missing/ambiguous physical branch or GridSFM edge mapping;
- nonpositive/missing ratings in a supposedly controllable line;
- failure of the intact AC baseline;
- checkpoint/source/environment hash drift;
- inability to represent modified Texas2k in the released GridSFM interface;
- changed objective, PAC weights, budgets, or reference definitions;
- accidental economic-cost optimization in native screening;
- silent candidate filtering, silent solver fallback, or missing failure rows;
- output reuse with mismatched hashes;
- any attempt to submit PACE jobs during implementation.

## 15. Known Risks And Planned Responses

| Risk | Consequence | Planned response |
| --- | --- | --- |
| Released GridSFM may not accept the Texas2k representation directly | surrogate path blocked | fail in adapter probe; document the exact incompatibility; do not reshape silently |
| Nonconvex native AC screening may converge locally or fail | incomplete/solver-sensitive ranking | fixed solver policy, explicit status, retained failures, exact references, no post-hoc tuning |
| The AC risk epigraph increases nonlinear model size | slow solves | smoke timing and memory evidence; parallelize topologies in production |
| K2 duplicates across parents | fewer unique states | global topology-key deduplication and deterministic budget refill |
| PACE partition/account syntax differs from assumptions | submission failure | configurable resources and runbook discovery commands |
| Ipopt gains little from extra cores | wasted allocation | begin at 8 CPUs; use telemetry before any 16-CPU benchmark |
| Gurobi unavailable for proxy generation | preflight failure | keep proxy backend isolated; implement a deterministic enumerative equivalent if validated before run |
| OneDrive/local path behavior affects artifacts | unreliable local publication | run outputs on PACE project/scratch, atomic writes, hash-verified retrieval |

## 16. Deferred Extensions

The following are explicitly outside this first Stage K implementation:

- quantile-based environmental hazard definitions;
- June 25 or multi-day environmental comparison;
- hospitals or other high-criticality artificial demand classes;
- N-1 contingency analysis and robust/stochastic optimization;
- GridFM OPF checkpoint comparison;
- GridSFM fine-tuning;
- warm-start experiments;
- heuristic topology baselines;
- production budgets or scientific claims from smoke data.

These remain natural extensions after the Texas2k smoke establishes decision
quality, computational behavior, and implementation validity.

## 17. Planned Implementation Completion Signal

After Gate 2 is genuinely satisfied, the implementation handoff will use:

```text
STAGE_K_SMOKE_TEST_IMPLEMENTED_READY_FOR_PACE_REVIEW
```

That signal means the package is ready for user review and PACE preflight. It
does not mean PACE execution or scientific validation is complete.
