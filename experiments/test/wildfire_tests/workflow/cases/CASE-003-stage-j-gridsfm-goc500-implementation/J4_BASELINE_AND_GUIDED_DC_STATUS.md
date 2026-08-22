# J4 Baseline And Guided-DC Status v001

```yaml
artifact_id: CASE-003-J4-BASELINE-GUIDED-DC-STATUS
artifact_version: v001
created_local: 2026-08-12T23:24:00-04:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: SUPERSEDED_BY_J5_J7_SMOKE_STATUS_FOR_LATER_GATES
sha256_when_frozen: null
```

## Intact Economic AC Baseline

Official PGLib OPF repository:

```text
local path: C:/Users/Caleb Lu/.gridfm_stage_j/repos/pglib-opf
commit: dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3
case: pglib_opf_case500_goc.m
```

PowerModels/IPOPT result:

```text
termination_status = LOCALLY_SOLVED
objective = 454945.97831120284
runtime_seconds = 7.707000017166138
baseMVA = 100.0
bus_count = 500
load_count = 281
generator_count = 224 total PGLib rows
active GridSFM generator rows = 171
branch_count = 733 total PGLib rows
active_branch_count = 728
pd_total = 177.72920733832
qd_total = 45.88223415012
```

This passes the initial exact intact AC baseline gate for `case500_goc`. Baseline branch loading was later exported and documented in `J5_J7_SMOKE_STATUS.md`. The remaining exact AC work is to implement fixed-`z,alpha` Reference A and fixed-`z` Reference B for finalists.

## Guided-DC Economic Recourse

Implemented:

```text
fixed external z
fixed full alpha_requested with source-less alpha_effective
economic quadratic generator-cost objective C_gen(Pg)
DC nodal active-power balance
AC-line and transformer DC branch equations
per-unit rateA thermal limits
one angle reference per connected component
infeasible candidates return evaluation_status = dc_infeasible
```

Official GOC-500 DC smoke:

```text
intact:
  evaluation_status = ok
  objective_cost = 440428.2348123738
  Pg variables = 171
  theta variables = 500
  flow variables = 728

N-1 branch 1:
  evaluation_status = ok
  objective_cost = 440442.6664027842
  Pg variables = 171
  theta variables = 500
  flow variables = 727
```

The DC objective is lower than the AC baseline objective, which is expected for an approximate DC economic recourse and should not be interpreted as a better AC feasible objective.

## Validation

Historical local sandbox at time of v001:

```text
python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q
14 passed, 1 skipped
```

The skip is the Gurobi solve under the sandbox user identity.

Historical escalated/user context at time of v001:

```text
python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q
15 passed
```

Current expanded validation after J5-J7 scaffolding:

```text
python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py -q --basetemp=tmp\pytest-stage-j-user
18 passed
```

## Remaining Gate

Before full S1-S3, Stage J still needs:

```text
exact Reference A/B finalist audits
approved full per-load alpha optimization/search strategy, or approved approximation
full J8 runner after alpha decision
result plotting/summary generation
post-run methodology fidelity audit
```
