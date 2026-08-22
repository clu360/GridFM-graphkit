# Stage J Complete Run Status

## Run Definition

`run_stage_j_complete.py` is the resumable executor for the primary Stage J
GOC-500 comparison. It retains the locked wildfire-side decision interface:

```text
z_l in {0, 1},  alpha_i in [0, 1] for every load bus i.
```

It evaluates the following per `(scenario, lambda_R)` setting:

```text
Guided-DC
Guided-GridSFM
TH-GridSFM-top1
TH-GridSFM-top2
```

The run uses the coupled sweep `lambda_R_proxy = lambda_R` for `lambda_R` in
`[0.0, 0.2, 0.5, 0.8, 1.0]`, wildfire scenarios `J-S1` through `J-S3`,
`K <= 2`, topology budget `100`, continuous-evaluation budget `20` per
topology, and `q = 5` for the full per-load-alpha optimizer.

## Execution And Evidence

Each method setting is resumable. Once all four Stage J8 method artifacts are
present, the runner builds an explicit finalist manifest and runs J9/J10 for
each finalist:

```text
Reference A: fixed-z, fixed-alpha economic AC-OPF
Reference B: fixed-z AC maximum-load-delivery / economic tie-break
Warm starts: cold, DC partial, GridSFM partial, and exact-AC ceiling
```

Large mutated GridSFM graphs remain in the external cache at
`C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\complete_run_v001`.
Lightweight summaries, candidate traces, topology pools, alpha decisions, and
AC-audit tables are copied to:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run/
```

## Completion Rule

The run is complete only when all 15 settings have successful Guided-DC,
Guided-GridSFM, and TH-GridSFM artifacts and all four finalists have Reference
A, Reference B, and four warm-start records. Failures are retained in the
per-setting external logs and surface as `PARTIAL_OR_FAILED`; the runner does
not silently substitute an alternate model or formulation.

## Completion Record: 2026-08-17

Status: `COMPLETE`

The complete run finished with all `3 x 5 = 15` scenario/lambda settings
marked successful for all methods and all post-hoc AC audits.

```text
topology summaries:           3,030 = 15 * (100 Guided-DC +
                                         100 Guided-GridSFM + 2 TH)
candidate evaluations:       60,600 = 15 * (2,000 + 2,000 + 40)
finalists:                       60 = 15 * 4
Reference A rows:                60 = 15 * 4
Reference B rows:               120 = 15 * 4 * (B1, B2)
warm-start rows:                240 = 15 * 4 * 4
state-fidelity rows:            480
full requested/effective alpha files: 60 each
```

Evidence root:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run/
```

The lightweight result tree includes aggregate tables under `core_results/`,
method/audit evidence in `settings/`, and reviewed visual summaries under
`figures/`. Large mutated GridSFM graphs remain indexed in the external cache.

## Recorded Limitations

- The continuous correction uses the approved `q=5` selected-load Powell
  approximation, not a tractable all-load simultaneous alpha search. The full
  requested/effective vectors are saved, but only five connected loads are
  actively varied per candidate.
- All successful GridSFM calls are marked `model_output_penalized`, which
  means a valid surrogate evaluation with nonzero PAC, not solver failure or
  AC-feasibility certification. `PAC_model=0` and `D_input=0` reflect the
  released API's absent predicted demand channels and input-integrity guard.
- Guided-DC recorded 387 `dc_infeasible` rejections and 12 Gurobi
  no-incumbent numerical events across 30,000 candidates. None was selected
  as a Guided-DC finalist. They remain in `candidate_evaluations_all.csv`.
- During the initial J-S3 AC-reference execution, a source-less-load indexing
  bug and JSON escaping bug were found, repaired, and only the affected exact
  AC references were rerun. Candidate evaluations were not changed.
