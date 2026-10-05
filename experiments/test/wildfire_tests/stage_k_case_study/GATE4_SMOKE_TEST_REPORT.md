# Stage K Gate 4 Phoenix Smoke-Test Report

## Verdict

The smoke test demonstrates that the frozen Stage K topology-screening workflow
is executable on modified Texas2k. All 63 configured candidate states completed:
21 each for released GridSFM, native DC-OPF, and native AC-OPF. DC and AC were
solver-eligible for every state. GridSFM returned a state for every candidate,
but all 21 states required the frozen physical-output penalty. Production is
therefore computationally feasible, while released-checkpoint GridSFM quality
on Texas2k remains a primary scientific limitation rather than a deployment
failure.

The stored Scenario-16 branch loading remains authoritative, with
`R_base=32.975425699285616`. The June 23, 2023 16:00 CDT cumulative weather
snapshot, 3,993 switchable transmission lines, squared-loading wildfire term,
and all-energized-line AC epigraph were preserved.

## Execution Contract

- Smoke search: one `lambda_R=0.8`, intact K=0, 10 shared K=1 states, and up to
  10 evaluator-specific K=2 states per evaluator.
- GridSFM smoke hardware: one V100, explicitly approved for smoke only after
  the A100 request remained queued. The source, v1.1 checkpoint, package, CUDA,
  device, and intact Texas2k probe passed the frozen V100 preflight contract.
- DC, AC, and references: PowerModels/Ipopt on CPU-GNR.
- Candidate outcomes: 63 attempted, 63 eligible, 24 diagnostic empirical
  nondominated points.
- Reference outcomes: all three Reference A solves and all three Reference B1
  maximum-service solves completed. GridSFM Reference B2 returned
  `OTHER_ERROR`; its B1 maximum-service conclusion remains valid, but its
  economic tie-break is not certified.

## Selected Decisions

| Evaluator | Final topology | K | Native risk | Native shed | Native trade objective |
|---|---:|---:|---:|---:|---:|
| GridSFM | 4872 | 1 | 1.253522 | 0.000000 | 1.002818 |
| DC-OPF | 2245 | 1 | 0.045135 | 0.438182 | 0.123744 |
| AC-OPF | 281;1950 | 2 | 0.133182 | 0.345085 | 0.175562 |

The evaluators disagree materially. AC versus DC and AC versus GridSFM each
share only two of five K=1 parents (Jaccard 0.25). DC versus GridSFM shares four
of five (Jaccard 0.667), despite very different state quality.

## Exact AC References

| Evaluator | Reference A risk | Reference A shed | Reference A objective | Delta risk A | Delta objective A | Reference B service gap |
|---|---:|---:|---:|---:|---:|---:|
| GridSFM | 1.106146 | 0.000000 | 0.884917 | +0.147377 | +0.117901 | approximately 0 |
| DC-OPF | 0.489755 | 0.438182 | 0.479440 | -0.444620 | -0.355696 | 0.438182 |
| AC-OPF | 0.479550 | 0.345085 | 0.452657 | -0.346368 | -0.277094 | 0.345085 |

Reference A shows that the native DC and native risk-aware AC objectives are
not directly interchangeable with economic AC-OPF at fixed topology and
service. This is expected from the two-level methodology and confirms why the
exact AC reference layer is necessary.

GridSFM's selected candidate predicted maximum loading 12.439 and 127 lines
above 1.0 p.u., with `PAC_operational=13.494` and `PAC_ac=1`. Its exact
Reference A is AC-feasible with maximum loading 1.0 and has lower normalized
risk (1.106146) than the native GridSFM estimate (1.253522), although both
exceed the normalized intact baseline of 1. The released checkpoint can participate in
the comparison, but its Texas2k native scores must be interpreted as penalized
surrogate rankings, not calibrated physical estimates.

Reference B1 found 100% maximum service for all three selected topologies. The
DC and AC selected recourse decisions therefore leave 43.82 and 34.51
percentage points of recoverable service, respectively. GridSFM already
selected effectively full service. GridSFM B2 failed its economic tie-break;
the audited table uses the certified B1 state for diagnostics (risk 0.912236,
maximum loading 0.999812, no overloaded lines) and leaves B2 economic cost
unset.

## Runtime And Resources

Observed evaluator job wall times were 1:46 for GridSFM, 12:30 for DC, and
29:47 for the final checkpoint-assisted AC job. Candidate-level accumulated
times are the appropriate clean-run extrapolation because the final AC repair
job reused valid checkpoints from earlier work.

| Evaluator | Smoke candidate time | Full serial estimate | Production scaling |
|---|---:|---:|---|
| GridSFM V100 | 54.49 s | 4.34 h | 71.67x topologies and 4x alpha budget |
| DC-OPF GNR | 201.43 s | 4.01 h | 71.67x topologies |
| AC-OPF GNR | 2,585.22 s | 51.47 h | 71.67x topologies |

The reference runner took 1:35:37 for three evaluators at one lambda. A direct
five-lambda serial estimate is about 7.97 hours. With the three evaluator
families launched concurrently, production wall time is dominated by AC:
approximately 51.5 hours for one serial AC stream plus about 8 hours for serial
references. Splitting evaluator work deterministically by lambda gives a rough
10.3-hour AC chunk and allows the five chunks to run concurrently when the
allocation permits. The downstream reference/report job must depend on every
evaluator chunk.

Peak resident memory was approximately 1.39 GB for GridSFM, 0.64 GB for DC,
0.94 GB for AC, and 1.61 GB for references. The 32 GB requests were safe but
conservative. Production screening plus references is roughly 63.5 CPU-GNR
node-hours and 4.34 V100 node-hours under serial workload extrapolation,
before queueing and retry allowance. A100 production timing and billing must
be measured separately; the V100 figure is a smoke-only conservative estimate.

## Deployment Findings

Two evaluator defects were found and corrected during the smoke run:

1. DC physical loading now uses active power magnitude rather than NaN
   reactive-flow fields from the DC model.
2. Generation-cost diagnostics now tolerate missing cost vectors without
   crashing candidate export.

Two reporting safeguards were added after the final references:

1. GridSFM production timing now includes the alpha-budget increase from 5 to
   20. The earlier 1.1-hour estimate was invalid; 4.34 hours is the corrected
   V100 estimate.
2. Reference B physical diagnostics use B2 only when its solver status is
   eligible. Otherwise they fall back to certified B1, omit economic tie-break
   cost, and report `PASS_WITH_WARNINGS`.

Phoenix completed all reference solves, but the combined two-hour reference
and aggregation job timed out after references because Python startup blocked
in `cl_sync_io_wait` on the cluster filesystem. Aggregation completed locally
in under ten seconds from downloaded immutable Parquet tables. Production
should keep aggregation independently resumable and should not rerun expensive
references when only reporting fails.

## Production Gate

The outer/inner methodology is validated for a full run. Before submission,
the production Slurm package should implement deterministic evaluator chunks,
join all chunk outputs before finalist selection, place references behind all
successful evaluator dependencies, and run aggregation as a separate
resumable step. No change to candidate budgets, objectives, topology rules, or
reference definitions is required.

The full run should not be represented as validating released GridSFM on
Texas2k. It will instead measure how a frozen out-of-distribution surrogate's
penalized topology ranking compares with DC and exact risk-aware AC screening,
followed by common exact AC references.

## Artifacts

The locally retrieved package is under `stage_k_smoke_audited/`. Primary
results are in `evaluators/`, exact references in `references/`, Slurm and GNU
time records in `runtime/`, and derived tables and figures in `report/`.
`report/derived/reference_b_audited.parquet` preserves the B1 fallback and B2
warning explicitly. The Stage K unit and contract suite passes 31 tests.
