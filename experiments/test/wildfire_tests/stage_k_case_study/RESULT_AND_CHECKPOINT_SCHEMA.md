# Stage K Result And Checkpoint Schema

## Identity Keys

Every candidate is identified by:

```text
run_id
config_sha256
input_sha256
evaluator
lambda_r
k
topology_key
alpha_service_identity
```

`topology_key` is `intact` or the ascending semicolon-delimited canonical
branch IDs. PowerModels IDs are canonical IDs plus one and never replace the
canonical key.

## Candidate Results

`<evaluator>_candidate_results.parquet` contains one row for every attempted
state, including failures. Required fields are:

```text
evaluator, lambda_r, k, topology_key, offline_branch_ids
status, eligible, search_objective
r_norm, l_shed_total, j_trade
pac_operational, pac_ac, pac_model, pac_total, j_total
max_loading, loading_gt_1_count
selected_load_ids, alpha_effective_json
elapsed_seconds, solver_seconds, iterations, peak_memory_mb
message, state_path, metadata
```

PAC and `j_total` are populated only for GridSFM. Reactive and voltage state is
unavailable for DC by definition. Missing method-inapplicable values remain
null; they are never replaced with zero.

Statuses include `optimal_success`, `locally_solved`, `infeasible`,
`iteration_limit`, `time_limit`, `numerical_failure`,
`input_mapping_failure`, `evaluator_exception`, and `duplicate_candidate`.

## State Outputs

PowerModels state directories contain:

```text
*_branch_state.csv
*_load_service.csv
*_bus_state.csv
*_gen_state.csv
*_summary.json
stdout.log
stderr.log
```

AC branch state stores both the physical two-ended loading and the optimization
epigraph. Reported loading and risk are recomputed from physical flows. The
epigraph is retained only for formulation verification.

GridSFM state directories retain the official raw candidate JSON and model
outputs through the Stage J evaluator path. Alpha evaluations are additionally
normalized into `gridsfm_alpha_evaluations.parquet`.

## Checkpoints And Resume

Candidate checkpoints are atomic JSON files under:

```text
evaluators/checkpoints/<evaluator>/lambda_<value>/<topology_key>.json
```

A checkpoint satisfies resume only when `status=complete` and every identity
key matches. Its stored `alpha_service_identity` must equal the result's full
effective service vector. Missing, partial, failed, quarantined, or mismatched
checkpoints raise an error and cannot satisfy completion.

`RUN_COMPLETE.json` makes the run immutable. Evaluator runners refuse to write
after it exists. A new scientific/configuration run requires a new run ID and
directory.

## References And Deltas

Reference tables use one row per evaluator/lambda finalist. Deltas are:

```text
delta_r_a = native_r_norm - reference_a_r_norm
delta_j_a = native_j_trade - reference_a_j_trade
delta_s_b = reference_b_max_service - selected_service
```

GridSFM `delta_j_a` compares `j_trade` with `j_trade`; PAC is reported
separately.

## Derived Evidence

The report package includes primary-source-derived tables for native candidate
comparison, diagnostic empirical nondominance at smoke `lambda_R=0.8`, K1 rank
and parent divergence, convergence/failures, runtime/resources, exact-reference
discrepancies, and full-run workload extrapolation. Figures are never the sole
retained evidence.
