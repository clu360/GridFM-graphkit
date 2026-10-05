# Stage K Gate 2 Implementation Handoff

```yaml
artifact_id: STAGE-K-GATE2-IMPLEMENTATION-HANDOFF
artifact_version: v001
created_utc: 2026-09-21
status: IMPLEMENTED_WITH_GRIDSFM_ENVIRONMENT_AMENDMENT_AWAITING_GATE3_PACE_REVIEW
pace_jobs_submitted: false
production_run_authorized: false
```

## Completion Signal

```text
STAGE_K_SMOKE_TEST_IMPLEMENTED_READY_FOR_PACE_REVIEW
```

This signal means the approved smoke experiment is implemented and locally
validated as far as the available Windows environment permits. It does not
certify Phoenix deployment, solver convergence, or scientific results.

## Frozen Configurations

- `config/smoke.yaml`: `lambda_R=[0.8]`, 10 shared K1, five parents per
  evaluator, two K2 children per parent, at most ten unique K2, `q=5`, and
  `B_alpha=5`.
- `config/full.yaml`: five lambdas, 50 shared K1, five parents, 50 children per
  parent, at most 250 unique K2, `q=5`, and `B_alpha=20`.

Both configurations use the same runner and differ only by values.

## Files Created Or Modified

```text
STAGE_K_IMPLEMENTATION_PLAN.md                         modified
GATE2_IMPLEMENTATION_HANDOFF.md                       created
RESULT_AND_CHECKPOINT_SCHEMA.md                       created
PACE_SMOKE_TEST_RUNBOOK.md                            created
config/smoke.yaml                                     created
config/full.yaml                                      created
src/__init__.py                                       created
src/config.py                                         created
src/io_utils.py                                       created
src/schemas.py                                        created
src/identity.py                                       created
src/service.py                                        created
src/objectives.py                                     created
src/topology_candidates.py                            created
src/gridsfm_adapter.py                                created
src/gridsfm_evaluator.py                              created
src/alpha_search.py                                   created
src/native_opf.py                                     created
src/native_opf/Project.toml                           created
src/native_opf/stage_k_native_opf.jl                  created
src/baseline_audit.py                                 created
src/evaluator_runner.py                               created
src/reference_runner.py                               created
src/aggregate.py                                      created
src/prepare_inputs.py                                 created
src/validate.py                                       created
scripts/preflight.py                                  created
scripts/run_evaluator.py                              created
scripts/run_references.py                             created
scripts/aggregate_run.py                              created
pace/stage_k_gridsfm_a100.sbatch                     created
pace/stage_k_dc_gnr.sbatch                           created
pace/stage_k_ac_gnr.sbatch                           created
pace/stage_k_references_gnr.sbatch                   created
pace/submit_smoke.sh                                  created
environment/GRIDSFM_ENVIRONMENT_CONTRACT.json        created by environment amendment
environment/gridsfm_a100_environment.yml             created by environment amendment
environment/gridsfm_a100_constraints.txt             created by environment amendment
environment/download_gridsfm_checkpoint.py           created by environment amendment
environment/setup_gridsfm_a100.sh                    created by environment amendment
environment/README.md                                created by environment amendment
src/gridsfm_environment.py                           created by environment amendment
tests/test_stage_k_gate2.py                           created
gate2_validation/prepared_inputs/canonical_bus.parquet generated
gate2_validation/prepared_inputs/canonical_load.parquet generated
gate2_validation/prepared_inputs/canonical_generator.parquet generated
gate2_validation/prepared_inputs/canonical_branch.parquet generated
gate2_validation/prepared_inputs/branch_model_mapping.parquet generated
gate2_validation/prepared_inputs/proxy_components.parquet generated
gate2_validation/prepared_inputs/shared_k1.parquet    generated
gate2_validation/prepared_inputs/input_manifest.json generated
gate2_validation/validation_report.json               generated
gate2_validation/gridsfm_local_probe.json             generated
```

## Canonical Texas2k Result

The generated and validated canonical package under
`gate2_validation/prepared_inputs/` contains:

```text
buses:                         2,751
Scenario-16 loads:             1,125
generators:                    1,099
online generators:               736
physical branches:             5,344
L_trans (switchable ac_line):  3,993
fixed transformers/other:      1,351
```

The 1,351 fixed branches exactly match both nonzero MATPOWER taps and unequal
endpoint voltage levels. All branches are in service and have positive `rateA`.

Scenario-16 raw parquet demand is 88,719.453 MW. The existing Stage K scaling
contract maps it per bus to the solved MATPOWER demand of 92,895.471 MW and
23,007.429 MVAr. The case-scaled values are authoritative for service,
islanding, GridSFM input, and native OPF.

## Baseline And Proxy

The stored June 23 Scenario-16 loading remains authoritative. The computed
transmission-line baseline is:

```text
R_base = 32.97542569928573
```

The new intact AC solve is implemented only as a consistency audit and cannot
replace this vector.

Stage K directly calls the Stage J implementation:

```text
scenario_builder.connectivity_service_impact_proxy
source SHA-256:
b7ef750859a7baef64f197b5c2f736178467f4d3b44ee0e2a698255cbb2b4f82
```

All 3,993 Texas2k single-line `c_l` values are zero: no eligible single
transmission-line outage creates a source-less load component under this
connectivity-only proxy. This is a data result, not a replacement proxy.

The exact-K proxy is separable:

```text
J_proxy = lambda_R
        + sum_open[(1-lambda_R)c_l - lambda_R*w_l/R_base]
```

Deterministic additive ranking is therefore exactly equivalent to repeated
binary optimization/no-good enumeration for the Stage K exact-K1 and
parent-fixed exact-K2 searches. Unit tests verify it against brute-force
enumeration.

The smoke shared K1 IDs are:

```text
281, 828, 829, 1548, 1950, 1951, 2245, 3272, 3273, 4872
```

## Evaluators

GridSFM builds the official raw schema with 2,751 bus nodes, 736 online
generator nodes, 1,125 load nodes, 3,993 AC-line edges, and 1,351 transformer
edges. It reuses the Stage J mutation, prediction, two-ended flow, PAC, and
topology-relative load-selection conventions. The checkpoint is loaded once
and retained for the GPU job. Powell is bounded by an exact unique-evaluation
cache; the five-evaluation smoke starts at full service and fills duplicate
requests with a deterministic bounded design.

Native DC uses PowerModels `DCPPowerModel` plus Ipopt with internal service,
active balance, generation/branch limits, DC flow equations, active thermal
limits, source-less clamping, squared active-flow risk, and `J_trade`.

Native AC uses `ACPPowerModel` plus Ipopt. For every energized `L_trans` line it
creates an epigraph bounding both squared apparent-power ends. Reported loading
is independently recomputed from physical flows, never from the epigraph.

Economic cost is excluded from both native screening objectives and recorded
only diagnostically.

## Exact References

Reference A fixes topology and the complete effective service vector, then
solves economic AC-OPF with the Texas2k generator costs. Reference B fixes only
topology, maximizes AC load delivery, and applies the Stage J economic tie-break.

Implemented discrepancies are:

```text
Delta_R_A = R_native - R_ReferenceA
Delta_J_A = J_trade_native - J_trade_ReferenceA
Delta_S_B = S_max_ReferenceB - S_selected
```

No warm-start path is present.

## Results And PACE Package

The implementation provides immutable candidate rows, complete failure
accounting, hash-bound atomic checkpoints, full-state pointers, sealed
finalists, Reference A/B outputs, diagnostic empirical nondominance, K1
ranking/parent divergence, convergence/failures, resource summaries, and
full-run workload extrapolation. See `RESULT_AND_CHECKPOINT_SCHEMA.md`.

PACE files include one A100 GridSFM job, GNR-targeted DC/AC/reference jobs,
explicit thread controls, `/usr/bin/time -v` telemetry, optional Slurm account
injection, complete configurable resource arguments, `afterok` dependencies,
and a submission helper that defaults to dry run and requires `--submit`.

`PACE_SMOKE_TEST_RUNBOOK.md` provides the Windows/VPN/SSH through final-summary
workflow. No account identifier is committed.

## Local Validation

```text
canonical preparation:                  PASS
stored baseline binding:                PASS
June 23 / Scenario 16 provenance:       PASS
canonical counts and mappings:          PASS
shared K1 generation:                   PASS
Python compileall:                      PASS
Stage K pytest suite:                   23 passed
official GridSFM raw loading:           PASS
official GridSFM preprocessing:         PASS
released v1.1 intact CPU inference:     PASS
released v1.1 K1 full evaluator probe:  PASS
Python diff whitespace check:           PASS
```

Official preprocessing produced:

```text
bus:       (2751, 16)
generator: (736, 11)
load:      (1125, 2)
branch_ac: (3993, 13)
branch_tr: (1351, 15)
cycle:     (2594, 4)
```

The released checkpoint returned bus state for 2,751 buses, generator state for
736 online generators, and two-ended flows for all 5,344 physical branches.
The full branch-281 K1 evaluator then completed exactly five unique alpha
evaluations and returned 5,343 active physical-branch flows with finite
`r_norm`, load shedding, PAC, and `J_total`. The local environment lacked the
optional `huggingface_hub` import used by the package initializer, so this
already-local checkpoint probe used a no-network import stub. Gate 3 still
requires the complete pinned GridSFM environment. That temporary local probe
mechanism is not referenced by the PACE setup, preflight, evaluator, or job.

## GridSFM Environment Amendment

The official Microsoft GridSFM environment is now frozen as:

```text
repository:            https://github.com/microsoft/GridSFM.git
commit:                1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6
package/version:       gridsfm 1.1.0
checkpoint repository: microsoft/GridSFM_Open
checkpoint revision:   1b41299b80252adf1869d5c3b479a4a402c52591
checkpoint file:       gridsfm_open_v1.1.pt
checkpoint SHA-256:    f8a4396122e603e8303afdebe3b093819c0f64dac0878394aed0bd63205fd831
Python:                3.11.9
torch:                 2.7.1+cu126
torch_geometric:       2.6.1
numpy:                 1.26.4
scipy:                 1.15.3
huggingface_hub:       0.34.4
lightning:             2.6.6
device:                exactly one visible NVIDIA A100 at cuda:0, capability 8.0
CUDA runtime/driver:   12.6 / compatible NVIDIA host driver
fallback:              none
```

Setup checks out and installs the official package, downloads and verifies the
checkpoint once, and makes it read-only. PACE inference sets
`HF_HUB_OFFLINE=1`. The A100 preflight writes the observed package,
checkpoint, source, and device manifest, and only passes after intact Texas2k
inference. The GridSFM job revalidates the same contract before screening.

## Gate 3 Deployment-Audit Amendment

The intact economic AC solve is a deployment sanity probe, not a reproduction
of the authoritative stored Scenario-16 dispatch. CPU preflight therefore
passes only when Julia/PowerModels/Ipopt returns an accepted termination status
and all 5,344 intact physical branches have unique, finite loading outputs.
The stored-versus-solved maximum loading error, weighted risk difference, and
largest discrepancies remain published as diagnostics with
`loading_comparison_gate=false`. The stored baseline, proxy weights, and
`R_base` remain unchanged.

## Gate 3 Validations Still Required

The local machine has no Julia executable and no Bash executable. Consequently
these checks are correctly deferred to Phoenix and must pass before submission:

- instantiate and syntax/run-test Julia, PowerModels, JuMP, and Ipopt;
- run the intact AC baseline consistency audit and review its tolerance result;
- probe one intact and one K1 native DC/AC solve on GNR;
- create and activate the pinned official GridSFM environment;
- verify imports, CUDA 12.6, and exactly one visible A100;
- verify and load the immutable local v1.1 checkpoint;
- run one intact Texas2k GridSFM inference and retain the observed environment manifest;
- validate live Phoenix account, GNR selector, A100 selector, wall times, and
  `sbatch --test-only` syntax;
- inspect dry-run commands before authorizing `--submit`.

No PACE job was submitted and the production configuration was not executed.
