# Texas2k Post-Run Analysis Plan

## Objective

After PACE writes a valid `RUN_COMPLETE.json`, retrieve the immutable
`stage_k_production_v001` package into `full_run/`. Build the final analysis
locally from the retrieved primary tables and states. Do not rerun or alter the
frozen optimization methodology while constructing figures.

The final report must connect every major methodological choice to numerical
results, a suitable figure, supporting primary evidence, and an explicit
interpretation. The goal is a clear basis for assessing Stage K and discussing
subsequent economic, contingency, hazard-model, and optimization extensions.

## Required Analysis

| Methodological choice | Result question | Figure | Supporting evidence |
|---|---|---|---|
| June 23, 2023, 16:00 CDT cumulative hazard with Scenario 16 loading | What environmental and loading conditions define the experiment? | Texas hazard/loading maps and distributions | Frozen snapshot, source hashes, branch table, environmental report |
| Shared K=1 and evaluator-specific K=2 search | Do evaluators rank shared lines similarly, and where do their search paths diverge? | K1 rank/parent divergence and topology-overlap figures | Shared K1, K2 provenance, candidate keys |
| Five `lambda_R` values | How does increasing wildfire-risk emphasis change service, risk, cost, and topology? | Finalist tradeoff curves across `lambda_R` | Candidate and finalist tables |
| GridSFM, DC-OPF, and direct AC-OPF screening | Which evaluator produces the strongest decisions, and at what computational cost? | Risk-service candidate scatter, convergence trajectories, and runtime comparison | Candidate rows, checkpoints, solver diagnostics, accounting |
| GridSFM 20-alpha continuous recourse and squared-loading PAC | Does surrogate recourse improve selection, and where does PAC influence ranking? | GridSFM alpha/PAC diagnostics and native-versus-reference comparison | Alpha evaluations, `J_trade`, `PAC_total`, `J_total`, state files |
| Exact AC Reference A | How accurately does each native evaluator value its selected decision? | Native versus Reference A risk/objective/cost and discrepancy plots | Reference A and `delta_r_a`/`delta_j_a` tables |
| Exact AC Reference B1/B2 | How much AC-feasible service remains, and what is its economic cost when B2 is eligible? | Selected versus maximum service and `delta_s_b`; B2 cost/status figure | Reference B, B1 certification, B2 status/logs |
| Fixed solver and failure policy | Are conclusions affected by convergence failures or warnings? | Convergence/failure-status figure | Failure rows, iterations, logs, B2 warning policy |
| PACE parallel execution | Is the methodology computationally practical at Texas2k scale? | Runtime/resource figure by evaluator and phase | Slurm accounting, elapsed time, memory, device, retries |

## Required Figure Suite

1. Evaluated risk-service scatterplots with nondominated points and finalists.
2. Best-so-far objective convergence by evaluator and `lambda_R`.
3. Reference A native-versus-exact AC comparisons for risk, objective, and
   economic cost.
4. Reference B selected-versus-B1 maximum service and service-gap figures,
   with B2 eligibility and cost shown explicitly.
5. Finalist risk, service, and exact-cost tradeoff curves across `lambda_R`.
6. Solver convergence, failure, and warning summaries.
7. Runtime, memory, CPU/GPU, and per-candidate performance comparisons.
8. K1 ranking, K2 parent, finalist-topology overlap, and divergence figures.
9. Texas maps of selected finalist lines over the frozen hazard/loading data.

Reuse the established Stage J/GOC-500 visual and analytical methodology where
the quantities are comparable. Adapt labels and comparisons to Stage K's three
evaluators, five risk weights, Texas geography, GridSFM PAC definition, and
Reference B1/B2 policy. Retain the primary table behind every figure.

## Completion Standard

The post-run analysis is complete only when:

- the retrieved package passes its hashes and final validation checks;
- candidate, finalist, Reference A/B, failure, and resource accounting are
  complete;
- every reported claim can be traced to a retained table or state artifact;
- figures distinguish native screening metrics from exact AC references;
- B2 warnings and unavailable economic costs are visible rather than imputed;
- methodological limitations and deferred extensions are stated; and
- the report provides a concise evidence-based basis for assessing results and
  selecting the next research step.
