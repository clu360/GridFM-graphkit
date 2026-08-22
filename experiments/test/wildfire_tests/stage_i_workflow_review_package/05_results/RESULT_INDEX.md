# Result Index

## Primary Complete Run

- Original path: `experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11`
- Package path: `05_results/primary_complete_run`
- Scope: full Stage H K<=2 comparison.
- Coverage documented in handoff: 5 scenarios, 5 lambda_R values, 2 rho values, 6 methods, 300 best rows.
- Scientific interpretation: primary retrospective comparison run.

## Companion Run: Corrected MLD

- Original path: `.../MLD/r5`
- Package path: `05_results/companion_runs/MLD_r5`
- Scope: literature-alignment MLD case with `lambda_R_proxy=1`, `lambda_R=0`.
- Scientific interpretation: companion sub-study, not the main lambda sweep.

## Companion Run: Proxy-Inner Lambda Sweep

- Original path: `.../proxy_inner_lambda_sweep/r2`
- Package path: `05_results/companion_runs/proxy_inner_lambda_sweep_r2`
- Scope: Stage E K2 and Stage I-a DC guided only, rho=0, separated proxy and inner lambda sweeps.
- Scientific interpretation: search-design companion study; no AC projection distances were generated for this run.

## Large Artifacts

Large CSVs above the package threshold are indexed instead of copied. See:

```text
10_machine_readable_inventory/large_artifacts.csv
05_results/large_artifact_summaries/*.json
```
