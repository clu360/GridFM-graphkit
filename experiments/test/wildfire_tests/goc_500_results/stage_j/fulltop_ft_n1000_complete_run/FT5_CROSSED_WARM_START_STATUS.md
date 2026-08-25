# FT5 Crossed Warm-Start Study Status

## Status

```text
FT5: COMPLETE
validation: PASS
settings: 15
finalist families: 5
initialization policies: 5
exact Reference A solves: 375
timing-only rerun: COMPLETE
primary runtime metric: IPOPT native solve time
```

FT5 holds every finalist's topology and alpha decisions fixed and changes only
the initialization supplied to the same exact economic AC Reference A problem.
It therefore measures solver-start behavior, not a change in decision quality,
constraints, or objective definition.

## Crossed Design

The retained finalist families are Guided-DC, released-checkpoint
Guided-GridSFM, fine-tuned Guided-GridSFM, and released-checkpoint TH top1/top2.
Every family is evaluated under:

1. Explicit generic cold start: `V=1`, `theta=0`, midpoint `Pg`, zero `Qg`.
2. DC partial start: `Pg`, `theta`.
3. Released GridSFM v1.1 full start: `Pg`, `Qg`, `V`, `theta`.
4. `fulltop_ft_n1000` full start: `Pg`, `Qg`, `V`, `theta`.
5. Exact Reference A solution start.

The primary metric is `result["solve_time"]`, which InfrastructureModels obtains
from `JuMP.solve_time` / `MOI.SolveTimeSec` around `JuMP.optimize!`. It therefore
times IPOPT after the PowerModels/JuMP model and the selected initial values
have been established. `power_models_total_seconds` retains the broader
PowerModels call duration, while model/result overhead and timing-rerun wall
time remain supporting metrics. GridSFM inference is deliberately outside the
primary timer. Frozen and fine-tuned checkpoint provenance is independent of
finalist provenance in every canonical row.

## Validation

```text
canonical rows:                         375
unique setting/finalist/start keys:     375
observations per 25 combinations:        15
successful exact solves:                375
native IPOPT timing rows:                375
maximum objective spread:      8.406234e-06
objective spread tolerance:            1e-3
maximum objective delta vs prior: 5.820766e-11
cold-start audit:                        PASS
full-state payload audit:                PASS
checkpoint provenance audit:             PASS
IPOPT timing validation:                  PASS
```

The objective spread confirms that the five starts converge to numerically
equivalent Reference A solutions for each fixed instance. Warm starts alter the
solver trajectory and runtime, not the solved problem.

## Paired Runtime Results

Across all 75 matched finalist instances per initialization policy:

| Initialization | Mean IPOPT s | Median IPOPT s | Q1 s | Q3 s |
|---|---:|---:|---:|---:|
| Cold | 5.0634 | 5.0450 | 4.9975 | 5.0905 |
| DC partial | 4.7734 | 4.7400 | 4.6480 | 4.8625 |
| Frozen GridSFM full | 4.9108 | 4.8720 | 4.8215 | 4.9410 |
| Fine-tuned GridSFM full | 4.9097 | 4.8820 | 4.8140 | 4.9515 |
| Exact primal | 4.8935 | 4.8720 | 4.7895 | 4.9380 |

The matched comparisons below report median IPOPT seconds saved. Positive
values favor the named start.

| Comparison | Median saved s | Wins | Losses |
|---|---:|---:|---:|
| DC vs cold | +0.329 | 71 | 4 |
| Frozen GridSFM vs cold | +0.186 | 69 | 6 |
| Fine-tuned GridSFM vs cold | +0.178 | 69 | 6 |
| Exact vs cold | +0.191 | 67 | 8 |
| Fine-tuned vs frozen GridSFM | +0.005 | 40 | 35 |

Fine-tuned versus frozen GridSFM by fixed finalist family is:

| Fixed finalist family | Median FT seconds saved | FT wins | FT losses |
|---|---:|---:|---:|
| Guided-DC | +0.003 | 8 | 7 |
| Guided-GridSFM (frozen) | -0.041 | 5 | 10 |
| Guided-GridSFM (fine-tuned) | -0.008 | 7 | 8 |
| TH-GridSFM-top1 | +0.058 | 14 | 1 |
| TH-GridSFM-top2 | -0.011 | 6 | 9 |

DC is the fastest and most consistent initialization in this study. Both full
GridSFM states usually improve IPOPT convergence relative to cold, but frozen
and fine-tuned are effectively tied in aggregate: the FT median advantage is
only 0.005 seconds with a 40/35 split. TH top1 is the clearest family-specific
FT advantage. The earlier broad-runtime result suggesting that FT was slower
for four families is superseded because it included PowerModels model-building
and result-processing time rather than isolating IPOPT.

## Interpretation Limits

- Each setting/start pair has one timed exact solve. The paired design controls
  the optimization instance but does not estimate repeated-run timing noise.
- IPOPT iteration count remains unavailable through the current structured
  PowerModels result, so native IPOPT solve time is the primary trajectory
  measure.
- Exact-start runtime is a best-information reference, not another predictive
  model, and need not be the fastest measured wall-clock run at this scale.
- GridSFM remains valuable for producing a complete AC state and for the Stage J
  search model; this study shows that state completeness alone does not
  guarantee faster IPOPT convergence on every fixed topology/alpha instance.

## Artifacts

```text
working root:
C:\Users\Caleb Lu\.gridfm_stage_j\results\stage_j\fulltop_ft_n1000_complete_run

canonical table:
core_results/warm_start_crossed.csv

primary figure:
figures/stage_j_guided_warm_start_study.png

direct IPOPT summary figure:
figures/stage_j_ipopt_warm_start_summary.png

statistical tables:
core_results/derived_visual_summaries/guided_warm_start_summary.csv
core_results/derived_visual_summaries/guided_warm_start_paired_comparisons.csv
core_results/derived_visual_summaries/ipopt_warm_start_summary.csv
core_results/derived_visual_summaries/ipopt_warm_start_win_loss.csv

validation:
FT5_CROSSED_WARM_START_VALIDATION.json
FT5_IPOPT_TIMING_VALIDATION.json
FT4_VALIDATION.json
```

The refreshed Git publication contains 398 artifacts, 312 Parquet tables, and
201,033 tabular rows. It reduces 188.3 MB of working evidence to 22.9 MB
(`87.9%`) and reports `PASS`. The canonical 375-row warm-start Parquet passed
readback comparison across all 52 columns. The publication also includes the
343-row evaluated native Pareto-frontier table.
