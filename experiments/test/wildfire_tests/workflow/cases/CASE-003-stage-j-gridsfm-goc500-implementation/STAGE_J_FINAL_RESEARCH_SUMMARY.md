# Stage J Final Research Summary

## Document Purpose

This document is the integrated research record for the completed Stage J
GOC-500 study. It consolidates the experimental motivation, frozen methodology,
GridSFM fine-tuning sequence, OPS application, quantitative results, code paths,
artifact provenance, interpretation, and limitations. It should be the first
document used to understand or write about Stage J.

The final publication package is:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/refined_finetune_study
```

`HISTORY.md` remains the chronological audit trail and intentionally preserves
superseded intermediate results. `FT6_SEQUENTIAL_FINETUNE_STATUS.md` is the
authoritative sealed model-evaluation record. `FT7_REFINED_FINETUNE_STATUS.md`
is the authoritative final execution and provenance record. This summary
connects those records into one scientific account.

## Executive Summary

Stage J asked whether GridSFM, a pretrained AC-OPF foundation model, could be
used as the inner electrical-state evaluator in an existing wildfire-aware
optimal power shutoff (OPS) search on the GOC-500 system, and whether supported
fine-tuning changed its behavior in that modified setting.

The outer OPS logic was held fixed. It selected line-status decisions `z` and
load-service decisions `alpha`; GridSFM did not choose either decision. Given a
candidate network and served load, GridSFM predicted the AC state and branch
flows used to score that candidate. A DC approximation provided the controlled
non-neural comparison. Exact fixed-decision AC OPF was then used to audit final
solutions rather than to guide the search.

The final study compared five methods:

| ID | Method | Training exposure |
| --- | --- | --- |
| DC | Guided-DC | DC approximation; no GridSFM checkpoint |
| M0 | Released GridSFM v1.1 | Released checkpoint |
| M1 | FullTop-1000 | M0 fine-tuned on 1,000 GOC-500 FullTop cases |
| M2 | FullTop-1500 | M1 continued on 500 new FullTop cases |
| M3 | FullTop-1000 + N-1-500 | M1 continued on 500 N-1 cases |

The principal findings are:

1. The released GridSFM checkpoint can be fine-tuned through the repository's
   official OPFData path, reloaded, and substituted into the unchanged Stage J
   evaluator without changing model architecture, output schema, or OPS logic.
2. Fine-tuning materially improved held-out state, flow, cost, and physics
   metrics. M2 was strongest on the FullTop-oriented OPS diagnostic, while M3
   was strongest on the held-out N-1 composite. This is evidence of
   distribution-specific adaptation, not universal model dominance.
3. In final OPS results, M2 had the lowest mean native objective among GridSFM
   variants, the smallest native-to-exact risk and objective discrepancies, and
   the lowest aggregate state NRMSE.
4. The empirical risk/load Pareto analysis used all eligible evaluated
   candidates, not only final solutions: 149,601 of 150,000 rows were eligible,
   producing 603 nondominated points over 15 scenario/model fronts.
5. A controlled 525-solve IPOPT experiment found modest warm-start benefits but
   no strong checkpoint ordering by solve time. The earlier apparent M2/M3
   slowdown was an execution-order artifact and is superseded.
6. A separate paired RQ1 benchmark found that frozen M0 returned a usable AC
   state `3.59x` faster in median total-evaluator time than fresh Reference A
   fixed-decision AC OPF on 54 unique refined-study decisions. This is an
   evaluator-screening result, not an end-to-end OPS speedup.

The study establishes technical compatibility and diagnostic value in this OPS
setting. It does not establish universal economic superiority, complete
security-constrained performance, or a universal solver-speed advantage.

## Research Questions

Stage J addressed three linked questions.

### Q1. Can GridSFM Be Applied To The Modified OPS Problem?

Can a released GridSFM checkpoint accept GOC-500 candidates whose topology and
served loads were modified by an external wildfire-aware search, return the AC
state families expected by the Stage J contracts, and support exact downstream
audits?

### Q2. Does Supported Fine-Tuning Improve The Model?

Does official fine-tuning on solved GOC-500 OPFData improve performance on
strictly held-out FullTop and N-1 test cases? Does adding more FullTop data have
a different effect from adding N-1 topology exposure?

### Q3. Do Those Improvements Transfer To OPS?

When each checkpoint evaluates the same type of Stage J candidate opportunity
set, does fine-tuning change selected solutions, native risk/load tradeoffs,
agreement with exact AC recourse, predicted-state fidelity, or IPOPT warm-start
behavior?

The intended contribution is therefore a controlled application and diagnostic
study. It is not a claim that GridSFM performs topology optimization internally.

### RQ1 Computational-Motivation Addendum

The RQ1 addendum isolates the runtime required to evaluate an already selected
fixed OPS candidate. All 75 M0/M1/M2/M3/DC refined-study provenance rows were
retained, then deduplicated into 54 statistical units using the complete
branch-status vector `z`, effective load-service vector `alpha_effective`,
scaled active demand `Pd`, and scaled reactive demand `Qd`. Exact float-hex
encodings and component hashes make the identity reproducible.

For the primary comparison, timing begins when an initialized evaluator
receives the fixed candidate and ends when it returns a usable electrical
state. It includes candidate mutation, GridSFM preprocessing or AC model
construction, computation, and state extraction. It excludes downstream
wildfire scoring, publication I/O, and one-time environment/model startup.
Reference A uses a fresh ACP model for every repetition and preserves the
existing standard initialization: `V=1`, `theta=0`, midpoint `Pg`, and `Qg=0`.

Frozen M0 used seven total-evaluator repetitions per decision. Reference A used
three repetitions after three persistent-process warmups; all 162 measured
solves were locally solved, returned finite states, and reproduced sealed exact
objectives within `1.56e-08`. Per-decision medians were the paired observations.

| Timing measure | Frozen M0 median | Reference A median | Paired speedup |
| --- | ---: | ---: | ---: |
| Total evaluator | 0.2973 s | 1.0919 s | 3.59x |
| Core forward / IPOPT solve | 0.2224 s | 0.7460 s | 3.35x |

The median total-evaluator speedup had a paired-bootstrap 95% interval of
`3.48-3.68x`; its geometric mean was `3.61x`. The result validates the
computational motivation for using released M0 to screen many candidates and
reserving exact AC OPF for a smaller finalist audit on this CPU/GOC-500 setup.
It does not imply that the complete OPS workflow is accelerated by `3.59x`.
The paper-facing before/after illustration uses the actual M0-selected
`m0:s1_l0p8` finalist (`RQ1U031`): lines 276 and wildfire target 473 are opened,
load 157 is actively curtailed, and the complete frozen-M0 output is displayed
with separate branch-loading and bus-voltage encodings. Its predicted voltage
range is `0.976-1.100 p.u.`, maximum predicted loading is `1.309 p.u.`, and
`PAC_total` is `2.62e-4`; these are prediction diagnostics, not an exact-
feasibility claim. All 54 unique decisions are used only for the paired timing
inference.

## Frozen Experimental Contract

### Division Of Responsibility

The outer search controls:

- binary line status `z_l`;
- continuous load-service fraction `alpha_i`;
- wildfire scenario and risk/load weight `lambda_R`;
- candidate generation, no-good cuts, and topology budget.

The inner evaluator controls only the electrical response assigned to a fixed
candidate. GridSFM returns predicted `Pg`, `Qg`, voltage magnitude, voltage
angle, two-ended active/reactive branch flows, and feasibility output. The DC
path returns its available approximation-backed state. Exact AC OPF is reserved
for post-selection audit and warm-start experiments.

### Load Service And Islanding

For a load in an energized component containing generation,
`alpha_effective = alpha`. For a load in a source-less component,
`alpha_effective = 0`. Active and reactive demand are scaled proportionally.
Total load shedding is computed outside the model so every method receives the
same accounting rule.

This source-less-component guard and the canonical GOC-500 branch identity map
are mandatory. They prevent invalid service claims and line-order mismatches
across GridSFM, PowerModels, Gurobi, and tabular artifacts.

### Risk And Objective

The intact exact AC solution supplies baseline line loading and the normalizer
`R_base`. The AC-state wildfire risk is

```text
R_ac = sum_l z_l * p_env,l * (max(|S_ij|, |S_ji|) / rateA_l)^2
R_norm = R_ac / R_base
```

The DC comparison uses the corresponding active-power loading approximation.
The shared tradeoff objective is

```text
J_trade = lambda_R * R_norm + (1 - lambda_R) * L_shed_total
```

GridSFM candidate selection also includes the predeclared physics penalty:

```text
J_total = J_trade + rho_phys * PAC_total
```

`model_output_penalized` therefore means the candidate was validly evaluated
and penalized for predicted physics residuals. It is not synonymous with a
failed run or an exact AC-feasibility certificate.

### DC Versus GridSFM Comparison Protocol

The DC and GridSFM paths share the symbolic `J_trade` construction, but they do
not instantiate the risk term from the same electrical representation:

```text
DC loading_l = |P_l| / rateA_l

GridSFM loading_l =
    max(sqrt(P_ij^2 + Q_ij^2), sqrt(P_ji^2 + Q_ji^2)) / rateA_l
```

Both square these loading values, weight them by the same `p_env`, and divide
by the same exact intact-case `R_base`. Nevertheless, the DC numerator is based
on active-power flow, while the GridSFM numerator is based on predicted
two-ended apparent-power flow. A shared formula and denominator therefore do
not make the resulting native risk values identical measurements.

The actual candidate-selection merits also differ:

```text
DC search merit      = J_trade
GridSFM search merit = J_total = J_trade + rho_phys * PAC_total
```

The GridSFM-only PAC term is a deliberate guard against operational and physics
inconsistency in a predicted AC state. It also means that the DC search merit
and GridSFM `J_total` must not be presented as values of one identical objective
function.

All Stage J comparisons must consequently follow this hierarchy:

| Comparison | Interpretation |
| --- | --- |
| M0 vs M1 vs M2 vs M3 native metrics | Directly comparable: identical architecture, AC loading definition, PAC construction, and search merit |
| DC vs GridSFM native `J_trade` | Qualified only: shared risk/load form, but method-specific electrical risk approximation |
| DC `J_trade` vs GridSFM `J_total` | Not a direct scalar comparison because only GridSFM includes PAC |
| Exact Reference A/B outcomes | Primary common AC basis for comparing decisions selected by DC and GridSFM |
| Controlled IPOPT warm starts | Direct solver experiment on fixed instances, while recognizing that DC supplies a partial state and GridSFM supplies a full state |
| Full state-fidelity metrics | Comparable among GridSFM checkpoints; DC is N/A where it lacks AC state families |

In this document, **native** means evaluated under the method's own electrical
representation. A lower native DC `J_trade` does not establish a lower AC
objective. Cross-method conclusions should be based primarily on exact
Reference A/B reevaluation and the controlled fixed-instance warm-start study.
The native DC/GridSFM comparison remains useful for studying how different
evaluators guide the same outer OPS workflow, but it is not an apples-to-apples
AC ranking.

### Shared Search Design

All methods retained the same:

- three diagnostic wildfire scenarios, J-S1 through J-S3;
- five risk weights `lambda_R = [0, 0.2, 0.5, 0.8, 1.0]`;
- outage budget `K <= 2`;
- 100 topology opportunities per scenario/weight setting;
- 20 continuous candidate evaluations per topology;
- five selected loads in the bounded continuous correction;
- wildfire inputs, baseline, load-shedding accounting, and exact references.

The shared Gurobi proxy ranks line opportunities using wildfire exposure and
intact-case loading with a common service-impact term. The continuous alpha
step is a bounded derivative-free approximation using five selected loads and
20 evaluations per topology. This approximation is a deliberate Stage J
limitation and is not presented as a globally solved continuous problem.

### Diagnostic Scenarios

The scenarios were constructed to exercise recognizable OPS conditions:

- J-S1: high wildfire risk with relatively low service impact;
- J-S2: high risk with high service impact;
- J-S3: high risk on an electrically redundant alternate-path pattern.

These are controlled diagnostics, not estimates of a real wildfire event
distribution. The small observed shedding range follows from this design,
especially `K <= 2`, the available redundancy, and the incentive structure.

### Exact References

Reference A solves economic AC OPF with the finalist's topology and alpha fixed.
Reference B first maximizes deliverable load under the fixed topology, then
solves an economic tie-break with service locked to that maximum. Reference A
is used for native-to-exact objective, risk, state, and warm-start comparisons;
Reference B tests whether a selected topology forces additional AC load loss.

## Experiment Progression

Stage J was developed through explicit internal and external gates. This kept
pipeline validation, model evaluation, and OPS conclusions separate.

| Phase | Purpose | Outcome |
| --- | --- | --- |
| Initial Stage J | Freeze GOC-500 OPS contract and compare DC, released GridSFM, and topology heuristics | Complete baseline package |
| FT0 | Ten-sample fine-tuning smoke test | Mechanics validated; no scientific claim |
| FT1 | Train FullTop-1000 M1 | Checkpoint and per-epoch validation recorded |
| FT2 | Sealed M0/M1 FullTop and N-1 evaluation | Fine-tuning benefit established before OPS |
| FT3/FT4 | Apply only M1 to frozen Stage J opportunities and merge prior evidence | First fine-tuned OPS comparison |
| FT5 | Cross finalist families and initializers; isolate native IPOPT time | Warm-start methodology corrected |
| FT6 | Train M2/M3 siblings and evaluate four models on a sealed 375+375 set | Sample/topology ablation completed; external approval gate |
| FT7 | Run only M2/M3 in OPS, assemble five methods, expand Pareto and warm-start studies | Final refined study completed |

### Original Complete Run

The first complete package evaluated Guided-DC, Guided-GridSFM M0, and two
frozen GridSFM topology-heuristic finalists. It produced 3,030 topology rows,
60,600 continuous candidate rows, 60 finalists, 60 Reference A solves, 120
Reference B solves, 240 warm-start rows, and 480 fidelity rows. The guided DC
and M0 paths each evaluated 30,000 candidates; the smaller heuristic path
evaluated 600.

That run established the mechanics and exposed the main approximation
differences. Mean native `J_trade` was 0.260473 for DC and 0.333630 for M0.
Mean exact Reference A risk was 0.674001 for DC and 0.680575 for M0, while mean
absolute native-to-exact risk discrepancy was 0.070265 and 0.074987,
respectively. Reference B found full service for all DC finalists; M0 finalists
had mean exact shedding 0.001050 and maximum 0.005248.

The run also resolved implementation issues in exact-reference indexing,
source-less component handling, and JSON serialization. These affected audit
wrappers rather than the frozen candidate decisions and were corrected before
the references were rerun. The DC inner search recorded 387 infeasible and 12
numerical no-incumbent evaluations among 30,000; none became a finalist.

### FT3 Through FT5 Bridge

After FT2 approval, only M1 was rerun through the same opportunity sets. Its
package contained 1,530 topology rows, 30,600 candidates, 45 finalists, 45
Reference A rows, 90 Reference B rows, 180 warm-start rows, and 360 fidelity
rows. Existing DC and M0 evidence was joined by scenario and lambda rather than
recomputed. At that stage, figures also retained the two historical frozen TH
comparisons.

The first candidate-level Pareto revision pooled 90,201 eligible evaluations
and retained 343 nondominated rows across DC, M0, M1, and the two TH series. It
established the all-evaluated-candidate method later used in FT7.

FT5 crossed five finalist families with cold, DC, frozen GridSFM, M1, and exact
primal starts for 375 fixed-instance solves. It then corrected the primary
timing boundary from the broader PowerModels call to solver-reported IPOPT
solve time. FT7 expanded this to four checkpoint starts and reran all 525
solves with balanced execution order and iteration capture. Consequently, FT5
is methodological history; FT7 is the final warm-start evidence.

## Fine-Tuning Methodology

### Compatibility Boundary

Fine-tuning used Microsoft/GridSFM commit
`1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6` and the released v1.1 architecture.
The implementation reused the official `OPFDataAdapterDataset`,
`SyntheticMixedDataset`, GridSFM transform, `finetune_opfdata`, checkpoint
loader, loss, and `eval_pass`. No new architecture, target schema, electrical
state generator, loss, or OPS-specific training objective was introduced.

The official adapter supports `fulltop` and `n1` variants. M3's N-1 continuation
is consequently a supported API extension. It is not described as a
reproduction of the white paper's documented FullTop-only GOC-500 experiment.
Two upstream private helpers, `_hash_state_dict` and `_predicted_flows`, were
used because equivalent public interfaces were unavailable; the GridSFM commit
and environment were frozen to contain that dependency.

### FT0: End-To-End Smoke Test

FT0 verified the complete mechanics on ten FullTop training and ten held-out
test graphs for two epochs:

```text
released checkpoint -> OPFData -> official fine-tune -> saved checkpoint
-> fresh-process reload -> held-out evaluation -> unchanged Stage J candidate
```

All 1,221 floating parameter tensors changed and remained finite. The FT0
checkpoint reloaded with the same output schema and produced different
predictions on the same candidate. Because of the tiny sample and two epochs,
FT0 was pipeline evidence only, not a scientific model-quality result.

### FT1 And FT2: FullTop-1000 Model

M1 used FullTop train indices 0-999, ten epochs, batch size 8, learning rate
`1e-4`, weight decay `1e-4`, infeasibility probability `0.3`, seed 42, and CPU
execution. Validation metrics were recorded by epoch. Training took 9,082.9 s
(151.4 min).

M1 was tested against M0 on 750 held-out FullTop and 750 held-out N-1 graphs.
Selected official metrics were:

| Test stratum | Metric | M0 | M1 | Relative change |
| --- | --- | ---: | ---: | ---: |
| FullTop, 750 | Loss | 0.093351 | 0.041518 | -55.53% |
| FullTop, 750 | Cost MAPE | 0.008835 | 0.007413 | -16.09% |
| FullTop, 750 | Branch-P MAE | 0.074686 | 0.040645 | -45.58% |
| FullTop, 750 | Feasibility accuracy | 1.000000 | 1.000000 | unchanged |
| N-1, 750 | Loss | 0.137594 | 0.085317 | -37.99% |
| N-1, 750 | Cost MAPE | 0.011588 | 0.008723 | -24.72% |
| N-1, 750 | Branch-P MAE | 0.077109 | 0.043933 | -43.02% |
| N-1, 750 | Feasibility accuracy | 0.998667 | 0.993333 | -0.53 points |

The test graphs in this evaluation were feasible, so the small feasibility-
classification change should not be interpreted as a balanced classifier
result. The main evidence was the broad improvement in regression and physics
metrics. FT2 did not yet prove transfer to OPS.

### FT6: Controlled Sample And Topology Ablation

M2 and M3 were sibling continuations from the byte-identical M1 checkpoint,
each with a fresh AdamW optimizer. The frozen data contract was:

| Use | Variant | Split | Indices | Count |
| --- | --- | --- | ---: | ---: |
| M2 continuation | FullTop | train | 1000-1499 | 500 |
| M3 continuation | N-1 | train | 0-499 | 500 |
| validation | FullTop | val | 0-374 | 375 |
| validation | N-1 | val | 0-374 | 375 |
| sealed test | FullTop | test | 0-374 | 375 |
| sealed test | N-1 | test | 0-374 | 375 |

Preflight fingerprinted 2,500 selected graph records. All tensors were finite,
identities were unique within each stratum, and no graph hash crossed training,
validation, or sealed-test uses. All records came from local OPFData caches.

Each branch completed ten epochs and 630/630 batches without skips, changed all
1,221 floating parameter tensors, retained the output schema, and passed a
fresh-process reload. M2 took 4,571.1 s (76.2 min), M3 took 4,524.7 s
(75.4 min), and the sealed four-model evaluation took 1,140.2 s (19.0 min).

### Checkpoint Provenance

| ID | SHA-256 |
| --- | --- |
| M0 released v1.1 | `F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831` |
| M1 FullTop-1000 | `A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD` |
| M2 FullTop-1500 | `08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6` |
| M3 FullTop-1000 + N-1-500 | `4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D` |

Checkpoints remain external to Git. Their paths and hashes are bound in the
training, evaluation, FT7 preflight, and publication manifests.

## Sealed Four-Model Evaluation

### FullTop Test, 375 Cases

| Model | Loss | Cost MAPE | Branch-P MAE | P-KCL residual | Feas. acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.093298 | 0.008437 | 0.074514 | 6.656e-4 | 1.000000 |
| M1 | 0.040794 | 0.007164 | 0.040482 | 2.631e-4 | 1.000000 |
| M2 | **0.038112** | 0.005520 | **0.035024** | **2.209e-4** | 1.000000 |
| M3 | 0.040381 | **0.005228** | 0.036046 | 2.398e-4 | 1.000000 |

### N-1 Test, 375 Cases

| Model | Loss | Cost MAPE | Branch-P MAE | P-KCL residual | Feas. acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 | 0.118472 | 0.011977 | 0.076878 | 6.742e-4 | 0.997333 |
| M1 | 0.070146 | 0.008770 | 0.043597 | 2.924e-4 | 0.992000 |
| M2 | 0.068235 | 0.007789 | 0.038692 | 2.578e-4 | 0.992000 |
| M3 | **0.056797** | **0.006347** | **0.038042** | **2.550e-4** | **1.000000** |

M2 improved every reported lower-is-better metric over M1 on both strata while
preserving feasibility accuracy. M3 had the strongest N-1 loss, cost MAPE, and
feasibility result. Relative to M2, however, it traded away some FullTop,
reactive-flow, Q-KCL, and thermal-overload performance. The controlled result
separates the effect of 500 additional samples from the effect of 500 samples
drawn from a changed topology distribution.

## Final FT7 OPS Study

After the external FT6 review gate, only M2 and M3 were newly run through OPS.
DC, M0, and M1 were reused as hash-bound evidence. Topology pools and Stage J
settings were unchanged. The prior topology-heuristic (TH) runs remain archived
in the original complete packages but were intentionally excluded from this
fine-tuning-focused comparison.

Each new model produced 15 settings, 1,500 topology rows, 30,000 candidate
rows, 15 finalists, 15 Reference A rows, 30 Reference B rows, and 120 fidelity
rows. The combined five-method package contains:

| Artifact family | Rows |
| --- | ---: |
| Topology objectives | 7,500 |
| Candidate evaluations | 150,000 |
| Finalists | 75 |
| Reference A | 75 |
| Reference B | 150 |
| State fidelity | 600 |
| Controlled warm starts | 525 |

All exact solves succeeded, all Reference B service locks passed, and source
provenance and primary-key uniqueness passed.

### Native OPS Outcomes, With Method-Specific Risk

| Method | Mean native method-specific `J_trade` | Mean exact Ref. A cost |
| --- | ---: | ---: |
| DC | 0.260473 | 460,634.43 |
| M0 | 0.333630 | 453,440.84 |
| M1 | 0.308432 | 454,311.95 |
| M2 | **0.303515** | **451,374.56** |
| M3 | 0.309792 | 452,511.89 |

DC had the lowest native `J_trade`, but this is a DC active-power risk score,
not an apples-to-apples AC objective ranking against GridSFM. Its mean exact
Reference A cost was the highest. The M0/M1/M2/M3 native comparison is direct
because all four checkpoints share the same AC loading definition, PAC
construction, architecture, and state completeness.

Relative to M0, M2 reduced mean native `J_trade` by about 9.0% and M3 by about
7.1%. M2 improved on M1 by about 1.6%; M3 was about 0.4% higher than M1. M2
also yielded the lowest mean exact cost among the five methods in this
diagnostic, but the study does not claim general economic dominance from these
15 selected solutions.

### Native-To-Exact Agreement

| Method | Mean abs. risk discrepancy | Mean abs. `J_trade` discrepancy | Mean state NRMSE |
| --- | ---: | ---: | ---: |
| DC | 0.070265 | 0.035777 | N/A across full state families |
| M0 | 0.074987 | 0.030791 | 0.207259 |
| M1 | 0.027569 | 0.009981 | 0.147720 |
| M2 | **0.017622** | **0.005112** | **0.120588** |
| M3 | 0.021816 | 0.007044 | 0.141500 |

State NRMSE averages the seven comparable GridSFM state families. DC does not
provide all reactive, voltage, and two-ended branch-flow families, so missing
DC diagnostics are reported as N/A rather than zero. M2's strongest FullTop OPS
agreement is consistent with its extra FullTop continuation; M3's main benefit
remains its stronger N-1 held-out behavior.

### Reference B Load Delivery

Reference B asks whether the fixed finalist topology can support more load than
the selected alpha requested. In the original complete run, all DC finalists
supported full service. Frozen M0 finalists averaged 0.001050 exact shedding,
with a maximum of 0.005248. These small values are consistent with the
diagnostic design and should not be inflated into a broad resilience claim.
The final publication contains the full five-method Reference B table.

## Empirical Pareto Analysis

The final risk/load figures do not plot only the retained finalist for each
lambda or only one alpha per topology. For each scenario/model pair, all five
lambda-directed candidate pools are combined. Rows with status `ok` or
`model_output_penalized` and finite `R_norm` and `L_shed_total` are eligible.
Duplicate coordinates are removed, then points dominated in both minimized
coordinates are discarded.

```text
150,000 total candidates
149,601 eligible finite candidates
603 nondominated points
15 fronts = 3 scenarios x 5 methods
```

| Method | Nondominated points |
| --- | ---: |
| DC | 111 |
| M0 | 138 |
| M1 | 81 |
| M2 | 93 |
| M3 | 180 |

Scenario totals were 184 for J-S1, 196 for J-S2, and 223 for J-S3. Frontier
size measures sampled diversity, not frontier quality. These are empirical
native fronts over evaluated candidates, not exhaustive feasible-system fronts
and not exact-AC Pareto fronts. Lambda controls candidate discovery; the plotted
risk/load coordinates remain candidate properties.

The five curves must be read as **method-native search diagnostics**. The four
GridSFM frontiers are directly comparable to one another. The DC frontier uses
active-power loading while the GridSFM frontiers use predicted apparent-power
loading, so apparent DC/GridSFM dominance in these plots does not establish
dominance under one common AC risk metric. Constructing a common exact-AC
candidate frontier would require exact AC reevaluation of the candidate set, a
different and substantially larger experiment than the finalist audits in this
study.

## Controlled IPOPT Warm-Start Study

### Construction

The study fixes each of 75 Reference A instances:

```text
5 finalist families x 3 scenarios x 5 lambda settings = 75 instances
```

Every fixed topology and alpha decision is solved from seven initialization
policies, producing 525 rows:

- cold: `V=1`, `theta=0`, `Pg=(Pmin+Pmax)/2`, `Qg=0`;
- DC partial state;
- M0 full state: `Pg,Qg,V,theta`;
- M1 full state: `Pg,Qg,V,theta`;
- M2 full state: `Pg,Qg,V,theta`;
- M3 full state: `Pg,Qg,V,theta`;
- exact Reference A primal state.

This is a crossed initializer study: every start is applied to every finalist
family, rather than applying a model only to the solution it selected. The
same PowerModels/JuMP formulation and IPOPT options are used after initial
values are loaded.

The primary timing field is solver-reported IPOPT solve time. Model
construction, initial-value loading, GridSFM inference, and export are excluded.
Exact A supplies a converged primal state but not IPOPT dual variables, barrier
state, or an optimizer hot restart. It is therefore a best-information primal
reference, not an expectation of near-zero runtime.

### Controlled Result

The final run rotated start order. Each start appeared in every execution
position 10 or 11 times. All 525 solves succeeded, all iteration counts were
positive, and maximum within-instance objective spread was `8.406234e-06`,
below the `1e-3` equivalence threshold.

| Start | Mean time (s) | Median time (s) | Time wins vs cold | Mean iterations | Median iterations | Iteration wins vs cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cold | 5.221 | 5.113 | reference | 33.64 | 33 | reference |
| DC | 5.151 | 5.075 | 37/75 | 31.57 | 30 | 64/75 |
| M0 | 5.160 | 5.104 | 39/75 | 32.25 | 31 | 50/75 |
| M1 | 5.108 | **5.039** | **48/75** | 32.45 | 31 | 51/75 |
| M2 | **5.104** | 5.055 | 45/75 | 32.49 | 31 | 50/75 |
| M3 | 5.159 | 5.067 | 41/75 | 32.32 | 31 | 51/75 |
| Exact A primal | 5.216 | 5.089 | 42/75 | 32.43 | 31 | 44/75 |

The controlled rerun supersedes the earlier mixed-batch conclusion that M2 and
M3 were several seconds slower. M1 and M2 are effectively tied: M2's mean is
lower by only 0.004 s, while M1 has the lower median and more paired time wins.
All GridSFM starts reduce the median iteration count from 33 to 31. DC has the
lowest median iteration count at 30 but not the lowest mean time. Iterations
and solve time are related but not interchangeable because individual IPOPT
iterations can have different computational costs.

The defensible result is a modest warm-start benefit with no strong checkpoint
speed ordering. Better GridSFM prediction quality does not automatically imply
faster nonlinear optimization.

## Findings And Interpretation

### 1. Direct Extension Was Successful

The work demonstrates a clean boundary between the OPS decision process and
GridSFM's supported AC-state prediction role. The model was neither retrained
on the OPS objective nor asked to make unsupported topology decisions.

### 2. Fine-Tuning Improved Fidelity

M1 sharply improved over the released checkpoint. M2 further improved broad
FullTop and N-1 metrics after additional FullTop samples. M3 improved the
targeted N-1 composite after explicit N-1 exposure, with identifiable reactive
and thermal tradeoffs. These results support evaluating a checkpoint on the
distribution where it will be used rather than treating more fine-tuning as a
single scalar notion of improvement.

### 3. FullTop Accuracy Transferred Most Clearly To FullTop-Oriented OPS

M2 was best among GridSFM variants in native OPS objective, native-to-exact
agreement, and aggregate state fidelity. M3 remained much better than M0 and
generally competitive with M1, but its strongest evidence appeared on held-out
N-1 cases rather than this FullTop-oriented OPS diagnostic.

### 4. DC And GridSFM Answer Different Approximation Questions

DC can rank favorably under its native scalar objective and provide an effective
partial warm start. GridSFM supplies a much richer AC state and enables
reactive, voltage, branch-flow, and physics diagnostics unavailable to DC.
Native DC and GridSFM scores should therefore be compared together with exact
recourse, not read as interchangeable measurements of the same state.

### 5. Pareto Evidence Is More Informative At Candidate Level

Using all evaluated candidates reveals the sampled tradeoff surface and avoids
reducing each lambda search to one endpoint. The front remains conditioned on
the search budget and opportunity sets, so it is a diagnostic of evaluated
solutions rather than the mathematical Pareto frontier of the full OPS problem.

### 6. Warm-Start Claims Must Use Controlled Solver Timing

The final study isolates IPOPT's reported solve time and balances execution
order. This changed the interpretation of an intermediate result. The corrected
evidence supports modest convergence help, while showing that prediction
fidelity and solve speed are distinct outcomes.

### 7. Evaluator Speed And Warm-Start Speed Are Separate Claims

The RQ1 benchmark asks how quickly M0 or exact AC OPF can produce a usable state
for a fixed candidate. The warm-start experiment asks whether an already
available state reduces a later IPOPT solve. Frozen M0 was materially faster as
an approximate evaluator, even though GridSFM checkpoint quality did not induce
a strong ordering in controlled IPOPT convergence. These findings are
complementary and must not be combined into one runtime number.

## Limitations And Claim Boundary

- Only one GOC-500 base system and three constructed wildfire scenarios were
  studied.
- `K <= 2`, system redundancy, and the diagnostic objective produced a narrow
  load-shedding range. The work tests method behavior more than severe-event
  welfare consequences.
- The five-load, 20-evaluation alpha correction is budgeted and approximate.
- Pareto fronts are empirical, native, and search-budget dependent.
- Exact AC audits are fixed-decision recourse checks, not security-constrained
  optimization over contingencies.
- M3 learned from N-1 OPFData, but the final OPS scenarios are not a complete
  N-1 security-constrained study.
- The feasibility test sets were highly or entirely feasible and do not support
  broad classification claims.
- DC and GridSFM do not expose identical state families.
- GridSFM inference time is excluded from the IPOPT warm-start timing metric;
  end-to-end deployment speed is a separate question.
- The RQ1 evaluator benchmark includes GridSFM inference and exact model/solve
  construction but excludes outer-search, wildfire-scoring, and publication
  costs; its speedup is hardware- and implementation-specific.
- Exact-primal initialization excludes dual and barrier state.
- The study does not establish statistical generalization across grids,
  wildfire distributions, solver implementations, or hardware.

## Implementation Map

### Core Stage J

| Responsibility | Code |
| --- | --- |
| Contracts and exact-result schema | `stage_j_gridsfm_goc500/contracts.py` |
| Tabular schemas | `stage_j_gridsfm_goc500/schemas.py` |
| GOC-500 identity and mutation guards | `stage_j_gridsfm_goc500/goc500_adapter.py` |
| Source-aware load service and shedding | `stage_j_gridsfm_goc500/load_service.py` |
| Scenario construction | `stage_j_gridsfm_goc500/scenario_builder.py` |
| Shared Gurobi topology proxy | `stage_j_gridsfm_goc500/outer_proxy.py` |
| Fixed-topology DC recourse | `stage_j_gridsfm_goc500/dc_economic_recourse.py` |
| GridSFM candidate evaluation | `stage_j_gridsfm_goc500/gridsfm_evaluator.py` |
| Bounded alpha search | `stage_j_gridsfm_goc500/alpha_optimizer.py` |
| Complete-run orchestration | `stage_j_gridsfm_goc500/run_stage_j_complete.py` |
| Model variant selection | `stage_j_gridsfm_goc500/model_selection.py` |
| Exact AC Reference A/B wrapper | `stage_j_gridsfm_goc500/stage_j_ac_reference.jl` |

All paths in this and the following tables are relative to
`experiments/test/wildfire_tests`.

### Fine-Tuning And Refined Study

| Responsibility | Code |
| --- | --- |
| FT0/FT1 training and fresh reload | `stage_j_gridsfm_goc500/finetune/run_finetune.py` |
| FT2 M0/M1 evaluation | `stage_j_gridsfm_goc500/finetune/run_ft2_evaluation.py` |
| FT3/FT4 M1 OPS extension | `stage_j_gridsfm_goc500/finetune/run_ft3_ft4.py` |
| FT6 data/provenance preflight | `stage_j_gridsfm_goc500/finetune/run_ft6_preflight.py` |
| FT6 sibling training | `stage_j_gridsfm_goc500/finetune/run_ft6_training.py` |
| Sealed four-model evaluation | `stage_j_gridsfm_goc500/finetune/run_ft6_evaluation.py` |
| FT6 tables and figures | `stage_j_gridsfm_goc500/finetune/summarize_ft6.py` |
| FT7 M2/M3 OPS and assembly | `stage_j_gridsfm_goc500/finetune/run_ft7_refined_study.py` |
| Seven-start state preparation | `stage_j_gridsfm_goc500/finetune/run_ft7_warm_starts.py` |
| Controlled IPOPT timing/iterations | `stage_j_gridsfm_goc500/finetune/run_ft7_iteration_timing.py` |
| Final derivations and figures | `stage_j_gridsfm_goc500/finetune/summarize_ft7_refined_study.py` |
| Compact Parquet publication | `stage_j_gridsfm_goc500/finetune/publish_ft7_refined_study.py` |
| RQ1 canonical identity/preflight | `stage_j_gridsfm_goc500/finetune/run_rq1_preflight.py` |
| RQ1 frozen-M0 timing | `stage_j_gridsfm_goc500/finetune/run_rq1_m0_timing.py` |
| RQ1 Reference A timing | `stage_j_gridsfm_goc500/finetune/run_rq1_reference_timing.py` |
| RQ1 figures and publication | `stage_j_gridsfm_goc500/finetune/summarize_rq1_frozen_evaluator.py` |

### Verification

The focused regression suites are:

```text
tests/test_wildfire_stage_j_gridsfm_goc500.py
tests/test_wildfire_stage_j_model_selection.py
tests/test_wildfire_stage_j_ft7.py
tests/test_wildfire_stage_j_rq1.py
```

The final recorded verification result was 44 passed, 2 skipped, with three
deprecation warnings.

## Reproducibility Guide

### Prerequisites

The repository alone is insufficient for a full rerun. The following external
state is required:

- the pinned GridSFM environment and commit;
- local FullTop and N-1 OPFData caches;
- M0 through M3 checkpoint files matching the hashes above;
- Gurobi, Julia, PowerModels, JuMP, and IPOPT installations;
- the external Stage J working root used by the JSON configurations.

Review every JSON configuration before execution because paths are machine-
specific and the complete run takes hours. Manifests fail closed on checkpoint,
source-package, row-count, and method-identity mismatches.

From `experiments/test/wildfire_tests`, the principal commands are:

```powershell
# Focused tests
python -m pytest tests/test_wildfire_stage_j_gridsfm_goc500.py `
  tests/test_wildfire_stage_j_model_selection.py `
  tests/test_wildfire_stage_j_ft7.py

# FT1 and FT2
python stage_j_gridsfm_goc500/finetune/run_finetune.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft1_fulltop_1000.json
python stage_j_gridsfm_goc500/finetune/run_ft2_evaluation.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft2_fulltop_n1_test.json

# FT6 preflight, sibling training, sealed evaluation, and review products
python stage_j_gridsfm_goc500/finetune/run_ft6_preflight.py `
  --m2-config stage_j_gridsfm_goc500/finetune/configs/ft6_m2_fulltop_500.json `
  --m3-config stage_j_gridsfm_goc500/finetune/configs/ft6_m3_n1_500.json `
  --evaluation-config stage_j_gridsfm_goc500/finetune/configs/ft6_evaluation_375_375.json `
  --output-dir <ft6-working-output> `
  --plan workflow/cases/CASE-003-stage-j-gridsfm-goc500-implementation/FT6_FT7_PLAN.md `
  --white-paper <GridSFM-white-paper.pdf>
python stage_j_gridsfm_goc500/finetune/run_ft6_training.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft6_m2_fulltop_500.json
python stage_j_gridsfm_goc500/finetune/run_ft6_training.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft6_m3_n1_500.json
python stage_j_gridsfm_goc500/finetune/run_ft6_evaluation.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft6_evaluation_375_375.json
python stage_j_gridsfm_goc500/finetune/summarize_ft6.py `
  --root <ft6-working-output> --output <ft6-review-output>

# FT7 fail-closed preflight and the only two new OPS model runs
python stage_j_gridsfm_goc500/finetune/run_ft7_refined_study.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json `
  --preflight-only
python stage_j_gridsfm_goc500/finetune/run_ft7_refined_study.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json --model m2
python stage_j_gridsfm_goc500/finetune/run_ft7_refined_study.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json --model m3

# Seven-start states, controlled solve-time/iteration rerun, and publication
python stage_j_gridsfm_goc500/finetune/run_ft7_warm_starts.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json --workers 4
python stage_j_gridsfm_goc500/finetune/run_ft7_iteration_timing.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json --workers 4
python stage_j_gridsfm_goc500/finetune/summarize_ft7_refined_study.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json
python stage_j_gridsfm_goc500/finetune/publish_ft7_refined_study.py `
  --config stage_j_gridsfm_goc500/finetune/configs/ft7_refined_finetune.json

# RQ1 frozen-evaluator preflight, timing, and compact publication
python stage_j_gridsfm_goc500/finetune/run_rq1_preflight.py `
  --config stage_j_gridsfm_goc500/finetune/configs/rq1_frozen_m0_evaluator.json
python stage_j_gridsfm_goc500/finetune/run_rq1_m0_timing.py `
  --config stage_j_gridsfm_goc500/finetune/configs/rq1_frozen_m0_evaluator.json
python stage_j_gridsfm_goc500/finetune/run_rq1_reference_timing.py `
  --config stage_j_gridsfm_goc500/finetune/configs/rq1_frozen_m0_evaluator.json
python stage_j_gridsfm_goc500/finetune/summarize_rq1_frozen_evaluator.py `
  --config stage_j_gridsfm_goc500/finetune/configs/rq1_frozen_m0_evaluator.json
```

The training scripts also provide explicit fresh-process reload modes. The
committed configuration and manifests, rather than this abbreviated command
list, are authoritative for exact paths and parameters.

## Final Artifact Index

The compact final package contains 48 physical files. The publication manifest
counts 38 primary artifacts and lists the remaining ten status/provenance files
as supporting records:

```text
20 Parquet tables
15 PNG figures
12 JSON status/provenance records
1 Markdown result summary
0 CSV files
0 checkpoint files
```

Important tables include `candidate_evaluations_all.parquet`,
`evaluated_native_pareto_frontiers.parquet`, `reference_a_all.parquet`,
`reference_b_all.parquet`, `state_fidelity_all.parquet`,
`warm_start_crossed.parquet`, `ipopt_warm_start_summary.parquet`, and
`ipopt_iteration_summary.parquet`. `PUBLICATION_MANIFEST.json` records each
published path, row count, byte size, and hash.

Parquet is the publication format for large tabular experiment evidence because
typed columnar storage and compression reduce repository size substantially
while preserving the validated data. Future large experimental tables should be
validated, converted to Parquet, read back, and hash/row-count recorded before
publication. CSV may remain a temporary working format outside Git.

The separate RQ1 addendum package is
`goc_500_results/stage_j/rq1_frozen_m0_evaluator_study`. It preserves 75
provenance rows, 54 unique-decision records, raw repetition tables, paired and
summary tables, two figures, phase validations, status records, and a complete
hash manifest without publishing CSV or checkpoint files.

## Paper-Facing Conclusion

Stage J provides evidence that a pretrained AC-OPF foundation model can be
integrated as a modular evaluator within a wildfire-aware topology and load-
service search, then improved through its supported OPFData fine-tuning path.
Fine-tuning improved held-out electrical predictions and, most clearly for the
additional FullTop model, improved agreement between native OPS evaluation and
exact AC recourse. Explicit N-1 exposure produced the strongest N-1 held-out
composite but did not dominate the FullTop-oriented OPS study, demonstrating a
meaningful distribution tradeoff.

The results also clarify what GridSFM does and does not contribute. It supplies
a rich AC-state approximation that supports diagnostics unavailable to DC; it
does not select shutdown decisions, guarantee exact feasibility, or guarantee
faster IPOPT convergence. Candidate-level Pareto analysis and controlled
warm-start timing make those distinctions visible. Native DC and GridSFM
objectives are method-specific search diagnostics, not one common AC ranking;
the exact Reference A/B audits provide the primary cross-method outcome basis.
The strongest workshop-paper framing is therefore a diagnostic study of
foundation-model transfer into a modified OPS workflow, with transparent exact
recourse and fine-tuning ablations, rather than a claim of a completed
real-world wildfire policy tool.
