# Stage J GOC-500 Complete Run

## Scope

This result package contains the locked Stage J S1-S3 comparison of Guided-DC,
Guided-GridSFM, TH-GridSFM-top1, and TH-GridSFM-top2. It uses the coupled sweep
`lambda_R_proxy=lambda_R` for `lambda_R=[0, 0.2, 0.5, 0.8, 1]`, `K<=2`, topology
budget 100, continuous budget 20 per topology, and the approved `q=5`
selected-load alpha correction.

`RUN_STATUS.json` is `COMPLETE`: all 15 setting bundles include four finalists,
Reference A, Reference B B1/B2, and cold/DC/GridSFM/exact warm starts.

## Primary Evidence

| Artifact | Records | Meaning |
| --- | ---: | --- |
| `core_results/topology_objectives_all.csv` | 3,030 | Best continuous result per evaluated topology |
| `core_results/candidate_evaluations_all.csv` | 60,600 | Every bounded alpha evaluation, including rejections |
| `core_results/method_finalists_all.csv` | 60 | Four finalists per scenario/lambda setting |
| `core_results/reference_a_all.csv` | 60 | Fixed-topology/fixed-alpha economic AC-OPF |
| `core_results/reference_b_all.csv` | 120 | Fixed-topology AC maximum-load-delivery plus B2 tie-break |
| `core_results/warm_start_all.csv` | 240 | Four starts per Reference A instance |
| `core_results/state_fidelity_all.csv` | 480 | Component-level native-to-Reference-A state comparisons |

The `settings/` directory contains method-level artifacts and all 60 full
requested/effective alpha-vector pairs. Large per-candidate GridSFM graph files
remain in the external cache named in `RUN_CONFIG.json`.

## Reviewed Figures

- `figures/j_trade_by_topology_rank_j-s1.png` through `j-s3.png`: common
  wildfire/service objective for every evaluated topology.
- `figures/j_trade_convergence_j-s1.png` through `j-s3.png`: best common
  objective observed over bounded candidate evaluations.
- `figures/evaluated_risk_load_scatter_j-s1.png` through `j-s3.png`: evaluated
  `R_norm` and total load-shedding outcomes by lambda and method.
- `figures/stage_j_guided_reference_a_discrepancy_distance.png`: Guided-DC
  versus Guided-GridSFM finalists compared against fixed-topology/fixed-alpha
  Reference A economic AC-OPF cost, native-vs-AC discrepancy, and
  native-state distance to Reference A.
- `figures/stage_j_guided_reference_b_mld.png`: Reference B maximum-load-
  delivery outcomes for Guided-DC and Guided-GridSFM finalist topologies,
  plotted as AC served-load fraction `1 - L_shed_ac_mld`.
- `figures/stage_j_guided_state_distance_heatmap.png`: component-level
  native-to-Reference-A state-distance summary by method.
- `figures/stage_j_guided_warm_start_study.png`: Reference A AC-OPF warm-start
  runtime comparison for cold, DC, GridSFM, and exact/reference-state starts.

## Runtime Evidence

Summed candidate-evaluation time, excluding the exact AC audit solves:

```text
Guided-DC:       1,650.08 s across 30,000 evaluations
Guided-GridSFM:  9,740.38 s across 30,000 evaluations
TH-GridSFM:        191.25 s across    600 evaluations
```

See `core_results/candidate_runtime_summary.csv` for setting-level detail.

## Guided-DC Versus Guided-GridSFM Findings

The complete run gives a direct guided-method comparison over the same
`15 = 3 scenarios x 5 lambda_R` settings. The key distinction is:

```text
Guided-DC:
  finalist topology/load-service decisions selected by DC recourse

Guided-GridSFM:
  finalist topology/load-service decisions selected by frozen GridSFM
  OPF-surrogate recourse plus PAC search merit
```

Both finalist families are then audited by the same exact AC references and
warm-start study.

### Native Wildfire-Service Tradeoff

`core_results/method_finalists_all.csv` shows that Guided-DC selected lower
common native wildfire/service objective finalists on average:

| Method | Rows | Avg `J_trade` | Avg `R_norm` | Avg `L_shed_total` | Avg `PAC_total` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Guided-DC | 15 | 0.260473 | 0.603736 | 0.002963 | 0.00000000 |
| Guided-GridSFM | 15 | 0.333630 | 0.754192 | 0.007015 | 0.00011008 |

This supports the current Stage J finding that, under the approved `q=5`
bounded-alpha approximation and shared guided topology budget, Guided-DC found
better native `J_trade` frontier points than Guided-GridSFM on average.

### Reference A: Fixed-Topology/Fixed-Alpha Economic AC-OPF

Reference A fixes each finalist's selected topology `z*` and load-service
command `alpha*`, then solves exact economic AC-OPF. It therefore checks the
AC outcome of the selected decision without allowing extra load recovery.

| Method | Rows | Avg Reference A solver seconds | Avg Reference A wall seconds | Avg Reference A objective | Avg AC `R_norm` | Avg AC `L_shed` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Guided-DC | 15 | 6.6661 | 12.0682 | 460634.43 | 0.674001 | 0.002963 |
| Guided-GridSFM | 15 | 6.7369 | 12.1179 | 453440.84 | 0.680575 | 0.007015 |

The Reference A load-shedding values exactly match the native finalist
`L_shed_total`, as expected, because Reference A holds `alpha*` fixed. The
exact-AC risk discrepancy and state-distance diagnostics are:

| Method | Avg `|native R_norm - AC R_norm|` | Avg `|native L_shed - AC L_shed|` | Avg native-state distance to Reference A |
| --- | ---: | ---: | ---: |
| Guided-DC | 0.070265 | 0.000000 | 0.022693 |
| Guided-GridSFM | 0.074987 | 0.000000 | 0.207259 |

Guided-DC finalist instances are slightly faster for Reference A AC-OPF and
closer to Reference A in the aggregated normalized state-distance metric. The
important counterpoint is that Guided-GridSFM finalists have lower average
Reference A economic objective, so the comparison separates wildfire-service
quality and downstream economic dispatch cost rather than collapsing them into
one score.

### Reference B: Fixed-Topology Maximum Load Delivery

Reference B fixes only the finalist topology and asks what exact AC operation
can recover when load delivery is optimized. B1 maximizes delivered load, and
B2 applies the economic tie-break at the recovered service level.

| Method | Rows | Avg Reference B MLD `L_shed` | Max Reference B MLD `L_shed` | Avg served-load fraction | Min served-load fraction | Avg Reference B MLD `R_norm` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Guided-DC | 15 | 0.000000 | 0.000000 | 1.000000 | 1.000000 | 0.584971 |
| Guided-GridSFM | 15 | 0.001050 | 0.005248 | 0.998950 | 0.994752 | 0.624316 |

This is the clearest AC-recovery result: all Guided-DC selected topologies
support full load delivery under exact AC Reference B, while Guided-GridSFM
topologies recover nearly all load but retain small residual MLD shedding in
some cases.

### Warm-Start Runtime Study

For each finalist Reference A instance, the warm-start study re-solves the same
fixed-`z*`, fixed-`alpha*` exact AC-OPF from cold, DC, GridSFM, and exact/
reference-state starts. The figure reports raw solver seconds and seconds
saved relative to the respective cold start for the same finalist instance.

| Finalist family | Warm start | Avg solver seconds | Avg solver seconds saved vs respective cold |
| --- | --- | ---: | ---: |
| Guided-DC | DC partial | 6.4526 | 0.2198 |
| Guided-DC | GridSFM partial | 6.4794 | 0.1930 |
| Guided-DC | Exact/reference-state | 6.4665 | 0.2059 |
| Guided-GridSFM | DC partial | 6.4756 | 0.2468 |
| Guided-GridSFM | GridSFM partial | 6.4771 | 0.2453 |
| Guided-GridSFM | Exact/reference-state | 6.5145 | 0.2079 |

Warm starts provide modest solver-time reductions in both finalist families.
The relative savings are slightly larger for Guided-GridSFM finalist instances,
but the raw Reference A solver times remain slightly lower for Guided-DC
finalists. Therefore the cleaner runtime finding is that Guided-DC selected
finalists are somewhat easier exact-AC instances in raw solve time, while both
DC and GridSFM starts offer small warm-start benefits.

### Overall Evidence-Based Interpretation

For this Stage J complete run, Guided-DC is the stronger method on the common
wildfire-service decision surface: it has lower average native `J_trade`, lower
average native `R_norm`, lower average native `L_shed_total`, slightly faster
Reference A exact-AC solve times, lower native-to-Reference-A state distance,
and full Reference B maximum-load-delivery recovery for all guided finalists.

Guided-GridSFM remains scientifically useful but is not the winning guided
selector in this particular `q=5`, `K<=2`, 100-topology-budget study. Its
advantages in the retained evidence are lower average Reference A economic
objective and an AC-OPF surrogate state channel that can be audited directly.
The result should therefore be framed as a nuanced comparison, not as a blanket
failure of GridSFM: the frozen GridSFM OPF-surrogate does not yet outperform
the DC guided selector on the wildfire-service topology/alpha objective, but it
provides a distinct AC-state prediction and economic-dispatch lens for the same
wildfire decisions.

## Interpretation Boundary

The exported `J_trade` values compare the common wildfire/service objective.
GridSFM `J_total` additionally includes the frozen PAC merit and should not be
used as a common DC-versus-GridSFM objective. A successful GridSFM call marked
`model_output_penalized` is not an AC-feasibility certificate; Reference A/B
and the state-fidelity tables are the downstream exact-AC evidence.

The alpha search is a selected-five-load computational approximation. Do not
describe this run as an exact joint optimization over all GOC-500 load-bus
alpha values.
