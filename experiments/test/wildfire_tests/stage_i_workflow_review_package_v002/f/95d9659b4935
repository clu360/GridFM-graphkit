# Stage H Stage I-a Proxy-Inner Lambda Findings

This note documents the Stage I-a summary figures generated from the
`proxy_inner_lambda_sweep/r2` run. It is intended as a clean handoff for
presentation planning and for the next implementation pass.

## Result Location

Main Stage H run used for the coupled baseline:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/main_results/r11
```

Proxy-inner lambda sweep used for the expanded Stage I-a search:

```text
experiments/test/wildfire_tests/results/leq/stage_h/
  DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2
```

Generated summary figures are stored under each scenario:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S*/
  stage_i_a_main_vs_best_proxy/
    stage_i_a_pareto_main_vs_best_proxy.png
    stage_i_a_traditional_objective_main_vs_best_proxy.png
```

The figure manifest is:

```text
proxy_inner_lambda_sweep/r2/tables/stage_i_a_summary_figure_manifest.csv
```

The best proxy setting table is:

```text
proxy_inner_lambda_sweep/r2/tables/stage_i_a_best_proxy_setting_for_summary_figures.csv
```

## Methodological Purpose

The main Stage H results use a coupled lambda convention:

```text
lambda_R_proxy = lambda_R
```

This means the topology proposal objective and the fixed-topology continuous
recourse objective use the same wildfire-risk weight.

The proxy-inner lambda sweep tests whether a larger search design improves the
Stage I-a tradeoff front. It separates:

```text
lambda_R_proxy in {0, 0.2, 0.5, 0.8, 1}
lambda_R       in {0, 0.2, 0.5, 0.8, 1}
```

For each scenario and proxy value, Stage I-a generates guided `K <= 2`
topologies, then evaluates those topologies under every inner `lambda_R`.
The comparison answers:

```text
Does changing the topology-search risk weight expose better DC recourse
solutions than the coupled lambda_R_proxy = lambda_R convention?
```

This study is rho-invariant for Stage I-a DC and was run with:

```text
rho_phys = 0
```

## Equations Used In The Figures

The traditional lambda objective used for fair comparison is:

```text
J(lambda_R) =
lambda_R R_norm + (1 - lambda_R) L_shed
```

For Stage I-a DC:

```text
L_shed =
sum_{i in D} Pd_i^base (1 - s_i)
/
sum_{i in D} Pd_i^base
```

The normalized risk is:

```text
R_norm =
R_raw,DC / R_baseline,shared
```

with:

```text
R_raw,DC =
sum_{l in L_risk} p_l^env (f_l / F_l^max)^2
```

and the shared denominator:

```text
R_baseline,shared =
sum_{l in L_risk} p_l^env (baseline_loading_l^stored)^2
```

The Pareto figures use:

```text
x = L_shed
y = R_norm
```

Both axes are minimized. A point `a` dominates point `b` when:

```text
L_shed(a) <= L_shed(b)
R_norm(a) <= R_norm(b)
```

and at least one inequality is strict. The nondominated set is the Pareto front.

## Best Proxy Setting Selection

For the new summary figures, one proxy setting is selected per scenario by:

```text
lambda_R_proxy^*(s) =
argmin_p mean_{lambda_R in {0.2, 0.5, 0.8, 1}} J_s(p, lambda_R)
```

Ties are broken by lower mean `R_norm`, then lower mean `L_shed`.
`lambda_R = 0` is excluded from this selection rule because the objective
becomes pure load shedding and can mask whether the proxy dimension improves
wildfire-risk/load-shedding tradeoffs.

Selected Stage I-a proxy settings:

```text
S1: lambda_R_proxy = 1.0
S2: lambda_R_proxy = 0.5
S3: lambda_R_proxy = 0.5
S4: lambda_R_proxy = 0.0
S5: lambda_R_proxy = 1.0
```

Corresponding mean nonzero-objective values:

```text
S1: mean J = 0.127657, mean R_norm = 0.206646, mean L_shed = 0.105722
S2: mean J = 0.151548, mean R_norm = 0.239961, mean L_shed = 0.113631
S3: mean J = 0.218266, mean R_norm = 0.342016, mean L_shed = 0.122228
S4: mean J = 0.159708, mean R_norm = 0.241100, mean L_shed = 0.116734
S5: mean J = 0.056211, mean R_norm = 0.080604, mean L_shed = 0.102037
```

The common operational diagnostic for these selected Stage I-a DC settings is
zero in the summary table, which is expected for a physically consistent DC
recourse formulation.

## Generated Summary Images

For each scenario, two summary images were generated.

### S1

Pareto comparison:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S1/
  stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png
```

Traditional objective convergence:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S1/
  stage_i_a_main_vs_best_proxy/stage_i_a_traditional_objective_main_vs_best_proxy.png
```

Reading: the best proxy setting is `lambda_R_proxy = 1.0`. The proxy sweep
mainly improves or matches the coupled result in risk-sensitive regions. This
supports the idea that risk-heavy topology proposal can reveal useful
topologies even when the inner recourse lambda is not exactly 1.

### S2

Pareto comparison:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S2/
  stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png
```

Traditional objective convergence:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S2/
  stage_i_a_main_vs_best_proxy/stage_i_a_traditional_objective_main_vs_best_proxy.png
```

Reading: the best proxy setting is `lambda_R_proxy = 0.5`. The improvement is
more moderate than S1 or S5. The main value of the expanded search here is not
a dramatic frontier shift, but a more balanced topology pool for nonzero
lambda values.

### S3

Pareto comparison:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S3/
  stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png
```

Traditional objective convergence:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S3/
  stage_i_a_main_vs_best_proxy/stage_i_a_traditional_objective_main_vs_best_proxy.png
```

Reading: the best mean proxy setting is `lambda_R_proxy = 0.5`. The expanded
search adds frontier diversity, but the best objective is not uniformly better
at every inner lambda. This is an important scenario for presentation because
it shows the method improves the search space, not necessarily every scalarized
objective slice.

### S4

Pareto comparison:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S4/
  stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png
```

Traditional objective convergence:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S4/
  stage_i_a_main_vs_best_proxy/stage_i_a_traditional_objective_main_vs_best_proxy.png
```

Reading: the best proxy setting is `lambda_R_proxy = 0.0`, which is
methodologically interesting. In S4, a load-delivery-oriented topology proposal
still exposes topologies that become competitive after inner recourse applies
nonzero wildfire-risk weights. This supports keeping proxy and inner lambda
separable rather than assuming the coupled convention is always best.

### S5

Pareto comparison:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S5/
  stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png
```

Traditional objective convergence:

```text
proxy_inner_lambda_sweep/r2/plots/by_scenario/S5/
  stage_i_a_main_vs_best_proxy/stage_i_a_traditional_objective_main_vs_best_proxy.png
```

Reading: the best proxy setting is `lambda_R_proxy = 1.0`. This is one of the
clearest cases where the proxy-inner sweep improves the risk-sensitive side of
the Stage I-a frontier while retaining comparable load-shedding behavior.

## Main Findings

1. Decoupling `lambda_R_proxy` and inner `lambda_R` is useful for Stage I-a.
   It expands the topology pool and can improve the discovered Pareto front
   without changing the fixed-topology DC recourse formulation.

2. The best proxy setting is scenario-dependent. S1 and S5 prefer
   risk-heavy topology search, S2 and S3 prefer a balanced proxy, and S4
   prefers a load-delivery proxy. This argues against locking
   `lambda_R_proxy = lambda_R` as the only experiment design.

3. The benefit is clearest on Pareto-front diversity and selected scalarized
   objectives, not on every single lambda slice. This should be framed as
   "expanded search finds additional useful tradeoff candidates", not as
   "proxy-inner always dominates coupled search".

4. Stage I-a remains physically stable under the DC diagnostic. The selected
   summary settings have approximately zero common operational diagnostic,
   which distinguishes the DC comparison from the current Stage E GridFM
   behavior where physically unrealistic predicted loading can dominate
   `R_norm` and confound interpretation.

5. The comparison is most useful as evidence about search design. It shows
   that topology proposal and continuous recourse can have different best
   objective weights, especially when the topology decision is sparse
   (`K <= 2`) and the same topology can support multiple recourse priorities.

## Presentation Guidance

Use the S1, S4, and S5 summary figures as the strongest presentation examples.

S1 and S5 show the intuitive case:

```text
risk-heavy proxy search -> better risk-sensitive frontier behavior
```

S4 shows the non-intuitive case:

```text
load-delivery proxy search -> useful topologies for later risk-aware recourse
```

This contrast is useful because it motivates the two-level framing:

```text
outer proxy lambda controls topology discovery
inner lambda controls operational recourse evaluation
```

The traditional objective convergence figures should be used to explain how
the best-so-far objective evolves across topology iterations. The Pareto
figures should be used to explain the broader set of nondominated tradeoffs
that are hidden when only one final scalarized solution is shown.

## Limitations And Next Checks

The proxy-inner sweep did not generate AC projection distances. Therefore, we
cannot yet claim that the expanded proxy-inner search improves AC projection
distance or AC-feasible realizability. The correct next check is:

```text
Run AC projection on selected proxy-inner finalists and compare against
main_results/r11 Stage I-a finalists.
```

The current summary selects a single best proxy setting per scenario using
mean nonzero `J`. That is a useful presentation reduction, but it is not the
only defensible selection rule. Other views may select the best proxy by:

```text
minimum R_norm
minimum L_shed
maximum target recall
maximum precision
minimum AC projection distance, once available
```

Stage E K2 was intentionally excluded from these new summary images. The goal
of these figures is to isolate the Stage I-a DC behavior without the current
GridFM physical-inconsistency scale dominating the axes.

## Pickup Checklist

The work represented by this note is complete for tonight:

```text
5 scenarios x 2 summary images = 10 Stage I-a comparison images
```

The images compare the coupled main Stage I-a result against the best
proxy-inner Stage I-a setting per scenario. The best proxy settings are:

```text
S1: 1.0
S2: 0.5
S3: 0.5
S4: 0.0
S5: 1.0
```

For the next session, start from:

```text
experiments/test/wildfire_tests/stage_i_dc_comparison/
  STAGE_H_DC_COMPARISON_PROGRESS.md
  STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md
```

Then inspect:

```text
proxy_inner_lambda_sweep/r2/tables/stage_i_a_summary_figure_manifest.csv
proxy_inner_lambda_sweep/r2/tables/stage_i_a_best_proxy_setting_for_summary_figures.csv
```

The most natural next analysis is:

```text
AC projection on selected proxy-inner Stage I-a finalists.
```

That would let us test whether the improved DC tradeoff-front candidates also
move closer to AC-feasible operation, rather than only improving the DC
objective/frontier.

## Reproduction

Generate or refresh the summary figures with:

```text
python experiments/test/wildfire_tests/stage_i_dc_comparison/run_proxy_inner_stage_ia_summary_figures.py
```

The script reads:

```text
main_results/r11/tables/all_evaluated_stage_h_points.csv
proxy_inner_lambda_sweep/r2/tables/best_by_scenario_proxy_inner_stage.csv
proxy_inner_lambda_sweep/r2/tables/all_evaluated_proxy_inner_points.csv
```

and writes the per-scenario summary images and manifest listed above.
