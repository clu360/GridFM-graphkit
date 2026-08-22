# 7-31 Meeting Slides: Compact Result Picklist

Purpose: provide a short, professor-facing set of slides that directly matches the meeting notes in `7-31 Meeting Prep.docx`. This is meant to help zoom out, explain the research progression, show the most useful evidence, and create a productive framing discussion.

Use this as the main meeting flow. The embedded figures are actual result artifacts already present in the repository/package.

## Slide 1 - Project Evolution In One View

**Use:** a compact hand-made table, not a plot.

| Stage | What Changed | Why It Mattered |
|---|---|---|
| A-B | First GridFM wildfire/load objective, then automatic line scoring/groups | Proved the workflow could run and gave a repeatable way to define risky regions |
| C-D | Added de-energization, then enumerated `K <= 2` shutoffs | Showed topology decisions drive much larger risk reduction than continuous-only changes |
| E | Added Gurobi proxy topology search + GridFM evaluation | Moved from exhaustive enumeration toward scalable guided search |
| F | Built five decision-quality scenarios | Created interpretable tests for whether decisions match scenario motivation |
| G | Fixed baseline loading, islanding, load-service, and PAC/model-consistency accounting | Revealed GridFM physical-realism and command-faithfulness limitations |
| H | Added Rhodes-inspired TH/AH heuristic baselines | Connected our setup to prior wildfire shutoff methodology |
| I | Added Stage I-a DC guided recourse, Stage I-b DC MIQP, MLD, proxy-inner sweep, AC projection | Created a transparent benchmark for GridFM and clarified current framing |

**Message:**  
The project evolved from “can GridFM support wildfire/load optimization?” into a comparative study of GridFM-guided topology control versus heuristic and DC optimization methods.

**Pull from:**  
`STAGE_A_I_DEVELOPMENT_MAP.md`, `FINAL_ALIGNMENT_REPORT.md`, and the meeting prep document.

---

## Slide 2 - Why Stage G Changed The Framing

**Best figure:** Stage E K2 GridFM load-shedding discrepancy.

![Stage E K2 GridFM load-shedding discrepancy](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/summary/load_shed_cmd_gridfm_hybrid_stage_e_k2_only.png>)

**What it does:**  
Shows that commanded load shedding, raw GridFM-predicted load shedding, effective GridFM load shedding after islanding/clipping, and hybrid load shedding are not telling the same story.

**Message:**  
This is the cleanest evidence that our GridFM formulation exposed a model bottleneck: the model can receive commanded decision values but still predict grid states whose load-service interpretation changes substantially after validity checks.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/summary/load_shed_cmd_gridfm_hybrid_stage_e_k2_only.png`

---

## Slide 3 - Main Result Across Methods

**Best figure:** best `J_true` by method across lambda and rho.

![Best J_true across methods](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/summary/best_j_true_by_method_lambda_rho.png>)

**What it does:**  
Compares Stage E K2 GridFM, Stage I-a DC guided K2, Stage I-b DC MIQP K2, TH top-1/top-2, and AH K2 using the common `J_true` view.

**Message:**  
The main comparison makes the GridFM limitation visible: Stage E K2 becomes much worse when GridFM physics/model artifacts dominate. The DC and heuristic methods remain much more compact on the same `J_true` scale.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/summary/best_j_true_by_method_lambda_rho.png`

---

## Slide 4 - DC/Heuristic Comparison Without GridFM Scale Effects

**Best figure:** best `J_true`, excluding Stage E GridFM.

![Best J_true excluding Stage E GridFM](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/summary/best_j_true_by_method_lambda_rho_no_stage_e.png>)

**What it does:**  
Removes the Stage E GridFM curve so the advisor can actually see Stage I-a, Stage I-b, TH, and AH relative behavior.

**Message:**  
This is the clearest high-level comparison once GridFM is not visually dominating the axis. Stage I-b is usually strongest or very close on objective, Stage I-a is close, and TH/AH are useful sparse literature-inspired reference points.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/summary/best_j_true_by_method_lambda_rho_no_stage_e.png`

---

## Slide 5 - Pareto View: Why GridFM Needs Careful Qualification

**Best figure:** S4 Pareto scatter with all methods.

![S4 Pareto scatter with all methods](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S4/pareto_frontier_scatter.png>)

**What it does:**  
Shows all evaluated risk/load solutions for S4 under `rho=0`, including Stage E K2 GridFM, DC methods, and heuristics.

**Message:**  
This is the most intuitive visual for explaining why the current safest framing is comparative feasibility/decision quality rather than “GridFM solves OPS.” The GridFM points occupy a very different scale, while DC/heuristic solutions remain in a physically grounded region.

**Exact copied path:**  
`experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S4/pareto_frontier_scatter.png`

---

## Slide 6 - Stage I-a Versus Stage I-b: Compact Formulation Tradeoff

**Best figure:** S4, `rho=0`, `lambda_R=1`, Pareto scatter excluding Stage E GridFM.

![S4 no-Stage-E Pareto scatter at lambda_R=1](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S4/pareto_frontier_scatter_no_stage_e_by_lambda/pareto_frontier_scatter_no_stage_e_lambda_R_1.png>)

**What it does:**  
Zooms into the DC/heuristic methods for the S4 islanding scenario at pure wildfire-risk weighting.

**Message:**  
This is the best slide for the Stage I-a versus Stage I-b discussion. Stage I-b MIQP can find a lower-load-shed solution at similar or better risk, because it solves topology and continuous DC operation jointly. Stage I-a is still valuable because its guided outer-inner structure is closer to the GridFM methodology and often matches scenario target logic more directly.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/per_rho/rho0/S4/pareto_frontier_scatter_no_stage_e_by_lambda/pareto_frontier_scatter_no_stage_e_lambda_R_1.png`

---

## Slide 7 - Decision Quality: Expected Versus Selected Shutoffs

**Best figure:** S4 expected-versus-selected shutoffs.

![S4 expected versus selected shutoffs](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S4/expected_vs_selected_shutoff_lines.png>)

**What it does:**  
Shows selected lines, target hits, misses, non-targets, recall, precision, and islanding flag for each method/lambda setting.

**Message:**  
This connects results back to the motivation for the five scenarios. For S4, the target is line `77`, and the islanding pair should be avoided. Stage I-a hits `77` at balanced/risk-heavy settings, while Stage I-b avoids islanding but prefers non-target alternatives under the compact DC objective. This cleanly separates scalar objective quality from target-alignment behavior.

**Exact copied path:**  
`experiments/test/wildfire_tests/stage_i_workflow_review_package/05_results/primary_complete_run/plots/per_rho/rho0/S4/expected_vs_selected_shutoff_lines.png`

---

## Slide 8 - AC Projection: Distance To Feasible Operation

**Best figure:** AC projection distance for selected finalists.

![AC projection distance](<../../../stage_i_workflow_review_package/05_results/primary_complete_run/plots/projection_distance_by_lambda.png>)

**What it does:**  
Shows how far selected finalists are from a fixed-topology AC-feasible point when projection succeeds.

**Message:**  
This is the bridge from DC/GridFM outputs toward operational feasibility. Projection distance is not an AC topology optimizer and not just a residual; it asks how much the selected operating point must move to become AC-feasible under the same topology.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/main_results/r11/plots/projection_distance_by_lambda.png`

---

## Slide 9 - Literature Alignment: MLD Substudy

**Best figure:** MLD S4 Pareto scatter.

![MLD S4 Pareto scatter](<../../../stage_i_workflow_review_package/05_results/companion_runs/MLD_r5/plots/per_rho/rho0/S4/pareto_frontier_scatter.png>)

**What it does:**  
Shows the special MLD-style comparison, where the inner problem is load-delivery focused rather than the main wildfire/load lambda sweep.

**Message:**  
This slide connects the work back to Rhodes-style MLD/heuristic literature. It should be framed as a literature-alignment sub-study, not the main generalized formulation.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/MLD/r5/plots/per_rho/rho0/S4/pareto_frontier_scatter.png`

---

## Slide 10 - Methodological Next Step: Proxy-Inner Lambda Sweep

**Best figure:** S5 Stage I-a main coupled versus best proxy setting.

![S5 Stage I-a proxy-inner comparison](<../../../stage_i_workflow_review_package/05_results/companion_runs/proxy_inner_lambda_sweep_r2/plots/by_scenario/S5/stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png>)

**What it does:**  
Compares the original coupled convention `lambda_R_proxy = lambda_R` against the best decoupled Stage I-a proxy setting.

**Message:**  
This shows why separating topology-search lambda from inner recourse lambda is methodologically useful. The best proxy setting is scenario-dependent: `S1=1.0`, `S2=0.5`, `S3=0.5`, `S4=0.0`, `S5=1.0`. This supports keeping the outer-inner structure as a flexible research direction, even though Stage I-b is faster as a compact DC MIQP.

**Exact live path:**  
`experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_i/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2/plots/by_scenario/S5/stage_i_a_main_vs_best_proxy/stage_i_a_pareto_main_vs_best_proxy.png`

---

## Short Meeting Version

If time is tight, show only these:

1. Slide 1: Stage A-I roadmap.
2. Slide 2: GridFM load-shedding discrepancy.
3. Slide 4: DC/heuristic comparison without Stage E GridFM.
4. Slide 6: S4 no-Stage-E Pareto at `lambda_R=1`.
5. Slide 8: AC projection distance.
6. Slide 10: proxy-inner lambda sweep.

## Suggested Closing Discussion

Use these questions to guide the professor conversation:

1. Should the near-term framing be comparative feasibility/decision quality rather than a GridFM-centered OPS solver claim?
2. Should the next technical step prioritize better surrogate models such as newer GridFM/GridSFM, or move toward DC/AC/SC-OPS extensions?
3. Should we complete AC projection coverage before making stronger operational claims?
4. Is the paper direction an insights/limitations paper, a comparative methodology paper, or a longer-term robust/security-constrained formulation?
