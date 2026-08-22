# Method Comparison Matrix

| Method | Topology mechanism | Continuous recourse | Physics model | Primary feasibility representation | Final evaluation | Expected topology budget | Implementation evidence |
|---|---|---|---|---|---|---|---|
| Stage E K2 GridFM | Guided proxy/no-good topology search | GridFM-linked continuous recourse | GridFM-predicted AC-style state with soft diagnostics | GridFM PAC and common diagnostics | Shared metrics and AC projection | K<=2 | `run_stage_h_dc_comparison.py`, `main_results/r11` |
| Stage I-a | Guided topology search | Fixed-topology DC recourse | DC | Hard DC constraints plus common diagnostic | Shared metrics and AC projection | K<=2 | `dc_formulation.py`, `stage_i_a_topology_pool.csv` |
| Stage I-b | Joint mixed-integer optimization | Joint DC dispatch/load service | DC MIQP | Hard DC constraints, solver diagnostics | Shared metrics and AC projection | K<=2 | `dc_formulation.py`, `stage_i_b_solution_pool.csv` |
| TH | Heuristic ranking | Budgeted heuristic evaluation/recourse | Depends on evaluator path | Heuristic topology plus evaluator | Shared metrics | K<=2 | `heuristic_topology_pool.csv`, `expected_vs_selected_by_rho.csv` |
| AH | Connected-area heuristic | Budgeted heuristic evaluation/recourse | Depends on evaluator path | Heuristic topology plus evaluator | Shared metrics | K<=2 | `ah_k2_audit.csv`, `heuristic_topology_pool.csv` |
