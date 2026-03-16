"""
Run a Pareto sweep over lambda_s / lambda_w tradeoffs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test import NeuralSolverWrapper, load_gnn_model, load_gps_model, load_single_test_scenario
from experiments.test.improved_optimization.config import build_default_experiment_config
from experiments.test.improved_optimization.io_utils import create_run_directory, ensure_results_structure
from experiments.test.improved_optimization.optimization_problem import ImprovedDispatchOptimizationProblem
from experiments.test.improved_optimization.plots import plot_pareto_frontier
from experiments.test.improved_optimization.reporting import save_dataframe
from experiments.test.improved_optimization.selection import (
    TargetedDispatchDecisionSpec,
    build_selected_decision_spec,
    compute_generator_relief_scores,
    compute_load_relief_scores,
)


def main(model_name: str = "gnn") -> None:
    model_name = model_name.lower()
    config = build_default_experiment_config(model_name=model_name, device="cpu")
    paths = ensure_results_structure(config.results_root)
    run_dir = create_run_directory(paths["pareto"], f"{model_name}_pareto")

    context = load_single_test_scenario(
        scenario_idx=config.scenario_idx,
        scenario_id=config.scenario_id,
        config_name=config.config_name,
    )
    if model_name == "gps":
        model, _ = load_gps_model(context.config_dict, repo_root=context.repo_root, device=config.device)
    else:
        model = load_gnn_model(context.args, repo_root=context.repo_root, device=config.device)
    scenario = context.scenario
    screening_solver = NeuralSolverWrapper(
        model,
        model_name,
        scenario,
        TargetedDispatchDecisionSpec(scenario, [], [], max_shed_fraction=config.selection.max_shed_fraction),
        device=config.device,
    )
    generator_scores, _ = compute_generator_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    load_scores, _ = compute_load_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    decision_spec = build_selected_decision_spec(scenario, generator_scores, load_scores, config.selection)
    solver = NeuralSolverWrapper(model, model_name, scenario, decision_spec, device=config.device)

    rows = []
    for lambda_s in config.pareto.lambda_s_values[: config.pareto.max_runs]:
        config.objective.lambda_s = float(lambda_s)
        config.objective.lambda_w = float(config.pareto.lambda_w)
        config.objective.lambda_g = float(config.pareto.lambda_g)
        problem = ImprovedDispatchOptimizationProblem(scenario, decision_spec, solver, config)
        result = problem.optimize()
        rows.append(
            {
                "lambda_s": lambda_s,
                "lambda_w": config.objective.lambda_w,
                "lambda_g": config.objective.lambda_g,
                "success": result["success"],
                "n_iter": result["n_iter"],
                "objective": result["final"]["objective"],
                "wildfire_cost": result["final"]["wildfire_cost"],
                "load_shedding_cost": result["final"]["load_shedding_cost"],
                "generator_deviation_cost": result["final"]["generator_deviation_cost"],
                "total_shed_mw": result["final"]["total_shed_mw"],
                "max_risky_line_loading": result["final"]["max_risky_line_loading"],
                "selected_generators": ",".join(map(str, decision_spec.selected_generator_buses.tolist())),
                "selected_loads": ",".join(map(str, decision_spec.selected_load_buses.tolist())),
            }
        )

    df = pd.DataFrame(rows)
    csv_path = save_dataframe(df, run_dir / "pareto_results.csv")
    plot_pareto_frontier(df, "total_shed_mw", "wildfire_cost", run_dir / "pareto_wildfire_vs_shed.png", "Pareto frontier: wildfire vs shed MW")
    plot_pareto_frontier(df, "wildfire_cost", "objective", run_dir / "pareto_objective_vs_wildfire.png", "Pareto frontier: objective vs wildfire")
    plot_pareto_frontier(df, "total_shed_mw", "max_risky_line_loading", run_dir / "pareto_loading_vs_shed.png", "Pareto frontier: max risky loading vs shed MW")
    print(f"[OK] {model_name.upper()} Pareto sweep complete. Results saved to {csv_path}")


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else "gnn"
    main(model_name=selected_model)
