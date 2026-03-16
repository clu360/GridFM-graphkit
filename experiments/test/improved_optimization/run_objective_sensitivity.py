"""
Run objective sensitivity studies for the improved optimization workflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test import NeuralSolverWrapper, load_gnn_model, load_gps_model, load_single_test_scenario
from experiments.test.improved_optimization.config import build_default_experiment_config
from experiments.test.improved_optimization.io_utils import create_run_directory, ensure_results_structure
from experiments.test.improved_optimization.optimization_problem import ImprovedDispatchOptimizationProblem
from experiments.test.improved_optimization.plots import plot_sensitivity_curve
from experiments.test.improved_optimization.reporting import save_dataframe
from experiments.test.improved_optimization.selection import (
    TargetedDispatchDecisionSpec,
    build_selected_decision_spec,
    compute_generator_relief_scores,
    compute_load_relief_scores,
)


def _fresh_config(model_name: str):
    return build_default_experiment_config(model_name=model_name, device="cpu")


def _build_problem(config):
    context = load_single_test_scenario(
        scenario_idx=config.scenario_idx,
        scenario_id=config.scenario_id,
        config_name=config.config_name,
    )
    if config.model_name == "gps":
        model, _ = load_gps_model(context.config_dict, repo_root=context.repo_root, device=config.device)
    else:
        model = load_gnn_model(context.args, repo_root=context.repo_root, device=config.device)
    scenario = context.scenario
    screening_solver = NeuralSolverWrapper(
        model,
        config.model_name,
        scenario,
        TargetedDispatchDecisionSpec(scenario, [], [], max_shed_fraction=config.selection.max_shed_fraction),
        device=config.device,
    )
    generator_scores, _ = compute_generator_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    load_scores, _ = compute_load_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    decision_spec = build_selected_decision_spec(scenario, generator_scores, load_scores, config.selection)
    solver = NeuralSolverWrapper(model, config.model_name, scenario, decision_spec, device=config.device)
    return scenario, decision_spec, ImprovedDispatchOptimizationProblem(scenario, decision_spec, solver, config)


def main(model_name: str = "gnn") -> None:
    model_name = model_name.lower()
    config = _fresh_config(model_name)
    paths = ensure_results_structure(config.results_root)
    run_dir = create_run_directory(paths["sensitivity"], f"{model_name}_sensitivity")

    rows = []
    base_config = _fresh_config(model_name)
    for value in base_config.sensitivity.lambda_s_values:
        config = _fresh_config(model_name)
        config.objective.lambda_s = float(value)
        _, _, problem = _build_problem(config)
        result = problem.optimize()
        rows.append({"metric_group": "lambda_s", "x_value": value, "objective": result["final"]["objective"], "wildfire_cost": result["final"]["wildfire_cost"], "total_shed_mw": result["final"]["total_shed_mw"]})
    for value in base_config.sensitivity.wildfire_threshold_values:
        config = _fresh_config(model_name)
        config.wildfire.wildfire_threshold = float(value)
        _, _, problem = _build_problem(config)
        result = problem.optimize()
        rows.append({"metric_group": "wildfire_threshold", "x_value": value, "objective": result["final"]["objective"], "wildfire_cost": result["final"]["wildfire_cost"], "total_shed_mw": result["final"]["total_shed_mw"]})
    for value in base_config.sensitivity.softplus_alpha_values:
        config = _fresh_config(model_name)
        config.wildfire.softplus_alpha = float(value)
        _, _, problem = _build_problem(config)
        result = problem.optimize()
        rows.append({"metric_group": "softplus_alpha", "x_value": value, "objective": result["final"]["objective"], "wildfire_cost": result["final"]["wildfire_cost"], "total_shed_mw": result["final"]["total_shed_mw"]})

    config = _fresh_config(model_name)
    scenario, decision_spec, problem = _build_problem(config)
    base = decision_spec.u_base.copy()
    sweep_rows = []
    for idx in range(min(2, decision_spec.n_total)):
        span = decision_spec.u_max[idx] - decision_spec.u_min[idx]
        offsets = np.linspace(
            -config.sensitivity.local_sweep_scale,
            config.sensitivity.local_sweep_scale,
            config.sensitivity.local_sweep_points,
        ) * span
        for offset in offsets:
            u = base.copy()
            u[idx] = np.clip(u[idx] + offset, decision_spec.u_min[idx], decision_spec.u_max[idx])
            breakdown = problem.compute_objective_breakdown(u)
            sweep_rows.append(
                {
                    "metric_group": f"local_control_{idx}",
                    "x_value": offset,
                    "objective": breakdown["objective"],
                    "wildfire_cost": breakdown["wildfire_cost"],
                    "total_shed_mw": breakdown["total_shed_mw"],
                }
            )

    df = pd.DataFrame(rows + sweep_rows)
    csv_path = save_dataframe(df, run_dir / "sensitivity_results.csv")
    plot_sensitivity_curve(df[df["metric_group"].isin(["lambda_s", "wildfire_threshold", "softplus_alpha"])], "x_value", "objective", run_dir / "sensitivity_objective.png", "Objective sensitivity")
    plot_sensitivity_curve(df[df["metric_group"].str.startswith("local_control_")], "x_value", "objective", run_dir / "local_landscape.png", "Local objective landscape")
    print(f"[OK] {model_name.upper()} sensitivity analysis complete. Results saved to {csv_path}")


if __name__ == "__main__":
    selected_model = sys.argv[1] if len(sys.argv) > 1 else "gnn"
    main(model_name=selected_model)
