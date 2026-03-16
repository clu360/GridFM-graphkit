"""
Run a single targeted wildfire-aware surrogate optimization experiment.
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
from experiments.test.improved_optimization.plots import plot_objective_history, plot_risky_line_comparison
from experiments.test.improved_optimization.reporting import (
    append_run_summary_block,
    save_dataframe,
    save_objective_history_csv,
    save_optimization_summary_json,
    write_markdown_summary,
)
from experiments.test.improved_optimization.selection import (
    TargetedDispatchDecisionSpec,
    build_selected_decision_spec,
    compute_generator_relief_scores,
    compute_load_relief_scores,
)
from experiments.test.improved_optimization.validation import validate_run
from experiments.test.improved_optimization.wildfire_metrics import summarize_top_risky_lines


def _empty_spec(scenario, selection_cfg) -> TargetedDispatchDecisionSpec:
    return TargetedDispatchDecisionSpec(
        scenario,
        selected_generator_buses=[],
        selected_load_buses=[],
        max_shed_fraction=selection_cfg.max_shed_fraction,
    )


def run_single_experiment(model_name: str = "gnn") -> dict:
    config = build_default_experiment_config(model_name=model_name, device="cpu")
    paths = ensure_results_structure(config.results_root)
    run_dir = create_run_directory(paths["improved_runs"], f"{model_name}_improved")

    context = load_single_test_scenario(
        scenario_idx=config.scenario_idx,
        scenario_id=config.scenario_id,
        config_name=config.config_name,
    )
    scenario = context.scenario

    if model_name.lower() == "gps":
        model, _ = load_gps_model(context.config_dict, repo_root=context.repo_root, device=config.device)
    else:
        model = load_gnn_model(context.args, repo_root=context.repo_root, device=config.device)

    screening_solver = NeuralSolverWrapper(
        model,
        model_name.lower(),
        scenario,
        _empty_spec(scenario, config.selection),
        device=config.device,
    )
    generator_scores, _ = compute_generator_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    load_scores, _ = compute_load_relief_scores(scenario, screening_solver, config.selection, config.wildfire)
    decision_spec = build_selected_decision_spec(scenario, generator_scores, load_scores, config.selection)
    solver = NeuralSolverWrapper(model, model_name.lower(), scenario, decision_spec, device=config.device)
    problem = ImprovedDispatchOptimizationProblem(scenario, decision_spec, solver, config)
    result = problem.optimize()

    validation = validate_run(decision_spec, result["final"]["predictions"], result["final"]["objective"])
    history_path = save_objective_history_csv(result["history"], run_dir / "objective_history.csv")
    save_dataframe(generator_scores, run_dir / "generator_scores.csv")
    save_dataframe(load_scores, run_dir / "load_scores.csv")
    save_dataframe(decision_spec.decision_metadata(), run_dir / "selected_decisions.csv")
    risky_before = summarize_top_risky_lines(
        result["baseline"]["loading"],
        result["baseline"]["branch_risk_terms"],
        result["baseline"]["line_weights"],
        top_n=config.wildfire.top_n_risky_lines,
    )
    risky_after = summarize_top_risky_lines(
        result["final"]["loading"],
        result["final"]["branch_risk_terms"],
        result["final"]["line_weights"],
        top_n=config.wildfire.top_n_risky_lines,
    )
    save_dataframe(risky_before, run_dir / "top_risky_lines_baseline.csv")
    save_dataframe(risky_after, run_dir / "top_risky_lines_optimized.csv")
    plot_objective_history(result["history"], run_dir / "objective_history.png")
    plot_risky_line_comparison(risky_before, risky_after, run_dir / "risky_line_comparison.png", f"{model_name.upper()} risky-line loading")

    summary = {
        "model_name": model_name.lower(),
        "success": result["success"],
        "message": result["message"],
        "n_iter": result["n_iter"],
        "baseline_objective": result["baseline"]["objective"],
        "optimized_objective": result["final"]["objective"],
        "baseline_wildfire_cost": result["baseline"]["wildfire_cost"],
        "optimized_wildfire_cost": result["final"]["wildfire_cost"],
        "optimized_total_shed_mw": result["final"]["total_shed_mw"],
        "selected_generator_buses": decision_spec.selected_generator_buses.tolist(),
        "selected_load_buses": decision_spec.selected_load_buses.tolist(),
        "validation_all_passed": validation["all_passed"],
        "history_csv": str(history_path),
        "run_dir": str(run_dir),
    }
    save_optimization_summary_json(summary, run_dir / "summary.json")

    report_path = config.package_root / "methodology_report.md"
    if not report_path.exists():
        write_markdown_summary(
            report_path,
            "Improved Optimization Methodology Report",
            [
                "This report is generated by the improved optimization scripts.",
                "",
                "## Automated Run Summaries",
                "",
            ],
        )
    append_run_summary_block(report_path, f"{model_name.upper()} improved optimization run", summary, config.repo_root)

    print(f"[OK] {model_name.upper()} run complete: objective {result['baseline']['objective']:.6f} -> {result['final']['objective']:.6f}")
    print(f"[OK] Selected generators: {decision_spec.selected_generator_buses.tolist()}")
    print(f"[OK] Selected loads: {decision_spec.selected_load_buses.tolist()}")
    print(f"[OK] Results written to {run_dir}")
    return summary


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gnn"
    run_single_experiment(model_name=model_name)
