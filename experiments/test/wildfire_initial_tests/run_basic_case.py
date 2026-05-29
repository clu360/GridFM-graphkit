from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_initial_tests.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_initial_tests.decision_vector import (
    FirstPassDecisionVector,
    auto_select_decision_buses,
)
from experiments.test.wildfire_initial_tests.gridfm_runner import GridFMRunner, load_gridfm_model
from experiments.test.wildfire_initial_tests.optimization_problem import FirstPassOptimizationProblem
from experiments.test.wildfire_initial_tests.plot_network_changes import plot_network_changes
from experiments.test.wildfire_initial_tests.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_initial_tests.reporting import (
    git_metadata,
    make_run_dir,
    write_dataframe,
    write_json,
)
from experiments.test.wildfire_initial_tests.scenario import load_first_pass_context
from experiments.test.wildfire_initial_tests.state_extraction import compare_states, extract_state_quantities
from experiments.test.wildfire_initial_tests.validation import (
    validate_baseline_prediction,
    validate_optimization_result,
)
from experiments.test.wildfire_initial_tests.wildfire_risk import (
    compute_counterfactual_line_impacts,
    compute_grouped_wildfire_risk,
    risk_summary_dict,
)
from experiments.test.wildfire_initial_tests.wildfire_scenario import (
    build_synthetic_wildfire_scenario,
    validate_connected_line_group,
)


def _build_decision_vector(config, scenario) -> FirstPassDecisionVector:
    if config.decision.selected_generator_buses or config.decision.selected_load_buses:
        selected_g = np.asarray(config.decision.selected_generator_buses, dtype=int)
        selected_l = np.asarray(config.decision.selected_load_buses, dtype=int)
    else:
        selected_g, selected_l = auto_select_decision_buses(
            scenario,
            config.decision.auto_select_generators,
            config.decision.auto_select_loads,
        )
    return FirstPassDecisionVector(
        scenario,
        selected_g,
        selected_l,
        delta_pg_bound_mw=config.decision.delta_pg_bound_mw,
        alpha_min=config.decision.alpha_min,
        alpha_max=config.decision.alpha_max,
    )


def _json_prediction_summary(config, scenario, state):
    return {
        "model_type": config.model.model_type.lower(),
        "scenario_id": scenario.scenario_id,
        "num_buses": int(scenario.num_buses),
        "num_lines": int(state["num_lines"]),
        "prediction_has_nan": bool(state["prediction_has_nan"]),
        "prediction_has_inf": bool(state["prediction_has_inf"]),
        "max_voltage": float(state["max_voltage"]),
        "min_voltage": float(state["min_voltage"]),
        "max_loading_ratio": float(state["max_loading_ratio"]),
        "state_extraction_passed": bool(state["state_extraction_passed"]),
    }


def _perturbation_smoke_test(decision_vector, runner, scenario, baseline_state, config):
    u = decision_vector.u_base.copy()
    if decision_vector.n_generators:
        u[0] = min(1.0, decision_vector.u_max[0])
    if decision_vector.n_loads:
        u[decision_vector.n_generators] = max(0.99, decision_vector.u_min[decision_vector.n_generators])
    bounds_ok, bounds_message = decision_vector.check_bounds(u)
    pred = runner.predict(u)
    state = extract_state_quantities(
        scenario,
        pred,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    metrics = compare_states(baseline_state, state)
    metrics.update(
        {
            "bounds_ok": bool(bounds_ok),
            "bounds_message": bounds_message,
            "prediction_has_nan": bool(state["prediction_has_nan"]),
            "prediction_has_inf": bool(state["prediction_has_inf"]),
            "perturbed_prediction_correct_shape": bool(len(state["Vm"]) == scenario.num_buses),
            "passed": bool(
                bounds_ok
                and not state["prediction_has_nan"]
                and not state["prediction_has_inf"]
                and len(state["Vm"]) == scenario.num_buses
                and metrics["nonzero_response"]
            ),
        }
    )
    return metrics


def _risk_before_after_frame(before: pd.DataFrame, after: pd.DataFrame, key: str) -> pd.DataFrame:
    return before.merge(after, on=key, suffixes=("_before", "_after"))


def write_summary_markdown(run_dir: Path, summary: dict) -> None:
    path = run_dir / "run_summary.md"
    lines = [
        "# Wildfire First Pass Summary",
        "",
        "## What Was Implemented",
        "",
        "A reduced fixed-topology wildfire predict-then-optimize experiment path was added under `experiments/test/wildfire_initial_tests`.",
        "",
        "## Reused Modules",
        "",
        "- `experiments.test.pipeline_utils` for scenario and checkpoint loading.",
        "- `experiments.test.neural_solver.NeuralSolverWrapper` for in-memory GridFM inference.",
        "- `experiments.test.scenario_data.ScenarioData` for the IEEE-30 scenario representation.",
        "- `experiments.test.overload_penalty` for branch loading reconstruction.",
        "",
        "## New Modules Added",
        "",
        "- `config.py`, `scenario.py`, `decision_vector.py`, `gridfm_runner.py`, `state_extraction.py`",
        "- `wildfire_scenario.py`, `wildfire_risk.py`, `objective.py`, `optimization_problem.py`",
        "- `validation.py`, `reporting.py`, `run_basic_case.py`, `run_stability_sweep.py`",
        "- `plot_optimization_behavior.py`, `plot_network_changes.py`",
        "",
        "## First-Pass Assumptions",
        "",
        "The topology is fixed, all buses and lines are energized, Qg is fixed at baseline, and the optimizer only controls selected real-power generator redispatch and selected load-service fractions.",
        "",
        "## Relation To Full Formulation",
        "",
        "The full formulation includes Pg, Qg, alpha, bus energization, and line energization. This first pass keeps Qg, y, and z fixed and tests whether the continuous reduced controls can lower grouped wildfire exposure.",
        "",
        "The optimized scalar objective is `lambda_R * (R_group / R_baseline) + lambda_L * L_shed_weighted`, where `L_shed_weighted = sum_n (Pd_n / sum_m Pd_m) * (1 - alpha_n)`. Generator movement is recorded as a diagnostic, not used as a cost term.",
        "",
        "`R_group` uses `z_l * p_env_l * loading_l^2 * I_l(u)` summed over configured line groups. `z_l` is fixed at 1, and `I_l(u)` is the relative equal-weight served-load loss from a counterfactual one-line GridFM outage.",
        "",
        "## GridFM Surrogate Use",
        "",
        "GridFM is loaded once in memory and called through `NeuralSolverWrapper`; the CLI is not called inside optimization iterations.",
        "",
        "## Grouped Wildfire Scenario",
        "",
        "A synthetic high-risk corridor is represented as a `WildfireLineGroup`; risk is summed at line and group levels with counterfactual `impact` values written to the per-line CSVs.",
        "",
        "## Basic Run",
        "",
        f"Latest run directory: `{run_dir}`",
        "",
        "## Stability Sweep",
        "",
        "Implemented as `run_stability_sweep.py`; run results are only reported here after execution.",
        "",
        "## Validation Checks",
        "",
        "Baseline prediction validity, perturbation response, decision bounds, finite objective components, final prediction validity, alpha preservation, and risk/objective reduction are checked.",
        "",
        "## Tests Added",
        "",
        "Unit and smoke tests are intended under `tests/test_wildfire_initial_tests_*.py`.",
        "",
        "## Commands Run",
        "",
        f"- `python experiments/test/wildfire_initial_tests/run_basic_case.py --config {summary['config_path']}`",
        "",
        "## Final Results",
        "",
        f"- baseline prediction worked: {summary['baseline_prediction_passed']}",
        f"- perturbation produced a nonzero response: {summary['perturbation_nonzero_response']}",
        f"- optimization reduced total objective: {summary['objective_reduced']}",
        f"- optimization reduced grouped wildfire risk: {summary['wildfire_risk_reduced']}",
        f"- final mean alpha: {summary['mean_alpha']}",
        f"- final min alpha: {summary['min_alpha']}",
        f"- max absolute Delta_Pg: {summary['max_abs_delta_pg']}",
        f"- optimizer success/failure status: {summary['optimizer_success']} ({summary['optimizer_message']})",
        f"- number of objective evaluations: {summary['num_objective_evals']}",
        f"- GNN and GPS both ran if tested: {summary.get('both_models_tested', 'not tested')}",
        f"- stability sweep summary if run: {summary.get('stability_sweep_summary', 'not run')}",
        f"- validation failures: {summary.get('validation_failures', [])}",
        f"- optimization behavior plot: {summary.get('optimization_behavior_plot', 'not available')}",
        f"- topology change plot: {summary.get('topology_change_plot', 'not available')}",
        "",
        "## Commands to Reproduce",
        "",
        "```powershell",
        "python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/basic_gps.yaml",
        "python experiments/test/wildfire_initial_tests/run_basic_case.py --config experiments/test/wildfire_initial_tests/configs/basic_gnn.yaml",
        "python experiments/test/wildfire_initial_tests/run_stability_sweep.py --config experiments/test/wildfire_initial_tests/configs/stability_sweep.yaml",
        "python experiments/test/wildfire_initial_tests/plot_optimization_behavior.py --run-dir experiments/test/wildfire_initial_tests/results/<run_name>",
        "python experiments/test/wildfire_initial_tests/plot_network_changes.py --run-dir experiments/test/wildfire_initial_tests/results/<run_name>",
        "pytest tests/test_wildfire_initial_tests_scenario.py",
        "pytest tests/test_wildfire_initial_tests_risk.py",
        "pytest tests/test_wildfire_initial_tests_decision_vector.py",
        "pytest tests/test_wildfire_initial_tests_objective.py",
        "pytest tests/test_wildfire_initial_tests_basic_run.py",
        "```",
        "",
        "## Limitations And Next Steps",
        "",
        "This is not a full wildfire-resilience-aware OPF. Next steps are to calibrate risk inputs, improve feasibility diagnostics, compare against a physical solver, and decide whether the reduced formulation should move toward topology controls.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_basic_case(config_path: Path) -> dict:
    config = load_first_pass_config(config_path)
    np.random.seed(config.random_seed)
    run_dir = make_run_dir(config.output_root_path(), config.output.run_name)
    write_config_copy(config, run_dir / "config.yaml")
    write_json(run_dir / "metadata.json", {**git_metadata(), "config_path": str(config_path)})

    context = load_first_pass_context(config)
    scenario = context.scenario
    decision_vector = _build_decision_vector(config, scenario)
    model = load_gridfm_model(config, context)
    runner = GridFMRunner(
        model,
        config.model.model_type,
        scenario,
        decision_vector,
        device=config.model.device,
    )

    baseline_prediction = runner.predict(decision_vector.u_base)
    baseline_state = extract_state_quantities(
        scenario,
        baseline_prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    baseline_prediction_json = _json_prediction_summary(config, scenario, baseline_state)
    write_json(run_dir / "baseline_prediction.json", baseline_prediction_json)
    baseline_gate = validate_baseline_prediction(baseline_state)
    if not baseline_gate["passed"]:
        write_json(run_dir / "validation_summary.json", baseline_gate)
        return {"run_dir": str(run_dir), "baseline_prediction_passed": False}

    perturbation = _perturbation_smoke_test(decision_vector, runner, scenario, baseline_state, config)
    write_json(run_dir / "perturbation_smoke_test.json", perturbation)

    wildfire = build_synthetic_wildfire_scenario(
        baseline_state["loading_ratio"],
        selected_line_ids=config.wildfire.selected_line_ids,
        selection_method=config.wildfire.selection_method,
        num_high_risk_lines=config.wildfire.num_high_risk_lines,
        high_hazard=config.wildfire.high_hazard,
        default_hazard=config.wildfire.default_hazard,
        default_impact=config.wildfire.default_impact,
        group_weight=config.wildfire.group_weight,
        hazard_multiplier=config.wildfire.hazard_multiplier,
    )
    if config.wildfire.selection_method == "manual_connected":
        validate_connected_line_group(scenario.edge_index, wildfire.line_groups[0].line_ids)
    write_json(run_dir / "wildfire_scenario.json", wildfire.to_dict())
    baseline_line_impact = compute_counterfactual_line_impacts(
        decision_vector.u_base,
        scenario,
        runner,
        wildfire,
        baseline_prediction,
    )
    baseline_risk, baseline_line_risk, baseline_group_risk = compute_grouped_wildfire_risk(
        baseline_state["loading_ratio"],
        wildfire,
        impact=baseline_line_impact,
    )
    write_dataframe(run_dir / "baseline_wildfire_risk.csv", baseline_line_risk)
    write_json(
        run_dir / "baseline_wildfire_risk_summary.json",
        risk_summary_dict(baseline_risk, baseline_line_risk, baseline_group_risk),
    )

    if config.objective.normalize_terms:
        if config.objective.risk_normalizer <= 0.0:
            config.objective.risk_normalizer = float(max(baseline_risk, 1e-12))
        if config.objective.load_shedding_normalizer <= 0.0:
            config.objective.load_shedding_normalizer = 1.0
        write_json(
            run_dir / "objective_normalizers.json",
            {
                "normalize_terms": config.objective.normalize_terms,
                "risk_normalizer": config.objective.risk_normalizer,
                "load_shedding_normalizer": config.objective.load_shedding_normalizer,
                "risk_normalizer_source": "baseline_grouped_wildfire_risk",
                "load_shedding_normalizer_source": "demand_weighted_fraction_already_normalized",
                "load_shedding_metric": "demand_weighted_fraction",
            },
        )

    problem = FirstPassOptimizationProblem(scenario, decision_vector, runner, wildfire, config)
    baseline_components = problem.evaluate(decision_vector.u_base)
    write_json(run_dir / "baseline_objective_components.json", baseline_components)
    write_dataframe(run_dir / "decision_vector_initial.csv", decision_vector.metadata_frame(decision_vector.u_base))

    result = problem.optimize()
    final_components = result["final"]
    write_json(run_dir / "final_objective_components.json", final_components)
    write_dataframe(run_dir / "objective_trace.csv", result["trace"].to_frame())
    write_dataframe(run_dir / "decision_vector_final.csv", decision_vector.metadata_frame(result["u_final"]))

    final_prediction = runner.predict(result["u_final"])
    final_state = extract_state_quantities(
        scenario,
        final_prediction,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    final_line_impact = compute_counterfactual_line_impacts(
        result["u_final"],
        scenario,
        runner,
        wildfire,
        final_prediction,
    )
    final_risk, final_line_risk, final_group_risk = compute_grouped_wildfire_risk(
        final_state["loading_ratio"],
        wildfire,
        impact=final_line_impact,
    )
    write_dataframe(
        run_dir / "risk_by_line_before_after.csv",
        _risk_before_after_frame(baseline_line_risk, final_line_risk, "line_id"),
    )
    write_dataframe(
        run_dir / "risk_by_group_before_after.csv",
        _risk_before_after_frame(baseline_group_risk, final_group_risk, "group_name"),
    )

    validation = validate_optimization_result(
        baseline_components,
        final_components,
        decision_vector,
        result["u_final"],
        result["success"],
        result["message"],
    )
    validation.update({"baseline_prediction": baseline_gate, "perturbation": perturbation})
    write_json(run_dir / "validation_summary.json", validation)

    optimization_summary = {
        "model_type": config.model.model_type.lower(),
        "scenario_id": scenario.scenario_id,
        "selected_generator_buses": decision_vector.selected_generator_buses.tolist(),
        "selected_load_buses": decision_vector.selected_load_buses.tolist(),
        "baseline_objective": float(baseline_components["objective_total"]),
        "final_objective": float(final_components["objective_total"]),
        "baseline_grouped_wildfire_risk": float(baseline_components["wildfire_group_risk"]),
        "final_grouped_wildfire_risk": float(final_components["wildfire_group_risk"]),
        "optimizer_success": bool(result["success"]),
        "optimizer_message": result["message"],
        "num_objective_evals": int(result["num_objective_evals"]),
        "objective_failure_count": int(result["trace"].failure_count),
        "objective_terms_normalized": bool(config.objective.normalize_terms),
        "risk_normalizer": float(config.objective.risk_normalizer),
        "load_shedding_normalizer": float(config.objective.load_shedding_normalizer),
        "load_shedding_metric": "demand_weighted_fraction",
        "lambda_R": float(config.objective.lambda_R),
        "lambda_L": float(config.objective.lambda_L),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "optimization_summary.json", optimization_summary)

    visualization_summary = {
        "optimization_behavior_plot": None,
        "topology_change_plot": None,
        "errors": [],
    }
    try:
        visualization_summary["optimization_behavior_plot"] = str(plot_optimization_behavior(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"optimization_behavior: {exc}")
    try:
        visualization_summary["topology_change_plot"] = str(plot_network_changes(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"topology_change: {exc}")
    write_json(run_dir / "visualization_summary.json", visualization_summary)

    final_summary = {
        "config_path": str(config_path),
        "baseline_prediction_passed": baseline_gate["passed"],
        "perturbation_nonzero_response": perturbation["nonzero_response"],
        "objective_reduced": validation["objective_reduced"],
        "wildfire_risk_reduced": validation["wildfire_risk_reduced"],
        "mean_alpha": validation["mean_alpha"],
        "min_alpha": validation["min_alpha"],
        "max_abs_delta_pg": validation["max_abs_delta_pg"],
        "optimizer_success": result["success"],
        "optimizer_message": result["message"],
        "num_objective_evals": result["num_objective_evals"],
        "validation_failures": [k for k, v in validation.items() if isinstance(v, bool) and not v],
        "optimization_behavior_plot": visualization_summary["optimization_behavior_plot"] or "not available",
        "topology_change_plot": visualization_summary["topology_change_plot"] or "not available",
    }
    write_summary_markdown(run_dir, final_summary)
    print(f"[OK] Basic wildfire first-pass run complete: {run_dir}")
    print(f"[OK] Objective: {optimization_summary['baseline_objective']:.6f} -> {optimization_summary['final_objective']:.6f}")
    print(f"[OK] Group risk: {optimization_summary['baseline_grouped_wildfire_risk']:.6f} -> {optimization_summary['final_grouped_wildfire_risk']:.6f}")
    if visualization_summary["errors"]:
        print(f"[WARN] Visualization errors: {visualization_summary['errors']}")
    else:
        print(f"[OK] Figures: {run_dir / 'figures'}")
    return {**optimization_summary, **final_summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    run_basic_case(args.config)


if __name__ == "__main__":
    main()
