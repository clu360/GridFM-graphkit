from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import load_first_pass_config, write_config_copy
from experiments.test.wildfire_tests.shared.decision_vector import (
    FirstPassDecisionVector,
    auto_select_decision_buses,
)
from experiments.test.wildfire_tests.shared.gridfm_runner import GridFMRunner, load_gridfm_model
from experiments.test.wildfire_tests.shared.optimization_problem import FirstPassOptimizationProblem
from experiments.test.wildfire_tests.shared.plot_network_changes import plot_network_changes
from experiments.test.wildfire_tests.shared.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_tests.shared.paths import CONFIGS_ROOT, RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_a_first_pass.run_connected_corridor_tradeoffs import MODEL_CONFIGS, TRADEOFF_SETS
from experiments.test.wildfire_tests.shared.scenario import load_first_pass_context
from experiments.test.wildfire_tests.shared.state_extraction import extract_state_quantities
from experiments.test.wildfire_tests.shared.wildfire_risk import (
    compute_counterfactual_line_impacts,
    compute_grouped_wildfire_risk,
    risk_summary_dict,
)
from experiments.test.wildfire_tests.shared.wildfire_setup import (
    automatic_visualization_summary_fields,
    build_wildfire_for_baseline,
    write_automatic_group_artifacts,
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


def build_problem(config_path: Path):
    config = load_first_pass_config(config_path)
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
    wildfire, baseline_line_impact, automatic_artifacts = build_wildfire_for_baseline(
        config,
        scenario,
        runner,
        decision_vector,
        baseline_prediction,
        baseline_state,
    )
    baseline_risk, baseline_line_risk, baseline_group_risk = compute_grouped_wildfire_risk(
        baseline_state["loading_ratio"],
        wildfire,
        impact=baseline_line_impact,
    )
    if config.objective.normalize_terms:
        if config.objective.risk_normalizer <= 0.0:
            config.objective.risk_normalizer = float(max(baseline_risk, 1e-12))
        if config.objective.load_shedding_normalizer <= 0.0:
            config.objective.load_shedding_normalizer = 1.0

    problem = FirstPassOptimizationProblem(scenario, decision_vector, runner, wildfire, config)
    return (
        config,
        scenario,
        decision_vector,
        runner,
        problem,
        baseline_prediction,
        baseline_state,
        baseline_line_impact,
        baseline_risk,
        baseline_line_risk,
        baseline_group_risk,
        automatic_artifacts,
    )


def build_grid_seed_candidates(
    problem: FirstPassOptimizationProblem,
    num_points: int,
    max_seeds: int,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    decision_vector = problem.decision_vector
    rows = []
    u_base = decision_vector.u_base.copy()
    baseline = problem.evaluate(u_base, record=False)
    rows.append(
        {
            "seed_source": "baseline",
            "decision_label": "baseline",
            "decision_index": -1,
            "decision_value": np.nan,
            "objective_total": float(baseline["objective_total"]),
            "wildfire_group_risk": float(baseline["wildfire_group_risk"]),
            "load_shedding": float(baseline["load_shedding"]),
            "equal_bus_load_shedding": float(baseline.get("equal_bus_load_shedding", np.nan)),
            "unserved_demand_mw": float(baseline.get("unserved_demand_mw", np.nan)),
            "u": u_base.copy(),
        }
    )

    metadata = decision_vector.metadata_frame(u_base)
    for _, meta in metadata.iterrows():
        idx = int(meta["decision_index"])
        label = f"{meta['decision_type']}_{idx}_bus{int(meta['bus_idx'])}"
        values = np.linspace(float(meta["lower_bound"]), float(meta["upper_bound"]), int(num_points))
        for value in values:
            u = u_base.copy()
            u[idx] = float(value)
            components = problem.evaluate(u, record=False)
            rows.append(
                {
                    "seed_source": "one_variable_grid",
                    "decision_label": label,
                    "decision_index": idx,
                    "decision_value": float(value),
                    "objective_total": float(components["objective_total"]),
                    "wildfire_group_risk": float(components["wildfire_group_risk"]),
                    "load_shedding": float(components["load_shedding"]),
                    "equal_bus_load_shedding": float(components.get("equal_bus_load_shedding", np.nan)),
                    "unserved_demand_mw": float(components.get("unserved_demand_mw", np.nan)),
                    "u": u,
                }
            )

    frame = pd.DataFrame(rows)
    sorted_frame = frame.sort_values("objective_total", kind="mergesort")
    starts: list[np.ndarray] = []
    selected_rows = []
    seen = set()
    for _, row in sorted_frame.iterrows():
        u = np.asarray(row["u"], dtype=float)
        key = tuple(np.round(u, decimals=10).tolist())
        if key in seen:
            continue
        seen.add(key)
        starts.append(u)
        selected_rows.append(row)
        if len(starts) >= int(max_seeds):
            break

    seed_frame = pd.DataFrame(selected_rows).drop(columns=["u"]).reset_index(drop=True)
    seed_frame.insert(0, "start_idx", np.arange(len(seed_frame), dtype=int))
    return starts, seed_frame


def _result_rows(result: dict, decision_vector: FirstPassDecisionVector) -> pd.DataFrame:
    rows = []
    for item in result["multistart_results"]:
        delta_pg, alpha = decision_vector.split_decision_vector(item["u_final"])
        rows.append(
            {
                "start_idx": int(item["start_idx"]),
                "success": bool(item["success"]),
                "message": item["message"],
                "n_iter": int(item["n_iter"]),
                "num_objective_evals": int(item["num_objective_evals"]),
                "start_objective": float(item["start_objective"]),
                "final_objective": float(item["final"]["objective_total"]),
                "final_wildfire_group_risk": float(item["final"]["wildfire_group_risk"]),
                "final_load_shedding": float(item["final"]["load_shedding"]),
                "final_equal_bus_load_shedding": float(item["final"].get("equal_bus_load_shedding", np.nan)),
                "final_unserved_demand_mw": float(item["final"].get("unserved_demand_mw", np.nan)),
                "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
                "min_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
                "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
                "is_best": int(item["start_idx"]) == int(result["best_start_idx"]),
            }
        )
    return pd.DataFrame(rows)


def _risk_before_after_frame(before: pd.DataFrame, after: pd.DataFrame, key: str) -> pd.DataFrame:
    return before.merge(after, on=key, suffixes=("_before", "_after"))


def run_multistart_optimization(
    config_path: str | Path,
    num_seed_points: int = 11,
    max_seeds: int = 5,
    output_root: str | Path | None = None,
) -> Path:
    config_path = Path(config_path)
    (
        config,
        scenario,
        decision_vector,
        runner,
        problem,
        _baseline_prediction,
        _baseline_state,
        _baseline_line_impact,
        baseline_risk,
        baseline_line_risk,
        baseline_group_risk,
        automatic_artifacts,
    ) = build_problem(config_path)
    if output_root is None:
        output_root = RESULTS_ROOT / "stage_b" / "multistart" / "manual"
    output_root = Path(output_root)
    run_dir = output_root / f"multistart_{config.model.model_type.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_config_copy(config, run_dir / "config.yaml")
    write_json(run_dir / "wildfire_scenario.json", problem.wildfire.to_dict())
    write_dataframe(run_dir / "baseline_wildfire_risk.csv", baseline_line_risk)
    write_json(
        run_dir / "baseline_wildfire_risk_summary.json",
        risk_summary_dict(baseline_risk, baseline_line_risk, baseline_group_risk),
    )
    automatic_metadata = write_automatic_group_artifacts(
        run_dir,
        automatic_artifacts,
        baseline_risk,
        config,
    )
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
    write_json(run_dir / "baseline_objective_components.json", problem.evaluate(decision_vector.u_base))
    write_dataframe(run_dir / "decision_vector_initial.csv", decision_vector.metadata_frame(decision_vector.u_base))

    starts, seed_frame = build_grid_seed_candidates(problem, num_seed_points, max_seeds)
    write_dataframe(run_dir / "selected_starts.csv", seed_frame)
    result = problem.optimize_multistart(starts)
    write_json(run_dir / "final_objective_components.json", result["final"])
    write_dataframe(run_dir / "multistart_results.csv", _result_rows(result, decision_vector))
    best_trace = result["trace"].to_frame()
    write_dataframe(run_dir / "best_objective_trace.csv", best_trace)
    write_dataframe(run_dir / "objective_trace.csv", best_trace)
    write_dataframe(run_dir / "best_decision_vector.csv", decision_vector.metadata_frame(result["u_final"]))
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
        problem.wildfire,
        final_prediction,
    )
    _final_risk, final_line_risk, final_group_risk = compute_grouped_wildfire_risk(
        final_state["loading_ratio"],
        problem.wildfire,
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

    summary = {
        "analysis_type": "grid_seeded_multistart",
        "config_path": str(config_path),
        "model_type": config.model.model_type.lower(),
        "num_seed_points_per_variable": int(num_seed_points),
        "num_starts": int(result["num_starts"]),
        "best_start_idx": int(result["best_start_idx"]),
        "best_start_index": int(result["best_start_idx"]),
        "baseline_objective": float(result["baseline"]["objective_total"]),
        "baseline_grouped_risk": float(baseline_risk),
        "best_start_objective": float(result["start_objective"]),
        "best_seed_objective": float(result["start_objective"]),
        "best_final_objective": float(result["final"]["objective_total"]),
        "best_objective": float(result["final"]["objective_total"]),
        "best_final_wildfire_group_risk": float(result["final"]["wildfire_group_risk"]),
        "best_grouped_risk": float(result["final"]["wildfire_group_risk"]),
        "best_final_load_shedding": float(result["final"]["load_shedding"]),
        "best_load_shedding": float(result["final"]["load_shedding"]),
        "best_final_equal_bus_load_shedding": float(result["final"].get("equal_bus_load_shedding", np.nan)),
        "best_final_unserved_demand_mw": float(result["final"].get("unserved_demand_mw", np.nan)),
        "optimizer_success": bool(result["success"]),
        "optimizer_message": result["message"],
        "lambda_R": float(config.objective.lambda_R),
        "lambda_L": float(config.objective.lambda_L),
        "risk_normalizer": float(config.objective.risk_normalizer),
        "load_shedding_normalizer": float(config.objective.load_shedding_normalizer),
        "load_shedding_metric": "demand_weighted_fraction",
        "selected_starts_csv": str(run_dir / "selected_starts.csv"),
        "multistart_results_csv": str(run_dir / "multistart_results.csv"),
        "best_objective_trace_csv": str(run_dir / "best_objective_trace.csv"),
        "objective_trace_csv": str(run_dir / "objective_trace.csv"),
        "run_dir": str(run_dir),
        "selection_method": config.wildfire.selection_method,
        "requested_top_fraction": automatic_metadata.get("requested_top_fraction"),
        "realized_selected_fraction": automatic_metadata.get("realized_selected_fraction"),
        "num_selected_lines": automatic_metadata.get("num_selected_lines"),
        "num_groups": automatic_metadata.get("num_groups"),
        "collapsed_to_single_group": automatic_metadata.get("collapsed_to_single_group"),
        "largest_group_num_lines": automatic_metadata.get("largest_group_num_lines"),
        "largest_group_fraction_of_selected_lines": automatic_metadata.get(
            "largest_group_fraction_of_selected_lines"
        ),
    }
    write_json(run_dir / "optimization_summary.json", summary)
    visualization_summary = {
        "optimization_behavior_plot": None,
        "topology_change_plot": None,
        "errors": [],
    }
    if automatic_metadata:
        visualization_summary.update(automatic_visualization_summary_fields(automatic_metadata))
    try:
        visualization_summary["optimization_behavior_plot"] = str(plot_optimization_behavior(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"optimization_behavior: {exc}")
    try:
        visualization_summary["topology_change_plot"] = str(plot_network_changes(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"topology_change: {exc}")
    summary["optimization_behavior_plot"] = visualization_summary["optimization_behavior_plot"] or "not available"
    summary["topology_change_plot"] = visualization_summary["topology_change_plot"] or "not available"
    summary["visualization_errors"] = visualization_summary["errors"]
    write_json(run_dir / "visualization_summary.json", visualization_summary)
    write_json(run_dir / "optimization_summary.json", summary)
    write_json(run_dir / "analysis_summary.json", summary)
    print(f"[OK] Multistart optimization written to {run_dir}")
    print(f"[OK] Objective: {summary['baseline_objective']:.6f} -> {summary['best_final_objective']:.6f}")
    return run_dir


def _multistart_root() -> Path:
    return RESULTS_ROOT / "stage_b" / "multistart"


def run_multistart_tradeoff_sets(
    num_seed_points: int = 11,
    max_seeds: int = 5,
    clear: bool = False,
) -> Path:
    root = _multistart_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected_parent = RESULTS_ROOT.resolve()
        if resolved_root.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    generated_configs = CONFIGS_ROOT / "generated_multistart_tradeoffs"
    generated_configs.mkdir(parents=True, exist_ok=True)
    rows = []
    for set_name, weights in TRADEOFF_SETS.items():
        for model_type, config_name in MODEL_CONFIGS.items():
            base_config_path = CONFIGS_ROOT / config_name
            config = load_first_pass_config(base_config_path)
            config.model.model_type = model_type
            config.decision.alpha_min = 0.0
            config.objective.lambda_R = float(weights["lambda_R"])
            config.objective.lambda_L = float(weights["lambda_L"])
            config.objective.risk_normalizer = 0.0
            config.objective.load_shedding_normalizer = 0.0
            config.output.output_root = str(root / set_name / model_type)
            config.output.run_name = f"multistart_{set_name}_{model_type}"

            run_config_path = generated_configs / f"{set_name}_{model_type}.yaml"
            write_config_copy(config, run_config_path)
            run_dir = run_multistart_optimization(
                run_config_path,
                num_seed_points=num_seed_points,
                max_seeds=max_seeds,
                output_root=root / set_name / model_type,
            )
            summary_path = run_dir / "analysis_summary.json"
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            rows.append(
                {
                    "tradeoff_set": set_name,
                    "description": weights["description"],
                    "model_type": model_type,
                    "lambda_R": float(weights["lambda_R"]),
                    "lambda_L": float(weights["lambda_L"]),
                    "baseline_objective": summary["baseline_objective"],
                    "best_start_objective": summary["best_start_objective"],
                    "best_final_objective": summary["best_final_objective"],
                    "best_final_wildfire_group_risk": summary["best_final_wildfire_group_risk"],
                    "best_final_load_shedding": summary["best_final_load_shedding"],
                    "best_final_equal_bus_load_shedding": summary.get("best_final_equal_bus_load_shedding"),
                    "best_final_unserved_demand_mw": summary.get("best_final_unserved_demand_mw"),
                    "optimizer_success": summary["optimizer_success"],
                    "optimizer_message": summary["optimizer_message"],
                    "num_starts": summary["num_starts"],
                    "best_start_idx": summary["best_start_idx"],
                    "run_dir": str(run_dir),
                    "optimization_behavior_plot": summary.get("optimization_behavior_plot"),
                }
            )

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "multistart_tradeoff_summary.csv"
    summary_json = root / "multistart_tradeoff_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "num_runs": int(len(rows)),
            "num_seed_points_per_variable": int(num_seed_points),
            "max_seeds": int(max_seeds),
            "summary_csv": str(summary_csv),
            "tradeoff_sets": TRADEOFF_SETS,
        },
    )
    print(f"[OK] Multistart tradeoff summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIGS_ROOT / "connected_corridor_gps.yaml",
    )
    parser.add_argument("--num-seed-points", type=int, default=11)
    parser.add_argument("--max-seeds", type=int, default=5)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--tradeoff-sets", action="store_true", help="Run canonical risk_leaning/balanced/service_leaning cases for GNN and GPS.")
    parser.add_argument("--clear", action="store_true", help="Delete existing results/stage_b/multistart before tradeoff-set generation.")
    args = parser.parse_args()
    if args.tradeoff_sets:
        run_multistart_tradeoff_sets(
            num_seed_points=args.num_seed_points,
            max_seeds=args.max_seeds,
            clear=args.clear,
        )
        return
    run_multistart_optimization(
        args.config,
        num_seed_points=args.num_seed_points,
        max_seeds=args.max_seeds,
        output_root=args.output_root,
    )


if __name__ == "__main__":
    main()
