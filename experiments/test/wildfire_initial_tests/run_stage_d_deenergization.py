from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_initial_tests.config import write_config_copy
from experiments.test.wildfire_initial_tests.plot_network_changes import plot_network_changes
from experiments.test.wildfire_initial_tests.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_initial_tests.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_initial_tests.run_stage_c_psps_baseline import (
    CASE_FOLDERS,
    CASES,
    MODEL_CONFIGS,
    _build_model_context,
    _candidate_group_maps,
    _edge_array,
    _group_risk_frame,
    _line_risk_frames,
    compact_fraction_label,
    fraction_label,
)
from experiments.test.wildfire_initial_tests.stage_c_psps import (
    affected_buses,
    apply_environmental_case,
    demand_weighted_load_shed_from_prediction,
    disconnected_component_count,
    predict_psps_state,
)
from experiments.test.wildfire_initial_tests.stage_d_deenergization import (
    LAMBDA_CASES,
    LAMBDA_FOLDERS,
    disconnected_buses_from_reference,
    enumerate_deenergization_subsets,
    expected_subset_count,
    line_risk_total,
    objective_for_lambda,
    select_best_subset,
)
from experiments.test.wildfire_initial_tests.wildfire_setup import write_automatic_group_artifacts


EVALUATION_MODE = "limited_enumerated_z_only"


def _stage_d_root() -> Path:
    return REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "stage_d_deenergization"


def _parse_line_ids(value) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value) or str(value).strip() == "":
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [int(item) for item in parsed]
    except Exception:
        pass
    return [int(item) for item in text.replace("[", "").replace("]", "").split(",") if item.strip()]


def _candidate_line_risk_scores(
    scenario,
    candidate_line_ids,
    line_to_group,
    p_env_by_line,
    baseline_loading,
    fixed_impact,
):
    edge_array = _edge_array(scenario)
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    rows = []
    for line_id in range(int(edge_array.shape[1])):
        rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "group_id": line_to_group.get(line_id, ""),
                "p_env": float(p_env_by_line.get(line_id, 0.0)),
                "loading_base": float(baseline_loading[line_id]),
                "I_l": float(fixed_impact[line_id]),
                "baseline_line_risk": float(
                    p_env_by_line.get(line_id, 0.0) * baseline_loading[line_id] ** 2 * fixed_impact[line_id]
                ),
                "candidate_for_deenergization": bool(line_id in candidate_set),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(
    scenario,
    candidate_line_ids,
    best_deenergized_line_ids,
    line_to_group,
    p_env_by_line,
    fixed_impact,
    baseline_loading,
):
    edge_array = _edge_array(scenario)
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    best_set = set(int(line_id) for line_id in best_deenergized_line_ids)
    rows = []
    for line_id in range(int(edge_array.shape[1])):
        selected = line_id in best_set
        rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "group_id": line_to_group.get(line_id, ""),
                "p_env": float(p_env_by_line.get(line_id, 0.0)),
                "I_l": float(fixed_impact[line_id]),
                "baseline_line_risk": float(
                    p_env_by_line.get(line_id, 0.0) * baseline_loading[line_id] ** 2 * fixed_impact[line_id]
                ),
                "candidate_for_deenergization": bool(line_id in candidate_set),
                "optimized_z_l": int(0 if selected else 1),
                "deenergized": bool(selected),
                "selected_in_best_solution": bool(selected),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_subset(
    model_context: dict,
    candidate_line_ids: list[int],
    p_env_by_line: dict[int, float],
    fixed_impact: np.ndarray,
    baseline_risk: float,
    subset: tuple[int, ...],
    eval_id: int,
) -> dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    runner = model_context["runner"]
    decision_vector = model_context["decision_vector"]
    baseline_prediction = model_context["baseline_prediction"]
    baseline_state = model_context["baseline_state"]
    deenergized = [int(line_id) for line_id in subset]
    z_by_line = {
        int(line_id): (0 if int(line_id) in set(deenergized) else 1)
        for line_id in range(int(scenario.edge_index.shape[1]))
    }
    try:
        if deenergized:
            prediction, state = predict_psps_state(
                scenario,
                runner,
                decision_vector.u_base,
                deenergized,
                standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
            )
        else:
            prediction = baseline_prediction
            state = baseline_state
        loading = np.asarray(state["loading_ratio"], dtype=float)
        R_group, _risk_by_line = line_risk_total(
            loading,
            p_env_by_line,
            fixed_impact,
            z_by_line,
            candidate_line_ids,
        )
        L_norm = demand_weighted_load_shed_from_prediction(prediction, scenario) if deenergized else 0.0
        affected = affected_buses(scenario.edge_index, deenergized)
        disconnected = disconnected_buses_from_reference(scenario.edge_index, scenario.num_buses, deenergized)
        components = disconnected_component_count(scenario.edge_index, scenario.num_buses, deenergized)
        return {
            "eval_id": int(eval_id),
            "deenergized_line_ids": deenergized,
            "num_deenergized_lines": int(len(deenergized)),
            "R_group": float(R_group),
            "R_norm": float(R_group / max(float(baseline_risk), 1e-12)),
            "L_norm": float(L_norm),
            "demand_weighted_load_shed": float(L_norm),
            "num_disconnected_components": int(components),
            "affected_bus_ids": [int(bus) for bus in affected],
            "disconnected_bus_ids": [int(bus) for bus in disconnected],
            "valid_prediction": bool(True),
            "status": "ok",
            "error": "",
        }
    except Exception as exc:
        affected = affected_buses(model_context["scenario"].edge_index, deenergized)
        disconnected = disconnected_buses_from_reference(
            model_context["scenario"].edge_index,
            model_context["scenario"].num_buses,
            deenergized,
        )
        components = disconnected_component_count(
            model_context["scenario"].edge_index,
            model_context["scenario"].num_buses,
            deenergized,
        )
        return {
            "eval_id": int(eval_id),
            "deenergized_line_ids": deenergized,
            "num_deenergized_lines": int(len(deenergized)),
            "R_group": np.nan,
            "R_norm": np.nan,
            "L_norm": np.nan,
            "demand_weighted_load_shed": np.nan,
            "num_disconnected_components": int(components),
            "affected_bus_ids": [int(bus) for bus in affected],
            "disconnected_bus_ids": [int(bus) for bus in disconnected],
            "valid_prediction": bool(False),
            "status": "failed",
            "error": str(exc),
        }


def _evaluate_all_subsets(model_context: dict, candidate_line_ids: list[int], p_env_by_line, fixed_impact, baseline_risk, max_deenergized_lines):
    subsets = enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=max_deenergized_lines)
    rows = [
        _evaluate_subset(
            model_context,
            candidate_line_ids,
            p_env_by_line,
            fixed_impact,
            baseline_risk,
            subset,
            eval_id,
        )
        for eval_id, subset in enumerate(subsets)
    ]
    return pd.DataFrame(rows)


def _stage_c_comparison_row(stage_c_summary: pd.DataFrame | None, model_type: str, case_name: str, grouping_top_fraction: float):
    if stage_c_summary is None or stage_c_summary.empty:
        return None
    matches = stage_c_summary[
        (stage_c_summary["model_type"].astype(str).str.lower() == model_type.lower())
        & (stage_c_summary["environmental_risk_case"].astype(str) == case_name)
        & np.isclose(stage_c_summary["grouping_top_fraction"].astype(float), float(grouping_top_fraction))
        & (stage_c_summary["status"].astype(str) == "ok")
    ]
    if matches.empty:
        return None
    return matches.iloc[-1].to_dict()


def _load_stage_c_summary() -> pd.DataFrame | None:
    path = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "stage_c_psps" / "stage_c_psps_summary.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _evaluation_to_csv_frame(evaluations: pd.DataFrame, lambda_case: str, lambda_R: float, lambda_L: float) -> pd.DataFrame:
    frame = evaluations.copy()
    frame["lambda_case"] = lambda_case
    frame["lambda_R"] = float(lambda_R)
    frame["lambda_L"] = float(lambda_L)
    frame["objective"] = frame.apply(lambda row: objective_for_lambda(row, lambda_R, lambda_L), axis=1)
    return frame[
        [
            "eval_id",
            "lambda_case",
            "lambda_R",
            "lambda_L",
            "deenergized_line_ids",
            "num_deenergized_lines",
            "R_group",
            "R_norm",
            "L_norm",
            "objective",
            "demand_weighted_load_shed",
            "num_disconnected_components",
            "affected_bus_ids",
            "disconnected_bus_ids",
            "valid_prediction",
            "status",
            "error",
        ]
    ]


def _objective_trace(frame: pd.DataFrame) -> pd.DataFrame:
    trace = frame.rename(columns={"eval_id": "eval_idx"}).copy()
    trace["objective_total"] = trace["objective"]
    trace["wildfire_group_risk"] = trace["R_group"]
    trace["load_shedding"] = trace["L_norm"]
    trace["normalized_wildfire_group_risk"] = trace["R_norm"]
    trace["normalized_load_shedding"] = trace["L_norm"]
    trace["risk_objective_term"] = trace["lambda_R"] * trace["R_norm"]
    trace["load_shedding_objective_term"] = trace["lambda_L"] * trace["L_norm"]
    trace["generator_movement"] = 0.0
    trace["max_loading_ratio"] = np.nan
    trace["mean_alpha"] = 1.0
    trace["max_abs_delta_pg"] = 0.0
    trace["message"] = trace["deenergized_line_ids"].apply(lambda value: f"enumerated_z={value}")
    return trace


def _write_lambda_run(
    model_context: dict,
    case_name: str,
    env,
    output_root: Path,
    grouping_top_fraction: float,
    max_deenergized_lines: int,
    lambda_case: str,
    lambda_R: float,
    lambda_L: float,
    candidate_line_ids: list[int],
    num_evaluated_subsets: int,
    evaluations: pd.DataFrame,
    stage_c_eval: dict | None,
    stage_c_line_ids: list[int],
    baseline_risk: float,
) -> dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    wildfire = model_context["wildfire"]
    automatic_artifacts = model_context["automatic_artifacts"]
    fixed_impact = model_context["fixed_impact"]
    consequence_df = model_context["consequence_df"]
    baseline_state = model_context["baseline_state"]
    line_to_group = _candidate_group_maps(wildfire)
    baseline_loading = np.asarray(baseline_state["loading_ratio"], dtype=float)

    valid_evaluations = evaluations[evaluations["status"] == "ok"].copy()
    best = select_best_subset(valid_evaluations, lambda_case, lambda_R, lambda_L)
    best_set = set(best.deenergized_line_ids)
    z_by_line = {line_id: (0 if line_id in best_set else 1) for line_id in range(len(baseline_loading))}
    if best.deenergized_line_ids:
        _prediction, best_state = predict_psps_state(
            scenario,
            model_context["runner"],
            decision_vector.u_base,
            best.deenergized_line_ids,
            standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
        )
    else:
        best_state = baseline_state
    best_loading = np.asarray(best_state["loading_ratio"], dtype=float)
    baseline_risk_by_total, baseline_risk_by_line = line_risk_total(
        baseline_loading,
        env.p_env_by_line,
        fixed_impact,
        {line_id: 1 for line_id in range(len(baseline_loading))},
        candidate_line_ids,
    )
    _best_risk_total, best_risk_by_line = line_risk_total(
        best_loading,
        env.p_env_by_line,
        fixed_impact,
        z_by_line,
        candidate_line_ids,
    )

    run_dir = make_run_dir(output_root, "run")
    write_config_copy(config, run_dir / "config.yaml")
    write_json(run_dir / "metadata.json", {**git_metadata(), "config_path": str(model_context["config_path"])})
    write_json(run_dir / "wildfire_scenario.json", wildfire.to_dict())
    write_dataframe(run_dir / "fixed_line_consequence_scores.csv", consequence_df)
    automatic_metadata = write_automatic_group_artifacts(run_dir, automatic_artifacts, baseline_risk, config)

    candidate_scores = _candidate_line_risk_scores(
        scenario,
        candidate_line_ids,
        line_to_group,
        env.p_env_by_line,
        baseline_loading,
        fixed_impact,
    )
    write_dataframe(run_dir / "candidate_line_risk_scores.csv", candidate_scores)
    eval_frame = _evaluation_to_csv_frame(evaluations, lambda_case, lambda_R, lambda_L)
    write_dataframe(run_dir / "deenergization_candidate_evaluations.csv", eval_frame)
    decisions = _decision_table(
        scenario,
        candidate_line_ids,
        best.deenergized_line_ids,
        line_to_group,
        env.p_env_by_line,
        fixed_impact,
        baseline_loading,
    )
    write_dataframe(run_dir / "optimized_deenergization_decisions.csv", decisions)
    write_dataframe(run_dir / "decision_vector_initial.csv", decision_vector.metadata_frame(decision_vector.u_base))
    write_dataframe(run_dir / "decision_vector_final.csv", decision_vector.metadata_frame(decision_vector.u_base))

    baseline_line_df, final_line_df = _line_risk_frames(
        scenario,
        candidate_line_ids,
        line_to_group,
        env.p_env_by_line,
        baseline_loading,
        best_loading,
        fixed_impact,
        baseline_risk_by_line,
        best_risk_by_line,
    )
    write_dataframe(run_dir / "risk_by_line_before_after.csv", baseline_line_df.merge(final_line_df, on="line_id", suffixes=("_before", "_after")))
    write_dataframe(
        run_dir / "risk_by_group_before_after.csv",
        _group_risk_frame(wildfire, baseline_risk_by_line).merge(
            _group_risk_frame(wildfire, best_risk_by_line),
            on="group_name",
            suffixes=("_before", "_after"),
        ),
    )
    trace = _objective_trace(eval_frame)
    write_dataframe(run_dir / "objective_trace.csv", trace)

    affected = affected_buses(scenario.edge_index, best.deenergized_line_ids)
    disconnected = disconnected_buses_from_reference(scenario.edge_index, scenario.num_buses, best.deenergized_line_ids)
    components = int(best["num_disconnected_components"]) if isinstance(best, dict) else disconnected_component_count(
        scenario.edge_index,
        scenario.num_buses,
        best.deenergized_line_ids,
    )
    stage_c_available = stage_c_eval is not None
    stage_c_objective = None
    objective_improvement = None
    risk_change = None
    load_change = None
    stage_c_risk = None
    stage_c_R_group = None
    stage_c_load = None
    if stage_c_eval is not None:
        stage_c_R_group = float(stage_c_eval["R_group"])
        stage_c_risk = float(stage_c_eval["R_norm"])
        stage_c_load = float(stage_c_eval["L_norm"])
        stage_c_objective = float(lambda_R * stage_c_risk + lambda_L * stage_c_load)
        objective_improvement = float(stage_c_objective - best.objective)
        risk_change = float(best.R_norm - stage_c_risk)
        load_change = float(best.L_norm - stage_c_load)

    summary = {
        "stage": "D",
        "evaluation_mode": EVALUATION_MODE,
        "max_deenergized_lines": int(max_deenergized_lines),
        "model_type": config.model.model_type.lower(),
        "environmental_risk_case": case_name,
        "lambda_case": lambda_case,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "grouping_top_fraction": float(grouping_top_fraction),
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_evaluated_subsets": int(num_evaluated_subsets),
        "best_deenergized_line_ids": [int(line_id) for line_id in best.deenergized_line_ids],
        "best_num_deenergized_lines": int(len(best.deenergized_line_ids)),
        "num_deenergized_lines": int(len(best.deenergized_line_ids)),
        "deenergized_line_ids": [int(line_id) for line_id in best.deenergized_line_ids],
        "best_R_group": float(best.R_group),
        "best_R_norm": float(best.R_norm),
        "best_L_norm": float(best.L_norm),
        "best_objective": float(best.objective),
        "best_demand_weighted_load_shed": float(best.demand_weighted_load_shed),
        "R_group": float(best.R_group),
        "R_norm": float(best.R_norm),
        "L_norm": float(best.L_norm),
        "objective": float(best.objective),
        "demand_weighted_load_shed": float(best.demand_weighted_load_shed),
        "risk_reduction_fraction": float(1.0 - best.R_norm),
        "num_groups": int(len(wildfire.line_groups)),
        "largest_group_id": env.largest_group_id,
        "largest_group_num_lines": int(len(env.largest_group_line_ids)),
        "manual_high_risk_group_id": env.manual_high_risk_group_id,
        "num_disconnected_components": int(components),
        "affected_bus_ids": [int(bus) for bus in affected],
        "disconnected_bus_ids": [int(bus) for bus in disconnected],
        "baseline_R_group": float(baseline_risk_by_total),
        "stage_c_comparison_available": bool(stage_c_available),
        "stage_c_deenergized_line_ids": [int(line_id) for line_id in stage_c_line_ids],
        "stage_c_objective": stage_c_objective,
        "stage_c_risk": stage_c_risk,
        "stage_c_R_group": stage_c_R_group,
        "stage_c_R_norm": stage_c_risk,
        "stage_c_demand_weighted_load_shed": stage_c_load,
        "stage_d_deenergized_line_ids": [int(line_id) for line_id in best.deenergized_line_ids],
        "stage_d_objective": float(best.objective),
        "stage_d_risk": float(best.R_norm),
        "stage_d_R_group": float(best.R_group),
        "stage_d_R_norm": float(best.R_norm),
        "stage_d_demand_weighted_load_shed": float(best.demand_weighted_load_shed),
        "objective_improvement_vs_stage_c": objective_improvement,
        "risk_change_vs_stage_c": risk_change,
        "load_shed_change_vs_stage_c": load_change,
        "run_dir": str(run_dir),
        "status": "ok",
        "error": "",
    }
    write_json(run_dir / "optimization_summary.json", summary)

    visualization_summary = {
        "stage": "D",
        "model_type": config.model.model_type.lower(),
        "environmental_risk_case": case_name,
        "lambda_case": lambda_case,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "grouping_top_fraction": float(grouping_top_fraction),
        "evaluation_mode": EVALUATION_MODE,
        "max_deenergized_lines": int(max_deenergized_lines),
        "group_ids": automatic_metadata.get("group_ids"),
        "group_line_ids": automatic_metadata.get("group_line_ids"),
        "largest_group_id": env.largest_group_id,
        "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
        "optimized_deenergized_line_ids": [int(line_id) for line_id in best.deenergized_line_ids],
        "num_optimized_deenergized_lines": int(len(best.deenergized_line_ids)),
        "affected_bus_ids": [int(bus) for bus in affected],
        "disconnected_bus_ids": [int(bus) for bus in disconnected],
        "num_disconnected_components": int(components),
        "stage_c_comparison_available": bool(stage_c_available),
        "stage_c_deenergized_line_ids": [int(line_id) for line_id in stage_c_line_ids],
        "optimization_behavior_plot": None,
        "topology_change_plot": None,
        "errors": [],
    }
    write_json(run_dir / "visualization_summary.json", visualization_summary)
    try:
        visualization_summary["optimization_behavior_plot"] = str(plot_optimization_behavior(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"optimization_behavior: {exc}")
    try:
        visualization_summary["topology_change_plot"] = str(plot_network_changes(run_dir))
    except Exception as exc:
        visualization_summary["errors"].append(f"topology_change: {exc}")
    write_json(run_dir / "visualization_summary.json", visualization_summary)
    write_json(run_dir / "optimization_summary.json", summary)
    return summary


def run_stage_d_deenergization(
    grouping_top_fraction: float = 0.30,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    lambda_cases: list[str] | None = None,
    evaluation_mode: str = EVALUATION_MODE,
    max_deenergized_lines: int = 2,
    clear: bool = False,
) -> Path:
    if evaluation_mode != EVALUATION_MODE:
        raise ValueError(f"Stage D v1 only supports evaluation_mode={EVALUATION_MODE}.")
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    lambda_cases = list(LAMBDA_CASES) if lambda_cases is None else lambda_cases
    root = _stage_d_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected_parent = (REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results").resolve()
        if resolved_root.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    stage_c_summary = _load_stage_c_summary()
    threshold_root = root / compact_fraction_label("t", grouping_top_fraction)
    rows = []
    for model_type in models:
        try:
            context = _build_model_context(model_type, grouping_top_fraction)
            config = context["config"]
            wildfire = context["wildfire"]
            fixed_impact = context["fixed_impact"]
            baseline_state = context["baseline_state"]
            group_summary = context["automatic_artifacts"]["group_summary"]
            candidate_line_ids = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
            num_candidate_lines = int(len(candidate_line_ids))
            num_evaluated_subsets = expected_subset_count(num_candidate_lines, max_deenergized_lines=max_deenergized_lines)
            if num_candidate_lines <= 0:
                raise ValueError("Stage D candidate line set is empty.")
            baseline_loading = np.asarray(baseline_state["loading_ratio"], dtype=float)
            for case_name in cases:
                env = apply_environmental_case(wildfire, group_summary, case_name)
                baseline_risk, _baseline_risk_by_line = line_risk_total(
                    baseline_loading,
                    env.p_env_by_line,
                    fixed_impact,
                    {line_id: 1 for line_id in range(len(baseline_loading))},
                    candidate_line_ids,
                )
                if baseline_risk <= 1e-12:
                    raise ValueError(f"Baseline grouped risk is zero for {model_type}/{case_name}.")
                evaluations = _evaluate_all_subsets(
                    context,
                    candidate_line_ids,
                    env.p_env_by_line,
                    fixed_impact,
                    baseline_risk,
                    max_deenergized_lines,
                )
                stage_c_row = _stage_c_comparison_row(stage_c_summary, model_type, case_name, grouping_top_fraction)
                stage_c_line_ids = _parse_line_ids(stage_c_row["deenergized_line_ids"]) if stage_c_row else []
                stage_c_eval = (
                    _evaluate_subset(
                        context,
                        candidate_line_ids,
                        env.p_env_by_line,
                        fixed_impact,
                        baseline_risk,
                        tuple(stage_c_line_ids),
                        eval_id=-1,
                    )
                    if stage_c_line_ids
                    else None
                )
                for lambda_case in lambda_cases:
                    lambda_R, lambda_L = LAMBDA_CASES[lambda_case]
                    output_root = (
                        threshold_root
                        / model_type
                        / CASE_FOLDERS.get(case_name, case_name)
                        / LAMBDA_FOLDERS.get(lambda_case, lambda_case)
                    )
                    rows.append(
                        _write_lambda_run(
                            context,
                            case_name,
                            env,
                            output_root,
                            grouping_top_fraction,
                            max_deenergized_lines,
                            lambda_case,
                            lambda_R,
                            lambda_L,
                            candidate_line_ids,
                            num_evaluated_subsets,
                            evaluations,
                            stage_c_eval if stage_c_eval and stage_c_eval.get("status") == "ok" else None,
                            stage_c_line_ids,
                            baseline_risk,
                        )
                    )
        except Exception as exc:
            for case_name in cases:
                for lambda_case in lambda_cases:
                    lambda_R, lambda_L = LAMBDA_CASES[lambda_case]
                    rows.append(
                        {
                            "model_type": model_type,
                            "environmental_risk_case": case_name,
                            "lambda_case": lambda_case,
                            "lambda_R": lambda_R,
                            "lambda_L": lambda_L,
                            "grouping_top_fraction": float(grouping_top_fraction),
                            "evaluation_mode": evaluation_mode,
                            "max_deenergized_lines": int(max_deenergized_lines),
                            "status": "failed",
                            "error": str(exc),
                            "run_dir": "",
                        }
                    )

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "stage_d_deenergization_summary.csv"
    summary_json = root / "stage_d_deenergization_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "grouping_top_fraction": float(grouping_top_fraction),
            "evaluation_mode": EVALUATION_MODE,
            "max_deenergized_lines": int(max_deenergized_lines),
            "lambda_cases": {case: {"lambda_R": LAMBDA_CASES[case][0], "lambda_L": LAMBDA_CASES[case][1]} for case in lambda_cases},
            "models": models,
            "cases": cases,
            "summary_csv": str(summary_csv),
        },
    )
    print(f"[OK] Stage D de-energization summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--lambda-cases", nargs="+", choices=sorted(LAMBDA_CASES), default=list(LAMBDA_CASES))
    parser.add_argument("--evaluation-mode", default=EVALUATION_MODE, choices=[EVALUATION_MODE])
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--clear", action="store_true", help="Delete existing results/stage_d_deenergization first.")
    args = parser.parse_args()
    run_stage_d_deenergization(
        grouping_top_fraction=args.grouping_top_fraction,
        models=args.models,
        cases=args.cases,
        lambda_cases=args.lambda_cases,
        evaluation_mode=args.evaluation_mode,
        max_deenergized_lines=args.max_deenergized_lines,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
