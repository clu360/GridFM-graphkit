from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.plot_network_changes import plot_network_changes
from experiments.test.wildfire_tests.shared.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_setup import write_automatic_group_artifacts
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    CASE_FOLDERS,
    CASES,
    MODEL_CONFIGS,
    _build_model_context,
    _candidate_group_maps,
    compact_fraction_label,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    affected_buses,
    apply_environmental_case,
    demand_weighted_load_shed_from_prediction,
    disconnected_component_count,
    predict_psps_state,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    disconnected_buses_from_reference,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    GurobiUnavailableError,
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    METHOD_NAME,
    STAGE_E_LAMBDA_CASES,
    STAGE_E_LAMBDA_FOLDERS,
    compute_proxy_metrics,
    compute_true_metrics,
    deenergized_from_y,
    normalize_true_exposure,
    vector_string,
    z_from_y,
)


EVALUATION_MODE = "gurobi_master_gridfm_fixed_u"


def _stage_e_root() -> Path:
    return RESULTS_ROOT / "leq" / "stage_e" / "gurobi_gridfm"


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _line_consequence_frame(consequence_df: pd.DataFrame) -> pd.DataFrame:
    frame = consequence_df.copy()
    if "c_l" not in frame.columns:
        source = "I_l" if "I_l" in frame.columns else "service_loss_fraction"
        frame["c_l"] = frame[source].astype(float)
    return frame


def _consequence_by_line(consequence_df: pd.DataFrame) -> dict[int, float]:
    frame = _line_consequence_frame(consequence_df)
    return {int(row["line_id"]): float(row["c_l"]) for _, row in frame.iterrows()}


def _candidate_score_frame(scenario, candidate_line_ids, line_to_group, p_env_by_line, baseline_loading, c_by_line, proxy_type):
    edge_array = _edge_array(scenario)
    candidate_set = {int(line_id) for line_id in candidate_line_ids}
    rows = []
    for line_id in range(int(edge_array.shape[1])):
        p_env = float(p_env_by_line.get(line_id, 0.0))
        loading = float(baseline_loading[line_id])
        rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "group_id": line_to_group.get(line_id, ""),
                "p_env": p_env,
                "loading_base": loading,
                "c_l": float(c_by_line.get(line_id, 0.0)),
                "proxy_type": proxy_type,
                "proxy_R_coeff_env_loading_base": float(p_env * loading**2),
                "candidate_for_deenergization": bool(line_id in candidate_set),
            }
        )
    return pd.DataFrame(rows)


def _decision_table(scenario, candidate_line_ids, best_y, line_to_group, p_env_by_line, baseline_loading, c_by_line):
    edge_array = _edge_array(scenario)
    candidate_set = {int(line_id) for line_id in candidate_line_ids}
    rows = []
    for line_id in range(int(edge_array.shape[1])):
        y_l = int(best_y.get(line_id, 0))
        rows.append(
            {
                "line_id": int(line_id),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "group_id": line_to_group.get(line_id, ""),
                "p_env": float(p_env_by_line.get(line_id, 0.0)),
                "loading_base": float(baseline_loading[line_id]),
                "c_l": float(c_by_line.get(line_id, 0.0)),
                "candidate_for_deenergization": bool(line_id in candidate_set),
                "gurobi_y_l": y_l,
                "optimized_z_l": int(1 - y_l),
                "deenergized": bool(y_l == 1),
                "selected_in_best_solution": bool(y_l == 1),
            }
        )
    return pd.DataFrame(rows)


def _risk_by_line_before_after(scenario, candidate_line_ids, line_to_group, p_env_by_line, baseline_loading, final_loading, baseline_by_line, final_by_line):
    edge_array = _edge_array(scenario)
    candidate_set = {int(line_id) for line_id in candidate_line_ids}
    rows = []
    for line_id in range(int(edge_array.shape[1])):
        before = {
            "line_id": int(line_id),
            "from_bus_before": int(edge_array[0, line_id]),
            "to_bus_before": int(edge_array[1, line_id]),
            "loading_ratio_before": float(baseline_loading[line_id]),
            "hazard_before": float(p_env_by_line.get(line_id, 0.0)),
            "risk_before": float(baseline_by_line.get(line_id, 0.0)),
            "group_id_before": line_to_group.get(line_id, ""),
            "selected_candidate_high_risk_before": bool(line_id in candidate_set),
        }
        after = {
            "from_bus_after": int(edge_array[0, line_id]),
            "to_bus_after": int(edge_array[1, line_id]),
            "loading_ratio_after": float(final_loading[line_id]),
            "hazard_after": float(p_env_by_line.get(line_id, 0.0)),
            "risk_after": float(final_by_line.get(line_id, 0.0)),
            "group_id_after": line_to_group.get(line_id, ""),
            "selected_candidate_high_risk_after": bool(line_id in candidate_set),
        }
        rows.append({**before, **after})
    return pd.DataFrame(rows)


def _group_risk_frame(wildfire, risk_by_line: dict[int, float], suffix: str) -> pd.DataFrame:
    rows = []
    for group in wildfire.line_groups:
        raw = float(sum(risk_by_line.get(int(line_id), 0.0) for line_id in group.line_ids))
        rows.append(
            {
                "group_name": group.name,
                f"line_ids_{suffix}": ",".join(str(int(line_id)) for line_id in group.line_ids),
                f"group_weight_{suffix}": float(group.group_weight),
                f"raw_group_risk_{suffix}": raw,
                f"weighted_group_risk_{suffix}": float(raw * group.group_weight),
            }
        )
    return pd.DataFrame(rows)


def _objective_trace(eval_frame: pd.DataFrame) -> pd.DataFrame:
    trace = eval_frame.rename(columns={"iteration": "eval_idx"}).copy()
    trace["objective_total"] = trace["true_objective"]
    trace["wildfire_group_risk"] = trace["true_R_raw"]
    trace["load_shedding"] = trace["true_L_shed"]
    trace["normalized_wildfire_group_risk"] = trace["true_R_norm"]
    trace["normalized_load_shedding"] = trace["true_L_shed"]
    trace["risk_objective_term"] = trace["lambda_R_true"] * trace["true_R_norm"]
    trace["load_shedding_objective_term"] = trace["lambda_L_true"] * trace["true_L_shed"]
    trace["generator_movement"] = 0.0
    trace["max_loading_ratio"] = np.nan
    trace["mean_alpha"] = 1.0
    trace["max_abs_delta_pg"] = 0.0
    trace["message"] = trace["deenergized_line_ids"].apply(lambda value: f"gurobi_y={value}")
    return trace


def _evaluate_candidate(
    model_context: dict,
    candidate_line_ids: list[int],
    p_env_by_line: dict[int, float],
    baseline_R_raw_new: float,
    baseline_loading: np.ndarray,
    c_by_line: dict[int, float],
    y_by_line: dict[int, int],
    lambda_case_name: str,
    lambda_R: float,
    lambda_L: float,
    lambda_P: float,
    proxy_type: str,
    iteration: int,
    best_objective_so_far: float,
    gridfm_calls_before: int,
    baseline_L_shed: float,
    start_time: float,
) -> tuple[dict, dict, dict | None]:
    config = model_context["config"]
    scenario = model_context["scenario"]
    runner = model_context["runner"]
    decision_vector = model_context["decision_vector"]
    baseline_prediction = model_context["baseline_prediction"]
    baseline_state = model_context["baseline_state"]
    num_lines = int(scenario.edge_index.shape[1])
    deenergized = deenergized_from_y(y_by_line)
    z_by_line = z_from_y(y_by_line, num_lines)
    proxy = compute_proxy_metrics(
        candidate_line_ids,
        p_env_by_line,
        baseline_loading,
        c_by_line,
        y_by_line,
        lambda_R,
        lambda_L,
        proxy_type=proxy_type,
    )
    status = "ok"
    error_message = ""
    prediction = baseline_prediction
    state = baseline_state
    gridfm_calls = int(gridfm_calls_before)
    try:
        if deenergized:
            prediction, state = predict_psps_state(
                scenario,
                runner,
                decision_vector.u_base,
                deenergized,
                standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
            )
            gridfm_calls += 1
        loading = np.asarray(state["loading_ratio"], dtype=float)
        true_L_shed = demand_weighted_load_shed_from_prediction(prediction, scenario) if deenergized else 0.0
        true = compute_true_metrics(
            loading,
            p_env_by_line,
            z_by_line,
            candidate_line_ids,
            baseline_R_raw_new,
            true_L_shed,
            true_P_AC=0.0,
            lambda_R_true=lambda_R,
            lambda_L_true=lambda_L,
            lambda_P=lambda_P,
        )
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        loading = np.asarray(baseline_state["loading_ratio"], dtype=float)
        true = {
            "true_R_raw": np.nan,
            "true_R_norm": np.nan,
            "true_L_shed": np.nan,
            "true_P_AC": np.nan,
            "true_objective": np.nan,
            "true_exposure_by_line": {},
        }
    improved = bool(status == "ok" and float(true["true_objective"]) < float(best_objective_so_far))
    current_best = float(true["true_objective"]) if improved else float(best_objective_so_far)
    row = {
        "iteration": int(iteration),
        "method_name": METHOD_NAME,
        "lambda_case_name": lambda_case_name,
        "lambda_R_master": float(lambda_R),
        "lambda_L_master": float(lambda_L),
        "lambda_R_true": float(lambda_R),
        "lambda_L_true": float(lambda_L),
        "lambda_P": float(lambda_P),
        "model_type": config.model.model_type.lower(),
        "candidate_lines": [int(line_id) for line_id in candidate_line_ids],
        "K": int(sum(y_by_line.values()) if False else len([line_id for line_id in candidate_line_ids if y_by_line.get(line_id, 0) in (0, 1)])),
        "proxy_type": proxy_type,
        "y_vector": vector_string(candidate_line_ids, y_by_line),
        "z_vector": vector_string(candidate_line_ids, {line_id: 1 - int(y_by_line.get(line_id, 0)) for line_id in candidate_line_ids}),
        "deenergized_line_ids": [int(line_id) for line_id in deenergized],
        "energized_line_ids": [int(line_id) for line_id in candidate_line_ids if int(y_by_line.get(line_id, 0)) == 0],
        "num_deenergized_lines": int(len(deenergized)),
        **{key: proxy[key] for key in ["proxy_R_hat", "proxy_L_hat", "proxy_objective"]},
        **{key: true[key] for key in ["true_R_raw", "true_R_norm", "true_L_shed", "true_P_AC", "true_objective"]},
        "baseline_R_raw_new": float(baseline_R_raw_new),
        "baseline_R_norm": 1.0,
        "baseline_L_shed": float(baseline_L_shed),
        "baseline_P_AC": 0.0,
        "is_best_so_far": improved,
        "best_objective_so_far": current_best,
        "num_gridfm_calls": int(gridfm_calls),
        "runtime_seconds": float(time.perf_counter() - start_time),
        "status": status,
        "error_flag": bool(status != "ok"),
        "error_message": error_message,
        "affected_bus_ids": [int(bus) for bus in affected_buses(scenario.edge_index, deenergized)],
        "disconnected_bus_ids": [
            int(bus) for bus in disconnected_buses_from_reference(scenario.edge_index, scenario.num_buses, deenergized)
        ],
        "num_disconnected_components": int(disconnected_component_count(scenario.edge_index, scenario.num_buses, deenergized)),
    }
    return row, true["true_exposure_by_line"], (state if status == "ok" else None)


def _run_lambda_case(
    model_context: dict,
    case_name: str,
    env,
    output_root: Path,
    grouping_top_fraction: float,
    max_deenergized_lines: int,
    evaluation_budget: int,
    lambda_case_name: str,
    lambda_R: float,
    lambda_L: float,
    proxy_type: str,
) -> dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    wildfire = model_context["wildfire"]
    automatic_artifacts = model_context["automatic_artifacts"]
    consequence_df = _line_consequence_frame(model_context["consequence_df"])
    c_by_line = _consequence_by_line(consequence_df)
    baseline_prediction = model_context["baseline_prediction"]
    baseline_state = model_context["baseline_state"]
    baseline_loading = np.asarray(baseline_state["loading_ratio"], dtype=float)
    candidate_line_ids = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
    if not candidate_line_ids:
        raise ValueError("Stage E candidate line set is empty.")
    baseline_z = {line_id: 1 for line_id in range(len(baseline_loading))}
    baseline_R_raw_new, baseline_by_line = compute_operational_wildfire_exposure(
        baseline_loading,
        env.p_env_by_line,
        baseline_z,
        candidate_line_ids,
    )
    normalize_true_exposure(baseline_R_raw_new, baseline_R_raw_new)
    baseline_L_shed = demand_weighted_load_shed_from_prediction(baseline_prediction, scenario)
    line_to_group = _candidate_group_maps(wildfire)
    config.objective.lambda_R = float(lambda_R)
    config.objective.lambda_L = float(lambda_L)

    run_dir = make_run_dir(output_root, "run")
    write_config_copy(config, run_dir / "config.yaml")
    write_json(run_dir / "metadata.json", {**git_metadata(), "config_path": str(model_context["config_path"])})
    write_json(run_dir / "wildfire_scenario.json", wildfire.to_dict())
    write_dataframe(run_dir / "line_consequence_scores.csv", consequence_df)
    automatic_metadata = write_automatic_group_artifacts(run_dir, automatic_artifacts, baseline_R_raw_new, config)
    write_dataframe(
        run_dir / "candidate_line_risk_scores.csv",
        _candidate_score_frame(scenario, candidate_line_ids, line_to_group, env.p_env_by_line, baseline_loading, c_by_line, proxy_type),
    )
    baseline_metrics = {
        "baseline_R_raw_new": float(baseline_R_raw_new),
        "baseline_R_norm": 1.0,
        "baseline_L_shed": float(baseline_L_shed),
        "baseline_P_AC": 0.0,
        "baseline_loading_by_line": {str(idx): float(value) for idx, value in enumerate(baseline_loading)},
        "p_env_by_line": {str(line_id): float(value) for line_id, value in env.p_env_by_line.items()},
        "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
        "risk_scope": "candidate_group_scope",
        "risk_formula": "sum_l z_l * p_env_l * loading_l^2",
    }
    write_json(run_dir / "baseline_metrics.json", baseline_metrics)

    rows = []
    evaluated_y = []
    best_objective = float("inf")
    best_y = {line_id: 0 for line_id in candidate_line_ids}
    best_exposure_by_line = baseline_by_line
    best_loading = baseline_loading
    gridfm_calls = 0
    start_time = time.perf_counter()
    for iteration in range(int(evaluation_budget)):
        try:
            proposal = solve_gurobi_master_next_candidate(
                candidate_line_ids,
                env.p_env_by_line,
                baseline_loading,
                c_by_line,
                lambda_R,
                lambda_L,
                max_deenergized_lines=max_deenergized_lines,
                evaluated_y_vectors=evaluated_y,
                proxy_type=proxy_type,
            )
        except Exception as exc:
            if iteration == 0:
                raise
            rows.append(
                {
                    "iteration": int(iteration),
                    "method_name": METHOD_NAME,
                    "lambda_case_name": lambda_case_name,
                    "lambda_R_master": float(lambda_R),
                    "lambda_L_master": float(lambda_L),
                    "lambda_R_true": float(lambda_R),
                    "lambda_L_true": float(lambda_L),
                    "lambda_P": 0.0,
                    "model_type": config.model.model_type.lower(),
                    "candidate_lines": [int(line_id) for line_id in candidate_line_ids],
                    "K": int(max_deenergized_lines),
                    "proxy_type": proxy_type,
                    "status": "exhausted_or_failed",
                    "error_flag": True,
                    "error_message": str(exc),
                    "runtime_seconds": float(time.perf_counter() - start_time),
                    "num_gridfm_calls": int(gridfm_calls),
                }
            )
            break
        y_by_line = proposal["y_by_line"]
        row, exposure_by_line, state = _evaluate_candidate(
            model_context,
            candidate_line_ids,
            env.p_env_by_line,
            baseline_R_raw_new,
            baseline_loading,
            c_by_line,
            y_by_line,
            lambda_case_name,
            lambda_R,
            lambda_L,
            lambda_P=0.0,
            proxy_type=proxy_type,
            iteration=iteration,
            best_objective_so_far=best_objective,
            gridfm_calls_before=gridfm_calls,
            baseline_L_shed=baseline_L_shed,
            start_time=start_time,
        )
        row["K"] = int(max_deenergized_lines)
        gridfm_calls = int(row["num_gridfm_calls"])
        if row["status"] == "ok" and float(row["true_objective"]) < best_objective:
            best_objective = float(row["true_objective"])
            best_y = dict(y_by_line)
            best_exposure_by_line = exposure_by_line
            best_loading = np.asarray(state["loading_ratio"], dtype=float) if state is not None else best_loading
            row["is_best_so_far"] = True
            row["best_objective_so_far"] = best_objective
        rows.append(row)
        evaluated_y.append(dict(y_by_line))

    eval_frame = pd.DataFrame(rows)
    write_dataframe(run_dir / "gurobi_candidate_evaluations.csv", eval_frame)
    if len(eval_frame):
        write_dataframe(run_dir / "objective_trace.csv", _objective_trace(eval_frame[eval_frame["status"] == "ok"].copy()))
    else:
        write_dataframe(run_dir / "objective_trace.csv", pd.DataFrame())
    write_dataframe(
        run_dir / "gurobi_best_decision.csv",
        _decision_table(scenario, candidate_line_ids, best_y, line_to_group, env.p_env_by_line, baseline_loading, c_by_line),
    )
    write_dataframe(
        run_dir / "optimized_deenergization_decisions.csv",
        _decision_table(scenario, candidate_line_ids, best_y, line_to_group, env.p_env_by_line, baseline_loading, c_by_line),
    )
    write_dataframe(
        run_dir / "risk_by_line_before_after.csv",
        _risk_by_line_before_after(
            scenario,
            candidate_line_ids,
            line_to_group,
            env.p_env_by_line,
            baseline_loading,
            best_loading,
            baseline_by_line,
            best_exposure_by_line,
        ),
    )
    write_dataframe(
        run_dir / "risk_by_group_before_after.csv",
        _group_risk_frame(wildfire, baseline_by_line, "before").merge(
            _group_risk_frame(wildfire, best_exposure_by_line, "after"),
            on="group_name",
        ),
    )
    write_dataframe(run_dir / "decision_vector_initial.csv", decision_vector.metadata_frame(decision_vector.u_base))
    write_dataframe(run_dir / "decision_vector_final.csv", decision_vector.metadata_frame(decision_vector.u_base))

    ok = eval_frame[eval_frame["status"] == "ok"].copy() if len(eval_frame) else pd.DataFrame()
    best_row = ok.sort_values("true_objective", kind="mergesort").iloc[0].to_dict() if len(ok) else {}
    best_deenergized = best_row.get("deenergized_line_ids", [])
    summary = {
        "stage": "E",
        "method_name": METHOD_NAME,
        "evaluation_mode": EVALUATION_MODE,
        "model_type": config.model.model_type.lower(),
        "environmental_risk_case": case_name,
        "lambda_case": lambda_case_name,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "lambda_R_master": float(lambda_R),
        "lambda_L_master": float(lambda_L),
        "lambda_R_true": float(lambda_R),
        "lambda_L_true": float(lambda_L),
        "lambda_P": 0.0,
        "grouping_top_fraction": float(grouping_top_fraction),
        "max_deenergized_lines": int(max_deenergized_lines),
        "K": int(max_deenergized_lines),
        "evaluation_budget": int(evaluation_budget),
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_evaluated_candidates": int(len(ok)),
        "num_gridfm_calls": int(gridfm_calls),
        "proxy_type": proxy_type,
        "best_deenergized_line_ids": best_deenergized,
        "best_num_deenergized_lines": int(len(best_deenergized)) if isinstance(best_deenergized, list) else 0,
        "best_true_R_raw": best_row.get("true_R_raw"),
        "best_true_R_norm": best_row.get("true_R_norm"),
        "best_true_L_shed": best_row.get("true_L_shed"),
        "best_true_P_AC": best_row.get("true_P_AC"),
        "best_true_objective": best_row.get("true_objective"),
        "baseline_R_raw_new": float(baseline_R_raw_new),
        "baseline_R_norm": 1.0,
        "baseline_L_shed": float(baseline_L_shed),
        "baseline_P_AC": 0.0,
        "risk_scope": "candidate_group_scope",
        "true_risk_formula": "sum_l z_l * p_env_l * loading_l^2",
        "gurobi_role": "proxy topology candidate generator",
        "gridfm_role": "true post-topology evaluator",
        "run_dir": str(run_dir),
        "status": "ok" if len(ok) else "failed",
        "error": "" if len(ok) else "No successful candidate evaluations.",
    }
    write_json(run_dir / "optimization_summary.json", summary)
    visualization_summary = {
        "stage": "E",
        "method_name": METHOD_NAME,
        "model_type": config.model.model_type.lower(),
        "environmental_risk_case": case_name,
        "lambda_case": lambda_case_name,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "grouping_top_fraction": float(grouping_top_fraction),
        "evaluation_mode": EVALUATION_MODE,
        "max_deenergized_lines": int(max_deenergized_lines),
        "group_ids": automatic_metadata.get("group_ids"),
        "group_line_ids": automatic_metadata.get("group_line_ids"),
        "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
        "optimized_deenergized_line_ids": best_deenergized,
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


def run_stage_e_gurobi_gridfm(
    grouping_top_fraction: float = 0.30,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    lambda_cases: list[str] | None = None,
    max_deenergized_lines: list[int] | None = None,
    evaluation_budget: int = 20,
    proxy_type: str = DEFAULT_PROXY_TYPE,
    clear: bool = False,
) -> Path:
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    lambda_cases = list(STAGE_E_LAMBDA_CASES) if lambda_cases is None else lambda_cases
    max_deenergized_lines = [1, 2] if max_deenergized_lines is None else [int(item) for item in max_deenergized_lines]
    root = _stage_e_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected = (RESULTS_ROOT / "leq" / "stage_e").resolve()
        if resolved_root.parent != expected:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    rows = []
    threshold_root = root / compact_fraction_label("t", grouping_top_fraction)
    for model_type in models:
        try:
            context = _build_model_context(model_type, grouping_top_fraction)
            wildfire = context["wildfire"]
            group_summary = context["automatic_artifacts"]["group_summary"]
            for case_name in cases:
                env = apply_environmental_case(wildfire, group_summary, case_name)
                for k in max_deenergized_lines:
                    for lambda_case_name in lambda_cases:
                        lambda_R, lambda_L = STAGE_E_LAMBDA_CASES[lambda_case_name]
                        output_root = (
                            threshold_root
                            / f"k{int(k)}"
                            / model_type
                            / CASE_FOLDERS.get(case_name, case_name)
                            / STAGE_E_LAMBDA_FOLDERS.get(lambda_case_name, lambda_case_name)
                        )
                        rows.append(
                            _run_lambda_case(
                                context,
                                case_name,
                                env,
                                output_root,
                                grouping_top_fraction,
                                int(k),
                                int(evaluation_budget),
                                lambda_case_name,
                                lambda_R,
                                lambda_L,
                                proxy_type,
                            )
                        )
        except Exception as exc:
            for case_name in cases:
                for k in max_deenergized_lines:
                    for lambda_case_name in lambda_cases:
                        lambda_R, lambda_L = STAGE_E_LAMBDA_CASES[lambda_case_name]
                        rows.append(
                            {
                                "stage": "E",
                                "method_name": METHOD_NAME,
                                "model_type": model_type,
                                "environmental_risk_case": case_name,
                                "lambda_case": lambda_case_name,
                                "lambda_R_master": lambda_R,
                                "lambda_L_master": lambda_L,
                                "lambda_R_true": lambda_R,
                                "lambda_L_true": lambda_L,
                                "lambda_P": 0.0,
                                "grouping_top_fraction": float(grouping_top_fraction),
                                "max_deenergized_lines": int(k),
                                "K": int(k),
                                "evaluation_budget": int(evaluation_budget),
                                "proxy_type": proxy_type,
                                "status": "failed",
                                "error": str(exc),
                                "run_dir": "",
                            }
                        )

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "stage_e_gurobi_gridfm_summary.csv"
    summary_json = root / "stage_e_gurobi_gridfm_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "grouping_top_fraction": float(grouping_top_fraction),
            "evaluation_mode": EVALUATION_MODE,
            "max_deenergized_lines": [int(item) for item in max_deenergized_lines],
            "evaluation_budget": int(evaluation_budget),
            "proxy_type": proxy_type,
            "lambda_cases": {
                case: {"lambda_R": STAGE_E_LAMBDA_CASES[case][0], "lambda_L": STAGE_E_LAMBDA_CASES[case][1]}
                for case in lambda_cases
            },
            "models": models,
            "cases": cases,
            "summary_csv": str(summary_csv),
        },
    )
    proxy_summary = summary_frame[
        [
            col
            for col in [
                "model_type",
                "environmental_risk_case",
                "lambda_case",
                "K",
                "num_evaluated_candidates",
                "num_gridfm_calls",
                "best_true_R_norm",
                "best_true_L_shed",
                "best_true_objective",
                "run_dir",
                "status",
            ]
            if col in summary_frame.columns
        ]
    ].copy()
    write_dataframe(root / "proxy_vs_true_ranking_summary.csv", proxy_summary)
    runtime_summary = summary_frame[
        [col for col in ["model_type", "environmental_risk_case", "lambda_case", "K", "num_gridfm_calls", "run_dir", "status"] if col in summary_frame.columns]
    ].copy()
    write_dataframe(root / "runtime_summary.csv", runtime_summary)
    print(f"[OK] Stage E Gurobi/GridFM summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--lambda-cases", nargs="+", choices=sorted(STAGE_E_LAMBDA_CASES), default=list(STAGE_E_LAMBDA_CASES))
    parser.add_argument("--max-deenergized-lines", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--evaluation-budget", type=int, default=20)
    parser.add_argument("--proxy-type", choices=["env_loading_base", "env_only"], default=DEFAULT_PROXY_TYPE)
    parser.add_argument("--clear", action="store_true", help="Delete existing results/leq/stage_e/gurobi_gridfm first.")
    args = parser.parse_args()
    run_stage_e_gurobi_gridfm(
        grouping_top_fraction=args.grouping_top_fraction,
        models=args.models,
        cases=args.cases,
        lambda_cases=args.lambda_cases,
        max_deenergized_lines=args.max_deenergized_lines,
        evaluation_budget=args.evaluation_budget,
        proxy_type=args.proxy_type,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
