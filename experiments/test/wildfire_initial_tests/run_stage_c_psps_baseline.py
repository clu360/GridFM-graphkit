from __future__ import annotations

import argparse
import json
import shutil
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
from experiments.test.wildfire_initial_tests.plot_network_changes import plot_network_changes
from experiments.test.wildfire_initial_tests.plot_optimization_behavior import plot_optimization_behavior
from experiments.test.wildfire_initial_tests.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_initial_tests.scenario import load_first_pass_context
from experiments.test.wildfire_initial_tests.stage_c_psps import (
    affected_buses,
    apply_environmental_case,
    compute_fixed_line_consequence_scores,
    compute_line_risk_with_z,
    demand_weighted_load_shed_from_prediction,
    disconnected_component_count,
    predict_psps_state,
    psps_line_count,
    select_psps_lines,
)
from experiments.test.wildfire_initial_tests.state_extraction import extract_state_quantities
from experiments.test.wildfire_initial_tests.wildfire_setup import (
    build_wildfire_for_baseline,
    write_automatic_group_artifacts,
)


MODEL_CONFIGS = {
    "gps": "automatic_multigroup_gps.yaml",
    "gnn": "automatic_multigroup_gnn.yaml",
}

CASES = ["auto_env", "largest_group_high"]
CASE_FOLDERS = {
    "auto_env": "auto",
    "largest_group_high": "lgh",
}
STAGE_C_LAMBDA_R = 0.999001
STAGE_C_LAMBDA_L = 0.000999


def fraction_label(prefix: str, value: float) -> str:
    text = f"{float(value):.2f}" if abs(float(value) * 100.0 - round(float(value) * 100.0)) < 1e-9 else f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{prefix}_" + text.replace(".", "p")


def compact_fraction_label(prefix: str, value: float) -> str:
    text = f"{float(value):.2f}" if abs(float(value) * 100.0 - round(float(value) * 100.0)) < 1e-9 else f"{float(value):.3f}".rstrip("0").rstrip(".")
    return f"{prefix}" + text.replace(".", "p")


def _stage_c_root() -> Path:
    return REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results" / "stage_c_psps"


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


def _edge_array(scenario) -> np.ndarray:
    return scenario.edge_index.cpu().numpy() if hasattr(scenario.edge_index, "cpu") else np.asarray(scenario.edge_index)


def _candidate_group_maps(wildfire):
    line_to_group = {}
    for group in wildfire.line_groups:
        for line_id in group.line_ids:
            line_to_group[int(line_id)] = group.name
    return line_to_group


def _line_risk_frames(
    scenario,
    candidate_line_ids,
    line_to_group,
    p_env_by_line,
    baseline_loading,
    post_loading,
    fixed_impact,
    baseline_risk_by_line,
    post_risk_by_line,
):
    edge_array = _edge_array(scenario)
    rows_before = []
    rows_after = []
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    for line_id in range(int(edge_array.shape[1])):
        base_row = {
            "line_id": int(line_id),
            "loading_ratio": float(baseline_loading[line_id]),
            "hazard": float(p_env_by_line.get(line_id, 0.0)),
            "impact": float(fixed_impact[line_id]),
            "risk": float(baseline_risk_by_line.get(line_id, 0.0)),
            "group_id": line_to_group.get(line_id, ""),
            "selected_candidate_high_risk": bool(line_id in candidate_set),
        }
        after_row = dict(base_row)
        after_row["loading_ratio"] = float(post_loading[line_id])
        after_row["risk"] = float(post_risk_by_line.get(line_id, 0.0))
        rows_before.append(base_row)
        rows_after.append(after_row)
    return pd.DataFrame(rows_before), pd.DataFrame(rows_after)


def _group_risk_frame(wildfire, risk_by_line: dict[int, float]) -> pd.DataFrame:
    rows = []
    for group in wildfire.line_groups:
        raw = float(sum(risk_by_line.get(int(line_id), 0.0) for line_id in group.line_ids))
        rows.append(
            {
                "group_name": group.name,
                "line_ids": ",".join(str(int(line_id)) for line_id in group.line_ids),
                "group_weight": float(group.group_weight),
                "raw_group_risk": raw,
                "weighted_group_risk": float(raw * group.group_weight),
            }
        )
    return pd.DataFrame(rows)


def _before_after(before: pd.DataFrame, after: pd.DataFrame, key: str) -> pd.DataFrame:
    return before.merge(after, on=key, suffixes=("_before", "_after"))


def _psps_tables(
    scenario,
    candidate_line_ids,
    deenergized_line_ids,
    line_to_group,
    p_env_by_line,
    baseline_loading,
    fixed_impact,
    baseline_psps_risk,
    grouping_top_fraction,
    psps_top_fraction,
):
    edge_array = _edge_array(scenario)
    candidate_set = set(int(line_id) for line_id in candidate_line_ids)
    deenergized_set = set(int(line_id) for line_id in deenergized_line_ids)
    num_candidate = len(candidate_set)
    num_psps = len(deenergized_set)
    realized_psps_fraction = float(num_psps / num_candidate) if num_candidate else 0.0
    rank_by_line = {
        int(line_id): rank
        for rank, line_id in enumerate(
            sorted(candidate_set, key=lambda item: (-float(baseline_psps_risk[item]), int(item))),
            start=1,
        )
    }
    score_rows = []
    decision_rows = []
    for line_id in range(int(edge_array.shape[1])):
        is_candidate = line_id in candidate_set
        deenergized = line_id in deenergized_set
        row = {
            "line_id": int(line_id),
            "from_bus": int(edge_array[0, line_id]),
            "to_bus": int(edge_array[1, line_id]),
            "group_id": line_to_group.get(line_id, ""),
            "p_env": float(p_env_by_line.get(line_id, 0.0)),
            "loading_base": float(baseline_loading[line_id]),
            "I_l": float(fixed_impact[line_id]),
            "baseline_psps_risk": float(baseline_psps_risk.get(line_id, 0.0)),
            "rank": rank_by_line.get(line_id, ""),
            "selected_candidate_high_risk": bool(is_candidate),
            "deenergized": bool(deenergized),
            "z_l": int(0 if deenergized else 1),
        }
        score_rows.append(row)
        if is_candidate:
            decision_rows.append(
                {
                    "line_id": int(line_id),
                    "from_bus": int(edge_array[0, line_id]),
                    "to_bus": int(edge_array[1, line_id]),
                    "group_id": line_to_group.get(line_id, ""),
                    "baseline_psps_risk": float(baseline_psps_risk[line_id]),
                    "grouping_top_fraction": float(grouping_top_fraction),
                    "psps_top_fraction": float(psps_top_fraction),
                    "realized_psps_fraction": realized_psps_fraction,
                    "num_candidate_lines": int(num_candidate),
                    "num_psps_lines": int(num_psps),
                    "deenergized": bool(deenergized),
                    "z_l": int(0 if deenergized else 1),
                    "affected_buses": f"{int(edge_array[0, line_id])},{int(edge_array[1, line_id])}",
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(decision_rows)


def _build_model_context(model_type: str, grouping_top_fraction: float):
    config_path = REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "configs" / MODEL_CONFIGS[model_type]
    config = load_first_pass_config(config_path)
    config.model.model_type = model_type
    config.wildfire.selection_method = "automatic_risk_components"
    config.wildfire.risk_score = {
        **(config.wildfire.risk_score or {}),
        "formula": "p_env_times_loading_squared_times_impact",
        "threshold_method": "top_fraction",
        "top_fraction": float(grouping_top_fraction),
        "candidate_p_env": float((config.wildfire.risk_score or {}).get("candidate_p_env", 1.0)),
    }
    config.objective.lambda_R = STAGE_C_LAMBDA_R
    config.objective.lambda_L = STAGE_C_LAMBDA_L
    config.objective.risk_normalizer = 0.0
    config.objective.load_shedding_normalizer = 1.0

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
    wildfire, _baseline_line_impact, automatic_artifacts = build_wildfire_for_baseline(
        config,
        scenario,
        runner,
        decision_vector,
        baseline_prediction,
        baseline_state,
    )
    fixed_impact, consequence_df = compute_fixed_line_consequence_scores(
        scenario,
        runner,
        decision_vector.u_base,
    )
    return {
        "config": config,
        "config_path": config_path,
        "scenario": scenario,
        "decision_vector": decision_vector,
        "runner": runner,
        "baseline_prediction": baseline_prediction,
        "baseline_state": baseline_state,
        "wildfire": wildfire,
        "automatic_artifacts": automatic_artifacts,
        "fixed_impact": fixed_impact,
        "consequence_df": consequence_df,
    }


def run_stage_c_case(
    model_context: dict,
    case_name: str,
    output_root: Path,
    grouping_top_fraction: float,
    psps_top_fraction: float,
) -> dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    runner = model_context["runner"]
    wildfire = model_context["wildfire"]
    automatic_artifacts = model_context["automatic_artifacts"]
    fixed_impact = model_context["fixed_impact"]
    baseline_state = model_context["baseline_state"]
    consequence_df = model_context["consequence_df"]
    group_summary = automatic_artifacts["group_summary"]
    candidate_line_ids = sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})
    line_to_group = _candidate_group_maps(wildfire)
    env = apply_environmental_case(wildfire, group_summary, case_name)
    baseline_loading = np.asarray(baseline_state["loading_ratio"], dtype=float)
    baseline_psps_risk = {
        int(line_id): float(env.p_env_by_line[int(line_id)] * baseline_loading[int(line_id)] ** 2 * fixed_impact[int(line_id)])
        for line_id in candidate_line_ids
    }
    deenergized_line_ids = select_psps_lines(candidate_line_ids, baseline_psps_risk, psps_top_fraction)
    num_psps_lines = psps_line_count(len(candidate_line_ids), psps_top_fraction)
    realized_psps_fraction = float(num_psps_lines / len(candidate_line_ids))
    z_by_line = {int(line_id): (0 if int(line_id) in set(deenergized_line_ids) else 1) for line_id in range(len(baseline_loading))}

    baseline_risk, baseline_risk_by_line = compute_line_risk_with_z(
        baseline_loading,
        env.p_env_by_line,
        fixed_impact,
        {line_id: 1 for line_id in range(len(baseline_loading))},
        candidate_line_ids,
    )
    post_prediction, post_state = predict_psps_state(
        scenario,
        runner,
        decision_vector.u_base,
        deenergized_line_ids,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    post_loading = np.asarray(post_state["loading_ratio"], dtype=float)
    post_risk, post_risk_by_line = compute_line_risk_with_z(
        post_loading,
        env.p_env_by_line,
        fixed_impact,
        z_by_line,
        candidate_line_ids,
    )
    load_shed = demand_weighted_load_shed_from_prediction(post_prediction, scenario)
    risk_scale = max(float(baseline_risk), 1e-12)
    baseline_normalized_risk = 1.0
    post_normalized_risk = float(post_risk / risk_scale)
    baseline_objective = float(STAGE_C_LAMBDA_R * baseline_normalized_risk)
    post_objective = float(STAGE_C_LAMBDA_R * post_normalized_risk + STAGE_C_LAMBDA_L * load_shed)

    run_name = "run"
    run_dir = make_run_dir(output_root, run_name)
    write_config_copy(config, run_dir / "config.yaml")
    write_json(run_dir / "metadata.json", {**git_metadata(), "config_path": str(model_context["config_path"])})
    write_json(run_dir / "wildfire_scenario.json", wildfire.to_dict())
    write_dataframe(run_dir / "fixed_line_consequence_scores.csv", consequence_df)

    automatic_metadata = write_automatic_group_artifacts(run_dir, automatic_artifacts, baseline_risk, config)
    score_df, decision_df = _psps_tables(
        scenario,
        candidate_line_ids,
        deenergized_line_ids,
        line_to_group,
        env.p_env_by_line,
        baseline_loading,
        fixed_impact,
        baseline_psps_risk,
        grouping_top_fraction,
        psps_top_fraction,
    )
    write_dataframe(run_dir / "psps_line_risk_scores.csv", score_df)
    write_dataframe(run_dir / "psps_deenergization_decisions.csv", decision_df)
    write_dataframe(run_dir / "decision_vector_initial.csv", decision_vector.metadata_frame(decision_vector.u_base))
    write_dataframe(run_dir / "decision_vector_final.csv", decision_vector.metadata_frame(decision_vector.u_base))

    baseline_line_df, post_line_df = _line_risk_frames(
        scenario,
        candidate_line_ids,
        line_to_group,
        env.p_env_by_line,
        baseline_loading,
        post_loading,
        fixed_impact,
        baseline_risk_by_line,
        post_risk_by_line,
    )
    baseline_group_df = _group_risk_frame(wildfire, baseline_risk_by_line)
    post_group_df = _group_risk_frame(wildfire, post_risk_by_line)
    write_dataframe(run_dir / "risk_by_line_before_after.csv", _before_after(baseline_line_df, post_line_df, "line_id"))
    write_dataframe(run_dir / "risk_by_group_before_after.csv", _before_after(baseline_group_df, post_group_df, "group_name"))

    objective_trace = pd.DataFrame(
        [
            {
                "eval_idx": 0,
                "objective_total": baseline_objective,
                "wildfire_group_risk": float(baseline_risk),
                "load_shedding": 0.0,
                "normalized_wildfire_group_risk": baseline_normalized_risk,
                "normalized_load_shedding": 0.0,
                "risk_objective_term": float(STAGE_C_LAMBDA_R * baseline_normalized_risk),
                "load_shedding_objective_term": 0.0,
                "generator_movement": 0.0,
                "max_loading_ratio": float(baseline_state["max_loading_ratio"]),
                "mean_alpha": 1.0,
                "max_abs_delta_pg": 0.0,
                "message": "baseline_all_energized",
            },
            {
                "eval_idx": 1,
                "objective_total": post_objective,
                "wildfire_group_risk": float(post_risk),
                "load_shedding": float(load_shed),
                "normalized_wildfire_group_risk": post_normalized_risk,
                "normalized_load_shedding": float(load_shed),
                "risk_objective_term": float(STAGE_C_LAMBDA_R * post_normalized_risk),
                "load_shedding_objective_term": float(STAGE_C_LAMBDA_L * load_shed),
                "generator_movement": 0.0,
                "max_loading_ratio": float(post_state["max_loading_ratio"]),
                "mean_alpha": 1.0,
                "max_abs_delta_pg": 0.0,
                "message": "post_psps_threshold",
            },
        ]
    )
    write_dataframe(run_dir / "objective_trace.csv", objective_trace)

    affected = affected_buses(scenario.edge_index, deenergized_line_ids)
    components = disconnected_component_count(scenario.edge_index, scenario.num_buses, deenergized_line_ids)
    deenergized_group_ids = sorted({line_to_group[line_id] for line_id in deenergized_line_ids if line_id in line_to_group}, key=lambda item: int(item.split("_")[1]))
    summary = {
        "model_type": config.model.model_type.lower(),
        "scenario_label": f"{fraction_label('threshold', grouping_top_fraction)}/{fraction_label('psps_top', psps_top_fraction)}",
        "environmental_risk_case": case_name,
        "lambda_R": STAGE_C_LAMBDA_R,
        "lambda_L": STAGE_C_LAMBDA_L,
        "grouping_top_fraction": float(grouping_top_fraction),
        "psps_top_fraction": float(psps_top_fraction),
        "realized_psps_fraction": realized_psps_fraction,
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_psps_lines": int(num_psps_lines),
        "num_groups": int(len(wildfire.line_groups)),
        "largest_group_id": env.largest_group_id,
        "largest_group_num_lines": int(len(env.largest_group_line_ids)),
        "manual_high_risk_group_id": env.manual_high_risk_group_id,
        "num_deenergized_lines": int(len(deenergized_line_ids)),
        "deenergized_line_ids": [int(line_id) for line_id in deenergized_line_ids],
        "baseline_all_energized_risk": float(baseline_risk),
        "post_psps_risk": float(post_risk),
        "risk_reduction_fraction": float((baseline_risk - post_risk) / risk_scale),
        "demand_weighted_load_shed": float(load_shed),
        "load_shedding_metric": "demand_weighted_fraction_from_post_psps_prediction",
        "fixed_consequence_metric": "baseline_demand_weighted_one_line_outage",
        "num_disconnected_components": int(components),
        "affected_bus_ids": [int(bus) for bus in affected],
        "objective_value": float(post_objective),
        "evaluation_mode": "psps_only",
        "optimizer_success": None,
        "optimizer_message": "not_run_psps_only",
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "optimization_summary.json", summary)

    visualization_summary = {
        "optimization_behavior_plot": None,
        "topology_change_plot": None,
        "errors": [],
        "scenario_label": summary["scenario_label"],
        "grouping_top_fraction": float(grouping_top_fraction),
        "psps_top_fraction": float(psps_top_fraction),
        "realized_psps_fraction": realized_psps_fraction,
        "environmental_risk_case": case_name,
        "group_ids": automatic_metadata.get("group_ids"),
        "group_line_ids": automatic_metadata.get("group_line_ids"),
        "group_bus_ids": automatic_metadata.get("group_bus_ids"),
        "largest_group_id": env.largest_group_id,
        "largest_group_line_ids": env.largest_group_line_ids,
        "manual_high_risk_group_id": env.manual_high_risk_group_id,
        "deenergized_line_ids": [int(line_id) for line_id in deenergized_line_ids],
        "deenergized_group_ids": deenergized_group_ids,
        "num_deenergized_lines": int(len(deenergized_line_ids)),
        "num_candidate_lines": int(len(candidate_line_ids)),
        "collapsed_to_single_group": automatic_metadata.get("collapsed_to_single_group"),
        "num_disconnected_components": int(components),
        "affected_bus_ids": [int(bus) for bus in affected],
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

    return {**summary, "status": "ok", "error": ""}


def run_stage_c_psps_baseline(
    grouping_top_fraction: float = 0.30,
    psps_top_fraction: float = 0.10,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    clear: bool = False,
) -> Path:
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    root = _stage_c_root()
    if clear and root.exists():
        resolved_root = root.resolve()
        expected_parent = (REPO_ROOT / "experiments" / "test" / "wildfire_initial_tests" / "results").resolve()
        if resolved_root.parent != expected_parent:
            raise ValueError(f"Refusing to delete unexpected path: {resolved_root}")
        shutil.rmtree(resolved_root)
    root.mkdir(parents=True, exist_ok=True)

    threshold_root = root / compact_fraction_label("t", grouping_top_fraction) / compact_fraction_label("p", psps_top_fraction) / "r"
    rows = []
    for model_type in models:
        try:
            context = _build_model_context(model_type, grouping_top_fraction)
            for case_name in cases:
                output_root = threshold_root / model_type / CASE_FOLDERS.get(case_name, case_name)
                try:
                    rows.append(
                        run_stage_c_case(
                            context,
                            case_name,
                            output_root,
                            grouping_top_fraction,
                            psps_top_fraction,
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model_type": model_type,
                            "environmental_risk_case": case_name,
                            "grouping_top_fraction": float(grouping_top_fraction),
                            "psps_top_fraction": float(psps_top_fraction),
                            "status": "failed",
                            "error": str(exc),
                            "run_dir": "",
                        }
                    )
        except Exception as exc:
            for case_name in cases:
                rows.append(
                    {
                        "model_type": model_type,
                        "environmental_risk_case": case_name,
                        "grouping_top_fraction": float(grouping_top_fraction),
                        "psps_top_fraction": float(psps_top_fraction),
                        "status": "failed",
                        "error": str(exc),
                        "run_dir": "",
                    }
                )

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "stage_c_psps_summary.csv"
    summary_json = root / "stage_c_psps_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "grouping_top_fraction": float(grouping_top_fraction),
            "psps_top_fraction": float(psps_top_fraction),
            "models": models,
            "cases": cases,
            "evaluation_mode": "psps_only",
            "summary_csv": str(summary_csv),
        },
    )
    print(f"[OK] Stage C PSPS summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--psps-top-fraction", type=float, default=0.10)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--clear", action="store_true", help="Delete existing results/stage_c_psps first.")
    args = parser.parse_args()
    run_stage_c_psps_baseline(
        grouping_top_fraction=args.grouping_top_fraction,
        psps_top_fraction=args.psps_top_fraction,
        models=args.models,
        cases=args.cases,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
