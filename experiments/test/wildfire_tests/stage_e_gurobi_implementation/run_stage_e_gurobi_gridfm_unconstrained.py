from __future__ import annotations

import argparse
import shutil
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.plot_network_changes import plot_network_changes
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.shared.wildfire_setup import write_automatic_group_artifacts
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    CASE_FOLDERS,
    CASES,
    MODEL_CONFIGS,
    _build_model_context,
    _candidate_group_maps,
    compact_fraction_label,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    apply_environmental_case,
    demand_weighted_load_shed_from_prediction,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.run_stage_e_gurobi_gridfm import (
    _candidate_score_frame,
    _consequence_by_line,
    _decision_table,
    _evaluate_candidate,
    _group_risk_frame,
    _line_consequence_frame,
    _objective_trace,
    _risk_by_line_before_after,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    METHOD_NAME,
    STAGE_E_LAMBDA_CASES,
    STAGE_E_LAMBDA_FOLDERS,
    normalize_true_exposure,
)


UNCONSTRAINED_METHOD_NAME = f"{METHOD_NAME}_unconstrained"
EVALUATION_MODE = "gurobi_master_gridfm_fixed_u_unconstrained"


def _stage_e_unconstrained_root() -> Path:
    return RESULTS_ROOT / "leq" / "stage_e" / "unconstrained"


def _plot_unconstrained_objective_trace(eval_frame: pd.DataFrame, run_dir: Path, summary: dict) -> Path:
    ok = eval_frame[eval_frame["status"].eq("ok")].copy()
    if ok.empty:
        raise ValueError("Cannot plot unconstrained trace without successful evaluations.")
    ok["candidate_eval"] = np.arange(1, len(ok) + 1)
    ok["evaluated_objective"] = ok["true_objective"].astype(float)
    ok["best_so_far"] = ok["evaluated_objective"].cummin()
    best = ok.loc[ok["evaluated_objective"].idxmin()]

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    output_path = figure_dir / "unconstrained_objective_trace.png"

    fig, ax = plt.subplots(figsize=(10.0, 5.6), constrained_layout=True)
    ax.plot(
        ok["candidate_eval"],
        ok["evaluated_objective"],
        color="#8A8A8A",
        linewidth=1.2,
        alpha=0.55,
        label="Evaluated topology objective",
    )
    ax.scatter(
        ok["candidate_eval"],
        ok["evaluated_objective"],
        color="#F58518",
        s=24,
        alpha=0.78,
        label="GridFM topology evaluations",
    )
    ax.step(
        ok["candidate_eval"],
        ok["best_so_far"],
        where="post",
        color="#1F77B4",
        linewidth=2.4,
        label="Best objective so far",
    )
    ax.scatter(
        [best["candidate_eval"]],
        [best["evaluated_objective"]],
        color="#D62728",
        s=72,
        zorder=5,
        label="Final best topology",
    )
    ax.annotate(
        str(best["deenergized_line_ids"]).replace(" ", ""),
        xy=(best["candidate_eval"], best["evaluated_objective"]),
        xytext=(6, 10),
        textcoords="offset points",
        fontsize=8,
        rotation=20,
    )
    ax.set_title(
        f"Unconstrained Stage E: {summary['model_type']} / {summary['environmental_risk_case']} / "
        f"{summary['lambda_case']} / {summary['evaluation_budget']} proposals\n"
        f"J={summary['lambda_R_true']:g} R_norm + {summary['lambda_L_true']:g} L_shed; "
        f"best {best['deenergized_line_ids']} with R_norm={best['true_R_norm']:.4f}, "
        f"L_shed={best['true_L_shed']:.4f}, J={best['true_objective']:.4f}"
    )
    ax.set_xlabel("Unconstrained Stage E candidate topology evaluation")
    ax.set_ylabel("True GridFM objective")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _run_unconstrained_lambda_case(
    model_context: dict,
    case_name: str,
    env,
    output_root: Path,
    grouping_top_fraction: float,
    evaluation_budget: int,
    lambda_case_name: str,
    lambda_R: float,
    lambda_L: float,
    proxy_type: str,
    generate_run_figures: bool = True,
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
        raise ValueError("Stage E unconstrained candidate line set is empty.")

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
        _candidate_score_frame(
            scenario,
            candidate_line_ids,
            line_to_group,
            env.p_env_by_line,
            baseline_loading,
            c_by_line,
            proxy_type,
        ),
    )
    write_json(
        run_dir / "baseline_metrics.json",
        {
            "baseline_R_raw_new": float(baseline_R_raw_new),
            "baseline_R_norm": 1.0,
            "baseline_L_shed": float(baseline_L_shed),
            "baseline_P_AC": 0.0,
            "baseline_loading_by_line": {str(idx): float(value) for idx, value in enumerate(baseline_loading)},
            "p_env_by_line": {str(line_id): float(value) for line_id, value in env.p_env_by_line.items()},
            "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
            "risk_scope": "candidate_group_scope",
            "risk_formula": "sum_l z_l * p_env_l * loading_l^2",
            "unconstrained_topology_budget": True,
        },
    )

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
                max_deenergized_lines=None,
                evaluated_y_vectors=evaluated_y,
                proxy_type=proxy_type,
            )
        except Exception as exc:
            if iteration == 0:
                raise
            rows.append(
                {
                    "iteration": int(iteration),
                    "method_name": UNCONSTRAINED_METHOD_NAME,
                    "lambda_case_name": lambda_case_name,
                    "lambda_R_master": float(lambda_R),
                    "lambda_L_master": float(lambda_L),
                    "lambda_R_true": float(lambda_R),
                    "lambda_L_true": float(lambda_L),
                    "lambda_P": 0.0,
                    "model_type": config.model.model_type.lower(),
                    "candidate_lines": [int(line_id) for line_id in candidate_line_ids],
                    "K": "unconstrained",
                    "max_deenergized_lines": "",
                    "topology_budget_mode": "unconstrained",
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
        row["method_name"] = UNCONSTRAINED_METHOD_NAME
        row["K"] = "unconstrained"
        row["max_deenergized_lines"] = ""
        row["topology_budget_mode"] = "unconstrained"
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
    ok = eval_frame[eval_frame["status"].eq("ok")].copy() if len(eval_frame) else pd.DataFrame()
    if len(ok):
        write_dataframe(run_dir / "objective_trace.csv", _objective_trace(ok.copy()))
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

    best_row = ok.sort_values("true_objective", kind="mergesort").iloc[0].to_dict() if len(ok) else {}
    best_deenergized = best_row.get("deenergized_line_ids", [])
    summary = {
        "stage": "E",
        "study": "unconstrained_topology_budget",
        "method_name": UNCONSTRAINED_METHOD_NAME,
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
        "max_deenergized_lines": None,
        "K": "unconstrained",
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
        "gurobi_role": "unconstrained proxy topology candidate generator",
        "gridfm_role": "true post-topology evaluator",
        "run_dir": str(run_dir),
        "status": "ok" if len(ok) else "failed",
        "error": "" if len(ok) else "No successful candidate evaluations.",
    }

    visualization_summary = {
        "stage": "E",
        "study": "unconstrained_topology_budget",
        "method_name": UNCONSTRAINED_METHOD_NAME,
        "model_type": config.model.model_type.lower(),
        "environmental_risk_case": case_name,
        "lambda_case": lambda_case_name,
        "lambda_R": float(lambda_R),
        "lambda_L": float(lambda_L),
        "grouping_top_fraction": float(grouping_top_fraction),
        "evaluation_mode": EVALUATION_MODE,
        "max_deenergized_lines": None,
        "group_ids": automatic_metadata.get("group_ids"),
        "group_line_ids": automatic_metadata.get("group_line_ids"),
        "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
        "optimized_deenergized_line_ids": best_deenergized,
        "unconstrained_objective_trace_plot": None,
        "topology_change_plot": None,
        "errors": [],
    }
    write_json(run_dir / "optimization_summary.json", summary)
    write_json(run_dir / "visualization_summary.json", visualization_summary)
    if generate_run_figures:
        try:
            visualization_summary["unconstrained_objective_trace_plot"] = str(_plot_unconstrained_objective_trace(eval_frame, run_dir, summary))
        except Exception as exc:
            visualization_summary["errors"].append(f"unconstrained_objective_trace: {exc}")
        try:
            visualization_summary["topology_change_plot"] = str(plot_network_changes(run_dir))
        except Exception as exc:
            visualization_summary["errors"].append(f"topology_change: {exc}")
    write_json(run_dir / "visualization_summary.json", visualization_summary)
    write_json(run_dir / "optimization_summary.json", summary)
    return summary


def run_stage_e_gurobi_gridfm_unconstrained(
    grouping_top_fraction: float = 0.30,
    models: list[str] | None = None,
    cases: list[str] | None = None,
    lambda_cases: list[str] | None = None,
    evaluation_budget: int = 100,
    proxy_type: str = DEFAULT_PROXY_TYPE,
    clear: bool = False,
) -> Path:
    models = ["gps"] if models is None else models
    cases = list(CASES) if cases is None else cases
    lambda_cases = list(STAGE_E_LAMBDA_CASES) if lambda_cases is None else lambda_cases
    root = _stage_e_unconstrained_root()
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
                for lambda_case_name in lambda_cases:
                    lambda_R, lambda_L = STAGE_E_LAMBDA_CASES[lambda_case_name]
                    output_root = (
                        threshold_root
                        / model_type
                        / CASE_FOLDERS.get(case_name, case_name)
                        / STAGE_E_LAMBDA_FOLDERS.get(lambda_case_name, lambda_case_name)
                    )
                    rows.append(
                        _run_unconstrained_lambda_case(
                            context,
                            case_name,
                            env,
                            output_root,
                            grouping_top_fraction,
                            int(evaluation_budget),
                            lambda_case_name,
                            lambda_R,
                            lambda_L,
                            proxy_type,
                        )
                    )
        except Exception as exc:
            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            for case_name in cases:
                for lambda_case_name in lambda_cases:
                    lambda_R, lambda_L = STAGE_E_LAMBDA_CASES[lambda_case_name]
                    rows.append(
                        {
                            "stage": "E",
                            "study": "unconstrained_topology_budget",
                            "method_name": UNCONSTRAINED_METHOD_NAME,
                            "model_type": model_type,
                            "environmental_risk_case": case_name,
                            "lambda_case": lambda_case_name,
                            "lambda_R_master": lambda_R,
                            "lambda_L_master": lambda_L,
                            "lambda_R_true": lambda_R,
                            "lambda_L_true": lambda_L,
                            "lambda_P": 0.0,
                            "grouping_top_fraction": float(grouping_top_fraction),
                            "max_deenergized_lines": None,
                            "K": "unconstrained",
                            "evaluation_budget": int(evaluation_budget),
                            "proxy_type": proxy_type,
                            "status": "failed",
                            "error": error_text,
                            "run_dir": "",
                        }
                    )

    summary_frame = pd.DataFrame(rows)
    summary_csv = root / "stage_e_unconstrained_gridfm_summary.csv"
    summary_json = root / "stage_e_unconstrained_gridfm_summary.json"
    write_dataframe(summary_csv, summary_frame)
    write_json(
        summary_json,
        {
            "study": "unconstrained_topology_budget",
            "num_runs": int(len(rows)),
            "num_successful_runs": int((summary_frame["status"] == "ok").sum()) if len(summary_frame) else 0,
            "grouping_top_fraction": float(grouping_top_fraction),
            "evaluation_mode": EVALUATION_MODE,
            "max_deenergized_lines": None,
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
    print(f"[OK] Stage E unconstrained summary written to {summary_csv}")
    return summary_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grouping-top-fraction", type=float, default=0.30)
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gps"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=CASES)
    parser.add_argument("--lambda-cases", nargs="+", choices=sorted(STAGE_E_LAMBDA_CASES), default=list(STAGE_E_LAMBDA_CASES))
    parser.add_argument("--evaluation-budget", type=int, default=100)
    parser.add_argument("--proxy-type", choices=["env_loading_base", "env_only"], default=DEFAULT_PROXY_TYPE)
    parser.add_argument("--clear", action="store_true", help="Delete existing unconstrained Stage E results first.")
    args = parser.parse_args()
    run_stage_e_gurobi_gridfm_unconstrained(
        grouping_top_fraction=args.grouping_top_fraction,
        models=args.models,
        cases=args.cases,
        lambda_cases=args.lambda_cases,
        evaluation_budget=args.evaluation_budget,
        proxy_type=args.proxy_type,
        clear=args.clear,
    )


if __name__ == "__main__":
    main()
