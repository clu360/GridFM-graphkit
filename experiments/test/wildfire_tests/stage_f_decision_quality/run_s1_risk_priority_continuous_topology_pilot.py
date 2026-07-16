from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import scipy.optimize as opt

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    predict_psps_state,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    deenergized_from_y,
    normalize_true_exposure,
    z_from_y,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    RISK_SCOPE,
    _consequence_by_line,
    _csv_ints,
    _edge_array,
    _risk_line_ids,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    get_scenarios,
    p_env_for_scenario,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_f" / "continuous_topology_pilot" / "s1_risk_priority_k2"
LAMBDA_CASE = "risk_priority"
LAMBDA_R = 0.8
LAMBDA_L = 0.2
SCENARIO_ID = "S1"
STAGE_E_NAME = "stage_e_k2_gurobi_continuous_pilot"


def _fixed_z(num_lines: int, shutoff_line_ids: Iterable[int]) -> Dict[int, int]:
    shutoff = {int(line_id) for line_id in shutoff_line_ids}
    return {line_id: int(line_id not in shutoff) for line_id in range(int(num_lines))}


def _evaluate_revised_objective(
    model_context: dict,
    u: np.ndarray,
    shutoff_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    risk_line_ids: List[int],
) -> Dict:
    config = model_context["config"]
    scenario = model_context["scenario"]
    runner = model_context["runner"]
    decision_vector = model_context["decision_vector"]
    num_lines = int(_edge_array(scenario).shape[1])
    prediction, state = predict_psps_state(
        scenario,
        runner,
        u,
        shutoff_line_ids,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )
    loading = np.asarray(state["loading_ratio"], dtype=float)
    if not np.all(np.isfinite(loading)):
        raise FloatingPointError("GridFM loading contains NaN or inf.")
    if state.get("prediction_has_nan") or state.get("prediction_has_inf"):
        raise FloatingPointError("GridFM prediction contains NaN or inf.")
    R_wf, exposure_by_line = compute_operational_wildfire_exposure(
        loading,
        p_env_by_line,
        _fixed_z(num_lines, shutoff_line_ids),
        risk_line_ids,
    )
    R_norm = normalize_true_exposure(R_wf, baseline_R)
    L_shed_alpha = decision_vector.load_shedding(u)
    J_true = float(LAMBDA_R * R_norm + LAMBDA_L * L_shed_alpha)
    return {
        "R_wf": float(R_wf),
        "R_base_s": float(baseline_R),
        "R_norm": float(R_norm),
        "L_shed": float(L_shed_alpha),
        "J_true": float(J_true),
        "risk_contribution": float(LAMBDA_R * R_norm),
        "load_contribution": float(LAMBDA_L * L_shed_alpha),
        "max_loading_ratio": float(state.get("max_loading_ratio", np.nan)),
        "min_voltage": float(state.get("min_voltage", np.nan)),
        "max_voltage": float(state.get("max_voltage", np.nan)),
        "num_nan": int(state.get("num_nan", 0)),
        "num_inf": int(state.get("num_inf", 0)),
        "true_exposure_by_line": ";".join(f"{int(k)}:{float(v):.12g}" for k, v in sorted(exposure_by_line.items())),
        "prediction": prediction,
        "state": state,
    }


def _optimize_fixed_topology(
    model_context: dict,
    shutoff_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    risk_line_ids: List[int],
    maxiter: int,
    ftol: float | None = None,
    gtol: float | None = None,
    eps: float | None = None,
) -> Dict:
    config = model_context["config"]
    decision_vector = model_context["decision_vector"]
    trace_rows: list[Dict] = []
    penalty = float(config.objective.invalid_prediction_penalty)

    def objective(u_raw: np.ndarray) -> float:
        eval_id = len(trace_rows)
        try:
            components = _evaluate_revised_objective(
                model_context,
                np.asarray(u_raw, dtype=float),
                shutoff_line_ids,
                p_env_by_line,
                baseline_R,
                risk_line_ids,
            )
            value = float(components["J_true"])
            message = "ok"
        except Exception as exc:
            components = {
                "J_true": penalty,
                "R_norm": np.nan,
                "L_shed": np.nan,
                "R_wf": np.nan,
                "max_loading_ratio": np.nan,
                "min_voltage": np.nan,
                "max_voltage": np.nan,
            }
            value = penalty
            message = f"invalid_prediction: {exc}"
        if eval_id < 5000:
            trace_rows.append(
                {
                    "objective_eval": int(eval_id),
                    "J_true": float(value),
                    "R_norm": float(components.get("R_norm", np.nan)),
                    "L_shed": float(components.get("L_shed", np.nan)),
                    "R_wf": float(components.get("R_wf", np.nan)),
                    "max_loading_ratio": float(components.get("max_loading_ratio", np.nan)),
                    "min_voltage": float(components.get("min_voltage", np.nan)),
                    "max_voltage": float(components.get("max_voltage", np.nan)),
                    "message": message,
                }
            )
        return value

    u0 = decision_vector.u_base.copy()
    bounds = list(zip(decision_vector.u_min, decision_vector.u_max))
    options = {
        "maxiter": int(maxiter),
        "ftol": config.optimizer.ftol if ftol is None else float(ftol),
        "gtol": config.optimizer.gtol if gtol is None else float(gtol),
        "eps": config.optimizer.eps if eps is None else float(eps),
    }
    fixed_components = _evaluate_revised_objective(
        model_context,
        u0,
        shutoff_line_ids,
        p_env_by_line,
        baseline_R,
        risk_line_ids,
    )
    started = time.perf_counter()
    result = opt.minimize(
        objective,
        u0,
        method=config.optimizer.method,
        bounds=bounds,
        options=options,
    )
    runtime = float(time.perf_counter() - started)
    final_components = _evaluate_revised_objective(
        model_context,
        np.asarray(result.x, dtype=float),
        shutoff_line_ids,
        p_env_by_line,
        baseline_R,
        risk_line_ids,
    )
    delta_pg, alpha = decision_vector.split_decision_vector(result.x)
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "n_iter": int(result.nit),
        "num_objective_evals": int(result.nfev),
        "runtime_seconds": runtime,
        "fixed_control_components": fixed_components,
        "final_components": final_components,
        "u_final": np.asarray(result.x, dtype=float),
        "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
        "mean_alpha": float(np.mean(alpha)) if len(alpha) else 1.0,
        "min_alpha": float(np.min(alpha)) if len(alpha) else 1.0,
        "trace": pd.DataFrame(trace_rows),
    }


def _row_from_candidate(
    scenario_name: str,
    candidate_line_ids: List[int],
    risk_line_ids: List[int],
    eval_id: int,
    proposal: Dict,
    optimize_result: Dict,
) -> Dict:
    y_by_line = {int(k): int(v) for k, v in proposal["y_by_line"].items()}
    shutoff = deenergized_from_y(y_by_line)
    fixed = optimize_result["fixed_control_components"]
    final = optimize_result["final_components"]
    return {
        "stage": STAGE_E_NAME,
        "scenario_id": SCENARIO_ID,
        "scenario_name": scenario_name,
        "lambda_case": LAMBDA_CASE,
        "lambda_R": LAMBDA_R,
        "lambda_L": LAMBDA_L,
        "eval_id": int(eval_id),
        "candidate_set": "fixed_t0p30",
        "risk_scope": RISK_SCOPE,
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_risk_lines": int(len(risk_line_ids)),
        "num_shutoff_lines": int(len(shutoff)),
        "shutoff_line_ids": _csv_ints(shutoff),
        "fixed_R_norm": float(fixed["R_norm"]),
        "fixed_L_shed": float(fixed["L_shed"]),
        "fixed_J_true": float(fixed["J_true"]),
        "optimized_R_norm": float(final["R_norm"]),
        "optimized_L_shed": float(final["L_shed"]),
        "optimized_J_true": float(final["J_true"]),
        "optimized_risk_contribution": float(final["risk_contribution"]),
        "optimized_load_contribution": float(final["load_contribution"]),
        "optimized_R_wf": float(final["R_wf"]),
        "R_base_s": float(final["R_base_s"]),
        "max_loading_ratio": float(final["max_loading_ratio"]),
        "min_voltage": float(final["min_voltage"]),
        "max_voltage": float(final["max_voltage"]),
        "optimizer_success": bool(optimize_result["success"]),
        "optimizer_message": str(optimize_result["message"]),
        "optimizer_iterations": int(optimize_result["n_iter"]),
        "optimizer_objective_evals": int(optimize_result["num_objective_evals"]),
        "optimizer_runtime_seconds": float(optimize_result["runtime_seconds"]),
        "max_abs_delta_pg": float(optimize_result["max_abs_delta_pg"]),
        "mean_alpha": float(optimize_result["mean_alpha"]),
        "min_alpha": float(optimize_result["min_alpha"]),
        "proxy_R_hat": float(proposal.get("proxy_R_hat", np.nan)),
        "proxy_L_hat": float(proposal.get("proxy_L_hat", np.nan)),
        "proxy_objective": float(proposal.get("proxy_objective", np.nan)),
        "proxy_R_denominator": float(proposal.get("proxy_R_denominator", np.nan)),
        "gurobi_objective": float(proposal.get("gurobi_objective", np.nan)),
        "gurobi_status": int(proposal.get("gurobi_status", -1)),
    }


def run_s1_risk_priority_continuous_pilot(
    model_type: str = "gnn",
    evaluation_budget: int = 100,
    max_deenergized_lines: int = 2,
    optimizer_maxiter: int = 80,
) -> Path:
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_type={model_type}; supported={sorted(MODEL_CONFIGS)}")
    started = time.perf_counter()
    run_dir = make_run_dir(RESULT_ROOT, "run")
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    model_context = _build_model_context(model_type, grouping_top_fraction=0.30)
    config = model_context["config"]
    write_config_copy(config, run_dir / f"config_{model_type}.yaml")
    scenario = model_context["scenario"]
    scenario_def = get_scenarios([SCENARIO_ID])[0]
    candidate_line_ids = list(FIXED_T0P30_CANDIDATE_LINE_IDS)
    num_lines = int(_edge_array(scenario).shape[1])
    _validate_candidate_line_ids(candidate_line_ids, num_lines)
    risk_line_ids = _risk_line_ids(num_lines)
    p_env = p_env_for_scenario(scenario_def, num_lines)
    baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    baseline_R, baseline_by_line = _scenario_baseline_exposure(baseline_loading, p_env, risk_line_ids)
    c_by_line = _consequence_by_line(model_context["consequence_df"])

    evaluated_y: list[dict[int, int]] = []
    rows = []
    for eval_id in range(int(evaluation_budget)):
        proposal = solve_gurobi_master_next_candidate(
            candidate_line_ids,
            p_env,
            baseline_loading,
            c_by_line,
            LAMBDA_R,
            LAMBDA_L,
            max_deenergized_lines=max_deenergized_lines,
            evaluated_y_vectors=evaluated_y,
            proxy_type=DEFAULT_PROXY_TYPE,
        )
        y_by_line = {int(k): int(v) for k, v in proposal["y_by_line"].items()}
        evaluated_y.append(y_by_line)
        shutoff = deenergized_from_y(y_by_line)
        optimize_result = _optimize_fixed_topology(
            model_context,
            shutoff,
            p_env,
            baseline_R,
            risk_line_ids,
            maxiter=optimizer_maxiter,
        )
        row = _row_from_candidate(
            scenario_def.scenario_name,
            candidate_line_ids,
            risk_line_ids,
            eval_id,
            proposal,
            optimize_result,
        )
        rows.append(row)
        trace = optimize_result["trace"].copy()
        if not trace.empty:
            trace.insert(0, "eval_id", int(eval_id))
            trace.insert(1, "shutoff_line_ids", _csv_ints(shutoff))
            write_dataframe(traces_dir / f"candidate_{eval_id:03d}_trace.csv", trace)
        print(
            f"[{eval_id + 1}/{evaluation_budget}] shutoff={row['shutoff_line_ids'] or 'none'} "
            f"fixed_J={row['fixed_J_true']:.6f} opt_J={row['optimized_J_true']:.6f} "
            f"success={row['optimizer_success']}",
            flush=True,
        )

    candidates = pd.DataFrame(rows)
    best = candidates.sort_values(
        by=["optimized_J_true", "optimized_L_shed", "num_shutoff_lines", "shutoff_line_ids"],
        kind="mergesort",
    ).head(1)
    successful = candidates[candidates["optimizer_success"].astype(bool)].copy()
    best_successful = successful.sort_values(
        by=["optimized_J_true", "optimized_L_shed", "num_shutoff_lines", "shutoff_line_ids"],
        kind="mergesort",
    ).head(1)
    fixed_best = candidates.sort_values(
        by=["fixed_J_true", "fixed_L_shed", "num_shutoff_lines", "shutoff_line_ids"],
        kind="mergesort",
    ).head(1)
    write_dataframe(run_dir / "continuous_candidate_evaluations.csv", candidates)
    write_dataframe(run_dir / "continuous_best_decision.csv", best)
    write_dataframe(run_dir / "continuous_best_successful_decision.csv", best_successful)
    write_dataframe(run_dir / "fixed_control_best_among_same_candidates.csv", fixed_best)
    write_json(
        run_dir / "metadata.json",
        {
            **git_metadata(),
            "stage": "F",
            "study": "s1_risk_priority_continuous_topology_pilot",
            "model_type": model_type,
            "scenario_id": SCENARIO_ID,
            "scenario_name": scenario_def.scenario_name,
            "lambda_case": LAMBDA_CASE,
            "lambda_R": LAMBDA_R,
            "lambda_L": LAMBDA_L,
            "max_deenergized_lines": int(max_deenergized_lines),
            "evaluation_budget": int(evaluation_budget),
            "optimizer_maxiter": int(optimizer_maxiter),
            "candidate_line_ids": [int(line_id) for line_id in candidate_line_ids],
            "risk_scope": RISK_SCOPE,
            "risk_line_ids": [int(line_id) for line_id in risk_line_ids],
            "num_risk_lines": int(len(risk_line_ids)),
            "p_env_high_line_ids": [int(line_id) for line_id, value in sorted(p_env.items()) if abs(float(value) - 1.0) <= 1e-12],
            "p_env_low_value": 0.05,
            "R_base_s": float(baseline_R),
            "baseline_exposure_by_line": {str(int(k)): float(v) for k, v in sorted(baseline_by_line.items())},
            "continuous_true_objective": "lambda_R * (sum_l_in_all_lines z_l * p_env_l * loading_l(u,z)^2 / R_base_s) + lambda_L * demand_weighted_load_shedding(alpha)",
            "impact_or_consequence_in_true_objective": False,
            "gurobi_proxy_uses_consequence_c_l": True,
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S1 risk-priority Stage F continuous topology pilot.")
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGS), default="gnn")
    parser.add_argument("--evaluation-budget", type=int, default=100)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--optimizer-maxiter", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_s1_risk_priority_continuous_pilot(
        model_type=args.model,
        evaluation_budget=args.evaluation_budget,
        max_deenergized_lines=args.max_deenergized_lines,
        optimizer_maxiter=args.optimizer_maxiter,
    )
    print(f"Wrote S1 continuous topology pilot results to {run_dir}")


if __name__ == "__main__":
    main()
