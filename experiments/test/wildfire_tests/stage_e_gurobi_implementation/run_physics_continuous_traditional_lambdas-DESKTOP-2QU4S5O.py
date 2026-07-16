from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import _build_model_context
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    apply_environmental_case,
    demand_weighted_service,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    DEFAULT_PHYSICS_WEIGHTS,
    _active_line_ids,
    _physics_components,
    _predict_topology_state,
    recourse_bounds_with_island_limits,
    source_less_island_buses,
    weighted_pac_total,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import normalize_true_exposure


CASE_ROOT = RESULTS_ROOT / "leq" / "stage_e" / "physics_infeasibility_case_study"
CONTINUOUS_ROOT = CASE_ROOT / "with_continuous_optimization"
TRADITIONAL_LAMBDAS = [0.8, 0.5, 0.2]
STAGES = [
    "stage_c_risk_only",
    "stage_d_k2_exhaustive",
    "stage_e_k2",
    "stage_e_unconstrained",
]
STAGE_LABELS = {
    "stage_c_risk_only": "Stage C",
    "stage_d_k2_exhaustive": "Stage D",
    "stage_e_k2": "Stage E constrained",
    "stage_e_unconstrained": "Stage E unconstrained",
}
RHO_FOLDERS = {
    0.0: "rho0_no_physics",
    100.0: "rho100_with_physics",
}


class CallBudgetReached(RuntimeError):
    pass


def _long_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    return "\\\\?\\" + str(path.resolve())


def _mkdir(path: Path) -> None:
    os.makedirs(_long_path(path), exist_ok=True)


def _write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    _mkdir(path.parent)
    frame.to_csv(_long_path(path), index=False)


def _write_json(path: Path, data: Dict) -> None:
    _mkdir(path.parent)
    with open(_long_path(path), "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def _write_config_copy(config, path: Path) -> None:
    _mkdir(path.parent)
    with open(_long_path(path), "w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)


def _savefig(fig, path: Path) -> None:
    _mkdir(path.parent)
    fig.savefig(_long_path(path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_run_dir(root: Path) -> Path:
    run_dir = root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    _mkdir(run_dir)
    return run_dir


def _parse_topology(value) -> List[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return sorted(int(item) for item in text.split(",") if item != "")


def _topology_key(values: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted(int(item) for item in values))


def _candidate_line_ids(wildfire) -> List[int]:
    return sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})


def _baseline_exposure(model_context: dict, candidate_line_ids: List[int], p_env_by_line: Dict[int, float]) -> float:
    scenario = model_context["scenario"]
    loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    z_all_on = {line_id: 1 for line_id in range(int(scenario.edge_index.shape[1]))}
    raw, _by_line = compute_operational_wildfire_exposure(
        loading,
        p_env_by_line,
        z_all_on,
        candidate_line_ids,
    )
    return float(raw)


def _fixed_result_dir(rho_phys: float) -> Path:
    folder = RHO_FOLDERS[float(rho_phys)]
    candidates = [
        CASE_ROOT / "without_continuous_optimization" / folder,
        CASE_ROOT / folder,
    ]
    for root in candidates:
        if not root.exists():
            continue
        runs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("run_"))
        if runs:
            return runs[-1]
    raise FileNotFoundError(f"No fixed-control source run found for rho_phys={rho_phys}.")


def _source_topologies(rho_phys: float, stage_d_limit: int | None, stage_e_budget: int | None) -> Dict[float, Dict[str, pd.DataFrame]]:
    run_dir = _fixed_result_dir(rho_phys)
    path = run_dir / "combined_stage_cde_physics_results.csv"
    frame = pd.read_csv(_long_path(path))
    frame = frame[
        frame["gridfm_status"].eq("ok")
        & frame["stage"].isin(STAGES)
        & frame["lambda_R"].round(10).isin(TRADITIONAL_LAMBDAS)
    ].copy()
    frame["topology_key"] = frame["shutoff_line_ids"].fillna("").astype(str)
    result: Dict[float, Dict[str, pd.DataFrame]] = {}
    for lambda_R in TRADITIONAL_LAMBDAS:
        result[lambda_R] = {}
        local = frame[np.isclose(frame["lambda_R"], lambda_R)]
        for stage in STAGES:
            stage_frame = local[local["stage"].eq(stage)].copy()
            if stage == "stage_d_k2_exhaustive":
                stage_frame = stage_frame.sort_values("eval_id", kind="mergesort")
                if stage_d_limit is not None:
                    stage_frame = stage_frame.head(int(stage_d_limit))
            elif stage.startswith("stage_e"):
                stage_frame = stage_frame.sort_values("stage_e_iteration", kind="mergesort")
                if stage_e_budget is not None:
                    stage_frame = stage_frame.head(int(stage_e_budget))
            else:
                stage_frame = stage_frame.head(1)
            result[lambda_R][stage] = stage_frame.reset_index(drop=True)
    return result


def _evaluate_control(
    model_context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    removed: List[int],
    active: List[int],
    z_by_line: Dict[int, int],
    source_less: List[int],
    u: np.ndarray,
    lambda_R: float,
    rho_phys: float,
) -> Dict:
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    prediction, state, _keep = _predict_topology_state(model_context, np.asarray(u, dtype=float), removed)
    components = _physics_components(model_context, np.asarray(u, dtype=float), state, active, source_less)
    pac = weighted_pac_total(components, DEFAULT_PHYSICS_WEIGHTS)
    raw, exposure_by_line = compute_operational_wildfire_exposure(
        np.asarray(state["loading_ratio"], dtype=float),
        p_env_by_line,
        z_by_line,
        candidate_line_ids,
    )
    r_norm = normalize_true_exposure(raw, baseline_R)
    _served, predicted_service = demand_weighted_service(prediction, scenario)
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    total_demand = max(float(np.sum(demand)), 1e-12)
    l_shed_prediction = float(np.sum(demand * (1.0 - predicted_service)) / total_demand)
    full_alpha = decision_vector.full_alpha(np.asarray(u, dtype=float))
    l_shed_alpha = float(np.sum(demand * (1.0 - full_alpha)) / total_demand)
    lambda_L = 1.0 - float(lambda_R)
    j_no_phys = float(lambda_R) * r_norm + lambda_L * l_shed_prediction
    j_true = j_no_phys + float(rho_phys) * pac
    delta_pg, alpha_selected = decision_vector.split_decision_vector(np.asarray(u, dtype=float))
    return {
        "prediction": prediction,
        "state": state,
        "components": components,
        "exposure_by_line": exposure_by_line,
        "R_raw": float(raw),
        "R_norm": float(r_norm),
        "L_shed_prediction": l_shed_prediction,
        "L_shed_alpha_commanded": l_shed_alpha,
        "PAC_total": float(pac),
        "J_no_phys": float(j_no_phys),
        "J_true": float(j_true),
        "risk_contribution": float(lambda_R) * float(r_norm),
        "load_contribution": lambda_L * l_shed_prediction,
        "physics_contribution": float(rho_phys) * float(pac),
        "predicted_service_fraction": predicted_service,
        "alpha_full": full_alpha,
        "delta_pg": delta_pg,
        "alpha_selected": alpha_selected,
    }


def _alpha_diagnostics(
    model_context: dict,
    eval_id: int,
    stage: str,
    lambda_R: float,
    rho_phys: float,
    topology_key: str,
    components: Dict,
) -> tuple[List[Dict], Dict]:
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    demand = np.maximum(np.asarray(scenario.Pd_base, dtype=float), 0.0)
    selected = np.asarray(decision_vector.selected_load_buses, dtype=int)
    predicted = np.asarray(components["predicted_service_fraction"], dtype=float)
    alpha_full = np.asarray(components["alpha_full"], dtype=float)
    total_selected_demand = max(float(np.sum(demand[selected])), 1e-12)
    rows = []
    weighted_abs = 0.0
    weighted_signed = 0.0
    max_abs = 0.0
    for bus in selected:
        signed = float(predicted[bus] - alpha_full[bus])
        absolute = abs(signed)
        weight = float(demand[bus] / total_selected_demand)
        weighted_abs += weight * absolute
        weighted_signed += weight * signed
        max_abs = max(max_abs, absolute)
        rows.append(
            {
                "eval_id": int(eval_id),
                "stage": stage,
                "lambda_R": float(lambda_R),
                "rho_phys": float(rho_phys),
                "topology_key": topology_key,
                "bus_id": int(bus),
                "Pd_base": float(demand[bus]),
                "alpha_commanded_service_fraction": float(alpha_full[bus]),
                "gridfm_predicted_service_fraction": float(predicted[bus]),
                "signed_service_fraction_mismatch": signed,
                "absolute_service_fraction_mismatch": absolute,
                "selected_demand_weight": weight,
            }
        )
    summary = {
        "alpha_weighted_abs_mismatch": float(weighted_abs),
        "alpha_weighted_signed_mismatch": float(weighted_signed),
        "alpha_max_abs_mismatch": float(max_abs),
        "predicted_selected_bus_load_shed": float(
            np.sum(demand[selected] * (1.0 - predicted[selected])) / max(float(np.sum(demand)), 1e-12)
        ),
    }
    return rows, summary


def _optimize_topology(
    model_context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    shutoff_line_ids: List[int],
    stage: str,
    lambda_R: float,
    rho_phys: float,
    eval_id: int,
    call_budget: int,
) -> tuple[Dict, List[Dict], List[Dict]]:
    config = model_context["config"]
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    removed = sorted({int(line_id) for line_id in shutoff_line_ids})
    topology_key = _topology_key(removed)
    num_lines = int(scenario.edge_index.shape[1])
    active = _active_line_ids(num_lines, removed)
    z_by_line = {line_id: int(line_id not in set(removed)) for line_id in range(num_lines)}
    source_less = source_less_island_buses(scenario, removed)
    lower, upper, forced_alpha_count = recourse_bounds_with_island_limits(decision_vector, source_less)
    u0 = np.minimum(np.maximum(np.asarray(decision_vector.u_base, dtype=float), lower), upper)
    bounds = list(zip(lower.tolist(), upper.tolist()))
    traces: List[Dict] = []
    best: Dict | None = None
    calls = 0
    invalid_calls = 0
    started = time.perf_counter()
    scipy_success = False
    scipy_message = ""
    termination_reason = ""

    def objective(u_raw: np.ndarray) -> float:
        nonlocal calls, invalid_calls, best
        if calls >= int(call_budget):
            raise CallBudgetReached(f"GridFM objective-call budget {call_budget} reached.")
        calls += 1
        u = np.minimum(np.maximum(np.asarray(u_raw, dtype=float), lower), upper)
        call_started = time.perf_counter()
        try:
            components = _evaluate_control(
                model_context,
                candidate_line_ids,
                p_env_by_line,
                baseline_R,
                removed,
                active,
                z_by_line,
                source_less,
                u,
                lambda_R,
                rho_phys,
            )
            value = float(components["J_true"])
            valid = bool(np.isfinite(value))
            error = ""
        except Exception as exc:
            components = {}
            value = float(config.objective.invalid_prediction_penalty)
            valid = False
            error = str(exc)
            invalid_calls += 1
        runtime = float(time.perf_counter() - call_started)
        improved = bool(valid and (best is None or value < float(best["J_true"]) - 1e-15))
        if improved:
            best = {
                **components,
                "u": u.copy(),
                "call_idx": int(calls),
                "J_true": value,
            }
        traces.append(
            {
                "eval_id": int(eval_id),
                "stage": stage,
                "lambda_R": float(lambda_R),
                "lambda_L": float(1.0 - lambda_R),
                "rho_phys": float(rho_phys),
                "topology_key": topology_key,
                "call_idx": int(calls),
                "J_true": value,
                "R_norm": components.get("R_norm", np.nan),
                "L_shed_prediction": components.get("L_shed_prediction", np.nan),
                "L_shed_alpha_commanded": components.get("L_shed_alpha_commanded", np.nan),
                "PAC_total": components.get("PAC_total", np.nan),
                "valid": valid,
                "is_best_within_topology": improved,
                "call_runtime_seconds": runtime,
                "error": error,
            }
        )
        return value

    try:
        result = minimize(
            objective,
            u0,
            method=config.optimizer.method,
            bounds=bounds,
            options={
                "maxiter": 1000,
                "ftol": float(config.optimizer.ftol),
                "gtol": float(config.optimizer.gtol),
                "eps": float(config.optimizer.eps),
                "disp": False,
            },
        )
        scipy_success = bool(result.success)
        scipy_message = str(result.message)
        termination_reason = "scipy_converged" if result.success else "scipy_terminated"
    except CallBudgetReached as exc:
        scipy_message = str(exc)
        termination_reason = "call_budget_reached"
    except Exception as exc:
        scipy_message = str(exc)
        termination_reason = "optimizer_exception"

    runtime_seconds = float(time.perf_counter() - started)
    if best is None:
        raise RuntimeError(
            f"No valid continuous-control evaluation for stage={stage}, lambda_R={lambda_R}, "
            f"rho_phys={rho_phys}, topology={topology_key}."
        )
    alpha_rows, alpha_summary = _alpha_diagnostics(
        model_context,
        eval_id,
        stage,
        lambda_R,
        rho_phys,
        topology_key,
        best,
    )
    delta_pg = np.asarray(best["delta_pg"], dtype=float)
    alpha_selected = np.asarray(best["alpha_selected"], dtype=float)
    row = {
        "eval_id": int(eval_id),
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "case_name": "auto_env",
        "model": "gnn",
        "lambda_R": float(lambda_R),
        "lambda_L": float(1.0 - lambda_R),
        "rho_phys": float(rho_phys),
        "topology_key": topology_key,
        "shutoff_line_ids": topology_key,
        "num_shutoff_lines": int(len(removed)),
        "best_call_idx": int(best["call_idx"]),
        "gridfm_calls": int(calls),
        "invalid_calls": int(invalid_calls),
        "call_budget": int(call_budget),
        "budget_exhausted": bool(calls >= int(call_budget)),
        "termination_reason": termination_reason,
        "scipy_success": bool(scipy_success),
        "scipy_message": scipy_message,
        "runtime_seconds": runtime_seconds,
        "J_true": float(best["J_true"]),
        "J_no_phys": float(best["J_no_phys"]),
        "R_raw": float(best["R_raw"]),
        "R_norm": float(best["R_norm"]),
        "L_shed_prediction": float(best["L_shed_prediction"]),
        "L_shed_alpha_commanded": float(best["L_shed_alpha_commanded"]),
        "PAC_total": float(best["PAC_total"]),
        "risk_contribution": float(best["risk_contribution"]),
        "load_contribution": float(best["load_contribution"]),
        "physics_contribution": float(best["physics_contribution"]),
        "PAC_voltage_limits": float(best["components"]["voltage_limits"]),
        "PAC_thermal_limits": float(best["components"]["thermal_limits"]),
        "PAC_generator_limits": float(best["components"]["generator_limits"]),
        "PAC_island_source_feasibility": float(best["components"]["island_source_feasibility"]),
        "source_less_bus_ids": _topology_key(source_less),
        "source_less_selected_alpha_forced_zero": int(forced_alpha_count),
        "max_abs_delta_pg": float(np.max(np.abs(delta_pg))) if len(delta_pg) else 0.0,
        "mean_alpha": float(np.mean(alpha_selected)) if len(alpha_selected) else 1.0,
        "min_alpha": float(np.min(alpha_selected)) if len(alpha_selected) else 1.0,
        "u_best_json": json.dumps(np.asarray(best["u"], dtype=float).tolist()),
        "delta_pg_json": json.dumps(delta_pg.tolist()),
        "alpha_selected_json": json.dumps(alpha_selected.tolist()),
        **alpha_summary,
    }
    return row, traces, alpha_rows


def _best_by_stage(topology_results: pd.DataFrame) -> pd.DataFrame:
    idx = topology_results.groupby(["lambda_R", "rho_phys", "stage"], dropna=False)["J_true"].idxmin()
    return topology_results.loc[idx].sort_values(["lambda_R", "stage"], ascending=[False, True]).reset_index(drop=True)


def _runtime_summary(topology_results: pd.DataFrame) -> pd.DataFrame:
    return (
        topology_results.groupby(["lambda_R", "rho_phys", "stage", "stage_label"], dropna=False)
        .agg(
            num_topologies=("eval_id", "count"),
            total_runtime_seconds=("runtime_seconds", "sum"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            total_gridfm_calls=("gridfm_calls", "sum"),
            mean_gridfm_calls=("gridfm_calls", "mean"),
            budget_exhaustion_rate=("budget_exhausted", "mean"),
            scipy_convergence_rate=("scipy_success", "mean"),
            invalid_calls=("invalid_calls", "sum"),
        )
        .reset_index()
    )


def _stage_e_vs_stage_d_gap(best: pd.DataFrame) -> pd.DataFrame:
    d = best[best["stage"].eq("stage_d_k2_exhaustive")]
    rows = []
    for stage in ["stage_e_k2", "stage_e_unconstrained"]:
        e = best[best["stage"].eq(stage)]
        merged = e.merge(d, on=["lambda_R", "lambda_L", "rho_phys"], suffixes=("_stage_e", "_stage_d"))
        for _, row in merged.iterrows():
            denominator = max(abs(float(row["J_true_stage_d"])), 1e-12)
            rows.append(
                {
                    "lambda_R": float(row["lambda_R"]),
                    "lambda_L": float(row["lambda_L"]),
                    "rho_phys": float(row["rho_phys"]),
                    "stage_e_method": stage,
                    "stage_e_J_true": float(row["J_true_stage_e"]),
                    "stage_d_J_true": float(row["J_true_stage_d"]),
                    "gap_J_true": float(row["J_true_stage_e"] - row["J_true_stage_d"]),
                    "relative_gap_J_true": float(
                        (row["J_true_stage_e"] - row["J_true_stage_d"]) / denominator
                    ),
                    "stage_e_topology": row["topology_key_stage_e"],
                    "stage_d_topology": row["topology_key_stage_d"],
                }
            )
    return pd.DataFrame(rows)


def _plot_outputs(
    topology_results: pd.DataFrame,
    best: pd.DataFrame,
    runtime: pd.DataFrame,
    output_dir: Path,
) -> None:
    _mkdir(output_dir)
    colors = {
        "stage_c_risk_only": "#7F7F7F",
        "stage_d_k2_exhaustive": "#4C78A8",
        "stage_e_k2": "#F58518",
        "stage_e_unconstrained": "#54A24B",
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for ax, lambda_R in zip(axes, TRADITIONAL_LAMBDAS):
        local = topology_results[np.isclose(topology_results["lambda_R"], lambda_R)].copy()
        max_iterations = max(int(local["stage_topology_index"].max()), 1)
        topology_lines = []
        for stage in STAGES:
            stage_rows = local[local["stage"].eq(stage)].sort_values(
                "stage_topology_index",
                kind="mergesort",
            )
            if stage_rows.empty:
                continue
            best_so_far = stage_rows["J_true"].cummin()
            if stage == "stage_c_risk_only":
                final_value = float(best_so_far.iloc[-1])
                ax.axhline(
                    final_value,
                    color=colors[stage],
                    linestyle="--",
                    linewidth=1.7,
                    label=STAGE_LABELS[stage],
                )
                endpoint_x = max_iterations
            else:
                ax.step(
                    stage_rows["stage_topology_index"],
                    best_so_far,
                    where="post",
                    color=colors[stage],
                    linewidth=2.0,
                    label=STAGE_LABELS[stage],
                )
                final_value = float(best_so_far.iloc[-1])
                endpoint_x = int(stage_rows["stage_topology_index"].iloc[-1])
            ax.scatter([endpoint_x], [final_value], color="#D62728", s=34, zorder=5)
            final_row = stage_rows.loc[stage_rows["J_true"].idxmin()]
            topology = str(final_row["topology_key"]) or "none"
            topology_lines.append(f"{STAGE_LABELS[stage]}: [{topology}]")
        ax.set_title(f"lambda_R={lambda_R:.1f}, lambda_L={1-lambda_R:.1f}")
        ax.set_xlabel("Topology iteration")
        ax.set_ylabel("Best topology objective so far, J_true")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        ax.text(
            0.98,
            0.76,
            "\n".join(topology_lines),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox={"facecolor": "white", "edgecolor": "#C8C8C8", "alpha": 0.88, "pad": 4},
        )
    rho_value = float(best["rho_phys"].iloc[0])
    fig.suptitle(f"Traditional-Lambda Continuous Topology Search, rho_phys={rho_value:g}")
    _savefig(fig, output_dir / "traditional_lambda_objective_comparison.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    for ax, lambda_R in zip(axes, TRADITIONAL_LAMBDAS):
        local = best[np.isclose(best["lambda_R"], lambda_R)].copy()
        local["stage_order"] = local["stage"].map({stage: idx for idx, stage in enumerate(STAGES)})
        local = local.sort_values("stage_order")
        x = np.arange(len(local))
        width = 0.25
        ax.bar(x - width, local["L_shed_alpha_commanded"], width, label="Alpha-commanded")
        ax.bar(x, local["predicted_selected_bus_load_shed"], width, label="Predicted selected buses")
        ax.bar(x + width, local["L_shed_prediction"], width, label="Predicted system-wide")
        ax.set_xticks(x, local["stage_label"], rotation=25)
        ax.set_title(f"lambda_R={lambda_R:.1f}")
        ax.set_ylabel("Demand-weighted load shed")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    fig.suptitle(f"Alpha Command Versus GridFM-Predicted Load Shedding, rho_phys={rho_value:g}")
    _savefig(fig, output_dir / "alpha_consistency_comparison.png")

    runtime_plot = runtime.copy()
    runtime_plot["runtime_minutes"] = runtime_plot["total_runtime_seconds"] / 60.0
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for stage in STAGES:
        local = runtime_plot[runtime_plot["stage"].eq(stage)].sort_values("lambda_R")
        axes[0].plot(local["lambda_R"], local["runtime_minutes"], marker="o", label=STAGE_LABELS[stage])
        axes[1].plot(local["lambda_R"], local["total_gridfm_calls"], marker="o", label=STAGE_LABELS[stage])
    axes[0].set_xlabel("lambda_R")
    axes[0].set_ylabel("Total runtime (minutes)")
    axes[1].set_xlabel("lambda_R")
    axes[1].set_ylabel("Total GridFM calls")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Continuous Optimization Runtime and Call Usage, rho_phys={rho_value:g}")
    _savefig(fig, output_dir / "runtime_and_call_usage.png")


def run_continuous_study(
    rho_phys: float,
    call_budget: int = 100,
    stage_d_limit: int | None = None,
    stage_e_budget: int | None = None,
) -> Path:
    family_root = CONTINUOUS_ROOT / RHO_FOLDERS[float(rho_phys)]
    run_dir = _make_run_dir(family_root)
    plots_dir = run_dir / "plots"
    overall_started = time.perf_counter()
    model_context = _build_model_context("gnn", grouping_top_fraction=0.30)
    _write_config_copy(model_context["config"], run_dir / "config.yaml")
    wildfire = model_context["wildfire"]
    env_case = apply_environmental_case(
        wildfire,
        model_context["automatic_artifacts"]["group_summary"],
        "auto_env",
    )
    candidate_line_ids = _candidate_line_ids(wildfire)
    baseline_R = _baseline_exposure(model_context, candidate_line_ids, env_case.p_env_by_line)
    sources = _source_topologies(rho_phys, stage_d_limit, stage_e_budget)
    topology_rows: List[Dict] = []
    trace_rows: List[Dict] = []
    alpha_rows: List[Dict] = []
    eval_id = 0
    for lambda_R in TRADITIONAL_LAMBDAS:
        for stage in STAGES:
            source = sources[lambda_R][stage]
            for index, source_row in source.iterrows():
                topology = _parse_topology(source_row.get("shutoff_line_ids", ""))
                row, traces, alpha = _optimize_topology(
                    model_context,
                    candidate_line_ids,
                    env_case.p_env_by_line,
                    baseline_R,
                    topology,
                    stage,
                    lambda_R,
                    rho_phys,
                    eval_id,
                    call_budget,
                )
                row["stage_topology_index"] = int(index + 1)
                topology_rows.append(row)
                trace_rows.extend(traces)
                alpha_rows.extend(alpha)
                eval_id += 1
                if eval_id % 25 == 0:
                    _write_dataframe(run_dir / "progress_topology_results.csv", pd.DataFrame(topology_rows))
                    _write_json(
                        run_dir / "progress.json",
                        {
                            "completed_topology_optimizations": int(eval_id),
                            "latest_stage": stage,
                            "latest_lambda_R": float(lambda_R),
                            "latest_topology": row["topology_key"],
                            "elapsed_seconds": float(time.perf_counter() - overall_started),
                        },
                    )
                    print(
                        f"[rho={rho_phys:g}] completed {eval_id} topology optimizations; "
                        f"latest={stage} lambda_R={lambda_R:.1f} topology={row['topology_key'] or 'none'}",
                        flush=True,
                    )

    topology_results = pd.DataFrame(topology_rows)
    traces = pd.DataFrame(trace_rows)
    alpha_diagnostics = pd.DataFrame(alpha_rows)
    best = _best_by_stage(topology_results)
    runtime = _runtime_summary(topology_results)
    gap = _stage_e_vs_stage_d_gap(best)
    _write_dataframe(run_dir / "combined_continuous_results.csv", topology_results)
    _write_dataframe(run_dir / "best_within_topology.csv", topology_results)
    _write_dataframe(run_dir / "best_by_stage_lambda.csv", best)
    _write_dataframe(run_dir / "continuous_objective_call_trace.csv", traces)
    _write_dataframe(run_dir / "alpha_consistency_diagnostics.csv", alpha_diagnostics)
    _write_dataframe(run_dir / "stage_e_vs_stage_d_gap.csv", gap)
    _write_dataframe(run_dir / "runtime_summary.csv", runtime)
    _plot_outputs(topology_results, best, runtime, plots_dir)
    _write_json(
        run_dir / "metadata.json",
        {
            **git_metadata(),
            "study": "physics_infeasibility_traditional_lambdas_with_continuous_optimization",
            "case_name": "auto_env",
            "model": "gnn",
            "rho_phys": float(rho_phys),
            "lambda_R_values": TRADITIONAL_LAMBDAS,
            "lambda_L_rule": "1 - lambda_R",
            "stages": STAGES,
            "call_budget_per_topology": int(call_budget),
            "continuous_decision": "[Delta_Pg, alpha] over 3 selected generator buses and 5 selected load buses",
            "within_topology_selection": "lowest valid J_true observed within the GridFM call budget",
            "across_topology_selection": "lowest retained within-topology J_true for each fixed stage, lambda, and rho_phys",
            "true_objective": "lambda_R * R_norm + lambda_L * L_shed_prediction_system_wide + rho_phys * PAC_total",
            "load_shedding_metric": "demand-weighted GridFM-predicted system-wide service loss",
            "alpha_diagnostic": "commanded selected-bus service versus GridFM-predicted selected-bus service",
            "stage_d_limit": stage_d_limit,
            "stage_e_budget": stage_e_budget,
            "num_topology_optimizations": int(len(topology_results)),
            "num_objective_calls": int(topology_results["gridfm_calls"].sum()),
            "runtime_seconds": float(time.perf_counter() - overall_started),
            "pareto_frontier_generated": False,
            "traditional_lambda_topology_search_plot_generated": True,
            "stage_specific_lambda_sweep_plots_generated": False,
        },
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the traditional-lambda physics case study with budgeted continuous optimization."
    )
    parser.add_argument("--rho-phys", nargs="+", type=float, choices=[0.0, 100.0], default=[0.0, 100.0])
    parser.add_argument("--call-budget", type=int, default=100)
    parser.add_argument("--stage-d-limit", type=int, default=None)
    parser.add_argument("--stage-e-budget", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for rho_phys in args.rho_phys:
        run_dir = run_continuous_study(
            rho_phys=float(rho_phys),
            call_budget=int(args.call_budget),
            stage_d_limit=args.stage_d_limit,
            stage_e_budget=args.stage_e_budget,
        )
        print(f"Wrote continuous traditional-lambda study to {run_dir}")


if __name__ == "__main__":
    main()
