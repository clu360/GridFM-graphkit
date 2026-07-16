from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT as SHARED_RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
    compact_fraction_label,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    apply_environmental_case,
    psps_line_count,
    select_psps_lines,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    GurobiUnavailableError,
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    evaluate_topology_with_physics_recourse,
    risk_only_stage_c_scores,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    STAGE_E_LAMBDA_CASES,
    candidate_y_from_deenergized,
    compute_proxy_metrics,
)


USER_CASES = {
    "auto_env": "auto_env",
    "lfg": "largest_group_high",
}

DEFAULT_RHO_PHYS = [0.0, 100.0]
CASE_STUDY_ROOT = SHARED_RESULTS_ROOT / "leq" / "stage_e" / "physics_infeasibility_case_study"
RESULT_ROOT = CASE_STUDY_ROOT / "without_continuous_optimization"
TRADITIONAL_LAMBDA_R = [0.8, 0.5, 0.2]
RHO_OUTPUT_FOLDERS = {
    0.0: "rho0_no_physics",
    100.0: "rho100_with_physics",
}


def _candidate_line_ids(wildfire) -> List[int]:
    return sorted({int(line_id) for group in wildfire.line_groups for line_id in group.line_ids})


def _consequence_by_line(consequence_df: pd.DataFrame) -> Dict[int, float]:
    frame = consequence_df.copy()
    source = "c_l" if "c_l" in frame.columns else "I_l"
    return {int(row["line_id"]): float(row[source]) for _, row in frame.iterrows()}


def _z_all_on(num_lines: int) -> Dict[int, int]:
    return {int(line_id): 1 for line_id in range(int(num_lines))}


def _baseline_exposure(model_context: dict, candidate_line_ids: List[int], p_env_by_line: Dict[int, float]) -> float:
    scenario = model_context["scenario"]
    baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    raw, _by_line = compute_operational_wildfire_exposure(
        baseline_loading,
        p_env_by_line,
        _z_all_on(int(scenario.edge_index.shape[1])),
        candidate_line_ids,
    )
    return float(raw)


def _as_list(value: str) -> List[int]:
    if value is None or str(value) == "":
        return []
    return [int(item) for item in str(value).split(",") if str(item) != ""]


def _lambda_name(lambda_r: float) -> str:
    return f"lr{float(lambda_r):.2f}".replace(".", "p")


def _rho_output_folder(rho_phys: float) -> str:
    value = float(rho_phys)
    if value in RHO_OUTPUT_FOLDERS:
        return RHO_OUTPUT_FOLDERS[value]
    return f"rho{value:.2f}".replace(".", "p").replace("-", "m")


def _clear_tree(path: Path) -> None:
    def _onexc(function, failed_path, exc_info):
        if isinstance(exc_info, FileNotFoundError):
            return
        if isinstance(exc_info, BaseException) and isinstance(exc_info.__cause__, FileNotFoundError):
            return
        try:
            os.chmod(failed_path, 0o700)
            function(failed_path)
        except FileNotFoundError:
            return

    try:
        shutil.rmtree(path, onexc=_onexc)
    except FileNotFoundError:
        return
    except OSError:
        isolated = path.with_name(f"{path.name}_broken_onedrive_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        path.rename(isolated)


def _long_path(path: Path) -> str:
    text = str(path)
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return text
    return "\\\\?\\" + str(path.resolve())


def _mkdir(path: Path) -> None:
    os.makedirs(_long_path(path), exist_ok=True)


def _savefig(fig, path: Path, **kwargs) -> None:
    fig.savefig(_long_path(path), **kwargs)


def _write_config_copy(config, path: Path) -> None:
    _mkdir(path.parent)
    with open(_long_path(path), "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (float, np.floating)) and pd.isna(value):
        return None
    return value


def _write_json(path: Path, data: Dict) -> None:
    _mkdir(path.parent)
    with open(_long_path(path), "w", encoding="utf-8") as f:
        json.dump(_json_safe(data), f, indent=2)


def _write_dataframe(path: Path, df: pd.DataFrame) -> None:
    _mkdir(path.parent)
    df.to_csv(_long_path(path), index=False)


def _make_run_dir(output_root: Path, run_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{run_name}_{stamp}"
    _mkdir(run_dir)
    return run_dir


def _lambda_sweep_values(lambda_step: float) -> List[float]:
    step = float(lambda_step)
    if step <= 0.0 or step > 1.0:
        raise ValueError(f"lambda-step must be in (0, 1], got {lambda_step}.")
    count = int(round(1.0 / step))
    values = [round(idx * step, 10) for idx in range(count + 1)]
    if not np.isclose(values[-1], 1.0):
        values.append(1.0)
    return [float(round(value, 2)) for value in values]


def _lambda_settings(lambda_cases: Sequence[str] | None, lambda_step: float | None) -> List[Dict[str, float | str]]:
    if lambda_step is not None:
        return [
            {
                "lambda_case": _lambda_name(lambda_R),
                "lambda_R": float(lambda_R),
                "lambda_L": float(1.0 - float(lambda_R)),
            }
            for lambda_R in _lambda_sweep_values(lambda_step)
        ]
    cases = list(STAGE_E_LAMBDA_CASES) if lambda_cases is None else list(lambda_cases)
    return [
        {
            "lambda_case": str(name),
            "lambda_R": float(STAGE_E_LAMBDA_CASES[str(name)][0]),
            "lambda_L": float(STAGE_E_LAMBDA_CASES[str(name)][1]),
        }
        for name in cases
    ]


def nondominated_mask(frame: pd.DataFrame, columns: Iterable[str] = ("R_norm", "L_shed")) -> np.ndarray:
    values = frame[list(columns)].to_numpy(dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    mask = np.zeros(len(frame), dtype=bool)
    for i, point in enumerate(values):
        if not finite[i]:
            continue
        dominated = False
        for j, other in enumerate(values):
            if i == j or not finite[j]:
                continue
            if np.all(other <= point + 1e-12) and np.any(other < point - 1e-12):
                dominated = True
                break
        mask[i] = not dominated
    return mask


def _best_by_stage(combined: pd.DataFrame) -> pd.DataFrame:
    ok = combined[combined["gridfm_status"] == "ok"].copy()
    if ok.empty:
        return ok
    group_cols = ["case_name", "model", "lambda_case", "lambda_R", "lambda_L", "rho_phys", "stage"]
    idx = ok.groupby(group_cols, dropna=False)["J_true"].idxmin()
    return ok.loc[idx].sort_values(group_cols).reset_index(drop=True)


def _stage_e_vs_stage_d_gap(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["case_name", "model", "lambda_case", "lambda_R", "lambda_L", "rho_phys"]
    d = best[best["stage"] == "stage_d_k2_exhaustive"]
    e = best[best["stage"] == "stage_e_k2"]
    merged = e.merge(d, on=keys, suffixes=("_stage_e", "_stage_d"))
    for _, row in merged.iterrows():
        denom = max(abs(float(row["J_true_stage_d"])), 1e-12)
        rows.append(
            {
                **{key: row[key] for key in keys},
                "stage_e_eval_id": int(row["eval_id_stage_e"]),
                "stage_d_eval_id": int(row["eval_id_stage_d"]),
                "stage_e_J_true": float(row["J_true_stage_e"]),
                "stage_d_J_true": float(row["J_true_stage_d"]),
                "gap_J_true": float(row["J_true_stage_e"] - row["J_true_stage_d"]),
                "relative_gap_J_true": float((row["J_true_stage_e"] - row["J_true_stage_d"]) / denom),
                "stage_e_J_no_phys": float(row["J_no_phys_stage_e"]),
                "stage_d_J_no_phys": float(row["J_no_phys_stage_d"]),
                "gap_J_no_phys": float(row["J_no_phys_stage_e"] - row["J_no_phys_stage_d"]),
            }
        )
    return pd.DataFrame(rows)


def _physics_sensitivity(best: pd.DataFrame) -> pd.DataFrame:
    keys = ["case_name", "model", "lambda_case", "lambda_R", "lambda_L", "stage"]
    zero = best[best["rho_phys"] == 0].copy()
    phys = best[best["rho_phys"] == 100].copy()
    merged = phys.merge(zero, on=keys, suffixes=("_rho100", "_rho0"))
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                **{key: row[key] for key in keys},
                "J_true_rho0": float(row["J_true_rho0"]),
                "J_true_rho100": float(row["J_true_rho100"]),
                "PAC_total_rho0": float(row["PAC_total_rho0"]),
                "PAC_total_rho100": float(row["PAC_total_rho100"]),
                "R_norm_delta": float(row["R_norm_rho100"] - row["R_norm_rho0"]),
                "L_shed_delta": float(row["L_shed_rho100"] - row["L_shed_rho0"]),
                "PAC_total_delta": float(row["PAC_total_rho100"] - row["PAC_total_rho0"]),
                "topology_changed": str(row["shutoff_line_ids_rho100"]) != str(row["shutoff_line_ids_rho0"]),
                "shutoff_line_ids_rho0": row["shutoff_line_ids_rho0"],
                "shutoff_line_ids_rho100": row["shutoff_line_ids_rho100"],
            }
        )
    return pd.DataFrame(rows)


def _all_candidate_points(combined: pd.DataFrame) -> pd.DataFrame:
    ok = combined[combined["gridfm_status"] == "ok"].copy()
    if ok.empty:
        return ok
    points = ok.dropna(subset=["R_norm", "L_shed"]).copy()
    if points.empty:
        return points
    points["line_id_key"] = points["shutoff_line_ids"].fillna("").astype(str)
    return points.reset_index(drop=True)


def _unique_candidate_points(all_points: pd.DataFrame) -> pd.DataFrame:
    if all_points.empty:
        return all_points.copy()
    sort_cols = ["case_name", "rho_phys", "stage", "line_id_key", "lambda_R", "J_true"]
    present = [col for col in sort_cols if col in all_points.columns]
    return (
        all_points.sort_values(present, kind="mergesort")
        .drop_duplicates(subset=["case_name", "rho_phys", "stage", "line_id_key"], keep="first")
        .reset_index(drop=True)
    )


def _pareto_frontier_points(unique_points: pd.DataFrame) -> pd.DataFrame:
    if unique_points.empty:
        return unique_points.copy()
    rows = []
    group_cols = ["case_name", "rho_phys", "stage"]
    for key, group in unique_points.groupby(group_cols, dropna=False, sort=True):
        local = group.dropna(subset=["R_norm", "L_shed"]).copy()
        if local.empty:
            continue
        local["is_pareto_frontier"] = nondominated_mask(local, ["R_norm", "L_shed"])
        local["pareto_rank_scope"] = "|".join(str(part) for part in key)
        rows.append(local[local["is_pareto_frontier"]].copy())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def _traditional_lambda_summary(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return best.copy()
    mask = best["lambda_R"].astype(float).round(2).isin(TRADITIONAL_LAMBDA_R)
    return best[mask].sort_values(["case_name", "rho_phys", "lambda_R", "stage"], ascending=[True, True, False, True]).reset_index(drop=True)


def _traditional_objective_comparison(traditional: pd.DataFrame) -> pd.DataFrame:
    if traditional.empty:
        return traditional.copy()
    columns = [
        "case_name",
        "rho_phys",
        "lambda_case",
        "lambda_R",
        "lambda_L",
        "stage",
        "proposal_method",
        "shutoff_line_ids",
        "J_true",
        "J_no_phys",
        "R_norm",
        "L_shed",
        "PAC_total",
        "PAC_voltage_limits",
        "PAC_thermal_limits",
        "PAC_generator_limits",
        "PAC_island_source_feasibility",
        "source_less_bus_ids",
        "source_less_selected_alpha_forced_zero",
        "scipy_success",
        "gridfm_status",
    ]
    present = [col for col in columns if col in traditional.columns]
    return traditional[present].copy()


def _best_so_far(values: pd.Series) -> pd.Series:
    return values.astype(float).cummin()


def _plot_outputs(
    combined: pd.DataFrame,
    traditional: pd.DataFrame,
    frontier: pd.DataFrame,
    output_dir: Path,
) -> None:
    _mkdir(output_dir)
    ok = combined[combined["gridfm_status"] == "ok"].copy()
    if ok.empty:
        return

    stage_specs = [
        ("stage_c_risk_only", "Stage C: risk-only heuristic"),
        ("stage_d_k2_exhaustive", "Stage D: exhaustive k<=2"),
        ("stage_e_k2", "Stage E: constrained K<=2"),
        ("stage_e_unconstrained", "Stage E: unconstrained"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (stage, title) in zip(axes.flat, stage_specs):
        stage_points = ok[ok["stage"] == stage].copy()
        stage_points["line_id_key"] = stage_points["shutoff_line_ids"].fillna("").astype(str)
        stage_points = stage_points.drop_duplicates("line_id_key", keep="first")
        stage_frontier = frontier[frontier["stage"] == stage].copy() if not frontier.empty else pd.DataFrame()
        ax.scatter(
            stage_points["L_shed"],
            stage_points["R_norm"],
            s=20,
            color="#B8B8B8",
            alpha=0.35,
            linewidths=0,
            label="Evaluated topologies",
        )
        if not stage_frontier.empty:
            ordered = stage_frontier.sort_values(["L_shed", "R_norm"], kind="mergesort")
            ax.scatter(
                ordered["L_shed"],
                ordered["R_norm"],
                s=50,
                color="#D62728",
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
                label="Nondominated frontier",
            )
            ax.plot(ordered["L_shed"], ordered["R_norm"], color="#D62728", linewidth=1.5, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("GridFM-predicted load shed, L_shed")
        ax.set_ylabel("Wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Stage-wise Pareto Frontiers From Lambda Sweep, lambda_R = 0.00..1.00 step 0.05")
    _savefig(fig, output_dir / "pareto_frontier_scatter.png", dpi=180)
    plt.close(fig)

    if not traditional.empty:
        colors = {
            "stage_c_risk_only": "#7F7F7F",
            "stage_d_k2_exhaustive": "#4C78A8",
            "stage_e_k2": "#F58518",
            "stage_e_unconstrained": "#54A24B",
        }
        labels = {
            "stage_c_risk_only": "Stage C",
            "stage_d_k2_exhaustive": "Stage D",
            "stage_e_k2": "Stage E constrained",
            "stage_e_unconstrained": "Stage E unconstrained",
        }
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
        traditional_lambdas = [0.8, 0.5, 0.2]
        for ax, lambda_R in zip(axes, traditional_lambdas):
            lambda_rows = ok[np.isclose(ok["lambda_R"].astype(float), lambda_R)].copy()
            stage_groups = {}
            for stage, _title in stage_specs:
                group = lambda_rows[lambda_rows["stage"] == stage].copy()
                if group.empty:
                    continue
                if stage.startswith("stage_e"):
                    group = group.sort_values("stage_e_iteration", kind="mergesort")
                else:
                    group = group.sort_values("eval_id", kind="mergesort")
                group["candidate_eval"] = np.arange(1, len(group) + 1)
                group["best_so_far"] = _best_so_far(group["J_true"])
                stage_groups[stage] = group
            max_evaluations = max([len(group) for group in stage_groups.values()] or [1])
            topology_lines = []
            for stage, _title in stage_specs:
                group = stage_groups.get(stage)
                if group is None:
                    continue
                if stage == "stage_c_risk_only":
                    final_row = group.iloc[0]
                    ax.axhline(
                        float(final_row["J_true"]),
                        color=colors[stage],
                        linestyle="--",
                        linewidth=1.5,
                        label=labels[stage],
                    )
                    final_x = max_evaluations
                    final_y = float(final_row["J_true"])
                else:
                    ax.step(
                        group["candidate_eval"],
                        group["best_so_far"],
                        where="post",
                        color=colors[stage],
                        linewidth=2.0,
                        label=labels[stage],
                    )
                    final_x = int(group["candidate_eval"].iloc[-1])
                    final_y = float(group["best_so_far"].iloc[-1])
                    final_row = group.loc[group["J_true"].idxmin()]
                topology = str(final_row.get("shutoff_line_ids", "")) or "none"
                ax.scatter([final_x], [final_y], color="#D62728", s=42, zorder=5)
                topology_lines.append(f"{labels[stage]}: [{topology}]")
            if topology_lines:
                ax.text(
                    0.98,
                    0.78,
                    "\n".join(topology_lines),
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                    ha="right",
                    bbox={"facecolor": "white", "edgecolor": "#C8C8C8", "alpha": 0.88, "pad": 4},
                )
            ax.set_xlim(left=0, right=max_evaluations * 1.03)
            ax.set_title(f"lambda_R={lambda_R:.1f}, lambda_L={1.0 - lambda_R:.1f}")
            ax.set_xlabel("Topology evaluation")
            ax.set_ylabel("Best objective so far")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="upper right", fontsize=8)
        rho_value = float(ok["rho_phys"].iloc[0])
        fig.suptitle(
            f"Traditional-Lambda Objective Convergence, rho_phys={rho_value:g}\n"
            f"J = lambda_R R_norm + lambda_L L_shed + {rho_value:g} PAC_total"
        )
        _savefig(fig, output_dir / "traditional_lambda_objective_comparison.png", dpi=180)
        plt.close(fig)

        residual_cols = ["PAC_voltage_limits", "PAC_thermal_limits", "PAC_generator_limits", "PAC_island_source_feasibility"]
        residual_cols = [col for col in residual_cols if col in traditional.columns]
        residual = traditional[["stage", "lambda_case", *residual_cols]].copy()
        residual["label"] = residual["stage"] + "|" + residual["lambda_case"]
        fig, ax = plt.subplots(figsize=(10, 5))
        residual.set_index("label")[residual_cols].plot.bar(stacked=True, ax=ax)
        ax.set_ylabel("Physics residual components")
        ax.tick_params(axis="x", labelrotation=75)
        fig.tight_layout()
        _savefig(fig, output_dir / "traditional_lambda_physics_residuals.png", dpi=180)
        plt.close(fig)

    for stage, name, x_col in [
        ("stage_d_k2_exhaustive", "stage_d_best_so_far.png", "eval_id"),
        ("stage_e_k2", "stage_e_k2_best_so_far.png", "stage_e_iteration"),
        ("stage_e_unconstrained", "stage_e_unconstrained_best_so_far.png", "stage_e_iteration"),
    ]:
        trace = ok[ok["stage"] == stage].copy()
        displayed_lambdas = np.arange(0.0, 1.01, 0.2)
        trace = trace[
            trace["lambda_R"].astype(float).apply(
                lambda value: bool(np.any(np.isclose(float(value), displayed_lambdas)))
            )
        ].copy()
        if trace.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        line_styles = ["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))]
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(displayed_lambdas)))
        for curve_idx, (lambda_R, group) in enumerate(trace.groupby("lambda_R", dropna=False, sort=True)):
            group = group.sort_values(x_col)
            candidate_eval = np.arange(1, len(group) + 1)
            best_so_far = _best_so_far(group["J_true"])
            ax.plot(
                candidate_eval,
                best_so_far,
                color=colors[curve_idx],
                linestyle=line_styles[curve_idx],
                linewidth=1.8,
                label=f"lambda_R={float(lambda_R):.1f}",
            )
            ax.scatter(
                [candidate_eval[-1]],
                [best_so_far.iloc[-1]],
                color=colors[curve_idx],
                s=24,
                zorder=4,
            )
        ax.set_xlabel("Topology evaluation")
        ax.set_ylabel("Best-so-far J_true")
        ax.set_title(f"{stage}: lambda_R shown at 0.2 increments")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        _savefig(fig, output_dir / name, dpi=180)
        plt.close(fig)


def _evaluate_and_collect(
    rows: List[Dict],
    traces: List[Dict],
    model_context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_R: float,
    shutoff_line_ids: Iterable[int],
    lambda_case: str,
    lambda_R: float,
    lambda_L: float,
    rho_phys: float,
    stage: str,
    case_name: str,
    eval_id: int,
    proposal_method: str,
    topology_budget: int | None = None,
    stage_e_iteration: int | None = None,
    stage_c_proposal_score: float | None = None,
    proxy_fields: Dict | None = None,
    recourse_maxiter: int | None = None,
    optimize_recourse: bool = True,
) -> None:
    result = evaluate_topology_with_physics_recourse(
        model_context=model_context,
        candidate_line_ids=candidate_line_ids,
        p_env_by_line=p_env_by_line,
        baseline_R_raw=baseline_R,
        shutoff_line_ids=shutoff_line_ids,
        lambda_R=lambda_R,
        lambda_L=lambda_L,
        rho_phys=rho_phys,
        stage=stage,
        case_name=case_name,
        lambda_case=lambda_case,
        eval_id=eval_id,
        proposal_method=proposal_method,
        model_name="gnn",
        topology_budget=topology_budget,
        stage_e_iteration=stage_e_iteration,
        stage_c_proposal_score=stage_c_proposal_score,
        proxy_fields=proxy_fields,
        recourse_maxiter=recourse_maxiter,
        optimize_recourse=optimize_recourse,
    )
    rows.append(result.row)
    traces.extend(result.trace)


def _run_stage_e_proposals(
    rows: List[Dict],
    traces: List[Dict],
    model_context: dict,
    candidate_line_ids: List[int],
    p_env_by_line: Dict[int, float],
    baseline_loading: np.ndarray,
    c_by_line: Dict[int, float],
    baseline_R: float,
    lambda_case: str,
    lambda_R: float,
    lambda_L: float,
    rho_phys: float,
    case_name: str,
    stage: str,
    max_deenergized_lines: int | None,
    budget: int,
    eval_id_start: int,
    recourse_maxiter: int | None,
    optimize_recourse: bool,
) -> int:
    evaluated_y: List[Dict[int, int]] = []
    eval_id = int(eval_id_start)
    for iteration in range(1, int(budget) + 1):
        try:
            proposal = solve_gurobi_master_next_candidate(
                candidate_line_ids,
                p_env_by_line,
                baseline_loading,
                c_by_line,
                lambda_R,
                lambda_L,
                max_deenergized_lines=max_deenergized_lines,
                evaluated_y_vectors=evaluated_y,
                proxy_type=DEFAULT_PROXY_TYPE,
            )
        except Exception as exc:
            rows.append(
                {
                    "stage": stage,
                    "case_name": case_name,
                    "model": "gnn",
                    "candidate_set": "t0p30",
                    "lambda_case": lambda_case,
                    "lambda_R": float(lambda_R),
                    "lambda_L": float(lambda_L),
                    "rho_phys": float(rho_phys),
                    "eval_id": eval_id,
                    "proposal_method": "gurobi_master",
                    "topology_budget": np.nan if max_deenergized_lines is None else int(max_deenergized_lines),
                    "stage_e_iteration": int(iteration),
                    "gridfm_status": "proposal_failed",
                    "gridfm_error": str(exc),
                }
            )
            return eval_id + 1
        y_by_line = {int(k): int(v) for k, v in proposal["y_by_line"].items()}
        evaluated_y.append(y_by_line)
        shutoff = [line_id for line_id, value in y_by_line.items() if int(value) == 1]
        _evaluate_and_collect(
            rows,
            traces,
            model_context,
            candidate_line_ids,
            p_env_by_line,
            baseline_R,
            shutoff,
            lambda_case,
            lambda_R,
            lambda_L,
            rho_phys,
            stage,
            case_name,
            eval_id,
            "gurobi_master",
            topology_budget=max_deenergized_lines,
            stage_e_iteration=iteration,
            proxy_fields=proposal,
            recourse_maxiter=recourse_maxiter,
            optimize_recourse=optimize_recourse,
        )
        eval_id += 1
    return eval_id


def run_physics_infeasibility_case_study(
    cases: List[str],
    lambda_cases: List[str] | None,
    lambda_step: float | None,
    rho_phys_values: List[float],
    stage_e_k2_budget: int,
    stage_e_unconstrained_budget: int,
    recourse_maxiter: int | None = 10,
    optimize_recourse: bool = True,
    stage_d_limit: int | None = None,
    output_root: Path = RESULT_ROOT,
    clear: bool = False,
) -> Path:
    if clear and output_root.exists():
        expected_root = RESULT_ROOT
        if output_root.parent != expected_root and output_root != expected_root:
            raise ValueError(f"Refusing to clear unexpected output root: {output_root}")
        _clear_tree(output_root)
    run_dir = _make_run_dir(output_root, "run")
    plots_dir = run_dir / "plots"
    settings = _lambda_settings(lambda_cases, lambda_step)

    model_context = _build_model_context("gnn", 0.30)
    config = model_context["config"]
    _write_config_copy(config, run_dir / "config.yaml")

    wildfire = model_context["wildfire"]
    group_summary = model_context["automatic_artifacts"]["group_summary"]
    candidate_line_ids = _candidate_line_ids(wildfire)
    baseline_loading = np.asarray(model_context["baseline_state"]["loading_ratio"], dtype=float)
    c_by_line = _consequence_by_line(model_context["consequence_df"])

    rows: List[Dict] = []
    traces: List[Dict] = []
    eval_id = 0

    for user_case in cases:
        internal_case = USER_CASES[user_case]
        env_case = apply_environmental_case(wildfire, group_summary, internal_case)
        p_env_by_line = env_case.p_env_by_line
        baseline_R = _baseline_exposure(model_context, candidate_line_ids, p_env_by_line)
        stage_c_scores = risk_only_stage_c_scores(candidate_line_ids, p_env_by_line, baseline_loading)
        stage_c_shutoff = select_psps_lines(candidate_line_ids, stage_c_scores, 0.10)
        stage_c_score_sum = float(sum(stage_c_scores[line_id] for line_id in stage_c_shutoff))
        stage_d_subsets = enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=2)
        if stage_d_limit is not None:
            limited = [subset for subset in stage_d_subsets if len(subset) == 0]
            limited.extend(subset for subset in stage_d_subsets if len(subset) > 0)
            stage_d_subsets = limited[: max(1, int(stage_d_limit))]

        for setting in settings:
            lambda_case = str(setting["lambda_case"])
            lambda_R = float(setting["lambda_R"])
            lambda_L = float(setting["lambda_L"])
            for rho_phys in rho_phys_values:
                _evaluate_and_collect(
                    rows,
                    traces,
                    model_context,
                    candidate_line_ids,
                    p_env_by_line,
                    baseline_R,
                    stage_c_shutoff,
                    lambda_case,
                    lambda_R,
                    lambda_L,
                    rho_phys,
                    "stage_c_risk_only",
                    user_case,
                    eval_id,
                    "risk_only_transmission_heuristic",
                    topology_budget=psps_line_count(len(candidate_line_ids), 0.10),
                    stage_c_proposal_score=stage_c_score_sum,
                    recourse_maxiter=recourse_maxiter,
                    optimize_recourse=optimize_recourse,
                )
                eval_id += 1

                for subset in stage_d_subsets:
                    _evaluate_and_collect(
                        rows,
                        traces,
                        model_context,
                        candidate_line_ids,
                        p_env_by_line,
                        baseline_R,
                        subset,
                        lambda_case,
                        lambda_R,
                        lambda_L,
                        rho_phys,
                        "stage_d_k2_exhaustive",
                        user_case,
                        eval_id,
                        "exhaustive_k_le_2",
                        topology_budget=2,
                        recourse_maxiter=recourse_maxiter,
                        optimize_recourse=optimize_recourse,
                    )
                    eval_id += 1

                eval_id = _run_stage_e_proposals(
                    rows,
                    traces,
                    model_context,
                    candidate_line_ids,
                    p_env_by_line,
                    baseline_loading,
                    c_by_line,
                    baseline_R,
                    lambda_case,
                    lambda_R,
                    lambda_L,
                    rho_phys,
                    user_case,
                    "stage_e_k2",
                    max_deenergized_lines=2,
                    budget=stage_e_k2_budget,
                    eval_id_start=eval_id,
                    recourse_maxiter=recourse_maxiter,
                    optimize_recourse=optimize_recourse,
                )
                eval_id = _run_stage_e_proposals(
                    rows,
                    traces,
                    model_context,
                    candidate_line_ids,
                    p_env_by_line,
                    baseline_loading,
                    c_by_line,
                    baseline_R,
                    lambda_case,
                    lambda_R,
                    lambda_L,
                    rho_phys,
                    user_case,
                    "stage_e_unconstrained",
                    max_deenergized_lines=None,
                    budget=stage_e_unconstrained_budget,
                    eval_id_start=eval_id,
                    recourse_maxiter=recourse_maxiter,
                    optimize_recourse=optimize_recourse,
                )

    combined = pd.DataFrame(rows)
    _write_dataframe(run_dir / "combined_stage_cde_physics_results.csv", combined)
    _write_json(run_dir / "combined_stage_cde_physics_results.json", {"rows": combined.to_dict(orient="records")})

    traces_df = pd.DataFrame(traces)
    _write_dataframe(run_dir / "physics_recourse_trace.csv", traces_df)

    best = _best_by_stage(combined)
    gap = _stage_e_vs_stage_d_gap(best)
    all_points = _all_candidate_points(combined)
    unique_points = _unique_candidate_points(all_points)
    frontier = _pareto_frontier_points(unique_points)
    traditional = _traditional_lambda_summary(best)
    traditional_comparison = _traditional_objective_comparison(traditional)

    _write_dataframe(run_dir / "best_by_stage.csv", best)
    _write_dataframe(run_dir / "stage_e_vs_stage_d_gap.csv", gap)
    _write_dataframe(run_dir / "all_candidate_points.csv", all_points)
    _write_dataframe(run_dir / "unique_candidate_points.csv", unique_points)
    _write_dataframe(run_dir / "pareto_frontier_points.csv", frontier)
    _write_dataframe(run_dir / "traditional_lambda_summary.csv", traditional)
    _write_dataframe(run_dir / "traditional_lambda_objective_comparison.csv", traditional_comparison)

    metadata = {
        **git_metadata(),
        "model": "gnn",
        "cases": cases,
        "case_mapping": USER_CASES,
        "candidate_set": compact_fraction_label("t", 0.30),
        "stage_c_psps_fraction": 0.10,
        "stage_c_proposal": "risk_only_transmission_heuristic",
        "stage_d_subsets": "k<=2 including no-shutoff",
        "stage_d_limit": None if stage_d_limit is None else int(stage_d_limit),
        "stage_e_constrained_k": 2,
        "stage_e_k2_budget": int(stage_e_k2_budget),
        "stage_e_unconstrained_budget": int(stage_e_unconstrained_budget),
        "recourse_maxiter": None if recourse_maxiter is None else int(recourse_maxiter),
        "continuous_recourse_optimized": bool(optimize_recourse),
        "rho_phys_values": [float(value) for value in rho_phys_values],
        "lambda_step": None if lambda_step is None else float(lambda_step),
        "lambda_settings": settings,
        "traditional_lambda_R_values": [float(value) for value in TRADITIONAL_LAMBDA_R],
        "final_wildfire_metric": "sum_l z_l * p_env_l * loading_l^2 / R_base",
        "final_wildfire_metric_excludes": ["I_l", "c_l", "impact", "consequence"],
        "load_shedding_metric": "demand-weighted GridFM-predicted service loss after applying each topology",
        "load_shedding_formula": "sum_i Pd_base_i * (1 - clip(Pd_pred_i / Pd_base_i, 0, 1)) / sum_i Pd_base_i",
        "pareto_frontier_objectives": ["R_norm", "L_shed"],
        "pareto_frontier_scope": "computed independently for each stage within fixed case and rho_phys",
        "pac_total_role": "objective penalty through rho_phys and diagnostic metadata; not a Pareto frontier axis",
        "continuous_recourse": (
            "SciPy reduced recourse over [Delta_Pg, alpha] for every evaluated topology unless row failure is recorded"
            if optimize_recourse
            else "disabled; fixed-control evaluation at clipped decision_vector.u_base"
        ),
        "outputs_are_additive": True,
    }
    _write_json(run_dir / "metadata.json", metadata)
    _plot_outputs(combined, traditional, frontier, plots_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the physics-aware Stage C/D/E case study.")
    parser.add_argument("--cases", nargs="+", choices=sorted(USER_CASES), default=["auto_env", "lfg"])
    parser.add_argument("--lambda-cases", nargs="+", choices=sorted(STAGE_E_LAMBDA_CASES), default=list(STAGE_E_LAMBDA_CASES))
    parser.add_argument("--lambda-step", type=float, default=None, help="Generate lambda_R = 0..1 sweep with this step; overrides --lambda-cases.")
    parser.add_argument("--rho-phys", nargs="+", type=float, default=DEFAULT_RHO_PHYS)
    parser.add_argument("--stage-e-budget", type=int, default=None, help="Backward-compatible alias that sets both Stage E budgets.")
    parser.add_argument("--stage-e-k2-budget", type=int, default=100)
    parser.add_argument("--stage-e-unconstrained-budget", type=int, default=100)
    parser.add_argument("--recourse-maxiter", type=int, default=10, help="Maximum SciPy iterations per topology recourse solve.")
    parser.add_argument("--no-continuous-recourse", action="store_true", help="Evaluate each topology at clipped u_base without SciPy recourse.")
    parser.add_argument("--stage-d-limit", type=int, default=None, help="Optional cap for Stage D topology evaluations; full run defaults to exhaustive k<=2.")
    parser.add_argument("--smoke", action="store_true", help="Use auto_env, balanced, rho={0,100}, and 2 Stage E evaluations.")
    parser.add_argument("--clear", action="store_true", help="Clear the physics case-study output root before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = ["auto_env"] if args.smoke else args.cases
    lambda_cases = ["balanced"] if args.smoke else args.lambda_cases
    lambda_step = None if args.smoke else args.lambda_step
    rho_phys = [0.0, 100.0] if args.smoke else args.rho_phys
    if args.stage_e_budget is not None:
        stage_e_k2_budget = int(args.stage_e_budget)
        stage_e_unconstrained_budget = int(args.stage_e_budget)
    else:
        stage_e_k2_budget = int(args.stage_e_k2_budget)
        stage_e_unconstrained_budget = int(args.stage_e_unconstrained_budget)
    if args.smoke:
        stage_e_k2_budget = 2
        stage_e_unconstrained_budget = 2
    stage_d_limit = 8 if args.smoke and args.stage_d_limit is None else args.stage_d_limit
    if args.clear and RESULT_ROOT.exists():
        expected_parent = CASE_STUDY_ROOT
        if RESULT_ROOT.parent != expected_parent:
            raise ValueError(f"Refusing to clear unexpected result root: {RESULT_ROOT}")
        _clear_tree(RESULT_ROOT)
    run_dirs = []
    for rho_value in rho_phys:
        output_root = RESULT_ROOT / _rho_output_folder(float(rho_value))
        run_dir = run_physics_infeasibility_case_study(
            cases=cases,
            lambda_cases=lambda_cases,
            lambda_step=lambda_step,
            rho_phys_values=[float(rho_value)],
            stage_e_k2_budget=stage_e_k2_budget,
            stage_e_unconstrained_budget=stage_e_unconstrained_budget,
            recourse_maxiter=int(args.recourse_maxiter),
            optimize_recourse=not bool(args.no_continuous_recourse),
            stage_d_limit=stage_d_limit,
            output_root=output_root,
            clear=False,
        )
        run_dirs.append(run_dir)
    for run_dir in run_dirs:
        print(f"Wrote physics infeasibility case study to {run_dir}")


if __name__ == "__main__":
    main()
