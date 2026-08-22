from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.gridfm_support.branch_metadata import canonicalize_line_ids, physical_line_ids
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
)
from experiments.test.wildfire_tests.stage_c_psps_baseline.stage_c_psps import (
    demand_weighted_load_shed_from_prediction,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.physics_infeasibility_evaluator import (
    DEFAULT_PHYSICS_WEIGHTS,
    _active_line_ids,
    _physics_components,
    _predict_topology_state,
    source_less_island_buses,
    weighted_pac_total,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    candidate_y_from_deenergized,
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
    _parse_line_ids,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    DecisionQualityScenario,
    HIGH_P_ENV_DEFAULT,
    LOW_P_ENV_DEFAULT,
    get_scenarios,
    p_env_for_scenario,
)


RESULT_ROOT = (
    RESULTS_ROOT
    / "leq"
    / "stage_f"
    / "decision_quality_analysis"
    / "with_physics_infeasibility"
    / "rho100"
)
STAGE_D = "stage_d_k2_exhaustive"
STAGE_E_K2 = "stage_e_k2"
STAGE_E_UNCONSTRAINED = "stage_e_unconstrained"
STAGES = [STAGE_D, STAGE_E_K2, STAGE_E_UNCONSTRAINED]
STAGE_LABELS = {
    STAGE_D: "Stage D exhaustive k<=2",
    STAGE_E_K2: "Stage E constrained K<=2",
    STAGE_E_UNCONSTRAINED: "Stage E unconstrained",
}
TRADITIONAL_LAMBDAS = [0.8, 0.5, 0.2]
EXPECTED_COMPARISON_LAMBDAS = [1.0, 0.8, 0.5, 0.2, 0.0]


def _lambda_values(step: float) -> List[float]:
    if step <= 0.0 or step > 1.0:
        raise ValueError("lambda_step must be in (0, 1].")
    count = int(round(1.0 / float(step)))
    values = [round(index * float(step), 10) for index in range(count + 1)]
    if not np.isclose(values[-1], 1.0):
        values.append(1.0)
    return [float(round(value, 2)) for value in values]


def _lambda_case(lambda_r: float) -> str:
    return f"lr{float(lambda_r):.2f}".replace(".", "p")


def _canonicalize_decision_quality_scenario(
    scenario_def: DecisionQualityScenario,
    grid_scenario,
) -> DecisionQualityScenario:
    return replace(
        scenario_def,
        target_high_risk_line_ids=tuple(canonicalize_line_ids(grid_scenario, scenario_def.target_high_risk_line_ids)),
        suppressed_line_ids=tuple(canonicalize_line_ids(grid_scenario, scenario_def.suppressed_line_ids)),
        expected_target_set=tuple(canonicalize_line_ids(grid_scenario, scenario_def.expected_target_set)),
    )


def _wrapped_line_ids(value, chunk_size: int = 6) -> str:
    values = _parse_line_ids(value)
    if not values:
        return ""
    chunks = [values[index : index + int(chunk_size)] for index in range(0, len(values), int(chunk_size))]
    return "\n".join(",".join(str(line_id) for line_id in chunk) for chunk in chunks)


def _fixed_topology_metrics(
    model_context: dict,
    scenario_def: DecisionQualityScenario,
    p_env_by_line: Dict[int, float],
    baseline_r: float,
    shutoff_line_ids: Iterable[int],
    eval_id: int,
    stage: str,
    topology_iteration: int,
    rho_phys: float,
    proxy_fields: Dict | None = None,
) -> Dict:
    scenario = model_context["scenario"]
    decision_vector = model_context["decision_vector"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    risk_line_ids = physical_line_ids(scenario)
    removed = sorted({int(line_id) for line_id in shutoff_line_ids})
    num_lines = int(_edge_array(scenario).shape[1])
    active = _active_line_ids(num_lines, removed, scenario=scenario)
    z_by_line = z_from_y(candidate_y_from_deenergized(candidate_line_ids, removed), num_lines)
    source_less = source_less_island_buses(scenario, removed)
    status = "ok"
    error = ""
    try:
        prediction, state, _keep = _predict_topology_state(
            model_context,
            np.asarray(decision_vector.u_base, dtype=float),
            removed,
        )
        components = _physics_components(
            model_context,
            np.asarray(decision_vector.u_base, dtype=float),
            state,
            active,
            source_less,
        )
        pac_total = weighted_pac_total(components, DEFAULT_PHYSICS_WEIGHTS)
        r_raw, exposure_by_line = compute_operational_wildfire_exposure(
            np.asarray(state["loading_ratio"], dtype=float),
            p_env_by_line,
            z_by_line,
            risk_line_ids,
        )
        r_norm = normalize_true_exposure(r_raw, baseline_r)
        l_shed = demand_weighted_load_shed_from_prediction(prediction, scenario) if removed else 0.0
        max_loading = float(state.get("max_loading_ratio", np.nan))
        min_voltage = float(state.get("min_voltage", np.nan))
        max_voltage = float(state.get("max_voltage", np.nan))
    except Exception as exc:
        status = "failed"
        error = str(exc)
        components = {name: np.nan for name in DEFAULT_PHYSICS_WEIGHTS}
        pac_total = np.nan
        r_raw = np.nan
        r_norm = np.nan
        l_shed = np.nan
        exposure_by_line = {}
        max_loading = np.nan
        min_voltage = np.nan
        max_voltage = np.nan

    proxy_fields = proxy_fields or {}
    return {
        "stage": stage,
        "stage_label": STAGE_LABELS[stage],
        "scenario_id": scenario_def.scenario_id,
        "scenario_name": scenario_def.scenario_name,
        "rho_phys": float(rho_phys),
        "eval_id": int(eval_id),
        "topology_iteration": int(topology_iteration),
        "candidate_set": "fixed_t0p30",
        "risk_scope": RISK_SCOPE,
        "num_candidate_lines": int(len(candidate_line_ids)),
        "num_risk_lines": int(len(risk_line_ids)),
        "num_shutoff_lines": int(len(removed)),
        "shutoff_line_ids": _csv_ints(removed),
        "line_id_key": _csv_ints(removed),
        "R_raw": float(r_raw) if np.isfinite(r_raw) else np.nan,
        "R_base_s": float(baseline_r),
        "R_norm": float(r_norm) if np.isfinite(r_norm) else np.nan,
        "L_shed": float(l_shed) if np.isfinite(l_shed) else np.nan,
        "PAC_total": float(pac_total) if np.isfinite(pac_total) else np.nan,
        "PAC_voltage_limits": float(components["voltage_limits"]) if np.isfinite(components["voltage_limits"]) else np.nan,
        "PAC_thermal_limits": float(components["thermal_limits"]) if np.isfinite(components["thermal_limits"]) else np.nan,
        "PAC_generator_limits": float(components["generator_limits"]) if np.isfinite(components["generator_limits"]) else np.nan,
        "PAC_island_source_feasibility": (
            float(components["island_source_feasibility"])
            if np.isfinite(components["island_source_feasibility"])
            else np.nan
        ),
        "source_less_bus_ids": _csv_ints(source_less),
        "creates_source_less_island": bool(len(source_less) > 0),
        "max_loading_ratio": max_loading,
        "min_voltage": min_voltage,
        "max_voltage": max_voltage,
        "fixed_control_u_base": True,
        "continuous_recourse_optimized": False,
        "status": status,
        "error": error,
        "proxy_R_hat": proxy_fields.get("proxy_R_hat", np.nan),
        "proxy_L_hat": proxy_fields.get("proxy_L_hat", np.nan),
        "proxy_objective": proxy_fields.get("proxy_objective", np.nan),
        "gurobi_objective": proxy_fields.get("gurobi_objective", np.nan),
        "gurobi_status": proxy_fields.get("gurobi_status", np.nan),
        "true_exposure_by_line": ";".join(
            f"{int(line_id)}:{float(value):.12g}" for line_id, value in sorted(exposure_by_line.items())
        ),
    }


def _score_rows(metrics: pd.DataFrame, lambda_r: float, rho_phys: float) -> pd.DataFrame:
    frame = metrics.copy()
    lambda_l = 1.0 - float(lambda_r)
    frame["lambda_case"] = _lambda_case(lambda_r)
    frame["lambda_R"] = float(lambda_r)
    frame["lambda_L"] = float(lambda_l)
    frame["J_no_phys"] = float(lambda_r) * frame["R_norm"] + lambda_l * frame["L_shed"]
    frame["J_true"] = frame["J_no_phys"] + float(rho_phys) * frame["PAC_total"]
    frame["risk_contribution"] = float(lambda_r) * frame["R_norm"]
    frame["load_contribution"] = lambda_l * frame["L_shed"]
    frame["physics_contribution"] = float(rho_phys) * frame["PAC_total"]
    return frame


def _best_by_stage(scored: pd.DataFrame) -> pd.DataFrame:
    ok = scored[scored["status"].eq("ok")].copy()
    if ok.empty:
        return ok
    keys = ["model_type", "scenario_id", "scenario_name", "lambda_R", "lambda_L", "rho_phys", "stage"]
    ordered = ok.sort_values(
        keys + ["J_true", "L_shed", "num_shutoff_lines", "line_id_key"],
        kind="mergesort",
    )
    return ordered.groupby(keys, as_index=False, dropna=False).first()


def _expected_outcome(scenario: DecisionQualityScenario, lambda_r: float) -> str:
    if np.isclose(lambda_r, 1.0):
        base = "Pure risk: expect one or two high-risk target lines if they are available in the candidate set."
    elif np.isclose(lambda_r, 0.8):
        base = "Risk priority: expect one or two target lines unless physics infeasibility makes them operationally unattractive."
    elif np.isclose(lambda_r, 0.5):
        base = "Balanced: expect a target subset only when risk reduction justifies load-service and physics consequences."
    elif np.isclose(lambda_r, 0.2):
        base = "Service priority: expect no shutoff or a low-impact target/alternative."
    else:
        base = "Pure service: the empty topology is normally expected unless another topology improves the service/physics metric."
    if scenario.scenario_id == "S2":
        return base + " Line 23 should become less attractive as service/physics weight increases."
    if scenario.scenario_id == "S4":
        return base + " The pair [77,79] should be avoided when the source-less-island physics residual is active."
    if scenario.scenario_id == "S5":
        return base + " Distributed lower-impact alternatives may be preferable to a concentrated local pair."
    return base


def _expected_vs_observed(best: pd.DataFrame, scenarios: List[DecisionQualityScenario]) -> pd.DataFrame:
    rows = []
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    selected = best[
        best["lambda_R"].astype(float).apply(
            lambda value: any(np.isclose(float(value), expected) for expected in EXPECTED_COMPARISON_LAMBDAS)
        )
    ].copy()
    for _, row in selected.iterrows():
        scenario = by_id[str(row["scenario_id"])]
        expected = set(int(line_id) for line_id in scenario.expected_target_set)
        observed = set(_parse_line_ids(row.get("shutoff_line_ids", "")))
        observed_targets = observed.intersection(expected)
        observed_non_targets = observed.difference(expected)
        missed_targets = expected.difference(observed)
        target_recall = float(len(observed_targets) / max(len(expected), 1))
        target_precision = float(len(observed_targets) / len(observed)) if observed else 0.0
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_name": scenario.scenario_name,
                "stage": row["stage"],
                "stage_label": row["stage_label"],
                "lambda_R": float(row["lambda_R"]),
                "lambda_L": float(row["lambda_L"]),
                "rho_phys": float(row["rho_phys"]),
                "expected_target_line_ids": _csv_ints(expected),
                "expected_possible_behavior": _expected_outcome(scenario, float(row["lambda_R"])),
                "observed_shutoff_line_ids": row.get("shutoff_line_ids", ""),
                "observed_target_subset": _csv_ints(observed_targets),
                "observed_non_target_lines": _csv_ints(observed_non_targets),
                "expected_targets_not_selected": _csv_ints(missed_targets),
                "num_expected_target_lines": int(len(expected)),
                "num_observed_shutoff_lines": int(len(observed)),
                "num_observed_target_lines": int(len(observed_targets)),
                "num_observed_non_target_lines": int(len(observed_non_targets)),
                "target_recall": target_recall,
                "target_precision": target_precision,
                "target_overlap_fraction": target_recall,
                "creates_source_less_island": bool(row.get("creates_source_less_island", False)),
                "R_norm": float(row["R_norm"]),
                "L_shed": float(row["L_shed"]),
                "PAC_total": float(row["PAC_total"]),
                "J_true": float(row["J_true"]),
            }
        )
    return pd.DataFrame(rows)


def _stage_e_vs_stage_d_gap(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["model_type", "scenario_id", "scenario_name", "lambda_R", "lambda_L", "rho_phys"]
    stage_d = best[best["stage"].eq(STAGE_D)]
    for stage_e_name in [STAGE_E_K2, STAGE_E_UNCONSTRAINED]:
        stage_e = best[best["stage"].eq(stage_e_name)]
        merged = stage_e.merge(stage_d, on=keys, suffixes=("_stage_e", "_stage_d"))
        for _, row in merged.iterrows():
            denominator = max(abs(float(row["J_true_stage_d"])), 1e-12)
            rows.append(
                {
                    **{key: row[key] for key in keys},
                    "stage_e_method": stage_e_name,
                    "stage_e_shutoff_line_ids": row["shutoff_line_ids_stage_e"],
                    "stage_d_shutoff_line_ids": row["shutoff_line_ids_stage_d"],
                    "stage_e_J_true": float(row["J_true_stage_e"]),
                    "stage_d_J_true": float(row["J_true_stage_d"]),
                    "gap_J_true": float(row["J_true_stage_e"] - row["J_true_stage_d"]),
                    "relative_gap_J_true": float(
                        (row["J_true_stage_e"] - row["J_true_stage_d"]) / denominator
                    ),
                    "stage_e_R_norm": float(row["R_norm_stage_e"]),
                    "stage_d_R_norm": float(row["R_norm_stage_d"]),
                    "stage_e_L_shed": float(row["L_shed_stage_e"]),
                    "stage_d_L_shed": float(row["L_shed_stage_d"]),
                    "stage_e_PAC_total": float(row["PAC_total_stage_e"]),
                    "stage_d_PAC_total": float(row["PAC_total_stage_d"]),
                }
            )
    return pd.DataFrame(rows)


def _nondominated_mask(frame: pd.DataFrame) -> np.ndarray:
    values = frame[["R_norm", "L_shed"]].to_numpy(dtype=float)
    finite = np.all(np.isfinite(values), axis=1)
    result = np.zeros(len(frame), dtype=bool)
    for index, point in enumerate(values):
        if not finite[index]:
            continue
        dominated = False
        for other_index, other in enumerate(values):
            if index == other_index or not finite[other_index]:
                continue
            if np.all(other <= point + 1e-12) and np.any(other < point - 1e-12):
                dominated = True
                break
        result[index] = not dominated
    return result


def _pareto_tables(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = scored[scored["status"].eq("ok")].copy()
    ok["line_id_key"] = ok["shutoff_line_ids"].fillna("").astype(str)
    unique = (
        ok.sort_values(["scenario_id", "stage", "line_id_key", "lambda_R", "J_true"], kind="mergesort")
        .drop_duplicates(["scenario_id", "stage", "line_id_key"], keep="first")
        .reset_index(drop=True)
    )
    pieces = []
    for _, group in unique.groupby(["scenario_id", "stage"], sort=True, dropna=False):
        local = group.copy()
        local["is_pareto_frontier"] = _nondominated_mask(local)
        pieces.append(local)
    points = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    frontier = points[points["is_pareto_frontier"]].copy() if not points.empty else pd.DataFrame()
    return points, frontier


def _plot_scenario_outputs(
    scenario_id: str,
    scored: pd.DataFrame,
    best: pd.DataFrame,
    expected_comparison: pd.DataFrame,
    pareto_points: pd.DataFrame,
    frontier: pd.DataFrame,
    output_dir: Path,
    rho_phys: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {STAGE_D: "#4C78A8", STAGE_E_K2: "#F58518", STAGE_E_UNCONSTRAINED: "#54A24B"}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    for ax, lambda_r in zip(axes, TRADITIONAL_LAMBDAS):
        local = scored[
            scored["scenario_id"].eq(scenario_id) & np.isclose(scored["lambda_R"].astype(float), lambda_r)
        ].copy()
        if local.empty:
            ax.set_title(f"lambda_R={lambda_r:.1f}, lambda_L={1-lambda_r:.1f}")
            ax.text(0.5, 0.5, "Not evaluated in this run", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        topology_lines = []
        max_iterations = max(int(local["topology_iteration"].max()), 1)
        for stage in STAGES:
            group = local[local["stage"].eq(stage)].sort_values("topology_iteration", kind="mergesort")
            if group.empty:
                continue
            best_so_far = group["J_true"].astype(float).cummin()
            ax.step(
                group["topology_iteration"],
                best_so_far,
                where="post",
                color=colors[stage],
                linewidth=2,
                label=STAGE_LABELS[stage],
            )
            final_row = group.loc[group["J_true"].idxmin()]
            ax.scatter(
                [int(group["topology_iteration"].iloc[-1])],
                [float(best_so_far.iloc[-1])],
                color="#D62728",
                s=38,
                zorder=5,
            )
            topology_lines.append(f"{STAGE_LABELS[stage]}: [{final_row['shutoff_line_ids'] or 'none'}]")
        ax.text(
            0.98,
            0.78,
            "\n".join(topology_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox={"facecolor": "white", "edgecolor": "#C8C8C8", "alpha": 0.88, "pad": 4},
        )
        ax.set_xlim(left=0, right=max_iterations * 1.03)
        ax.set_title(f"lambda_R={lambda_r:.1f}, lambda_L={1-lambda_r:.1f}")
        ax.set_xlabel("Topology iteration")
        ax.set_ylabel("Best topology objective so far, J_true")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(
        f"{scenario_id}: Physics-Aware Decision Quality, rho_phys={rho_phys:g}\n"
        f"J = lambda_R R_norm + lambda_L L_shed + {rho_phys:g} PAC_total"
    )
    fig.savefig(output_dir / "traditional_lambda_objective_convergence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    for ax, stage in zip(axes, STAGES):
        points = pareto_points[
            pareto_points["scenario_id"].eq(scenario_id) & pareto_points["stage"].eq(stage)
        ]
        front = frontier[frontier["scenario_id"].eq(scenario_id) & frontier["stage"].eq(stage)]
        ax.scatter(points["L_shed"], points["R_norm"], s=20, color="#B8B8B8", alpha=0.35, label="Evaluated")
        if not front.empty:
            ordered = front.sort_values(["L_shed", "R_norm"], kind="mergesort")
            ax.scatter(ordered["L_shed"], ordered["R_norm"], s=48, color="#D62728", label="Nondominated")
            ax.plot(ordered["L_shed"], ordered["R_norm"], color="#D62728", linewidth=1.4)
        ax.set_title(STAGE_LABELS[stage])
        ax.set_xlabel("GridFM-predicted load shed, L_shed")
        ax.set_ylabel("Wildfire exposure, R_norm")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"{scenario_id}: Stage-wise Pareto Frontiers, lambda_R=0..1 step 0.05")
    fig.savefig(output_dir / "pareto_frontier_scatter.png", dpi=180)
    plt.close(fig)

    comparison = expected_comparison[expected_comparison["scenario_id"].eq(scenario_id)].copy()
    if not comparison.empty:
        expected_targets = str(comparison["expected_target_line_ids"].iloc[0])
        fig, axes = plt.subplots(
            len(STAGES),
            len(EXPECTED_COMPARISON_LAMBDAS),
            figsize=(18, 8),
            constrained_layout=True,
        )
        for row_index, stage in enumerate(STAGES):
            for column_index, lambda_r in enumerate(EXPECTED_COMPARISON_LAMBDAS):
                ax = axes[row_index, column_index]
                row = comparison[
                    comparison["stage"].eq(stage)
                    & np.isclose(comparison["lambda_R"].astype(float), lambda_r)
                ]
                ax.set_xticks([])
                ax.set_yticks([])
                if row.empty:
                    ax.set_facecolor("#F2F2F2")
                    ax.text(0.5, 0.5, "Not evaluated", ha="center", va="center", transform=ax.transAxes)
                    continue
                item = row.iloc[0]
                recall = float(item.get("target_recall", item.get("target_overlap_fraction", 0.0)))
                precision = float(item.get("target_precision", 0.0))
                creates_island = bool(item["creates_source_less_island"])
                if creates_island:
                    facecolor = "#F8C9C9"
                elif recall > 0.0:
                    facecolor = "#D8EFD3"
                elif str(item["observed_shutoff_line_ids"]).strip() in {"", "nan"}:
                    facecolor = "#E8E8E8"
                else:
                    facecolor = "#F6E3B4"
                ax.set_facecolor(facecolor)
                observed = _wrapped_line_ids(item["observed_shutoff_line_ids"])
                target_hits = _wrapped_line_ids(item["observed_target_subset"])
                non_targets = _wrapped_line_ids(item["observed_non_target_lines"])
                text = (
                    f"Selected:\n[{observed}]\n"
                    f"Target hits: [{target_hits}]\n"
                    f"Other:\n[{non_targets}]\n"
                    f"Recall={recall:.2f}, precision={precision:.2f}\n"
                    f"PAC={float(item['PAC_total']):.3g}\n"
                    f"Island={'yes' if creates_island else 'no'}"
                )
                fontsize = 6.2 if len(_parse_line_ids(item["observed_shutoff_line_ids"])) > 10 else 8
                ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=fontsize, transform=ax.transAxes)
                if row_index == 0:
                    ax.set_title(f"lambda_R={lambda_r:.1f}", fontsize=10)
                if column_index == 0:
                    ax.set_ylabel(STAGE_LABELS[stage], fontsize=9)
        fig.suptitle(
            f"{scenario_id}: Expected Versus Selected Shutoff Lines, rho_phys={rho_phys:g}\n"
            f"Expected target set: [{expected_targets}]"
        )
        fig.savefig(output_dir / "expected_vs_selected_shutoff_lines.png", dpi=180)
        plt.close(fig)


def run_stage_f_physics_decision_quality(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    rho_phys: float = 100.0,
    lambda_step: float = 0.05,
    max_deenergized_lines: int = 2,
    stage_e_budget: int = 100,
    stage_d_limit: int | None = None,
    clear: bool = False,
    result_root: Path | None = None,
    stage_name: str = "F",
    study_name: str = "decision_quality_with_physics_infeasibility",
    p_env_low_value: float = LOW_P_ENV_DEFAULT,
    p_env_high_value: float = HIGH_P_ENV_DEFAULT,
    p_env_suppressed_value: float | None = None,
    p_env_extra_suppressed_line_ids: Iterable[int] = (),
) -> Path:
    models = ["gnn"] if models is None else list(models)
    scenarios = get_scenarios(scenario_ids)
    lambdas = _lambda_values(lambda_step)
    default_root = RESULT_ROOT if np.isclose(rho_phys, 100.0) else RESULT_ROOT.parent / f"rho{rho_phys:g}"
    root = Path(result_root) if result_root is not None else default_root
    if clear and root.exists():
        resolved = root.resolve()
        expected_parent = root.parent.resolve()
        if resolved.parent != expected_parent:
            raise ValueError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    run_dir = make_run_dir(root, "run")
    plots_dir = run_dir / "plots"
    all_scored = []
    model_metadata = []
    scenarios_for_reporting = scenarios
    started = time.perf_counter()

    for model_type in models:
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model_type={model_type}.")
        context = _build_model_context(model_type, grouping_top_fraction=0.30)
        write_config_copy(context["config"], run_dir / f"config_{model_type}.yaml")
        scenario = context["scenario"]
        candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
        num_lines = int(_edge_array(scenario).shape[1])
        _validate_candidate_line_ids(candidate_line_ids, num_lines)
        baseline_loading = np.asarray(context["baseline_state"]["loading_ratio"], dtype=float)
        c_by_line = _consequence_by_line(context["consequence_df"])
        extra_suppressed_line_ids = canonicalize_line_ids(scenario, p_env_extra_suppressed_line_ids)
        canonical_scenarios = [
            _canonicalize_decision_quality_scenario(scenario_def, scenario) for scenario_def in scenarios
        ]
        scenarios_for_reporting = canonical_scenarios

        for scenario_def in canonical_scenarios:
            p_env = p_env_for_scenario(
                scenario_def,
                num_lines,
                low_value=p_env_low_value,
                high_value=p_env_high_value,
                suppressed_value=p_env_suppressed_value,
                extra_suppressed_line_ids=extra_suppressed_line_ids,
            )
            baseline_r, _ = _scenario_baseline_exposure(
                baseline_loading,
                p_env,
                physical_line_ids(scenario),
            )
            subsets = enumerate_deenergization_subsets(
                candidate_line_ids,
                max_deenergized_lines=max_deenergized_lines,
            )
            if stage_d_limit is not None:
                subsets = subsets[: int(stage_d_limit)]
            stage_d_metrics = pd.DataFrame(
                [
                    _fixed_topology_metrics(
                        context,
                        scenario_def,
                        p_env,
                        baseline_r,
                        subset,
                        eval_id=index,
                        stage=STAGE_D,
                        topology_iteration=index + 1,
                        rho_phys=rho_phys,
                    )
                    for index, subset in enumerate(subsets)
                ]
            )
            stage_d_metrics["model_type"] = model_type
            for lambda_r in lambdas:
                all_scored.append(_score_rows(stage_d_metrics, lambda_r, rho_phys))

            for lambda_r in lambdas:
                lambda_l = 1.0 - lambda_r
                for stage, budget in [(STAGE_E_K2, max_deenergized_lines), (STAGE_E_UNCONSTRAINED, None)]:
                    evaluated_y = []
                    rows = []
                    for iteration in range(int(stage_e_budget)):
                        proposal = solve_gurobi_master_next_candidate(
                            candidate_line_ids,
                            p_env,
                            baseline_loading,
                            c_by_line,
                            lambda_r,
                            lambda_l,
                            max_deenergized_lines=budget,
                            evaluated_y_vectors=evaluated_y,
                            proxy_type=DEFAULT_PROXY_TYPE,
                        )
                        y_by_line = {int(key): int(value) for key, value in proposal["y_by_line"].items()}
                        evaluated_y.append(y_by_line)
                        rows.append(
                            _fixed_topology_metrics(
                                context,
                                scenario_def,
                                p_env,
                                baseline_r,
                                deenergized_from_y(y_by_line),
                                eval_id=iteration,
                                stage=stage,
                                topology_iteration=iteration + 1,
                                rho_phys=rho_phys,
                                proxy_fields=proposal,
                            )
                        )
                    frame = pd.DataFrame(rows)
                    frame["model_type"] = model_type
                    all_scored.append(_score_rows(frame, lambda_r, rho_phys))
                print(
                    f"[{model_type} {scenario_def.scenario_id}] completed lambda_R={lambda_r:.2f}",
                    flush=True,
                )

        model_metadata.append(
            {
                "model_type": model_type,
                "candidate_line_ids": candidate_line_ids,
                "num_candidate_lines": len(candidate_line_ids),
                "num_lines": num_lines,
            }
        )

    scored = pd.concat(all_scored, ignore_index=True, sort=False)
    best = _best_by_stage(scored)
    expected = _expected_vs_observed(best, scenarios_for_reporting)
    gap = _stage_e_vs_stage_d_gap(best)
    pareto_points, frontier = _pareto_tables(scored)
    write_dataframe(run_dir / "all_physics_aware_evaluations.csv", scored)
    write_dataframe(run_dir / "best_by_scenario_lambda_stage.csv", best)
    write_dataframe(run_dir / "expected_vs_observed_line_subsets.csv", expected)
    write_dataframe(run_dir / "stage_e_vs_stage_d_gap.csv", gap)
    write_dataframe(run_dir / "pareto_unique_points.csv", pareto_points)
    write_dataframe(run_dir / "pareto_frontier_points.csv", frontier)
    write_json(
        run_dir / "scenario_definitions.json",
        {
            "low_p_env": float(p_env_low_value),
            "high_p_env": float(p_env_high_value),
            "suppressed_p_env": (
                None if p_env_suppressed_value is None else float(p_env_suppressed_value)
            ),
            "extra_suppressed_line_ids": sorted({int(line_id) for line_id in p_env_extra_suppressed_line_ids}),
            "scenarios": [scenario.to_dict() for scenario in scenarios_for_reporting],
        },
    )
    write_json(
        run_dir / "metadata.json",
        {
            **git_metadata(),
            "stage": stage_name,
            "study": study_name,
            "rho_phys": float(rho_phys),
            "lambda_values": lambdas,
            "lambda_step": float(lambda_step),
            "traditional_lambda_values": TRADITIONAL_LAMBDAS,
            "expected_comparison_lambda_values": EXPECTED_COMPARISON_LAMBDAS,
            "stage_e_budget_per_lambda": int(stage_e_budget),
            "max_deenergized_lines_stage_d_and_stage_e_k2": int(max_deenergized_lines),
            "stage_e_unconstrained": True,
            "fixed_control_u_base": True,
            "continuous_recourse_optimized": False,
            "source_less_island_handling": "PAC penalty only; no alpha forcing in fixed-control study",
            "true_objective": "lambda_R * R_norm + lambda_L * L_shed + rho_phys * PAC_total",
            "risk_scope": RISK_SCOPE,
            "pareto_scope": "independent nondominated R_norm/L_shed frontier per scenario and stage",
            "runtime_seconds": float(time.perf_counter() - started),
            "model_metadata": model_metadata,
            "p_env_low_value": float(p_env_low_value),
            "p_env_high_value": float(p_env_high_value),
            "p_env_suppressed_value": (
                None if p_env_suppressed_value is None else float(p_env_suppressed_value)
            ),
            "p_env_extra_suppressed_line_ids": sorted({int(line_id) for line_id in p_env_extra_suppressed_line_ids}),
        },
    )
    for scenario in scenarios_for_reporting:
        _plot_scenario_outputs(
            scenario.scenario_id,
            scored,
            best,
            expected,
            pareto_points,
            frontier,
            plots_dir / scenario.scenario_id,
            rho_phys,
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-control Stage F decision quality with a physics-infeasibility penalty."
    )
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gnn"])
    parser.add_argument("--scenarios", nargs="+", choices=[scenario.scenario_id for scenario in get_scenarios()], default=None)
    parser.add_argument("--rho-phys", type=float, default=100.0)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--stage-e-budget", type=int, default=100)
    parser.add_argument("--stage-d-limit", type=int, default=None)
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_stage_f_physics_decision_quality(
        models=args.models,
        scenario_ids=args.scenarios,
        rho_phys=args.rho_phys,
        lambda_step=args.lambda_step,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_budget=args.stage_e_budget,
        stage_d_limit=args.stage_d_limit,
        clear=args.clear,
    )
    print(f"Wrote physics-aware Stage F decision-quality results to {run_dir}")


if __name__ == "__main__":
    main()
