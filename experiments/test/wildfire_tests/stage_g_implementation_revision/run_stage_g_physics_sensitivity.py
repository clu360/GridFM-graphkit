from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import canonicalize_line_ids, physical_line_ids
from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.stage_c_psps_baseline.run_stage_c_psps_baseline import (
    MODEL_CONFIGS,
    _build_model_context,
)
from experiments.test.wildfire_tests.stage_d_deenergization.stage_d_deenergization import (
    enumerate_deenergization_subsets,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    solve_gurobi_master_next_candidate,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    DEFAULT_PROXY_TYPE,
    deenergized_from_y,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    _consequence_by_line,
    _csv_ints,
    _edge_array,
    _parse_line_ids,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    EXPECTED_COMPARISON_LAMBDAS,
    STAGE_D,
    STAGE_E_K2,
    STAGE_E_UNCONSTRAINED,
    STAGE_LABELS,
    STAGES,
    TRADITIONAL_LAMBDAS,
    _best_by_stage,
    _canonicalize_decision_quality_scenario,
    _expected_vs_observed,
    _fixed_topology_metrics,
    _lambda_case,
    _lambda_values,
    _nondominated_mask,
    _pareto_tables,
    _plot_scenario_outputs,
    _score_rows,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    HIGH_P_ENV_DEFAULT,
    LOW_P_ENV_DEFAULT,
    DecisionQualityScenario,
    get_scenarios,
    p_env_for_scenario,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_decision_quality import (
    DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_g" / "physics_infeasibility_sensitivity"
DEFAULT_RHO_VALUES = [0.0, 10.0, 20.0, 50.0, 100.0]
P_ENV_CALIBRATION_MODE = "rank_calibrated_non_target_gridfm_outliers"
P_ENV_CALIBRATION_FLOOR = 1e-6
P_ENV_TARGET_MARGIN = 0.9


def _line_id_key(line_ids: Iterable[int]) -> str:
    return _csv_ints(sorted({int(line_id) for line_id in line_ids}))


def calibrate_p_env_for_scenario(
    scenario_def: DecisionQualityScenario,
    num_lines: int,
    physical_ids: Sequence[int],
    baseline_loading: Sequence[float],
    impact_by_line: Dict[int, float],
    outlier_line_ids: Iterable[int],
    low_value: float = LOW_P_ENV_DEFAULT,
    high_value: float = HIGH_P_ENV_DEFAULT,
    floor: float = P_ENV_CALIBRATION_FLOOR,
    target_margin: float = P_ENV_TARGET_MARGIN,
) -> tuple[Dict[int, float], pd.DataFrame]:
    physical_ids = [int(line_id) for line_id in physical_ids]
    outlier_ids = {int(line_id) for line_id in outlier_line_ids}
    target_ids = {int(line_id) for line_id in scenario_def.target_high_risk_line_ids}
    p_env = p_env_for_scenario(
        scenario_def,
        num_lines,
        low_value=low_value,
        high_value=high_value,
    )
    coeff = {
        line_id: float(baseline_loading[line_id]) ** 2 * float(impact_by_line.get(line_id, 0.0))
        for line_id in physical_ids
    }
    target_scores = [float(high_value) * coeff[line_id] for line_id in target_ids if line_id in coeff]
    positive_target_scores = [score for score in target_scores if np.isfinite(score) and score > 0.0]
    weakest_target_score = min(positive_target_scores) if positive_target_scores else np.nan
    old_p_env = {line_id: float(p_env.get(line_id, low_value)) for line_id in physical_ids}

    for line_id in physical_ids:
        if line_id not in outlier_ids or line_id in target_ids:
            continue
        line_coeff = coeff[line_id]
        if not np.isfinite(weakest_target_score) or line_coeff <= 0.0:
            p_env[line_id] = float(low_value)
            continue
        calibrated = min(float(low_value), float(target_margin) * float(weakest_target_score) / float(line_coeff))
        p_env[line_id] = max(float(floor), float(calibrated))

    old_scores = {line_id: old_p_env[line_id] * coeff[line_id] for line_id in physical_ids}
    new_scores = {line_id: float(p_env.get(line_id, 0.0)) * coeff[line_id] for line_id in physical_ids}
    old_rank = {line_id: rank for rank, line_id in enumerate(sorted(physical_ids, key=lambda x: (-old_scores[x], x)), start=1)}
    new_rank = {line_id: rank for rank, line_id in enumerate(sorted(physical_ids, key=lambda x: (-new_scores[x], x)), start=1)}

    rows = []
    for line_id in physical_ids:
        rows.append(
            {
                "scenario_id": scenario_def.scenario_id,
                "line_id": int(line_id),
                "is_target": bool(line_id in target_ids),
                "is_gridfm_heavy_loading_outlier": bool(line_id in outlier_ids),
                "old_p_env": float(old_p_env[line_id]),
                "calibrated_p_env": float(p_env.get(line_id, 0.0)),
                "loading_ratio": float(baseline_loading[line_id]),
                "impact": float(impact_by_line.get(line_id, 0.0)),
                "outlier_coeff_loading_squared_impact": float(coeff[line_id]),
                "weakest_target_baseline_score": float(weakest_target_score) if np.isfinite(weakest_target_score) else np.nan,
                "old_baseline_risk_score": float(old_scores[line_id]),
                "calibrated_baseline_risk_score": float(new_scores[line_id]),
                "old_rank": int(old_rank[line_id]),
                "new_rank": int(new_rank[line_id]),
                "calibration_mode": P_ENV_CALIBRATION_MODE,
            }
        )
    return {int(k): float(v) for k, v in p_env.items()}, pd.DataFrame(rows)


def rescore_topology_pool(metric_pool: pd.DataFrame, rho_values: Sequence[float]) -> pd.DataFrame:
    pieces = []
    for rho in rho_values:
        frame = metric_pool.copy()
        lambda_r = frame["lambda_R"].astype(float)
        lambda_l = 1.0 - lambda_r
        frame["rho_phys"] = float(rho)
        frame["lambda_L"] = lambda_l
        frame["lambda_case"] = frame["lambda_R"].astype(float).map(_lambda_case)
        frame["J_no_phys"] = lambda_r * frame["R_norm"].astype(float) + lambda_l * frame["L_shed"].astype(float)
        frame["J_true"] = frame["J_no_phys"] + float(rho) * frame["PAC_total"].astype(float)
        frame["risk_contribution"] = lambda_r * frame["R_norm"].astype(float)
        frame["load_contribution"] = lambda_l * frame["L_shed"].astype(float)
        frame["physics_contribution"] = float(rho) * frame["PAC_total"].astype(float)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True, sort=False)


def methodology_fidelity_checks(
    metric_pool: pd.DataFrame,
    rescored: pd.DataFrame,
    calibration: pd.DataFrame,
    scenario,
    physical_ids: Sequence[int],
    rho_values: Sequence[float],
    candidate_line_ids: Sequence[int],
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, severity: str, details: str) -> None:
        rows.append({"check_name": name, "passed": bool(passed), "severity": severity, "details": details})

    target_rows = calibration[calibration["is_target"].astype(bool)]
    add(
        "explicit_targets_keep_high_p_env",
        target_rows.empty or np.allclose(target_rows["calibrated_p_env"].astype(float), HIGH_P_ENV_DEFAULT),
        "hard",
        "Explicit scenario targets must remain at p_env=1.0.",
    )
    ordinary = calibration[
        ~calibration["is_target"].astype(bool) & ~calibration["is_gridfm_heavy_loading_outlier"].astype(bool)
    ]
    add(
        "ordinary_non_targets_keep_low_p_env",
        ordinary.empty or np.allclose(ordinary["calibrated_p_env"].astype(float), LOW_P_ENV_DEFAULT),
        "hard",
        "Ordinary non-target physical lines must remain at p_env=0.05.",
    )
    reduced = calibration[
        ~calibration["is_target"].astype(bool) & calibration["is_gridfm_heavy_loading_outlier"].astype(bool)
    ]
    add(
        "only_non_target_outliers_reduced",
        reduced.empty or (reduced["calibrated_p_env"].astype(float) <= LOW_P_ENV_DEFAULT + 1e-12).all(),
        "hard",
        "Only non-target GridFM-heavy-loading outlier lines may be reduced below the ordinary low value.",
    )
    physical_set = {int(line_id) for line_id in physical_ids}
    add(
        "candidate_lines_are_physical",
        set(int(line_id) for line_id in candidate_line_ids).issubset(physical_set),
        "hard",
        "Topology candidates must use canonical physical line IDs.",
    )
    self_loop_count = int(np.sum(np.asarray(scenario.is_self_loop)[list(physical_set)])) if physical_set else 0
    add(
        "risk_candidates_exclude_self_loops",
        self_loop_count == 0,
        "hard",
        f"Self-loop count among physical risk lines: {self_loop_count}.",
    )
    canonical_values = [int(scenario.canonical_line_id[line_id]) for line_id in physical_set]
    add(
        "risk_candidates_exclude_directed_duplicates",
        len(canonical_values) == len(set(canonical_values)),
        "hard",
        "Each physical risk line must have one canonical branch ID.",
    )
    add(
        "fixed_control_no_continuous_recourse",
        metric_pool["continuous_recourse_optimized"].eq(False).all()
        and metric_pool["fixed_control_u_base"].eq(True).all(),
        "hard",
        "This case study must remain fixed-control with no continuous recourse.",
    )
    rho_count = rescored["rho_phys"].nunique()
    add(
        "rho_values_present",
        rho_count == len(set(float(value) for value in rho_values)),
        "hard",
        f"Found {rho_count} rho values; expected {len(set(float(value) for value in rho_values))}.",
    )
    invariant_columns = ["R_norm", "L_shed", "PAC_total", "R_raw", "R_base_s"]
    drift = []
    for _, group in rescored.groupby("topology_pool_id", sort=False):
        for column in invariant_columns:
            values = group[column].astype(float).to_numpy()
            finite = values[np.isfinite(values)]
            if len(finite) and float(np.max(finite) - np.min(finite)) > 1e-10:
                drift.append((int(group["topology_pool_id"].iloc[0]), column))
                break
    add(
        "rho_does_not_change_intrinsic_metrics",
        not drift,
        "hard",
        f"Intrinsic metric drift examples: {drift[:5]}",
    )
    required_columns = [
        "p_env_calibration_mode",
        "rho_phys",
        "lambda_R",
        "lambda_L",
        "stage",
        "shutoff_line_ids",
        "J_true",
        "J_no_phys",
        "risk_contribution",
        "load_contribution",
        "physics_contribution",
    ]
    missing = [column for column in required_columns if column not in rescored.columns]
    add(
        "rescored_table_records_objective_components",
        not missing,
        "hard",
        f"Missing columns: {missing}",
    )
    return pd.DataFrame(rows)


def _metric_pool_id(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["topology_pool_id"] = np.arange(len(frame), dtype=int)
    return frame


def _build_metric_pool_for_model(
    model_type: str,
    scenario_ids: List[str] | None,
    lambda_step: float,
    max_deenergized_lines: int,
    stage_e_budget: int,
    stage_d_limit: int | None,
    outlier_line_ids: Sequence[int],
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, List[DecisionQualityScenario], dict]:
    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    write_config_copy(context["config"], run_dir / "inputs" / f"config_{model_type}.yaml")
    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    num_lines = int(_edge_array(scenario).shape[1])
    _validate_candidate_line_ids(candidate_line_ids, num_lines)
    baseline_loading = np.asarray(context["baseline_state"]["loading_ratio"], dtype=float)
    c_by_line = _consequence_by_line(context["consequence_df"])
    physical_ids = physical_line_ids(scenario)
    canonical_outlier_ids = canonicalize_line_ids(scenario, outlier_line_ids)
    canonical_scenarios = [
        _canonicalize_decision_quality_scenario(scenario_def, scenario)
        for scenario_def in get_scenarios(scenario_ids)
    ]
    lambdas = _lambda_values(lambda_step)
    metric_rows = []
    calibration_frames = []

    for scenario_def in canonical_scenarios:
        p_env, calibration = calibrate_p_env_for_scenario(
            scenario_def,
            num_lines,
            physical_ids,
            baseline_loading,
            c_by_line,
            canonical_outlier_ids,
        )
        calibration_frames.append(calibration)
        baseline_r, _ = _scenario_baseline_exposure(baseline_loading, p_env, physical_ids)
        subsets = enumerate_deenergization_subsets(candidate_line_ids, max_deenergized_lines=max_deenergized_lines)
        if stage_d_limit is not None:
            subsets = subsets[: int(stage_d_limit)]
        stage_d_rows = [
            _fixed_topology_metrics(
                context,
                scenario_def,
                p_env,
                baseline_r,
                subset,
                eval_id=index,
                stage=STAGE_D,
                topology_iteration=index + 1,
                rho_phys=0.0,
            )
            for index, subset in enumerate(subsets)
        ]
        stage_d_frame = pd.DataFrame(stage_d_rows)
        stage_d_frame["model_type"] = model_type
        stage_d_frame["p_env_calibration_mode"] = P_ENV_CALIBRATION_MODE
        for lambda_r in lambdas:
            local = stage_d_frame.copy()
            local["lambda_R"] = float(lambda_r)
            local["lambda_L"] = 1.0 - float(lambda_r)
            local["lambda_case"] = _lambda_case(lambda_r)
            local["generation_lambda_R"] = float(lambda_r)
            local["generation_lambda_L"] = 1.0 - float(lambda_r)
            metric_rows.append(local)

        for lambda_r in lambdas:
            lambda_l = 1.0 - float(lambda_r)
            for stage, budget in [(STAGE_E_K2, max_deenergized_lines), (STAGE_E_UNCONSTRAINED, None)]:
                evaluated_y = []
                rows = []
                for iteration in range(int(stage_e_budget)):
                    proposal = solve_gurobi_master_next_candidate(
                        candidate_line_ids,
                        p_env,
                        baseline_loading,
                        c_by_line,
                        float(lambda_r),
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
                            rho_phys=0.0,
                            proxy_fields=proposal,
                        )
                    )
                frame = pd.DataFrame(rows)
                frame["model_type"] = model_type
                frame["p_env_calibration_mode"] = P_ENV_CALIBRATION_MODE
                frame["lambda_R"] = float(lambda_r)
                frame["lambda_L"] = lambda_l
                frame["lambda_case"] = _lambda_case(lambda_r)
                frame["generation_lambda_R"] = float(lambda_r)
                frame["generation_lambda_L"] = lambda_l
                metric_rows.append(frame)
            print(f"[{model_type} {scenario_def.scenario_id}] metric pool completed lambda_R={lambda_r:.2f}", flush=True)

    metric_pool = _metric_pool_id(pd.concat(metric_rows, ignore_index=True, sort=False))
    calibration = pd.concat(calibration_frames, ignore_index=True, sort=False)
    metadata = {
        "scenario": scenario,
        "candidate_line_ids": candidate_line_ids,
        "physical_line_ids": physical_ids,
        "canonical_outlier_line_ids": canonical_outlier_ids,
        "num_lines": num_lines,
    }
    return metric_pool, calibration, canonical_scenarios, metadata


def _write_per_rho_outputs(
    rescored: pd.DataFrame,
    scenarios: List[DecisionQualityScenario],
    rho_values: Sequence[float],
    plots_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    best_all = _best_by_stage(rescored)
    expected_all = _expected_vs_observed(best_all, scenarios)
    for rho in rho_values:
        scored_rho = rescored[np.isclose(rescored["rho_phys"].astype(float), float(rho))].copy()
        best_rho = best_all[np.isclose(best_all["rho_phys"].astype(float), float(rho))].copy()
        expected_rho = expected_all[np.isclose(expected_all["rho_phys"].astype(float), float(rho))].copy()
        pareto_points, frontier = _pareto_tables(scored_rho)
        rho_dir = plots_dir / "per_rho" / f"rho{float(rho):g}"
        for scenario_def in scenarios:
            _plot_scenario_outputs(
                scenario_def.scenario_id,
                scored_rho,
                best_rho,
                expected_rho,
                pareto_points,
                frontier,
                rho_dir / scenario_def.scenario_id,
                float(rho),
            )
    return best_all, expected_all, rescored


def _selected_contains_line(value: str, line_id: int) -> bool:
    return int(line_id) in set(_parse_line_ids(value))


def _plot_cross_rho(best: pd.DataFrame, plots_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    physics_rows = []
    cost_rows = []
    frequency_rows = []
    rho_values = sorted(float(value) for value in best["rho_phys"].dropna().unique())
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(rho_values), 1)))
    color_by_rho = {rho: colors[index] for index, rho in enumerate(rho_values)}

    for (scenario_id, stage), group in best.groupby(["scenario_id", "stage"], sort=True):
        output_dir = plots_dir / "cross_rho" / str(scenario_id) / str(stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        for rho in rho_values:
            local = group[np.isclose(group["rho_phys"].astype(float), rho)].sort_values("lambda_R")
            if local.empty:
                continue
            ax.plot(local["R_norm"], local["L_shed"], marker="o", linewidth=1.5, markersize=3.5, color=color_by_rho[rho], label=f"rho={rho:g}")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Risk/Load Selected Path")
        ax.set_xlabel("R_norm")
        ax.set_ylabel("L_shed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(output_dir / "risk_load_pareto_overlay_all_lambdas.png", dpi=180)
        plt.close(fig)

        unique = group.drop_duplicates(["rho_phys", "line_id_key", "R_norm", "L_shed"]).copy()
        unique["is_nondominated_selected"] = _nondominated_mask(unique)
        fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
        ax.scatter(unique["R_norm"], unique["L_shed"], color="#C8C8C8", alpha=0.35, s=28, label="Selected")
        nd = unique[unique["is_nondominated_selected"]]
        for rho in rho_values:
            local = nd[np.isclose(nd["rho_phys"].astype(float), rho)]
            if local.empty:
                continue
            ax.scatter(local["R_norm"], local["L_shed"], color=color_by_rho[rho], s=55, label=f"rho={rho:g}")
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Nondominated Selected Points")
        ax.set_xlabel("R_norm")
        ax.set_ylabel("L_shed")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(output_dir / "nondominated_points_colored_by_rho_all_lambdas.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for lambda_r in TRADITIONAL_LAMBDAS:
            local = group[np.isclose(group["lambda_R"].astype(float), lambda_r)].sort_values("rho_phys")
            if local.empty:
                continue
            ax.plot(local["rho_phys"], local["PAC_total"], marker="o", linewidth=2, label=f"lambda_R={lambda_r:g}")
            for _, row in local.iterrows():
                physics_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage": stage,
                        "lambda_R": float(row["lambda_R"]),
                        "rho_phys": float(row["rho_phys"]),
                        "PAC_total": float(row["PAC_total"]),
                        "PAC_voltage_limits": float(row["PAC_voltage_limits"]),
                        "PAC_thermal_limits": float(row["PAC_thermal_limits"]),
                        "PAC_generator_limits": float(row["PAC_generator_limits"]),
                        "PAC_island_source_feasibility": float(row["PAC_island_source_feasibility"]),
                        "shutoff_line_ids": row["shutoff_line_ids"],
                    }
                )
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Physics Sensitivity")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("PAC_total")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(output_dir / "physics_feasibility_sensitivity_traditional_lambdas.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)
        for lambda_r in TRADITIONAL_LAMBDAS:
            local = group[np.isclose(group["lambda_R"].astype(float), lambda_r)].sort_values("rho_phys")
            if local.empty:
                continue
            rho0 = local[np.isclose(local["rho_phys"].astype(float), 0.0)]
            baseline = float(rho0["J_no_phys"].iloc[0]) if not rho0.empty else float(local["J_no_phys"].iloc[0])
            delta = local["J_no_phys"].astype(float) - baseline
            ax.plot(local["rho_phys"], delta, marker="o", linewidth=2, label=f"lambda_R={lambda_r:g}")
            for (_, row), value in zip(local.iterrows(), delta):
                cost_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "stage": stage,
                        "lambda_R": float(row["lambda_R"]),
                        "rho_phys": float(row["rho_phys"]),
                        "nonphysics_tradeoff_T": float(row["J_no_phys"]),
                        "cost_of_feasibility_vs_rho0": float(value),
                        "PAC_total": float(row["PAC_total"]),
                        "shutoff_line_ids": row["shutoff_line_ids"],
                    }
                )
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Cost of Feasibility")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("T(rho) - T(rho=0)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(output_dir / "cost_of_feasibility_traditional_lambdas.png", dpi=180)
        plt.close(fig)

        freq_values = []
        for rho in rho_values:
            local = group[np.isclose(group["rho_phys"].astype(float), rho)]
            frequency = float(local["shutoff_line_ids"].fillna("").map(lambda value: _selected_contains_line(value, 23)).mean()) if not local.empty else np.nan
            freq_values.append(frequency)
            frequency_rows.append(
                {
                    "scenario_id": scenario_id,
                    "stage": stage,
                    "rho_phys": float(rho),
                    "line23_selection_frequency": frequency,
                    "num_selected_rows": int(len(local)),
                }
            )
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        ax.plot(rho_values, freq_values, marker="o", linewidth=2, color="#4C78A8")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"{scenario_id} {STAGE_LABELS.get(stage, stage)}: Line 23 Frequency")
        ax.set_xlabel("rho_phys")
        ax.set_ylabel("Fraction of lambda sweep selections")
        ax.grid(True, alpha=0.25)
        fig.savefig(output_dir / "line23_frequency_by_rho.png", dpi=180)
        plt.close(fig)

    return pd.DataFrame(physics_rows), pd.DataFrame(cost_rows), pd.DataFrame(frequency_rows)


def _plot_summary(best: pd.DataFrame, expected: pd.DataFrame, cost: pd.DataFrame, line23: pd.DataFrame, plots_dir: Path) -> pd.DataFrame:
    summary_dir = plots_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    tradeoff = (
        best.groupby(["scenario_id", "rho_phys"], as_index=False)
        .agg(
            avg_PAC_total=("PAC_total", "mean"),
            avg_R_norm=("R_norm", "mean"),
            avg_L_shed=("L_shed", "mean"),
            avg_J_no_phys=("J_no_phys", "mean"),
            avg_J_true=("J_true", "mean"),
        )
        .sort_values(["scenario_id", "rho_phys"])
    )

    def line_plot(table: pd.DataFrame, y: str, title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
        for scenario_id, group in table.groupby("scenario_id", sort=True):
            ax.plot(group["rho_phys"], group[y], marker="o", linewidth=2, label=scenario_id)
        ax.set_title(title)
        ax.set_xlabel("rho_phys")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(summary_dir / filename, dpi=180)
        plt.close(fig)

    line_plot(tradeoff, "avg_PAC_total", "Average PAC by Rho", "Average PAC_total", "avg_pac_by_rho.png")
    target = expected.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        avg_target_recall=("target_recall", "mean"),
        avg_target_precision=("target_precision", "mean"),
        avg_target_overlap=("target_overlap_fraction", "mean"),
    )
    line_plot(target, "avg_target_recall", "Target Recall by Rho", "Average target recall", "target_recall_by_rho.png")
    line_plot(target, "avg_target_precision", "Target Precision by Rho", "Average target precision", "target_precision_by_rho.png")
    line_plot(target, "avg_target_overlap", "Target Recall by Rho", "Average target recall", "target_overlap_by_rho.png")
    cost_summary = cost.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        avg_cost_of_feasibility=("cost_of_feasibility_vs_rho0", "mean")
    )
    line_plot(cost_summary, "avg_cost_of_feasibility", "Average Cost of Feasibility by Rho", "Average T(rho)-T(0)", "avg_cost_of_feasibility_by_rho.png")
    line23_summary = line23.groupby(["scenario_id", "rho_phys"], as_index=False).agg(
        line23_selection_frequency=("line23_selection_frequency", "mean")
    )
    line_plot(line23_summary, "line23_selection_frequency", "Line 23 Frequency by Rho", "Selection frequency", "line23_frequency_by_rho_all_scenarios.png")
    return tradeoff


def run_stage_g_physics_infeasibility_sensitivity(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    lambda_step: float = 0.05,
    max_deenergized_lines: int = 2,
    stage_e_budget: int = 100,
    stage_d_limit: int | None = None,
    outlier_line_ids: Sequence[int] = DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS,
    output_root: Path = RESULT_ROOT,
) -> Path:
    models = ["gnn"] if models is None else list(models)
    if len(models) != 1:
        raise ValueError("Stage G physics sensitivity currently supports one model per run.")
    started = time.perf_counter()
    run_dir = make_run_dir(output_root, "run")
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    model_type = models[0]
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_type={model_type}.")

    metric_pool, calibration, scenarios, metadata = _build_metric_pool_for_model(
        model_type,
        scenario_ids,
        lambda_step,
        max_deenergized_lines,
        stage_e_budget,
        stage_d_limit,
        outlier_line_ids,
        run_dir,
    )
    rescored = rescore_topology_pool(metric_pool, rho_values)
    best, expected, rescored = _write_per_rho_outputs(rescored, scenarios, rho_values, plots_dir)
    physics, cost, line23 = _plot_cross_rho(best, plots_dir)
    tradeoff = _plot_summary(best, expected, cost, line23, plots_dir)
    checks = methodology_fidelity_checks(
        metric_pool,
        rescored,
        calibration,
        metadata["scenario"],
        metadata["physical_line_ids"],
        rho_values,
        metadata["candidate_line_ids"],
    )

    write_dataframe(tables_dir / "topology_metric_pool.csv", metric_pool)
    write_dataframe(tables_dir / "rho_rescored_objectives.csv", rescored)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_stage.csv", best)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "p_env_calibration_by_scenario.csv", calibration)
    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_dataframe(tables_dir / "physics_sensitivity.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    write_json(
        inputs_dir / "metadata.json",
        {
            **git_metadata(),
            "stage": "G",
            "study": "physics_infeasibility_sensitivity",
            "rho_phys_values": [float(value) for value in rho_values],
            "lambda_step": float(lambda_step),
            "lambda_values": _lambda_values(lambda_step),
            "traditional_lambda_values": TRADITIONAL_LAMBDAS,
            "expected_comparison_lambda_values": EXPECTED_COMPARISON_LAMBDAS,
            "stage_e_budget_per_lambda": int(stage_e_budget),
            "max_deenergized_lines_stage_d_and_stage_e_k2": int(max_deenergized_lines),
            "stage_d_limit": None if stage_d_limit is None else int(stage_d_limit),
            "model_type": model_type,
            "fixed_control_u_base": True,
            "continuous_recourse_optimized": False,
            "p_env_calibration_mode": P_ENV_CALIBRATION_MODE,
            "outlier_line_ids_requested": [int(line_id) for line_id in outlier_line_ids],
            "outlier_line_ids_canonical": [int(line_id) for line_id in metadata["canonical_outlier_line_ids"]],
            "true_objective": "lambda_R * R_norm + (1-lambda_R) * L_shed + rho_phys * PAC_total",
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    write_json(
        inputs_dir / "scenario_definitions.json",
        {
            "scenarios": [scenario.to_dict() for scenario in scenarios],
            "canonical_physical_branch_ids": True,
        },
    )
    write_json(inputs_dir / "rho_values.json", {"rho_phys_values": [float(value) for value in rho_values]})
    write_json(inputs_dir / "lambda_values.json", {"lambda_values": _lambda_values(lambda_step)})

    hard_failures = checks[(~checks["passed"].astype(bool)) & checks["severity"].eq("hard")]
    if not hard_failures.empty:
        raise RuntimeError(
            "Methodology fidelity checks failed: "
            + "; ".join(f"{row.check_name}: {row.details}" for row in hard_failures.itertuples(index=False))
        )
    return run_dir


def rescore_existing_stage_g_physics_sensitivity_run(
    run_dir: Path,
    rho_values: Sequence[float],
    model_type: str = "gnn",
) -> Path:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    scenario = context["scenario"]
    physical_ids = physical_line_ids(scenario)
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    scenarios = [
        _canonicalize_decision_quality_scenario(scenario_def, scenario)
        for scenario_def in get_scenarios()
    ]

    metric_pool = pd.read_csv(tables_dir / "topology_metric_pool.csv")
    calibration = pd.read_csv(tables_dir / "p_env_calibration_by_scenario.csv")
    rescored = rescore_topology_pool(metric_pool, rho_values)
    best, expected, rescored = _write_per_rho_outputs(rescored, scenarios, rho_values, plots_dir)
    physics, cost, line23 = _plot_cross_rho(best, plots_dir)
    tradeoff = _plot_summary(best, expected, cost, line23, plots_dir)
    checks = methodology_fidelity_checks(
        metric_pool,
        rescored,
        calibration,
        scenario,
        physical_ids,
        rho_values,
        candidate_line_ids,
    )

    write_dataframe(tables_dir / "rho_rescored_objectives.csv", rescored)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_stage.csv", best)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_dataframe(tables_dir / "physics_sensitivity.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    write_json(inputs_dir / "rho_values.json", {"rho_phys_values": [float(value) for value in rho_values]})
    write_json(
        inputs_dir / "rescore_metadata.json",
        {
            **git_metadata(),
            "rescore_run_dir": str(run_dir),
            "rho_phys_values": [float(value) for value in rho_values],
            "rescore_method": "reuse_existing_topology_metric_pool",
            "gridfm_or_gurobi_rerun": False,
        },
    )

    hard_failures = checks[(~checks["passed"].astype(bool)) & checks["severity"].eq("hard")]
    if not hard_failures.empty:
        raise RuntimeError(
            "Methodology fidelity checks failed after rescore: "
            + "; ".join(f"{row.check_name}: {row.details}" for row in hard_failures.itertuples(index=False))
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage G physics infeasibility rho sensitivity case study.")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gnn"])
    parser.add_argument("--scenarios", nargs="+", choices=[scenario.scenario_id for scenario in get_scenarios()], default=None)
    parser.add_argument("--rho-phys", nargs="+", type=float, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--stage-e-budget", type=int, default=100)
    parser.add_argument("--stage-d-limit", type=int, default=None)
    parser.add_argument("--outlier-line-ids", nargs="+", type=int, default=list(DEFAULT_GRIDFM_HEAVY_LOADING_LINE_IDS))
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--rescore-existing-run",
        type=Path,
        default=None,
        help="Reuse an existing topology_metric_pool.csv and regenerate rho-dependent tables/plots for the supplied rho list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rescore_existing_run is not None:
        run_dir = rescore_existing_stage_g_physics_sensitivity_run(
            args.rescore_existing_run,
            rho_values=args.rho_phys,
            model_type=args.models[0],
        )
        print(f"Rescored existing Stage G physics sensitivity run at {run_dir}")
        return
    scenario_ids = args.scenarios
    rho_values = args.rho_phys
    stage_e_budget = args.stage_e_budget
    stage_d_limit = args.stage_d_limit
    if args.smoke:
        scenario_ids = ["S1"] if scenario_ids is None else scenario_ids[:1]
        rho_values = [0.0, 10.0]
        stage_e_budget = min(int(stage_e_budget), 2)
        stage_d_limit = 5 if stage_d_limit is None else min(int(stage_d_limit), 5)
    run_dir = run_stage_g_physics_infeasibility_sensitivity(
        models=args.models,
        scenario_ids=scenario_ids,
        rho_values=rho_values,
        lambda_step=args.lambda_step,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_budget=stage_e_budget,
        stage_d_limit=stage_d_limit,
        outlier_line_ids=args.outlier_line_ids,
        output_root=args.output_root,
    )
    print(f"Wrote Stage G physics infeasibility sensitivity case study to {run_dir}")


if __name__ == "__main__":
    main()
