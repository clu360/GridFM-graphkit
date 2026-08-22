from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import canonicalize_line_ids, physical_line_ids
from experiments.test.wildfire_tests.shared.config import write_config_copy
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.state_extraction import compute_line_loading_ratios
from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
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
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    EXPECTED_COMPARISON_LAMBDAS,
    STAGE_D,
    STAGE_E_K2,
    STAGE_E_UNCONSTRAINED,
    TRADITIONAL_LAMBDAS,
    _best_by_stage,
    _canonicalize_decision_quality_scenario,
    _expected_vs_observed,
    _fixed_topology_metrics,
    _lambda_case,
    _lambda_values,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    HIGH_P_ENV_DEFAULT,
    LOW_P_ENV_DEFAULT,
    DecisionQualityScenario,
    get_scenarios,
    p_env_for_scenario,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_physics_sensitivity import (
    _plot_cross_rho,
    _plot_summary,
    _write_per_rho_outputs,
    rescore_topology_pool,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_g" / "physics_infeasibility_sensitivity_scenario_baseline_revision"
MARGIN_RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_g" / "physics_infeasibility_sensitivity_scenario_baseline_margin_revision"
DEFAULT_RHO_VALUES = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
BASELINE_LOADING_SOURCE = "stored_scenario_vm_va"
POST_TOPOLOGY_EVALUATION_SOURCE = "gridfm_inference_fixed_controls"
P_ENV_MODE = "targets_high_non_targets_0p05"
P_ENV_MODE_TARGET_MARGIN = "targets_high_non_targets_margin_0p25_without_line32"
TARGET_MARGIN_RATIO = 0.25
EXCLUDED_MARGIN_TARGET_LINE_IDS = (32,)
LINE23_SCENARIO_BASELINE_LOADING_REFERENCE = 0.5243973363957892


def scenario_baseline_loading_vector(context: dict) -> np.ndarray:
    scenario = context["scenario"]
    config = context["config"]
    return compute_line_loading_ratios(
        scenario,
        scenario.Vm_base,
        scenario.Va_base,
        standard_rate_a_mva=config.wildfire.standard_rate_a_mva,
    )


def p_env_for_scenario_baseline_revision(
    scenario_def: DecisionQualityScenario,
    num_lines: int,
    physical_ids: Sequence[int],
    low_value: float = LOW_P_ENV_DEFAULT,
    high_value: float = HIGH_P_ENV_DEFAULT,
) -> tuple[Dict[int, float], pd.DataFrame]:
    p_env = p_env_for_scenario(
        scenario_def,
        num_lines,
        low_value=low_value,
        high_value=high_value,
    )
    target_ids = {int(line_id) for line_id in scenario_def.target_high_risk_line_ids}
    rows = []
    for line_id in [int(item) for item in physical_ids]:
        rows.append(
            {
                "scenario_id": scenario_def.scenario_id,
                "line_id": line_id,
                "is_target": bool(line_id in target_ids),
                "p_env": float(p_env.get(line_id, low_value)),
                "p_env_mode": P_ENV_MODE,
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
            }
        )
    return {int(key): float(value) for key, value in p_env.items()}, pd.DataFrame(rows)


def remove_margin_excluded_targets(scenario_def: DecisionQualityScenario) -> DecisionQualityScenario:
    excluded = {int(line_id) for line_id in EXCLUDED_MARGIN_TARGET_LINE_IDS}
    targets = tuple(int(line_id) for line_id in scenario_def.target_high_risk_line_ids if int(line_id) not in excluded)
    expected = tuple(int(line_id) for line_id in scenario_def.expected_target_set if int(line_id) not in excluded)
    suppressed = tuple(int(line_id) for line_id in scenario_def.suppressed_line_ids if int(line_id) not in excluded)
    return replace(
        scenario_def,
        target_high_risk_line_ids=targets,
        expected_target_set=expected,
        suppressed_line_ids=suppressed,
    )


def p_env_for_target_margin_scenario(
    scenario_def: DecisionQualityScenario,
    num_lines: int,
    physical_ids: Sequence[int],
    baseline_loading: Sequence[float],
    low_value: float = LOW_P_ENV_DEFAULT,
    high_value: float = HIGH_P_ENV_DEFAULT,
    margin_ratio: float = TARGET_MARGIN_RATIO,
) -> tuple[Dict[int, float], pd.DataFrame]:
    physical_ids = [int(line_id) for line_id in physical_ids]
    target_ids = {int(line_id) for line_id in scenario_def.target_high_risk_line_ids}
    loading = np.asarray(baseline_loading, dtype=float)
    coeff = {line_id: float(loading[line_id]) ** 2 for line_id in physical_ids}
    target_scores = [float(high_value) * coeff[line_id] for line_id in target_ids if line_id in coeff]
    positive_target_scores = [score for score in target_scores if np.isfinite(score) and score > 0.0]
    weakest_target_score = min(positive_target_scores) if positive_target_scores else np.nan
    cap = float(margin_ratio) * float(weakest_target_score) if np.isfinite(weakest_target_score) else np.nan
    p_env = {line_id: float(low_value) for line_id in range(int(num_lines))}
    for line_id in physical_ids:
        if line_id in target_ids:
            p_env[line_id] = float(high_value)
            continue
        line_coeff = coeff[line_id]
        if np.isfinite(cap) and line_coeff > 0.0:
            p_env[line_id] = min(float(low_value), float(cap) / float(line_coeff))
        else:
            p_env[line_id] = float(low_value)

    rows = []
    for line_id in physical_ids:
        rows.append(
            {
                "scenario_id": scenario_def.scenario_id,
                "line_id": int(line_id),
                "is_target": bool(line_id in target_ids),
                "p_env": float(p_env.get(line_id, low_value)),
                "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
                "target_margin_ratio": float(margin_ratio),
                "weakest_target_score": float(weakest_target_score) if np.isfinite(weakest_target_score) else np.nan,
                "non_target_cap_score": float(cap) if np.isfinite(cap) else np.nan,
                "line_score_after_calibration": float(p_env.get(line_id, 0.0) * coeff[line_id]),
                "excluded_margin_target_line_ids": _csv_ints(EXCLUDED_MARGIN_TARGET_LINE_IDS),
            }
        )
    return {int(key): float(value) for key, value in p_env.items()}, pd.DataFrame(rows)


def build_scenario_baseline_loading_ranking(
    context: dict,
    baseline_loading: np.ndarray,
    p_env_frames: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    scenario = context["scenario"]
    edge_array = _edge_array(scenario)
    rates = np.asarray(scenario.rate_a, dtype=float)
    physical_ids = np.asarray(scenario.physical_branch_id, dtype=int)
    mapping_status = np.asarray(scenario.branch_mapping_status, dtype=object)
    is_self_loop = np.asarray(scenario.is_self_loop, dtype=bool)
    c_by_line = _consequence_by_line(context["consequence_df"])
    p_env_table = pd.concat(list(p_env_frames), ignore_index=True, sort=False) if p_env_frames else pd.DataFrame()
    mean_p_env = p_env_table.groupby("line_id")["p_env"].mean().to_dict() if not p_env_table.empty else {}

    rows = []
    for line_id in physical_line_ids(scenario):
        line_id = int(line_id)
        loading = float(baseline_loading[line_id])
        p_env_mean = float(mean_p_env.get(line_id, LOW_P_ENV_DEFAULT))
        impact = float(c_by_line.get(line_id, 0.0))
        rows.append(
            {
                "canonical_line_id": line_id,
                "physical_branch_id": int(physical_ids[line_id]),
                "directed_line_ids": ",".join(
                    str(int(item))
                    for item in scenario.physical_branch_directed_line_ids.get(int(physical_ids[line_id]), [line_id])
                ),
                "from_bus": int(edge_array[0, line_id]),
                "to_bus": int(edge_array[1, line_id]),
                "rate_a_mva": float(rates[line_id]),
                "loading_ratio": loading,
                "loading_squared": float(loading**2),
                "loading_pct": float(100.0 * loading),
                "mean_p_env_across_scenarios": p_env_mean,
                "impact": impact,
                "risk_score_mean_p_env_loading_squared": float(p_env_mean * loading**2),
                "proxy_score_mean_p_env_loading_squared_impact": float(p_env_mean * loading**2 * impact),
                "is_self_loop": bool(is_self_loop[line_id]),
                "mapping_status": str(mapping_status[line_id]),
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["loading_ratio", "canonical_line_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=int))
    return frame


def scenario_baseline_methodology_fidelity_checks(
    metric_pool: pd.DataFrame,
    rescored: pd.DataFrame,
    p_env_table: pd.DataFrame,
    ranking: pd.DataFrame,
    scenario,
    physical_ids: Sequence[int],
    rho_values: Sequence[float],
    candidate_line_ids: Sequence[int],
    p_env_mode: str = P_ENV_MODE,
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, severity: str, details: str) -> None:
        rows.append({"check_name": name, "passed": bool(passed), "severity": severity, "details": details})

    required_metric_flags = {
        "baseline_loading_source": BASELINE_LOADING_SOURCE,
        "R_base_s_source": BASELINE_LOADING_SOURCE,
        "proxy_loading_source": BASELINE_LOADING_SOURCE,
        "post_topology_evaluation_source": POST_TOPOLOGY_EVALUATION_SOURCE,
        "p_env_mode": p_env_mode,
    }
    for column, expected in required_metric_flags.items():
        add(
            f"{column}_recorded",
            column in metric_pool.columns and metric_pool[column].astype(str).eq(expected).all(),
            "hard",
            f"Expected {column}={expected} for all topology metric rows.",
        )

    add(
        "fixed_control_no_continuous_recourse",
        metric_pool["continuous_recourse_optimized"].eq(False).all()
        and metric_pool["fixed_control_u_base"].eq(True).all(),
        "hard",
        "This study must remain fixed-control with no SciPy continuous recourse.",
    )
    add(
        "rho_rescoring_reuses_metric_pool",
        rescored.groupby("topology_pool_id").size().nunique() == 1
        and rescored["rho_phys"].nunique() == len(set(float(value) for value in rho_values)),
        "hard",
        "Each topology_pool_id should appear once per rho value.",
    )

    physical_set = {int(line_id) for line_id in physical_ids}
    candidates = {int(line_id) for line_id in candidate_line_ids}
    add(
        "candidate_lines_are_physical",
        candidates.issubset(physical_set),
        "hard",
        "Candidate line IDs must be canonical physical branch IDs.",
    )
    self_loop_count = int(np.sum(np.asarray(scenario.is_self_loop, dtype=bool)[list(physical_set)])) if physical_set else 0
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
        "Physical risk lines must be unique canonical branches.",
    )

    target_rows = p_env_table[p_env_table["is_target"].astype(bool)]
    non_target_rows = p_env_table[~p_env_table["is_target"].astype(bool)]
    add(
        "target_lines_have_high_p_env",
        target_rows.empty or np.allclose(target_rows["p_env"].astype(float), HIGH_P_ENV_DEFAULT),
        "hard",
        "Explicit scenario targets must use p_env=1.0.",
    )
    if p_env_mode == P_ENV_MODE_TARGET_MARGIN:
        add(
            "non_target_lines_do_not_exceed_low_p_env",
            non_target_rows.empty or (non_target_rows["p_env"].astype(float) <= LOW_P_ENV_DEFAULT + 1e-12).all(),
            "hard",
            "Target-margin non-target lines must not exceed the ordinary low p_env value.",
        )
        margin_failures = []
        for scenario_id, group in p_env_table.groupby("scenario_id", sort=False):
            targets = group[group["is_target"].astype(bool)]
            non_targets = group[~group["is_target"].astype(bool)]
            if targets.empty or non_targets.empty:
                continue
            cap_values = non_targets["non_target_cap_score"].astype(float).dropna().unique()
            if len(cap_values) != 1:
                margin_failures.append(str(scenario_id))
                continue
            cap = float(cap_values[0])
            non_scores = non_targets["line_score_after_calibration"].astype(float)
            if (non_scores > cap + 1e-10).any():
                margin_failures.append(str(scenario_id))
        add(
            "non_targets_below_target_margin_cap",
            not margin_failures,
            "hard",
            f"Scenario failures: {margin_failures[:5]}",
        )
    else:
        add(
            "non_target_lines_have_low_p_env",
            non_target_rows.empty or np.allclose(non_target_rows["p_env"].astype(float), LOW_P_ENV_DEFAULT),
            "hard",
            "All non-target physical lines must use p_env=0.05.",
        )
        add(
            "rank_calibrated_suppression_disabled",
            p_env_table["p_env_mode"].astype(str).eq(P_ENV_MODE).all() and (non_target_rows["p_env"].astype(float) >= LOW_P_ENV_DEFAULT - 1e-12).all(),
            "hard",
            "No non-target should be suppressed below the ordinary low p_env value.",
        )

    max_loading = float(ranking["loading_ratio"].max()) if not ranking.empty else np.nan
    num_gt_1 = int((ranking["loading_ratio"].astype(float) > 1.0).sum()) if not ranking.empty else -1
    line23 = ranking[ranking["canonical_line_id"].astype(int).eq(23)]
    line23_loading = float(line23["loading_ratio"].iloc[0]) if not line23.empty else np.nan
    add(
        "scenario_baseline_max_loading_below_one",
        np.isfinite(max_loading) and max_loading < 1.0,
        "hard",
        f"Scenario-baseline max loading: {max_loading:.12g}.",
    )
    add(
        "scenario_baseline_no_branches_over_100pct",
        num_gt_1 == 0,
        "hard",
        f"Scenario-baseline branches > 100%: {num_gt_1}.",
    )
    add(
        "line23_scenario_baseline_loading_expected",
        np.isfinite(line23_loading) and abs(line23_loading - LINE23_SCENARIO_BASELINE_LOADING_REFERENCE) < 1e-3,
        "hard",
        f"Line 23 scenario-baseline loading: {line23_loading:.12g}.",
    )

    invariant_columns = [
        "R_raw",
        "R_norm",
        "L_shed",
        "PAC_total",
        "max_loading_ratio",
        "gridfm_status",
        "shutoff_line_ids",
    ]
    drift = []
    for _, group in rescored.groupby("topology_pool_id", sort=False):
        for column in invariant_columns:
            values = group[column]
            if pd.api.types.is_numeric_dtype(values):
                arr = values.astype(float).to_numpy()
                finite = arr[np.isfinite(arr)]
                if len(finite) and float(np.max(finite) - np.min(finite)) > 1e-10:
                    drift.append((int(group["topology_pool_id"].iloc[0]), column))
            else:
                if values.astype(str).nunique(dropna=False) > 1:
                    drift.append((int(group["topology_pool_id"].iloc[0]), column))
        if drift:
            break
    add(
        "rho_does_not_change_intrinsic_metrics",
        not drift,
        "hard",
        f"Intrinsic metric drift examples: {drift[:5]}",
    )

    recomputed_failures = []
    p_env_by_scenario = {
        scenario_id: {int(row.line_id): float(row.p_env) for row in frame.itertuples(index=False)}
        for scenario_id, frame in p_env_table.groupby("scenario_id", sort=False)
    }
    loading_by_line = {int(row.canonical_line_id): float(row.loading_ratio) for row in ranking.itertuples(index=False)}
    loading = np.zeros(max(loading_by_line.keys(), default=-1) + 1, dtype=float)
    for line_id, value in loading_by_line.items():
        loading[line_id] = value
    for scenario_id, group in metric_pool.groupby("scenario_id", sort=False):
        expected, _ = compute_operational_wildfire_exposure(
            loading,
            p_env_by_scenario[str(scenario_id)],
            {line_id: 1 for line_id in range(len(loading))},
            [int(line_id) for line_id in physical_ids],
        )
        observed = group["R_base_s"].astype(float).dropna().unique()
        if len(observed) != 1 or abs(float(observed[0]) - float(expected)) > 1e-10:
            recomputed_failures.append(str(scenario_id))
    add(
        "R_base_s_recomputes_from_scenario_baseline",
        not recomputed_failures,
        "hard",
        f"Scenario failures: {recomputed_failures[:5]}",
    )

    required_columns = [
        "p_env_mode",
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


def validate_plot_outputs(
    run_dir: Path,
    scenario_ids: Sequence[str],
    rho_values: Sequence[float],
    stages: Sequence[str],
) -> pd.DataFrame:
    rows = []

    def add(path: Path, category: str) -> None:
        rows.append(
            {
                "category": category,
                "relative_path": str(path.relative_to(run_dir)),
                "exists": bool(path.exists()),
                "size_bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        )

    plots_dir = run_dir / "plots"
    per_rho_files = [
        "expected_vs_selected_shutoff_lines.png",
        "pareto_frontier_scatter.png",
        "traditional_lambda_objective_convergence.png",
    ]
    cross_rho_files = [
        "risk_load_pareto_overlay_all_lambdas.png",
        "nondominated_points_colored_by_rho_all_lambdas.png",
        "physics_feasibility_sensitivity_traditional_lambdas.png",
        "cost_of_feasibility_traditional_lambdas.png",
        "line23_frequency_by_rho.png",
    ]
    for rho in rho_values:
        for scenario_id in scenario_ids:
            for filename in per_rho_files:
                add(plots_dir / "per_rho" / f"rho{float(rho):g}" / scenario_id / filename, "per_rho")
    for scenario_id in scenario_ids:
        for stage in stages:
            for filename in cross_rho_files:
                add(plots_dir / "cross_rho" / scenario_id / stage / filename, "cross_rho")
    for filename in [
        "avg_pac_by_rho.png",
        "target_overlap_by_rho.png",
        "avg_cost_of_feasibility_by_rho.png",
        "line23_frequency_by_rho_all_scenarios.png",
    ]:
        add(plots_dir / "summary" / filename, "summary")
    return pd.DataFrame(rows)


def build_scenario_design_audit(run_dir: Path, model_type: str = "gnn") -> tuple[pd.DataFrame, pd.DataFrame]:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    ranking = pd.read_csv(tables_dir / "scenario_baseline_loading_ranking.csv")
    p_env = pd.read_csv(tables_dir / "p_env_by_scenario.csv")
    best = pd.read_csv(tables_dir / "best_by_rho_scenario_lambda_stage.csv")
    expected = pd.read_csv(tables_dir / "expected_vs_selected_by_rho.csv")
    line23 = pd.read_csv(tables_dir / "line23_frequency_by_rho.csv")
    if "target_recall" not in expected.columns and "target_overlap_fraction" in expected.columns:
        expected["target_recall"] = expected["target_overlap_fraction"]
    if "target_precision" not in expected.columns:
        observed_counts = expected.get("observed_shutoff_line_ids", pd.Series(dtype=str)).fillna("").astype(str).map(
            lambda value: len([item for item in value.split(",") if item.strip()])
        )
        target_counts = expected.get("num_observed_target_lines", pd.Series(0, index=expected.index)).astype(float)
        expected["target_precision"] = np.where(observed_counts.astype(float) > 0.0, target_counts / observed_counts, 0.0)

    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    scenario = context["scenario"]
    candidate_line_ids = set(canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS))
    p_env_modes = set(str(value) for value in p_env.get("p_env_mode", pd.Series(dtype=str)).dropna().unique())
    canonical_scenarios_raw = [
        _canonicalize_decision_quality_scenario(scenario_def, scenario)
        for scenario_def in get_scenarios()
    ]
    if P_ENV_MODE_TARGET_MARGIN in p_env_modes:
        canonical_scenarios = [remove_margin_excluded_targets(scenario_def) for scenario_def in canonical_scenarios_raw]
    else:
        canonical_scenarios = canonical_scenarios_raw

    target_rows = []
    scenario_rows = []
    for scenario_def in canonical_scenarios:
        scenario_id = scenario_def.scenario_id
        local_p = p_env[p_env["scenario_id"].astype(str).eq(scenario_id)][["line_id", "is_target", "p_env"]]
        local = ranking.merge(local_p, left_on="canonical_line_id", right_on="line_id", how="inner")
        local["risk_score_no_impact"] = local["p_env"].astype(float) * local["loading_ratio"].astype(float) ** 2
        local["risk_score_with_impact"] = local["risk_score_no_impact"] * local["impact"].astype(float)
        local["rank_no_impact"] = local["risk_score_no_impact"].rank(method="min", ascending=False).astype(int)
        local["rank_with_impact"] = local["risk_score_with_impact"].rank(method="min", ascending=False).astype(int)
        targets = local[local["is_target"].astype(bool)].copy()
        non_targets = local[~local["is_target"].astype(bool)].copy()
        max_non_target_score = float(non_targets["risk_score_no_impact"].max()) if not non_targets.empty else np.nan
        max_non_target_score_impact = float(non_targets["risk_score_with_impact"].max()) if not non_targets.empty else np.nan

        selected_counts = {}
        scenario_best = best[best["scenario_id"].astype(str).eq(scenario_id)]
        for text in scenario_best["shutoff_line_ids"].fillna("").astype(str):
            for item in [part for part in text.split(",") if part != ""]:
                line_id = int(item)
                selected_counts[line_id] = selected_counts.get(line_id, 0) + 1
        total_best_rows = max(int(len(scenario_best)), 1)

        for row in targets.itertuples(index=False):
            line_id = int(row.canonical_line_id)
            score = float(row.risk_score_no_impact)
            score_impact = float(row.risk_score_with_impact)
            target_rows.append(
                {
                    "scenario_id": scenario_id,
                    "target_line_id": line_id,
                    "in_candidate_set": bool(line_id in candidate_line_ids),
                    "loading_ratio": float(row.loading_ratio),
                    "p_env": float(row.p_env),
                    "impact": float(row.impact),
                    "rank_loading_only": int(row.rank),
                    "rank_penv_loading2": int(row.rank_no_impact),
                    "rank_penv_loading2_impact": int(row.rank_with_impact),
                    "score_penv_loading2": score,
                    "score_margin_vs_top_non_target": float(score / max(max_non_target_score, 1e-12)),
                    "score_margin_with_impact_vs_top_non_target": float(score_impact / max(max_non_target_score_impact, 1e-12)),
                    "selected_frequency_all_best_rows": float(selected_counts.get(line_id, 0) / total_best_rows),
                }
            )

        expected_local = expected[expected["scenario_id"].astype(str).eq(scenario_id)]
        line23_local = line23[line23["scenario_id"].astype(str).eq(scenario_id)]
        rho0_expected = expected_local[np.isclose(expected_local["rho_phys"].astype(float), 0.0)]
        rho0_recall = float(rho0_expected["target_recall"].mean())
        rho0_precision = float(rho0_expected["target_precision"].mean())
        rho0_overlap = float(rho0_expected["target_overlap_fraction"].mean())
        recall_by_rho = expected_local.groupby("rho_phys")["target_recall"].mean()
        precision_by_rho = expected_local.groupby("rho_phys")["target_precision"].mean()
        overlap_by_rho = expected_local.groupby("rho_phys")["target_overlap_fraction"].mean()
        best_recall = float(recall_by_rho.max())
        best_precision = float(precision_by_rho.max())
        best_overlap = float(overlap_by_rho.max())
        rho0_line23 = float(line23_local[np.isclose(line23_local["rho_phys"].astype(float), 0.0)]["line23_selection_frequency"].mean())
        best_target_rho = float(recall_by_rho.idxmax())
        weakest_rank = int(targets["rank_no_impact"].max()) if not targets.empty else -1
        weakest_rank_impact = int(targets["rank_with_impact"].max()) if not targets.empty else -1
        min_margin = float(
            min(
                (
                    float(row.risk_score_no_impact) / max(max_non_target_score, 1e-12)
                    for row in targets.itertuples(index=False)
                ),
                default=np.nan,
            )
        )
        missing_candidate_targets = sorted(
            int(line_id) for line_id in scenario_def.expected_target_set if int(line_id) not in candidate_line_ids
        )
        weak_target_lines = sorted(
            int(row.canonical_line_id)
            for row in targets.itertuples(index=False)
            if int(row.rank_no_impact) > 10 or float(row.risk_score_no_impact) < max_non_target_score
        )
        top_selected = sorted(selected_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        top_selected_non_targets = [
            int(line_id)
            for line_id, _count in top_selected
            if int(line_id) not in {int(value) for value in scenario_def.expected_target_set}
        ]
        if missing_candidate_targets or weak_target_lines:
            diagnosis = "scenario_design_weakness"
        elif rho0_overlap < 0.5 and min_margin >= 1.0:
            diagnosis = "methodology_or_evaluation_failure"
        else:
            diagnosis = "mixed_or_acceptable"
        scenario_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_name": scenario_def.scenario_name,
                "num_targets": int(len(targets)),
                "missing_candidate_targets": _csv_ints(missing_candidate_targets),
                "weak_target_lines": _csv_ints(weak_target_lines),
                "weakest_target_rank_penv_loading2": weakest_rank,
                "weakest_target_rank_penv_loading2_impact": weakest_rank_impact,
                "min_target_margin_vs_top_non_target": min_margin,
                "rho0_target_recall": rho0_recall,
                "rho0_target_precision": rho0_precision,
                "best_target_recall": best_recall,
                "best_target_precision": best_precision,
                "best_target_recall_rho": best_target_rho,
                "rho0_target_overlap": rho0_overlap,
                "best_target_overlap": best_overlap,
                "best_target_overlap_rho": best_target_rho,
                "rho0_line23_frequency": rho0_line23,
                "top_selected_lines": _csv_ints([line_id for line_id, _count in top_selected]),
                "top_selected_non_target_lines": _csv_ints(top_selected_non_targets),
                "diagnosis": diagnosis,
            }
        )
    return pd.DataFrame(scenario_rows), pd.DataFrame(target_rows)


def run_scenario_design_audit(run_dir: Path, model_type: str = "gnn") -> Path:
    run_dir = Path(run_dir)
    tables_dir = run_dir / "tables"
    scenario_audit, target_audit = build_scenario_design_audit(run_dir, model_type=model_type)
    write_dataframe(tables_dir / "scenario_design_audit.csv", scenario_audit)
    write_dataframe(tables_dir / "scenario_target_rank_audit.csv", target_audit)
    print(scenario_audit.to_csv(index=False))
    print(f"Scenario design audit written to: {tables_dir / 'scenario_design_audit.csv'}")
    print(f"Target rank audit written to: {tables_dir / 'scenario_target_rank_audit.csv'}")
    return run_dir


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
    run_dir: Path,
    p_env_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[DecisionQualityScenario], dict]:
    context = _build_model_context(model_type, grouping_top_fraction=0.30)
    write_config_copy(context["config"], run_dir / "inputs" / f"config_{model_type}.yaml")
    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    num_lines = int(_edge_array(scenario).shape[1])
    _validate_candidate_line_ids(candidate_line_ids, num_lines)
    baseline_loading = scenario_baseline_loading_vector(context)
    c_by_line = _consequence_by_line(context["consequence_df"])
    physical_ids = physical_line_ids(scenario)
    canonical_scenarios_raw = [
        _canonicalize_decision_quality_scenario(scenario_def, scenario)
        for scenario_def in get_scenarios(scenario_ids)
    ]
    if p_env_mode == P_ENV_MODE_TARGET_MARGIN:
        canonical_scenarios = [remove_margin_excluded_targets(scenario_def) for scenario_def in canonical_scenarios_raw]
    else:
        canonical_scenarios = canonical_scenarios_raw
    lambdas = _lambda_values(lambda_step)
    metric_rows = []
    p_env_frames = []

    for scenario_def in canonical_scenarios:
        if p_env_mode == P_ENV_MODE_TARGET_MARGIN:
            p_env, p_env_frame = p_env_for_target_margin_scenario(
                scenario_def,
                num_lines,
                physical_ids,
                baseline_loading,
            )
        else:
            p_env, p_env_frame = p_env_for_scenario_baseline_revision(
                scenario_def,
                num_lines,
                physical_ids,
            )
        p_env_frames.append(p_env_frame)
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
        stage_d_frame["p_env_mode"] = p_env_mode
        stage_d_frame["baseline_loading_source"] = BASELINE_LOADING_SOURCE
        stage_d_frame["R_base_s_source"] = BASELINE_LOADING_SOURCE
        stage_d_frame["proxy_loading_source"] = BASELINE_LOADING_SOURCE
        stage_d_frame["post_topology_evaluation_source"] = POST_TOPOLOGY_EVALUATION_SOURCE
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
                frame["p_env_mode"] = p_env_mode
                frame["baseline_loading_source"] = BASELINE_LOADING_SOURCE
                frame["R_base_s_source"] = BASELINE_LOADING_SOURCE
                frame["proxy_loading_source"] = BASELINE_LOADING_SOURCE
                frame["post_topology_evaluation_source"] = POST_TOPOLOGY_EVALUATION_SOURCE
                frame["lambda_R"] = float(lambda_r)
                frame["lambda_L"] = lambda_l
                frame["lambda_case"] = _lambda_case(lambda_r)
                frame["generation_lambda_R"] = float(lambda_r)
                frame["generation_lambda_L"] = lambda_l
                metric_rows.append(frame)
            print(f"[{model_type} {scenario_def.scenario_id}] scenario-baseline pool completed lambda_R={lambda_r:.2f}", flush=True)

    p_env_table = pd.concat(p_env_frames, ignore_index=True, sort=False)
    ranking = build_scenario_baseline_loading_ranking(context, baseline_loading, p_env_frames)
    metric_pool = _metric_pool_id(pd.concat(metric_rows, ignore_index=True, sort=False))
    if "gridfm_status" not in metric_pool.columns and "status" in metric_pool.columns:
        metric_pool["gridfm_status"] = metric_pool["status"]
    metadata = {
        "scenario": scenario,
        "candidate_line_ids": candidate_line_ids,
        "physical_line_ids": physical_ids,
        "num_lines": num_lines,
        "baseline_loading": baseline_loading,
    }
    return metric_pool, p_env_table, ranking, canonical_scenarios, metadata


def run_stage_g_scenario_baseline_physics_sensitivity(
    models: List[str] | None = None,
    scenario_ids: List[str] | None = None,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    lambda_step: float = 0.05,
    max_deenergized_lines: int = 2,
    stage_e_budget: int = 100,
    stage_d_limit: int | None = None,
    output_root: Path = RESULT_ROOT,
    p_env_mode: str = P_ENV_MODE,
) -> Path:
    models = ["gnn"] if models is None else list(models)
    if len(models) != 1:
        raise ValueError("Stage G scenario-baseline sensitivity currently supports one model per run.")
    started = time.perf_counter()
    run_dir = make_run_dir(output_root, "run")
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    model_type = models[0]
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model_type={model_type}.")

    metric_pool, p_env_table, ranking, scenarios, metadata = _build_metric_pool_for_model(
        model_type,
        scenario_ids,
        lambda_step,
        max_deenergized_lines,
        stage_e_budget,
        stage_d_limit,
        run_dir,
        p_env_mode,
    )
    rescored = rescore_topology_pool(metric_pool, rho_values)
    best, expected, rescored = _write_per_rho_outputs(rescored, scenarios, rho_values, plots_dir)
    physics, cost, line23 = _plot_cross_rho(best, plots_dir)
    tradeoff = _plot_summary(best, expected, cost, line23, plots_dir)
    checks = scenario_baseline_methodology_fidelity_checks(
        metric_pool,
        rescored,
        p_env_table,
        ranking,
        metadata["scenario"],
        metadata["physical_line_ids"],
        rho_values,
        metadata["candidate_line_ids"],
        p_env_mode=p_env_mode,
    )
    plot_checks = validate_plot_outputs(
        run_dir,
        [scenario.scenario_id for scenario in scenarios],
        rho_values,
        [STAGE_D, STAGE_E_K2, STAGE_E_UNCONSTRAINED],
    )

    write_dataframe(tables_dir / "scenario_baseline_loading_ranking.csv", ranking)
    write_dataframe(tables_dir / "topology_metric_pool.csv", metric_pool)
    write_dataframe(tables_dir / "rho_rescored_objectives.csv", rescored)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_stage.csv", best)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_dataframe(tables_dir / "plot_output_checks.csv", plot_checks)
    write_dataframe(tables_dir / "physics_sensitivity.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    write_json(
        inputs_dir / "metadata.json",
        {
            **git_metadata(),
            "stage": "G",
            "study": "physics_infeasibility_sensitivity_scenario_baseline_revision",
            "rho_phys_values": [float(value) for value in rho_values],
            "lambda_step": float(lambda_step),
            "lambda_values": _lambda_values(lambda_step),
            "traditional_lambda_values": TRADITIONAL_LAMBDAS,
            "expected_comparison_lambda_values": EXPECTED_COMPARISON_LAMBDAS,
            "stage_e_budget_per_lambda": int(stage_e_budget),
            "max_deenergized_lines_stage_d_and_stage_e_k2": int(max_deenergized_lines),
            "stage_d_limit": None if stage_d_limit is None else int(stage_d_limit),
            "model_type": model_type,
            "baseline_loading_source": BASELINE_LOADING_SOURCE,
            "R_base_s_source": BASELINE_LOADING_SOURCE,
            "proxy_loading_source": BASELINE_LOADING_SOURCE,
            "post_topology_evaluation_source": POST_TOPOLOGY_EVALUATION_SOURCE,
            "fixed_control_u_base": True,
            "continuous_recourse_optimized": False,
            "p_env_mode": p_env_mode,
            "target_margin_ratio": TARGET_MARGIN_RATIO if p_env_mode == P_ENV_MODE_TARGET_MARGIN else None,
            "excluded_margin_target_line_ids": list(EXCLUDED_MARGIN_TARGET_LINE_IDS) if p_env_mode == P_ENV_MODE_TARGET_MARGIN else [],
            "rho_rescoring_reuses_metric_pool": True,
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
    missing_plots = plot_checks[(~plot_checks["exists"].astype(bool)) | (plot_checks["size_bytes"].astype(int) <= 0)]
    if not hard_failures.empty:
        raise RuntimeError(
            "Methodology fidelity checks failed: "
            + "; ".join(f"{row.check_name}: {row.details}" for row in hard_failures.itertuples(index=False))
        )
    if not missing_plots.empty:
        raise RuntimeError(
            "Plot output checks failed: "
            + "; ".join(str(row.relative_path) for row in missing_plots.head(10).itertuples(index=False))
        )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage G scenario-baseline physics infeasibility sensitivity.")
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_CONFIGS), default=["gnn"])
    parser.add_argument("--scenarios", nargs="+", choices=[scenario.scenario_id for scenario in get_scenarios()], default=None)
    parser.add_argument("--rho-phys", nargs="+", type=float, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--max-deenergized-lines", type=int, default=2)
    parser.add_argument("--stage-e-budget", type=int, default=100)
    parser.add_argument("--stage-d-limit", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--p-env-mode",
        choices=[P_ENV_MODE, P_ENV_MODE_TARGET_MARGIN],
        default=P_ENV_MODE,
        help="Scenario p_env construction mode.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--audit-existing-run",
        type=Path,
        default=None,
        help="Write scenario_design_audit.csv and scenario_target_rank_audit.csv for an existing completed run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.audit_existing_run is not None:
        run_scenario_design_audit(args.audit_existing_run, model_type=args.models[0])
        return
    scenario_ids = args.scenarios
    rho_values = args.rho_phys
    stage_e_budget = args.stage_e_budget
    stage_d_limit = args.stage_d_limit
    if args.smoke:
        scenario_ids = ["S1"] if scenario_ids is None else scenario_ids[:1]
        rho_values = [0.0, 0.1]
        stage_e_budget = min(int(stage_e_budget), 2)
        stage_d_limit = 5 if stage_d_limit is None else min(int(stage_d_limit), 5)
    output_root = args.output_root
    if args.p_env_mode == P_ENV_MODE_TARGET_MARGIN and output_root == RESULT_ROOT:
        output_root = MARGIN_RESULT_ROOT
    run_dir = run_stage_g_scenario_baseline_physics_sensitivity(
        models=args.models,
        scenario_ids=scenario_ids,
        rho_values=rho_values,
        lambda_step=args.lambda_step,
        max_deenergized_lines=args.max_deenergized_lines,
        stage_e_budget=stage_e_budget,
        stage_d_limit=stage_d_limit,
        output_root=output_root,
        p_env_mode=args.p_env_mode,
    )
    print(f"Wrote Stage G scenario-baseline physics infeasibility sensitivity to {run_dir}")


if __name__ == "__main__":
    main()
