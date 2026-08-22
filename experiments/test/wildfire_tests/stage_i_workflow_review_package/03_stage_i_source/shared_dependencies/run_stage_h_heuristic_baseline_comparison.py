from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.gridfm_support.branch_metadata import canonicalize_line_ids, physical_line_ids
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.shared.wildfire_scenario import connected_components_from_line_ids
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_decision_quality import (
    FIXED_T0P30_CANDIDATE_LINE_IDS,
    _edge_array,
    _scenario_baseline_exposure,
    _validate_candidate_line_ids,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    _canonicalize_decision_quality_scenario,
)
from experiments.test.wildfire_tests.stage_f_decision_quality.scenario_definitions import (
    DecisionQualityScenario,
    get_scenarios,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    _expected_vs_observed_all_lambdas,
    _fixed_vs_continuous,
    _line_key,
    _make_continuous_context,
    _methodology_checks,
    _parse_line_key,
    _plot_cross_and_summary,
    _run_continuous_pool,
    _runtime_summary,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_scenario_baseline_physics_sensitivity import (
    BASELINE_LOADING_SOURCE,
    P_ENV_MODE_TARGET_MARGIN,
    build_scenario_baseline_loading_ranking,
    p_env_for_target_margin_scenario,
    remove_margin_excluded_targets,
    scenario_baseline_loading_vector,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_h" / "heuristic_baseline_comparison"
DEFAULT_STAGE_G_REFERENCE_RUN = (
    RESULTS_ROOT
    / "leq"
    / "stage_g"
    / "physics_infeasibility_revised_continuous_implementation"
    / "continuous_run"
)
DEFAULT_LAMBDAS = [0.8, 0.5, 0.2]
DEFAULT_RHO_VALUES = [0.0, 2.0]
DEFAULT_TH_PERCENTILES = [60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
DEFAULT_TH_LINE_COUNTS = [5, 4, 3, 2, 1]
DEFAULT_AH_TOP_FRACTION = 0.30
DEFAULT_CALL_BUDGET = 100

METHOD_TH = "TH"
METHOD_AH = "AH"
METHOD_STAGE_G = "Stage G revised continuous"
AH_STAGE = "AH_connected_top30"


def _json_ints(values: Iterable[int]) -> str:
    return json.dumps([int(value) for value in values])


def _json_floats_by_line(values: Dict[int, float]) -> str:
    return json.dumps({str(int(key)): float(value) for key, value in sorted(values.items())}, sort_keys=True)


def _as_float_list(values: Sequence[float]) -> List[float]:
    return [float(value) for value in values]


def _lambda_case(lambda_r: float) -> str:
    return f"lambda_R={float(lambda_r):.2f}"


def _stage_for_th_percentile(percentile: float) -> str:
    return f"TH_p{int(round(float(percentile)))}"


def _stage_label_for_th_percentile(percentile: float) -> str:
    return f"TH p{int(round(float(percentile)))}"


def _stage_for_th_line_count(line_count: int) -> str:
    return f"TH_top{int(line_count)}"


def _stage_label_for_th_line_count(line_count: int) -> str:
    return f"TH top {int(line_count)}"


def _candidate_scores(
    candidate_line_ids: Sequence[int],
    p_env: Dict[int, float],
    baseline_loading: np.ndarray,
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for line_id in sorted(int(item) for item in candidate_line_ids):
        scores[line_id] = float(p_env.get(line_id, 0.0)) * float(baseline_loading[line_id]) ** 2
    if not scores:
        raise ValueError("candidate_line_ids must not be empty.")
    if not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("heuristic candidate scores must be finite.")
    return scores


def select_threshold_lines(candidate_scores: Dict[int, float], percentile: float) -> tuple[List[int], float]:
    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty.")
    threshold = float(np.percentile(list(candidate_scores.values()), float(percentile)))
    selected = sorted(int(line_id) for line_id, score in candidate_scores.items() if float(score) >= threshold)
    if not selected:
        max_line = max(candidate_scores, key=lambda line_id: (float(candidate_scores[line_id]), -int(line_id)))
        selected = [int(max_line)]
    return selected, threshold


def select_top_fraction_lines(candidate_scores: Dict[int, float], top_fraction: float) -> List[int]:
    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty.")
    if not (0.0 < float(top_fraction) <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")
    count = max(1, int(math.ceil(len(candidate_scores) * float(top_fraction))))
    ordered = sorted(candidate_scores, key=lambda line_id: (-float(candidate_scores[line_id]), int(line_id)))
    return [int(line_id) for line_id in ordered[:count]]


def select_top_k_lines(candidate_scores: Dict[int, float], line_count: int) -> List[int]:
    if not candidate_scores:
        raise ValueError("candidate_scores must not be empty.")
    count = int(line_count)
    if count <= 0:
        raise ValueError("line_count must be positive.")
    count = min(count, len(candidate_scores))
    ordered = sorted(candidate_scores, key=lambda line_id: (-float(candidate_scores[line_id]), int(line_id)))
    return [int(line_id) for line_id in ordered[:count]]


def area_heuristic_groups(edge_index, candidate_scores: Dict[int, float], top_fraction: float) -> pd.DataFrame:
    selected = select_top_fraction_lines(candidate_scores, top_fraction)
    components = connected_components_from_line_ids(edge_index, selected)
    rows = []
    selected_set = set(int(line_id) for line_id in selected)
    for group_idx, component in enumerate(components, start=1):
        component_lines = [int(line_id) for line_id in component["line_ids"] if int(line_id) in selected_set]
        if not component_lines:
            continue
        scores = [float(candidate_scores[line_id]) for line_id in component_lines]
        rows.append(
            {
                "ah_group_id": f"AH_G{group_idx}",
                "ah_group_line_ids": _line_key(component_lines),
                "ah_group_bus_ids": _json_ints(component["bus_ids"]),
                "ah_group_num_lines": int(len(component_lines)),
                "ah_group_average_score": float(np.mean(scores)),
                "ah_group_total_score": float(np.sum(scores)),
                "ah_group_min_score": float(np.min(scores)),
                "ah_group_max_score": float(np.max(scores)),
            }
        )
    if not rows:
        raise ValueError("AH top-fraction selection produced no connected groups.")
    groups = pd.DataFrame(rows)
    groups = groups.sort_values(
        ["ah_group_average_score", "ah_group_total_score", "ah_group_num_lines", "ah_group_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    groups["ah_selected_group"] = False
    groups.loc[0, "ah_selected_group"] = True
    return groups


def _canonical_scenarios(scenario_ids: Sequence[str], grid_scenario) -> List[DecisionQualityScenario]:
    return [
        remove_margin_excluded_targets(_canonicalize_decision_quality_scenario(scenario_def, grid_scenario))
        for scenario_def in get_scenarios(scenario_ids)
    ]


def _build_stage_h_topology_pool(
    context: dict,
    scenario_ids: Sequence[str],
    lambda_values: Sequence[float],
    th_line_counts: Sequence[int],
    ah_top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[DecisionQualityScenario], dict, pd.DataFrame, pd.DataFrame]:
    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    physical_ids = physical_line_ids(scenario)
    num_lines = int(_edge_array(scenario).shape[1])
    _validate_candidate_line_ids(candidate_line_ids, num_lines)
    baseline_loading = scenario_baseline_loading_vector(context)
    canonical_scenarios = _canonical_scenarios(scenario_ids, scenario)

    rows = []
    p_env_frames = []
    th_audit_rows = []
    ah_audit_frames = []

    for scenario_def in canonical_scenarios:
        p_env, p_env_frame = p_env_for_target_margin_scenario(
            scenario_def,
            num_lines,
            physical_ids,
            baseline_loading,
        )
        baseline_r, _ = _scenario_baseline_exposure(baseline_loading, p_env, physical_ids)
        p_env_frames.append(p_env_frame)
        scores = _candidate_scores(candidate_line_ids, p_env, baseline_loading)
        score_json = _json_floats_by_line(scores)

        for line_count in th_line_counts:
            selected = select_top_k_lines(scores, int(line_count))
            selected_scores = [float(scores[line_id]) for line_id in selected]
            th_audit_rows.append(
                {
                    "scenario_id": scenario_def.scenario_id,
                    "scenario_name": scenario_def.scenario_name,
                    "th_line_count": int(line_count),
                    "selected_line_ids": _line_key(selected),
                    "selected_line_scores_json": _json_floats_by_line({line_id: scores[line_id] for line_id in selected}),
                    "num_selected_lines": int(len(selected)),
                    "selected_score_mean": float(np.mean(selected_scores)),
                    "selected_score_min": float(np.min(selected_scores)),
                    "selected_score_max": float(np.max(selected_scores)),
                }
            )
            for lambda_r in lambda_values:
                rows.append(
                    _topology_row(
                        scenario_def,
                        baseline_r,
                        p_env,
                        scores,
                        selected,
                        lambda_r,
                        _stage_for_th_line_count(int(line_count)),
                        _stage_label_for_th_line_count(int(line_count)),
                        METHOD_TH,
                        f"top_{int(line_count)}_lines",
                        th_line_count=int(line_count),
                    )
                )

        ah_groups = area_heuristic_groups(_edge_array(scenario), scores, ah_top_fraction)
        ah_frame = ah_groups.copy()
        ah_frame.insert(0, "scenario_name", scenario_def.scenario_name)
        ah_frame.insert(0, "scenario_id", scenario_def.scenario_id)
        ah_frame["ah_top_fraction"] = float(ah_top_fraction)
        ah_frame["candidate_score_json"] = score_json
        ah_audit_frames.append(ah_frame)
        chosen = ah_groups[ah_groups["ah_selected_group"].astype(bool)].iloc[0]
        chosen_lines = _parse_line_key(chosen["ah_group_line_ids"])
        for lambda_r in lambda_values:
            rows.append(
                _topology_row(
                    scenario_def,
                    baseline_r,
                    p_env,
                    scores,
                    chosen_lines,
                    lambda_r,
                    AH_STAGE,
                    "AH connected top 30%",
                    METHOD_AH,
                    "connected_top30_highest_average_score",
                    ah_top_fraction=float(ah_top_fraction),
                    ah_group_id=chosen["ah_group_id"],
                    ah_group_line_ids=chosen["ah_group_line_ids"],
                    ah_group_average_score=float(chosen["ah_group_average_score"]),
                    ah_group_total_score=float(chosen["ah_group_total_score"]),
                )
            )
        print(f"[stage_h topology pool] {scenario_def.scenario_id}", flush=True)

    topology_pool = pd.DataFrame(rows).reset_index(drop=True)
    topology_pool["topology_iteration"] = topology_pool.groupby(
        ["scenario_id", "lambda_R", "heuristic_method"], sort=False
    ).cumcount() + 1
    p_env_table = pd.concat(p_env_frames, ignore_index=True, sort=False)
    ranking = build_scenario_baseline_loading_ranking(context, baseline_loading, p_env_frames)
    th_audit = pd.DataFrame(th_audit_rows)
    ah_audit = pd.concat(ah_audit_frames, ignore_index=True, sort=False)
    metadata = {
        "candidate_line_ids": candidate_line_ids,
        "physical_line_ids": physical_ids,
        "baseline_loading": baseline_loading.tolist(),
        "th_line_counts": [int(value) for value in th_line_counts],
        "ah_top_fraction": float(ah_top_fraction),
        "heuristic_methodology": {
            "TH": "Rank-based top-k shutoff over scenario-specific p_env_l * baseline_loading_l^2 candidate-line scores.",
            "AH": "Connected-network analogue: top 30 percent scored candidate lines grouped by network connected components; highest average-score group is shut off.",
        },
    }
    return topology_pool, p_env_table, ranking, canonical_scenarios, metadata, th_audit, ah_audit


def _topology_row(
    scenario_def: DecisionQualityScenario,
    baseline_r: float,
    p_env: Dict[int, float],
    scores: Dict[int, float],
    shutoff_line_ids: Sequence[int],
    lambda_r: float,
    stage: str,
    stage_label: str,
    heuristic_method: str,
    method_variant: str,
    **metadata,
) -> dict:
    selected_scores = [float(scores[int(line_id)]) for line_id in shutoff_line_ids]
    return {
        "scenario_id": scenario_def.scenario_id,
        "scenario_name": scenario_def.scenario_name,
        "stage": stage,
        "stage_label": stage_label,
        "lambda_R": float(lambda_r),
        "lambda_L": float(1.0 - float(lambda_r)),
        "lambda_case": _lambda_case(lambda_r),
        "proposal_method": f"stage_h_{heuristic_method.lower()}_{method_variant}",
        "heuristic_method": heuristic_method,
        "method_variant": method_variant,
        "shutoff_line_ids": _line_key(shutoff_line_ids),
        "baseline_R": float(baseline_r),
        "p_env_json": json.dumps({str(int(key)): float(value) for key, value in sorted(p_env.items())}, sort_keys=True),
        "topology_proxy_excludes_pac": True,
        "candidate_score_json": _json_floats_by_line(scores),
        "selected_score_mean": float(np.mean(selected_scores)) if selected_scores else 0.0,
        "selected_score_total": float(np.sum(selected_scores)) if selected_scores else 0.0,
        "selected_line_count": int(len(shutoff_line_ids)),
        **metadata,
    }


def _best_by_rho_scenario_lambda_method(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[np.isfinite(results["J_true"].astype(float))].copy()
    keys = ["model_type", "scenario_id", "lambda_R", "rho_phys", "stage", "stage_label", "heuristic_method", "method_variant"]
    if ok.empty:
        return ok
    return (
        ok.sort_values(keys + ["J_true", "L_shed", "num_shutoff_lines", "line_id_key"], kind="mergesort")
        .groupby(keys, as_index=False, dropna=False)
        .first()
        .reset_index(drop=True)
    )


def _best_method_oracle(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return best
    keys = ["model_type", "scenario_id", "lambda_R", "rho_phys", "heuristic_method"]
    return (
        best.sort_values(keys + ["J_true", "L_shed", "num_shutoff_lines", "line_id_key"], kind="mergesort")
        .groupby(keys, as_index=False, dropna=False)
        .first()
        .reset_index(drop=True)
    )


def _load_stage_g_reference(reference_run: Path, lambda_values: Sequence[float], rho_values: Sequence[float]) -> pd.DataFrame:
    path = reference_run / "tables" / "best_by_rho_scenario_lambda_stage.csv"
    read_path = path
    if not path.exists() and sys.platform.startswith("win"):
        extended = "\\\\?\\" + str(path.resolve())
        if Path(extended).exists():
            read_path = Path(extended)
    if not read_path.exists():
        return pd.DataFrame()
    ref = pd.read_csv(read_path)
    ref = ref[
        ref["lambda_R"].astype(float).isin([float(value) for value in lambda_values])
        & ref["rho_phys"].astype(float).isin([float(value) for value in rho_values])
    ].copy()
    if ref.empty:
        return ref
    ref["heuristic_method"] = METHOD_STAGE_G
    ref["method_variant"] = ref["stage"].astype(str)
    ref["display_method"] = ref.get("stage_label", ref["stage"]).astype(str)
    return ref.reset_index(drop=True)


def _combine_expected_with_reference(
    expected: pd.DataFrame,
    reference_best: pd.DataFrame,
    scenarios: List[DecisionQualityScenario],
) -> pd.DataFrame:
    combined = expected.copy()
    if not reference_best.empty:
        scenario_ids = {str(scenario.scenario_id) for scenario in scenarios}
        reference_best = reference_best[reference_best["scenario_id"].astype(str).isin(scenario_ids)].copy()
    if not reference_best.empty:
        ref_expected = _expected_vs_observed_all_lambdas(reference_best, scenarios)
        ref_expected["heuristic_method"] = METHOD_STAGE_G
        ref_expected["method_variant"] = ref_expected["stage"].astype(str)
        combined = pd.concat([ref_expected, combined], ignore_index=True, sort=False)
    return combined


def _read_checkpoint_tables(checkpoints_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(checkpoints_dir / "continuous_recourse_results.csv"),
        pd.read_csv(checkpoints_dir / "continuous_objective_call_trace.csv"),
        pd.read_csv(checkpoints_dir / "load_shedding_provenance.csv"),
        pd.read_csv(checkpoints_dir / "wildfire_risk_provenance.csv"),
        pd.read_csv(checkpoints_dir / "controlled_state_consistency.csv"),
        pd.read_csv(checkpoints_dir / "masking_clamping_audit.csv"),
    )


def finalize_stage_h_run(
    run_dir: Path,
    model_type: str,
    scenario_ids: Sequence[str],
    lambda_values: Sequence[float],
    rho_values: Sequence[float],
    delta_qg_bound_mvar: float,
    stage_g_reference_run: Path,
) -> Path:
    context = _make_continuous_context(model_type, delta_qg_bound_mvar)
    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    scenarios = _canonical_scenarios(scenario_ids, scenario)

    tables_dir = run_dir / "tables"
    checkpoints_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    topology_pool = pd.read_csv(tables_dir / "heuristic_topology_pool.csv")
    p_env_table = pd.read_csv(tables_dir / "scenario_p_env_target_margin.csv")
    ranking = pd.read_csv(tables_dir / "scenario_baseline_loading_ranking.csv")
    results, traces, load_prov, risk_prov, controlled, mask = _read_checkpoint_tables(checkpoints_dir)
    best = _best_by_rho_scenario_lambda_method(results)
    best_oracle = _best_method_oracle(best)
    expected = _expected_vs_observed_all_lambdas(best, scenarios)
    expected = expected.merge(
        best[["scenario_id", "lambda_R", "rho_phys", "stage", "heuristic_method", "method_variant"]],
        on=["scenario_id", "lambda_R", "rho_phys", "stage"],
        how="left",
    )
    reference_best = _load_stage_g_reference(stage_g_reference_run, lambda_values, rho_values)
    reference_best = reference_best[reference_best["scenario_id"].astype(str).isin({str(item) for item in scenario_ids})].copy() if not reference_best.empty else reference_best
    expected_with_ref = _combine_expected_with_reference(expected, reference_best, scenarios)
    fixed_vs = _fixed_vs_continuous(results)
    checks = _methodology_checks(
        results,
        p_env_table,
        ranking,
        load_prov,
        risk_prov,
        controlled,
        mask,
        scenario,
        candidate_line_ids,
    )

    write_dataframe(tables_dir / "continuous_recourse_results.csv", results)
    write_dataframe(tables_dir / "continuous_objective_call_trace.csv", traces)
    write_dataframe(tables_dir / "load_shedding_provenance.csv", load_prov)
    write_dataframe(tables_dir / "wildfire_risk_provenance.csv", risk_prov)
    write_dataframe(tables_dir / "controlled_state_consistency.csv", controlled)
    write_dataframe(tables_dir / "masking_clamping_audit.csv", mask)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_method.csv", best)
    write_dataframe(tables_dir / "best_oracle_by_rho_scenario_lambda_method.csv", best_oracle)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho_method.csv", expected_with_ref)
    write_dataframe(tables_dir / "runtime_and_solver_diagnostics.csv", _runtime_summary(results))
    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_dataframe(tables_dir / "stage_g_reference_best_all_stages.csv", reference_best)
    write_dataframe(tables_dir / "topology_proposal_pool.csv", topology_pool)
    write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "fixed_vs_continuous_comparison.csv", fixed_vs)
    physics, cost, line23, tradeoff = _plot_cross_and_summary(results, best, expected, mask, plots_dir)
    write_dataframe(tables_dir / "physics_sensitivity_by_rho.csv", physics)
    write_dataframe(tables_dir / "cost_of_feasibility_by_rho.csv", cost)
    write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
    write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
    _plot_stage_h_outputs(results, best, expected_with_ref, reference_best, plots_dir)
    write_json(
        run_dir / "finalization_summary.json",
        {
            "status": "complete",
            "topology_rows_before_rho": int(len(topology_pool)),
            "continuous_evaluations": int(len(results)),
            "hard_methodology_failures": int((~checks["passed"].astype(bool) & checks["severity"].astype(str).eq("hard")).sum()),
        },
    )
    return run_dir


def _plot_stage_h_outputs(results: pd.DataFrame, best: pd.DataFrame, expected: pd.DataFrame, reference_best: pd.DataFrame, plots_dir: Path) -> None:
    _plot_per_rho(results, best, expected, reference_best, plots_dir)
    _plot_summary(best, expected, reference_best, plots_dir)


def _plot_per_rho(results: pd.DataFrame, best: pd.DataFrame, expected: pd.DataFrame, reference_best: pd.DataFrame, plots_dir: Path) -> None:
    for (rho, scenario_id), _ in results.groupby(["rho_phys", "scenario_id"], sort=True):
        out = plots_dir / "per_rho" / f"rho{rho:g}" / str(scenario_id)
        out.mkdir(parents=True, exist_ok=True)
        local_best = best[
            np.isclose(best["rho_phys"].astype(float), float(rho)) & best["scenario_id"].astype(str).eq(str(scenario_id))
        ].copy()
        local_ref = reference_best[
            np.isclose(reference_best.get("rho_phys", pd.Series(dtype=float)).astype(float), float(rho))
            & reference_best.get("scenario_id", pd.Series(dtype=str)).astype(str).eq(str(scenario_id))
        ].copy() if not reference_best.empty else pd.DataFrame()
        _plot_pareto(local_best, local_ref, out / "pareto_frontier_scatter.png")
        _plot_expected_table(
            expected[
                np.isclose(expected["rho_phys"].astype(float), float(rho))
                & expected["scenario_id"].astype(str).eq(str(scenario_id))
            ],
            out / "expected_vs_selected_shutoff_lines.png",
        )
        _plot_lambda_objective(local_best, local_ref, out / "traditional_lambda_objective_convergence.png")
        _plot_shutoffs_vs_objective(local_best, local_ref, out / "num_shutoffs_vs_objective_by_method.png")
        _plot_th_topk_sensitivity(local_best, out / "th_topk_sensitivity.png")


def _plot_pareto(local_best: pd.DataFrame, local_ref: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    all_points = []
    if not local_ref.empty:
        markers = {
            "Stage D exhaustive": "x",
            "Stage E k2": "P",
            "Stage E unconstrained": "*",
        }
        for label, frame in local_ref.groupby("stage_label", sort=True):
            all_points.append(frame.assign(plot_family=str(label)))
            ax.scatter(
                frame["R_norm"].astype(float),
                frame["L_shed"].astype(float),
                marker=markers.get(str(label), "x"),
                s=90,
                label=str(label),
                alpha=0.85,
            )
    th = local_best[local_best["heuristic_method"].astype(str).eq(METHOD_TH)].copy()
    if not th.empty:
        all_points.append(th.assign(plot_family="TH top-k"))
        scatter = ax.scatter(
            th["R_norm"].astype(float),
            th["L_shed"].astype(float),
            c=th["th_line_count"].astype(float),
            cmap="viridis_r",
            marker="o",
            s=58,
            edgecolor="white",
            linewidth=0.5,
            label="TH top-k",
        )
        fig.colorbar(scatter, ax=ax, label="TH selected line count")
    ah = local_best[local_best["heuristic_method"].astype(str).eq(METHOD_AH)].copy()
    if not ah.empty:
        all_points.append(ah.assign(plot_family="AH"))
        ax.scatter(ah["R_norm"].astype(float), ah["L_shed"].astype(float), color="tab:green", marker="s", s=75, label="AH")
    if all_points:
        combined = pd.concat(all_points, ignore_index=True, sort=False)
        points = combined.drop_duplicates(["R_norm", "L_shed"]).sort_values(["R_norm", "L_shed"]).copy()
        nondominated = []
        best_l = np.inf
        for _, row in points.iterrows():
            l_shed = float(row["L_shed"])
            is_nd = l_shed < best_l - 1e-10
            nondominated.append(is_nd)
            if is_nd:
                best_l = l_shed
        nd = points[nondominated]
        if len(nd) >= 2:
            ax.plot(nd["R_norm"].astype(float), nd["L_shed"].astype(float), color="black", linewidth=1.4, alpha=0.7, label="nondominated envelope")
    ax.set_xlabel("R_norm")
    ax.set_ylabel("L_shed")
    ax.set_title("Risk-load comparison: Stage G baselines vs Stage H heuristics")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_expected_table(frame: pd.DataFrame, path: Path) -> None:
    if frame.empty:
        return
    work = frame.copy()
    work["row_label"] = work.get("stage_label", work["stage"]).astype(str)
    lambdas = sorted(work["lambda_R"].astype(float).unique(), reverse=True)
    rows = sorted(work["row_label"].unique())
    cell_text = []
    for row_label in rows:
        row_cells = []
        for lambda_r in lambdas:
            cell = work[work["row_label"].eq(row_label) & np.isclose(work["lambda_R"].astype(float), lambda_r)]
            if cell.empty:
                row_cells.append("")
                continue
            item = cell.sort_values("J_true").iloc[0]
            observed = str(item.get("observed_shutoff_line_ids", ""))
            recall = float(item.get("target_recall", item.get("target_overlap_fraction", 0.0)))
            precision = float(item.get("target_precision", 0.0))
            pac = float(item.get("PAC_total", 0.0))
            row_cells.append(f"{observed or '-'}\nrecall={recall:.2f}\nprecision={precision:.2f}\nPAC={pac:.3g}")
        cell_text.append(row_cells)
    fig, ax = plt.subplots(figsize=(max(8, 2.4 * len(lambdas)), max(4, 0.55 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(cellText=cell_text, rowLabels=rows, colLabels=[f"lambda_R={value:.1f}" for value in lambdas], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    ax.set_title("Expected vs selected shutoff lines")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_lambda_objective(local_best: pd.DataFrame, local_ref: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not local_ref.empty:
        for label, ref in local_ref.groupby("stage_label", sort=True):
            ref = ref.sort_values("lambda_R")
            ax.plot(ref["lambda_R"], ref["J_true"], marker="x", linewidth=2, label=str(label))
    for label, frame in local_best.groupby("stage_label", sort=True):
        frame = frame.sort_values("lambda_R")
        alpha = 0.45 if str(label).startswith("TH") else 0.95
        lw = 1.0 if str(label).startswith("TH") else 2.0
        ax.plot(frame["lambda_R"], frame["J_true"], marker="o", alpha=alpha, linewidth=lw, label=str(label))
    ax.set_xlabel("lambda_R")
    ax.set_ylabel("J_true")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_shutoffs_vs_objective(local_best: pd.DataFrame, local_ref: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not local_ref.empty:
        ax.scatter(local_ref["num_shutoff_lines"], local_ref["J_true"], color="black", marker="x", s=80, label=METHOD_STAGE_G)
    for method, frame in local_best.groupby("heuristic_method", sort=True):
        ax.scatter(frame["num_shutoff_lines"], frame["J_true"], s=55, label=str(method), alpha=0.8)
    ax.set_xlabel("Selected shutoff line count")
    ax.set_ylabel("J_true")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_th_topk_sensitivity(local_best: pd.DataFrame, path: Path) -> None:
    th = local_best[local_best["heuristic_method"].astype(str).eq(METHOD_TH)].copy()
    if th.empty:
        return
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    metrics = ["R_norm", "L_shed", "PAC_total", "J_true", "num_shutoff_lines"]
    for ax, metric in zip(axes.ravel(), metrics):
        for lambda_r, frame in th.groupby("lambda_R", sort=True):
            frame = frame.sort_values("th_line_count")
            ax.plot(frame["th_line_count"], frame[metric], marker="o", label=f"lambda_R={float(lambda_r):.1f}")
        ax.set_title(metric)
        ax.grid(alpha=0.25)
    axes.ravel()[-1].axis("off")
    axes.ravel()[0].legend(fontsize=8)
    for ax in axes[-1, :2]:
        ax.set_xlabel("TH selected line count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary(best: pd.DataFrame, expected: pd.DataFrame, reference_best: pd.DataFrame, plots_dir: Path) -> None:
    out = plots_dir / "summary"
    out.mkdir(parents=True, exist_ok=True)
    combined = _combined_best_for_summary(best, reference_best)
    if combined.empty:
        return
    _bar_summary(combined, ["display_method", "lambda_R", "rho_phys"], "J_true", out / "average_objective_by_method_lambda_rho.png")
    _multi_metric_summary(combined, out / "average_risk_load_pac_by_method.png")
    if not expected.empty:
        expected_plot = expected.assign(display_method=expected.get("stage_label", expected["stage"]))
        _bar_summary(expected_plot, ["display_method"], "target_recall", out / "target_recall_by_method.png")
        _bar_summary(expected_plot, ["display_method"], "target_precision", out / "target_precision_by_method.png")
        _optional_multi_metric_summary(expected_plot, ["target_recall", "target_precision"], out / "target_recall_precision_by_method.png")
        _bar_summary(expected_plot, ["display_method"], "target_overlap_fraction", out / "target_overlap_by_method.png")
    _bar_summary(combined, ["display_method"], "num_shutoff_lines", out / "shutoff_count_by_method.png")
    _optional_multi_metric_summary(
        combined,
        ["PAC_operational", "PAC_AC", "PAC_model_consistency"],
        out / "average_operational_ac_model_pac_by_method.png",
    )
    _optional_multi_metric_summary(
        combined,
        ["L_shed_cmd", "L_shed_gridfm_raw", "L_shed_gridfm_effective", "L_shed_hybrid"],
        out / "load_shed_cmd_gridfm_hybrid_by_method.png",
    )
    _optional_multi_metric_summary(
        combined,
        ["PAC_cmd_load", "PAC_cmd_Pg", "PAC_cmd_Qg", "PAC_generator_limits_raw"],
        out / "model_alignment_violation_by_method.png",
    )
    _optional_multi_metric_summary(
        combined,
        ["PAC_p_balance", "PAC_q_balance"],
        out / "ac_balance_violation_by_method.png",
    )
    th = best[best["heuristic_method"].astype(str).eq(METHOD_TH)].copy()
    if not th.empty and "th_line_count" in th.columns:
        chosen = _best_method_oracle(th)
        _bar_summary(chosen.assign(display_method=chosen["scenario_id"].astype(str) + " lambda=" + chosen["lambda_R"].astype(str)), ["display_method", "rho_phys"], "th_line_count", out / "best_th_topk_by_scenario_lambda_rho.png")


def _combined_best_for_summary(best: pd.DataFrame, reference_best: pd.DataFrame) -> pd.DataFrame:
    frames = []
    if not reference_best.empty:
        ref = reference_best.copy()
        ref["display_method"] = ref.get("stage_label", ref.get("stage", METHOD_STAGE_G)).astype(str)
        frames.append(ref)
    if not best.empty:
        work = best.copy()
        work["display_method"] = work["stage_label"].astype(str)
        frames.append(work)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _bar_summary(frame: pd.DataFrame, group_keys: List[str], value: str, path: Path) -> None:
    summary = frame.groupby(group_keys, as_index=False)[value].mean()
    labels = summary[group_keys].astype(str).agg(" | ".join, axis=1)
    fig, ax = plt.subplots(figsize=(max(8, 0.42 * len(summary)), 5))
    ax.bar(range(len(summary)), summary[value].astype(float))
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel(f"mean {value}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _multi_metric_summary(frame: pd.DataFrame, path: Path) -> None:
    summary = frame.groupby("display_method", as_index=False)[["R_norm", "L_shed", "PAC_total"]].mean()
    labels = summary["display_method"].astype(str).tolist()
    x = np.arange(len(summary))
    width = 0.26
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(summary)), 5))
    for offset, metric in zip([-width, 0, width], ["R_norm", "L_shed", "PAC_total"]):
        ax.bar(x + offset, summary[metric].astype(float), width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _optional_multi_metric_summary(frame: pd.DataFrame, metrics: List[str], path: Path) -> None:
    available = [metric for metric in metrics if metric in frame.columns]
    if not available or frame.empty:
        return
    work = frame.copy()
    for metric in available:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    summary = work.groupby("display_method", as_index=False)[available].mean()
    labels = summary["display_method"].astype(str).tolist()
    x = np.arange(len(summary))
    width = min(0.8 / max(len(available), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(summary)), 5))
    offsets = (np.arange(len(available)) - (len(available) - 1) / 2.0) * width
    for offset, metric in zip(offsets, available):
        ax.bar(x + offset, summary[metric].astype(float), width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def resume_stage_h_run(
    run_dir: Path,
    call_budget: int | None = None,
    stage_g_reference_run: Path | None = None,
) -> Path:
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Cannot resume Stage H run without metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    model_type = str(metadata.get("model_type", "gnn"))
    scenario_ids = list(metadata.get("scenario_ids", ["S1", "S2", "S3", "S4", "S5"]))
    lambda_values = [float(value) for value in metadata.get("lambda_values", DEFAULT_LAMBDAS)]
    rho_values = [float(value) for value in metadata.get("rho_phys", DEFAULT_RHO_VALUES)]
    delta_qg_bound_mvar = float(metadata.get("delta_qg_bound_mvar", 10.0))
    effective_call_budget = int(call_budget if call_budget is not None else metadata.get("call_budget", DEFAULT_CALL_BUDGET))
    effective_reference = Path(stage_g_reference_run or metadata.get("stage_g_reference_run", DEFAULT_STAGE_G_REFERENCE_RUN))

    context = _make_continuous_context(model_type, delta_qg_bound_mvar)
    scenario = context["scenario"]
    candidate_line_ids = canonicalize_line_ids(scenario, FIXED_T0P30_CANDIDATE_LINE_IDS)
    tables_dir = run_dir / "tables"
    checkpoints_dir = run_dir / "checkpoints"
    progress_path = run_dir / "progress.json"
    topology_pool = pd.read_csv(tables_dir / "heuristic_topology_pool.csv")
    _run_continuous_pool(
        context,
        topology_pool,
        candidate_line_ids,
        rho_values,
        effective_call_budget,
        checkpoint_dir=checkpoints_dir,
        progress_path=progress_path,
    )
    return finalize_stage_h_run(
        run_dir=run_dir,
        model_type=model_type,
        scenario_ids=scenario_ids,
        lambda_values=lambda_values,
        rho_values=rho_values,
        delta_qg_bound_mvar=delta_qg_bound_mvar,
        stage_g_reference_run=effective_reference,
    )


def run_stage_h_heuristic_baseline_comparison(
    model_types: Sequence[str],
    scenario_ids: Sequence[str],
    lambda_values: Sequence[float],
    rho_values: Sequence[float],
    th_line_counts: Sequence[int],
    ah_top_fraction: float,
    output_root: Path,
    call_budget: int,
    delta_qg_bound_mvar: float,
    stage_g_reference_run: Path,
    audit_only: bool,
) -> List[Path]:
    run_dirs = []
    for model_type in model_types:
        context = _make_continuous_context(model_type, delta_qg_bound_mvar)
        topology_pool, p_env_table, ranking, scenarios, metadata, th_audit, ah_audit = _build_stage_h_topology_pool(
            context,
            scenario_ids,
            lambda_values,
            th_line_counts,
            ah_top_fraction,
        )
        run_dir = make_run_dir(output_root, f"run_{model_type}")
        tables_dir = run_dir / "tables"
        checkpoints_dir = run_dir / "checkpoints"
        plots_dir = run_dir / "plots"
        tables_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        write_dataframe(tables_dir / "heuristic_topology_pool.csv", topology_pool)
        write_dataframe(tables_dir / "th_topk_audit_by_scenario.csv", th_audit)
        write_dataframe(tables_dir / "ah_group_audit_by_scenario.csv", ah_audit)
        write_dataframe(tables_dir / "scenario_p_env_target_margin.csv", p_env_table)
        write_dataframe(tables_dir / "scenario_baseline_loading_ranking.csv", ranking)
        write_json(
            run_dir / "run_metadata.json",
            {
                "stage": "stage_h_heuristic_baseline_comparison",
                "model_type": model_type,
                "scenario_ids": list(scenario_ids),
                "lambda_values": _as_float_list(lambda_values),
                "rho_phys": _as_float_list(rho_values),
                "th_line_counts": [int(value) for value in th_line_counts],
                "ah_top_fraction": float(ah_top_fraction),
                "call_budget": int(call_budget),
                "delta_qg_bound_mvar": float(delta_qg_bound_mvar),
                "audit_only": bool(audit_only),
                "stage_g_reference_run": str(stage_g_reference_run),
                "expected_topology_rows_before_rho": int(len(topology_pool)),
                "expected_continuous_evaluations": int(len(topology_pool) * len(rho_values)),
                "baseline_loading_source": BASELINE_LOADING_SOURCE,
                "p_env_mode": P_ENV_MODE_TARGET_MARGIN,
                "load_shed_mode": "hybrid",
                "stage_h_reference_methods": ["Stage D exhaustive", "Stage E k2", "Stage E unconstrained", "TH", "AH"],
                "git": git_metadata(),
                **metadata,
            },
        )
        if audit_only:
            write_json(run_dir / "progress.json", {"status": "audit_only", "completed_topology_rho_optimizations": 0})
            run_dirs.append(run_dir)
            continue

        progress_path = run_dir / "progress.json"
        results, traces, load_prov, risk_prov, controlled, mask = _run_continuous_pool(
            context,
            topology_pool,
            metadata["candidate_line_ids"],
            rho_values,
            call_budget,
            checkpoint_dir=checkpoints_dir,
            progress_path=progress_path,
        )
        best = _best_by_rho_scenario_lambda_method(results)
        best_oracle = _best_method_oracle(best)
        expected = _expected_vs_observed_all_lambdas(best, scenarios)
        expected = expected.merge(
            best[["scenario_id", "lambda_R", "rho_phys", "stage", "heuristic_method", "method_variant"]],
            on=["scenario_id", "lambda_R", "rho_phys", "stage"],
            how="left",
        )
        reference_best = _load_stage_g_reference(stage_g_reference_run, lambda_values, rho_values)
        reference_best = reference_best[reference_best["scenario_id"].astype(str).isin({str(item) for item in scenario_ids})].copy() if not reference_best.empty else reference_best
        expected_with_ref = _combine_expected_with_reference(expected, reference_best, scenarios)
        fixed_vs = _fixed_vs_continuous(results)
        checks = _methodology_checks(
            results,
            p_env_table,
            ranking,
            load_prov,
            risk_prov,
            controlled,
            mask,
            context["scenario"],
            metadata["candidate_line_ids"],
        )

        write_dataframe(tables_dir / "continuous_recourse_results.csv", results)
        write_dataframe(tables_dir / "continuous_objective_call_trace.csv", traces)
        write_dataframe(tables_dir / "load_shedding_provenance.csv", load_prov)
        write_dataframe(tables_dir / "wildfire_risk_provenance.csv", risk_prov)
        write_dataframe(tables_dir / "controlled_state_consistency.csv", controlled)
        write_dataframe(tables_dir / "masking_clamping_audit.csv", mask)
        write_dataframe(tables_dir / "best_by_rho_scenario_lambda_method.csv", best)
        write_dataframe(tables_dir / "best_oracle_by_rho_scenario_lambda_method.csv", best_oracle)
        write_dataframe(tables_dir / "expected_vs_selected_by_rho_method.csv", expected_with_ref)
        write_dataframe(tables_dir / "runtime_and_solver_diagnostics.csv", _runtime_summary(results))
        write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
        write_dataframe(tables_dir / "stage_g_reference_best_all_stages.csv", reference_best)
        write_dataframe(tables_dir / "topology_proposal_pool.csv", topology_pool)
        write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
        write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
        write_dataframe(tables_dir / "fixed_vs_continuous_comparison.csv", fixed_vs)
        physics, cost, line23, tradeoff = _plot_cross_and_summary(results, best, expected, mask, plots_dir)
        write_dataframe(tables_dir / "physics_sensitivity_by_rho.csv", physics)
        write_dataframe(tables_dir / "cost_of_feasibility_by_rho.csv", cost)
        write_dataframe(tables_dir / "line23_frequency_by_rho.csv", line23)
        write_dataframe(tables_dir / "rho_tradeoff_summary.csv", tradeoff)
        _plot_stage_h_outputs(results, best, expected_with_ref, reference_best, plots_dir)
        run_dirs.append(run_dir)
    return run_dirs


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Stage H TH/AH heuristic baseline comparison.")
    parser.add_argument("--models", nargs="+", default=["gnn"])
    parser.add_argument("--scenario-ids", nargs="+", default=["S1", "S2", "S3", "S4", "S5"])
    parser.add_argument("--lambda-values", nargs="+", type=float, default=DEFAULT_LAMBDAS)
    parser.add_argument("--rho-phys", nargs="+", type=float, default=DEFAULT_RHO_VALUES)
    parser.add_argument("--th-line-counts", nargs="+", type=int, default=DEFAULT_TH_LINE_COUNTS)
    parser.add_argument("--ah-top-fraction", type=float, default=DEFAULT_AH_TOP_FRACTION)
    parser.add_argument("--call-budget", type=int, default=DEFAULT_CALL_BUDGET)
    parser.add_argument("--delta-qg-bound-mvar", type=float, default=10.0)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument("--stage-g-reference-run", type=Path, default=DEFAULT_STAGE_G_REFERENCE_RUN)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--finalize-run", type=Path, default=None)
    parser.add_argument("--resume-run", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.resume_run is not None:
        run_dir = resume_stage_h_run(
            run_dir=args.resume_run,
            call_budget=args.call_budget,
            stage_g_reference_run=args.stage_g_reference_run,
        )
        print(run_dir)
        return
    if args.finalize_run is not None:
        run_dir = finalize_stage_h_run(
            run_dir=args.finalize_run,
            model_type=args.models[0],
            scenario_ids=args.scenario_ids,
            lambda_values=args.lambda_values,
            rho_values=args.rho_phys,
            delta_qg_bound_mvar=args.delta_qg_bound_mvar,
            stage_g_reference_run=args.stage_g_reference_run,
        )
        print(run_dir)
        return
    run_dirs = run_stage_h_heuristic_baseline_comparison(
        model_types=args.models,
        scenario_ids=args.scenario_ids,
        lambda_values=args.lambda_values,
        rho_values=args.rho_phys,
        th_line_counts=args.th_line_counts,
        ah_top_fraction=args.ah_top_fraction,
        output_root=args.output_root,
        call_budget=args.call_budget,
        delta_qg_bound_mvar=args.delta_qg_bound_mvar,
        stage_g_reference_run=args.stage_g_reference_run,
        audit_only=args.audit_only,
    )
    for run_dir in run_dirs:
        print(run_dir)


if __name__ == "__main__":
    main()
