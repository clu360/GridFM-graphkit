from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.reporting import git_metadata, make_run_dir, write_dataframe, write_json
from experiments.test.wildfire_tests.stage_f_decision_quality.run_stage_f_physics_decision_quality import (
    _nondominated_mask,
)
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    STAGE_E_K2,
    _expected_vs_observed_all_lambdas,
    _make_continuous_context,
    run_revised_continuous_implementation,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.ac_projection import project_finalists_with_cache
from experiments.test.wildfire_tests.stage_i_dc_comparison.dc_formulation import (
    AH_K2,
    AH_K2_LABEL,
    STAGE_I_A,
    STAGE_I_A_LABEL,
    STAGE_I_B,
    STAGE_I_B_LABEL,
    TH_TOP1,
    TH_TOP1_LABEL,
    TH_TOP2,
    TH_TOP2_LABEL,
    build_budgeted_heuristic_pool,
    build_dc_network,
    build_stage_ia_topology_pool,
    canonical_scenarios_for_context,
    evaluate_stage_ia_pool,
    evaluate_stage_ib_grid,
    solve_stage_ia_dc_recourse,
)


RESULT_ROOT = RESULTS_ROOT / "leq" / "stage_h" / "DC Approximation + Baseline Heuristic Comparison"
MAIN_RESULTS = RESULT_ROOT / "main_results"
MLD_RESULTS = RESULT_ROOT / "MLD"
DEFAULT_LAMBDAS = [0.0, 0.2, 0.5, 0.8, 1.0]
DEFAULT_RHO_VALUES = [0.0, 2.0]
DEFAULT_SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


def _line_key(values) -> str:
    if values is None or (isinstance(values, float) and np.isnan(values)):
        return ""
    if isinstance(values, str):
        return values
    return ",".join(str(int(v)) for v in sorted({int(x) for x in values}))


def _parse_line_key(value) -> list[int]:
    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return sorted({int(part) for part in text.split(",") if part.strip()})


def _long_path(path: Path) -> str:
    path = Path(path)
    if os.name == "nt":
        absolute = str(path.absolute())
        return absolute if absolute.startswith("\\\\?\\") else "\\\\?\\" + absolute
    return str(path)


def _mkdir(path: Path) -> None:
    path = Path(path)
    if os.name == "nt":
        Path(_long_path(path)).mkdir(parents=True, exist_ok=True)
    else:
        path.mkdir(parents=True, exist_ok=True)


def _make_short_run_dir(output_root: Path) -> Path:
    output_root = Path(output_root)
    _mkdir(output_root)
    for idx in range(1000):
        name = "r" if idx == 0 else f"r{idx}"
        run_dir = output_root / name
        if not run_dir.exists():
            _mkdir(run_dir)
            return run_dir
    stamp = datetime.now().strftime("%H%M%S")
    run_dir = output_root / f"r{stamp}"
    _mkdir(run_dir)
    return run_dir


def _savefig(fig, path: Path, **kwargs) -> None:
    _mkdir(Path(path).parent)
    staging_dir = REPO_ROOT / "tmp" / "stage_h_plot_staging"
    _mkdir(staging_dir)
    suffix = Path(path).suffix or ".png"
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(staging_dir))
    tmp_path = Path(handle.name)
    handle.close()
    try:
        fig.savefig(tmp_path, **kwargs)
        os.replace(str(tmp_path), _long_path(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _best_by_method(results: pd.DataFrame) -> pd.DataFrame:
    ok = results[np.isfinite(results["J_true"].astype(float))].copy()
    if ok.empty:
        return ok
    keys = ["model_type", "scenario_id", "scenario_name", "lambda_R", "lambda_L", "rho_phys", "stage", "stage_label"]
    for optional in ["lambda_R_proxy", "lambda_L_proxy", "lambda_proxy_case"]:
        if optional in ok.columns:
            keys.append(optional)
    return (
        ok.sort_values(keys + ["J_true", "L_shed", "num_shutoff_lines", "line_id_key"], kind="mergesort")
        .groupby(keys, as_index=False, dropna=False)
        .first()
        .reset_index(drop=True)
    )


def _load_stage_e_best(
    stage_e_run: Path,
    scenario_ids: Sequence[str],
    lambda_values: Sequence[float],
    rho_values: Sequence[float],
    proxy_lambda_values: Sequence[float] | None = None,
) -> pd.DataFrame:
    path = stage_e_run / "tables" / "best_by_rho_scenario_lambda_stage.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    requested_proxy = None if proxy_lambda_values is None else [float(v) for v in proxy_lambda_values]
    if "lambda_R_proxy" not in frame.columns:
        frame["lambda_R_proxy"] = frame["lambda_R"].astype(float)
        frame["lambda_L_proxy"] = 1.0 - frame["lambda_R"].astype(float)
        frame["lambda_proxy_case"] = frame["lambda_case"] if "lambda_case" in frame.columns else frame["lambda_R"].map(lambda v: f"lambda_{float(v):g}")
        if requested_proxy is not None and sorted(requested_proxy) != sorted(float(v) for v in lambda_values):
            raise ValueError(
                f"Stage E reference run {stage_e_run} has no lambda_R_proxy column and cannot be reused for separated proxy lambdas {requested_proxy}."
            )
    frame = frame[
        frame["stage"].astype(str).eq(STAGE_E_K2)
        & frame["scenario_id"].astype(str).isin([str(v) for v in scenario_ids])
        & frame["lambda_R"].astype(float).isin([float(v) for v in lambda_values])
        & frame["rho_phys"].astype(float).isin([float(v) for v in rho_values])
    ].copy()
    if requested_proxy is not None:
        frame = frame[frame["lambda_R_proxy"].astype(float).isin(requested_proxy)].copy()
    if frame.empty:
        return frame
    frame["method_family"] = "Stage E"
    return frame.reset_index(drop=True)


def _method_display_order() -> dict[str, int]:
    return {
        "Stage E k2": 0,
        STAGE_I_A_LABEL: 1,
        STAGE_I_B_LABEL: 2,
        TH_TOP1_LABEL: 3,
        TH_TOP2_LABEL: 4,
        AH_K2_LABEL: 5,
    }


def _plot_pareto(best: pd.DataFrame, out_dir: Path) -> None:
    for (rho, scenario_id), local in best.groupby(["rho_phys", "scenario_id"], sort=True):
        fig, ax = plt.subplots(figsize=(8.3, 5.8))
        for label, frame in sorted(local.groupby("stage_label"), key=lambda item: _method_display_order().get(str(item[0]), 99)):
            frame = frame.sort_values("lambda_R")
            x = frame["L_shed"].astype(float).to_numpy()
            y = frame["R_norm"].astype(float).to_numpy()
            marker = "o"
            linestyle = "-"
            if str(label).startswith("TH") or str(label).startswith("AH"):
                marker = "s" if str(label).startswith("AH") else "^"
                linestyle = "None"
            ax.plot(x, y, marker=marker, linestyle=linestyle, label=str(label), alpha=0.88)
            for _, row in frame.iterrows():
                ax.annotate(f"{float(row['lambda_R']):.1f}", (float(row["L_shed"]), float(row["R_norm"])), fontsize=7, alpha=0.75)
        comparable = local[~local["stage_label"].astype(str).str.startswith(("TH", "AH"))].copy()
        if not comparable.empty:
            points = comparable.drop_duplicates(["L_shed", "R_norm"]).sort_values(["L_shed", "R_norm"])
            nd = points[_nondominated_mask(points)] if len(points) else points
            if len(nd) >= 2:
                ax.plot(nd["L_shed"], nd["R_norm"], color="black", linewidth=1.1, alpha=0.5, label="combined nondominated")
        ax.set_xlabel("L_shed")
        ax.set_ylabel("R_norm")
        ax.set_title(f"Pareto scatter: {scenario_id}, rho={float(rho):g}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "per_rho" / f"rho{float(rho):g}" / str(scenario_id) / "pareto_frontier_scatter.png"
        _savefig(fig, path, dpi=180)
        plt.close(fig)


def _plot_metric_lines(best: pd.DataFrame, projections: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("L_shed", "effective_load_shedding_by_lambda.png", "Effective load shedding"),
        ("PAC_common_op_overlap", "common_operational_diagnostic_by_lambda.png", "Common operational diagnostic"),
    ]
    for metric, filename, ylabel in metrics:
        if metric not in best:
            continue
        for (rho, scenario_id), local in best.groupby(["rho_phys", "scenario_id"], sort=True):
            fig, ax = plt.subplots(figsize=(8.0, 5.0))
            for label, frame in local.groupby("stage_label", sort=True):
                frame = frame.sort_values("lambda_R")
                ax.plot(frame["lambda_R"], frame[metric], marker="o", label=str(label))
            ax.set_xlabel("lambda_R")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel}: {scenario_id}, rho={float(rho):g}")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            path = out_dir / "per_rho" / f"rho{float(rho):g}" / str(scenario_id) / filename
            _savefig(fig, path, dpi=180)
            plt.close(fig)

    if not projections.empty:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        plot_frame = projections.copy()
        plot_frame["plot_value"] = plot_frame["D_proj_total"].astype(float)
        for label, frame in plot_frame.groupby("stage_label", sort=True):
            ax.plot(frame["lambda_R"], frame["plot_value"], marker="o", linestyle="None", label=str(label))
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("AC projection distance")
        ax.set_title("AC projection distance for selected finalists")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / "projection_distance_by_lambda.png"
        _savefig(fig, path, dpi=180)
        plt.close(fig)


def _plot_expected_table(expected: pd.DataFrame, out_dir: Path) -> None:
    if expected.empty:
        return
    for (rho, scenario_id), frame in expected.groupby(["rho_phys", "scenario_id"], sort=True):
        work = frame.copy()
        labels = sorted(work["stage_label"].astype(str).unique(), key=lambda v: _method_display_order().get(v, 99))
        lambdas = sorted(work["lambda_R"].astype(float).unique())
        fig, ax = plt.subplots(figsize=(max(8.0, len(lambdas) * 1.45), max(3.5, len(labels) * 0.75)))
        ax.axis("off")
        cell_text = []
        colors = []
        for label in labels:
            row_text = []
            row_colors = []
            for lam in lambdas:
                cell = work[work["stage_label"].astype(str).eq(label) & np.isclose(work["lambda_R"].astype(float), lam)]
                if cell.empty:
                    row_text.append("")
                    row_colors.append("#f2f2f2")
                    continue
                item = cell.iloc[0]
                recall = float(item.get("target_recall", 0.0))
                precision = float(item.get("target_precision", 0.0))
                selected = str(item.get("observed_shutoff_line_ids", ""))
                row_text.append(f"{selected or '-'}\nR={recall:.2f} P={precision:.2f}")
                row_colors.append("#b7e4c7" if recall >= 0.999 else "#fff3b0" if recall > 0.0 else "#f8c7c7")
            cell_text.append(row_text)
            colors.append(row_colors)
        table = ax.table(cellText=cell_text, rowLabels=labels, colLabels=[f"{v:.1f}" for v in lambdas], cellColours=colors, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.5)
        ax.set_title(f"Expected vs selected shutoffs: {scenario_id}, rho={float(rho):g}")
        fig.tight_layout()
        path = out_dir / "per_rho" / f"rho{float(rho):g}" / str(scenario_id) / "expected_vs_selected_shutoff_lines.png"
        _savefig(fig, path, dpi=180)
        plt.close(fig)


def _plot_solver_diagnostics(best: pd.DataFrame, out_dir: Path) -> None:
    frame = best[best["stage"].astype(str).eq(STAGE_I_B)].copy()
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for scenario_id, local in frame.groupby("scenario_id", sort=True):
        axes[0].plot(local["lambda_R"], local["mip_gap"], marker="o", label=str(scenario_id))
        axes[1].plot(local["lambda_R"], local["runtime_seconds"], marker="o", label=str(scenario_id))
    axes[0].set_xlabel("lambda_R")
    axes[0].set_ylabel("MIP gap")
    axes[1].set_xlabel("lambda_R")
    axes[1].set_ylabel("Runtime seconds")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "stage_i_b_solver_diagnostics.png"
    _savefig(fig, path, dpi=180)
    plt.close(fig)


def _run_dc_side(
    context: dict,
    scenario_ids: Sequence[str],
    lambda_values: Sequence[float],
    rho_values: Sequence[float],
    topology_budget: int,
    proxy_lambda_values: Sequence[float] | None = None,
):
    scenario = context["scenario"]
    network = build_dc_network(scenario)
    topology_pool, p_env_table, ranking, scenarios, metadata = build_stage_ia_topology_pool(
        context,
        scenario_ids,
        lambda_values,
        topology_budget,
        proxy_lambda_values=proxy_lambda_values,
    )
    ia_results = evaluate_stage_ia_pool(scenario, network, topology_pool, rho_values)
    ib_results = evaluate_stage_ib_grid(scenario, network, topology_pool, rho_values)
    heuristic_pool, ah_audit = build_budgeted_heuristic_pool(context, scenario_ids, lambda_values, proxy_lambda_values=proxy_lambda_values)
    heuristic_rows = []
    for idx, row in enumerate(heuristic_pool.itertuples(index=False)):
        p_env = {int(k): float(v) for k, v in json.loads(row.p_env_json).items()}
        recourse = solve_stage_ia_dc_recourse(scenario, network, p_env, float(row.baseline_R), _parse_line_key(row.shutoff_line_ids), float(row.lambda_R), result_id=100000 + idx)
        for rho in rho_values:
            heuristic_rows.append(
                {
                    **row._asdict(),
                    **recourse,
                    "rho_phys": float(rho),
                    "model_type": "dc_heuristic",
                    "candidate_set": "t0p30",
                    "line_id_key": row.shutoff_line_ids,
                    "num_shutoff_lines": len(_parse_line_key(row.shutoff_line_ids)),
                    "post_topology_evaluation_source": "fixed_topology_dc_recourse_for_budgeted_heuristic",
                }
            )
    heuristic_results = pd.DataFrame(heuristic_rows)
    return network, topology_pool, p_env_table, ranking, scenarios, metadata, ia_results, ib_results, heuristic_pool, ah_audit, heuristic_results


def run_stage_h_dc_comparison(
    *,
    scenario_ids: Sequence[str] = DEFAULT_SCENARIOS,
    lambda_values: Sequence[float] = DEFAULT_LAMBDAS,
    proxy_lambda_values: Sequence[float] | None = None,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    topology_budget: int = 100,
    call_budget: int = 100,
    smoke: bool = False,
    smoke_lambda_values: Sequence[float] | None = None,
    skip_gridfm: bool = False,
    stage_e_reference_run: Path | None = None,
    output_root: Path = MAIN_RESULTS,
) -> Path:
    if smoke:
        scenario_ids = list(scenario_ids)[:1]
        if smoke_lambda_values is not None:
            lambda_values = [float(v) for v in smoke_lambda_values]
        elif [float(v) for v in lambda_values] == [float(v) for v in DEFAULT_LAMBDAS]:
            lambda_values = [0.5]
        rho_values = [0.0, 2.0]
        topology_budget = min(int(topology_budget), 3)
        call_budget = min(int(call_budget), 5)
    run_dir = _make_short_run_dir(Path(output_root))
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    _mkdir(inputs_dir)
    context = _make_continuous_context("gnn", 5.0)
    write_json(
        inputs_dir / "metadata.json",
        {
            **git_metadata(),
            "study": "stage_h_dc_approximation_baseline_heuristic_comparison",
            "scenario_ids": list(scenario_ids),
            "lambda_values": [float(v) for v in lambda_values],
            "proxy_lambda_values": None if proxy_lambda_values is None else [float(v) for v in proxy_lambda_values],
            "rho_phys_values": [float(v) for v in rho_values],
            "topology_budget": int(topology_budget),
            "call_budget": int(call_budget),
            "smoke": bool(smoke),
            "result_folder_label": "DC Approximation + Baseline Heuristic Comparison",
        },
    )

    stage_e_best = pd.DataFrame()
    stage_e_run = None
    if stage_e_reference_run is not None:
        stage_e_run = Path(stage_e_reference_run)
        stage_e_best = _load_stage_e_best(stage_e_run, scenario_ids, lambda_values, rho_values, proxy_lambda_values=proxy_lambda_values)
        write_json(inputs_dir / "stage_e_reference_run.json", {"path": str(stage_e_run), "source": "provided"})
    elif not skip_gridfm:
        stage_e_root = REPO_ROOT / "tmp" / "stage_h_dc_stage_e_reference" / ("smoke" if smoke else "full")
        stage_e_run = run_revised_continuous_implementation(
            models=["gnn"],
            scenario_ids=list(scenario_ids),
            lambda_values=[float(v) for v in lambda_values],
            proxy_lambda_values=None if proxy_lambda_values is None else [float(v) for v in proxy_lambda_values],
            rho_values=[float(v) for v in rho_values],
            stages=[STAGE_E_K2],
            stage_e_budget=int(topology_budget),
            call_budget=int(call_budget),
            output_root=stage_e_root,
        )
        stage_e_best = _load_stage_e_best(stage_e_run, scenario_ids, lambda_values, rho_values, proxy_lambda_values=proxy_lambda_values)
        write_json(inputs_dir / "stage_e_reference_run.json", {"path": str(stage_e_run)})

    network, topology_pool, p_env_table, ranking, scenarios, metadata, ia_results, ib_results, heuristic_pool, ah_audit, heuristic_results = _run_dc_side(
        context,
        scenario_ids,
        lambda_values,
        rho_values,
        int(topology_budget),
        proxy_lambda_values,
    )
    dc_results = pd.concat([ia_results, ib_results, heuristic_results], ignore_index=True, sort=False)
    all_best = pd.concat([stage_e_best, _best_by_method(dc_results)], ignore_index=True, sort=False)
    expected = _expected_vs_observed_all_lambdas(all_best, scenarios)
    projections = project_finalists_with_cache(all_best, context, network, cache_path=tables_dir / "ac_projection_cache.csv")

    checks = methodology_fidelity_checks(all_best, p_env_table, ranking, projections, network.branch_audit, smoke=smoke)
    write_dataframe(tables_dir / "stage_i_a_topology_pool.csv", topology_pool)
    write_dataframe(tables_dir / "heuristic_topology_pool.csv", heuristic_pool)
    write_dataframe(tables_dir / "ah_k2_audit.csv", ah_audit)
    write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
    write_dataframe(tables_dir / "scenario_baseline_loading_ranking.csv", ranking)
    write_dataframe(tables_dir / "dc_recourse_results.csv", dc_results)
    write_dataframe(tables_dir / "best_by_rho_scenario_lambda_stage.csv", all_best)
    write_dataframe(tables_dir / "expected_vs_selected_by_rho.csv", expected)
    write_dataframe(tables_dir / "ac_projection_distances.csv", projections)
    write_dataframe(tables_dir / "methodology_fidelity_checks.csv", checks)
    write_json(tables_dir / "dc_branch_model_audit.json", network.branch_audit)

    _plot_pareto(all_best, plots_dir)
    _plot_metric_lines(all_best, projections, plots_dir)
    _plot_expected_table(expected, plots_dir)
    _plot_solver_diagnostics(all_best, plots_dir)
    write_json(
        run_dir / "finalization_summary.json",
        {
            "status": "complete" if bool(checks["passed"].all()) else "complete_with_methodology_warnings",
            "num_best_rows": int(len(all_best)),
            "num_projection_rows": int(len(projections)),
            "num_methodology_checks": int(len(checks)),
            "num_failed_checks": int((~checks["passed"].astype(bool)).sum()),
            "stage_e_reference_run": None if stage_e_run is None else str(stage_e_run),
        },
    )
    return run_dir


def methodology_fidelity_checks(
    best: pd.DataFrame,
    p_env_table: pd.DataFrame,
    ranking: pd.DataFrame,
    projections: pd.DataFrame,
    branch_audit: dict,
    *,
    smoke: bool,
) -> pd.DataFrame:
    rows = []

    def add(name: str, passed: bool, severity: str, detail: str = ""):
        rows.append({"check_name": name, "passed": bool(passed), "severity": severity, "detail": detail})

    add("all_shutoffs_leq_2", bool((best["num_shutoff_lines"].fillna(0).astype(int) <= 2).all()), "hard")
    add("stage_d_absent", not best["stage"].astype(str).str.contains("stage_d", case=False).any(), "hard")
    add("stage_e_unconstrained_absent", not best["stage"].astype(str).str.contains("unconstrained", case=False).any(), "hard")
    add("stage_i_a_present", STAGE_I_A in set(best["stage"].astype(str)), "hard")
    add("stage_i_b_present", STAGE_I_B in set(best["stage"].astype(str)), "hard")
    add("branch_tap_shift_audited", "dc_branch_model_used" in branch_audit, "hard", str(branch_audit))
    dc = best[best["stage"].astype(str).isin([STAGE_I_A, STAGE_I_B])]
    residual_tol = 5e-4
    max_balance = float(dc["max_abs_nodal_balance_residual"].max()) if not dc.empty else 0.0
    max_angle_flow = float(dc["max_abs_angle_flow_residual"].max()) if not dc.empty else 0.0
    add(
        "dc_residuals_near_zero",
        bool(dc.empty or ((dc["max_abs_nodal_balance_residual"].fillna(0).astype(float) <= residual_tol) & (dc["max_abs_angle_flow_residual"].fillna(0).astype(float) <= residual_tol)).all()),
        "hard",
        f"tol={residual_tol:g}; max_balance={max_balance:.6g}; max_angle_flow={max_angle_flow:.6g}",
    )
    add("projection_rows_saved", not projections.empty, "hard")
    add(
        "projection_backend_status_visible",
        bool(not projections.empty and projections["projection_status"].astype(str).str.len().gt(0).all()),
        "hard",
    )
    solved_fraction = float(projections["projection_success"].astype(bool).mean()) if not projections.empty and "projection_success" in projections else 0.0
    finite_count = int(projections["D_proj_total"].notna().sum()) if not projections.empty and "D_proj_total" in projections else 0
    add(
        "projection_distances_or_failures_logged",
        bool(
            not projections.empty
            and "D_proj_total" in projections
            and "projection_status" in projections
            and projections["projection_status"].astype(str).str.len().gt(0).all()
        ),
        "hard",
        f"finite_distances={finite_count}/{len(projections)}; solved_fraction={solved_fraction:.3f}",
    )
    add("shared_baseline_tables_saved", bool(not p_env_table.empty and not ranking.empty), "hard")
    add("smoke_mode_recorded" if smoke else "full_mode_recorded", True, "info")
    return pd.DataFrame(rows)


def run_mld_substudy(
    *,
    scenario_ids: Sequence[str] = DEFAULT_SCENARIOS,
    rho_values: Sequence[float] = DEFAULT_RHO_VALUES,
    smoke: bool = False,
    stage_e_reference_run: Path | None = None,
) -> Path:
    lambda_values = [0.0]
    return run_stage_h_dc_comparison(
        scenario_ids=scenario_ids,
        lambda_values=lambda_values,
        proxy_lambda_values=[1.0],
        rho_values=rho_values,
        topology_budget=3 if smoke else 100,
        call_budget=5 if smoke else 100,
        smoke=smoke,
        smoke_lambda_values=[0.0],
        output_root=MLD_RESULTS,
        stage_e_reference_run=stage_e_reference_run,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage H DC approximation + baseline heuristic comparison.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-gridfm", action="store_true")
    parser.add_argument("--scenario-id", action="append", dest="scenario_ids")
    parser.add_argument("--lambda-r", action="append", type=float, dest="lambda_values")
    parser.add_argument("--proxy-lambda-r", action="append", type=float, dest="proxy_lambda_values")
    parser.add_argument("--rho", action="append", type=float, dest="rho_values")
    parser.add_argument("--topology-budget", type=int, default=100)
    parser.add_argument("--call-budget", type=int, default=100)
    parser.add_argument("--mld", action="store_true")
    parser.add_argument("--stage-e-reference-run", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    scenario_ids = args.scenario_ids or DEFAULT_SCENARIOS
    rho_values = args.rho_values or DEFAULT_RHO_VALUES
    if args.mld:
        run_dir = run_mld_substudy(
            scenario_ids=scenario_ids,
            rho_values=rho_values,
            smoke=bool(args.smoke),
            stage_e_reference_run=args.stage_e_reference_run,
        )
    else:
        run_dir = run_stage_h_dc_comparison(
            scenario_ids=scenario_ids,
            lambda_values=args.lambda_values or DEFAULT_LAMBDAS,
            proxy_lambda_values=args.proxy_lambda_values,
            rho_values=rho_values,
            topology_budget=int(args.topology_budget),
            call_budget=int(args.call_budget),
            smoke=bool(args.smoke),
            skip_gridfm=bool(args.skip_gridfm),
            stage_e_reference_run=args.stage_e_reference_run,
        )
    print(run_dir)


if __name__ == "__main__":
    main()
