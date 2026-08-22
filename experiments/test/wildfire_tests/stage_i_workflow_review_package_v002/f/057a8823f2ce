from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.reporting import git_metadata, write_dataframe, write_json
from experiments.test.wildfire_tests.stage_g_implementation_revision.run_stage_g_revised_continuous_implementation import (
    STAGE_E_K2,
    _make_continuous_context,
    _run_continuous_pool,
    _topology_candidates_for_context,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.dc_formulation import (
    STAGE_I_A,
    STAGE_I_A_LABEL,
    build_dc_network,
    build_stage_ia_topology_pool,
    evaluate_stage_ia_pool,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_dc_comparison import (
    DEFAULT_LAMBDAS,
    DEFAULT_SCENARIOS,
    RESULT_ROOT,
    _make_short_run_dir,
)


STUDY_ROOT = RESULT_ROOT / "proxy_inner_lambda_sweep"
STAGE_E_LABEL = "Stage E K2 GridFM"
STAGE_LABELS = {STAGE_E_K2: STAGE_E_LABEL, STAGE_I_A: STAGE_I_A_LABEL}
STAGE_ORDER = [STAGE_E_K2, STAGE_I_A]
COLORS = {STAGE_E_K2: "#4C78A8", STAGE_I_A: "#F58518"}
MARKERS = {STAGE_E_K2: "o", STAGE_I_A: "s"}


def _long_path(path: Path) -> str:
    path = Path(path)
    if os.name == "nt":
        absolute = str(path.absolute())
        return absolute if absolute.startswith("\\\\?\\") else "\\\\?\\" + absolute
    return str(path)


def _mkdir(path: Path) -> None:
    if os.name == "nt":
        Path(_long_path(path)).mkdir(parents=True, exist_ok=True)
    else:
        Path(path).mkdir(parents=True, exist_ok=True)


def _savefig(fig, path: Path, **kwargs) -> None:
    _mkdir(path.parent)
    staging = Path.cwd() / "tmp" / "stage_h_plot_staging"
    _mkdir(staging)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=Path(path).suffix or ".png", dir=str(staging))
    tmp_path = Path(handle.name)
    handle.close()
    try:
        fig.savefig(tmp_path, **kwargs)
        os.replace(str(tmp_path), _long_path(path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _line_key(value) -> str:
    values = _parse_lines(value)
    return ",".join(str(v) for v in values)


def _parse_lines(value) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    return sorted({int(float(part.strip())) for part in text.split(",") if part.strip()})


def _nondominated_mask(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series([], dtype=bool, index=frame.index)
    points = frame[["L_shed", "R_norm"]].astype(float).to_numpy()
    keep = np.ones(len(points), dtype=bool)
    for idx, (x, y) in enumerate(points):
        dominated = (
            (points[:, 0] <= x + 1e-12)
            & (points[:, 1] <= y + 1e-12)
            & ((points[:, 0] < x - 1e-12) | (points[:, 1] < y - 1e-12))
        )
        keep[idx] = not bool(np.any(dominated))
    return pd.Series(keep, index=frame.index)


def _lambda_case(value: float) -> str:
    return f"lambda_{float(value):g}".replace(".", "p")


def _expand_proxy_topology_pool(pool: pd.DataFrame, lambda_values: Sequence[float]) -> pd.DataFrame:
    """Reuse each proxy-generated topology across the requested inner lambdas."""
    if pool.empty:
        return pool.copy()
    rows = []
    for _, row in pool.iterrows():
        proxy_lambda = float(row.get("lambda_R_proxy", row.get("lambda_R", np.nan)))
        for lambda_r in lambda_values:
            expanded = row.copy()
            expanded["lambda_R"] = float(lambda_r)
            expanded["lambda_L"] = 1.0 - float(lambda_r)
            expanded["lambda_case"] = _lambda_case(float(lambda_r))
            expanded["lambda_R_proxy"] = proxy_lambda
            expanded["lambda_L_proxy"] = 1.0 - proxy_lambda
            expanded["lambda_proxy_case"] = _lambda_case(proxy_lambda)
            rows.append(expanded)
    return pd.DataFrame(rows).reset_index(drop=True)


def _normalize_rows(frame: pd.DataFrame, stage: str | None = None) -> pd.DataFrame:
    out = frame.copy()
    if stage is not None:
        out["stage"] = stage
    out["stage_label"] = out["stage"].map(STAGE_LABELS).fillna(out.get("stage_label", out["stage"]))
    if "line_id_key" not in out:
        out["line_id_key"] = out.get("shutoff_line_ids", "").map(_line_key)
    else:
        out["line_id_key"] = out["line_id_key"].fillna(out.get("shutoff_line_ids", "")).map(_line_key)
    if "num_shutoff_lines" not in out:
        out["num_shutoff_lines"] = out["line_id_key"].map(lambda value: len(_parse_lines(value)))
    if "lambda_R_proxy" not in out:
        out["lambda_R_proxy"] = out["lambda_R"].astype(float)
        out["lambda_L_proxy"] = 1.0 - out["lambda_R_proxy"].astype(float)
    out["J_traditional_no_phys"] = (
        out["lambda_R"].astype(float) * out["R_norm"].astype(float)
        + (1.0 - out["lambda_R"].astype(float)) * out["L_shed"].astype(float)
    )
    out["stage_order"] = out["stage"].map({stage_name: idx for idx, stage_name in enumerate(STAGE_ORDER)}).fillna(99).astype(int)
    return out


def _best_by_proxy_inner(all_eval: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario_id", "scenario_name", "rho_phys", "lambda_R_proxy", "lambda_R", "stage", "stage_label"]
    ok = all_eval[np.isfinite(all_eval["J_traditional_no_phys"].astype(float))].copy()
    return (
        ok.sort_values(keys + ["J_traditional_no_phys", "L_shed", "R_norm", "topology_iteration"], kind="mergesort")
        .groupby(keys, as_index=False, dropna=False)
        .first()
        .reset_index(drop=True)
    )


def _plot_proxy_folder(local: pd.DataFrame, best: pd.DataFrame, out: Path, scenario_id: str, proxy_lambda: float, rho: float) -> None:
    _mkdir(out)
    lambdas = sorted(float(v) for v in local["lambda_R"].dropna().unique())

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    for stage in STAGE_ORDER:
        frame = local[local["stage"].astype(str).eq(stage)].copy()
        if frame.empty:
            continue
        ax.scatter(
            frame["L_shed"],
            frame["R_norm"],
            s=16 if len(frame) > 100 else 42,
            alpha=0.24 if len(frame) > 100 else 0.72,
            color=COLORS[stage],
            marker=MARKERS[stage],
            label=f"{STAGE_LABELS[stage]} evaluated",
        )
        unique = frame.drop_duplicates(["L_shed", "R_norm", "line_id_key"]).copy()
        front = unique[_nondominated_mask(unique)].sort_values(["L_shed", "R_norm"], kind="mergesort")
        if not front.empty:
            ax.plot(front["L_shed"], front["R_norm"], color=COLORS[stage], linewidth=2.0)
            ax.scatter(front["L_shed"], front["R_norm"], s=58, color=COLORS[stage], marker=MARKERS[stage], edgecolor="#202020", linewidth=0.6)
    values = local["R_norm"].astype(float)
    min_pos = values[values > 0].min() if (values > 0).any() else 1.0
    if len(values) and float(values.max()) / max(float(min_pos), 1e-9) > 100:
        ax.set_yscale("log")
        ylabel = "R_norm (log scale)"
    else:
        ylabel = "R_norm"
    ax.set_xlabel("L_shed")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{scenario_id}: all evaluated points, proxy lambda={proxy_lambda:g}, rho={rho:g}")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, out / "pareto_frontier_scatter.png", dpi=190)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(lambdas), figsize=(max(15.0, 4.0 * len(lambdas)), 5.6), squeeze=False)
    for ax, lambda_r in zip(axes[0], lambdas):
        lam_frame = local[np.isclose(local["lambda_R"].astype(float), lambda_r)].copy()
        max_iter = max(int(lam_frame["topology_iteration"].max()), 1) if len(lam_frame) else 1
        y_values: list[float] = []
        for stage in STAGE_ORDER:
            frame = lam_frame[lam_frame["stage"].astype(str).eq(stage)].copy()
            if frame.empty:
                continue
            frame = frame.sort_values(["topology_iteration", "J_traditional_no_phys", "line_id_key"], kind="mergesort")
            iter_best = frame.groupby("topology_iteration", as_index=False, dropna=False).first()
            iter_best["best_so_far"] = iter_best["J_traditional_no_phys"].astype(float).cummin()
            y_values.extend(iter_best["best_so_far"].astype(float).tolist())
            ax.step(iter_best["topology_iteration"], iter_best["best_so_far"], where="post", color=COLORS[stage], linewidth=2.0, label=STAGE_LABELS[stage])
            ax.scatter(iter_best["topology_iteration"].iloc[-1], iter_best["best_so_far"].iloc[-1], color=COLORS[stage], s=30, edgecolor="#202020", linewidth=0.4)
        finite = [value for value in y_values if np.isfinite(value)]
        if finite and max(finite) > 10 * max(min([value for value in finite if value > 0] or [1e-3]), 1e-3):
            ax.set_yscale("symlog", linthresh=1e-3)
            ylabel = "Best-so-far J (symlog)"
        else:
            ylabel = "Best-so-far J"
        ax.set_xlim(left=0, right=max_iter * 1.03)
        ax.set_title(f"inner lambda={lambda_r:g}")
        ax.set_xlabel("Topology iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    handles, labels = axes[0][-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.suptitle(
        f"{scenario_id}: traditional objective convergence, proxy lambda={proxy_lambda:g}, rho={rho:g}\n"
        "J = lambda_R R_norm + (1 - lambda_R) L_shed; physics penalty excluded",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.90])
    _savefig(fig, out / "traditional_lambda_objective_convergence.png", dpi=190)
    plt.close(fig)

    summary = best[np.isclose(best["lambda_R_proxy"].astype(float), proxy_lambda)].copy()
    if not summary.empty:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
        metrics = [("J_traditional_no_phys", "Best J"), ("L_shed", "L_shed"), ("R_norm", "R_norm")]
        for ax, (metric, title) in zip(axes, metrics):
            for stage in STAGE_ORDER:
                frame = summary[summary["stage"].astype(str).eq(stage)].sort_values("lambda_R")
                if frame.empty:
                    continue
                ax.plot(frame["lambda_R"], frame[metric], marker=MARKERS[stage], color=COLORS[stage], linewidth=2, label=STAGE_LABELS[stage])
            ax.set_title(title)
            ax.set_xlabel("inner lambda_R")
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("value")
        axes[-1].legend(fontsize=7)
        fig.suptitle(f"{scenario_id}: best values across inner lambda, proxy lambda={proxy_lambda:g}")
        _savefig(fig, out / "best_metrics_by_inner_lambda.png", dpi=190)
        plt.close(fig)


def _plot_scenario_heatmaps(best: pd.DataFrame, out: Path, scenario_id: str, rho: float) -> None:
    scenario = best[best["scenario_id"].astype(str).eq(str(scenario_id)) & np.isclose(best["rho_phys"].astype(float), float(rho))].copy()
    if scenario.empty:
        return
    proxies = sorted(float(v) for v in scenario["lambda_R_proxy"].dropna().unique())
    inners = sorted(float(v) for v in scenario["lambda_R"].dropna().unique())
    for metric, title in [("J_traditional_no_phys", "best_J_no_physics"), ("L_shed", "best_L_shed"), ("R_norm", "best_R_norm")]:
        fig, axes = plt.subplots(1, len(STAGE_ORDER), figsize=(6.2 * len(STAGE_ORDER), 5.2), squeeze=False)
        for ax, stage in zip(axes[0], STAGE_ORDER):
            pivot = (
                scenario[scenario["stage"].astype(str).eq(stage)]
                .pivot_table(index="lambda_R_proxy", columns="lambda_R", values=metric, aggfunc="min")
                .reindex(index=proxies, columns=inners)
            )
            data = pivot.to_numpy(dtype=float)
            image = ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
            ax.set_xticks(range(len(inners)), [f"{v:g}" for v in inners])
            ax.set_yticks(range(len(proxies)), [f"{v:g}" for v in proxies])
            ax.set_xlabel("inner lambda_R")
            ax.set_ylabel("proxy lambda_R")
            ax.set_title(STAGE_LABELS[stage])
            for row_idx in range(data.shape[0]):
                for col_idx in range(data.shape[1]):
                    value = data[row_idx, col_idx]
                    if np.isfinite(value):
                        ax.text(col_idx, row_idx, f"{value:.3g}", ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(f"{scenario_id}: {title}, rho={rho:g}")
        fig.tight_layout()
        _savefig(fig, out / f"{title}_heatmap.png", dpi=190)
        plt.close(fig)


def _write_checks(run_dir: Path, all_eval: pd.DataFrame, best: pd.DataFrame, expected_rows: int, plot_count: int) -> pd.DataFrame:
    checks = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check_name": name, "passed": bool(passed), "severity": "hard", "detail": detail})

    add("only_stage_e_and_stage_i_a", set(all_eval["stage"].astype(str).unique()).issubset(set(STAGE_ORDER)), ",".join(sorted(all_eval["stage"].astype(str).unique())))
    add("all_proxy_inner_combinations_present", len(best[["scenario_id", "lambda_R_proxy", "lambda_R", "stage"]].drop_duplicates()) == expected_rows, f"observed={len(best[['scenario_id','lambda_R_proxy','lambda_R','stage']].drop_duplicates())}; expected={expected_rows}")
    add("rho_fixed", sorted(float(v) for v in all_eval["rho_phys"].dropna().unique()) == [0.0], str(sorted(float(v) for v in all_eval["rho_phys"].dropna().unique())))
    add("proxy_lambda_column_saved", "lambda_R_proxy" in all_eval.columns and "lambda_R_proxy" in best.columns)
    add("traditional_objective_saved", "J_traditional_no_phys" in all_eval.columns and "J_traditional_no_phys" in best.columns)
    add("plots_generated", plot_count > 0, f"plot_count={plot_count}")
    frame = pd.DataFrame(checks)
    write_dataframe(run_dir / "tables" / "methodology_fidelity_checks.csv", frame)
    return frame


def run_proxy_inner_lambda_sweep(
    *,
    scenario_ids: Sequence[str] = DEFAULT_SCENARIOS,
    lambda_values: Sequence[float] = DEFAULT_LAMBDAS,
    proxy_lambda_values: Sequence[float] = DEFAULT_LAMBDAS,
    rho_value: float = 0.0,
    topology_budget: int = 100,
    call_budget: int = 100,
    output_root: Path = STUDY_ROOT,
    resume_run_dir: Path | None = None,
) -> Path:
    started = time.perf_counter()
    run_dir = Path(resume_run_dir) if resume_run_dir is not None else _make_short_run_dir(Path(output_root))
    _mkdir(run_dir)
    tables_dir = run_dir / "tables"
    plots_dir = run_dir / "plots"
    inputs_dir = run_dir / "inputs"
    _mkdir(inputs_dir)
    lambda_values = [float(v) for v in lambda_values]
    proxy_lambda_values = [float(v) for v in proxy_lambda_values]
    scenario_ids = [str(v) for v in scenario_ids]
    rho_values = [float(rho_value)]
    write_json(
        inputs_dir / "metadata.json",
        {
            **git_metadata(),
            "study": "stage_h_proxy_inner_lambda_sweep",
            "scenario_ids": scenario_ids,
            "lambda_values": lambda_values,
            "proxy_lambda_values": proxy_lambda_values,
            "rho_values": rho_values,
            "stages": STAGE_ORDER,
            "topology_budget": int(topology_budget),
            "call_budget": int(call_budget),
            "objective_for_comparison": "lambda_R * R_norm + (1 - lambda_R) * L_shed",
            "physics_penalty_excluded_from_traditional_objective": True,
        },
    )

    context = _make_continuous_context("gnn", 5.0)
    stage_e_proxy_pool, stage_e_p_env, stage_e_ranking, stage_e_scenarios, stage_e_metadata = _topology_candidates_for_context(
        context,
        list(scenario_ids),
        proxy_lambda_values,
        None,
        [STAGE_E_K2],
        None,
        int(topology_budget),
    )
    stage_e_topology_pool = _expand_proxy_topology_pool(stage_e_proxy_pool, lambda_values)
    checkpoint_root = Path("tmp") / "stage_h_proxy_inner_checkpoints" / run_dir.name
    _mkdir(checkpoint_root)
    stage_e, stage_e_traces, stage_e_load, stage_e_risk, stage_e_controlled, stage_e_mask = _run_continuous_pool(
        context,
        stage_e_topology_pool,
        list(stage_e_metadata["candidate_line_ids"]),
        rho_values,
        int(call_budget),
        checkpoint_dir=checkpoint_root / "stage_e_checkpoints",
        progress_path=None,
    )
    write_json(
        inputs_dir / "stage_e_generation.json",
        {
            "topology_generation_axis": "lambda_R_proxy",
            "continuous_recourse_axis": "lambda_R",
            "candidate_line_ids": [int(v) for v in stage_e_metadata["candidate_line_ids"]],
            "num_proxy_topology_rows": int(len(stage_e_proxy_pool)),
            "num_expanded_recourse_rows": int(len(stage_e_topology_pool)),
            "checkpoint_root": str(checkpoint_root),
        },
    )

    network = build_dc_network(context["scenario"])
    topology_pool_proxy, p_env_table, ranking, scenarios, metadata = build_stage_ia_topology_pool(
        context,
        scenario_ids,
        proxy_lambda_values,
        int(topology_budget),
        proxy_lambda_values=None,
    )
    topology_pool = _expand_proxy_topology_pool(topology_pool_proxy, lambda_values)
    stage_i_a = evaluate_stage_ia_pool(context["scenario"], network, topology_pool, rho_values)
    stage_i_a = stage_i_a[stage_i_a["stage"].astype(str).eq(STAGE_I_A)].copy()

    stage_e = _normalize_rows(stage_e, stage=STAGE_E_K2)
    stage_i_a = _normalize_rows(stage_i_a, stage=STAGE_I_A)
    common = sorted(set(stage_e.columns).union(stage_i_a.columns))
    all_eval = pd.concat([stage_e.reindex(columns=common), stage_i_a.reindex(columns=common)], ignore_index=True, sort=False)
    all_eval = all_eval[np.isfinite(all_eval["R_norm"].astype(float)) & np.isfinite(all_eval["L_shed"].astype(float))].copy()
    all_eval = all_eval.sort_values(["scenario_id", "lambda_R_proxy", "lambda_R", "stage_order", "topology_iteration"], kind="mergesort").reset_index(drop=True)
    best = _best_by_proxy_inner(all_eval)

    write_dataframe(tables_dir / "stage_e_proxy_topology_pool.csv", stage_e_proxy_pool)
    write_dataframe(tables_dir / "stage_e_topology_pool.csv", stage_e_topology_pool)
    write_dataframe(tables_dir / "stage_e_continuous_recourse_results.csv", stage_e)
    write_dataframe(tables_dir / "stage_e_continuous_objective_call_trace.csv", stage_e_traces)
    write_dataframe(tables_dir / "stage_e_load_shedding_provenance.csv", stage_e_load)
    write_dataframe(tables_dir / "stage_e_wildfire_risk_provenance.csv", stage_e_risk)
    write_dataframe(tables_dir / "stage_e_controlled_state_consistency.csv", stage_e_controlled)
    write_dataframe(tables_dir / "stage_e_masking_clamping_audit.csv", stage_e_mask)
    write_dataframe(tables_dir / "stage_i_a_proxy_topology_pool.csv", topology_pool_proxy)
    write_dataframe(tables_dir / "stage_i_a_topology_pool.csv", topology_pool)
    write_dataframe(tables_dir / "stage_i_a_dc_recourse_results.csv", stage_i_a)
    write_dataframe(tables_dir / "all_evaluated_proxy_inner_points.csv", all_eval)
    write_dataframe(tables_dir / "best_by_scenario_proxy_inner_stage.csv", best)
    write_dataframe(tables_dir / "p_env_by_scenario.csv", p_env_table)
    write_dataframe(tables_dir / "scenario_baseline_loading_ranking.csv", ranking)
    write_json(tables_dir / "dc_branch_model_audit.json", network.branch_audit)

    frontier_rows = []
    for (scenario_id, proxy_lambda, rho), local in all_eval.groupby(["scenario_id", "lambda_R_proxy", "rho_phys"], sort=True):
        proxy_folder = plots_dir / "by_scenario" / str(scenario_id) / f"proxy_lambda_{float(proxy_lambda):g}"
        local_best = best[
            best["scenario_id"].astype(str).eq(str(scenario_id))
            & np.isclose(best["lambda_R_proxy"].astype(float), float(proxy_lambda))
            & np.isclose(best["rho_phys"].astype(float), float(rho))
        ].copy()
        _plot_proxy_folder(local, local_best, proxy_folder, str(scenario_id), float(proxy_lambda), float(rho))
        for stage in STAGE_ORDER:
            frame = local[local["stage"].astype(str).eq(stage)].drop_duplicates(["L_shed", "R_norm", "line_id_key"]).copy()
            front = frame[_nondominated_mask(frame)].copy()
            for _, row in front.iterrows():
                frontier_rows.append(
                    {
                        "scenario_id": scenario_id,
                        "rho_phys": float(rho),
                        "lambda_R_proxy": float(proxy_lambda),
                        "lambda_R": float(row["lambda_R"]),
                        "stage": stage,
                        "stage_label": STAGE_LABELS[stage],
                        "topology_iteration": int(row["topology_iteration"]),
                        "line_id_key": row["line_id_key"],
                        "L_shed": float(row["L_shed"]),
                        "R_norm": float(row["R_norm"]),
                        "J_traditional_no_phys": float(row["J_traditional_no_phys"]),
                    }
                )
    for (scenario_id, rho), _ in all_eval.groupby(["scenario_id", "rho_phys"], sort=True):
        _plot_scenario_heatmaps(best, plots_dir / "by_scenario" / str(scenario_id), str(scenario_id), float(rho))

    frontier = pd.DataFrame(frontier_rows)
    write_dataframe(tables_dir / "pareto_frontier_points_by_proxy.csv", frontier)
    plot_count = sum(1 for path in Path(_long_path(plots_dir)).rglob("*.png"))
    expected_rows = len(scenario_ids) * len(proxy_lambda_values) * len(lambda_values) * len(STAGE_ORDER)
    checks = _write_checks(run_dir, all_eval, best, expected_rows, int(plot_count))
    write_json(
        run_dir / "finalization_summary.json",
        {
            "status": "complete" if bool(checks["passed"].all()) else "complete_with_methodology_warnings",
            "num_all_evaluated_rows": int(len(all_eval)),
            "num_best_rows": int(len(best)),
            "num_frontier_rows": int(len(frontier)),
            "num_methodology_checks": int(len(checks)),
            "num_failed_checks": int((~checks["passed"].astype(bool)).sum()),
            "num_plot_files": int(plot_count),
            "stage_e_topology_generation": "proxy_topology_pool_expanded_across_inner_lambda",
            "runtime_seconds": float(time.perf_counter() - started),
        },
    )
    return run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage H proxy-lambda versus inner-lambda sweep.")
    parser.add_argument("--scenario-id", action="append", dest="scenario_ids")
    parser.add_argument("--lambda-r", action="append", type=float, dest="lambda_values")
    parser.add_argument("--proxy-lambda-r", action="append", type=float, dest="proxy_lambda_values")
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--topology-budget", type=int, default=100)
    parser.add_argument("--call-budget", type=int, default=100)
    parser.add_argument("--output-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--resume-run-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_dir = run_proxy_inner_lambda_sweep(
        scenario_ids=args.scenario_ids or DEFAULT_SCENARIOS,
        lambda_values=args.lambda_values or DEFAULT_LAMBDAS,
        proxy_lambda_values=args.proxy_lambda_values or DEFAULT_LAMBDAS,
        rho_value=float(args.rho),
        topology_budget=int(args.topology_budget),
        call_budget=int(args.call_budget),
        output_root=args.output_root,
        resume_run_dir=args.resume_run_dir,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
