from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_miqp_pool_refresh import (  # noqa: E402
    _long_path,
    _mkdir,
    _nondominated_mask,
    _savefig,
)
from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_dc_comparison import (  # noqa: E402
    RESULT_ROOT,
)


PROXY_RUN = RESULT_ROOT / "proxy_inner_lambda_sweep" / "r2"
MAIN_RUN = RESULT_ROOT / "main_results" / "r11"
STAGE = "stage_i_a_dc_k2"
OUT_SUBDIR = "stage_i_a_main_vs_best_proxy"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(_long_path(path), low_memory=False)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _lambda_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _best_proxy_by_scenario(proxy_best: pd.DataFrame) -> pd.DataFrame:
    work = proxy_best[
        proxy_best["stage"].astype(str).eq(STAGE)
        & np.isclose(proxy_best["rho_phys"].astype(float), 0.0)
        & (proxy_best["lambda_R"].astype(float) > 0.0)
    ].copy()
    rows = []
    for scenario_id, local in work.groupby("scenario_id", sort=True):
        summary = (
            local.groupby("lambda_R_proxy", as_index=False)
            .agg(
                mean_nonzero_J=("J_traditional_no_phys", "mean"),
                mean_nonzero_R=("R_norm", "mean"),
                mean_nonzero_L=("L_shed", "mean"),
                mean_nonzero_common_op=("PAC_common_op_overlap", "mean"),
            )
            .sort_values(["mean_nonzero_J", "mean_nonzero_R", "mean_nonzero_L"], kind="mergesort")
        )
        best = summary.iloc[0]
        rows.append(
            {
                "scenario_id": scenario_id,
                "best_lambda_R_proxy": float(best["lambda_R_proxy"]),
                "selection_rule": "min mean J over lambda_R_inner > 0; tie by mean R_norm then mean L_shed",
                "mean_nonzero_J": float(best["mean_nonzero_J"]),
                "mean_nonzero_R": float(best["mean_nonzero_R"]),
                "mean_nonzero_L": float(best["mean_nonzero_L"]),
                "mean_nonzero_common_op": float(best["mean_nonzero_common_op"]),
            }
        )
    return pd.DataFrame(rows)


def _best_rows_by_lambda(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lambda_r, local in frame.groupby("lambda_R", sort=True):
        local = local.dropna(subset=["J_traditional_no_phys", "R_norm", "L_shed"]).copy()
        if local.empty:
            continue
        best = local.sort_values(["J_traditional_no_phys", "R_norm", "L_shed"], kind="mergesort").iloc[0]
        rows.append(best)
    return pd.DataFrame(rows)


def _plot_pareto(
    scenario_id: str,
    proxy_lambda: float,
    main_local: pd.DataFrame,
    proxy_local: pd.DataFrame,
    out: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.8, 6.6), constrained_layout=True)
    specs = [
        ("Main coupled", main_local, "#F58518", "s"),
        (f"Proxy sweep best setting (lambda_R_proxy={proxy_lambda:g})", proxy_local, "#54A24B", "D"),
    ]
    for label, frame, color, marker in specs:
        frame = frame.dropna(subset=["L_shed", "R_norm"]).copy()
        if frame.empty:
            continue
        ax.scatter(
            frame["L_shed"],
            frame["R_norm"],
            s=22,
            marker=marker,
            color=color,
            edgecolor=color,
            alpha=0.18,
            linewidth=0.35,
            label=f"{label} evaluated",
        )
        unique = frame.drop_duplicates(["L_shed", "R_norm", "line_id_key"]).copy()
        nd = unique[_nondominated_mask(unique)].sort_values(["L_shed", "R_norm"], kind="mergesort")
        if not nd.empty:
            ax.plot(nd["L_shed"], nd["R_norm"], color=color, linewidth=2.2, alpha=0.95, label=f"{label} nondominated")
            ax.scatter(nd["L_shed"], nd["R_norm"], s=88, marker=marker, color=color, edgecolor="#202020", linewidth=0.8)
        selected = _best_rows_by_lambda(frame)
        for _, row in selected.iterrows():
            ax.annotate(
                f"{float(row['lambda_R']):g}",
                (float(row["L_shed"]), float(row["R_norm"])),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
                color=color,
            )
    ax.set_title(f"{scenario_id}: Stage I-a Pareto Comparison\nmain coupled vs best proxy setting")
    ax.set_xlabel("L_shed")
    ax.set_ylabel("R_norm")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    path = out / "stage_i_a_pareto_main_vs_best_proxy.png"
    _savefig(fig, path, dpi=190)
    plt.close(fig)
    return path


def _best_so_far(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.dropna(subset=["topology_iteration", "J_traditional_no_phys"]).copy()
    if work.empty:
        return work
    work = work.sort_values(["topology_iteration", "J_traditional_no_phys", "line_id_key"], kind="mergesort")
    best = np.inf
    rows = []
    for iteration, local in work.groupby("topology_iteration", sort=True):
        row = local.iloc[0]
        best = min(best, float(row["J_traditional_no_phys"]))
        rows.append(
            {
                "topology_iteration": int(iteration),
                "J_traditional_no_phys": float(row["J_traditional_no_phys"]),
                "best_so_far": float(best),
                "line_id_key": row.get("line_id_key", ""),
            }
        )
    return pd.DataFrame(rows)


def _plot_traditional_objective(
    scenario_id: str,
    proxy_lambda: float,
    main_local: pd.DataFrame,
    proxy_local: pd.DataFrame,
    out: Path,
) -> Path:
    lambdas = sorted(float(v) for v in sorted(set(main_local["lambda_R"].dropna()).union(set(proxy_local["lambda_R"].dropna()))))
    fig, axes = plt.subplots(1, len(lambdas), figsize=(max(4.0 * len(lambdas), 8.0), 5.4), sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    y_values = []
    for ax, lambda_r in zip(axes, lambdas):
        main_frame = main_local[np.isclose(main_local["lambda_R"].astype(float), lambda_r)]
        proxy_frame = proxy_local[np.isclose(proxy_local["lambda_R"].astype(float), lambda_r)]
        for label, frame, color, marker in [
            ("Main coupled", main_frame, "#F58518", "s"),
            (f"Best proxy {proxy_lambda:g}", proxy_frame, "#54A24B", "D"),
        ]:
            trace = _best_so_far(frame)
            if trace.empty:
                continue
            y_values.extend(trace["best_so_far"].tolist())
            ax.step(trace["topology_iteration"], trace["best_so_far"], where="post", color=color, linewidth=2.0, label=label)
            ax.scatter(
                trace["topology_iteration"].iloc[-1],
                trace["best_so_far"].iloc[-1],
                color=color,
                marker=marker,
                edgecolor="#202020",
                s=54,
                zorder=3,
            )
        ax.set_title(f"lambda_R={lambda_r:g}", fontsize=10)
        ax.set_xlabel("Topology iteration")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Best-so-far J")
    if y_values:
        ymin, ymax = min(y_values), max(y_values)
        pad = max((ymax - ymin) * 0.08, 1e-4)
        for ax in axes:
            ax.set_ylim(max(0.0, ymin - pad), ymax + pad)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncols=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"{scenario_id}: Stage I-a Traditional Objective Convergence\n"
        "J = lambda_R R_norm + (1 - lambda_R) L_shed",
        fontsize=12,
    )
    path = out / "stage_i_a_traditional_objective_main_vs_best_proxy.png"
    _savefig(fig, path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def generate() -> pd.DataFrame:
    proxy_tables = PROXY_RUN / "tables"
    main_tables = MAIN_RUN / "tables"
    proxy_best = _numeric(
        _read_csv(proxy_tables / "best_by_scenario_proxy_inner_stage.csv"),
        ["rho_phys", "lambda_R", "lambda_R_proxy", "J_traditional_no_phys", "R_norm", "L_shed", "PAC_common_op_overlap"],
    )
    best_proxy = _best_proxy_by_scenario(proxy_best)
    best_proxy.to_csv(_long_path(proxy_tables / "stage_i_a_best_proxy_setting_for_summary_figures.csv"), index=False)

    main_eval = _numeric(
        _read_csv(main_tables / "all_evaluated_stage_h_points.csv"),
        ["rho_phys", "lambda_R", "topology_iteration", "J_traditional_no_phys", "R_norm", "L_shed"],
    )
    proxy_eval = _numeric(
        _read_csv(proxy_tables / "all_evaluated_proxy_inner_points.csv"),
        ["rho_phys", "lambda_R", "lambda_R_proxy", "topology_iteration", "J_traditional_no_phys", "R_norm", "L_shed"],
    )
    main_eval = main_eval[main_eval["stage"].astype(str).eq(STAGE) & np.isclose(main_eval["rho_phys"].astype(float), 0.0)].copy()
    proxy_eval = proxy_eval[proxy_eval["stage"].astype(str).eq(STAGE) & np.isclose(proxy_eval["rho_phys"].astype(float), 0.0)].copy()

    manifest = []
    for row in best_proxy.itertuples(index=False):
        scenario_id = str(row.scenario_id)
        proxy_lambda = float(row.best_lambda_R_proxy)
        out = PROXY_RUN / "plots" / "by_scenario" / scenario_id / OUT_SUBDIR
        _mkdir(out)
        main_local = main_eval[main_eval["scenario_id"].astype(str).eq(scenario_id)].copy()
        proxy_local = proxy_eval[
            proxy_eval["scenario_id"].astype(str).eq(scenario_id)
            & np.isclose(proxy_eval["lambda_R_proxy"].astype(float), proxy_lambda)
        ].copy()
        pareto_path = _plot_pareto(scenario_id, proxy_lambda, main_local, proxy_local, out)
        objective_path = _plot_traditional_objective(scenario_id, proxy_lambda, main_local, proxy_local, out)
        manifest.append(
            {
                "scenario_id": scenario_id,
                "best_lambda_R_proxy": proxy_lambda,
                "pareto_image": str(pareto_path),
                "traditional_objective_image": str(objective_path),
                "num_main_points": int(len(main_local)),
                "num_proxy_points": int(len(proxy_local)),
            }
        )
    manifest_frame = pd.DataFrame(manifest)
    manifest_frame.to_csv(_long_path(proxy_tables / "stage_i_a_summary_figure_manifest.csv"), index=False)
    return manifest_frame


def main() -> None:
    manifest = generate()
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
