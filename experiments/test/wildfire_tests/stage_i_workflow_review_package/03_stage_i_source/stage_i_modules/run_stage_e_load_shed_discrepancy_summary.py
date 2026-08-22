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

from experiments.test.wildfire_tests.shared.reporting import write_dataframe, write_json
from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_dc_comparison import (
    RESULT_ROOT,
    _long_path,
)


RUN = RESULT_ROOT / "main_results" / "r11"
TABLES = RUN / "tables"
PLOTS = RUN / "plots" / "summary"
STAGE = "stage_e_k2"


ROW_TABLE = "stage_e_k2_load_shed_discrepancy_main_results.csv"
SUMMARY_TABLE = "stage_e_k2_load_shed_discrepancy_summary_by_rho_lambda.csv"
OVERALL_TABLE = "stage_e_k2_load_shed_discrepancy_summary_overall.csv"
METADATA_FILE = "stage_e_k2_load_shed_discrepancy_metadata.json"
LOAD_SHED_METRICS = [
    "L_shed_cmd",
    "L_shed_gridfm_raw",
    "L_shed_gridfm_effective",
    "L_shed_hybrid",
]
LOAD_SHED_LABELS = {
    "L_shed_cmd": "Commanded",
    "L_shed_gridfm_raw": "GridFM raw",
    "L_shed_gridfm_effective": "GridFM effective",
    "L_shed_hybrid": "Hybrid objective",
}
LOAD_SHED_COLORS = {
    "L_shed_cmd": "#111827",
    "L_shed_gridfm_raw": "#2563eb",
    "L_shed_gridfm_effective": "#f97316",
    "L_shed_hybrid": "#16a34a",
}
DIFF_METRICS = [
    "L_shed_gridfm_raw_minus_cmd",
    "L_shed_gridfm_effective_minus_cmd",
    "L_shed_hybrid_minus_cmd",
]
DIFF_LABELS = {
    "L_shed_gridfm_raw_minus_cmd": "Raw - commanded",
    "L_shed_gridfm_effective_minus_cmd": "Effective - commanded",
    "L_shed_hybrid_minus_cmd": "Hybrid - commanded",
}
DIFF_COLORS = {
    "L_shed_gridfm_raw_minus_cmd": "#2563eb",
    "L_shed_gridfm_effective_minus_cmd": "#f97316",
    "L_shed_hybrid_minus_cmd": "#16a34a",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(_long_path(path), low_memory=False)


def _mkdir(path: Path) -> None:
    os.makedirs(_long_path(path), exist_ok=True)


def _savefig(fig: plt.Figure, path: Path) -> None:
    _mkdir(path.parent)
    fig.savefig(_long_path(path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _summarize(grouped: pd.core.groupby.generic.DataFrameGroupBy) -> pd.DataFrame:
    return grouped.agg(
        n=("scenario_id", "count"),
        mean_L_shed_cmd=("L_shed_cmd", "mean"),
        mean_L_shed_gridfm_raw=("L_shed_gridfm_raw", "mean"),
        mean_L_shed_gridfm_effective=("L_shed_gridfm_effective", "mean"),
        mean_L_shed_hybrid=("L_shed_hybrid", "mean"),
        mean_raw_minus_cmd=("L_shed_gridfm_raw_minus_cmd", "mean"),
        mean_effective_minus_cmd=("L_shed_gridfm_effective_minus_cmd", "mean"),
        mean_hybrid_minus_cmd=("L_shed_hybrid_minus_cmd", "mean"),
        median_hybrid_minus_cmd=("L_shed_hybrid_minus_cmd", "median"),
        max_abs_raw_minus_cmd=("abs_L_shed_gridfm_raw_minus_cmd", "max"),
        max_abs_effective_minus_cmd=("abs_L_shed_gridfm_effective_minus_cmd", "max"),
        max_abs_hybrid_minus_cmd=("abs_L_shed_hybrid_minus_cmd", "max"),
        mean_hybrid_minus_effective=("L_shed_hybrid_minus_gridfm_effective", "mean"),
        mean_R_norm=("R_norm", "mean"),
        mean_PAC_total=("PAC_total", "mean"),
    ).reset_index()


def _plot_levels_by_lambda(rows: pd.DataFrame, path: Path) -> None:
    summary = rows.groupby(["rho_phys", "lambda_R"], as_index=False)[LOAD_SHED_METRICS].mean()
    rhos = sorted(summary["rho_phys"].dropna().unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(7.0 * len(rhos), 4.7), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos):
        local = summary[np.isclose(summary["rho_phys"], rho)].sort_values("lambda_R")
        for metric in LOAD_SHED_METRICS:
            ax.plot(
                local["lambda_R"],
                local[metric],
                marker="o",
                linewidth=2.0,
                color=LOAD_SHED_COLORS[metric],
                label=LOAD_SHED_LABELS[metric],
            )
        ax.set_title(f"rho_phys = {rho:g}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("Mean load shedding")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Stage E K2 GridFM load-shedding terms", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.96))
    _savefig(fig, path)


def _plot_discrepancy_by_lambda(rows: pd.DataFrame, path: Path) -> None:
    summary = rows.groupby(["rho_phys", "lambda_R"], as_index=False)[DIFF_METRICS].mean()
    rhos = sorted(summary["rho_phys"].dropna().unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(7.0 * len(rhos), 4.7), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos):
        local = summary[np.isclose(summary["rho_phys"], rho)].sort_values("lambda_R")
        ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.7)
        for metric in DIFF_METRICS:
            ax.plot(
                local["lambda_R"],
                local[metric],
                marker="o",
                linewidth=2.0,
                color=DIFF_COLORS[metric],
                label=DIFF_LABELS[metric],
            )
        ax.set_title(f"rho_phys = {rho:g}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("Mean discrepancy versus commanded")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Stage E K2 GridFM load-shedding discrepancy", y=1.03, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.96))
    _savefig(fig, path)


def _plot_hybrid_heatmap(rows: pd.DataFrame, path: Path) -> None:
    rhos = sorted(rows["rho_phys"].dropna().unique())
    scenarios = sorted(rows["scenario_id"].astype(str).unique())
    lambdas = sorted(rows["lambda_R"].dropna().unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(6.4 * len(rhos), 4.8), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    max_abs = float(np.nanmax(np.abs(rows["L_shed_hybrid_minus_cmd"].astype(float)))) if len(rows) else 0.0
    vmax = max(max_abs, 1e-6)
    im = None
    for ax, rho in zip(axes, rhos):
        local = rows[np.isclose(rows["rho_phys"], rho)]
        pivot = (
            local.pivot_table(
                index="scenario_id",
                columns="lambda_R",
                values="L_shed_hybrid_minus_cmd",
                aggfunc="mean",
            )
            .reindex(index=scenarios, columns=lambdas)
        )
        im = ax.imshow(pivot.values.astype(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"Hybrid - commanded, rho={rho:g}")
        ax.set_xticks(range(len(lambdas)))
        ax.set_xticklabels([f"{value:g}" for value in lambdas])
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_xlabel("lambda_R")
        for y_idx in range(len(scenarios)):
            for x_idx in range(len(lambdas)):
                value = pivot.values[y_idx, x_idx]
                if np.isfinite(value):
                    ax.text(x_idx, y_idx, f"{value:.2f}", ha="center", va="center", fontsize=8)
    if im is not None:
        cbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.62])
        fig.colorbar(im, cax=cbar_ax, label="L_shed_hybrid - L_shed_cmd")
    fig.suptitle("Stage E K2 scenario-level hybrid discrepancy", y=1.03, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.82, wspace=0.18)
    _savefig(fig, path)


def _plot_cmd_vs_gridfm(rows: pd.DataFrame, path: Path) -> None:
    compare_metrics = ["L_shed_gridfm_raw", "L_shed_gridfm_effective", "L_shed_hybrid"]
    fig, axes = plt.subplots(1, len(compare_metrics), figsize=(5.2 * len(compare_metrics), 4.8), sharex=True, sharey=True)
    max_value = float(
        np.nanmax(rows[["L_shed_cmd", *compare_metrics]].to_numpy(dtype=float))
    )
    min_value = float(
        np.nanmin(rows[["L_shed_cmd", *compare_metrics]].to_numpy(dtype=float))
    )
    lim_low = min(0.0, min_value) - 0.03
    lim_high = max_value + 0.03
    for ax, metric in zip(axes, compare_metrics):
        for rho, marker in [(0.0, "o"), (2.0, "^")]:
            local = rows[np.isclose(rows["rho_phys"], rho)]
            if local.empty:
                continue
            ax.scatter(
                local["L_shed_cmd"],
                local[metric],
                c=local["lambda_R"],
                cmap="viridis",
                vmin=0,
                vmax=1,
                marker=marker,
                edgecolor="#111827",
                linewidth=0.35,
                alpha=0.85,
                label=f"rho={rho:g}",
            )
        ax.plot([lim_low, lim_high], [lim_low, lim_high], color="#111827", linewidth=1.0, linestyle="--")
        ax.set_xlim(lim_low, lim_high)
        ax.set_ylim(lim_low, lim_high)
        ax.set_title(f"{LOAD_SHED_LABELS[metric]} vs commanded")
        ax.set_xlabel("L_shed_cmd")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Compared load-shed term")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    norm = plt.Normalize(0, 1)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.24, 0.014, 0.52])
    fig.colorbar(sm, cax=cbar_ax, label="lambda_R")
    fig.suptitle("Stage E K2 commanded versus GridFM/hybrid load shedding", y=1.03, fontsize=14)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.2, top=0.82, wspace=0.16)
    _savefig(fig, path)


def _plot_discrepancy_visuals(rows: pd.DataFrame) -> list[str]:
    outputs = [
        PLOTS / "stage_e_k2_load_shed_terms_by_lambda_rho.png",
        PLOTS / "stage_e_k2_load_shed_discrepancy_by_lambda_rho.png",
        PLOTS / "stage_e_k2_load_shed_hybrid_minus_cmd_heatmap.png",
        PLOTS / "stage_e_k2_load_shed_cmd_vs_gridfm_scatter.png",
    ]
    _plot_levels_by_lambda(rows, outputs[0])
    _plot_discrepancy_by_lambda(rows, outputs[1])
    _plot_hybrid_heatmap(rows, outputs[2])
    _plot_cmd_vs_gridfm(rows, outputs[3])
    return [str(path) for path in outputs]


def _plot_legacy_load_shed_cmd_gridfm_hybrid(rows: pd.DataFrame, path: Path) -> str:
    available = [metric for metric in LOAD_SHED_METRICS if metric in rows.columns]
    work = rows.copy()
    work["display_method"] = "Stage E K2 GridFM"
    for metric in available:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    summary = work.groupby("display_method", as_index=False)[available].mean()
    labels = summary["display_method"].astype(str).tolist()
    x = np.arange(len(summary))
    width = min(0.8 / max(len(available), 1), 0.22)
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = (np.arange(len(available)) - (len(available) - 1) / 2.0) * width
    for offset, metric in zip(offsets, available):
        ax.bar(x + offset, summary[metric].astype(float), width=width, label=metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _savefig(fig, path)
    return str(path)


def main() -> None:
    best = _read_csv(TABLES / "best_by_rho_scenario_lambda_stage.csv")
    numeric_columns = [
        "rho_phys",
        "lambda_R",
        "lambda_L",
        "R_norm",
        "J_true",
        "J_no_phys",
        "PAC_total",
        "L_shed",
        "L_shed_cmd",
        "L_shed_gridfm_raw",
        "L_shed_gridfm_effective",
        "L_shed_hybrid",
        "num_shutoff_lines",
        "topology_iteration",
    ]
    best = _numeric(best, numeric_columns)

    rows = best[best["stage"].astype(str).eq(STAGE)].copy()
    rows = rows.sort_values(["rho_phys", "scenario_id", "lambda_R"], kind="mergesort")

    keep_columns = [
        "scenario_id",
        "scenario_name",
        "rho_phys",
        "lambda_R",
        "lambda_L",
        "stage",
        "stage_label",
        "shutoff_line_ids",
        "line_id_key",
        "num_shutoff_lines",
        "topology_iteration",
        "L_shed",
        "L_shed_cmd",
        "L_shed_gridfm_raw",
        "L_shed_gridfm_effective",
        "L_shed_hybrid",
        "R_norm",
        "J_no_phys",
        "J_true",
        "PAC_total",
        "L_shed_source",
        "load_shed_mode",
        "termination_reason",
    ]
    rows = rows[[column for column in keep_columns if column in rows.columns]].copy()

    rows["L_shed_gridfm_raw_minus_cmd"] = rows["L_shed_gridfm_raw"] - rows["L_shed_cmd"]
    rows["L_shed_gridfm_effective_minus_cmd"] = rows["L_shed_gridfm_effective"] - rows["L_shed_cmd"]
    rows["L_shed_hybrid_minus_cmd"] = rows["L_shed_hybrid"] - rows["L_shed_cmd"]
    rows["L_shed_hybrid_minus_gridfm_raw"] = rows["L_shed_hybrid"] - rows["L_shed_gridfm_raw"]
    rows["L_shed_hybrid_minus_gridfm_effective"] = rows["L_shed_hybrid"] - rows["L_shed_gridfm_effective"]
    rows["abs_L_shed_gridfm_raw_minus_cmd"] = rows["L_shed_gridfm_raw_minus_cmd"].abs()
    rows["abs_L_shed_gridfm_effective_minus_cmd"] = rows["L_shed_gridfm_effective_minus_cmd"].abs()
    rows["abs_L_shed_hybrid_minus_cmd"] = rows["L_shed_hybrid_minus_cmd"].abs()

    summary = _summarize(rows.groupby(["rho_phys", "lambda_R"], dropna=False))
    overall = _summarize(rows.assign(scope="all").groupby(["scope"], dropna=False))
    legacy_plot = _plot_legacy_load_shed_cmd_gridfm_hybrid(
        rows,
        PLOTS / "load_shed_cmd_gridfm_hybrid_stage_e_k2_only.png",
    )
    plot_outputs = [legacy_plot]

    write_dataframe(TABLES / ROW_TABLE, rows)
    write_dataframe(TABLES / SUMMARY_TABLE, summary)
    write_dataframe(TABLES / OVERALL_TABLE, overall)
    write_json(
        TABLES / METADATA_FILE,
        {
            "source_table": str(TABLES / "best_by_rho_scenario_lambda_stage.csv"),
            "stage_filter": STAGE,
            "reason_for_stage_filter": (
                "The command/raw/effective/hybrid load-shed discrepancy is a GridFM-evaluated "
                "quantity. DC rows have exact DC load service with raw/effective GridFM fields "
                "unset, and heuristic rows are baseline/ranking comparisons rather than "
                "GridFM-optimized continuous recourse."
            ),
            "row_table": ROW_TABLE,
            "summary_table": SUMMARY_TABLE,
            "overall_table": OVERALL_TABLE,
            "plots": plot_outputs,
            "num_rows": int(len(rows)),
            "rho_values": [float(value) for value in sorted(rows["rho_phys"].dropna().unique())],
            "lambda_R_values": [float(value) for value in sorted(rows["lambda_R"].dropna().unique())],
            "columns_interpreted": {
                "L_shed_cmd": "commanded/corrected load service from selected controls",
                "L_shed_gridfm_raw": "raw GridFM-implied load shedding before clipping/correction",
                "L_shed_gridfm_effective": "GridFM-implied load shedding after clipping/island correction",
                "L_shed_hybrid": "active Stage E K2 objective load-shed metric",
            },
        },
    )

    print(f"Wrote {TABLES / ROW_TABLE}")
    print(f"Wrote {TABLES / SUMMARY_TABLE}")
    print(f"Wrote {TABLES / OVERALL_TABLE}")
    for path in plot_outputs:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
