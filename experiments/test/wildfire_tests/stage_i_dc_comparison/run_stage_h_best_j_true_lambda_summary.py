"""Plot best J_true across the lambda_R sweep for Stage I r11 main results."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT = (
    REPO_ROOT
    / "experiments"
    / "test"
    / "wildfire_tests"
    / "results"
    / "leq"
    / "stage_i"
    / "DC Approximation + Baseline Heuristic Comparison"
    / "main_results"
    / "r11"
)
TABLES_DIR = RUN_ROOT / "tables"
PLOTS_DIR = RUN_ROOT / "plots" / "summary"

STAGE_ORDER = [
    "stage_e_k2",
    "stage_i_a_dc_k2",
    "stage_i_b_dc_miqp_k2",
    "th_top1",
    "th_top2",
    "ah_k2_budgeted",
]
STAGE_LABELS = {
    "stage_e_k2": "Stage E K2 GridFM",
    "stage_i_a_dc_k2": "Stage I-a DC guided K2",
    "stage_i_b_dc_miqp_k2": "Stage I-b DC MIQP K2",
    "th_top1": "TH top-1",
    "th_top2": "TH top-2",
    "ah_k2_budgeted": "AH K2 budgeted",
}
STAGE_COLORS = {
    "stage_e_k2": "#2563eb",
    "stage_i_a_dc_k2": "#16a34a",
    "stage_i_b_dc_miqp_k2": "#f97316",
    "th_top1": "#7c3aed",
    "th_top2": "#a855f7",
    "ah_k2_budgeted": "#dc2626",
}
STAGE_MARKERS = {
    "stage_e_k2": "o",
    "stage_i_a_dc_k2": "s",
    "stage_i_b_dc_miqp_k2": "^",
    "th_top1": "D",
    "th_top2": "P",
    "ah_k2_budgeted": "X",
}


def _long_path(path: Path) -> str:
    text = str(path.resolve())
    if len(text) >= 240 and not text.startswith("\\\\?\\"):
        return "\\\\?\\" + text
    return text


def _mkdir(path: Path) -> None:
    os.makedirs(_long_path(path), exist_ok=True)


def _read_best_table() -> pd.DataFrame:
    path = TABLES_DIR / "best_by_rho_scenario_lambda_stage.csv"
    frame = pd.read_csv(_long_path(path), low_memory=False)
    required = {"scenario_id", "rho_phys", "lambda_R", "stage", "J_true"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    frame = frame[list(required.union({"stage_label"}))].copy()
    for col in ["rho_phys", "lambda_R", "J_true"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame[np.isfinite(frame["J_true"])].copy()
    frame["stage"] = frame["stage"].astype(str)
    frame["scenario_id"] = frame["scenario_id"].astype(str)
    frame["method_label"] = frame["stage"].map(STAGE_LABELS).fillna(frame["stage"])
    return frame


def _build_summary(best: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        best.groupby(["rho_phys", "lambda_R", "stage", "method_label"], dropna=False)
        .agg(
            mean_J_true=("J_true", "mean"),
            min_J_true=("J_true", "min"),
            max_J_true=("J_true", "max"),
            std_J_true=("J_true", "std"),
            n_scenarios=("scenario_id", "nunique"),
            n_rows=("J_true", "size"),
        )
        .reset_index()
    )
    grouped["stage_order"] = grouped["stage"].map({s: i for i, s in enumerate(STAGE_ORDER)}).fillna(999)
    return grouped.sort_values(["rho_phys", "stage_order", "lambda_R"], kind="mergesort")


def _savefig(fig: plt.Figure, path: Path) -> None:
    _mkdir(path.parent)
    fig.savefig(_long_path(path), dpi=190, bbox_inches="tight")
    plt.close(fig)


def _plot_mean(summary: pd.DataFrame, output_path: Path, *, include_stage_e: bool) -> None:
    data = summary.copy()
    if not include_stage_e:
        data = data[data["stage"] != "stage_e_k2"].copy()
    rhos = sorted(data["rho_phys"].dropna().unique())
    fig, axes = plt.subplots(1, len(rhos), figsize=(7.2 * len(rhos), 5.0), sharey=False)
    if len(rhos) == 1:
        axes = [axes]
    for ax, rho in zip(axes, rhos):
        local_rho = data[np.isclose(data["rho_phys"], rho)]
        for stage in STAGE_ORDER:
            local = local_rho[local_rho["stage"] == stage].sort_values("lambda_R")
            if local.empty:
                continue
            ax.plot(
                local["lambda_R"],
                local["mean_J_true"],
                marker=STAGE_MARKERS.get(stage, "o"),
                linewidth=2.0,
                markersize=6,
                color=STAGE_COLORS.get(stage),
                label=STAGE_LABELS.get(stage, stage),
            )
        ax.set_title(f"rho_phys = {rho:g}")
        ax.set_xlabel("lambda_R")
        ax.set_ylabel("Mean best J_true across scenarios")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    title = "Stage I r11 best J_true across lambda_R"
    if not include_stage_e:
        title += " (excluding Stage E GridFM)"
    fig.suptitle(title, y=1.03, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.18, 1.0, 0.96))
    _savefig(fig, output_path)


def _plot_by_scenario(best: pd.DataFrame, output_path: Path, *, include_stage_e: bool) -> None:
    data = best.copy()
    if not include_stage_e:
        data = data[data["stage"] != "stage_e_k2"].copy()
    scenarios = sorted(data["scenario_id"].unique())
    rhos = sorted(data["rho_phys"].dropna().unique())
    fig, axes = plt.subplots(
        len(rhos),
        len(scenarios),
        figsize=(4.2 * len(scenarios), 3.8 * len(rhos)),
        sharex=True,
        sharey=False,
    )
    if len(rhos) == 1:
        axes = np.array([axes])
    if len(scenarios) == 1:
        axes = axes.reshape((len(rhos), 1))
    for row_idx, rho in enumerate(rhos):
        for col_idx, scenario in enumerate(scenarios):
            ax = axes[row_idx, col_idx]
            local_cell = data[
                (data["scenario_id"] == scenario) & np.isclose(data["rho_phys"], rho)
            ]
            for stage in STAGE_ORDER:
                local = local_cell[local_cell["stage"] == stage].sort_values("lambda_R")
                if local.empty:
                    continue
                ax.plot(
                    local["lambda_R"],
                    local["J_true"],
                    marker=STAGE_MARKERS.get(stage, "o"),
                    linewidth=1.8,
                    markersize=5,
                    color=STAGE_COLORS.get(stage),
                    label=STAGE_LABELS.get(stage, stage),
                )
            ax.set_title(f"{scenario}, rho={rho:g}")
            ax.set_xlabel("lambda_R")
            if col_idx == 0:
                ax.set_ylabel("Best J_true")
            ax.grid(True, alpha=0.22)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    title = "Stage I r11 per-scenario best J_true across lambda_R"
    if not include_stage_e:
        title += " (excluding Stage E GridFM)"
    fig.suptitle(title, y=1.02, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.96))
    _savefig(fig, output_path)


def _plot_legacy_bar(summary: pd.DataFrame, output_path: Path, *, include_stage_e: bool) -> None:
    data = summary.copy()
    if not include_stage_e:
        data = data[data["stage"] != "stage_e_k2"].copy()
    data = data.sort_values(["stage_order", "lambda_R", "rho_phys"], kind="mergesort").copy()
    labels = [
        f"{row.method_label} | lambda_R={row.lambda_R:g} | rho={row.rho_phys:g}"
        for row in data.itertuples(index=False)
    ]
    colors = [STAGE_COLORS.get(stage, "#64748b") for stage in data["stage"]]
    fig, ax = plt.subplots(figsize=(max(12.0, 0.26 * len(data)), 6.0))
    ax.bar(range(len(data)), data["mean_J_true"].astype(float), color=colors, alpha=0.92)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7.5)
    ax.set_ylabel("mean best J_true across scenarios")
    ax.grid(axis="y", alpha=0.25)
    title = "Average objective by method, lambda_R, and rho_phys"
    if not include_stage_e:
        title += " (excluding Stage E GridFM)"
    ax.set_title(title)
    fig.tight_layout()
    _savefig(fig, output_path)


def main() -> None:
    _mkdir(TABLES_DIR)
    _mkdir(PLOTS_DIR)
    best = _read_best_table()
    summary = _build_summary(best)
    summary_path = TABLES_DIR / "best_j_true_by_method_lambda_rho_summary.csv"
    summary.to_csv(_long_path(summary_path), index=False)

    _plot_mean(
        summary,
        PLOTS_DIR / "best_j_true_by_method_lambda_rho.png",
        include_stage_e=True,
    )
    _plot_mean(
        summary,
        PLOTS_DIR / "best_j_true_by_method_lambda_rho_no_stage_e.png",
        include_stage_e=False,
    )
    _plot_by_scenario(
        best,
        PLOTS_DIR / "best_j_true_by_scenario_method_lambda_rho.png",
        include_stage_e=True,
    )
    _plot_by_scenario(
        best,
        PLOTS_DIR / "best_j_true_by_scenario_method_lambda_rho_no_stage_e.png",
        include_stage_e=False,
    )
    _plot_legacy_bar(
        summary,
        PLOTS_DIR / "average_objective_by_method_lambda_rho.png",
        include_stage_e=True,
    )
    _plot_legacy_bar(
        summary,
        PLOTS_DIR / "average_objective_by_method_lambda_rho_no_stage_e.png",
        include_stage_e=False,
    )

    metadata = {
        "source_table": str(TABLES_DIR / "best_by_rho_scenario_lambda_stage.csv"),
        "summary_table": str(summary_path),
        "plots": [
            str(PLOTS_DIR / "best_j_true_by_method_lambda_rho.png"),
            str(PLOTS_DIR / "best_j_true_by_method_lambda_rho_no_stage_e.png"),
            str(PLOTS_DIR / "best_j_true_by_scenario_method_lambda_rho.png"),
            str(PLOTS_DIR / "best_j_true_by_scenario_method_lambda_rho_no_stage_e.png"),
            str(PLOTS_DIR / "average_objective_by_method_lambda_rho.png"),
            str(PLOTS_DIR / "average_objective_by_method_lambda_rho_no_stage_e.png"),
        ],
        "metric": "J_true",
        "n_best_rows": int(len(best)),
        "stages": STAGE_ORDER,
    }
    with open(_long_path(TABLES_DIR / "best_j_true_by_method_lambda_rho_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
        fh.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
