"""Plot Stage J exact-AC reference discrepancies and warm-start outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHODS = ["Guided-DC", "Guided-GridSFM"]
METHOD_COLORS = {
    "Guided-DC": "#0072B2",
    "Guided-GridSFM": "#D55E00",
}
SCENARIO_MARKERS = {
    "J-S1": "o",
    "J-S2": "s",
    "J-S3": "^",
}
WARM_START_ORDER = [
    "cold_start",
    "dc_partial_warm",
    "gridsfm_partial_warm",
    "gt_warm",
]
WARM_START_LABELS = {
    "cold_start": "Cold",
    "dc_partial_warm": "DC",
    "gridsfm_partial_warm": "GridSFM",
    "gt_warm": "Exact",
}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.24, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_mean(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _read_core(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core = root / "core_results"
    reference_a = _numeric(
        pd.read_csv(core / "reference_a_all.csv"),
        [
            "lambda_r",
            "reference_a_objective",
            "native_r_norm",
            "reference_a_r_norm_ac",
            "delta_r_norm_native_minus_ac",
            "native_l_shed_total",
            "reference_a_l_shed_total",
            "native_j_trade",
            "reference_a_j_true",
            "delta_j_true_native_minus_ac",
            "max_ac_loading",
            "reference_a_runtime_seconds",
        ],
    )
    reference_b = _numeric(
        pd.read_csv(core / "reference_b_all.csv"),
        [
            "lambda_r",
            "solver_objective",
            "runtime_seconds",
            "l_shed_ac_mld",
            "r_norm_ac_mld",
            "j_trade_ac_mld",
            "service_recovery",
            "max_ac_loading",
        ],
    )
    fidelity = _numeric(
        pd.read_csv(core / "state_fidelity_all.csv"),
        ["lambda_r", "mae", "rmse", "max", "nmae", "nrmse", "nmax"],
    )
    warm = _numeric(
        pd.read_csv(core / "warm_start_all.csv"),
        [
            "lambda_r",
            "solver_runtime_seconds",
            "wall_seconds",
            "start_construction_seconds",
            "end_to_end_seconds",
            "iteration_count",
            "objective",
        ],
    )
    return reference_a, reference_b, fidelity, warm


def _state_distance_summary(fidelity: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    rows = fidelity[fidelity["method"].isin(methods)].copy()
    rows = rows[rows["metric_family"].isin(["Pg", "Qg", "V", "Pij", "Qij", "Pji", "Qji"])]
    summary = (
        rows.groupby(["scenario_id", "lambda_r", "method"], dropna=False)
        .agg(
            ac_projection_distance_nrmse=("nrmse", _finite_mean),
            ac_projection_distance_nmae=("nmae", _finite_mean),
            available_metric_families=("nrmse", lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum())),
        )
        .reset_index()
    )
    return summary


def _plot_reference_a(
    reference_a: pd.DataFrame,
    state_distance: pd.DataFrame,
    figures: Path,
    methods: list[str],
) -> pd.DataFrame:
    rows = reference_a[reference_a["method"].isin(methods)].copy()
    rows = rows.merge(state_distance, on=["scenario_id", "lambda_r", "method"], how="left")
    rows["abs_delta_j_trade"] = rows["delta_j_true_native_minus_ac"].abs()
    rows["abs_delta_r_norm"] = rows["delta_r_norm_native_minus_ac"].abs()
    rows["delta_l_shed_native_minus_ac"] = rows["native_l_shed_total"] - rows["reference_a_l_shed_total"]
    rows["abs_delta_l_shed"] = rows["delta_l_shed_native_minus_ac"].abs()

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))
    axes = axes.ravel()
    for method in methods:
        method_rows = rows[rows["method"] == method]
        for scenario, scenario_rows in method_rows.groupby("scenario_id"):
            marker = SCENARIO_MARKERS.get(scenario, "o")
            label = f"{method} {scenario}"
            color = METHOD_COLORS[method]
            axes[0].plot(
                scenario_rows["lambda_r"],
                scenario_rows["reference_a_objective"],
                marker=marker,
                color=color,
                linewidth=1.5,
                alpha=0.85,
                label=label,
            )
            axes[1].plot(
                scenario_rows["lambda_r"],
                scenario_rows["abs_delta_r_norm"],
                marker=marker,
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )
            axes[2].plot(
                scenario_rows["lambda_r"],
                scenario_rows["abs_delta_l_shed"],
                marker=marker,
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )
            axes[3].plot(
                scenario_rows["lambda_r"],
                scenario_rows["ac_projection_distance_nrmse"],
                marker=marker,
                color=color,
                linewidth=1.5,
                alpha=0.85,
            )

    _style(axes[0], "Reference A Economic AC-OPF Cost", "lambda_R", "AC objective")
    _style(axes[1], "Native vs Reference A Risk Discrepancy", "lambda_R", "|native R_norm - AC R_norm|")
    _style(axes[2], "Native vs Reference A Load Discrepancy", "lambda_R", "|native L_shed - AC L_shed|")
    _style(axes[3], "Native State Distance To Reference A", "lambda_R", "mean normalized RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Stage J Guided Finalists: Exact AC Reference A And Native-State Discrepancy", y=0.99, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    _save(fig, figures / "stage_j_guided_reference_a_discrepancy_distance.png")
    return rows


def _plot_reference_b(reference_b: pd.DataFrame, figures: Path, methods: list[str]) -> pd.DataFrame:
    rows = reference_b[reference_b["method"].isin(methods)].copy()
    rows["served_fraction_ac_mld"] = 1.0 - rows["l_shed_ac_mld"]
    b1 = rows[rows["reference_b_stage"] == "B1_load_delivery"].copy()
    b2 = rows[rows["reference_b_stage"] == "B2_cost_tiebreak"].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))
    for method in methods:
        for scenario, scenario_rows in b1[b1["method"] == method].groupby("scenario_id"):
            axes[0].plot(
                scenario_rows["lambda_r"],
                scenario_rows["served_fraction_ac_mld"],
                marker=SCENARIO_MARKERS.get(scenario, "o"),
                color=METHOD_COLORS[method],
                linewidth=1.5,
                alpha=0.85,
                label=f"{method} {scenario}",
            )
            axes[1].plot(
                scenario_rows["lambda_r"],
                scenario_rows["r_norm_ac_mld"],
                marker=SCENARIO_MARKERS.get(scenario, "o"),
                color=METHOD_COLORS[method],
                linewidth=1.5,
                alpha=0.85,
            )
        for scenario, scenario_rows in b2[b2["method"] == method].groupby("scenario_id"):
            axes[2].plot(
                scenario_rows["lambda_r"],
                scenario_rows["solver_objective"],
                marker=SCENARIO_MARKERS.get(scenario, "o"),
                color=METHOD_COLORS[method],
                linewidth=1.5,
                alpha=0.85,
            )

    _style(axes[0], "Reference B1 Maximum Load Delivery", "lambda_R", "AC served-load fraction")
    axes[0].set_ylim(0.98, 1.001)
    _style(axes[1], "Reference B1 Wildfire Loading Outcome", "lambda_R", "AC MLD R_norm")
    _style(axes[2], "Reference B2 Economic Tie-Break Cost", "lambda_R", "solver objective")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Stage J Guided Finalists: Reference B Maximum-Load-Delivery Outcomes", y=0.99, fontsize=13, weight="bold")
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.9))
    _save(fig, figures / "stage_j_guided_reference_b_mld.png")
    return rows


def _plot_state_distance_heatmap(fidelity: pd.DataFrame, figures: Path, methods: list[str]) -> pd.DataFrame:
    rows = fidelity[fidelity["method"].isin(methods)].copy()
    rows = rows[rows["metric_family"].isin(["Pg", "Qg", "V", "Pij", "Qij", "Pji", "Qji"])]
    summary = (
        rows.groupby(["method", "metric_family"], dropna=False)
        .agg(mean_nrmse=("nrmse", _finite_mean), mean_nmae=("nmae", _finite_mean), records=("nrmse", "size"))
        .reset_index()
    )
    pivot = summary.pivot(index="method", columns="metric_family", values="mean_nrmse")
    pivot = pivot.reindex(index=methods, columns=["Pg", "Qg", "V", "Pij", "Qij", "Pji", "Qji"])

    fig, ax = plt.subplots(figsize=(9.8, 3.2))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Mean Native-To-Reference-A State Distance", fontsize=12, weight="bold")
    for i, method in enumerate(pivot.index):
        for j, metric in enumerate(pivot.columns):
            value = pivot.loc[method, metric]
            text = "N/A" if pd.isna(value) else f"{value:.3g}"
            ax.text(j, i, text, ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, label="mean normalized RMSE")
    fig.tight_layout()
    _save(fig, figures / "stage_j_guided_state_distance_heatmap.png")
    return summary


def _plot_warm_start(warm: pd.DataFrame, figures: Path, methods: list[str]) -> pd.DataFrame:
    rows = warm[warm["method"].isin(methods)].copy()
    key_cols = ["setting_code", "scenario_id", "lambda_r", "method", "finalist_backend", "finalist_topology_id"]
    cold = (
        rows[rows["warm_start_type"] == "cold_start"][key_cols + ["solver_runtime_seconds", "end_to_end_seconds"]]
        .rename(columns={"solver_runtime_seconds": "cold_solver_seconds", "end_to_end_seconds": "cold_end_to_end_seconds"})
    )
    rows = rows.merge(cold, on=key_cols, how="left")
    rows["solver_seconds_saved_vs_cold"] = rows["cold_solver_seconds"] - rows["solver_runtime_seconds"]
    rows["end_to_end_seconds_saved_vs_cold"] = rows["cold_end_to_end_seconds"] - rows["end_to_end_seconds"]
    rows["solver_speedup_vs_cold"] = rows["cold_solver_seconds"] / rows["solver_runtime_seconds"]
    rows["end_to_end_speedup_vs_cold"] = rows["cold_end_to_end_seconds"] / rows["end_to_end_seconds"]
    rows["warm_start_label"] = rows["warm_start_type"].map(WARM_START_LABELS).fillna(rows["warm_start_type"])

    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(5.0 * (len(methods) + 1), 4.6))
    axes = list(axes) if hasattr(axes, "__len__") else [axes]
    positions = range(len(WARM_START_ORDER))
    for axis, method in zip(axes[:-1], methods):
        method_rows = rows[rows["method"] == method]
        data = [
            method_rows[method_rows["warm_start_type"] == warm_type]["solver_runtime_seconds"].dropna()
            for warm_type in WARM_START_ORDER
        ]
        box = axis.boxplot(
            data,
            positions=list(positions),
            widths=0.56,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#D8E8F5")
            patch.set_edgecolor("#333333")
            patch.set_alpha(0.9)
        for idx, warm_type in enumerate(WARM_START_ORDER):
            values = method_rows[method_rows["warm_start_type"] == warm_type]["solver_runtime_seconds"].dropna()
            axis.scatter([idx] * len(values), values, s=18, alpha=0.55, color=METHOD_COLORS[method])
        axis.set_xticks(list(positions))
        axis.set_xticklabels([WARM_START_LABELS[w] for w in WARM_START_ORDER], rotation=18)
        _style(axis, f"{method}: Reference A Solver Runtime", "", "seconds")

    non_cold = rows[rows["warm_start_type"] != "cold_start"].copy()
    labels = [WARM_START_LABELS[w] for w in WARM_START_ORDER if w != "cold_start"]
    if len(methods) == 1:
        x_offsets = {methods[0]: 0.0}
    else:
        spacing = 0.32 / max(len(methods) - 1, 1)
        x_offsets = {method: -0.16 + index * spacing for index, method in enumerate(methods)}
    comparison_axis = axes[-1]
    for method in methods:
        method_rows = non_cold[non_cold["method"] == method]
        for idx, warm_type in enumerate([w for w in WARM_START_ORDER if w != "cold_start"]):
            values = method_rows[method_rows["warm_start_type"] == warm_type]["solver_seconds_saved_vs_cold"].dropna()
            comparison_axis.scatter(
                [idx + x_offsets[method]] * len(values),
                values,
                s=28,
                alpha=0.7,
                color=METHOD_COLORS[method],
                label=method if idx == 0 else None,
            )
            if not values.empty:
                comparison_axis.plot(
                    [idx + x_offsets[method] - 0.08, idx + x_offsets[method] + 0.08],
                    [values.median(), values.median()],
                    color="black",
                    linewidth=1.2,
                )
    comparison_axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.7)
    comparison_axis.set_xticks(range(len(labels)))
    comparison_axis.set_xticklabels(labels, rotation=18)
    _style(comparison_axis, "Solver Seconds Saved Relative To Respective Cold Start", "", "cold runtime - warm runtime")
    comparison_axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Stage J Warm-Start Study For Reference A AC-OPF", y=1.02, fontsize=13, weight="bold")
    fig.tight_layout()
    _save(fig, figures / "stage_j_guided_warm_start_study.png")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run",
    )
    args = parser.parse_args()

    root = Path(args.result_root).resolve()
    figures = root / "figures"
    summary_dir = root / "core_results" / "derived_visual_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    reference_a, reference_b, fidelity, warm = _read_core(root)
    available_methods = set(reference_a["method"].dropna())
    methods = [method for method in METHODS if method in available_methods]
    if not methods:
        raise ValueError(f"No supported guided methods found; expected one of {METHODS}")
    state_distance = _state_distance_summary(fidelity, methods)

    reference_a_summary = _plot_reference_a(reference_a, state_distance, figures, methods)
    reference_b_summary = _plot_reference_b(reference_b, figures, methods)
    fidelity_summary = _plot_state_distance_heatmap(fidelity, figures, methods)
    warm_summary = _plot_warm_start(warm, figures, methods)

    reference_a_summary.to_csv(summary_dir / "guided_reference_a_discrepancy_distance.csv", index=False)
    reference_b_summary.to_csv(summary_dir / "guided_reference_b_mld.csv", index=False)
    fidelity_summary.to_csv(summary_dir / "guided_state_distance_by_metric.csv", index=False)
    warm_summary.to_csv(summary_dir / "guided_warm_start_speedups.csv", index=False)


if __name__ == "__main__":
    main()
