"""Plot the controlled DC, frozen GridSFM, fine-tuned GridSFM, and TH comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.summarize_stage_j_complete import (
    METHOD_COLORS,
    METHOD_ORDER,
    build_model_comparison,
)


SCENARIO_MARKERS = {"J-S1": "o", "J-S2": "s", "J-S3": "^"}
METRIC_ORDER = ["Pg", "Qg", "V", "Pij", "Qij", "Pji", "Qji"]
WARM_START_ORDER = [
    "cold_start",
    "dc_partial_warm",
    "gridsfm_frozen_full_warm",
    "gridsfm_ft_full_warm",
    "gt_warm",
]
WARM_START_LABELS = {
    "cold_start": "Cold",
    "dc_partial_warm": "DC",
    "gridsfm_frozen_full_warm": "GridSFM frozen",
    "gridsfm_ft_full_warm": "GridSFM fine-tuned",
    "gt_warm": "Exact",
}
WARM_START_COLORS = {
    "cold_start": "#666666",
    "dc_partial_warm": "#0072B2",
    "gridsfm_frozen_full_warm": "#D55E00",
    "gridsfm_ft_full_warm": "#6A3D9A",
    "gt_warm": "#009E73",
}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _read_core(root: Path) -> dict[str, pd.DataFrame]:
    core = root / "core_results"
    return {
        "reference_a": _numeric(
            pd.read_csv(core / "reference_a_all.csv"),
            [
                "lambda_r", "reference_a_objective", "native_r_norm",
                "reference_a_r_norm_ac", "delta_r_norm_native_minus_ac",
                "native_l_shed_total", "reference_a_l_shed_total",
                "delta_j_true_native_minus_ac", "reference_a_runtime_seconds",
            ],
        ),
        "reference_b": _numeric(
            pd.read_csv(core / "reference_b_all.csv"),
            ["lambda_r", "solver_objective", "runtime_seconds", "l_shed_ac_mld", "r_norm_ac_mld"],
        ),
        "fidelity": _numeric(
            pd.read_csv(core / "state_fidelity_all.csv"),
            ["lambda_r", "nmae", "nrmse"],
        ),
        "warm": _numeric(
            pd.read_csv(core / "warm_start_all.csv"),
            ["lambda_r", "solver_runtime_seconds", "end_to_end_seconds"],
        ),
    }


def _merge(fine_tuned: dict[str, pd.DataFrame], frozen: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {key: build_model_comparison(fine_tuned[key], frozen[key]) for key in fine_tuned}


def _style(axis: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_title(title, fontsize=10, weight="bold")
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.22, linewidth=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, figures: Path, filename: str) -> None:
    """Stage through a short path so Windows/OneDrive cannot drop long filenames."""
    with tempfile.TemporaryDirectory(prefix="stage_j_plot_") as temporary:
        staged = Path(temporary) / filename
        fig.savefig(staged, dpi=180, bbox_inches="tight")
        if sys.platform == "win32":
            result = subprocess.run(
                [
                    "robocopy", temporary, str(figures), filename,
                    "/COPY:DAT", "/R:2", "/W:1", "/NFL", "/NDL", "/NJH", "/NJS", "/NP",
                ],
                check=False,
            )
            if result.returncode >= 8:
                raise RuntimeError(f"robocopy failed for {filename}: exit {result.returncode}")
        else:
            import shutil

            shutil.copy2(staged, figures / filename)
    plt.close(fig)


def _legend_handles() -> list[Line2D]:
    method_handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], linewidth=2, label=method)
        for method in METHOD_ORDER
    ]
    scenario_handles = [
        Line2D(
            [0], [0], color="#333333", marker=marker, linestyle="None",
            markersize=6, label=scenario,
        )
        for scenario, marker in SCENARIO_MARKERS.items()
    ]
    return method_handles + scenario_handles


def _plot_lines(
    axis: plt.Axes,
    rows: pd.DataFrame,
    y_column: str,
    *,
    reference_b_stage: str | None = None,
) -> None:
    if reference_b_stage is not None:
        rows = rows[rows["reference_b_stage"] == reference_b_stage]
    for method in METHOD_ORDER:
        method_rows = rows[rows["method"] == method]
        for scenario, scenario_rows in method_rows.groupby("scenario_id"):
            scenario_rows = scenario_rows.sort_values("lambda_r")
            axis.plot(
                scenario_rows["lambda_r"], scenario_rows[y_column],
                color=METHOD_COLORS[method], marker=SCENARIO_MARKERS.get(scenario, "o"),
                linewidth=1.25, markersize=4, alpha=0.82,
            )


def _state_distance(fidelity: pd.DataFrame) -> pd.DataFrame:
    rows = fidelity[fidelity["metric_family"].isin(METRIC_ORDER)].copy()
    return (
        rows.groupby(["scenario_id", "lambda_r", "method"], dropna=False)
        .agg(ac_projection_distance_nrmse=("nrmse", "mean"))
        .reset_index()
    )


def _plot_reference_a(data: dict[str, pd.DataFrame], figures: Path) -> pd.DataFrame:
    rows = data["reference_a"].merge(
        _state_distance(data["fidelity"]), on=["scenario_id", "lambda_r", "method"], how="left"
    )
    rows["abs_delta_r_norm"] = rows["delta_r_norm_native_minus_ac"].abs()
    rows["abs_delta_l_shed"] = (
        rows["native_l_shed_total"] - rows["reference_a_l_shed_total"]
    ).abs()
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    specs = [
        ("reference_a_objective", "Reference A Economic AC-OPF Cost", "AC objective"),
        ("abs_delta_r_norm", "Native vs Reference A Risk Discrepancy", "absolute risk discrepancy"),
        ("abs_delta_l_shed", "Native vs Reference A Load Discrepancy", "absolute load discrepancy"),
        ("ac_projection_distance_nrmse", "Native State Distance to Reference A", "mean normalized RMSE"),
    ]
    for axis, (column, title, ylabel) in zip(axes.ravel(), specs):
        _plot_lines(axis, rows, column)
        _style(axis, title, "lambda_R", ylabel)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Stage J Model Comparison: Exact AC Reference A", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    _save_figure(fig, figures, "stage_j_guided_reference_a_discrepancy_distance.png")
    return rows


def _plot_reference_b(data: dict[str, pd.DataFrame], figures: Path) -> pd.DataFrame:
    rows = data["reference_b"].copy()
    rows["served_fraction_ac_mld"] = 1.0 - rows["l_shed_ac_mld"]
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.8))
    _plot_lines(axes[0], rows, "served_fraction_ac_mld", reference_b_stage="B1_load_delivery")
    _plot_lines(axes[1], rows, "r_norm_ac_mld", reference_b_stage="B1_load_delivery")
    _plot_lines(axes[2], rows, "solver_objective", reference_b_stage="B2_cost_tiebreak")
    _style(axes[0], "Reference B1 Maximum Load Delivery", "lambda_R", "served-load fraction")
    _style(axes[1], "Reference B1 Wildfire Loading", "lambda_R", "AC MLD R_norm")
    _style(axes[2], "Reference B2 Economic Tie-Break", "lambda_R", "solver objective")
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("Stage J Model Comparison: Reference B Outcomes", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.18, 1, 0.92))
    _save_figure(fig, figures, "stage_j_guided_reference_b_mld.png")
    return rows


def _plot_fidelity(data: dict[str, pd.DataFrame], figures: Path) -> pd.DataFrame:
    rows = data["fidelity"]
    rows = rows[rows["metric_family"].isin(METRIC_ORDER)]
    summary = (
        rows.groupby(["method", "metric_family"], dropna=False)
        .agg(mean_nrmse=("nrmse", "mean"), mean_nmae=("nmae", "mean"), records=("nrmse", "size"))
        .reset_index()
    )
    pivot = summary.pivot(index="method", columns="metric_family", values="mean_nrmse")
    pivot = pivot.reindex(index=METHOD_ORDER, columns=METRIC_ORDER)
    fig, axis = plt.subplots(figsize=(10.8, 4.6))
    image = axis.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="magma")
    axis.set_xticks(range(len(METRIC_ORDER)), METRIC_ORDER)
    axis.set_yticks(range(len(METHOD_ORDER)), METHOD_ORDER)
    axis.set_title("Mean Native-to-Reference-A State Distance", fontsize=12, weight="bold")
    for row_index, method in enumerate(METHOD_ORDER):
        for column_index, metric in enumerate(METRIC_ORDER):
            value = pivot.loc[method, metric]
            axis.text(
                column_index, row_index, "N/A" if pd.isna(value) else f"{value:.3g}",
                ha="center", va="center", color="white", fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="mean normalized RMSE")
    fig.tight_layout()
    _save_figure(fig, figures, "stage_j_guided_state_distance_heatmap.png")
    return summary


def _warm_start_statistics(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        rows.groupby(["method", "warm_start_type", "warm_start_label"], dropna=False)
        .agg(
            observations=("solver_runtime_seconds", "size"),
            successful=("status", lambda values: values.isin(["LOCALLY_SOLVED", "OPTIMAL"]).sum()),
            mean_ipopt_solve_seconds=("solver_runtime_seconds", "mean"),
            median_ipopt_solve_seconds=("solver_runtime_seconds", "median"),
            std_ipopt_solve_seconds=("solver_runtime_seconds", "std"),
            q1_ipopt_solve_seconds=("solver_runtime_seconds", lambda values: values.quantile(0.25)),
            q3_ipopt_solve_seconds=("solver_runtime_seconds", lambda values: values.quantile(0.75)),
            min_ipopt_solve_seconds=("solver_runtime_seconds", "min"),
            max_ipopt_solve_seconds=("solver_runtime_seconds", "max"),
        )
        .reset_index()
    )

    paired_rows: list[dict[str, object]] = []
    keys = ["setting_code", "method"]
    pivot = rows.pivot(index=keys, columns="warm_start_type", values="solver_runtime_seconds")
    for method in METHOD_ORDER:
        method_rows = pivot.xs(method, level="method")
        for warm_type in WARM_START_ORDER[1:]:
            savings = method_rows["cold_start"] - method_rows[warm_type]
            paired_rows.append({
                "method": method,
                "comparison": f"{WARM_START_LABELS[warm_type]} vs Cold",
                "observations": int(savings.notna().sum()),
                "median_seconds_saved": float(savings.median()),
                "mean_seconds_saved": float(savings.mean()),
                "wins": int((savings > 0).sum()),
                "ties": int((savings.abs() <= 1e-9).sum()),
                "losses": int((savings < 0).sum()),
            })
        model_starts = [value for value in WARM_START_ORDER if value.startswith("gridsfm_")]
        for baseline, candidate in zip(model_starts, model_starts[1:]):
            checkpoint_delta = method_rows[baseline] - method_rows[candidate]
            paired_rows.append({
                "method": method,
                "comparison": f"{WARM_START_LABELS[candidate]} vs {WARM_START_LABELS[baseline]}",
                "observations": int(checkpoint_delta.notna().sum()),
                "median_seconds_saved": float(checkpoint_delta.median()),
                "mean_seconds_saved": float(checkpoint_delta.mean()),
                "wins": int((checkpoint_delta > 0).sum()),
                "ties": int((checkpoint_delta.abs() <= 1e-9).sum()),
                "losses": int((checkpoint_delta < 0).sum()),
            })
    return summary, pd.DataFrame(paired_rows)


def _plot_warm_start(
    data: dict[str, pd.DataFrame], figures: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = data["warm"].copy()
    key_columns = [
        "setting_code", "scenario_id", "lambda_r", "method",
        "finalist_backend", "finalist_topology_id",
    ]
    cold = rows[rows["warm_start_type"] == "cold_start"][key_columns + ["solver_runtime_seconds"]]
    cold = cold.rename(columns={"solver_runtime_seconds": "cold_solver_seconds"})
    rows = rows.merge(cold, on=key_columns, how="left")
    rows["solver_seconds_saved_vs_cold"] = rows["cold_solver_seconds"] - rows["solver_runtime_seconds"]

    fig, axis = plt.subplots(figsize=(14.6, 6.2))
    base_positions = list(range(len(METHOD_ORDER)))
    offsets = np.linspace(-0.38, 0.38, len(WARM_START_ORDER))
    box_width = min(0.13, 0.72 / len(WARM_START_ORDER))
    for warm_type, offset in zip(WARM_START_ORDER, offsets):
        samples = [
            rows[
                (rows["method"] == method)
                & (rows["warm_start_type"] == warm_type)
            ]["solver_runtime_seconds"].dropna()
            for method in METHOD_ORDER
        ]
        boxes = axis.boxplot(
            samples,
            positions=[position + offset for position in base_positions],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "#111111", "linewidth": 1.1},
            boxprops={"linewidth": 0.9},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"markersize": 2.5, "alpha": 0.45},
        )
        for box in boxes["boxes"]:
            box.set_facecolor(WARM_START_COLORS[warm_type])
            box.set_alpha(0.72)
    axis.set_xticks(base_positions, METHOD_ORDER, rotation=10, ha="right")
    _style(
        axis,
        "IPOPT Solve Time by Fixed Reference A Finalist and Initialization",
        "Fixed topology and alpha finalist family",
        "IPOPT solve time (seconds)",
    )
    axis.legend(
        handles=[
            Patch(facecolor=WARM_START_COLORS[value], alpha=0.72, label=WARM_START_LABELS[value])
            for value in WARM_START_ORDER
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=min(4, len(WARM_START_ORDER)),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save_figure(fig, figures, "stage_j_guided_warm_start_study.png")
    summary, paired = _warm_start_statistics(rows)
    return rows, summary, paired


def _plot_ipopt_timing_summary(
    rows: pd.DataFrame, figures: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        rows.groupby(["warm_start_type", "warm_start_label"], dropna=False)
        .agg(
            observations=("solver_runtime_seconds", "size"),
            mean_ipopt_solve_seconds=("solver_runtime_seconds", "mean"),
            median_ipopt_solve_seconds=("solver_runtime_seconds", "median"),
            std_ipopt_solve_seconds=("solver_runtime_seconds", "std"),
            q1_ipopt_solve_seconds=("solver_runtime_seconds", lambda values: values.quantile(0.25)),
            q3_ipopt_solve_seconds=("solver_runtime_seconds", lambda values: values.quantile(0.75)),
            min_ipopt_solve_seconds=("solver_runtime_seconds", "min"),
            max_ipopt_solve_seconds=("solver_runtime_seconds", "max"),
        )
        .reset_index()
    )
    overall["warm_start_order"] = overall["warm_start_type"].map(
        {value: index for index, value in enumerate(WARM_START_ORDER)}
    )
    overall = overall.sort_values("warm_start_order")

    pivot = rows.pivot(
        index=["setting_code", "method"],
        columns="warm_start_type",
        values="solver_runtime_seconds",
    )
    comparison_specs = [
        (f"{WARM_START_LABELS[value]} vs Cold", "cold_start", value)
        for value in WARM_START_ORDER[1:]
    ]
    paired_rows: list[dict[str, object]] = []
    for label, baseline, candidate in comparison_specs:
        savings = pivot[baseline] - pivot[candidate]
        paired_rows.append({
            "comparison": label,
            "observations": int(savings.notna().sum()),
            "mean_seconds_saved": float(savings.mean()),
            "median_seconds_saved": float(savings.median()),
            "q1_seconds_saved": float(savings.quantile(0.25)),
            "q3_seconds_saved": float(savings.quantile(0.75)),
            "wins": int((savings > 0).sum()),
            "ties": int((savings.abs() <= 1e-9).sum()),
            "losses": int((savings < 0).sum()),
        })
    paired = pd.DataFrame(paired_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1), gridspec_kw={"width_ratios": [1.2, 1]})
    samples = [
        rows[rows["warm_start_type"] == warm_type]["solver_runtime_seconds"].dropna()
        for warm_type in WARM_START_ORDER
    ]
    boxes = axes[0].boxplot(
        samples,
        positions=range(len(WARM_START_ORDER)),
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 1.4},
        flierprops={"markersize": 2.5, "alpha": 0.4},
    )
    for box, warm_type in zip(boxes["boxes"], WARM_START_ORDER):
        box.set_facecolor(WARM_START_COLORS[warm_type])
        box.set_alpha(0.7)
    axes[0].scatter(
        range(len(WARM_START_ORDER)),
        overall["mean_ipopt_solve_seconds"],
        marker="D",
        s=34,
        color="#111111",
        label="Mean",
        zorder=4,
    )
    axes[0].set_xticks(
        range(len(WARM_START_ORDER)),
        [WARM_START_LABELS[value] for value in WARM_START_ORDER],
        rotation=15,
        ha="right",
    )
    _style(axes[0], "Mean, Median, and Spread", "Initialization", "IPOPT solve time (seconds)")
    axes[0].legend(frameon=False, loc="upper right")

    y = list(range(len(paired)))
    axes[1].barh(y, paired["wins"], color="#009E73", label="Wins")
    axes[1].barh(y, paired["ties"], left=paired["wins"], color="#999999", label="Ties")
    axes[1].barh(
        y,
        paired["losses"],
        left=paired["wins"] + paired["ties"],
        color="#D55E00",
        label="Losses",
    )
    for index, row in paired.iterrows():
        axes[1].text(
            max(1, int(row["observations"]) + 1),
            index,
            f"median saved {row['median_seconds_saved']:+.3f}s",
            va="center",
            fontsize=8,
        )
    observations = int(paired["observations"].max()) if not paired.empty else 0
    axes[1].set_xlim(0, max(20, observations * 1.48))
    axes[1].set_yticks(y, paired["comparison"])
    axes[1].invert_yaxis()
    _style(axes[1], f"Paired Outcomes Across {observations} Instances", "matched instances", "")
    axes[1].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
    )

    fig.suptitle("Stage J IPOPT-Only Warm-Start Timing Summary", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save_figure(fig, figures, "stage_j_ipopt_warm_start_summary.png")
    return overall, paired


def _plot_ipopt_iteration_summary(
    rows: pd.DataFrame, figures: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        rows.groupby(["warm_start_type", "warm_start_label"], dropna=False)
        .agg(
            observations=("iteration_count", "size"),
            mean_iterations=("iteration_count", "mean"),
            median_iterations=("iteration_count", "median"),
            std_iterations=("iteration_count", "std"),
            min_iterations=("iteration_count", "min"),
            max_iterations=("iteration_count", "max"),
        )
        .reset_index()
    )
    overall["warm_start_order"] = overall["warm_start_type"].map(
        {value: index for index, value in enumerate(WARM_START_ORDER)}
    )
    overall = overall.sort_values("warm_start_order")

    pivot = rows.pivot(
        index=["setting_code", "method"],
        columns="warm_start_type",
        values="iteration_count",
    )
    paired_rows = []
    for warm_type in WARM_START_ORDER[1:]:
        saved = pivot["cold_start"] - pivot[warm_type]
        paired_rows.append({
            "comparison": f"{WARM_START_LABELS[warm_type]} vs Cold",
            "observations": int(saved.notna().sum()),
            "mean_iterations_saved": float(saved.mean()),
            "median_iterations_saved": float(saved.median()),
            "wins": int((saved > 0).sum()),
            "ties": int((saved == 0).sum()),
            "losses": int((saved < 0).sum()),
        })
    paired = pd.DataFrame(paired_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.1), gridspec_kw={"width_ratios": [1.2, 1]})
    samples = [
        rows[rows["warm_start_type"] == warm_type]["iteration_count"].dropna()
        for warm_type in WARM_START_ORDER
    ]
    boxes = axes[0].boxplot(
        samples,
        positions=range(len(WARM_START_ORDER)),
        widths=0.58,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#111111", "linewidth": 1.4},
        flierprops={"markersize": 2.5, "alpha": 0.4},
    )
    for box, warm_type in zip(boxes["boxes"], WARM_START_ORDER):
        box.set_facecolor(WARM_START_COLORS[warm_type])
        box.set_alpha(0.7)
    axes[0].scatter(
        range(len(WARM_START_ORDER)), overall["mean_iterations"],
        marker="D", s=34, color="#111111", label="Mean", zorder=4,
    )
    axes[0].set_xticks(
        range(len(WARM_START_ORDER)),
        [WARM_START_LABELS[value] for value in WARM_START_ORDER],
        rotation=15,
        ha="right",
    )
    _style(axes[0], "Mean, Median, and Spread", "Initialization", "IPOPT barrier iterations")
    axes[0].legend(frameon=False, loc="upper right")

    y = list(range(len(paired)))
    axes[1].barh(y, paired["wins"], color="#009E73", label="Fewer iterations")
    axes[1].barh(y, paired["ties"], left=paired["wins"], color="#999999", label="Ties")
    axes[1].barh(
        y, paired["losses"], left=paired["wins"] + paired["ties"],
        color="#D55E00", label="More iterations",
    )
    for index, row in paired.iterrows():
        axes[1].text(
            max(1, int(row["observations"]) + 1), index,
            f"median saved {row['median_iterations_saved']:+.0f}",
            va="center", fontsize=8,
        )
    observations = int(paired["observations"].max()) if not paired.empty else 0
    axes[1].set_xlim(0, max(20, observations * 1.48))
    axes[1].set_yticks(y, paired["comparison"])
    axes[1].invert_yaxis()
    _style(axes[1], f"Paired Outcomes Across {observations} Instances", "matched instances", "")
    axes[1].legend(
        frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3,
    )

    fig.suptitle("Stage J IPOPT Warm-Start Iteration Summary", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    _save_figure(fig, figures, "stage_j_ipopt_iteration_summary.png")
    return overall, paired


def plot_comparison(result_root: Path, frozen_root: Path) -> None:
    figures = result_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = _merge(_read_core(result_root), _read_core(frozen_root))
    core = result_root / "core_results"
    crossed_path = core / "warm_start_crossed.csv"
    if crossed_path.is_file():
        data["warm"] = _numeric(
            pd.read_csv(crossed_path),
            [
                "lambda_r", "solver_runtime_seconds", "wall_seconds",
                "start_construction_seconds", "end_to_end_seconds",
            ],
        )
    data["reference_a"].to_csv(core / "compare_ref_a.csv", index=False)
    data["reference_b"].to_csv(core / "compare_ref_b.csv", index=False)
    data["fidelity"].to_csv(core / "compare_fidelity.csv", index=False)
    data["warm"].to_csv(core / "compare_warm.csv", index=False)
    derived = core / "derived_visual_summaries"
    derived.mkdir(parents=True, exist_ok=True)
    _plot_reference_a(data, figures).to_csv(
        derived / "guided_reference_a_discrepancy_distance.csv", index=False
    )
    _plot_reference_b(data, figures).to_csv(
        derived / "guided_reference_b_mld.csv", index=False
    )
    _plot_fidelity(data, figures).to_csv(
        derived / "guided_state_distance_by_metric.csv", index=False
    )
    warm_rows, warm_summary, warm_paired = _plot_warm_start(data, figures)
    warm_rows.to_csv(derived / "guided_warm_start_speedups.csv", index=False)
    warm_summary.to_csv(derived / "guided_warm_start_summary.csv", index=False)
    warm_paired.to_csv(derived / "guided_warm_start_paired_comparisons.csv", index=False)
    ipopt_summary, ipopt_paired = _plot_ipopt_timing_summary(warm_rows, figures)
    ipopt_summary.to_csv(derived / "ipopt_warm_start_summary.csv", index=False)
    ipopt_paired.to_csv(derived / "ipopt_warm_start_win_loss.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--comparison-root", required=True)
    args = parser.parse_args()
    plot_comparison(Path(args.result_root), Path(args.comparison_root))


if __name__ == "__main__":
    main()
