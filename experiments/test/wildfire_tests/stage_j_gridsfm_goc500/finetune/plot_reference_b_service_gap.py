"""Plot native versus Reference B1 served-load fractions for the refined study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / "goc_500_results" / "stage_j" / "refined_finetune_study"
SOURCE = STUDY / "core_results" / "reference_b_all.parquet"
OUTPUT_DIR = STUDY / "figures"
PAPER_DIR = ROOT.parent / "paper" / "images"

METHODS = [
    ("Guided-DC", "Guided-DC"),
    ("Guided-GridSFM (released v1.1)", "Model 1\nReleased"),
    ("Guided-GridSFM (FullTop-1000)", "Model 2\nFullTop-1000"),
    ("Guided-GridSFM (FullTop-1500)", "Model 3\nFullTop-1500"),
    ("Guided-GridSFM (FullTop+N-1)", "Model 4\nFullTop+N-1"),
]
SCENARIOS = ["J-S1", "J-S2", "J-S3"]
LAMBDAS = [0.0, 0.2, 0.5, 0.8, 1.0]


def _setting_order() -> list[tuple[str, float]]:
    return [(scenario, value) for scenario in SCENARIOS for value in LAMBDAS]


def _prepare() -> pd.DataFrame:
    data = pd.read_parquet(SOURCE)
    data = data.loc[data["reference_b_stage"].eq("B1_load_delivery")].copy()
    data["reference_b1_served_fraction"] = 1.0 - data["l_shed_ac_mld"]
    data["native_served_fraction"] = (
        data["reference_b1_served_fraction"] - data["service_recovery"]
    )
    data["recoverable_service_fraction"] = (
        data["reference_b1_served_fraction"] - data["native_served_fraction"]
    )
    return data


def plot() -> tuple[Path, Path]:
    data = _prepare()
    order = _setting_order()
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        1,
        len(METHODS),
        figsize=(13.2, 6.4),
        sharex=True,
        sharey=True,
    )
    fig.subplots_adjust(left=0.095, right=0.995, top=0.89, bottom=0.16, wspace=0.08)

    for ax, (method, title) in zip(axes, METHODS):
        rows = data.loc[data["method"].eq(method)].set_index(["scenario_id", "lambda_r"])
        rows = rows.loc[order]
        native = rows["native_served_fraction"].to_numpy(dtype=float)
        ref_b1 = rows["reference_b1_served_fraction"].to_numpy(dtype=float)

        native_y = y - 0.09
        ref_b1_y = y + 0.09
        for left, right, yi_native, yi_ref in zip(native, ref_b1, native_y, ref_b1_y):
            ax.plot(
                [left, right], [yi_native, yi_ref],
                color="#707070", linewidth=1.5, zorder=1,
            )
        ax.scatter(
            native,
            native_y,
            s=30,
            color="#CC4C4C",
            edgecolor="white",
            linewidth=0.45,
            label="Selected service",
            zorder=3,
        )
        ax.scatter(
            ref_b1,
            ref_b1_y,
            s=34,
            marker="D",
            color="#167D8D",
            edgecolor="white",
            linewidth=0.45,
            label="Reference B1 maximum",
            zorder=4,
        )
        ax.axvline(1.0, color="#202020", linewidth=0.8, linestyle=":", zorder=0)
        ax.set_title(title, fontsize=10, weight="bold", pad=8)
        ax.set_xlim(0.9835, 1.0008)
        ax.set_xticks([0.985, 0.990, 0.995, 1.000])
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", color="#DADADA", linewidth=0.6)
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

    labels = [f"{scenario}, {value:.1f}" for scenario, value in order]
    axes[0].set_yticks(y, labels=labels, fontsize=8)
    axes[0].set_ylabel(r"Scenario and $\lambda_R$", fontsize=9)
    fig.text(0.545, 0.085, "Served-load fraction", ha="center", fontsize=10)

    for boundary in (4.5, 9.5):
        for ax in axes:
            ax.axhline(boundary, color="#9A9A9A", linewidth=0.8)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.545, 0.012),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "reference_b1_selected_vs_maximum_service.pdf"
    png = OUTPUT_DIR / "reference_b1_selected_vs_maximum_service.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(PAPER_DIR / pdf.name, bbox_inches="tight")
    fig.savefig(PAPER_DIR / png.name, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_boxplot() -> tuple[Path, Path]:
    data = _prepare()
    centers = np.arange(len(METHODS), dtype=float)
    native_positions = centers - 0.18
    ref_positions = centers + 0.18
    native_values = []
    ref_values = []

    for method, _ in METHODS:
        rows = data.loc[data["method"].eq(method)].set_index(["scenario_id", "lambda_r"])
        rows = rows.loc[_setting_order()]
        native_values.append(rows["native_served_fraction"].to_numpy(dtype=float))
        ref_values.append(rows["reference_b1_served_fraction"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    box_style = {
        "patch_artist": True,
        "widths": 0.28,
        "showfliers": False,
        "medianprops": {"color": "#202020", "linewidth": 1.5},
        "whiskerprops": {"color": "#505050", "linewidth": 1.0},
        "capprops": {"color": "#505050", "linewidth": 1.0},
    }
    native_boxes = ax.boxplot(native_values, positions=native_positions, **box_style)
    ref_boxes = ax.boxplot(ref_values, positions=ref_positions, **box_style)
    for box in native_boxes["boxes"]:
        box.set(facecolor="#E17A73", edgecolor="#A53D38", alpha=0.70)
    for box in ref_boxes["boxes"]:
        box.set(facecolor="#43A8B2", edgecolor="#126D78", alpha=0.72)

    jitter = np.linspace(-0.045, 0.045, len(_setting_order()))
    for index, (native, ref_b1) in enumerate(zip(native_values, ref_values)):
        x_native = native_positions[index] + jitter
        x_ref = ref_positions[index] + jitter
        for xn, xr, yn, yr in zip(x_native, x_ref, native, ref_b1):
            ax.plot([xn, xr], [yn, yr], color="#858585", linewidth=0.55, alpha=0.34, zorder=1)
        ax.scatter(
            x_native, native, s=14, color="#C94C46", alpha=0.78,
            edgecolor="white", linewidth=0.25, zorder=3,
        )
        ax.scatter(
            x_ref, ref_b1, s=15, marker="D", color="#138293", alpha=0.82,
            edgecolor="white", linewidth=0.25, zorder=4,
        )

    labels = [title.replace("\n", "\n") for _, title in METHODS]
    ax.set_xticks(centers, labels=labels, fontsize=9)
    ax.set_ylabel("Served-load fraction", fontsize=10)
    ax.set_ylim(0.9835, 1.0010)
    ax.set_yticks([0.985, 0.990, 0.995, 1.000])
    ax.axhline(1.0, color="#202020", linewidth=0.8, linestyle=":", zorder=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#E17A73", edgecolor="#A53D38", label="Selected service"),
            Patch(facecolor="#43A8B2", edgecolor="#126D78", label="Reference B1 maximum"),
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#C8C8C8",
        fontsize=9,
        ncol=2,
    )
    fig.tight_layout()

    pdf = OUTPUT_DIR / "reference_b1_service_boxplot.pdf"
    png = OUTPUT_DIR / "reference_b1_service_boxplot.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(PAPER_DIR / pdf.name, bbox_inches="tight")
    fig.savefig(PAPER_DIR / png.name, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_discrepancy_boxplot() -> tuple[Path, Path]:
    data = _prepare()
    values = []
    labels = []
    for method, title in METHODS:
        rows = data.loc[data["method"].eq(method)].set_index(["scenario_id", "lambda_r"])
        rows = rows.loc[_setting_order()]
        values.append(100.0 * rows["recoverable_service_fraction"].to_numpy(dtype=float))
        labels.append(title)

    colors = ["#6B7280", "#D55E00", "#6A3D9A", "#009E73", "#CC79A7"]
    positions = np.arange(1, len(METHODS) + 1, dtype=float)
    fig, ax = plt.subplots(figsize=(7.8, 4.55))
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#202020", "linewidth": 1.6},
        whiskerprops={"color": "#555555", "linewidth": 1.05},
        capprops={"color": "#555555", "linewidth": 1.05},
    )
    for box, color in zip(boxes["boxes"], colors):
        box.set(facecolor=color, edgecolor=color, alpha=0.64)

    jitter = np.linspace(-0.13, 0.13, len(_setting_order()))
    for x, group, color in zip(positions, values, colors):
        ax.scatter(
            x + jitter,
            group,
            s=21,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
            zorder=3,
        )
        mean = float(np.mean(group))
        ax.scatter(
            [x], [mean], marker="D", s=36, color="#202020",
            edgecolor="white", linewidth=0.45, zorder=4,
        )
        ax.text(
            x,
            max(group) + 0.055,
            f"mean {mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#303030",
        )

    ax.set_xticks(positions, labels=labels, fontsize=9)
    ax.set_ylabel("Gap to Reference B1 maximum\n(percentage points)", fontsize=10)
    ax.set_ylim(-0.04, 1.66)
    ax.axhline(0.0, color="#202020", linewidth=0.8, linestyle=":", zorder=0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    pdf = OUTPUT_DIR / "reference_b1_service_discrepancy_boxplot.pdf"
    png = OUTPUT_DIR / "reference_b1_service_discrepancy_boxplot.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(PAPER_DIR / pdf.name, bbox_inches="tight")
    fig.savefig(PAPER_DIR / png.name, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


if __name__ == "__main__":
    pdf_path, png_path = plot()
    print(pdf_path)
    print(png_path)
    box_pdf_path, box_png_path = plot_boxplot()
    print(box_pdf_path)
    print(box_png_path)
    gap_pdf_path, gap_png_path = plot_discrepancy_boxplot()
    print(gap_pdf_path)
    print(gap_png_path)
