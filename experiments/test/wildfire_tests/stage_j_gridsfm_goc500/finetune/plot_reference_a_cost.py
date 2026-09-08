"""Create the paper-facing Reference A economic AC-OPF cost comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / "goc_500_results" / "stage_j" / "refined_finetune_study"
SOURCE = STUDY / "core_results" / "reference_a_all.parquet"
OUTPUT_DIR = STUDY / "figures"
PAPER_DIR = ROOT.parent / "paper" / "images"

METHODS = [
    ("Guided-DC", "Guided-DC"),
    ("Guided-GridSFM (released v1.1)", "Model 1\nReleased"),
    ("Guided-GridSFM (FullTop-1000)", "Model 2\nFullTop-1000"),
    ("Guided-GridSFM (FullTop-1500)", "Model 3\nFullTop-1500"),
    ("Guided-GridSFM (FullTop+N-1)", "Model 4\nFullTop+N-1"),
]
SCENARIOS = [
    ("J-S1", "#168C91"),
    ("J-S2", "#D47A24"),
    ("J-S3", "#B45479"),
]


def plot() -> tuple[Path, Path]:
    data = pd.read_parquet(SOURCE)
    scenario_means = data.pivot_table(
        index="method",
        columns="scenario_id",
        values="reference_a_objective",
        aggfunc="mean",
    )
    overall_means = data.groupby("method")["reference_a_objective"].mean()

    x = np.arange(len(METHODS), dtype=float)
    offsets = np.array([-0.17, 0.0, 0.17])
    fig, ax = plt.subplots(figsize=(7.7, 4.45))

    for index, (method, _) in enumerate(METHODS):
        values = np.array([scenario_means.loc[method, scenario] for scenario, _ in SCENARIOS])
        ax.plot(
            [x[index], x[index]],
            [values.min(), values.max()],
            color="#A8A8A8",
            linewidth=1.2,
            zorder=1,
        )

    for offset, (scenario, color) in zip(offsets, SCENARIOS):
        values = [scenario_means.loc[method, scenario] for method, _ in METHODS]
        ax.scatter(
            x + offset,
            values,
            s=43,
            color=color,
            edgecolor="white",
            linewidth=0.55,
            label=f"{scenario} mean",
            zorder=3,
        )

    overall = np.array([overall_means.loc[method] for method, _ in METHODS])
    ax.scatter(
        x,
        overall,
        marker="D",
        s=51,
        color="#202020",
        edgecolor="white",
        linewidth=0.6,
        label="Overall mean",
        zorder=4,
    )
    for xi, value in zip(x, overall):
        ax.annotate(
            f"{value / 1000.0:.1f}",
            (xi, value),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=7.8,
            color="#202020",
        )

    ax.set_xticks(x, labels=[label for _, label in METHODS], fontsize=9)
    ax.set_ylabel(r"Reference A economic AC-OPF cost ($10^3$ objective units)", fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000.0:.0f}"))
    ax.set_ylim(446000, 470000)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "reference_a_economic_cost_by_model.pdf"
    png = OUTPUT_DIR / "reference_a_economic_cost_by_model.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(PAPER_DIR / pdf.name, bbox_inches="tight")
    fig.savefig(PAPER_DIR / png.name, dpi=320, bbox_inches="tight")
    plt.close(fig)
    return pdf, png


def plot_lambda_average() -> tuple[Path, Path]:
    data = pd.read_parquet(SOURCE)
    averaged = (
        data.groupby(["method", "lambda_r"], as_index=False)["reference_a_objective"]
        .mean()
        .sort_values(["method", "lambda_r"])
    )

    styles = [
        ("#6B7280", "X"),
        ("#D55E00", "o"),
        ("#6A3D9A", "s"),
        ("#009E73", "D"),
        ("#CC79A7", "^"),
    ]
    fig, ax = plt.subplots(figsize=(7.25, 4.35))
    for (method, label), (color, marker) in zip(METHODS, styles):
        rows = averaged.loc[averaged["method"].eq(method)]
        ax.plot(
            rows["lambda_r"],
            rows["reference_a_objective"],
            color=color,
            marker=marker,
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.5,
            linewidth=1.8,
            label=label.replace("\n", " "),
            zorder=3,
        )

    ax.set_xlabel(r"Risk weight $\lambda_R$", fontsize=10)
    ax.set_ylabel("Mean Reference A economic AC-OPF cost", fontsize=10)
    ax.set_xticks([0.0, 0.2, 0.5, 0.8, 1.0])
    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(443000, 467000)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.grid(color="#DADADA", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=8.2,
        handlelength=2.1,
        columnspacing=1.0,
    )
    fig.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTPUT_DIR / "reference_a_economic_cost_lambda_average.pdf"
    png = OUTPUT_DIR / "reference_a_economic_cost_lambda_average.png"
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
    lambda_pdf_path, lambda_png_path = plot_lambda_average()
    print(lambda_pdf_path)
    print(lambda_png_path)
