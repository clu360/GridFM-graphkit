"""Generate paper-facing Stage J state-distance and warm-start figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT.parent / "goc_500_results" / "stage_j" / "refined_finetune_study"
CORE = STUDY / "core_results"
FIGURES = STUDY / "figures" / "paper_results"
PAPER_IMAGES = ROOT.parent / "paper" / "images"

METHODS = [
    ("Guided-DC", "Guided-DC", "#6B7280"),
    ("Guided-GridSFM (released v1.1)", "Model 1: Released", "#D55E00"),
    ("Guided-GridSFM (FullTop-1000)", "Model 2: FullTop-1000", "#6A3D9A"),
    ("Guided-GridSFM (FullTop-1500)", "Model 3: FullTop-1500", "#009E73"),
    ("Guided-GridSFM (FullTop+N-1)", "Model 4: FullTop+N-1", "#CC79A7"),
]
STATE_FAMILIES = ["Pg", "Qg", "V", "Pij", "Qij", "Pji", "Qji"]
INITIALIZERS = [
    ("cold_start", "Cold", "#B7B7B7"),
    ("dc_partial_warm", "DC", "#6B7280"),
    ("gridsfm_m0_full_warm", "Model 1", "#D55E00"),
    ("gridsfm_m1_full_warm", "Model 2", "#6A3D9A"),
    ("gridsfm_m2_full_warm", "Model 3", "#009E73"),
    ("gridsfm_m3_full_warm", "Model 4", "#CC79A7"),
    ("gt_warm", "Exact A", "#242424"),
]


def _filesystem_path(path: Path) -> str:
    resolved = str(path.resolve())
    return rf"\\?\{resolved}" if os.name == "nt" else resolved


def _save(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{stem}.pdf"
    png = FIGURES / f"{stem}.png"
    fig.savefig(_filesystem_path(pdf), bbox_inches="tight", facecolor="white")
    fig.savefig(_filesystem_path(png), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_IMAGES / pdf.name, bbox_inches="tight", facecolor="white")
    fig.savefig(PAPER_IMAGES / png.name, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pdf, png


def plot_state_distance_heatmap() -> tuple[Path, Path]:
    data = pd.read_parquet(CORE / "state_fidelity_all.parquet")
    mean_nrmse = data.pivot_table(
        index="method", columns="metric_family", values="nrmse", aggfunc="mean"
    )
    matrix = np.array(
        [
            [mean_nrmse.loc[method, family] if family in mean_nrmse.columns else np.nan
             for family in STATE_FAMILIES]
            for method, _, _ in METHODS
        ],
        dtype=float,
    )

    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#F1F3F5")
    norm = Normalize(vmin=0.0, vmax=0.85)
    fig, ax = plt.subplots(figsize=(9.2, 4.75))
    image = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(STATE_FAMILIES)), labels=STATE_FAMILIES, fontsize=10)
    ax.set_yticks(
        np.arange(len(METHODS)),
        labels=[label for _, label, _ in METHODS],
        fontsize=9.5,
    )
    for tick, (_, _, color) in zip(ax.get_yticklabels(), METHODS):
        tick.set_color(color)
        tick.set_fontweight("bold")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                ax.text(col, row, "N/A", ha="center", va="center", color="#6B7280", fontsize=9)
                continue
            rgba = cmap(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            text_color = "#111111" if luminance > 0.56 else "white"
            label = f"{value:.3f}" if value >= 0.1 else f"{value:.4f}"
            ax.text(
                col, row, label, ha="center", va="center",
                color=text_color, fontsize=9.2, fontweight="bold",
            )

    ax.set_title("Mean Native-to-Reference-A State Distance", fontsize=12, weight="bold", pad=9)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    colorbar.set_label("Mean normalized RMSE", fontsize=9.5)
    colorbar.ax.tick_params(labelsize=8.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return _save(fig, "stage_j_guided_state_distance_heatmap")


def plot_guided_warm_start_study() -> tuple[Path, Path]:
    data = pd.read_parquet(CORE / "warm_start_crossed.parquet")
    centers = np.arange(len(METHODS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(INITIALIZERS))
    fig, ax = plt.subplots(figsize=(10.4, 5.45))

    for center, (family, _, _) in zip(centers, METHODS):
        family_rows = data.loc[data["finalist_family"].eq(family)]
        for offset, (warm_type, _, color) in zip(offsets, INITIALIZERS):
            values = family_rows.loc[
                family_rows["warm_start_type"].eq(warm_type), "ipopt_solve_time_seconds"
            ].to_numpy(dtype=float)
            result = ax.boxplot(
                [values],
                positions=[center + offset],
                widths=0.075,
                patch_artist=True,
                showfliers=True,
                medianprops={"color": "#171717", "linewidth": 1.15},
                whiskerprops={"color": "#3F3F3F", "linewidth": 0.9},
                capprops={"color": "#3F3F3F", "linewidth": 0.9},
                flierprops={
                    "marker": "o", "markersize": 2.8, "markerfacecolor": "none",
                    "markeredgecolor": color, "alpha": 0.70,
                },
            )
            result["boxes"][0].set(facecolor=color, edgecolor="#3F3F3F", alpha=0.72)

    paper_labels = [
        "Guided-DC",
        "Model 1\nReleased",
        "Model 2\nFullTop-1000",
        "Model 3\nFullTop-1500",
        "Model 4\nFullTop+N-1",
    ]
    ax.set_xticks(centers, labels=paper_labels, fontsize=10.5)
    for tick, (_, _, color) in zip(ax.get_xticklabels(), METHODS):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax.set_ylabel("IPOPT solve time (seconds)", fontsize=12)
    ax.set_xlabel(r"Fixed topology and load-service finalist family $(z,\alpha)$", fontsize=11.5)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.set_title(
        "IPOPT Solve Time by Fixed Reference A Finalist and Initialization",
        fontsize=13,
        weight="bold",
        pad=9,
    )
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        handles=[Patch(facecolor=color, edgecolor="#3F3F3F", alpha=0.72, label=label)
                 for _, label, color in INITIALIZERS],
        loc="upper center",
        bbox_to_anchor=(0.70, 0.985),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#B8B8B8",
        framealpha=0.94,
        fontsize=10,
        columnspacing=0.85,
        handlelength=1.5,
        handletextpad=0.45,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.22)
    return _save(fig, "stage_j_guided_warm_start_study")


if __name__ == "__main__":
    for output in (*plot_state_distance_heatmap(), *plot_guided_warm_start_study()):
        print(output)
