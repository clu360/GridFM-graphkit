"""
Matplotlib plots for Improved Optimization results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_objective_history(history, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    axes[0].plot(history.objective, label="Objective", color="tab:blue")
    axes[0].plot(history.wildfire_cost, label="Wildfire", color="tab:red")
    axes[0].plot(history.load_shedding_cost, label="Shedding", color="tab:green")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Objective and Components")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history.total_shed_mw, label="Total shed MW", color="tab:orange")
    axes[1].plot(history.max_risky_line_loading, label="Max risky loading", color="tab:purple")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Operational Response")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_pareto_frontier(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df[x], df[y], marker="o")
    for _, row in df.iterrows():
        ax.annotate(f"{row['lambda_s']:.2g}", (row[x], row[y]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_sensitivity_curve(df: pd.DataFrame, x: str, y: str, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for group_name, group in df.groupby("metric_group"):
        ax.plot(group[x], group[y], marker="o", label=group_name)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_risky_line_comparison(before_df: pd.DataFrame, after_df: pd.DataFrame, output_path: Path, title: str) -> Path:
    merged = before_df[["line_idx", "loading"]].rename(columns={"loading": "baseline_loading"}).merge(
        after_df[["line_idx", "loading"]].rename(columns={"loading": "optimized_loading"}),
        on="line_idx",
        how="outer",
    ).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(merged))
    ax.bar([i - 0.2 for i in x], merged["baseline_loading"], width=0.4, label="Baseline")
    ax.bar([i + 0.2 for i in x], merged["optimized_loading"], width=0.4, label="Optimized")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["line_idx"].astype(int).tolist(), rotation=45)
    ax.set_xlabel("Line index")
    ax.set_ylabel("Loading")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
