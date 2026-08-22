"""Create compact, reproducible Stage J complete-run summary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {
    "Guided-DC": "#0072B2",
    "Guided-GridSFM": "#D55E00",
    "TH-GridSFM-top1": "#009E73",
    "TH-GridSFM-top2": "#CC79A7",
}
METHOD_ORDER = list(METHOD_COLORS)


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _format_lambda(value: float) -> str:
    return f"lambda_R={value:g}"


def _save_topology_figure(topologies: pd.DataFrame, scenario: str, figures: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=False)
    for axis, lambda_r in zip(axes, sorted(topologies["lambda_r"].dropna().unique())):
        subset = topologies[(topologies["scenario_id"] == scenario) & (topologies["lambda_r"] == lambda_r)]
        for method in METHOD_ORDER:
            rows = subset[(subset["method"] == method) & (subset["best_found"])]
            if rows.empty:
                continue
            axis.scatter(
                rows["topology_rank"], rows["j_trade"], s=11,
                color=METHOD_COLORS[method], alpha=0.72, label=method,
            )
        axis.set_title(_format_lambda(lambda_r), fontsize=10)
        axis.set_xlabel("Topology rank")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("J_trade")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(f"{scenario}: common wildfire/service objective by topology", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.24)
    fig.savefig(figures / f"j_trade_by_topology_rank_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_figure(candidates: pd.DataFrame, scenario: str, figures: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=False)
    for axis, lambda_r in zip(axes, sorted(candidates["lambda_r"].dropna().unique())):
        subset = candidates[(candidates["scenario_id"] == scenario) & (candidates["lambda_r"] == lambda_r)]
        for method in METHOD_ORDER:
            rows = subset[(subset["method"] == method) & subset["j_trade"].notna()].copy()
            if rows.empty:
                continue
            rows["evaluation_order"] = (rows["topology_rank"] - 1) * 20 + rows["candidate_index"]
            rows = rows.sort_values("evaluation_order")
            axis.plot(
                rows["evaluation_order"], rows["j_trade"].cummin(), linewidth=1.1,
                color=METHOD_COLORS[method], label=method,
            )
        axis.set_title(_format_lambda(lambda_r), fontsize=10)
        axis.set_xlabel("Candidate evaluation")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Best J_trade seen")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(f"{scenario}: within-run candidate-evaluation convergence", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.24)
    fig.savefig(figures / f"j_trade_convergence_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_pareto_figure(topologies: pd.DataFrame, scenario: str, figures: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharex=False, sharey=False)
    for axis, lambda_r in zip(axes, sorted(topologies["lambda_r"].dropna().unique())):
        subset = topologies[(topologies["scenario_id"] == scenario) & (topologies["lambda_r"] == lambda_r)]
        for method in METHOD_ORDER:
            rows = subset[(subset["method"] == method) & (subset["best_found"])]
            if rows.empty:
                continue
            axis.scatter(rows["l_shed_total"], rows["r_norm"], s=13, color=METHOD_COLORS[method], alpha=0.72, label=method)
        axis.set_title(_format_lambda(lambda_r), fontsize=10)
        axis.set_xlabel("L_shed_total")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("R_norm")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(f"{scenario}: evaluated topology outcomes", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=4, frameon=False)
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.24)
    fig.savefig(figures / f"evaluated_risk_load_scatter_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run",
    )
    args = parser.parse_args()
    root = Path(args.result_root).resolve()
    core = root / "core_results"
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    topology = _numeric(
        pd.read_csv(core / "topology_objectives_all.csv"),
        ["topology_rank", "j_trade", "r_norm", "l_shed_total", "lambda_r"],
    )
    candidates = _numeric(
        pd.read_csv(core / "candidate_evaluations_all.csv"),
        ["topology_rank", "candidate_index", "j_trade", "lambda_r"],
    )
    topology["best_found"] = topology["best_found"].astype(str).str.lower().eq("true")

    finalists = _numeric(
        pd.read_csv(core / "method_finalists_all.csv"),
        ["lambda_r", "j_trade", "r_norm", "l_shed_total", "num_shutoffs", "pac_total", "j_total"],
    )
    finalists.to_csv(core / "finalist_outcomes_all.csv", index=False)

    runtime = candidates.groupby(["scenario_id", "lambda_r", "method"], dropna=False).agg(
        candidate_evaluations=("candidate_index", "size"),
        candidate_runtime_seconds=("runtime_seconds", "sum"),
        best_j_trade=("j_trade", "min"),
    ).reset_index()
    runtime.to_csv(core / "candidate_runtime_summary.csv", index=False)

    for scenario in sorted(topology["scenario_id"].dropna().unique()):
        _save_topology_figure(topology, scenario, figures)
        _save_convergence_figure(candidates, scenario, figures)
        _save_pareto_figure(topology, scenario, figures)


if __name__ == "__main__":
    main()
