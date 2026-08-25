"""Create compact, reproducible Stage J complete-run summary artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Guided-DC": "#0072B2",
    "Guided-GridSFM (frozen)": "#D55E00",
    "Guided-GridSFM (fine-tuned)": "#6A3D9A",
    "TH-GridSFM-top1": "#009E73",
    "TH-GridSFM-top2": "#CC79A7",
}
METHOD_ORDER = list(METHOD_COLORS)
PARETO_ELIGIBLE_STATUSES = {"ok", "model_output_penalized"}

FROZEN_METHOD_LABELS = {
    "Guided-DC": "Guided-DC",
    "Guided-GridSFM": "Guided-GridSFM (frozen)",
    "TH-GridSFM-top1": "TH-GridSFM-top1",
    "TH-GridSFM-top2": "TH-GridSFM-top2",
}
FT_METHOD_LABELS = {
    "Guided-GridSFM": "Guided-GridSFM (fine-tuned)",
}


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = frame.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _format_lambda(value: float) -> str:
    return f"lambda_R={value:g}"


def build_model_comparison(
    fine_tuned: pd.DataFrame,
    frozen: pd.DataFrame,
) -> pd.DataFrame:
    """Combine controlled Stage J variants without duplicating the TH baseline."""
    frozen_rows = frozen[frozen["method"].isin(FROZEN_METHOD_LABELS)].copy()
    frozen_rows["comparison_variant"] = frozen_rows["method"].map(
        {
            "Guided-DC": "dc",
            "Guided-GridSFM": "frozen",
            "TH-GridSFM-top1": "th_frozen",
            "TH-GridSFM-top2": "th_frozen",
        }
    )
    frozen_rows["method"] = frozen_rows["method"].map(FROZEN_METHOD_LABELS)

    fine_tuned_rows = fine_tuned[fine_tuned["method"].isin(FT_METHOD_LABELS)].copy()
    fine_tuned_rows["comparison_variant"] = "ft"
    fine_tuned_rows["method"] = fine_tuned_rows["method"].map(FT_METHOD_LABELS)
    return pd.concat([frozen_rows, fine_tuned_rows], ignore_index=True, sort=False)


def _save_topology_figure(topologies: pd.DataFrame, scenario: str, figures: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(19, 3.4), sharey=True)
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
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=5, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.24)
    fig.savefig(figures / f"j_trade_by_topology_rank_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _save_convergence_figure(candidates: pd.DataFrame, scenario: str, figures: Path) -> None:
    # Each lambda has a materially different J_trade range, so a shared axis
    # hides the within-run improvements at the lower objective weights.
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
        axis.set_ylabel("Best J_trade seen")
        axis.grid(alpha=0.2)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.suptitle(f"{scenario}: within-run candidate-evaluation convergence", y=0.98)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=5, frameon=False, fontsize=8)
    fig.subplots_adjust(top=0.72, bottom=0.17, wspace=0.24)
    fig.savefig(figures / f"j_trade_convergence_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_evaluated_pareto_frontiers(candidates: pd.DataFrame) -> pd.DataFrame:
    """Return per-scenario, per-method native fronts from all finite evaluations."""
    finite = candidates[
        candidates["method"].isin(METHOD_ORDER)
        & candidates["evaluation_status"].astype(str).str.lower().isin(PARETO_ELIGIBLE_STATUSES)
        & np.isfinite(candidates["r_norm"])
        & np.isfinite(candidates["l_shed_total"])
    ].copy()
    sort_columns = [
        "scenario_id", "method", "l_shed_total", "r_norm", "lambda_r",
        "topology_rank", "candidate_index",
    ]
    finite = finite.sort_values(sort_columns, kind="stable")

    fronts: list[pd.DataFrame] = []
    for (_, _), group in finite.groupby(["scenario_id", "method"], sort=False):
        evaluated_count = len(group)
        unique_coordinates = group.drop_duplicates(
            subset=["l_shed_total", "r_norm"], keep="first"
        )
        # At a fixed load-shedding value, only the lowest-risk point can survive.
        ordered = unique_coordinates.drop_duplicates(
            subset=["l_shed_total"], keep="first"
        ).sort_values(["l_shed_total", "r_norm"], kind="stable")
        running_best = ordered["r_norm"].cummin().shift(fill_value=np.inf)
        front = ordered[ordered["r_norm"] < running_best].copy()
        front["pareto_order"] = np.arange(1, len(front) + 1)
        front["evaluated_candidate_count"] = evaluated_count
        front["unique_risk_load_coordinate_count"] = len(unique_coordinates)
        front["pareto_front_size"] = len(front)
        front["frontier_scope"] = "scenario_method_all_lambda_evaluated_native"
        fronts.append(front)

    if not fronts:
        return finite.iloc[0:0].copy()
    return pd.concat(fronts, ignore_index=True, sort=False)


def _validate_pareto_frontiers(candidates: pd.DataFrame, frontiers: pd.DataFrame) -> None:
    finite = candidates[
        candidates["method"].isin(METHOD_ORDER)
        & candidates["evaluation_status"].astype(str).str.lower().isin(PARETO_ELIGIBLE_STATUSES)
        & np.isfinite(candidates["r_norm"])
        & np.isfinite(candidates["l_shed_total"])
    ]
    expected_groups = set(map(tuple, finite[["scenario_id", "method"]].drop_duplicates().to_numpy()))
    observed_groups = set(map(tuple, frontiers[["scenario_id", "method"]].drop_duplicates().to_numpy()))
    if observed_groups != expected_groups:
        raise ValueError(f"Pareto group mismatch: expected={expected_groups}, observed={observed_groups}")

    for key, frontier in frontiers.groupby(["scenario_id", "method"], sort=False):
        pool = finite[(finite["scenario_id"] == key[0]) & (finite["method"] == key[1])]
        for row in frontier.itertuples(index=False):
            weakly_better = (
                (pool["l_shed_total"] <= row.l_shed_total)
                & (pool["r_norm"] <= row.r_norm)
            )
            strictly_better = (
                (pool["l_shed_total"] < row.l_shed_total)
                | (pool["r_norm"] < row.r_norm)
            )
            if (weakly_better & strictly_better).any():
                raise ValueError(f"Dominated point retained in Pareto frontier for {key}")


def _save_pareto_figure(frontiers: pd.DataFrame, scenario: str, figures: Path) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.8))
    subset = frontiers[frontiers["scenario_id"] == scenario]
    for method in METHOD_ORDER:
        rows = subset[subset["method"] == method].sort_values("pareto_order")
        if rows.empty:
            continue
        color = METHOD_COLORS[method]
        if len(rows) == 1:
            axis.scatter(
                rows["l_shed_total"], rows["r_norm"], s=58, marker="D",
                color=color, edgecolor="white", linewidth=0.7, label=method, zorder=4,
            )
        else:
            axis.plot(
                rows["l_shed_total"], rows["r_norm"], marker="o", markersize=3.8,
                linewidth=1.8, color=color, label=method,
            )
    axis.set_title(f"{scenario}: Evaluated Native Pareto Frontiers", fontsize=13)
    axis.set_xlabel("L_shed_total")
    axis.set_ylabel("R_norm")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / f"evaluated_risk_load_scatter_{scenario.lower()}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run",
    )
    parser.add_argument(
        "--comparison-root",
        help="Frozen/DC complete-run root to merge into a fine-tuned result package.",
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
        ["topology_rank", "candidate_index", "j_trade", "r_norm", "l_shed_total", "lambda_r"],
    )
    topology["best_found"] = topology["best_found"].astype(str).str.lower().eq("true")

    if args.comparison_root:
        comparison_core = Path(args.comparison_root).resolve() / "core_results"
        frozen_topology = _numeric(
            pd.read_csv(comparison_core / "topology_objectives_all.csv"),
            ["topology_rank", "j_trade", "r_norm", "l_shed_total", "lambda_r"],
        )
        frozen_candidates = _numeric(
            pd.read_csv(comparison_core / "candidate_evaluations_all.csv"),
            ["topology_rank", "candidate_index", "j_trade", "r_norm", "l_shed_total", "lambda_r"],
        )
        frozen_topology["best_found"] = (
            frozen_topology["best_found"].astype(str).str.lower().eq("true")
        )
        topology = build_model_comparison(topology, frozen_topology)
        candidates = build_model_comparison(candidates, frozen_candidates)
        topology.to_csv(core / "compare_topology.csv", index=False)
        candidates.to_csv(core / "compare_candidates.csv", index=False)

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

    frontiers = build_evaluated_pareto_frontiers(candidates)
    _validate_pareto_frontiers(candidates, frontiers)
    derived = core / "derived_visual_summaries"
    derived.mkdir(parents=True, exist_ok=True)
    frontiers.to_csv(derived / "evaluated_native_pareto_frontiers.csv", index=False)

    for scenario in sorted(topology["scenario_id"].dropna().unique()):
        _save_topology_figure(topology, scenario, figures)
        _save_convergence_figure(candidates, scenario, figures)
        _save_pareto_figure(frontiers, scenario, figures)


if __name__ == "__main__":
    main()
