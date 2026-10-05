"""Build the evidence-backed Stage K Texas2k post-run analysis package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EVALUATORS = ("gridsfm", "dc", "ac")
LABELS = {"gridsfm": "GridSFM", "dc": "DC-OPF", "ac": "AC-OPF"}
COLORS = {"gridsfm": "#2878B5", "dc": "#E07A1F", "ac": "#2A9D6F"}


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _read(run: Path, relative: str) -> pd.DataFrame:
    return pd.read_parquet(run / relative)


def _offline_ids(value: object) -> list[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    text = text.strip("[]")
    return [int(part.strip()) for part in text.replace(",", ";").split(";") if part.strip()]


def _native_tables(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates, finalists = [], []
    for evaluator in EVALUATORS:
        candidate = _read(run, f"evaluators/{evaluator}_candidate_results.parquet")
        candidate["evaluator"] = evaluator
        candidate["candidate_order"] = np.arange(len(candidate))
        candidates.append(candidate)
        finalist = _read(run, f"evaluators/{evaluator}_finalists.parquet")
        finalist["evaluator"] = evaluator
        finalists.append(finalist)
    return pd.concat(candidates, ignore_index=True), pd.concat(finalists, ignore_index=True)


def _environment(run: Path, tables: Path, figures: Path) -> pd.DataFrame:
    branch = _read(run, "prepared/canonical_branch.parquet")
    active = branch.loc[branch["is_switchable_transmission"].astype(bool)].copy()
    active["linear_risk"] = active["p_env"] * active["baseline_loading"]
    active["squared_risk"] = active["p_env"] * active["baseline_loading"].pow(2)
    active.to_parquet(tables / "environment_branch_metrics.parquet", index=False)
    summary = active[["p_env", "baseline_loading", "linear_risk", "squared_risk"]].describe(
        percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]
    ).T.reset_index(names="metric")
    summary.to_csv(tables / "environment_distribution_summary.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, column, title in zip(
        axes.flat,
        ("p_env", "baseline_loading", "linear_risk", "squared_risk"),
        ("Cumulative weather hazard", "Scenario 16 loading", "Linear risk", "Squared-loading risk"),
    ):
        ax.hist(active[column].dropna(), bins=45, color="#3E6B89", edgecolor="white", linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Transmission lines")
        ax.grid(axis="y", alpha=0.2)
    _save(fig, figures, "01_environment_distributions")
    return active


def _candidate_figures(candidates: pd.DataFrame, finalists: pd.DataFrame, tables: Path, figures: Path) -> None:
    candidates.to_parquet(tables / "all_native_candidates.parquet", index=False)
    finalists.to_csv(tables / "native_finalists.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    for ax, evaluator in zip(axes, EVALUATORS):
        rows = candidates.loc[(candidates.evaluator == evaluator) & candidates.eligible]
        ax.scatter(rows.l_shed_total, rows.r_norm, s=10, alpha=0.22, color=COLORS[evaluator])
        selected = finalists.loc[finalists.evaluator == evaluator]
        ax.scatter(selected.l_shed_total, selected.r_norm, s=95, marker="*", color="#D62728", edgecolor="black")
        for row in selected.itertuples():
            ax.annotate(f"{row.lambda_r:g}", (row.l_shed_total, row.r_norm), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_title(LABELS[evaluator])
        ax.set_xlabel("Native load shedding fraction")
        ax.set_ylabel("Native normalized risk")
        ax.grid(alpha=0.2)
    _save(fig, figures, "02_native_risk_service_candidates")

    conv_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for ax, evaluator in zip(axes, EVALUATORS):
        for lambda_r, group in candidates.loc[candidates.evaluator == evaluator].groupby("lambda_r"):
            group = group.sort_values("candidate_order")
            values = group.search_objective.where(group.eligible, np.nan).cummin()
            ax.plot(np.arange(1, len(group) + 1), values, label=f"lambda={lambda_r:g}")
            conv_rows.extend({"evaluator": evaluator, "lambda_r": lambda_r, "attempt": i + 1, "best_objective": value}
                             for i, value in enumerate(values))
        ax.set_title(LABELS[evaluator])
        ax.set_ylabel("Best native objective")
        ax.grid(alpha=0.2)
        ax.legend(ncol=5, fontsize=8, frameon=False)
    axes[-1].set_xlabel("Candidate evaluations")
    pd.DataFrame(conv_rows).to_parquet(tables / "best_so_far_convergence.parquet", index=False)
    _save(fig, figures, "03_best_so_far_convergence")


def _reference_figures(run: Path, finalists: pd.DataFrame, tables: Path, figures: Path) -> pd.DataFrame:
    ref_a = _read(run, "references/reference_a.parquet")
    ref_b = _read(run, "references/reference_b.parquet")
    delta = _read(run, "references/reference_discrepancies.parquet")
    merged = finalists.merge(ref_a, on=["evaluator", "lambda_r", "topology_key"]).merge(
        ref_b, on=["evaluator", "lambda_r", "topology_key"]
    ).merge(
        delta[["evaluator", "lambda_r", "topology_key", "delta_r_a", "delta_j_a"]],
        on=["evaluator", "lambda_r", "topology_key"],
    )
    merged["risk_change_b_minus_a"] = merged["r_norm_reference_b"] - merged["r_norm_reference_a"]
    merged["risk_change_b_vs_a_percent"] = (
        100.0 * merged["risk_change_b_minus_a"] / merged["r_norm_reference_a"]
    )
    merged.to_parquet(tables / "finalist_native_and_ac_references.parquet", index=False)
    merged.to_csv(tables / "finalist_native_and_ac_references.csv", index=False)
    merged[[
        "evaluator", "lambda_r", "topology_key", "r_norm_reference_a",
        "r_norm_reference_b", "risk_change_b_minus_a", "risk_change_b_vs_a_percent",
        "l_shed_reference_a", "maximum_service_fraction",
    ]].to_csv(tables / "reference_b_risk_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    comparisons = [
        ("r_norm", "r_norm_reference_a", "Normalized risk"),
        ("j_trade", "j_trade_reference_a", "Tradeoff objective"),
        ("l_shed_total", "l_shed_reference_a", "Load shedding fraction"),
    ]
    for ax, (native, exact, title) in zip(axes, comparisons):
        for evaluator in EVALUATORS:
            rows = merged.loc[merged.evaluator == evaluator]
            ax.scatter(rows[native], rows[exact], label=LABELS[evaluator], color=COLORS[evaluator], s=48)
        bounds = [np.nanmin([merged[native].min(), merged[exact].min()]), np.nanmax([merged[native].max(), merged[exact].max()])]
        ax.plot(bounds, bounds, "--", color="#555555", linewidth=1)
        ax.set_xlabel(f"Native {title.lower()}")
        ax.set_ylabel(f"Exact AC Reference A {title.lower()}")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    _save(fig, figures, "04_reference_a_native_vs_exact")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for evaluator in EVALUATORS:
        rows = merged.loc[merged.evaluator == evaluator].sort_values("lambda_r")
        axes[0].plot(rows.lambda_r, rows.selected_service_fraction, "o-", color=COLORS[evaluator], label=LABELS[evaluator])
        axes[0].plot(rows.lambda_r, rows.maximum_service_fraction, "--", color=COLORS[evaluator], alpha=0.65)
        axes[1].plot(rows.lambda_r, rows.delta_s_b, "o-", color=COLORS[evaluator], label=LABELS[evaluator])
        axes[2].plot(rows.lambda_r, rows.economic_cost_tiebreak, "o-", color=COLORS[evaluator], label=LABELS[evaluator])
    axes[0].set_title("Selected service (solid) vs B1 maximum (dashed)")
    axes[0].set_ylabel("Service fraction")
    axes[1].set_title("Recoverable service under Reference B1")
    axes[1].set_ylabel("Delta service fraction")
    axes[2].set_title("Reference B2 economic tie-break")
    axes[2].set_ylabel("Generation cost at maximum service")
    for ax in axes:
        ax.set_xlabel("Risk weight lambda_R")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    _save(fig, figures, "05_reference_b_service_certification")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for evaluator in EVALUATORS:
        rows = merged.loc[merged.evaluator == evaluator].sort_values("lambda_r")
        axes[0].plot(rows.lambda_r, rows.r_norm_reference_a, "o-", color=COLORS[evaluator], label=LABELS[evaluator])
        axes[1].plot(rows.lambda_r, rows.l_shed_reference_a, "o-", color=COLORS[evaluator])
        axes[2].plot(rows.lambda_r, rows.economic_cost, "o-", color=COLORS[evaluator])
    for ax, title, ylabel in zip(axes, ("Exact risk", "Selected service loss", "Generation cost"),
                                ("Reference A normalized risk", "Reference A shedding fraction", "Reference A cost")):
        ax.set_title(title); ax.set_xlabel("Risk weight lambda_R"); ax.set_ylabel(ylabel); ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    _save(fig, figures, "06_exact_ac_tradeoffs")
    return merged


def _topology_figures(run: Path, finalists: pd.DataFrame, branch: pd.DataFrame, tables: Path, figures: Path) -> None:
    opened = []
    for row in finalists.itertuples():
        ids = _offline_ids(row.offline_branch_ids)
        for branch_id in ids:
            opened.append({"evaluator": row.evaluator, "lambda_r": row.lambda_r, "topology_key": row.topology_key,
                           "k": row.k, "canonical_branch_id": branch_id})
    opened = pd.DataFrame(opened)
    opened = opened.merge(branch, on="canonical_branch_id", how="left") if len(opened) else opened
    opened.to_csv(tables / "finalist_opened_lines.csv", index=False)
    counts = finalists.groupby(["evaluator", "k"]).size().rename("finalist_count").reset_index()
    counts.to_csv(tables / "finalist_open_line_counts.csv", index=False)

    k1 = _read(run, "report/derived/k1_ranking_parent_divergence.parquet")
    k1.to_csv(tables / "k1_ranking_parent_divergence.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (left, right), group in k1.groupby(["evaluator_left", "evaluator_right"]):
        label = f"{LABELS[left]} vs {LABELS[right]}"
        axes[0].plot(group.lambda_r, group.spearman_rank_correlation, "o-", label=label)
        axes[1].plot(group.lambda_r, group.top5_jaccard, "o-", label=label)
    axes[0].set_title("Shared K=1 rank agreement"); axes[0].set_ylabel("Spearman correlation")
    axes[1].set_title("K=2 parent-set agreement"); axes[1].set_ylabel("Top-5 Jaccard")
    for ax in axes:
        ax.set_xlabel("Risk weight lambda_R"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.2); ax.legend(fontsize=8, frameon=False)
    _save(fig, figures, "07_topology_search_divergence")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    base = branch.loc[branch.is_switchable_transmission.astype(bool)]
    for ax, evaluator in zip(axes, EVALUATORS):
        ax.scatter(base.from_lon, base.from_lat, c=base.squared_risk, cmap="YlOrRd", s=3, alpha=0.16, rasterized=True)
        rows = opened.loc[opened.evaluator == evaluator]
        for item in rows.itertuples():
            ax.plot([item.from_lon, item.to_lon], [item.from_lat, item.to_lat], color=COLORS[evaluator], linewidth=2.3, alpha=0.75)
            ax.text((item.from_lon + item.to_lon) / 2, (item.from_lat + item.to_lat) / 2,
                    str(item.canonical_branch_id), fontsize=7, color="#111111")
        ax.set_title(LABELS[evaluator]); ax.set_xlabel("Longitude"); ax.grid(alpha=0.12)
    axes[0].set_ylabel("Latitude")
    _save(fig, figures, "08_texas_finalist_opened_lines")


def _operations_figures(run: Path, candidates: pd.DataFrame, tables: Path, figures: Path) -> None:
    status = candidates.groupby(["evaluator", "status", "eligible"]).size().rename("count").reset_index()
    status.to_csv(tables / "solver_status_accounting.csv", index=False)
    runtime = candidates.groupby("evaluator").agg(
        attempts=("topology_key", "size"), eligible=("eligible", "sum"),
        elapsed_total_seconds=("elapsed_seconds", "sum"), elapsed_median_seconds=("elapsed_seconds", "median"),
        elapsed_mean_seconds=("elapsed_seconds", "mean"), solver_mean_seconds=("solver_seconds", "mean"),
        iterations_mean=("iterations", "mean"),
    ).reset_index()
    runtime.to_csv(tables / "runtime_summary.csv", index=False)
    resources = _read(run, "report/derived/job_resource_summary.parquet")
    resources.to_csv(tables / "job_resource_summary.csv", index=False)
    native_resources = resources.loc[resources.job.str.match(r"^(gridsfm|dc|ac)_lambda")].copy()
    native_resources["evaluator"] = native_resources.job.str.extract(r"^(gridsfm|dc|ac)")
    memory = native_resources.groupby("evaluator").peak_memory_mb.max().reindex(EVALUATORS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].bar([LABELS[x] for x in runtime.evaluator], runtime.elapsed_median_seconds,
                color=[COLORS[x] for x in runtime.evaluator])
    axes[0].set_yscale("log"); axes[0].set_ylabel("Median elapsed seconds (log scale)"); axes[0].set_title("Per-candidate runtime")
    eligible = candidates.groupby("evaluator").eligible.agg(["sum", "count"])
    axes[1].bar([LABELS[x] for x in eligible.index], eligible["sum"], color=[COLORS[x] for x in eligible.index])
    axes[1].bar([LABELS[x] for x in eligible.index], eligible["count"] - eligible["sum"], bottom=eligible["sum"], color="#C44E52", label="Ineligible")
    axes[1].set_title("Candidate convergence accounting"); axes[1].set_ylabel("Candidates"); axes[1].legend(frameon=False)
    axes[2].bar([LABELS[x] for x in memory.index], memory.values / 1024.0,
                color=[COLORS[x] for x in memory.index])
    axes[2].set_title("Maximum observed native-job memory")
    axes[2].set_ylabel("Peak resident memory (GiB)")
    for ax in axes: ax.grid(axis="y", alpha=0.2)
    _save(fig, figures, "09_runtime_and_convergence")

    alpha = _read(run, "evaluators/gridsfm_alpha_evaluations.parquet")
    alpha_summary = alpha.groupby(["lambda_r", "evaluation_index"]).objective.agg(["median", "mean", "min", "max"]).reset_index()
    alpha_summary.to_csv(tables / "gridsfm_alpha_search_summary.csv", index=False)
    grid = candidates.loc[candidates.evaluator == "gridsfm"]
    grid[["lambda_r", "topology_key", "j_trade", "pac_operational", "pac_ac", "pac_model", "pac_total", "j_total"]].to_parquet(
        tables / "gridsfm_pac_accounting.parquet", index=False
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for lambda_r, rows in alpha_summary.groupby("lambda_r"):
        axes[0].plot(rows.evaluation_index, rows["median"], label=f"lambda={lambda_r:g}")
    axes[0].set_title("GridSFM 20-evaluation alpha recourse"); axes[0].set_xlabel("Alpha evaluation"); axes[0].set_ylabel("Median objective")
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].scatter(grid.j_trade, grid.pac_total, s=10, alpha=0.3, color=COLORS["gridsfm"])
    axes[1].set_title("GridSFM physical-admissibility penalty"); axes[1].set_xlabel("J_trade"); axes[1].set_ylabel("PAC_total")
    for ax in axes: ax.grid(alpha=0.2)
    _save(fig, figures, "10_gridsfm_alpha_and_pac")


def analyze(run_dir: Path, output_dir: Path) -> None:
    tables, figures = output_dir / "tables", output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    candidates, finalists = _native_tables(run_dir)
    branch = _environment(run_dir, tables, figures)
    _candidate_figures(candidates, finalists, tables, figures)
    merged = _reference_figures(run_dir, finalists, tables, figures)
    _topology_figures(run_dir, finalists, branch, tables, figures)
    _operations_figures(run_dir, candidates, tables, figures)
    summary = {
        "candidate_attempts": int(len(candidates)), "eligible_candidates": int(candidates.eligible.sum()),
        "finalists": int(len(finalists)), "reference_a_rows": int(len(merged)), "reference_b_rows": int(len(merged)),
        "b2_failures": int((~merged.b2_solver_eligible.astype(bool)).sum()),
        "unique_finalist_topologies": int(finalists.topology_key.nunique()),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); analyze(args.run_dir, args.output_dir); return 0


if __name__ == "__main__":
    raise SystemExit(main())
