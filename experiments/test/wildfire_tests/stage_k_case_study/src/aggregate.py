"""Derived Stage K smoke tables, diagnostic nondominance, and figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io_utils import atomic_write_json
from .schemas import ELIGIBLE_STATUSES, classify_solver_status
from .config import load_config


def empirical_nondominated(frame: pd.DataFrame) -> pd.Series:
    """Return nondominated mask for jointly minimized risk and load shedding."""

    values = frame[["r_norm", "l_shed_total"]].to_numpy(dtype=float)
    mask = np.isfinite(values).all(axis=1)
    result = np.zeros(len(frame), dtype=bool)
    for i, point in enumerate(values):
        if not mask[i]:
            continue
        dominated = np.any(mask & np.all(values <= point, axis=1) & np.any(values < point, axis=1))
        result[i] = not dominated
    return pd.Series(result, index=frame.index)


def _k1_divergence(candidates: pd.DataFrame) -> pd.DataFrame:
    k1 = candidates.loc[(candidates["k"] == 1) & candidates["eligible"]].copy()
    rows = []
    for lambda_r, group in k1.groupby("lambda_r"):
        pivot = group.pivot(index="topology_key", columns="evaluator", values="search_objective")
        ranks = pivot.rank(method="min")
        methods = sorted(ranks.columns)
        for i, left in enumerate(methods):
            for right in methods[i + 1 :]:
                valid = ranks[[left, right]].dropna()
                left_top = set(valid.nsmallest(5, left).index)
                right_top = set(valid.nsmallest(5, right).index)
                rows.append(
                    {
                        "lambda_r": lambda_r,
                        "evaluator_left": left,
                        "evaluator_right": right,
                        "spearman_rank_correlation": valid[left].corr(valid[right], method="spearman"),
                        "top5_overlap_count": len(left_top & right_top),
                        "top5_union_count": len(left_top | right_top),
                        "top5_jaccard": len(left_top & right_top) / max(len(left_top | right_top), 1),
                        "left_only_parents": ";".join(sorted(left_top - right_top)),
                        "right_only_parents": ";".join(sorted(right_top - left_top)),
                    }
                )
    return pd.DataFrame(rows)


def _k1_rank_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (lambda_r, evaluator), group in candidates.loc[
        (candidates["k"] == 1) & candidates["eligible"]
    ].groupby(["lambda_r", "evaluator"]):
        ordered = group.sort_values(["search_objective", "topology_key"]).copy()
        ordered["native_rank"] = np.arange(1, len(ordered) + 1)
        ordered["selected_as_k2_parent"] = ordered["native_rank"] <= 5
        rows.append(ordered[[
            "lambda_r", "evaluator", "topology_key", "search_objective",
            "r_norm", "l_shed_total", "native_rank", "selected_as_k2_parent",
        ]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _parse_gnu_time(path: Path) -> dict[str, object]:
    row: dict[str, object] = {"job": path.name.replace("_job_time.txt", "")}
    if not path.exists():
        return row
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        value = value.strip()
        if key == "Maximum resident set size (kbytes)":
            row["peak_memory_mb"] = float(value) / 1024.0
        elif key == "Percent of CPU this job got":
            row["cpu_percent"] = float(value.rstrip("%"))
        elif key == "User time (seconds)":
            row["user_cpu_seconds"] = float(value)
        elif key == "System time (seconds)":
            row["system_cpu_seconds"] = float(value)
    return row


def full_run_workload_ratio(evaluator: str) -> dict[str, float]:
    """Scale smoke timing by topology count and evaluator-specific inner work."""

    topology_ratio = (301 * 5) / 21
    inner_ratio = 20 / 5 if evaluator == "gridsfm" else 1.0
    return {
        "topology_ratio_full_to_smoke": topology_ratio,
        "inner_evaluation_ratio_full_to_smoke": inner_ratio,
        "workload_ratio_full_to_smoke": topology_ratio * inner_ratio,
    }


def aggregate(
    run_dir: str | Path, output_dir: str | Path, config_path: str | Path | None = None
) -> dict[str, object]:
    run = Path(run_dir)
    output = Path(output_dir)
    config = load_config(config_path) if config_path is not None else None
    is_full = bool(config and config.get("mode") == "full")
    derived = output / "derived"
    figures = output / "figures"
    derived.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    candidate_frames = []
    for evaluator in ("gridsfm", "dc", "ac"):
        path = run / f"{evaluator}_candidate_results.parquet"
        frame = pd.read_parquet(path)
        frame["evaluator"] = evaluator
        candidate_frames.append(frame)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    candidates["empirical_nondominated"] = False
    for (_, evaluator), idx in candidates.groupby(["lambda_r", "evaluator"]).groups.items():
        candidates.loc[idx, "empirical_nondominated"] = empirical_nondominated(candidates.loc[idx]).to_numpy()
    candidates.to_parquet(derived / "native_candidate_comparison.parquet", index=False)
    candidates.loc[candidates["empirical_nondominated"]].to_parquet(derived / "nondominated_points.parquet", index=False)

    divergence = _k1_divergence(candidates)
    divergence.to_parquet(derived / "k1_ranking_parent_divergence.parquet", index=False)
    k1_ranks = _k1_rank_table(candidates)
    k1_ranks.to_parquet(derived / "common_k1_evaluator_rankings.parquet", index=False)
    k2_frames = []
    finalist_frames = []
    for evaluator in ("gridsfm", "dc", "ac"):
        k2_path = run / f"{evaluator}_k2_candidates.parquet"
        if k2_path.exists():
            table = pd.read_parquet(k2_path)
            table["evaluator"] = evaluator
            k2_frames.append(table)
        finalist_path = run / f"{evaluator}_finalists.parquet"
        if finalist_path.exists():
            table = pd.read_parquet(finalist_path)
            table["evaluator"] = evaluator
            finalist_frames.append(table)
    if k2_frames:
        pd.concat(k2_frames, ignore_index=True).to_parquet(derived / "k2_search_path_provenance.parquet", index=False)
    if finalist_frames:
        pd.concat(finalist_frames, ignore_index=True).to_parquet(derived / "finalist_comparison.parquet", index=False)
    failures = (
        candidates.groupby(["evaluator", "lambda_r", "k", "status"], dropna=False)
        .size().rename("count").reset_index()
    )
    failures.to_parquet(derived / "convergence_failures.parquet", index=False)
    runtime = (
        candidates.groupby(["evaluator", "lambda_r", "k"], dropna=False)
        .agg(
            attempted=("topology_key", "size"),
            elapsed_mean_seconds=("elapsed_seconds", "mean"),
            elapsed_median_seconds=("elapsed_seconds", "median"),
            solver_mean_seconds=("solver_seconds", "mean"),
            iterations_mean=("iterations", "mean"),
            peak_memory_max_mb=("peak_memory_mb", "max"),
        ).reset_index()
    )
    runtime.to_parquet(derived / "runtime_resource_summary.parquet", index=False)
    runtime_dir = run.parent / "runtime"
    job_resources = [_parse_gnu_time(path) for path in sorted(runtime_dir.glob("*_job_time.txt"))]
    if job_resources:
        pd.DataFrame(job_resources).to_parquet(derived / "job_resource_summary.parquet", index=False)
    sacct_path = runtime_dir / "slurm_sacct.psv"
    if sacct_path.exists() and sacct_path.stat().st_size > 0:
        pd.read_csv(sacct_path, sep="|").to_parquet(derived / "slurm_resource_accounting.parquet", index=False)

    prepared_manifest = run.parent / "prepared" / "input_manifest.json"
    if prepared_manifest.exists():
        baseline = json.loads(prepared_manifest.read_text(encoding="utf-8"))
        atomic_write_json(derived / "baseline_summary.json", {
            "baseline_authority": baseline["baseline_authority"],
            "baseline_replaced_by_new_ac_solve": baseline["baseline_replaced_by_new_ac_solve"],
            "r_base": baseline["r_base"],
            "counts": baseline["counts"],
        })

    reference_root = run.parent / "references"
    reference_a_path = reference_root / "reference_a.parquet"
    reference_b_path = reference_root / "reference_b.parquet"
    reference_path = reference_root / "reference_discrepancies.parquet"
    if reference_a_path.exists():
        pd.read_parquet(reference_a_path).to_parquet(derived / "reference_a.parquet", index=False)
    b2_failure_count = 0
    if reference_b_path.exists():
        reference_b = pd.read_parquet(reference_b_path)
        reference_b.to_parquet(derived / "reference_b.parquet", index=False)
        b2_failure_count = int((
            ~reference_b["b2_status"].astype(str).map(
                lambda value: classify_solver_status(value) in ELIGIBLE_STATUSES
            )
        ).sum())
    if reference_path.exists():
        pd.read_parquet(reference_path).to_parquet(derived / "reference_discrepancies.parquet", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"gridsfm": "#276FBF", "dc": "#F28E2B", "ac": "#279E68"}
    for evaluator, group in candidates.loc[candidates["eligible"]].groupby("evaluator"):
        ax.scatter(group["l_shed_total"], group["r_norm"], s=24, alpha=0.45, label=evaluator, color=colors[evaluator])
        front = group.loc[group["empirical_nondominated"]].sort_values("l_shed_total")
        ax.plot(front["l_shed_total"], front["r_norm"], color=colors[evaluator], linewidth=1.3)
    if finalist_frames:
        finalists = pd.concat(finalist_frames, ignore_index=True)
        for evaluator, group in finalists.groupby("evaluator"):
            ax.scatter(
                group["l_shed_total"], group["r_norm"], marker="*", s=150,
                color=colors[evaluator], edgecolor="black", linewidth=0.6, zorder=5,
            )
    ax.set_xlabel("Load shedding fraction")
    ax.set_ylabel("Normalized wildfire risk")
    ax.set_title(
        "Empirical nondominated set, full five-lambda production"
        if is_full else "Diagnostic empirical nondominated set, smoke lambda_R=0.8"
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "risk_service_nondominated.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(divergence):
        labels = [f"{row.evaluator_left} vs {row.evaluator_right}" for row in divergence.itertuples(index=False)]
        positions = np.arange(len(divergence))
        ax.bar(positions - 0.18, divergence["spearman_rank_correlation"], width=0.36, label="Spearman rank")
        ax.bar(positions + 0.18, divergence["top5_jaccard"], width=0.36, label="Top-5 Jaccard")
        ax.set_xticks(positions, labels, rotation=20, ha="right")
        ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Agreement")
    ax.set_title("Shared K1 ranking and selected-parent agreement")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "k1_rank_parent_divergence.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    status_pivot = failures.pivot_table(index="evaluator", columns="status", values="count", aggfunc="sum", fill_value=0)
    status_pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Candidate attempts")
    ax.set_xlabel("")
    ax.set_title("Convergence and failure accounting")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "convergence_failures.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for evaluator, group in candidates.groupby("evaluator"):
        ax.scatter(group["k"], group["elapsed_seconds"], alpha=0.5, s=22, label=evaluator)
    ax.set_xlabel("Number of opened lines K")
    ax.set_ylabel("Elapsed seconds")
    ax.set_title("Candidate runtime by evaluator")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "runtime_resources.png", dpi=180)
    plt.close(fig)

    extrapolation = {
        evaluator: {
            "observed_smoke_elapsed_seconds": float(group["elapsed_seconds"].fillna(0).sum()),
            "serial_full_estimate_seconds": (
                float(group["elapsed_seconds"].fillna(0).sum())
                * (1.0 if is_full else full_run_workload_ratio(evaluator)["workload_ratio_full_to_smoke"])
            ),
            **(
                {
                    "topology_ratio_full_to_smoke": 1.0,
                    "inner_evaluation_ratio_full_to_smoke": 1.0,
                    "workload_ratio_full_to_smoke": 1.0,
                }
                if is_full else full_run_workload_ratio(evaluator)
            ),
            "estimate_scope": "candidate evaluation time only; excludes scheduler queueing and job startup",
        }
        for evaluator, group in candidates.groupby("evaluator")
    }
    atomic_write_json(derived / "full_run_workload_extrapolation.json", extrapolation)
    summary_text = [
        "# Stage K Smoke Summary",
        "",
        (
            "This is the full five-lambda Stage K production result."
            if is_full else
            "This is a diagnostic smoke result at `lambda_R=0.8`, not a final multi-lambda Pareto frontier."
        ),
        "",
        f"- Candidate attempts: {len(candidates)}",
        f"- Eligible candidates: {int(candidates['eligible'].sum())}",
        f"- Failure rows retained: {int((~candidates['eligible']).sum())}",
        f"- Diagnostic empirical nondominated points: {int(candidates['empirical_nondominated'].sum())}",
        "- Final scientific acceptance remains subject to Gate 5 validation."
        if is_full else "- Production remains unapproved pending Gate 4 review.",
    ]
    (output / "smoke_summary.md").write_text("\n".join(summary_text) + "\n", encoding="utf-8")
    summary = {
        "status": "PASS" if b2_failure_count == 0 else "PASS_WITH_WARNINGS",
        "candidate_attempts": len(candidates),
        "eligible_candidates": int(candidates["eligible"].sum()),
        "diagnostic_nondominated_points": int(candidates["empirical_nondominated"].sum()),
        "reference_b2_failure_count": b2_failure_count,
    }
    atomic_write_json(output / "aggregation_summary.json", summary)
    if not is_full:
        atomic_write_json(run.parent / "RUN_COMPLETE.json", {
            "status": "complete",
            "aggregation_summary": summary,
            "note": "Gate 4 smoke package complete; production remains a separate approval gate.",
        })
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.run_dir, args.output_dir, args.config), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
