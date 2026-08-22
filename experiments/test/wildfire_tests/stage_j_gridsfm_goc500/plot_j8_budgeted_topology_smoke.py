"""Create compact figures for the J8 budgeted topology smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {
    "Guided-DC": "#2F6FDB",
    "Guided-GridSFM": "#C43C39",
    "TH-GridSFM": "#5E8C31",
    "TH-GridSFM-top1": "#5E8C31",
    "TH-GridSFM-top2": "#A2A331",
}


def _read_method(root: Path, subdir: str, label: str | None) -> tuple[pd.DataFrame, dict]:
    folder = root / subdir
    summary_json = next(folder.glob("j8_*_summary.json"))
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    summary_csv = Path(summary["summary_csv"])
    if not summary_csv.exists():
        suffix_by_subdir = {"gdc": "guided_dc", "gsfm": "guided_gridsfm", "th_gridsfm": "th_gridsfm"}
        suffix = suffix_by_subdir[subdir]
        summary_csv = folder / f"j8_{suffix}_topology_summary.csv"
    table = pd.read_csv(summary_csv)
    if label is not None:
        table["method"] = label
    for col in [
        "topology_rank",
        "search_objective",
        "j_trade",
        "j_total",
        "r_norm",
        "l_shed_total",
        "l_shed_control",
        "l_shed_island",
        "pac_total",
        "max_loading",
        "num_loading_gt_1",
    ]:
        if col in table:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    return table, summary


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_objective_by_rank(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for method, group in df.groupby("method", sort=False):
        color = METHOD_COLORS.get(method)
        objective_col = "search_objective"
        ax.plot(
            group["topology_rank"],
            group[objective_col],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{method} selection objective",
            color=color,
            alpha=0.9,
        )
        best = group.loc[group[objective_col].idxmin()]
        ax.scatter(
            [best["topology_rank"]],
            [best[objective_col]],
            s=85,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )
    _style_axes(ax, "J8 Selection Objective Across Topology Pool", "Topology rank", "Objective")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out / "j8_objective_by_topology_rank.png")


def plot_risk_load_by_rank(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    for method, group in df.groupby("method", sort=False):
        color = METHOD_COLORS.get(method)
        axes[0].plot(group["topology_rank"], group["r_norm"], marker="o", markersize=3, linewidth=1.4, label=method, color=color)
        axes[1].plot(group["topology_rank"], group["l_shed_total"], marker="o", markersize=3, linewidth=1.4, label=method, color=color)
    _style_axes(axes[0], "Normalized Wildfire Exposure", "Topology rank", "R_norm")
    _style_axes(axes[1], "Total Load Shedding", "Topology rank", "L_shed_total")
    axes[0].legend(frameon=False, fontsize=9)
    _save(fig, out / "j8_risk_load_by_topology_rank.png")


def plot_pareto(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for method, group in df.groupby("method", sort=False):
        color = METHOD_COLORS.get(method)
        ax.scatter(
            group["l_shed_total"],
            group["r_norm"],
            s=32,
            alpha=0.7,
            label=method,
            color=color,
        )
        best = group.loc[group["search_objective"].idxmin()]
        ax.scatter(
            [best["l_shed_total"]],
            [best["r_norm"]],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        ax.annotate(
            f"best {int(best['topology_rank'])}: {best['topology_id']}",
            (best["l_shed_total"], best["r_norm"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    _style_axes(ax, "J8 Pareto Scatter: Risk vs Load Shed", "L_shed_total", "R_norm")
    ax.legend(frameon=False, fontsize=9)
    _save(fig, out / "j8_pareto_scatter.png")


def plot_target_hits(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    offsets = {"Guided-DC": 0.06, "Guided-GridSFM": -0.02, "TH-GridSFM-top1": -0.08, "TH-GridSFM-top2": 0.12}
    for method, group in df.groupby("method", sort=False):
        hit = group["contains_any_target"].astype(str).str.lower().eq("true").astype(int)
        ax.scatter(
            group["topology_rank"],
            hit + offsets.get(method, 0.0),
            s=28,
            alpha=0.75,
            color=METHOD_COLORS.get(method),
            label=method,
        )
    ax.set_ylim(-0.2, 1.25)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["miss", "hit"])
    _style_axes(ax, "Target Branch Hit Across Topology Pool", "Topology rank", "Target branch 473 selected")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _save(fig, out / "j8_target_hit_by_rank.png")


def plot_gridsfm_penalties(gsfm: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), sharex=True)
    for method, group in gsfm.groupby("method", sort=False):
        color = METHOD_COLORS.get(method)
        axes[0].plot(group["topology_rank"], group["pac_total"], marker="o", markersize=3, linewidth=1.4, color=color, label=method)
        axes[1].plot(group["topology_rank"], group["max_loading"], marker="o", markersize=3, linewidth=1.4, color=color, label=method)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, label="thermal limit")
    _style_axes(axes[0], "GridSFM PAC_total Across Topology Pool", "Topology rank", "PAC_total")
    _style_axes(axes[1], "GridSFM Maximum Predicted Loading", "Topology rank", "max loading")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    _save(fig, out / "j8_gridsfm_pac_loading_by_rank.png")


def plot_runtime(summaries: dict[str, dict], out: Path) -> None:
    rows = []
    for label, payload in summaries.items():
        n = float(payload["num_candidate_evaluations"])
        runtime = float(payload["runtime_seconds"])
        rows.append(
            {
                "method": label,
                "runtime_seconds": runtime,
                "seconds_per_eval": runtime / n if n else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    axes[0].bar(table["method"], table["runtime_seconds"], color=[METHOD_COLORS[m] for m in table["method"]])
    axes[1].bar(table["method"], table["seconds_per_eval"], color=[METHOD_COLORS[m] for m in table["method"]])
    _style_axes(axes[0], "Total Runtime", "", "seconds")
    _style_axes(axes[1], "Runtime Per Candidate Evaluation", "", "seconds/eval")
    for ax in axes:
        ax.tick_params(axis="x", rotation=15)
    _save(fig, out / "j8_runtime_summary.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiments/test/wildfire_tests/goc_500_results/j8s/s1_l08")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve() if args.output_dir else root / "figures"
    gdc, gdc_summary = _read_method(root, "gdc", "Guided-DC")
    gsfm, gsfm_summary = _read_method(root, "gsfm", "Guided-GridSFM")
    frames = [gdc, gsfm]
    summaries = {"Guided-DC": gdc_summary, "Guided-GridSFM": gsfm_summary}
    if (root / "th_gridsfm").exists():
        th, th_summary = _read_method(root, "th_gridsfm", None)
        frames.append(th)
        summaries["TH-GridSFM"] = th_summary
    df = pd.concat(frames, ignore_index=True)

    plot_objective_by_rank(df, out)
    plot_risk_load_by_rank(df, out)
    plot_pareto(df, out)
    plot_target_hits(df, out)
    plot_gridsfm_penalties(df[df["method"].astype(str).str.startswith(("Guided-GridSFM", "TH-GridSFM"))], out)
    plot_runtime(summaries, out)

    manifest = pd.DataFrame(
        [
            {
                "figure": path.name,
                "path": str(path),
                "purpose": purpose,
            }
            for path, purpose in [
                (out / "j8_objective_by_topology_rank.png", "Selection objective over the shared 100-topology pool."),
                (out / "j8_risk_load_by_topology_rank.png", "R_norm and L_shed_total components by topology rank."),
                (out / "j8_pareto_scatter.png", "Risk/load-shed scatter with selected best points highlighted."),
                (out / "j8_target_hit_by_rank.png", "Whether each topology contains the J-S1 target branch."),
                (out / "j8_gridsfm_pac_loading_by_rank.png", "GridSFM PAC_total and maximum predicted loading by rank."),
                (out / "j8_runtime_summary.png", "Total runtime and seconds per candidate evaluation."),
            ]
        ]
    )
    manifest.to_csv(out / "j8_figure_manifest.csv", index=False)
    print(json.dumps({"status": "PASS", "figure_dir": str(out), "num_figures": 6}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
