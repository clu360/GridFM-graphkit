from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd


def _load_objective_weights(run_dir: Path) -> tuple[float, float] | None:
    summary_path = run_dir / "optimization_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    if "lambda_R" not in summary or "lambda_L" not in summary:
        return None
    return float(summary["lambda_R"]), float(summary["lambda_L"])


def plot_optimization_behavior(run_dir: Path, output_path: Path | None = None) -> Path:
    trace_path = run_dir / "objective_trace.csv"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing objective trace: {trace_path}")

    trace = pd.read_csv(trace_path)
    if trace.empty:
        raise ValueError(f"Objective trace is empty: {trace_path}")

    objective_weights = _load_objective_weights(run_dir)
    n_rows = 4 if objective_weights is not None else 3
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 9 if objective_weights is not None else 8), sharex=True)

    axes[0].plot(trace["eval_idx"], trace["objective_total"], color="#252525", linewidth=1.5)
    axes[0].set_ylabel("objective")
    axes[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(trace["eval_idx"], trace["wildfire_group_risk"], color="#c43c39", linewidth=1.5)
    axes[1].set_ylabel("group risk")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(trace["eval_idx"], trace["load_shedding"], color="#756bb1", linewidth=1.5)
    axes[2].set_ylabel("load shedding")
    axes[2].grid(True, alpha=0.25)

    if objective_weights is not None:
        lambda_r, lambda_l = objective_weights
        axes[3].axhline(lambda_r, color="#c43c39", linewidth=1.6, label=f"lambda_R = {lambda_r:g}")
        axes[3].axhline(lambda_l, color="#756bb1", linewidth=1.6, label=f"lambda_L = {lambda_l:g}")
        axes[3].set_ylim(-0.05, 1.05)
        axes[3].set_ylabel("weights")
        axes[3].legend(loc="best")
        axes[3].grid(True, alpha=0.25)

    axes[-1].set_xlabel("objective evaluation")

    fig.suptitle("Wildfire first-pass optimization behavior")
    fig.tight_layout()

    if output_path is None:
        output_path = run_dir / "figures" / "optimization_behavior.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = plot_optimization_behavior(args.run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
