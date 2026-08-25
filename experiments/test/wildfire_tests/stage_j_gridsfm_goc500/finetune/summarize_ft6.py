"""Build compact review artifacts from completed FT6 manifests."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("m0", "m1", "m2", "m3")
MODEL_LABELS = {
    "m0": "Released v1.1",
    "m1": "FullTop-1000",
    "m2": "Sequential FullTop-1500",
    "m3": "FullTop-1000 + N-1-500",
}
STRATA = ("fulltop", "n1", "combined")
METRICS = (
    "loss", "cost_mape", "pg_mae", "qg_mae", "V_mae", "theta_mae",
    "brP_mae", "brQ_mae", "kcl_P_resid", "kcl_Q_resid",
    "thermal_max_loading", "thermal_frac_overload", "feas_acc",
)


def _read(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _test_tables(manifest: dict[str, Any], output: Path) -> None:
    rows = []
    for model in MODELS:
        for stratum in STRATA:
            metrics = manifest["results"][model][stratum]["metrics"]
            rows.append({
                "model_id": model,
                "model_label": MODEL_LABELS[model],
                "stratum": stratum,
                **{key: metrics[key] for key in METRICS},
                "n_graphs": metrics["n_graphs"],
                "runtime_seconds": manifest["results"][model][stratum]["runtime_seconds"],
            })
    _write_csv(output / "ft6_test_metrics.csv", rows)

    comparisons = []
    for model in ("m0", "m2", "m3"):
        for stratum in STRATA:
            baseline = manifest["results"]["m1"][stratum]["metrics"]
            updated = manifest["results"][model][stratum]["metrics"]
            for metric in METRICS:
                old = float(baseline[metric])
                new = float(updated[metric])
                comparisons.append({
                    "baseline_model_id": "m1",
                    "model_id": model,
                    "model_label": MODEL_LABELS[model],
                    "stratum": stratum,
                    "metric": metric,
                    "baseline": old,
                    "updated": new,
                    "delta": new - old,
                    "percent_change": 100.0 * (new - old) / old if old else "",
                    "lower_is_better": metric != "feas_acc",
                })
    _write_csv(output / "ft6_comparison_to_m1.csv", comparisons)


def _training_tables(root: Path, output: Path) -> dict[str, list[dict[str, Any]]]:
    logs = {
        model: _read(root / model / f"{model}_training_log.json")
        for model in ("m2", "m3")
    }
    rows = []
    for model, log in logs.items():
        for record in log:
            rows.append({"model_id": model, **record})
    _write_csv(output / "ft6_training_trajectories.csv", rows)
    return logs


def _plot_training(logs: dict[str, list[dict[str, Any]]], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels = (
        ("val_fulltop_loss", "FullTop validation loss"),
        ("val_n1_loss", "N-1 validation loss"),
        ("val_fulltop_cost_mape", "FullTop cost MAPE"),
        ("val_n1_cost_mape", "N-1 cost MAPE"),
    )
    colors = {"m2": "#237a57", "m3": "#c85432"}
    for axis, (key, title) in zip(axes.flat, panels, strict=True):
        for model in ("m2", "m3"):
            x = [int(row["epoch"]) + 1 for row in logs[model]]
            y = [float(row[key]) for row in logs[model]]
            axis.plot(x, y, marker="o", linewidth=1.8, markersize=4,
                      color=colors[model], label=MODEL_LABELS[model])
        axis.set_title(title)
        axis.grid(alpha=0.25)
        axis.set_xticks(range(1, 11))
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("FT6 held-out validation trajectories", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(output / "ft6_training_trajectories.png", dpi=180)
    plt.close(fig)


def _plot_test_comparison(manifest: dict[str, Any], output: Path) -> None:
    metrics = ("loss", "cost_mape", "brP_mae", "kcl_P_resid")
    labels = ("Loss", "Cost MAPE", "Branch P MAE", "KCL P residual")
    models = ("m0", "m2", "m3")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, stratum in zip(axes, ("fulltop", "n1"), strict=True):
        baseline = manifest["results"]["m1"][stratum]["metrics"]
        values = np.array([
            [100.0 * (manifest["results"][model][stratum]["metrics"][metric]
                      - baseline[metric]) / baseline[metric] for metric in metrics]
            for model in models
        ])
        image = axis.imshow(values, cmap="RdYlGn_r", vmin=-60, vmax=60, aspect="auto")
        axis.set_title("FullTop test" if stratum == "fulltop" else "N-1 test")
        axis.set_xticks(range(len(metrics)), labels, rotation=25, ha="right")
        axis.set_yticks(range(len(models)), [MODEL_LABELS[model] for model in models])
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                axis.text(col, row, f"{values[row, col]:+.1f}%", ha="center", va="center",
                          fontsize=9, color="black")
    fig.colorbar(image, ax=axes, label="Change from FullTop-1000 (lower is better)",
                 fraction=0.025, pad=0.04)
    fig.suptitle("FT6 sealed-test error changes relative to M1", fontsize=14)
    fig.subplots_adjust(left=0.20, right=0.88, bottom=0.23, top=0.84, wspace=0.12)
    fig.savefig(output / "ft6_test_comparison_to_m1.png", dpi=180)
    plt.close(fig)


def run(root: Path, output: Path) -> None:
    evaluation = _read(root / "evaluation" / "ft6_evaluation_manifest.json")
    if evaluation["status"] != "FT6_P3_FOUR_MODEL_EVALUATION_PASS":
        raise RuntimeError("FT6 evaluation is not complete")
    output.mkdir(parents=True, exist_ok=True)
    _test_tables(evaluation, output)
    logs = _training_tables(root, output)
    _plot_training(logs, output)
    _plot_test_comparison(evaluation, output)
    print(json.dumps({
        "status": "FT6_REVIEW_ARTIFACTS_PASS",
        "output_dir": str(output),
        "artifacts": sorted(path.name for path in output.iterdir()),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
