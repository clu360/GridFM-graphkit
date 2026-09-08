"""Publication plots for the RQ1 frozen-M0 evaluator benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np

from stage_j_gridsfm_goc500.plot_goc500_model_grid import (
    _branch_records,
    _layout,
    _segments,
)


COLORS = {"m0": "#2878b5", "refa": "#d1495b"}


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _baseline(path: Path) -> dict[int, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["branch_id"]): float(row["baseline_loading"]) for row in csv.DictReader(handle)}


def _draw_loading(
    ax,
    branches,
    positions,
    loading: Mapping[int, float],
    norm,
    offline: set[int],
    targets: set[int],
    title: str,
    bus_voltage: Mapping[int, float] | None = None,
    voltage_norm=None,
    text_scale: float = 1.0,
):
    active = [row for row in branches if row["branch_id"] not in offline]
    values = np.array([loading.get(row["branch_id"], np.nan) for row in active])
    valid = np.nan_to_num(values, nan=0.0)
    collection = LineCollection(
        _segments(active, positions), cmap="viridis", norm=norm, array=valid,
        linewidths=0.55 + 1.7 * np.clip(valid / max(norm.vmax, 1e-9), 0.0, 1.0),
        alpha=0.88, zorder=1,
    )
    ax.add_collection(collection)
    target_rows = [row for row in branches if row["branch_id"] in targets]
    if target_rows:
        ax.add_collection(LineCollection(
            _segments(target_rows, positions), colors="#e3ad16", linewidths=5.0,
            alpha=0.95, zorder=5,
        ))
        for row, segment in zip(
            target_rows, _segments(target_rows, positions), strict=True
        ):
            if row["branch_id"] in offline:
                continue
            midpoint = np.mean(np.asarray(segment), axis=0)
            ax.annotate(
                f"Wildfire target line {row['branch_id']}", xy=midpoint,
                xytext=(-10, 11), textcoords="offset points",
                fontsize=10.5 * text_scale, color="black", fontweight="bold",
                ha="right", va="bottom", zorder=10,
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 1.6},
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "black", "alpha": 0.94},
            )
    opened = [row for row in branches if row["branch_id"] in offline]
    if opened:
        ax.add_collection(LineCollection(
            _segments(opened, positions), colors="#d1495b", linewidths=3.5,
            linestyles="dashed", alpha=0.95, zorder=6,
        ))
        for row, segment in zip(opened, _segments(opened, positions), strict=True):
            midpoint = np.mean(np.asarray(segment), axis=0)
            label = f"Open line {row['branch_id']}"
            is_target = row["branch_id"] in targets
            if is_target:
                label += " (matches target)"
            ax.annotate(
                label, xy=midpoint,
                xytext=(-12, 12) if is_target else (9, 8),
                textcoords="offset points",
                fontsize=10.5 * text_scale, color="black", fontweight="bold",
                ha="right" if is_target else "left", va="bottom", zorder=10,
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 1.6},
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "black", "alpha": 0.94},
            )
    ordered_buses = sorted(positions)
    points = np.array([positions[bus_id] for bus_id in ordered_buses])
    if bus_voltage is None:
        nodes = ax.scatter(
            points[:, 0], points[:, 1], s=5.5, color="#4b5563",
            edgecolors="white", linewidths=0.14, alpha=0.82, zorder=3,
        )
    else:
        nodes = ax.scatter(
            points[:, 0], points[:, 1],
            c=[bus_voltage[bus_id] for bus_id in ordered_buses],
            cmap="coolwarm", norm=voltage_norm, s=11.0,
            edgecolors="#253044", linewidths=0.22, alpha=0.96, zorder=3,
        )
    ax.set_title(
        title, fontsize=13 * text_scale, loc="center", pad=1,
        fontweight="bold",
    )
    ax.set_aspect("equal", anchor="N")
    x_span = float(np.ptp(points[:, 0]))
    y_span = float(np.ptp(points[:, 1]))
    x_pad = max(0.025, 0.035 * x_span)
    y_pad = max(0.025, 0.035 * y_span)
    ax.set_xlim(float(points[:, 0].min()) - x_pad, float(points[:, 0].max()) + x_pad)
    ax.set_ylim(float(points[:, 1].min()) - y_pad, float(points[:, 1].max()) + y_pad)
    ax.axis("off")
    return collection, nodes


def _load_bus_map(case: Mapping) -> dict[int, int]:
    metadata = case["metadata"]
    return {
        int(load_id): int(bus_id)
        for load_id, bus_id in zip(
            metadata["load_id_map"], metadata["load_bus_map"], strict=True
        )
    }


def _overlay_load_decisions(
    ax,
    positions,
    *,
    selected_bus_ids: set[int],
    curtailed_bus_ids: set[int],
    curtailed_labels: Mapping[int, str] | None = None,
    text_scale: float = 1.0,
) -> None:
    if selected_bus_ids:
        points = np.array([positions[bus_id] for bus_id in sorted(selected_bus_ids)])
        ax.scatter(
            points[:, 0], points[:, 1], s=76, facecolors="none",
            edgecolors="#00a6a6", linewidths=2.0, zorder=8,
        )
    if curtailed_bus_ids:
        points = np.array([positions[bus_id] for bus_id in sorted(curtailed_bus_ids)])
        ax.scatter(
            points[:, 0], points[:, 1], s=132, marker="*", color="#d81b60",
            edgecolors="white", linewidths=0.7, zorder=9,
        )
        for bus_id in sorted(curtailed_bus_ids):
            if not curtailed_labels or bus_id not in curtailed_labels:
                continue
            ax.annotate(
                curtailed_labels[bus_id], xy=positions[bus_id],
                xytext=(-10, -13), textcoords="offset points",
                fontsize=10.5 * text_scale, color="black", fontweight="bold",
                ha="right", va="top", zorder=10,
                arrowprops={"arrowstyle": "-", "color": "black", "linewidth": 1.6},
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "black", "alpha": 0.94},
            )


def plot_representative_grid(
    *, raw_case_path: Path, baseline_path: Path, state_path: Path,
    alpha_effective_path: Path, selected_load_ids: list[int],
    offline_branch_ids: list[int], target_branch_ids: list[int],
    output_path: Path,
) -> None:
    case = _read_json(raw_case_path)
    state = _read_json(state_path)
    branches = _branch_records(case)
    bus_ids = [int(value) for value in case["metadata"]["bus_id_map"]]
    positions = _layout(bus_ids, branches, seed=42)
    baseline = _baseline(baseline_path)
    candidate = {
        int(branch_id): float(value)
        for branch_id, value in zip(state["branch_ids"], state["loading"], strict=True)
    }
    bus_voltage = {
        int(bus_id): float(value)
        for bus_id, value in zip(state["bus_ids"], state["vm"], strict=True)
    }
    with alpha_effective_path.open(newline="", encoding="utf-8") as handle:
        alpha_effective = {
            int(row["load_id"]): float(row["alpha"]) for row in csv.DictReader(handle)
        }
    active_curtailed_load_ids = {
        load_id for load_id, alpha in alpha_effective.items() if alpha < 1.0 - 1e-9
    }
    load_bus = _load_bus_map(case)
    selected_bus_ids = {load_bus[load_id] for load_id in selected_load_ids}
    curtailed_bus_ids = {load_bus[load_id] for load_id in active_curtailed_load_ids}
    curtailed_labels = {
        load_bus[load_id]: f"Load {load_id}: alpha={alpha_effective[load_id]:.3f}"
        for load_id in active_curtailed_load_ids
    }
    finite = list(baseline.values()) + list(candidate.values())
    vmax = max(1.0, float(np.max(finite)))
    norm = Normalize(vmin=0.0, vmax=vmax)
    vm_min = min(bus_voltage.values())
    vm_max = max(bus_voltage.values())
    voltage_norm = TwoSlopeNorm(
        vmin=min(0.95, vm_min), vcenter=1.0, vmax=max(1.10, vm_max)
    )
    # Native full-text-width sizing avoids shrinking all type and line weights
    # when the figure is placed into a two-column paper at about 7.2 inches.
    text_scale = 0.72
    fig = plt.figure(figsize=(7.2, 3.4))
    grid = fig.add_gridspec(
        2, 3, width_ratios=(1, 1, 0.027), height_ratios=(1, 1),
        left=0.001, right=0.975, bottom=0.001, top=0.995,
        wspace=0.012, hspace=0.07,
    )
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1])]
    loading_cax = fig.add_subplot(grid[0, 2])
    voltage_cax = fig.add_subplot(grid[1, 2])
    fig.patch.set_facecolor("white")
    collection, _ = _draw_loading(
        axes[0], branches, positions, baseline, norm, set(), set(target_branch_ids),
        "A. Baseline intact operating condition",
        text_scale=text_scale,
    )
    _, voltage_nodes = _draw_loading(
        axes[1], branches, positions, candidate, norm, set(offline_branch_ids),
        set(target_branch_ids), "B. GridSFM predicted post-decision AC state",
        bus_voltage=bus_voltage, voltage_norm=voltage_norm,
        text_scale=text_scale,
    )
    _overlay_load_decisions(
        axes[1], positions,
        selected_bus_ids=selected_bus_ids,
        curtailed_bus_ids=curtailed_bus_ids,
        curtailed_labels=curtailed_labels,
        text_scale=text_scale,
    )
    colorbar = fig.colorbar(collection, cax=loading_cax)
    colorbar.set_label("Branch loading (p.u. of rateA)", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.8, width=0.8, length=3.0)
    voltage_bar = fig.colorbar(voltage_nodes, cax=voltage_cax)
    voltage_bar.set_label("Predicted voltage magnitude (p.u.)", fontsize=8.5)
    voltage_bar.ax.tick_params(labelsize=7.8, width=0.8, length=3.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_post_decision_grid(
    *, raw_case_path: Path, state_path: Path,
    alpha_effective_path: Path, selected_load_ids: list[int],
    offline_branch_ids: list[int], target_branch_ids: list[int],
    output_path: Path,
) -> None:
    """Render only the GridSFM-predicted post-decision electrical state."""
    case = _read_json(raw_case_path)
    state = _read_json(state_path)
    branches = _branch_records(case)
    bus_ids = [int(value) for value in case["metadata"]["bus_id_map"]]
    positions = _layout(bus_ids, branches, seed=42)
    candidate = {
        int(branch_id): float(value)
        for branch_id, value in zip(state["branch_ids"], state["loading"], strict=True)
    }
    bus_voltage = {
        int(bus_id): float(value)
        for bus_id, value in zip(state["bus_ids"], state["vm"], strict=True)
    }
    with alpha_effective_path.open(newline="", encoding="utf-8") as handle:
        alpha_effective = {
            int(row["load_id"]): float(row["alpha"]) for row in csv.DictReader(handle)
        }
    active_curtailed_load_ids = {
        load_id for load_id, alpha in alpha_effective.items() if alpha < 1.0 - 1e-9
    }
    load_bus = _load_bus_map(case)
    selected_bus_ids = {load_bus[load_id] for load_id in selected_load_ids}
    curtailed_bus_ids = {load_bus[load_id] for load_id in active_curtailed_load_ids}
    curtailed_labels = {
        load_bus[load_id]: f"Load {load_id}: alpha={alpha_effective[load_id]:.3f}"
        for load_id in active_curtailed_load_ids
    }

    norm = Normalize(vmin=0.0, vmax=max(1.0, max(candidate.values())))
    vm_min = min(bus_voltage.values())
    vm_max = max(bus_voltage.values())
    voltage_norm = TwoSlopeNorm(
        vmin=min(0.95, vm_min), vcenter=1.0, vmax=max(1.10, vm_max)
    )

    fig = plt.figure(figsize=(10.8, 7.4))
    grid = fig.add_gridspec(
        2, 2, width_ratios=(1, 0.025), height_ratios=(1, 1),
        left=0.005, right=0.955, bottom=0.075, top=0.995,
        wspace=0.025, hspace=0.12,
    )
    axis = fig.add_subplot(grid[:, 0])
    loading_cax = fig.add_subplot(grid[0, 1])
    voltage_cax = fig.add_subplot(grid[1, 1])
    fig.patch.set_facecolor("white")
    collection, voltage_nodes = _draw_loading(
        axis, branches, positions, candidate, norm, set(offline_branch_ids),
        set(target_branch_ids), "", bus_voltage=bus_voltage,
        voltage_norm=voltage_norm,
    )
    _overlay_load_decisions(
        axis, positions,
        selected_bus_ids=selected_bus_ids,
        curtailed_bus_ids=curtailed_bus_ids,
        curtailed_labels=curtailed_labels,
    )
    loading_bar = fig.colorbar(collection, cax=loading_cax)
    loading_bar.set_label("Predicted branch loading (p.u. of rateA)", fontsize=9)
    voltage_bar = fig.colorbar(voltage_nodes, cax=voltage_cax)
    voltage_bar.set_label("Predicted voltage magnitude (p.u.)", fontsize=9)
    fig.legend(
        handles=[
            Line2D([], [], color="#e3ad16", linewidth=3.6, label="Wildfire target branch"),
            Line2D([], [], color="#d1495b", linewidth=2.4, linestyle="--", label="Outer-search de-energized branch"),
            Line2D([], [], marker="o", markersize=7, markerfacecolor="none", markeredgecolor="#00a6a6", linestyle="None", label="Selected alpha-control bus"),
            Line2D([], [], marker="*", markersize=10, markerfacecolor="#d81b60", markeredgecolor="white", linestyle="None", label="Actively curtailed load bus"),
        ],
        loc="lower center", bbox_to_anchor=(0.47, 0.006), ncol=4,
        frameon=False, fontsize=8.8, columnspacing=1.3, handletextpad=0.55,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_runtime(paired, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for _, row in paired.iterrows():
        ax.plot(
            [0, 1], [row["m0_evaluator_median_seconds"], row["refa_evaluator_median_seconds"]],
            color="#a8b0ba", linewidth=0.65, alpha=0.38, zorder=1,
        )
    ax.scatter(
        np.zeros(len(paired)), paired["m0_evaluator_median_seconds"],
        color=COLORS["m0"], s=20, alpha=0.78, label="Frozen GridSFM M0", zorder=3,
    )
    ax.scatter(
        np.ones(len(paired)), paired["refa_evaluator_median_seconds"],
        color=COLORS["refa"], s=20, alpha=0.78, label="Reference A AC OPF", zorder=3,
    )
    medians = [
        paired["m0_evaluator_median_seconds"].median(),
        paired["refa_evaluator_median_seconds"].median(),
    ]
    ax.scatter([0, 1], medians, color="#111827", marker="D", s=48, label="Median", zorder=5)
    ax.set_xticks([0, 1], ["Frozen M0\nevaluator", "Reference A\nevaluator"])
    ax.set_ylabel("Per-instance median runtime (seconds, log scale)")
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", color="#d7dce2", linewidth=0.7, alpha=0.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("RQ1 paired evaluator runtime across 54 unique fixed decisions", loc="left", fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
