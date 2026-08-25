"""Plot a GOC-500 topology with a saved GridSFM-predicted operating state."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _read_numeric_csv(path: Path, key: str, fields: tuple[str, ...]) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[int(row[key])] = {
                field: float(row[field]) for field in fields if row.get(field) not in (None, "")
            }
    return rows


def _best_alpha(summary: dict[str, object]) -> dict[int, float]:
    raw = summary.get("best_alpha")
    if raw is None and isinstance(summary.get("best_topology"), dict):
        raw = summary["best_topology"].get("best_alpha_selected")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {int(k): float(v) for k, v in raw.items()} if isinstance(raw, dict) else {}


def _branch_records(case: dict[str, object]) -> list[dict[str, object]]:
    grid = case["grid"]
    metadata = case["metadata"]
    bus_ids = metadata["bus_id_map"]
    records: list[dict[str, object]] = []
    for edge_type, id_key in (
        ("ac_line", "ac_line_branch_ids"),
        ("transformer", "transformer_branch_ids"),
    ):
        edges = grid["edges"][edge_type]
        for branch_id, sender, receiver in zip(
            metadata[id_key], edges["senders"], edges["receivers"], strict=True
        ):
            records.append(
                {
                    "branch_id": int(branch_id),
                    "from_bus": int(bus_ids[int(sender)]),
                    "to_bus": int(bus_ids[int(receiver)]),
                    "edge_type": edge_type,
                }
            )
    return records


def _layout(bus_ids: list[int], branches: list[dict[str, object]], seed: int) -> dict[int, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(bus_ids)
    graph.add_edges_from((row["from_bus"], row["to_bus"]) for row in branches)
    positions = nx.spring_layout(graph, seed=seed, iterations=250, k=0.115)
    points = np.array([positions[bus_id] for bus_id in bus_ids])
    center = points.mean(axis=0)
    scale = np.max(np.abs(points - center), axis=0)
    scale[scale == 0] = 1.0
    return {bus_id: (positions[bus_id] - center) / scale for bus_id in bus_ids}


def _segments(
    branches: list[dict[str, object]], positions: dict[int, np.ndarray]
) -> list[list[np.ndarray]]:
    return [[positions[row["from_bus"]], positions[row["to_bus"]]] for row in branches]


def _draw_open_branches(
    ax: plt.Axes,
    branches: list[dict[str, object]],
    positions: dict[int, np.ndarray],
    offline_ids: set[int],
) -> None:
    opened = [row for row in branches if row["branch_id"] in offline_ids]
    if not opened:
        return
    ax.add_collection(
        LineCollection(
            _segments(opened, positions), colors="#d62728", linewidths=2.3,
            linestyles="dashed", zorder=6,
        )
    )
    for row in opened:
        midpoint = (positions[row["from_bus"]] + positions[row["to_bus"]]) / 2
        ax.annotate(
            f"open {row['branch_id']}", midpoint, xytext=(4, 4), textcoords="offset points",
            fontsize=7, color="#a31319", fontweight="bold", zorder=8,
        )


def _finish_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.07, 1.07)
    ax.set_ylim(-1.07, 1.07)
    ax.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--branch-state-csv", required=True)
    parser.add_argument("--bus-state-csv", required=True)
    parser.add_argument("--alpha-summary-json")
    parser.add_argument("--offline-branch-ids", default="")
    parser.add_argument("--checkpoint-label", default="GridSFM")
    parser.add_argument("--scenario-id", default="")
    parser.add_argument("--lambda-r", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-png", required=True)
    parser.add_argument("--layout-csv")
    parser.add_argument("--manifest-json")
    args = parser.parse_args()

    case_path = Path(args.case_json).expanduser().resolve()
    branch_path = Path(args.branch_state_csv).expanduser().resolve()
    bus_path = Path(args.bus_state_csv).expanduser().resolve()
    output_path = Path(args.output_png).expanduser().resolve()
    case = _load_json(case_path)
    bus_ids = [int(value) for value in case["metadata"]["bus_id_map"]]
    branches = _branch_records(case)
    branch_state = _read_numeric_csv(branch_path, "branch_id", ("loading",))
    bus_state = _read_numeric_csv(bus_path, "bus_id", ("vm", "va"))
    offline_ids = {
        int(value) for value in args.offline_branch_ids.replace(",", ";").split(";") if value.strip()
    }
    alphas: dict[int, float] = {}
    if args.alpha_summary_json:
        alphas = _best_alpha(_load_json(Path(args.alpha_summary_json).expanduser().resolve()))

    metadata = case["metadata"]
    generator_buses = set(int(value) for value in metadata["gen_bus_map"])
    load_to_bus = {
        int(load_id): int(bus_id)
        for load_id, bus_id in zip(metadata["load_id_map"], metadata["load_bus_map"], strict=True)
    }
    selected_load_buses = {load_to_bus[load_id] for load_id in alphas if load_id in load_to_bus}
    curtailed_load_buses = {
        load_to_bus[load_id] for load_id, alpha in alphas.items()
        if load_id in load_to_bus and alpha < 0.999
    }
    positions = _layout(bus_ids, branches, args.seed)

    active = [row for row in branches if row["branch_id"] not in offline_ids]
    loadings = np.array([branch_state.get(row["branch_id"], {}).get("loading", np.nan) for row in active])
    finite_loadings = loadings[np.isfinite(loadings)]
    loading_max = max(1.05, float(np.max(finite_loadings))) if finite_loadings.size else 1.05
    loading_norm = Normalize(vmin=0.0, vmax=loading_max)
    voltage = np.array([bus_state[bus_id]["vm"] for bus_id in bus_ids])
    voltage_norm = TwoSlopeNorm(
        vmin=min(0.999, float(voltage.min())),
        vcenter=1.0,
        vmax=max(1.001, float(voltage.max())),
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
    fig.subplots_adjust(left=0.025, right=0.925, bottom=0.115, top=0.82, wspace=0.10)
    fig.patch.set_facecolor("#f7f8fa")
    for ax in axes:
        ax.set_facecolor("#ffffff")

    axes[0].add_collection(
        LineCollection(_segments(active, positions), colors="#aeb7c2", linewidths=0.55, alpha=0.68)
    )
    bus_xy = np.array([positions[bus_id] for bus_id in bus_ids])
    axes[0].scatter(bus_xy[:, 0], bus_xy[:, 1], s=6, c="#34495e", linewidths=0, zorder=3)
    gen_xy = np.array([positions[bus_id] for bus_id in sorted(generator_buses)])
    axes[0].scatter(
        gen_xy[:, 0], gen_xy[:, 1], s=17, marker="s", facecolors="#179c8c",
        edgecolors="white", linewidths=0.35, zorder=4,
    )
    if selected_load_buses:
        selected_xy = np.array([positions[bus_id] for bus_id in sorted(selected_load_buses)])
        axes[0].scatter(
            selected_xy[:, 0], selected_xy[:, 1], s=42, facecolors="none",
            edgecolors="#e67e22", linewidths=1.25, zorder=5,
        )
    if curtailed_load_buses:
        curtailed_xy = np.array([positions[bus_id] for bus_id in sorted(curtailed_load_buses)])
        axes[0].scatter(
            curtailed_xy[:, 0], curtailed_xy[:, 1], s=80, marker="*",
            facecolors="#f4c542", edgecolors="#6c5300", linewidths=0.7, zorder=7,
        )
    _draw_open_branches(axes[0], branches, positions, offline_ids)
    axes[0].legend(
        handles=[
            Line2D([], [], marker="o", color="none", markerfacecolor="#34495e", markersize=4, label="Bus"),
            Line2D([], [], marker="s", color="none", markerfacecolor="#179c8c", markersize=6, label="Generator bus"),
            Line2D([], [], marker="o", color="#e67e22", markerfacecolor="none", markersize=7, label="Selected load-control bus"),
            Line2D([], [], marker="*", color="#6c5300", markerfacecolor="#f4c542", markersize=9, label="Actively curtailed load bus"),
            Line2D([], [], color="#d62728", linestyle="--", linewidth=2, label="Open branch"),
        ],
        loc="lower left", frameon=False, fontsize=8,
    )
    _finish_axis(axes[0], "A  Final topology and control locations")

    valid = np.nan_to_num(loadings, nan=0.0)
    widths = 0.35 + 1.45 * np.clip(valid / loading_max, 0.0, 1.0)
    branch_collection = LineCollection(
        _segments(active, positions), cmap="plasma", norm=loading_norm,
        array=valid, linewidths=widths, alpha=0.82, zorder=1,
    )
    axes[1].add_collection(branch_collection)
    voltage_scatter = axes[1].scatter(
        bus_xy[:, 0], bus_xy[:, 1], s=12, c=voltage, cmap="coolwarm",
        norm=voltage_norm, edgecolors="#263238", linewidths=0.18, zorder=3,
    )
    overloaded = [row for row in active if branch_state.get(row["branch_id"], {}).get("loading", 0.0) > 1.0]
    if overloaded:
        axes[1].add_collection(
            LineCollection(_segments(overloaded, positions), colors="#00bcd4", linewidths=2.2, zorder=4)
        )
        axes[1].legend(
            handles=[
                Line2D([], [], color="#00bcd4", linewidth=2.2, label="Predicted loading > 1.0 p.u."),
            ],
            loc="lower left", frameon=False, fontsize=8,
        )
    _draw_open_branches(axes[1], branches, positions, offline_ids)
    branch_cax = fig.add_axes((0.61, 0.075, 0.23, 0.025))
    voltage_cax = fig.add_axes((0.94, 0.28, 0.013, 0.37))
    fig.colorbar(
        branch_collection, cax=branch_cax, orientation="horizontal",
        label="Predicted branch loading (p.u.)",
    )
    fig.colorbar(
        voltage_scatter, cax=voltage_cax, orientation="vertical",
        label="Predicted voltage magnitude (p.u.)",
    )
    _finish_axis(axes[1], "B  Fine-tuned GridSFM predicted AC state")

    scenario_bits = [bit for bit in (args.scenario_id, f"lambda_r={args.lambda_r}" if args.lambda_r else "") if bit]
    max_loading = float(np.nanmax(loadings)) if finite_loadings.size else float("nan")
    fig.suptitle(
        "GOC-500 model-informed grid visualization", y=0.975,
        fontsize=17, fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        f"{args.checkpoint_label} | {' | '.join(scenario_bits)} | "
        f"500 buses, {len(branches)} branches | max loading {max_loading:.3f} p.u. | "
        f"voltage {voltage.min():.3f}-{voltage.max():.3f} p.u.",
        ha="center", va="top", fontsize=9.5, color="#46515c",
    )
    fig.text(
        0.5, 0.018, "Deterministic force-directed topological layout; positions are not geographic.",
        ha="center", fontsize=8, color="#59636e",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)

    layout_path = Path(args.layout_csv).expanduser().resolve() if args.layout_csv else output_path.with_suffix(".layout.csv")
    with layout_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("bus_id", "x", "y"))
        writer.writeheader()
        writer.writerows(
            {"bus_id": bus_id, "x": float(positions[bus_id][0]), "y": float(positions[bus_id][1])}
            for bus_id in bus_ids
        )

    manifest_path = Path(args.manifest_json).expanduser().resolve() if args.manifest_json else output_path.with_suffix(".manifest.json")
    manifest = {
        "case_json": str(case_path),
        "branch_state_csv": str(branch_path),
        "bus_state_csv": str(bus_path),
        "checkpoint_label": args.checkpoint_label,
        "scenario_id": args.scenario_id,
        "lambda_r": args.lambda_r,
        "offline_branch_ids": sorted(offline_ids),
        "selected_alpha": alphas,
        "layout": "networkx.spring_layout",
        "layout_seed": args.seed,
        "num_buses": len(bus_ids),
        "num_branches": len(branches),
        "max_predicted_loading": max_loading,
        "num_predicted_loading_gt_1": int(np.sum(loadings > 1.0)),
        "min_predicted_vm": float(voltage.min()),
        "max_predicted_vm": float(voltage.max()),
        "output_png": str(output_path),
        "layout_csv": str(layout_path),
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
