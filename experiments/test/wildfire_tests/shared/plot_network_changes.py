from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.shared.config import load_first_pass_config
from experiments.test.wildfire_tests.shared.paths import RESULTS_ROOT
from experiments.test.wildfire_tests.shared.scenario import load_first_pass_context


def _latest_run_dir(results_root: Path) -> Path:
    candidates = [
        path
        for path in results_root.iterdir()
        if path.is_dir() and (path / "optimization_summary.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No completed run directories found under {results_root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _edge_key(src: int, dst: int) -> tuple[int, int]:
    return (src, dst) if src <= dst else (dst, src)


def build_topology_graph(
    scenario,
    risk_df: pd.DataFrame,
    high_risk_lines: set[int],
    line_to_group: dict[int, str] | None = None,
) -> nx.Graph:
    line_to_group = line_to_group or {}
    graph = nx.Graph()
    graph.add_nodes_from(range(scenario.num_buses))

    edge_index = scenario.edge_index.cpu().numpy()
    for line_id, (src, dst) in enumerate(edge_index.T):
        src = int(src)
        dst = int(dst)
        key = _edge_key(src, dst)
        row = risk_df.loc[risk_df["line_id"] == line_id]
        loading_before = float(row["loading_ratio_before"].iloc[0]) if len(row) else 0.0
        loading_after = float(row["loading_ratio_after"].iloc[0]) if len(row) else 0.0
        risk_before = float(row["risk_before"].iloc[0]) if len(row) else 0.0
        risk_after = float(row["risk_after"].iloc[0]) if len(row) else 0.0

        if graph.has_edge(*key):
            data = graph.edges[key]
            data["line_ids"].append(line_id)
            data["loading_before"] = max(data["loading_before"], loading_before)
            data["loading_after"] = max(data["loading_after"], loading_after)
            data["risk_before"] += risk_before
            data["risk_after"] += risk_after
            data["high_risk"] = data["high_risk"] or line_id in high_risk_lines
            if line_id in line_to_group:
                data["group_ids"].append(line_to_group[line_id])
        else:
            graph.add_edge(
                *key,
                line_ids=[line_id],
                loading_before=loading_before,
                loading_after=loading_after,
                risk_before=risk_before,
                risk_after=risk_after,
                high_risk=line_id in high_risk_lines,
                group_ids=[line_to_group[line_id]] if line_id in line_to_group else [],
            )
    return graph


def plot_network_changes(run_dir: Path, output_path: Path | None = None) -> Path:
    config = load_first_pass_config(run_dir / "config.yaml")
    context = load_first_pass_context(config)
    scenario = context.scenario

    risk_df = pd.read_csv(run_dir / "risk_by_line_before_after.csv")
    decisions = pd.read_csv(run_dir / "decision_vector_final.csv")
    wildfire = _load_json(run_dir / "wildfire_scenario.json")
    summary = _load_json(run_dir / "optimization_summary.json")
    psps_path = run_dir / "psps_deenergization_decisions.csv"
    optimized_path = run_dir / "optimized_deenergization_decisions.csv"
    psps_df = pd.read_csv(psps_path) if psps_path.exists() else pd.DataFrame()
    optimized_df = pd.read_csv(optimized_path) if optimized_path.exists() else pd.DataFrame()
    if len(optimized_df):
        deenergized_lines = set(optimized_df.loc[optimized_df["deenergized"] == True, "line_id"].astype(int).tolist())
        deenergized_label = "Stage D optimized de-energized line"
    elif len(psps_df):
        deenergized_lines = set(psps_df.loc[psps_df["deenergized"] == True, "line_id"].astype(int).tolist())
        deenergized_label = "PSPS de-energized line"
    else:
        deenergized_lines = set()
        deenergized_label = "de-energized line"
    manual_high_risk_group_id = summary.get("manual_high_risk_group_id")

    high_risk_lines = {
        int(line_id)
        for group in wildfire["line_groups"]
        for line_id in group["line_ids"]
    }
    line_to_group = {
        int(line_id): group["name"]
        for group in wildfire["line_groups"]
        for line_id in group["line_ids"]
    }
    automatic_groups = config.wildfire.selection_method == "automatic_risk_components"
    graph = build_topology_graph(scenario, risk_df, high_risk_lines, line_to_group=line_to_group)
    pos = nx.spring_layout(graph, seed=30, k=0.72, iterations=200)

    fig, ax = plt.subplots(figsize=(13.5, 9.5))
    title = f"IEEE-30 topology with wildfire first-pass changes ({summary['model_type'].upper()})"
    if automatic_groups:
        num_groups = len({group["name"] for group in wildfire["line_groups"]})
        title += f" - automatic groups: {num_groups}"
        if num_groups == 1:
            title += " (collapsed to one group)"
    ax.set_title(title)

    normal_edges = [(u, v) for u, v, data in graph.edges(data=True) if not data["high_risk"]]
    high_edges = [(u, v) for u, v, data in graph.edges(data=True) if data["high_risk"]]
    reduced_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data["high_risk"] and data["risk_after"] < data["risk_before"]
    ]
    deenergized_edges = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if any(int(line_id) in deenergized_lines for line_id in data["line_ids"])
    ]
    reduced_edge_color = "#111111"
    deenergized_edge_color = "#e31a1c"

    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=normal_edges,
        edge_color="#c8c8c8",
        width=1.0,
        alpha=0.65,
        ax=ax,
    )
    group_palette = [
        "#d62728",
        "#9467bd",
        "#1f77b4",
        "#ff7f0e",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
    ]
    group_colors = {
        group["name"]: group_palette[idx % len(group_palette)]
        for idx, group in enumerate(wildfire["line_groups"])
    }
    if automatic_groups:
        for group_id, color in group_colors.items():
            group_edges = [
                edge
                for edge in high_edges
                if group_id in graph.edges[edge].get("group_ids", [])
            ]
            if not group_edges:
                continue
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=group_edges,
                edge_color=color,
                width=[
                    (3.4 if group_id == manual_high_risk_group_id else 2.0)
                    + 0.35 * graph.edges[edge]["loading_before"]
                    for edge in group_edges
                ],
                alpha=0.92,
                ax=ax,
            )
    else:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=high_edges,
            edge_color="#c43c39",
            width=[
                1.5 + 0.35 * graph.edges[edge]["loading_before"]
                for edge in high_edges
            ],
            alpha=0.9,
            ax=ax,
        )
    nx.draw_networkx_edges(
        graph,
        pos,
        edgelist=reduced_edges,
        edge_color=reduced_edge_color,
        width=4.4,
        alpha=0.85,
        style="dashed",
        ax=ax,
    )
    if deenergized_edges:
        nx.draw_networkx_edges(
            graph,
            pos,
            edgelist=deenergized_edges,
            edge_color=deenergized_edge_color,
            width=5.8,
            alpha=0.95,
            style="dashed",
            ax=ax,
        )
        x_points = []
        y_points = []
        for u, v in deenergized_edges:
            x_points.append((pos[u][0] + pos[v][0]) / 2.0)
            y_points.append((pos[u][1] + pos[v][1]) / 2.0)
        ax.scatter(
            x_points,
            y_points,
            marker="x",
            s=180,
            linewidths=3.0,
            color=deenergized_edge_color,
            zorder=8,
            label="_nolegend_",
        )

    node_colors = ["#f2f2f2"] * scenario.num_buses
    node_sizes = [420] * scenario.num_buses
    selected_loads = decisions[decisions["decision_type"] == "alpha"]
    selected_gens = decisions[decisions["decision_type"] == "delta_pg"]

    for _, row in selected_loads.iterrows():
        bus = int(row["bus_idx"])
        shed_fraction = max(0.0, 1.0 - float(row["value"]))
        node_colors[bus] = mcolors.to_hex(plt.get_cmap("Oranges")(0.35 + 0.55 * min(1.0, shed_fraction)))
        node_sizes[bus] = 520 + 2200 * shed_fraction

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#444444",
        linewidths=0.8,
        ax=ax,
    )

    if len(selected_gens):
        gen_nodes = [int(row["bus_idx"]) for _, row in selected_gens.iterrows()]
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=gen_nodes,
            node_shape="s",
            node_color="none",
            edgecolors="#2171b5",
            linewidths=2.4,
            node_size=700,
            ax=ax,
        )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels={i: str(i + 1) for i in graph.nodes},
        font_size=8,
        ax=ax,
    )

    edge_labels = {}
    deenergized_edge_labels = {}
    for u, v, data in graph.edges(data=True):
        if data["high_risk"]:
            off_ids = [int(line_id) for line_id in data["line_ids"] if int(line_id) in deenergized_lines]
            if off_ids:
                deenergized_edge_labels[(u, v)] = "OFF " + ",".join(str(line_id) for line_id in off_ids)
            else:
                edge_labels[(u, v)] = ",".join(str(line_id) for line_id in data["line_ids"])
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=deenergized_edge_labels,
        font_size=8,
        font_color=deenergized_edge_color,
        font_weight="bold",
        bbox={"facecolor": "white", "edgecolor": deenergized_edge_color, "alpha": 0.88, "pad": 1.5},
        ax=ax,
    )

    legend_items = [
        plt.Line2D([0], [0], color="#c8c8c8", lw=2, label="IEEE-30 topology line"),
        plt.Line2D([0], [0], color=reduced_edge_color, lw=4, linestyle="--", label="selected edge risk reduced"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e6550d", markeredgecolor="#444444", markersize=10, label="selected load bus (darker/larger = more shedding)"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="none", markeredgecolor="#2171b5", markeredgewidth=2, markersize=10, label="selected generator bus"),
    ]
    if deenergized_edges:
        legend_items.insert(
            2,
            plt.Line2D([0], [0], color=deenergized_edge_color, lw=5, linestyle="--", marker="x", markersize=8, label=deenergized_label),
        )
    if automatic_groups:
        group_items = [
            plt.Line2D(
                [0],
                [0],
                color=group_colors[group["name"]],
                lw=4 if group["name"] == manual_high_risk_group_id else 3,
                label=(
                    f"{group['name']} (largest high-risk group)"
                    if group["name"] == manual_high_risk_group_id
                    else group["name"]
                ),
            )
            for group in wildfire["line_groups"]
        ]
        legend_items[1:1] = group_items
    else:
        legend_items.insert(
            1,
            plt.Line2D([0], [0], color="#c43c39", lw=3, label="synthetic high-risk corridor"),
        )
    ax.legend(
        handles=legend_items,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.035),
        ncol=2,
        frameon=True,
        fontsize=8,
        columnspacing=1.1,
        handlelength=2.8,
    )

    ax.text(
        0.01,
        0.99,
        "Layout is topology-accurate, not geographic. Bus labels are 1-based; line labels are scenario edge IDs.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.85},
    )
    ax.axis("off")
    fig.tight_layout(rect=[0.0, 0.11, 1.0, 1.0])

    if output_path is None:
        output_path = run_dir / "figures" / "ieee30_network_changes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = _latest_run_dir(RESULTS_ROOT)
    output = plot_network_changes(run_dir, args.output)
    print(output)


if __name__ == "__main__":
    main()
