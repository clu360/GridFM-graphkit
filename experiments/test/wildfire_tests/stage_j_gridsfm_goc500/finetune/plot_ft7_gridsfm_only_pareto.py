"""Plot the refined FT7 empirical Pareto fronts for GridSFM models only."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_SPECS = {
    "Guided-GridSFM (released v1.1)": ("Model 1: released", "#D55E00", "o"),
    "Guided-GridSFM (FullTop-1000)": ("Model 2: FullTop-1000", "#6A3D9A", "s"),
    "Guided-GridSFM (FullTop-1500)": ("Model 3: FullTop-1500", "#009E73", "D"),
    "Guided-GridSFM (FullTop+N-1)": ("Model 4: FullTop+N-1", "#CC79A7", "^")
}


def _maximum_vertical_gap(
    first: pd.DataFrame, second: pd.DataFrame,
) -> tuple[float, float, float, float]:
    """Return x, first y, second y, and max gap on the shared x-domain."""
    first = first.sort_values("l_shed_total")
    second = second.sort_values("l_shed_total")
    lower = max(first["l_shed_total"].min(), second["l_shed_total"].min())
    upper = min(first["l_shed_total"].max(), second["l_shed_total"].max())
    if lower > upper:
        raise ValueError(
            "The Model 1 and Model 3 Pareto frontiers do not overlap in load shedding"
        )

    knots = np.unique(np.concatenate([
        first.loc[first["l_shed_total"].between(lower, upper), "l_shed_total"],
        second.loc[second["l_shed_total"].between(lower, upper), "l_shed_total"],
        np.asarray([lower, upper]),
    ]))
    first_y = np.interp(knots, first["l_shed_total"], first["r_norm"])
    second_y = np.interp(knots, second["l_shed_total"], second["r_norm"])
    index = int(np.argmax(first_y - second_y))
    return (
        float(knots[index]), float(first_y[index]), float(second_y[index]),
        float(first_y[index] - second_y[index]),
    )


def plot_gridsfm_frontiers(frontiers: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    gridsfm = frontiers[frontiers["method"].isin(MODEL_SPECS)].copy()
    outputs: list[Path] = []

    for scenario in ("J-S1", "J-S2", "J-S3"):
        subset = gridsfm[gridsfm["scenario_id"] == scenario]
        if subset.empty:
            raise ValueError(f"No GridSFM Pareto points found for {scenario}")

        # Render at native single-column dimensions so publication text is not
        # reduced from an oversized source canvas.
        fig, axis = plt.subplots(figsize=(3.5, 2.75))
        for method, (label, color, marker) in MODEL_SPECS.items():
            rows = subset[subset["method"] == method].sort_values("pareto_order")
            if rows.empty:
                raise ValueError(f"Missing {method} Pareto points for {scenario}")
            axis.plot(
                rows["l_shed_total"],
                rows["r_norm"],
                color=color,
                linewidth=1.25,
                marker=marker,
                markersize=2.8,
                markeredgecolor="white",
                markeredgewidth=0.45,
                label=label,
                zorder=3,
            )

        m0 = subset[subset["method"] == "Guided-GridSFM (released v1.1)"]
        m2 = subset[subset["method"] == "Guided-GridSFM (FullTop-1500)"]
        gap_x, m0_y, m2_y, max_gap = _maximum_vertical_gap(m0, m2)
        axis.annotate(
            "", xy=(gap_x, m0_y), xytext=(gap_x, m2_y),
            arrowprops={"arrowstyle": "<->", "color": "#374151", "linewidth": 1.4},
            zorder=4,
        )
        axis.annotate(
            f"Max Model 1 vs. Model 3 gap = {max_gap:.4f}\n"
            rf"at $L_{{\mathrm{{shed,total}}}}={gap_x:.4f}$",
            xy=(gap_x, (m0_y + m2_y) / 2.0),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=6.0,
            color="#374151",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 2.0},
            zorder=5,
        )

        axis.set_title(f"{scenario}: GridSFM Evaluated Pareto Frontiers", fontsize=8.2)
        axis.set_xlabel(
            r"Total load shedding, $L_{\mathrm{shed,total}}$", fontsize=7.2
        )
        axis.set_ylabel(r"Normalized wildfire risk, $R_{\mathrm{norm}}$", fontsize=7.2)
        axis.tick_params(labelsize=6.4)
        axis.grid(alpha=0.22, linewidth=0.5)
        axis.legend(
            frameon=False, fontsize=5.8, ncol=2, loc="upper center",
            bbox_to_anchor=(0.5, -0.25), columnspacing=0.9, handletextpad=0.4,
        )
        axis.margins(x=0.025, y=0.06)
        fig.tight_layout()

        output = output_dir / f"gridsfm_pareto_{scenario.lower()}.png"
        pdf_output = output.with_suffix(".pdf")
        fig.savefig(output, dpi=450, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_output, bbox_inches="tight", facecolor="white")
        if scenario == "J-S1":
            axis.set_title("")
            axis.legend(
                frameon=True,
                framealpha=0.92,
                facecolor="white",
                edgecolor="#D1D5DB",
                fontsize=5.3,
                ncol=1,
                loc="upper right",
                bbox_to_anchor=(0.998, 0.998),
                borderaxespad=0.0,
                borderpad=0.45,
                labelspacing=0.3,
                handlelength=2.0,
                handletextpad=0.45,
            )
            fig.tight_layout()
            no_title_output = output.with_name(f"{output.stem}_no_title.png")
            no_title_pdf_output = no_title_output.with_suffix(".pdf")
            fig.savefig(
                no_title_output, dpi=450, bbox_inches="tight", facecolor="white"
            )
            fig.savefig(no_title_pdf_output, bbox_inches="tight", facecolor="white")
            outputs.extend((no_title_output, no_title_pdf_output))
        plt.close(fig)
        outputs.extend((output, pdf_output))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frontiers",
        type=Path,
        default=Path(
            "experiments/test/wildfire_tests/goc_500_results/stage_j/"
            "refined_finetune_study/derived/evaluated_native_pareto_frontiers.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/test/wildfire_tests/goc_500_results/stage_j/"
            "refined_finetune_study/figures"
        ),
    )
    args = parser.parse_args()
    outputs = plot_gridsfm_frontiers(pd.read_parquet(args.frontiers), args.output_dir)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
