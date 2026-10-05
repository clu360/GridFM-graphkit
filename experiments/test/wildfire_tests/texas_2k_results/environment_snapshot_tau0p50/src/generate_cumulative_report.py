from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "reports"


def _describe(series: pd.Series) -> dict[str, float]:
    s = series.dropna()
    return {
        "count": int(s.shape[0]),
        "min": float(s.min()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
    }


def _line_map(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    colorbar_label: str,
    output_path: Path,
    label_top_n: int = 3,
) -> None:
    segments = [
        [(row.from_lon, row.from_lat), (row.to_lon, row.to_lat)]
        for row in df.itertuples(index=False)
        if np.isfinite(row.from_lon)
        and np.isfinite(row.from_lat)
        and np.isfinite(row.to_lon)
        and np.isfinite(row.to_lat)
    ]
    values = df[value_col].to_numpy(dtype=float)
    finite = np.isfinite(values)
    vmax = float(np.nanmax(values)) if finite.any() else 1.0
    if vmax <= 0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(10, 8))
    collection = LineCollection(
        segments,
        array=values[finite],
        cmap="viridis",
        linewidths=0.9,
        alpha=0.9,
    )
    collection.set_clim(0.0, vmax)
    ax.add_collection(collection)
    top = df.sort_values(value_col, ascending=False).head(label_top_n)
    midpoint_counts: dict[tuple[float, float], int] = {}
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        mid_lon = (row.from_lon + row.to_lon) / 2.0
        mid_lat = (row.from_lat + row.to_lat) / 2.0
        midpoint_key = (round(float(mid_lon), 6), round(float(mid_lat), 6))
        duplicate_idx = midpoint_counts.get(midpoint_key, 0)
        midpoint_counts[midpoint_key] = duplicate_idx + 1
        xy_offset = (7, 7 + 13 * duplicate_idx)
        ax.scatter(
            [mid_lon],
            [mid_lat],
            s=36,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        ax.annotate(
            f"{rank}: line {int(row.branch_id)}",
            xy=(mid_lon, mid_lat),
            xytext=xy_offset,
            textcoords="offset points",
            fontsize=8,
            color="#111111",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#666666", "alpha": 0.88},
            zorder=5,
        )
    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = fig.colorbar(collection, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    geometry = pd.read_parquet(DATA_DIR / "texas2k_branch_geometry.parquet")
    metrics = pd.read_parquet(DATA_DIR / "branch_environment_metrics_1600_CDT.parquet")
    snapshot = geometry.merge(metrics, on="branch_id", how="left", validate="one_to_one")

    snapshot["hazard_score_p_cumulative"] = snapshot["p_cumulative"].fillna(0.0)
    snapshot["baseline_loading"] = snapshot["scenario16_baseline_loading"].fillna(0.0)
    snapshot["wildfire_risk_loading_product"] = (
        snapshot["hazard_score_p_cumulative"] * snapshot["baseline_loading"]
    )
    snapshot["wildfire_risk_squared_loading_product"] = (
        snapshot["hazard_score_p_cumulative"] * snapshot["baseline_loading"] ** 2
    )
    snapshot["weather_timestamp"] = "2023-06-23 16:00 CDT"
    snapshot["weather_timestamp_utc"] = "2023-06-23T21:00:00+00:00"
    snapshot["electrical_scenario"] = 16

    out_cols = [
        "branch_id",
        "from_bus",
        "to_bus",
        "from_lon",
        "from_lat",
        "to_lon",
        "to_lat",
        "geo_length_km",
        "rateA",
        "status",
        "baseline_loading",
        "H_cumulative_raw",
        "hazard_score_p_cumulative",
        "wildfire_risk_loading_product",
        "wildfire_risk_squared_loading_product",
        "weather_coverage_valid",
        "weather_timestamp",
        "weather_timestamp_utc",
        "electrical_scenario",
    ]
    snapshot[out_cols].to_parquet(
        DATA_DIR / "cum_hazard_risk.parquet",
        index=False,
    )
    snapshot[out_cols].to_csv(
        DATA_DIR / "cum_hazard_risk.csv",
        index=False,
    )

    top_hazard = snapshot.sort_values("hazard_score_p_cumulative", ascending=False).head(25)
    top_risk = snapshot.sort_values("wildfire_risk_loading_product", ascending=False).head(25)
    top_squared_risk = snapshot.sort_values("wildfire_risk_squared_loading_product", ascending=False).head(25)
    table_cols = [
        "branch_id",
        "from_bus",
        "to_bus",
        "hazard_score_p_cumulative",
        "baseline_loading",
        "wildfire_risk_loading_product",
        "wildfire_risk_squared_loading_product",
        "H_cumulative_raw",
        "geo_length_km",
    ]
    top_hazard[table_cols].to_csv(TABLE_DIR / "top_pcum_hazard.csv", index=False)
    top_risk[table_cols].to_csv(TABLE_DIR / "top_pcum_x_loading.csv", index=False)
    top_squared_risk[table_cols].to_csv(
        TABLE_DIR / "top_pcum_x_loading2.csv",
        index=False,
    )

    _line_map(
        snapshot,
        "hazard_score_p_cumulative",
        "Texas2k Cumulative Weather Hazard, June 23 2023 16:00 CDT",
        "p_cumulative",
        FIGURE_DIR / "pcum_hazard_map.png",
    )
    _line_map(
        snapshot,
        "wildfire_risk_loading_product",
        "Texas2k Loading-Weighted Wildfire Risk, June 23 2023 16:00 CDT",
        "p_cumulative x baseline loading",
        FIGURE_DIR / "pcum_x_loading_map.png",
    )
    _line_map(
        snapshot,
        "wildfire_risk_squared_loading_product",
        "Texas2k Squared-Loading Wildfire Risk, June 23 2023 16:00 CDT",
        "p_cumulative x baseline loading^2",
        FIGURE_DIR / "pcum_x_loading2_map.png",
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].hist(snapshot["hazard_score_p_cumulative"], bins=50, color="#2f6f73", edgecolor="white", linewidth=0.4)
    axes[0].set_title("Weather Hazard")
    axes[0].set_xlabel("p_cumulative")
    axes[0].set_ylabel("Branch count")
    axes[0].grid(alpha=0.25)
    axes[1].hist(snapshot["wildfire_risk_loading_product"], bins=50, color="#8f4f2f", edgecolor="white", linewidth=0.4)
    axes[1].set_title("Wildfire Risk Term")
    axes[1].set_xlabel("p_cumulative x baseline loading")
    axes[1].set_ylabel("Branch count")
    axes[1].grid(alpha=0.25)
    axes[2].hist(snapshot["wildfire_risk_squared_loading_product"], bins=50, color="#5f5b9a", edgecolor="white", linewidth=0.4)
    axes[2].set_title("Squared-Loading Risk Term")
    axes[2].set_xlabel("p_cumulative x baseline loading^2")
    axes[2].set_ylabel("Branch count")
    axes[2].grid(alpha=0.25)
    fig.suptitle("Texas2k Branch Environmental Scores, June 23 2023 16:00 CDT")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pcum_hazard_risk_hist.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "weather_timestamp": "2023-06-23 16:00 CDT",
        "weather_timestamp_utc": "2023-06-23T21:00:00+00:00",
        "electrical_scenario": 16,
        "number_of_branches": int(snapshot.shape[0]),
        "valid_weather_coverage_branches": int(snapshot["weather_coverage_valid"].fillna(False).sum()),
        "hazard_score_definition": "p_cumulative",
        "wildfire_risk_definition": "p_cumulative * scenario16_baseline_loading",
        "squared_wildfire_risk_definition": "p_cumulative * scenario16_baseline_loading^2",
        "hazard_score_p_cumulative": _describe(snapshot["hazard_score_p_cumulative"]),
        "baseline_loading": _describe(snapshot["baseline_loading"]),
        "wildfire_risk_loading_product": _describe(snapshot["wildfire_risk_loading_product"]),
        "wildfire_risk_squared_loading_product": _describe(
            snapshot["wildfire_risk_squared_loading_product"]
        ),
    }
    (DATA_DIR / "cum_hazard_risk_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report = f"""# Stage K Cumulative Environmental Hazard And Wildfire Risk

## Scope

This report uses the June 23, 2023 16:00 CDT environmental snapshot for the modified Texas2k grid. The active weather-fused wildfire hazard is the branch-level cumulative FFWI exposure:

```text
hazard_score = p_cumulative
```

The loading-weighted wildfire risk term requested for this package is:

```text
wildfire_risk = p_cumulative * scenario16_baseline_loading
```

We also include the squared-loading wildfire risk term used in the earlier Stage-style risk convention:

```text
wildfire_risk_squared = p_cumulative * scenario16_baseline_loading^2
```

## Summary

- Branches: {summary["number_of_branches"]}
- Valid weather coverage branches: {summary["valid_weather_coverage_branches"]}
- Hazard median: {summary["hazard_score_p_cumulative"]["median"]:.6f}
- Hazard p95: {summary["hazard_score_p_cumulative"]["p95"]:.6f}
- Hazard max: {summary["hazard_score_p_cumulative"]["max"]:.6f}
- Loading-weighted risk median: {summary["wildfire_risk_loading_product"]["median"]:.6f}
- Loading-weighted risk p95: {summary["wildfire_risk_loading_product"]["p95"]:.6f}
- Loading-weighted risk max: {summary["wildfire_risk_loading_product"]["max"]:.6f}
- Squared-loading risk median: {summary["wildfire_risk_squared_loading_product"]["median"]:.6f}
- Squared-loading risk p95: {summary["wildfire_risk_squared_loading_product"]["p95"]:.6f}
- Squared-loading risk max: {summary["wildfire_risk_squared_loading_product"]["max"]:.6f}

## Outputs

- `figures/pcum_hazard_map.png`
- `figures/pcum_x_loading_map.png`
- `figures/pcum_x_loading2_map.png`
- `figures/pcum_hazard_risk_hist.png`
- `tables/top_pcum_hazard.csv`
- `tables/top_pcum_x_loading.csv`
- `tables/top_pcum_x_loading2.csv`
- `data/cum_hazard_risk.parquet`
- `data/cum_hazard_risk.csv`
- `data/cum_hazard_risk_summary.json`
"""
    (REPORT_DIR / "pcum_hazard_risk_report.md").write_text(
        report,
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
