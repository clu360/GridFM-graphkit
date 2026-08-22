from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.test.wildfire_tests.stage_i_dc_comparison.run_stage_h_miqp_pool_refresh import (  # noqa: E402
    AUGMENTED_POINTS_TABLE,
    EXCLUDE_GRIDFM_STAGE,
    RESULT_ROOT,
    _long_path,
    _plot_pareto,
)


DEFAULT_RUN = "main_results/r11"
OUTPUT_SUBDIR = "pareto_frontier_scatter_no_stage_e_by_lambda"


def _lambda_tag(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return text


def _rho_dir(value: float) -> str:
    return f"rho{float(value):g}"


def generate(run_dir: Path) -> pd.DataFrame:
    tables = run_dir / "tables"
    table = tables / AUGMENTED_POINTS_TABLE
    if not table.exists():
        table = tables / "all_evaluated_stage_h_points.csv"
    points = pd.read_csv(_long_path(table), low_memory=False)
    points = points[~points["stage"].astype(str).eq(EXCLUDE_GRIDFM_STAGE)].copy()
    points = points[
        np.isfinite(pd.to_numeric(points["rho_phys"], errors="coerce"))
        & np.isfinite(pd.to_numeric(points["lambda_R"], errors="coerce"))
        & np.isfinite(pd.to_numeric(points["L_shed"], errors="coerce"))
        & np.isfinite(pd.to_numeric(points["R_norm"], errors="coerce"))
    ].copy()
    points["rho_phys"] = points["rho_phys"].astype(float)
    points["lambda_R"] = points["lambda_R"].astype(float)

    rows = []
    for (rho, scenario_id, lambda_r), local in points.groupby(["rho_phys", "scenario_id", "lambda_R"], sort=True):
        out = run_dir / "plots" / "per_rho" / _rho_dir(float(rho)) / str(scenario_id) / OUTPUT_SUBDIR
        filename = f"pareto_frontier_scatter_no_stage_e_lambda_R_{_lambda_tag(float(lambda_r))}.png"
        title = (
            f"Main {scenario_id}, rho={float(rho):g}, lambda_R={float(lambda_r):g}: "
            "Pareto Scatter Excluding Stage E GridFM"
        )
        _plot_pareto(local, out, title, filename, include_gridfm=False)
        rows.append(
            {
                "run_dir": str(run_dir),
                "rho_phys": float(rho),
                "scenario_id": scenario_id,
                "lambda_R": float(lambda_r),
                "num_points": int(len(local)),
                "output_path": str(out / filename),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_path = tables / "per_lambda_no_stage_e_pareto_manifest.csv"
    os.makedirs(_long_path(manifest_path.parent), exist_ok=True)
    manifest.to_csv(_long_path(manifest_path), index=False)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN, help="Stage H run relative to the comparison root.")
    args = parser.parse_args()
    manifest = generate(RESULT_ROOT / args.run)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
