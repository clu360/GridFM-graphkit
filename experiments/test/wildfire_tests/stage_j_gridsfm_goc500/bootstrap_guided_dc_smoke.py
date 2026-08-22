"""J4/J-guided-DC smoke on official GridSFM case500_goc raw JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.dc_economic_recourse import solve_fixed_topology_economic_dc_opf
from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity
from stage_j_gridsfm_goc500.schemas import EvaluationStatus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--offline-branch-id", type=int, default=None)
    parser.add_argument("--uniform-alpha", type=float, default=1.0)
    args = parser.parse_args()

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    sample = gridsfm_root / "model" / "samples" / "case500_goc.pyg.json"
    with sample.open("r", encoding="utf-8") as handle:
        raw_case = json.load(handle)

    identity = build_goc500_identity(raw_case)
    alpha_requested = {load.canonical_load_id: args.uniform_alpha for load in identity.loads}
    first_ac = next(branch.canonical_branch_id for branch in identity.branches if branch.edge_family == "ac_line")
    offline_branch_id = args.offline_branch_id if args.offline_branch_id is not None else first_ac

    rows = []
    for label, offline in (("intact", []), ("n1", [offline_branch_id])):
        result = solve_fixed_topology_economic_dc_opf(
            raw_case=raw_case,
            identity=identity,
            offline_branch_ids=offline,
            alpha_requested=alpha_requested,
            time_limit_seconds=120,
        )
        rows.append(
            {
                "case": label,
                "offline_branch_ids": offline,
                "evaluation_status": result.evaluation_status.value,
                "objective_cost": result.objective_cost,
                "num_pg": len(result.pg_by_generator),
                "num_theta": len(result.theta_by_bus),
                "num_flow": len(result.flow_by_line),
                "message": result.message,
            }
        )
        if result.evaluation_status is not EvaluationStatus.OK:
            print(json.dumps({"status": "FAIL", "rows": rows}, indent=2))
            return 1

    print(json.dumps({"status": "PASS", "sample": str(sample), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
