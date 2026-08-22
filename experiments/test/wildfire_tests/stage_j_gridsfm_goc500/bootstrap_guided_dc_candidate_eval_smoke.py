"""J7 smoke for one Guided-DC fixed-(z, alpha) Stage J candidate."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.dc_economic_recourse import solve_fixed_topology_economic_dc_opf
from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity, source_less_load_ids
from stage_j_gridsfm_goc500.load_service import compute_load_shedding
from stage_j_gridsfm_goc500.metrics import compute_j_trade


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _scenario_targets(path: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["scenario_id"]] = [int(value) for value in row["target_branch_ids"].split(";") if value]
    return out


def _p_env(path: Path, scenario_id: str) -> dict[int, float]:
    out: dict[int, float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                out[int(row["branch_id"])] = float(row["p_env"])
    return out


def _r_base(path: Path, scenario_id: str) -> float:
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                return float(row["r_base"])
    raise KeyError(f"scenario {scenario_id} missing from {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    raw_case_path = gridsfm_root / "model" / "samples" / "case500_goc.pyg.json"
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    raw_case = _load_json(raw_case_path)
    identity = build_goc500_identity(raw_case)
    targets = _scenario_targets(input_dir / "stage_j_scenario_register.csv")
    offline_branch_ids = targets[args.scenario_id][:1]
    alpha_requested = {load.canonical_load_id: 1.0 for load in identity.loads}

    result = solve_fixed_topology_economic_dc_opf(
        raw_case=raw_case,
        identity=identity,
        offline_branch_ids=offline_branch_ids,
        alpha_requested=alpha_requested,
    )
    payload = {
        "status": "PASS" if result.objective_cost is not None else "FAIL",
        "scenario_id": args.scenario_id,
        "offline_branch_ids": offline_branch_ids,
        "lambda_r": args.lambda_r,
        "evaluation_status": result.evaluation_status.value,
        "objective_cost": result.objective_cost,
        "message": result.message,
    }
    if result.objective_cost is not None:
        p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
        r_base = _r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
        flow_loading = {
            branch.canonical_branch_id: abs(float(result.flow_by_line[branch.canonical_branch_id])) / float(branch.rate_a)
            for branch in identity.branches
            if branch.canonical_branch_id in result.flow_by_line
        }
        r_raw = sum(float(p_env.get(line_id, 0.0)) * float(loading) ** 2 for line_id, loading in flow_loading.items())
        r_norm = r_raw / r_base
        source_less = source_less_load_ids(identity, offline_branch_ids)
        breakdown = compute_load_shedding(
            [load.canonical_load_id for load in identity.loads],
            {load.canonical_load_id: load.pd_pre for load in identity.loads},
            alpha_requested,
            source_less,
        )
        payload.update(
            {
                "r_norm": r_norm,
                "l_shed_total": breakdown.l_shed_total,
                "l_shed_control": breakdown.l_shed_control,
                "l_shed_island": breakdown.l_shed_island,
                "j_trade": compute_j_trade(args.lambda_r, r_norm, breakdown.l_shed_total),
                "max_loading": max(flow_loading.values()) if flow_loading else None,
                "num_loading_gt_1": sum(1 for value in flow_loading.values() if value > 1.0),
            }
        )
    with output_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
