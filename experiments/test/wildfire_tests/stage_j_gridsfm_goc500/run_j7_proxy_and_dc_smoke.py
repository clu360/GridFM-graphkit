"""J7 one-scenario smoke: shared proxy pool plus Guided-DC evaluation."""

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
from stage_j_gridsfm_goc500.outer_proxy import solve_proxy_topology_pool
from stage_j_gridsfm_goc500.scenario_builder import load_baseline_loading_csv


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def _load_candidate_scores(path: Path) -> dict[int, float]:
    out = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[int(row["branch_id"])] = float(row["c_l"])
    return out


def _p_env(path: Path, scenario_id: str) -> dict[int, float]:
    out = {}
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
    raise KeyError(f"scenario {scenario_id} missing from scenario register")


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--baseline-loading-csv", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--pool-size", type=int, default=5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    raw_case = _load_json(gridsfm_root / "model" / "samples" / "case500_goc.pyg.json")
    identity = build_goc500_identity(raw_case, candidate_branch_ids=[
        int(row["branch_id"])
        for row in csv.DictReader((input_dir / "stage_j_candidate_line_scores.csv").open(newline=""))
    ])
    baseline_loading = load_baseline_loading_csv(args.baseline_loading_csv)
    c_by_line = _load_candidate_scores(input_dir / "stage_j_candidate_line_scores.csv")
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    r_base = _r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    candidate_ids = sorted(c_by_line)

    pool = solve_proxy_topology_pool(
        candidate_branch_ids=candidate_ids,
        p_env_by_line=p_env,
        baseline_loading=baseline_loading,
        c_by_line=c_by_line,
        lambda_r_proxy=args.lambda_r,
        k=args.k,
        pool_size=args.pool_size,
    )

    topology_rows = [
        {
            "method_family": "Guided",
            "rank": item.rank,
            "shutoff_branch_ids": ";".join(str(line_id) for line_id in item.shutoff_branch_ids),
            "proxy_objective": item.proxy_objective,
            "r_proxy": item.r_proxy,
            "l_proxy": item.l_proxy,
            "alpha_strategy": "smoke_fixed_full_vector_alpha_ones_not_full_alpha_search",
        }
        for item in pool
    ]

    th_ranked = sorted(candidate_ids, key=lambda line_id: float(p_env.get(line_id, 0.0)) * float(baseline_loading[line_id]) ** 2, reverse=True)
    for th_k in (1, 2):
        selected = tuple(th_ranked[:th_k])
        topology_rows.append(
            {
                "method_family": f"TH-GridSFM-top{th_k}",
                "rank": 1,
                "shutoff_branch_ids": ";".join(str(line_id) for line_id in selected),
                "proxy_objective": None,
                "r_proxy": None,
                "l_proxy": None,
                "alpha_strategy": "smoke_fixed_full_vector_alpha_ones_not_full_alpha_search",
            }
        )

    _write_rows(output_dir / "j7_topology_pool.csv", topology_rows)

    alpha_requested = {load.canonical_load_id: 1.0 for load in identity.loads}
    dc_rows = []
    for item in pool:
        result = solve_fixed_topology_economic_dc_opf(
            raw_case=raw_case,
            identity=identity,
            offline_branch_ids=item.shutoff_branch_ids,
            alpha_requested=alpha_requested,
        )
        row = {
            "method": "Guided-DC",
            "rank": item.rank,
            "shutoff_branch_ids": ";".join(str(line_id) for line_id in item.shutoff_branch_ids),
            "evaluation_status": result.evaluation_status.value,
            "objective_cost": result.objective_cost,
            "message": result.message,
        }
        if result.objective_cost is not None:
            flow_loading = {
                branch.canonical_branch_id: abs(float(result.flow_by_line[branch.canonical_branch_id])) / float(branch.rate_a)
                for branch in identity.branches
                if branch.canonical_branch_id in result.flow_by_line
            }
            r_raw = sum(float(p_env.get(line_id, 0.0)) * float(loading) ** 2 for line_id, loading in flow_loading.items())
            source_less = source_less_load_ids(identity, item.shutoff_branch_ids)
            breakdown = compute_load_shedding(
                [load.canonical_load_id for load in identity.loads],
                {load.canonical_load_id: load.pd_pre for load in identity.loads},
                alpha_requested,
                source_less,
            )
            row.update(
                {
                    "r_norm": r_raw / r_base,
                    "l_shed_total": breakdown.l_shed_total,
                    "l_shed_control": breakdown.l_shed_control,
                    "l_shed_island": breakdown.l_shed_island,
                    "j_trade": compute_j_trade(args.lambda_r, r_raw / r_base, breakdown.l_shed_total),
                    "max_loading": max(flow_loading.values()) if flow_loading else None,
                    "num_loading_gt_1": sum(1 for value in flow_loading.values() if value > 1.0),
                }
            )
        dc_rows.append(row)
    _write_rows(output_dir / "j7_guided_dc_results.csv", dc_rows)

    summary = {
        "status": "PASS",
        "scenario_id": args.scenario_id,
        "lambda_r_proxy": args.lambda_r,
        "lambda_r": args.lambda_r,
        "k": args.k,
        "pool_size": args.pool_size,
        "topology_pool_csv": str(output_dir / "j7_topology_pool.csv"),
        "guided_dc_results_csv": str(output_dir / "j7_guided_dc_results.csv"),
        "alpha_strategy": "smoke_fixed_full_vector_alpha_ones_not_full_alpha_search",
        "full_per_load_alpha_search_status": "NOT_EXECUTED_IN_J7_SMOKE",
    }
    with (output_dir / "j7_proxy_and_dc_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
