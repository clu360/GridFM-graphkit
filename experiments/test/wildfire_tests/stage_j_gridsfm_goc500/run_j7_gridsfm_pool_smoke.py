"""J7 one-scenario smoke: evaluate shared topology pool with GridSFM."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity
from stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from stage_j_gridsfm_goc500.schemas import PacWeights


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


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


def _load_pool(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def _line_ids(value: str) -> tuple[int, ...]:
    return tuple(sorted(int(part) for part in str(value).split(";") if part))


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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--topology-pool-csv", required=True)
    parser.add_argument("--pac-freeze-json", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from gridsfm import load_model

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_root = gridsfm_root / "model"
    raw_case_path = model_root / "samples" / "case500_goc.pyg.json"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_case = _load_json(raw_case_path)
    identity = build_goc500_identity(raw_case)
    freeze = _load_json(Path(args.pac_freeze_json))
    frozen = freeze["frozen_weights"]
    weights = PacWeights(
        rho_phys=float(frozen["rho_phys"]),
        w_op=float(frozen["w_op"]),
        w_ac=float(frozen["w_ac"]),
        w_model=float(frozen["w_model"]),
    )
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    r_base = _r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    alpha_requested = {load.canonical_load_id: 1.0 for load in identity.loads}

    model = load_model(str(checkpoint), device="cpu")
    rows = []
    cache = {}
    for pool_row in _load_pool(Path(args.topology_pool_csv)):
        shutoff = _line_ids(pool_row["shutoff_branch_ids"])
        if shutoff not in cache:
            cache[shutoff] = evaluate_gridsfm_candidate(
                raw_case=raw_case,
                identity=identity,
                model=model,
                offline_branch_ids=shutoff,
                alpha_requested=alpha_requested,
                p_env_by_line=p_env,
                r_base=r_base,
                lambda_r=args.lambda_r,
                weights=weights,
                work_dir=output_dir / "mutated_candidates" / ("lines_" + "_".join(str(x) for x in shutoff or ["intact"])),
            )
        result = cache[shutoff]
        obj = result.objective
        rows.append(
            {
                "method": "Guided-GridSFM" if pool_row["method_family"] == "Guided" else pool_row["method_family"],
                "rank": pool_row["rank"],
                "shutoff_branch_ids": pool_row["shutoff_branch_ids"],
                "evaluation_status": result.evaluation_status.value,
                "r_norm": None if obj is None else obj.r_norm,
                "l_shed_total": None if obj is None else obj.l_shed_total,
                "l_shed_control": None if result.load_shedding is None else result.load_shedding.l_shed_control,
                "l_shed_island": None if result.load_shedding is None else result.load_shedding.l_shed_island,
                "j_trade": None if obj is None else obj.j_trade,
                "pac_operational": None if obj is None else obj.pac_operational,
                "pac_ac": None if obj is None else obj.pac_ac,
                "pac_model": None if obj is None else obj.pac_model,
                "pac_total": None if obj is None else obj.pac_total,
                "j_total": None if obj is None else obj.j_total,
                "d_input": result.d_input,
                "feasibility_head": result.feasibility_head,
                "max_loading": max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
                "num_loading_gt_1": sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
                "pac_model_components": json.dumps(dict(result.pac_model_components), sort_keys=True),
                "pac_notes": result.pac_notes,
                "alpha_strategy": pool_row["alpha_strategy"],
                "message": result.message,
            }
        )

    results_path = output_dir / "j7_gridsfm_results.csv"
    _write_rows(results_path, rows)
    summary = {
        "status": "PASS",
        "scenario_id": args.scenario_id,
        "lambda_r": args.lambda_r,
        "results_csv": str(results_path),
        "num_rows": len(rows),
        "num_unique_topologies_evaluated": len(cache),
        "alpha_strategy": "smoke_fixed_full_vector_alpha_ones_not_full_alpha_search",
        "full_per_load_alpha_search_status": "NOT_EXECUTED_IN_J7_SMOKE",
        "pac_freeze_json": str(Path(args.pac_freeze_json).expanduser().resolve()),
    }
    with (output_dir / "j7_gridsfm_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
