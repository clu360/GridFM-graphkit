"""J5/J7 smoke for one real GridSFM Stage J candidate evaluation."""

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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--rho-phys", type=float, default=2.0)
    parser.add_argument("--uniform-alpha", type=float, default=1.0)
    parser.add_argument("--target-load-id", type=int, default=None)
    parser.add_argument("--target-alpha", type=float, default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    from gridsfm import load_model

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_root = gridsfm_root / "model"
    raw_case_path = model_root / "samples" / "case500_goc.pyg.json"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)

    raw_case = _load_json(raw_case_path)
    identity = build_goc500_identity(raw_case)
    targets = _scenario_targets(input_dir / "stage_j_scenario_register.csv")
    if args.scenario_id not in targets:
        raise KeyError(f"scenario {args.scenario_id} missing from scenario register")
    offline_branch_ids = targets[args.scenario_id][:1]
    alpha_requested = {load.canonical_load_id: float(args.uniform_alpha) for load in identity.loads}
    if args.target_load_id is not None:
        if args.target_alpha is None:
            raise ValueError("--target-alpha is required when --target-load-id is supplied")
        alpha_requested[int(args.target_load_id)] = float(args.target_alpha)

    model = load_model(str(checkpoint), device="cpu")
    result = evaluate_gridsfm_candidate(
        raw_case=raw_case,
        identity=identity,
        model=model,
        offline_branch_ids=offline_branch_ids,
        alpha_requested=alpha_requested,
        p_env_by_line=_p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id),
        r_base=_r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id),
        lambda_r=args.lambda_r,
        weights=PacWeights(rho_phys=args.rho_phys, w_op=1.0, w_ac=1.0, w_model=0.0),
        work_dir=output_json.parent / "mutated_candidates",
    )

    objective = result.objective
    payload = {
        "status": "PASS" if objective is not None else "FAIL",
        "scenario_id": args.scenario_id,
        "offline_branch_ids": offline_branch_ids,
        "lambda_r": args.lambda_r,
        "rho_phys": args.rho_phys,
        "uniform_alpha": args.uniform_alpha,
        "target_load_id": args.target_load_id,
        "target_alpha": args.target_alpha,
        "evaluation_status": result.evaluation_status.value,
        "r_norm": None if objective is None else objective.r_norm,
        "l_shed_total": None if objective is None else objective.l_shed_total,
        "j_trade": None if objective is None else objective.j_trade,
        "pac_operational": None if objective is None else objective.pac_operational,
        "pac_ac": None if objective is None else objective.pac_ac,
        "pac_model": None if objective is None else objective.pac_model,
        "pac_total": None if objective is None else objective.pac_total,
        "j_total": None if objective is None else objective.j_total,
        "d_input": result.d_input,
        "feasibility_head": result.feasibility_head,
        "max_loading": max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
        "num_loading_gt_1": sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
        "l_shed_control": None if result.load_shedding is None else result.load_shedding.l_shed_control,
        "l_shed_island": None if result.load_shedding is None else result.load_shedding.l_shed_island,
        "pac_model_components": dict(result.pac_model_components),
        "pac_notes": result.pac_notes,
        "message": result.message,
    }
    with output_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
