"""Calibrate and freeze Stage J GridSFM PAC weights on smoke states only."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--dominance-threshold", type=float, default=0.05)
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
    targets = _scenario_targets(input_dir / "stage_j_scenario_register.csv")
    target_list = [targets[sid][0] for sid in sorted(targets) if targets[sid]]
    smoke_states = [("intact", [])]
    smoke_states.extend((f"n1_{line_id}", [line_id]) for line_id in target_list)
    if len(target_list) >= 2:
        smoke_states.append((f"n2_{target_list[0]}_{target_list[1]}", target_list[:2]))

    model = load_model(str(checkpoint), device="cpu")
    provisional = PacWeights(rho_phys=2.0, w_op=1.0, w_ac=1.0, w_model=0.0)
    alpha_requested = {load.canonical_load_id: 1.0 for load in identity.loads}
    rows = []
    for state_id, offline in smoke_states:
        scenario_id = "J-S1"
        result = evaluate_gridsfm_candidate(
            raw_case=raw_case,
            identity=identity,
            model=model,
            offline_branch_ids=offline,
            alpha_requested=alpha_requested,
            p_env_by_line=_p_env(input_dir / "stage_j_p_env_by_scenario.csv", scenario_id),
            r_base=_r_base(input_dir / "stage_j_scenario_register.csv", scenario_id),
            lambda_r=args.lambda_r,
            weights=provisional,
            work_dir=output_dir / "mutated_candidates" / state_id,
        )
        obj = result.objective
        rows.append(
            {
                "state_id": state_id,
                "offline_branch_ids": ";".join(str(x) for x in offline),
                "evaluation_status": result.evaluation_status.value,
                "j_trade": None if obj is None else obj.j_trade,
                "pac_operational": None if obj is None else obj.pac_operational,
                "pac_ac": None if obj is None else obj.pac_ac,
                "pac_model": None if obj is None else obj.pac_model,
                "pac_total_provisional": None if obj is None else obj.pac_total,
                "rho_pac_over_j_trade": None
                if obj is None or obj.j_trade == 0.0
                else provisional.rho_phys * float(obj.pac_total) / abs(float(obj.j_trade)),
                "feasibility_head": result.feasibility_head,
                "max_loading": max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
                "num_loading_gt_1": sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
                "message": result.message,
            }
        )

    invalid_rows = [row for row in rows if row["j_trade"] is None or row["pac_total_provisional"] is None]
    if invalid_rows:
        failure = {
            "status": "FAIL",
            "reason": "PAC calibration requires every smoke state to produce objective and PAC components.",
            "invalid_state_ids": [row["state_id"] for row in invalid_rows],
            "num_smoke_states": len(rows),
        }
        with (output_dir / "PAC_WEIGHT_FREEZE.json").open("w") as fh:
            json.dump(failure, fh, indent=2, sort_keys=True)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1

    ratios = [float(row["rho_pac_over_j_trade"]) for row in rows if row["rho_pac_over_j_trade"] is not None]
    if not ratios:
        failure = {
            "status": "FAIL",
            "reason": "PAC calibration produced no finite rho*PAC/J_trade ratios.",
            "num_smoke_states": len(rows),
        }
        with (output_dir / "PAC_WEIGHT_FREEZE.json").open("w") as fh:
            json.dump(failure, fh, indent=2, sort_keys=True)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    median_ratio = statistics.median(ratios) if ratios else float("inf")
    max_ratio = max(ratios) if ratios else float("inf")
    if median_ratio <= float(args.dominance_threshold):
        frozen = provisional
        calibration_rule = "identity_weights_kept_because_median_rho_pac_over_j_trade_within_threshold"
    else:
        scale = float(args.dominance_threshold) / max(median_ratio, 1e-12)
        frozen = PacWeights(rho_phys=2.0, w_op=scale, w_ac=scale, w_model=0.0)
        calibration_rule = "op_ac_weights_scaled_to_threshold_using_smoke_median"

    rows_path = output_dir / "pac_calibration_smoke_rows.csv"
    with rows_path.open("w", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    freeze = {
        "status": "PASS",
        "calibration_scope": "intact_plus_selected_n1_n2_smoke_states_only",
        "calibration_rule": calibration_rule,
        "dominance_threshold_median_rho_pac_over_j_trade": args.dominance_threshold,
        "median_rho_pac_over_j_trade_provisional": median_ratio,
        "max_rho_pac_over_j_trade_provisional": max_ratio,
        "frozen_weights": {
            "rho_phys": frozen.rho_phys,
            "w_op": frozen.w_op,
            "w_ac": frozen.w_ac,
            "w_model": frozen.w_model,
        },
        "pac_model_status": "ZERO_WEIGHTED_MODEL_FLOWS_DETERMINISTIC_FROM_V_THETA_IN_OFFICIAL_GRIDSFM_MODEL",
        "smoke_rows_csv": str(rows_path),
        "num_smoke_states": len(rows),
    }
    freeze_path = output_dir / "PAC_WEIGHT_FREEZE.json"
    with freeze_path.open("w") as fh:
        json.dump(freeze, fh, indent=2, sort_keys=True)
    print(json.dumps(freeze, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
