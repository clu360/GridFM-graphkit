"""Measure repeatability/noise for one saved GridSFM Stage J candidate."""

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
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _p_env(path: Path, scenario_id: str) -> dict[int, float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                out[int(row["branch_id"])] = float(row["p_env"])
    return out


def _r_base(path: Path, scenario_id: str) -> float:
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                return float(row["r_base"])
    raise KeyError(f"scenario {scenario_id} missing from scenario register")


def _line_ids(value: str) -> tuple[int, ...]:
    return tuple(sorted(int(part) for part in str(value).replace(",", ";").split(";") if part.strip()))


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _spread(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "mean": statistics.mean(values),
        "stdev": statistics.pstdev(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pac-freeze-json", required=True)
    parser.add_argument("--alpha-summary-json", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--topology-ids", default="285;473")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from gridsfm import load_model

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    model_root = gridsfm_root / "model"
    checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_case = _load_json(model_root / "samples" / "case500_goc.pyg.json")
    identity = build_goc500_identity(raw_case)
    alpha_summary = _load_json(Path(args.alpha_summary_json).expanduser().resolve())
    alpha_requested = {int(k): float(v) for k, v in alpha_summary["best_alpha"].items()}
    freeze = _load_json(Path(args.pac_freeze_json).expanduser().resolve())
    frozen = freeze["frozen_weights"]
    weights = PacWeights(
        rho_phys=float(frozen["rho_phys"]),
        w_op=float(frozen["w_op"]),
        w_ac=float(frozen["w_ac"]),
        w_model=float(frozen["w_model"]),
    )
    model = load_model(str(checkpoint), device="cpu")

    rows = []
    for rep in range(1, args.repetitions + 1):
        result = evaluate_gridsfm_candidate(
            raw_case=raw_case,
            identity=identity,
            model=model,
            offline_branch_ids=_line_ids(args.topology_ids),
            alpha_requested=alpha_requested,
            p_env_by_line=_p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id),
            r_base=_r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id),
            lambda_r=args.lambda_r,
            weights=weights,
            work_dir=output_dir / f"repeat_{rep:03d}",
        )
        obj = result.objective
        rows.append(
            {
                "repeat": rep,
                "evaluation_status": result.evaluation_status.value,
                "r_norm": None if obj is None else obj.r_norm,
                "l_shed_total": None if obj is None else obj.l_shed_total,
                "j_trade": None if obj is None else obj.j_trade,
                "pac_total": None if obj is None else obj.pac_total,
                "j_total": None if obj is None else obj.j_total,
                "d_input": result.d_input,
                "max_loading": max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
                "num_loading_gt_1": sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
            }
        )

    _write_rows(output_dir / "gridsfm_repeatability_rows.csv", rows)
    numeric_cols = ["r_norm", "l_shed_total", "j_trade", "pac_total", "j_total", "max_loading"]
    spreads = {
        col: _spread([float(row[col]) for row in rows if row[col] is not None])
        for col in numeric_cols
        if all(row[col] is not None for row in rows)
    }
    max_objective_range = max(spreads[col]["range"] for col in ("j_trade", "j_total") if col in spreads)
    payload = {
        "scenario_id": args.scenario_id,
        "lambda_r": args.lambda_r,
        "topology_ids": list(_line_ids(args.topology_ids)),
        "repetitions": args.repetitions,
        "rows_csv": str(output_dir / "gridsfm_repeatability_rows.csv"),
        "spreads": spreads,
        "max_objective_range": max_objective_range,
        "recommended_epsilon_abs_floor": max(1e-9, 10.0 * max_objective_range),
        "notes": "Recommended floor is 10x observed repeated-evaluation objective range for this smoke candidate only.",
    }
    with (output_dir / "gridsfm_repeatability_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
