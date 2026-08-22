"""J7.5 v2 smoke: coordinate screen plus SciPy alpha search for GridSFM."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.alpha_optimizer import (
    AlphaEvaluation,
    ScreenedScipyAlphaConfig,
    optimize_screened_scipy_alpha,
)
from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity, source_less_load_ids
from stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from stage_j_gridsfm_goc500.load_service import compute_alpha_effective
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
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_alpha_csv(path: Path, alpha: dict[int, float], column_name: str) -> None:
    _write_rows(path, [{"load_id": load_id, column_name: value} for load_id, value in sorted(alpha.items())])


def _best_numeric_row(rows: list[dict[str, object]], column: str) -> dict[str, object] | None:
    candidates = []
    for row in rows:
        value = row.get(column)
        if value == "" or value is None:
            continue
        try:
            candidates.append((float(value), row))
        except (TypeError, ValueError):
            continue
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else None


def _write_best_flow_rows(path: Path, result) -> None:
    rows = []
    for line_id in sorted(result.flow_loading_by_line):
        rows.append(
            {
                "branch_id": line_id,
                "loading": result.flow_loading_by_line[line_id],
                "p_from": result.p_from_by_line.get(line_id),
                "q_from": result.q_from_by_line.get(line_id),
                "p_to": result.p_to_by_line.get(line_id),
                "q_to": result.q_to_by_line.get(line_id),
            }
        )
    _write_rows(path, rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pac-freeze-json", required=True)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.5)
    parser.add_argument("--topology-ids", default="285;473")
    parser.add_argument("--b-alpha", type=int, default=500)
    parser.add_argument("--screen-delta", type=float, default=0.10)
    parser.add_argument("--q-values", default="5,10,20")
    parser.add_argument("--scipy-maxfev-per-q", type=int, default=60)
    parser.add_argument("--alpha-round-decimals", type=int, default=6)
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
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    r_base = _r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    shutoff = _line_ids(args.topology_ids)
    load_ids = [load.canonical_load_id for load in identity.loads]
    source_less = source_less_load_ids(identity, shutoff)
    freeze = _load_json(Path(args.pac_freeze_json).expanduser().resolve())
    frozen = freeze["frozen_weights"]
    weights = PacWeights(
        rho_phys=float(frozen["rho_phys"]),
        w_op=float(frozen["w_op"]),
        w_ac=float(frozen["w_ac"]),
        w_model=float(frozen["w_model"]),
    )
    model = load_model(str(checkpoint), device="cpu")

    eval_counter = {"count": 0}

    def detailed_eval(alpha_requested, work_suffix: str):
        return evaluate_gridsfm_candidate(
            raw_case=raw_case,
            identity=identity,
            model=model,
            offline_branch_ids=shutoff,
            alpha_requested=alpha_requested,
            p_env_by_line=p_env,
            r_base=r_base,
            lambda_r=args.lambda_r,
            weights=weights,
            work_dir=output_dir / "mutated_candidates" / work_suffix,
        )

    def evaluator(alpha_requested):
        eval_counter["count"] += 1
        result = detailed_eval(alpha_requested, f"alpha_eval_{eval_counter['count']:05d}")
        obj = result.objective
        if obj is None:
            return AlphaEvaluation(
                evaluation_status=result.evaluation_status.value,
                search_objective=None,
                d_input=result.d_input,
                message=result.message,
            )
        return AlphaEvaluation(
            evaluation_status=result.evaluation_status.value,
            search_objective=obj.j_total,
            l_shed_total=obj.l_shed_total,
            l_shed_control=None if result.load_shedding is None else result.load_shedding.l_shed_control,
            l_shed_island=None if result.load_shedding is None else result.load_shedding.l_shed_island,
            r_norm=obj.r_norm,
            j_trade=obj.j_trade,
            pac_operational=obj.pac_operational,
            pac_ac=obj.pac_ac,
            pac_model=obj.pac_model,
            pac_total=obj.pac_total,
            j_total=obj.j_total,
            d_input=result.d_input,
            message=result.message,
            extra={
                "feasibility_head": result.feasibility_head,
                "max_loading": max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
                "num_loading_gt_1": sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
                "pac_model_components": dict(result.pac_model_components),
                "pac_notes": result.pac_notes,
            },
        )

    config = ScreenedScipyAlphaConfig(
        b_alpha=args.b_alpha,
        screen_delta=args.screen_delta,
        q_values=tuple(int(value) for value in args.q_values.split(",") if value.strip()),
        scipy_maxfev_per_q=args.scipy_maxfev_per_q,
        alpha_round_decimals=args.alpha_round_decimals,
    )
    search = optimize_screened_scipy_alpha(
        load_ids=load_ids,
        fixed_zero_load_ids=source_less,
        offline_branch_ids=shutoff,
        backend="Guided-GridSFM",
        topology_id=";".join(str(value) for value in shutoff),
        search_run_id=f"J7_5_V2_GRIDSFM_{args.scenario_id}_lambda{args.lambda_r:g}_{'_'.join(str(v) for v in shutoff)}",
        evaluator=evaluator,
        config=config,
        cache_context={
            "electrical_scenario_id": "e0",
            "wildfire_scenario_id": args.scenario_id,
            "lambda_r": args.lambda_r,
            "selection_objective": "J_total",
            "rho_phys": weights.rho_phys,
            "w_op": weights.w_op,
            "w_ac": weights.w_ac,
            "w_model": weights.w_model,
        },
    )

    trace_rows = list(search.trace_rows)
    _write_rows(output_dir / "j7_5_v2_gridsfm_alpha_trace.csv", trace_rows)
    _write_rows(output_dir / "j7_5_v2_gridsfm_alpha_checkpoints.csv", list(search.checkpoint_rows))
    _write_rows(output_dir / "j7_5_v2_gridsfm_screen_rows.csv", list(search.screen_rows))
    _write_rows(output_dir / "j7_5_v2_gridsfm_q_summary.csv", list(search.q_summary_rows))
    alpha_effective = compute_alpha_effective(load_ids, search.best_alpha, source_less)
    _write_alpha_csv(output_dir / "j7_5_v2_gridsfm_best_alpha_requested.csv", search.best_alpha, "alpha_requested")
    _write_alpha_csv(output_dir / "j7_5_v2_gridsfm_best_alpha_effective.csv", alpha_effective, "alpha_effective")

    best_result = detailed_eval(search.best_alpha, "best_alpha_finalist")
    _write_best_flow_rows(output_dir / "j7_5_v2_gridsfm_best_flow_loading.csv", best_result)

    best_j_trade_row = _best_numeric_row(trace_rows, "J_trade")
    best_j_total_row = _best_numeric_row(trace_rows, "J_total")
    best_payload = {
        "search_run_id": search.search_run_id,
        "backend": search.backend,
        "optimizer_version": "screened_scipy_v2",
        "scenario_id": args.scenario_id,
        "lambda_r": args.lambda_r,
        "topology_ids": list(shutoff),
        "source_less_load_ids": list(source_less),
        "pac_weights": {
            "rho_phys": weights.rho_phys,
            "w_op": weights.w_op,
            "w_ac": weights.w_ac,
            "w_model": weights.w_model,
        },
        "termination_reason": search.termination_reason,
        "actual_evaluation_count": search.actual_evaluation_count,
        "unique_alpha_count": search.unique_alpha_count,
        "cache_hit_count": search.cache_hit_count,
        "screening_best_improved": search.screening_best_improved,
        "selected_loads_by_q": {str(k): list(v) for k, v in search.selected_loads_by_q.items()},
        "best_alpha_hash": search.best_alpha_hash,
        "cache_namespace_hash": search.cache_namespace_hash,
        "best_alpha_requested_csv": str(output_dir / "j7_5_v2_gridsfm_best_alpha_requested.csv"),
        "best_alpha_effective_csv": str(output_dir / "j7_5_v2_gridsfm_best_alpha_effective.csv"),
        "best_alpha": {str(k): v for k, v in sorted(search.best_alpha.items())},
        "best_evaluation": None
        if search.best_evaluation is None
        else {
            "evaluation_status": search.best_evaluation.evaluation_status,
            "search_objective": search.best_evaluation.search_objective,
            "l_shed_total": search.best_evaluation.l_shed_total,
            "l_shed_control": search.best_evaluation.l_shed_control,
            "l_shed_island": search.best_evaluation.l_shed_island,
            "r_norm": search.best_evaluation.r_norm,
            "j_trade": search.best_evaluation.j_trade,
            "pac_operational": search.best_evaluation.pac_operational,
            "pac_ac": search.best_evaluation.pac_ac,
            "pac_model": search.best_evaluation.pac_model,
            "pac_total": search.best_evaluation.pac_total,
            "j_total": search.best_evaluation.j_total,
            "d_input": search.best_evaluation.d_input,
            "message": search.best_evaluation.message,
            "extra": dict(search.best_evaluation.extra),
        },
        "argmin_j_trade_row": best_j_trade_row,
        "argmin_j_total_row": best_j_total_row,
        "best_flow_loading_csv": str(output_dir / "j7_5_v2_gridsfm_best_flow_loading.csv"),
        "alpha_search_notes": (
            "J7.5 v2 GridSFM candidate selection minimizes J_total. The all-load coordinate screen "
            "selects top-q coordinates for bounded SciPy refinement; argmin J_trade is diagnostic only."
        ),
    }
    with (output_dir / "j7_5_v2_gridsfm_alpha_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2, sort_keys=True)
    print(json.dumps(best_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
