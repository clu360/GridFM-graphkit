"""J7.5 v2 smoke: coordinate screen plus SciPy alpha search for Guided-DC."""

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
from stage_j_gridsfm_goc500.dc_economic_recourse import solve_fixed_topology_economic_dc_opf
from stage_j_gridsfm_goc500.goc500_adapter import build_goc500_identity, source_less_load_ids
from stage_j_gridsfm_goc500.load_service import compute_alpha_effective, compute_load_shedding
from stage_j_gridsfm_goc500.metrics import compute_j_trade
from stage_j_gridsfm_goc500.scenario_builder import load_baseline_loading_csv


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--baseline-loading-csv", required=True)
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

    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_case = _load_json(gridsfm_root / "model" / "samples" / "case500_goc.pyg.json")
    identity = build_goc500_identity(raw_case)
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    r_base = _r_base(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    shutoff = _line_ids(args.topology_ids)
    load_ids = [load.canonical_load_id for load in identity.loads]
    pd_pre = {load.canonical_load_id: load.pd_pre for load in identity.loads}
    source_less = source_less_load_ids(identity, shutoff)
    baseline_loading = load_baseline_loading_csv(args.baseline_loading_csv)

    def evaluator(alpha_requested):
        result = solve_fixed_topology_economic_dc_opf(
            raw_case=raw_case,
            identity=identity,
            offline_branch_ids=shutoff,
            alpha_requested=alpha_requested,
        )
        if result.objective_cost is None:
            return AlphaEvaluation(
                evaluation_status=result.evaluation_status.value,
                search_objective=None,
                message=result.message,
            )
        flow_loading = {
            branch.canonical_branch_id: abs(float(result.flow_by_line[branch.canonical_branch_id])) / float(branch.rate_a)
            for branch in identity.branches
            if branch.canonical_branch_id in result.flow_by_line
        }
        r_raw = 0.0
        for line_id, p_env_value in p_env.items():
            r_raw += float(p_env_value) * float(flow_loading.get(int(line_id), 0.0)) ** 2
        r_norm = r_raw / r_base
        breakdown = compute_load_shedding(load_ids, pd_pre, alpha_requested, source_less)
        j_trade = compute_j_trade(args.lambda_r, r_norm, breakdown.l_shed_total)
        return AlphaEvaluation(
            evaluation_status=result.evaluation_status.value,
            search_objective=j_trade,
            l_shed_total=breakdown.l_shed_total,
            l_shed_control=breakdown.l_shed_control,
            l_shed_island=breakdown.l_shed_island,
            r_norm=r_norm,
            j_trade=j_trade,
            message=result.message,
            extra={
                "objective_cost": result.objective_cost,
                "max_loading": max(flow_loading.values()) if flow_loading else None,
                "num_loading_gt_1": sum(1 for value in flow_loading.values() if value > 1.0),
                "baseline_max_loading": max(baseline_loading.values()) if baseline_loading else None,
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
        backend="Guided-DC",
        topology_id=";".join(str(value) for value in shutoff),
        search_run_id=f"J7_5_V2_DC_{args.scenario_id}_lambda{args.lambda_r:g}_{'_'.join(str(v) for v in shutoff)}",
        evaluator=evaluator,
        config=config,
        cache_context={
            "electrical_scenario_id": "e0",
            "wildfire_scenario_id": args.scenario_id,
            "lambda_r": args.lambda_r,
            "selection_objective": "J_trade",
            "pac_config": "not_applicable_guided_dc",
        },
    )

    _write_rows(output_dir / "j7_5_v2_guided_dc_alpha_trace.csv", list(search.trace_rows))
    _write_rows(output_dir / "j7_5_v2_guided_dc_alpha_checkpoints.csv", list(search.checkpoint_rows))
    _write_rows(output_dir / "j7_5_v2_guided_dc_screen_rows.csv", list(search.screen_rows))
    _write_rows(output_dir / "j7_5_v2_guided_dc_q_summary.csv", list(search.q_summary_rows))
    alpha_effective = compute_alpha_effective(load_ids, search.best_alpha, source_less)
    _write_alpha_csv(output_dir / "j7_5_v2_guided_dc_best_alpha_requested.csv", search.best_alpha, "alpha_requested")
    _write_alpha_csv(output_dir / "j7_5_v2_guided_dc_best_alpha_effective.csv", alpha_effective, "alpha_effective")
    best_payload = {
        "search_run_id": search.search_run_id,
        "backend": search.backend,
        "optimizer_version": "screened_scipy_v2",
        "scenario_id": args.scenario_id,
        "lambda_r": args.lambda_r,
        "topology_ids": list(shutoff),
        "source_less_load_ids": list(source_less),
        "termination_reason": search.termination_reason,
        "actual_evaluation_count": search.actual_evaluation_count,
        "unique_alpha_count": search.unique_alpha_count,
        "cache_hit_count": search.cache_hit_count,
        "screening_best_improved": search.screening_best_improved,
        "selected_loads_by_q": {str(k): list(v) for k, v in search.selected_loads_by_q.items()},
        "best_alpha_hash": search.best_alpha_hash,
        "cache_namespace_hash": search.cache_namespace_hash,
        "best_alpha_requested_csv": str(output_dir / "j7_5_v2_guided_dc_best_alpha_requested.csv"),
        "best_alpha_effective_csv": str(output_dir / "j7_5_v2_guided_dc_best_alpha_effective.csv"),
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
            "message": search.best_evaluation.message,
            "extra": dict(search.best_evaluation.extra),
        },
        "alpha_search_notes": (
            "J7.5 v2 keeps the full per-load alpha vector, screens all controllable loads with a 0.10 "
            "coordinate perturbation, then runs SciPy only over selected top-q coordinates."
        ),
    }
    with (output_dir / "j7_5_v2_guided_dc_alpha_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(best_payload, fh, indent=2, sort_keys=True)
    print(json.dumps(best_payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
