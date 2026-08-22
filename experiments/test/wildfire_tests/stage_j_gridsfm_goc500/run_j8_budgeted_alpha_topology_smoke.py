"""J8 smoke: topology pool with shallow budgeted alpha correction.

This runner is intentionally small-budget. It does not run the full all-load
coordinate screen from J7.5. For each topology, it chooses a cheap deterministic
q-load subset and spends at most `continuous_eval_budget` backend evaluations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

WILDFIRE_TESTS_ROOT = Path(__file__).resolve().parents[1]
if str(WILDFIRE_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(WILDFIRE_TESTS_ROOT))

from stage_j_gridsfm_goc500.dc_economic_recourse import solve_fixed_topology_economic_dc_opf
from stage_j_gridsfm_goc500.goc500_adapter import GOC500Identity, build_goc500_identity, source_less_load_ids
from stage_j_gridsfm_goc500.gridsfm_evaluator import evaluate_gridsfm_candidate
from stage_j_gridsfm_goc500.load_service import compute_load_shedding
from stage_j_gridsfm_goc500.metrics import compute_j_trade
from stage_j_gridsfm_goc500.outer_proxy import ProxyTopology, solve_proxy_topology_pool
from stage_j_gridsfm_goc500.scenario_builder import load_baseline_loading_csv
from stage_j_gridsfm_goc500.schemas import EvaluationStatus, PacWeights


@dataclass(frozen=True)
class CandidateEvaluation:
    evaluation_status: str
    search_objective: float | None
    r_norm: float | None = None
    l_shed_total: float | None = None
    l_shed_control: float | None = None
    l_shed_island: float | None = None
    j_trade: float | None = None
    pac_operational: float | None = None
    pac_ac: float | None = None
    pac_model: float | None = None
    pac_total: float | None = None
    j_total: float | None = None
    d_input: float | None = None
    feasibility_head: float | None = None
    max_loading: float | None = None
    num_loading_gt_1: int | None = None
    objective_cost: float | None = None
    message: str = ""
    extra: Mapping[str, object] | None = None

    @property
    def eligible(self) -> bool:
        return self.search_objective is not None and math.isfinite(float(self.search_objective))


def _load_json(path: Path):
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _load_candidate_scores(path: Path) -> dict[int, float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[int(row["branch_id"])] = float(row["c_l"])
    return out


def _p_env(path: Path, scenario_id: str) -> dict[int, float]:
    out = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                out[int(row["branch_id"])] = float(row["p_env"])
    return out


def _scenario_row(path: Path, scenario_id: str) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["scenario_id"] == scenario_id:
                return row
    raise KeyError(f"scenario {scenario_id} missing from scenario register")


def _line_ids(value: str) -> tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    return tuple(sorted(int(part) for part in text.replace(",", ";").split(";") if part.strip()))


def _line_key(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in sorted(int(v) for v in values))


def _maybe(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _optional_float(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: _maybe(value) for key, value in row.items()} for row in rows])


def _read_topology_pool(path: Path, *, method_family: str = "Guided") -> list[ProxyTopology]:
    out = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("method_family", "Guided") != method_family:
                continue
            out.append(
                ProxyTopology(
                    rank=int(row["rank"]),
                    shutoff_branch_ids=_line_ids(row["shutoff_branch_ids"]),
                    proxy_objective=_optional_float(row.get("proxy_objective")),
                    r_proxy=_optional_float(row.get("r_proxy")),
                    l_proxy=_optional_float(row.get("l_proxy")),
                )
            )
    return out


def _write_topology_pool(path: Path, pool: list[ProxyTopology]) -> None:
    _write_rows(
        path,
        [
            {
                "method_family": "Guided",
                "rank": item.rank,
                "shutoff_branch_ids": _line_key(item.shutoff_branch_ids),
                "num_shutoffs": len(item.shutoff_branch_ids),
                "proxy_objective": item.proxy_objective,
                "r_proxy": item.r_proxy,
                "l_proxy": item.l_proxy,
                "alpha_strategy": "j8_budgeted_subset_scipy_no_full_coordinate_screen",
            }
            for item in pool
        ],
    )


def build_th_gridsfm_topology_pool(
    *,
    candidate_branch_ids: Iterable[int],
    p_env_by_line: Mapping[int, float],
    baseline_loading: Mapping[int, float],
    k: int = 2,
) -> list[ProxyTopology]:
    """Build explicit TH-1 and TH-2 topologies from wildfire baseline scores."""

    candidate_ids = sorted(int(value) for value in candidate_branch_ids)
    if not candidate_ids:
        raise ValueError("candidate_branch_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be positive for TH-GridSFM")
    scores = {
        line_id: float(p_env_by_line.get(line_id, 0.0)) * float(baseline_loading[line_id]) ** 2
        for line_id in candidate_ids
    }
    denom = sum(scores.values())
    if denom <= 0.0 or not math.isfinite(float(denom)):
        raise ValueError(f"TH wildfire score denominator must be positive and finite, got {denom}")
    ranked = sorted(candidate_ids, key=lambda line_id: (-scores[line_id], line_id))
    pool: list[ProxyTopology] = []
    for th_k in range(1, min(int(k), 2) + 1):
        selected = tuple(sorted(ranked[:th_k]))
        remaining = sum(scores[line_id] for line_id in candidate_ids if line_id not in selected) / denom
        pool.append(
            ProxyTopology(
                rank=th_k,
                shutoff_branch_ids=selected,
                proxy_objective=None,
                r_proxy=float(remaining),
                l_proxy=None,
            )
        )
    return pool


def _write_th_gridsfm_topology_pool(path: Path, pool: list[ProxyTopology]) -> None:
    _write_rows(
        path,
        [
            {
                "method_family": "TH-GridSFM",
                "rank": item.rank,
                "th_k": len(item.shutoff_branch_ids),
                "shutoff_branch_ids": _line_key(item.shutoff_branch_ids),
                "num_shutoffs": len(item.shutoff_branch_ids),
                "proxy_objective": None,
                "r_proxy": item.r_proxy,
                "l_proxy": None,
                "alpha_strategy": "j8_budgeted_subset_scipy_no_full_coordinate_screen",
                "topology_policy": "score_TH=p_env*baseline_loading^2",
            }
            for item in pool
        ],
    )


def select_budgeted_alpha_loads(
    identity: GOC500Identity,
    offline_branch_ids: Iterable[int],
    *,
    q: int,
    exclude_load_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Cheap deterministic q-load selector for shallow alpha correction.

    Loads on/near switched-line endpoints are preferred. Remaining slots are
    filled by largest pre-intervention demand. This avoids a GridSFM/DC screen.
    """

    offline = {int(v) for v in offline_branch_ids}
    excluded = {int(v) for v in exclude_load_ids}
    if q <= 0:
        return ()

    endpoint_buses: set[int] = set()
    for branch in identity.branches:
        if branch.canonical_branch_id in offline:
            endpoint_buses.add(branch.from_bus_id)
            endpoint_buses.add(branch.to_bus_id)

    adjacency: dict[int, set[int]] = {bus_id: set() for bus_id in identity.bus_ids}
    for branch in identity.branches:
        if branch.canonical_branch_id in offline:
            continue
        adjacency.setdefault(branch.from_bus_id, set()).add(branch.to_bus_id)
        adjacency.setdefault(branch.to_bus_id, set()).add(branch.from_bus_id)

    distance = {bus_id: math.inf for bus_id in identity.bus_ids}
    queue: deque[int] = deque()
    for bus_id in endpoint_buses:
        if bus_id in distance:
            distance[bus_id] = 0
            queue.append(bus_id)
    while queue:
        bus_id = queue.popleft()
        for nxt in adjacency.get(bus_id, ()):
            if distance[nxt] == math.inf:
                distance[nxt] = distance[bus_id] + 1
                queue.append(nxt)

    loads = [load for load in identity.loads if load.canonical_load_id not in excluded]
    if endpoint_buses:
        ranked = sorted(loads, key=lambda load: (distance.get(load.bus_id, math.inf), -float(load.pd_pre), load.canonical_load_id))
    else:
        ranked = sorted(loads, key=lambda load: (-float(load.pd_pre), load.canonical_load_id))
    return tuple(load.canonical_load_id for load in ranked[: int(q)])


def run_budgeted_alpha_search(
    *,
    load_ids: Iterable[int],
    selected_load_ids: Iterable[int],
    offline_branch_ids: Iterable[int],
    method: str,
    topology_rank: int,
    evaluator: Callable[[Mapping[int, float]], CandidateEvaluation],
    continuous_eval_budget: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[int, float]]:
    """Run bounded Powell search over selected loads with a hard call budget."""

    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - scipy is tested separately
        raise RuntimeError(f"scipy is required for J8 budgeted alpha search: {exc}") from exc

    ids = [int(v) for v in load_ids]
    selected = tuple(int(v) for v in selected_load_ids)
    base_alpha = {load_id: 1.0 for load_id in ids}
    trace: list[dict[str, object]] = []
    cache: dict[tuple[float, ...], CandidateEvaluation] = {}
    best_alpha = dict(base_alpha)
    best_eval: CandidateEvaluation | None = None

    def evaluate_vector(x_values) -> CandidateEvaluation:
        nonlocal best_alpha, best_eval
        rounded = tuple(round(float(np.clip(v, 0.0, 1.0)), 8) for v in x_values)
        if rounded in cache:
            return cache[rounded]
        if len(cache) >= int(continuous_eval_budget):
            return CandidateEvaluation(
                evaluation_status="budget_rejection",
                search_objective=None,
                message="continuous_eval_budget exhausted",
            )
        alpha = dict(base_alpha)
        for load_id, value in zip(selected, rounded):
            alpha[load_id] = float(value)
        start = time.time()
        evaluation = evaluator(alpha)
        runtime = time.time() - start
        cache[rounded] = evaluation
        row = {
            "method": method,
            "topology_rank": topology_rank,
            "topology_id": _line_key(offline_branch_ids),
            "candidate_index": len(cache),
            "candidate_type": "powell_budgeted_subset",
            "selected_load_ids": _line_key(selected),
            "selected_alpha_values": json.dumps({str(load_id): alpha[load_id] for load_id in selected}, sort_keys=True),
            "evaluation_status": evaluation.evaluation_status,
            "search_objective": _maybe(evaluation.search_objective),
            "r_norm": _maybe(evaluation.r_norm),
            "l_shed_total": _maybe(evaluation.l_shed_total),
            "l_shed_control": _maybe(evaluation.l_shed_control),
            "l_shed_island": _maybe(evaluation.l_shed_island),
            "j_trade": _maybe(evaluation.j_trade),
            "pac_operational": _maybe(evaluation.pac_operational),
            "pac_ac": _maybe(evaluation.pac_ac),
            "pac_model": _maybe(evaluation.pac_model),
            "pac_total": _maybe(evaluation.pac_total),
            "j_total": _maybe(evaluation.j_total),
            "d_input": _maybe(evaluation.d_input),
            "feasibility_head": _maybe(evaluation.feasibility_head),
            "max_loading": _maybe(evaluation.max_loading),
            "num_loading_gt_1": _maybe(evaluation.num_loading_gt_1),
            "objective_cost": _maybe(evaluation.objective_cost),
            "runtime_seconds": runtime,
            "message": evaluation.message,
        }
        for key, value in sorted((evaluation.extra or {}).items()):
            row[f"extra_{key}"] = _maybe(value)
        trace.append(row)
        if evaluation.eligible and (best_eval is None or float(evaluation.search_objective) < float(best_eval.search_objective)):
            best_eval = evaluation
            best_alpha = alpha
        return evaluation

    if not selected:
        evaluate_vector(())
    else:
        def objective(x_values) -> float:
            evaluation = evaluate_vector(x_values)
            if not evaluation.eligible:
                return 1e12
            return float(evaluation.search_objective)

        minimize(
            objective,
            np.ones(len(selected), dtype=float),
            method="Powell",
            bounds=[(0.0, 1.0)] * len(selected),
            options={"maxfev": int(continuous_eval_budget), "disp": False},
        )

    if best_eval is None:
        summary = {
            "method": method,
            "topology_rank": topology_rank,
            "topology_id": _line_key(offline_branch_ids),
            "selected_load_ids": _line_key(selected),
            "actual_evaluation_count": len(cache),
            "best_found": False,
            "evaluation_status": "no_eligible_candidate",
        }
        return summary, trace, best_alpha

    summary = {
        "method": method,
        "topology_rank": topology_rank,
        "topology_id": _line_key(offline_branch_ids),
        "selected_load_ids": _line_key(selected),
        "actual_evaluation_count": len(cache),
        "best_found": True,
        "evaluation_status": best_eval.evaluation_status,
        "search_objective": _maybe(best_eval.search_objective),
        "r_norm": _maybe(best_eval.r_norm),
        "l_shed_total": _maybe(best_eval.l_shed_total),
        "l_shed_control": _maybe(best_eval.l_shed_control),
        "l_shed_island": _maybe(best_eval.l_shed_island),
        "j_trade": _maybe(best_eval.j_trade),
        "pac_operational": _maybe(best_eval.pac_operational),
        "pac_ac": _maybe(best_eval.pac_ac),
        "pac_model": _maybe(best_eval.pac_model),
        "pac_total": _maybe(best_eval.pac_total),
        "j_total": _maybe(best_eval.j_total),
        "d_input": _maybe(best_eval.d_input),
        "feasibility_head": _maybe(best_eval.feasibility_head),
        "max_loading": _maybe(best_eval.max_loading),
        "num_loading_gt_1": _maybe(best_eval.num_loading_gt_1),
        "objective_cost": _maybe(best_eval.objective_cost),
        "best_alpha_selected": json.dumps({str(load_id): best_alpha[load_id] for load_id in selected}, sort_keys=True),
        "message": best_eval.message,
    }
    for key, value in sorted((best_eval.extra or {}).items()):
        summary[f"extra_{key}"] = _maybe(value)
    return summary, trace, best_alpha


def _make_dc_evaluator(*, raw_case, identity, shutoff, p_env, r_base, lambda_r) -> Callable[[Mapping[int, float]], CandidateEvaluation]:
    load_ids = [load.canonical_load_id for load in identity.loads]
    pd_pre = {load.canonical_load_id: load.pd_pre for load in identity.loads}
    source_less = source_less_load_ids(identity, shutoff)

    def evaluator(alpha_requested: Mapping[int, float]) -> CandidateEvaluation:
        result = solve_fixed_topology_economic_dc_opf(
            raw_case=raw_case,
            identity=identity,
            offline_branch_ids=shutoff,
            alpha_requested=alpha_requested,
        )
        if result.evaluation_status is not EvaluationStatus.OK or result.objective_cost is None:
            return CandidateEvaluation(
                evaluation_status=result.evaluation_status.value,
                search_objective=None,
                objective_cost=result.objective_cost,
                message=result.message,
            )
        flow_loading = {
            branch.canonical_branch_id: abs(float(result.flow_by_line[branch.canonical_branch_id])) / float(branch.rate_a)
            for branch in identity.branches
            if branch.canonical_branch_id in result.flow_by_line
        }
        r_raw = sum(float(p_env.get(line_id, 0.0)) * float(loading) ** 2 for line_id, loading in flow_loading.items())
        r_norm = float(r_raw / r_base)
        breakdown = compute_load_shedding(load_ids, pd_pre, alpha_requested, source_less)
        j_trade = compute_j_trade(lambda_r, r_norm, breakdown.l_shed_total)
        return CandidateEvaluation(
            evaluation_status=EvaluationStatus.OK.value,
            search_objective=j_trade,
            r_norm=r_norm,
            l_shed_total=breakdown.l_shed_total,
            l_shed_control=breakdown.l_shed_control,
            l_shed_island=breakdown.l_shed_island,
            j_trade=j_trade,
            max_loading=max(flow_loading.values()) if flow_loading else None,
            num_loading_gt_1=sum(1 for value in flow_loading.values() if value > 1.0),
            objective_cost=result.objective_cost,
            message=result.message,
        )

    return evaluator


def _make_gridsfm_evaluator(*, raw_case, identity, model, shutoff, p_env, r_base, lambda_r, weights, output_dir, topology_rank) -> Callable[[Mapping[int, float]], CandidateEvaluation]:
    counter = {"count": 0}

    def evaluator(alpha_requested: Mapping[int, float]) -> CandidateEvaluation:
        counter["count"] += 1
        result = evaluate_gridsfm_candidate(
            raw_case=raw_case,
            identity=identity,
            model=model,
            offline_branch_ids=shutoff,
            alpha_requested=alpha_requested,
            p_env_by_line=p_env,
            r_base=r_base,
            lambda_r=lambda_r,
            weights=weights,
            work_dir=output_dir / "mutated_candidates" / f"topology_{topology_rank:04d}" / f"alpha_{counter['count']:04d}",
        )
        obj = result.objective
        if obj is None:
            return CandidateEvaluation(
                evaluation_status=result.evaluation_status.value,
                search_objective=None,
                d_input=result.d_input,
                feasibility_head=result.feasibility_head,
                message=result.message,
            )
        return CandidateEvaluation(
            evaluation_status=result.evaluation_status.value,
            search_objective=obj.j_total,
            r_norm=obj.r_norm,
            l_shed_total=obj.l_shed_total,
            l_shed_control=None if result.load_shedding is None else result.load_shedding.l_shed_control,
            l_shed_island=None if result.load_shedding is None else result.load_shedding.l_shed_island,
            j_trade=obj.j_trade,
            pac_operational=obj.pac_operational,
            pac_ac=obj.pac_ac,
            pac_model=obj.pac_model,
            pac_total=obj.pac_total,
            j_total=obj.j_total,
            d_input=result.d_input,
            feasibility_head=result.feasibility_head,
            max_loading=max(result.flow_loading_by_line.values()) if result.flow_loading_by_line else None,
            num_loading_gt_1=sum(1 for value in result.flow_loading_by_line.values() if value > 1.0),
            message=result.message,
            extra={
                "pac_model_components": dict(result.pac_model_components),
                "pac_notes": result.pac_notes,
            },
        )

    return evaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["guided-dc", "guided-gridsfm", "th-gridsfm"], required=True)
    parser.add_argument("--gridsfm-root", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--baseline-loading-csv", required=True)
    parser.add_argument("--pac-freeze-json", default=None)
    parser.add_argument("--scenario-id", default="J-S1")
    parser.add_argument("--lambda-r", type=float, default=0.8)
    parser.add_argument("--lambda-r-proxy", type=float, default=None)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--topology-budget", type=int, default=100)
    parser.add_argument("--continuous-eval-budget", type=int, default=20)
    parser.add_argument("--q", type=int, default=5)
    parser.add_argument("--topology-pool-csv", default=None)
    parser.add_argument("--xdg-cache-home", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    start_time = time.time()
    if args.xdg_cache_home:
        os.environ["XDG_CACHE_HOME"] = str(Path(args.xdg_cache_home).expanduser().resolve())
    gridsfm_root = Path(args.gridsfm_root).expanduser().resolve()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_case = _load_json(gridsfm_root / "model" / "samples" / "case500_goc.pyg.json")
    candidate_scores_path = input_dir / "stage_j_candidate_line_scores.csv"
    candidate_ids = [
        int(row["branch_id"])
        for row in csv.DictReader(candidate_scores_path.open(newline="", encoding="utf-8"))
    ]
    identity = build_goc500_identity(raw_case, candidate_branch_ids=candidate_ids)
    baseline_loading = load_baseline_loading_csv(args.baseline_loading_csv)
    c_by_line = _load_candidate_scores(candidate_scores_path)
    p_env = _p_env(input_dir / "stage_j_p_env_by_scenario.csv", args.scenario_id)
    scenario = _scenario_row(input_dir / "stage_j_scenario_register.csv", args.scenario_id)
    r_base = float(scenario["r_base"])
    lambda_r_proxy = float(args.lambda_r if args.lambda_r_proxy is None else args.lambda_r_proxy)
    target_branch_ids = set(_line_ids(scenario["target_branch_ids"]))

    default_pool_name = "j8_th_gridsfm_topology_pool.csv" if args.method == "th-gridsfm" else "j8_topology_pool.csv"
    pool_csv = Path(args.topology_pool_csv).expanduser().resolve() if args.topology_pool_csv else output_dir / default_pool_name
    if pool_csv.exists():
        pool = _read_topology_pool(pool_csv, method_family="TH-GridSFM" if args.method == "th-gridsfm" else "Guided")
    else:
        if args.method == "th-gridsfm":
            pool = build_th_gridsfm_topology_pool(
                candidate_branch_ids=candidate_ids,
                p_env_by_line=p_env,
                baseline_loading=baseline_loading,
                k=args.k,
            )
            _write_th_gridsfm_topology_pool(pool_csv, pool)
        else:
            pool = solve_proxy_topology_pool(
                candidate_branch_ids=candidate_ids,
                p_env_by_line=p_env,
                baseline_loading=baseline_loading,
                c_by_line=c_by_line,
                lambda_r_proxy=lambda_r_proxy,
                k=args.k,
                pool_size=args.topology_budget,
            )
            _write_topology_pool(pool_csv, pool)

    model = None
    weights = None
    if args.method in {"guided-gridsfm", "th-gridsfm"}:
        if args.pac_freeze_json is None:
            raise ValueError("--pac-freeze-json is required for GridSFM-backed methods")
        from gridsfm import load_model

        model_root = gridsfm_root / "model"
        checkpoint = Path(args.checkpoint).expanduser().resolve() if args.checkpoint else model_root / "checkpoints" / "gridsfm_open_v1.1.pt"
        frozen = _load_json(Path(args.pac_freeze_json).expanduser().resolve())["frozen_weights"]
        weights = PacWeights(
            rho_phys=float(frozen["rho_phys"]),
            w_op=float(frozen["w_op"]),
            w_ac=float(frozen["w_ac"]),
            w_model=float(frozen["w_model"]),
        )
        model = load_model(str(checkpoint), device="cpu")

    load_ids = [load.canonical_load_id for load in identity.loads]
    summary_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    alpha_rows: list[dict[str, object]] = []

    for item in pool:
        shutoff = item.shutoff_branch_ids
        source_less = source_less_load_ids(identity, shutoff)
        selected_loads = select_budgeted_alpha_loads(identity, shutoff, q=args.q, exclude_load_ids=source_less)
        if args.method == "guided-dc":
            evaluator = _make_dc_evaluator(
                raw_case=raw_case,
                identity=identity,
                shutoff=shutoff,
                p_env=p_env,
                r_base=r_base,
                lambda_r=args.lambda_r,
            )
            method_label = "Guided-DC"
        else:
            evaluator = _make_gridsfm_evaluator(
                raw_case=raw_case,
                identity=identity,
                model=model,
                shutoff=shutoff,
                p_env=p_env,
                r_base=r_base,
                lambda_r=args.lambda_r,
                weights=weights,
                output_dir=output_dir,
                topology_rank=item.rank,
            )
            method_label = f"TH-GridSFM-top{len(shutoff)}" if args.method == "th-gridsfm" else "Guided-GridSFM"

        topology_summary, topology_trace, best_alpha = run_budgeted_alpha_search(
            load_ids=load_ids,
            selected_load_ids=selected_loads,
            offline_branch_ids=shutoff,
            method=method_label,
            topology_rank=item.rank,
            evaluator=evaluator,
            continuous_eval_budget=args.continuous_eval_budget,
        )
        topology_summary.update(
            {
                "scenario_id": args.scenario_id,
                "lambda_r": args.lambda_r,
                "lambda_r_proxy": lambda_r_proxy,
                "k": args.k,
                "topology_budget": args.topology_budget,
                "continuous_eval_budget": args.continuous_eval_budget,
                "q": args.q,
                "proxy_objective": item.proxy_objective,
                "r_proxy": item.r_proxy,
                "l_proxy": item.l_proxy,
                "num_shutoffs": len(shutoff),
                "contains_any_target": bool(set(shutoff) & target_branch_ids),
                "contains_all_targets": target_branch_ids.issubset(set(shutoff)),
                "target_branch_ids": scenario["target_branch_ids"],
                "source_less_load_ids": _line_key(source_less),
                "num_source_less_loads": len(source_less),
                "alpha_selection_policy": "topology_endpoint_neighborhood_then_largest_pd",
                "topology_policy": "score_TH=p_env*baseline_loading^2" if args.method == "th-gridsfm" else "guided_proxy_no_good_pool",
            }
        )
        summary_rows.append(topology_summary)
        for row in topology_trace:
            row.update(
                {
                    "scenario_id": args.scenario_id,
                    "lambda_r": args.lambda_r,
                    "lambda_r_proxy": lambda_r_proxy,
                    "proxy_objective": item.proxy_objective,
                    "r_proxy": item.r_proxy,
                    "l_proxy": item.l_proxy,
                    "num_shutoffs": len(shutoff),
                    "contains_any_target": bool(set(shutoff) & target_branch_ids),
                    "contains_all_targets": target_branch_ids.issubset(set(shutoff)),
                }
            )
            trace_rows.append(row)
        for load_id in selected_loads:
            alpha_rows.append(
                {
                    "method": method_label,
                    "topology_rank": item.rank,
                    "topology_id": _line_key(shutoff),
                    "load_id": load_id,
                    "best_alpha_requested": best_alpha.get(load_id, 1.0),
                    "selected_for_alpha_search": True,
                }
            )

    best_row = min(
        (row for row in summary_rows if row.get("best_found") is True and row.get("search_objective") != ""),
        key=lambda row: float(row["search_objective"]),
        default=None,
    )
    eligible_topology_count = sum(1 for row in summary_rows if row.get("best_found") is True)
    failed_topology_count = len(summary_rows) - eligible_topology_count
    trace_status_counts = dict(Counter(str(row.get("evaluation_status", "")) for row in trace_rows))
    topology_status_counts = dict(Counter(str(row.get("evaluation_status", "")) for row in summary_rows))
    if eligible_topology_count == len(pool):
        run_status = "PASS"
    elif eligible_topology_count > 0:
        # Fixed (z, alpha) DC recourse may be infeasible for some proposed
        # topologies. Those candidates are correctly rejected, rather than
        # making the method-level evaluation unusable when a valid finalist
        # remains. Preserve the count in the payload and topology table.
        run_status = "PASS_WITH_INFEASIBLE_CANDIDATES"
    else:
        run_status = "FAIL"
    method_stub = args.method.replace("-", "_")
    summary_csv = output_dir / f"j8_{method_stub}_topology_summary.csv"
    trace_csv = output_dir / f"j8_{method_stub}_alpha_trace.csv"
    alpha_csv = output_dir / f"j8_{method_stub}_best_selected_alpha.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(summary_csv, summary_rows)
    _write_rows(trace_csv, trace_rows)
    _write_rows(alpha_csv, alpha_rows)
    payload = {
        "status": run_status,
        "method": args.method,
        "scenario_id": args.scenario_id,
        "target_branch_ids": scenario["target_branch_ids"],
        "lambda_r": args.lambda_r,
        "lambda_r_proxy": lambda_r_proxy,
        "k_constraint": "<= " + str(args.k),
        "topology_budget": args.topology_budget,
        "continuous_eval_budget": args.continuous_eval_budget,
        "q": args.q,
        "topology_pool_csv": str(pool_csv),
        "summary_csv": str(summary_csv),
        "trace_csv": str(trace_csv),
        "alpha_csv": str(alpha_csv),
        "num_topologies": len(pool),
        "eligible_topology_count": eligible_topology_count,
        "failed_topology_count": failed_topology_count,
        "num_candidate_evaluations": len(trace_rows),
        "trace_status_counts": trace_status_counts,
        "topology_status_counts": topology_status_counts,
        "runtime_seconds": time.time() - start_time,
        "xdg_cache_home": str(Path(args.xdg_cache_home).expanduser().resolve()) if args.xdg_cache_home else "",
        "alpha_selection_policy": "topology_endpoint_neighborhood_then_largest_pd",
        "full_coordinate_screen": False,
        "best_topology": best_row,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"j8_{method_stub}_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
