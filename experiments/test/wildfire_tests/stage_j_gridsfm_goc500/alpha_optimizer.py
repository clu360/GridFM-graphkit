"""Full per-load alpha search for Stage J fixed-topology evaluations.

The optimizer is intentionally backend-neutral. It never embeds GridSFM,
DC-OPF, or AC-OPF in a differentiable solver; it proposes bounded full-vector
alpha candidates and delegates electrical scoring to a caller-supplied
fixed-(z, alpha) evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping


def topology_hash(offline_branch_ids: Iterable[int]) -> str:
    """Stable hash for topology-only no-good-cut identity."""

    payload = ",".join(str(int(line_id)) for line_id in sorted(int(v) for v in offline_branch_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def alpha_hash(alpha_by_load: Mapping[int, float]) -> str:
    """Stable hash for a full requested alpha vector."""

    parts = [f"{int(load_id)}:{float(alpha):.12f}" for load_id, alpha in sorted(alpha_by_load.items())]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class AlphaSearchConfig:
    """J7.5 derivative-free full-alpha search settings."""

    seed_values: tuple[float, ...] = (1.0, 0.98, 0.95)
    delta_schedule: tuple[float, ...] = (0.10, 0.05, 0.02, 0.01)
    b_alpha: int = 300
    epsilon_abs: float = 1e-9
    epsilon_rel: float = 1e-9
    time_limit_seconds: float | None = None
    checkpoint_counts: tuple[int, ...] = (1, 25, 50, 100, 250, 500)
    max_accepted_moves_per_delta: int = 25
    run_interaction_search: bool = False

    def validate(self) -> None:
        if not self.seed_values:
            raise ValueError("seed_values must be nonempty")
        if not self.delta_schedule:
            raise ValueError("delta_schedule must be nonempty")
        if self.b_alpha <= 0:
            raise ValueError("b_alpha must be positive")
        for name, values in (("seed_values", self.seed_values), ("delta_schedule", self.delta_schedule)):
            for value in values:
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    raise ValueError(f"{name} values must be finite and nonnegative, got {value}")
        for seed in self.seed_values:
            if seed > 1.0:
                raise ValueError(f"seed alpha must be in [0, 1], got {seed}")


@dataclass(frozen=True)
class AlphaEvaluation:
    """Normalized fixed-(z, alpha) evaluation returned by backend wrappers."""

    evaluation_status: str
    search_objective: float | None
    l_shed_total: float | None = None
    l_shed_control: float | None = None
    l_shed_island: float | None = None
    r_norm: float | None = None
    j_trade: float | None = None
    pac_operational: float | None = None
    pac_ac: float | None = None
    pac_model: float | None = None
    pac_total: float | None = None
    j_total: float | None = None
    d_input: float | None = None
    runtime_seconds: float | None = None
    message: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def eligible(self) -> bool:
        return self.search_objective is not None and math.isfinite(float(self.search_objective))


@dataclass(frozen=True)
class AlphaSearchResult:
    """Final search result and auditable trajectory."""

    search_run_id: str
    backend: str
    topology_id: str
    topology_hash: str
    cache_namespace_hash: str
    best_alpha: dict[int, float]
    best_alpha_hash: str
    best_evaluation: AlphaEvaluation | None
    trace_rows: tuple[dict[str, object], ...]
    checkpoint_rows: tuple[dict[str, object], ...]
    termination_reason: str
    actual_evaluation_count: int
    unique_alpha_count: int
    cache_hit_count: int
    completed_sweeps: int
    accepted_moves: int
    interaction_search_performed: bool


@dataclass(frozen=True)
class ScreenedScipyAlphaConfig:
    """J7.5 v2 settings: coordinate screening followed by local SciPy search."""

    seed_values: tuple[float, ...] = (1.0, 0.98, 0.95)
    screen_delta: float = 0.10
    q_values: tuple[int, ...] = (5, 10, 20)
    b_alpha: int = 500
    epsilon_abs: float = 1e-9
    epsilon_rel: float = 1e-9
    time_limit_seconds: float | None = None
    checkpoint_counts: tuple[int, ...] = (1, 25, 50, 100, 250, 500)
    scipy_method: str = "Powell"
    scipy_maxfev_per_q: int = 60
    scipy_xtol: float = 1e-3
    scipy_ftol: float = 1e-6
    alpha_round_decimals: int = 6

    def validate(self) -> None:
        if not self.seed_values:
            raise ValueError("seed_values must be nonempty")
        if self.b_alpha <= 0:
            raise ValueError("b_alpha must be positive")
        if not math.isfinite(float(self.screen_delta)) or not (0.0 < float(self.screen_delta) <= 1.0):
            raise ValueError("screen_delta must be finite and in (0, 1]")
        if not self.q_values:
            raise ValueError("q_values must be nonempty")
        if self.scipy_maxfev_per_q <= 0:
            raise ValueError("scipy_maxfev_per_q must be positive")
        if self.alpha_round_decimals < 0:
            raise ValueError("alpha_round_decimals must be nonnegative")
        for seed in self.seed_values:
            if not math.isfinite(float(seed)) or float(seed) < 0.0 or float(seed) > 1.0:
                raise ValueError(f"seed alpha must be in [0, 1], got {seed}")
        for q_value in self.q_values:
            if int(q_value) <= 0:
                raise ValueError(f"q_values must be positive, got {q_value}")


@dataclass(frozen=True)
class ScreenedScipyAlphaResult:
    """Final J7.5 v2 search result and q-sensitivity metadata."""

    search_run_id: str
    backend: str
    topology_id: str
    topology_hash: str
    cache_namespace_hash: str
    best_alpha: dict[int, float]
    best_alpha_hash: str
    best_evaluation: AlphaEvaluation | None
    trace_rows: tuple[dict[str, object], ...]
    checkpoint_rows: tuple[dict[str, object], ...]
    screen_rows: tuple[dict[str, object], ...]
    q_summary_rows: tuple[dict[str, object], ...]
    termination_reason: str
    actual_evaluation_count: int
    unique_alpha_count: int
    cache_hit_count: int
    selected_loads_by_q: Mapping[int, tuple[int, ...]]
    screening_best_improved: bool


Evaluator = Callable[[Mapping[int, float]], AlphaEvaluation]


def optimize_full_alpha(
    *,
    load_ids: Iterable[int],
    fixed_zero_load_ids: Iterable[int],
    offline_branch_ids: Iterable[int],
    backend: str,
    topology_id: str,
    search_run_id: str,
    evaluator: Evaluator,
    config: AlphaSearchConfig | None = None,
    cache_context: Mapping[str, object] | None = None,
) -> AlphaSearchResult:
    """Run the locked J7.5 full per-load alpha pattern search."""

    config = config or AlphaSearchConfig()
    config.validate()

    ids = tuple(int(load_id) for load_id in load_ids)
    if not ids:
        raise ValueError("load_ids must be nonempty")
    if len(set(ids)) != len(ids):
        raise ValueError("load_ids must be unique")
    fixed_zero = {int(load_id) for load_id in fixed_zero_load_ids}
    unknown_fixed = sorted(fixed_zero.difference(ids))
    if unknown_fixed:
        raise KeyError(f"fixed_zero_load_ids not in load_ids: {unknown_fixed}")

    topo_hash = topology_hash(offline_branch_ids)
    cache_namespace_hash = _cache_namespace_hash(
        {
            "backend": backend,
            "topology_id": topology_id,
            "topology_hash": topo_hash,
            **dict(cache_context or {}),
        }
    )
    trace_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], AlphaEvaluation] = {}
    cache_hit_count = 0
    actual_count = 0
    iteration = 0
    sweep_id = 0
    completed_sweeps = 0
    accepted_moves = 0
    start = time.monotonic()
    best_alpha: dict[int, float] | None = None
    best_eval: AlphaEvaluation | None = None
    best_row_index: int | None = None

    def timed_out() -> bool:
        return config.time_limit_seconds is not None and (time.monotonic() - start) >= config.time_limit_seconds

    def meaningful_improvement(candidate: float, incumbent: float) -> bool:
        improvement = incumbent - candidate
        rel = improvement / max(abs(incumbent), 1e-12)
        return improvement > config.epsilon_abs and rel > config.epsilon_rel

    def serialize_values(alpha: Mapping[int, float], changed: Iterable[int]) -> str:
        return ";".join(f"{int(load_id)}:{float(alpha[int(load_id)]):.6f}" for load_id in changed)

    def evaluate_candidate(
        alpha: Mapping[int, float],
        *,
        parent_hash: str | None,
        changed_load_ids: Iterable[int],
        previous_alpha: Mapping[int, float] | None,
        step_size: float | None,
        candidate_type: str,
        current_best: AlphaEvaluation | None,
    ) -> tuple[int, AlphaEvaluation, bool]:
        nonlocal actual_count, cache_hit_count, iteration

        iteration += 1
        changed = tuple(int(load_id) for load_id in changed_load_ids)
        ahash = alpha_hash(alpha)
        cache_key = (cache_namespace_hash, ahash)
        cache_hit = cache_key in cache
        if cache_hit:
            cache_hit_count += 1
            evaluation = cache[cache_key]
        else:
            if actual_count >= config.b_alpha:
                raise RuntimeError("B_ALPHA_EXHAUSTED")
            before = time.monotonic()
            evaluation = evaluator(dict(alpha))
            runtime = time.monotonic() - before
            if evaluation.runtime_seconds is None:
                evaluation = AlphaEvaluation(
                    evaluation_status=evaluation.evaluation_status,
                    search_objective=evaluation.search_objective,
                    l_shed_total=evaluation.l_shed_total,
                    l_shed_control=evaluation.l_shed_control,
                    l_shed_island=evaluation.l_shed_island,
                    r_norm=evaluation.r_norm,
                    j_trade=evaluation.j_trade,
                    pac_operational=evaluation.pac_operational,
                    pac_ac=evaluation.pac_ac,
                    pac_model=evaluation.pac_model,
                    pac_total=evaluation.pac_total,
                    j_total=evaluation.j_total,
                    d_input=evaluation.d_input,
                    runtime_seconds=runtime,
                    message=evaluation.message,
                    extra=evaluation.extra,
                )
            cache[cache_key] = evaluation
            actual_count += 1

        incumbent = float(current_best.search_objective) if current_best and current_best.eligible else None
        objective = float(evaluation.search_objective) if evaluation.eligible else None
        improvement_abs = None if incumbent is None or objective is None else incumbent - objective
        improvement_rel = None if incumbent is None or objective is None else improvement_abs / max(abs(incumbent), 1e-12)
        row = {
            "search_run_id": search_run_id,
            "search_iteration": iteration,
            "sweep_id": sweep_id,
            "topology_id": topology_id,
            "topology_hash": topo_hash,
            "cache_namespace_hash": cache_namespace_hash,
            "backend": backend,
            "alpha_hash": ahash,
            "parent_alpha_hash": parent_hash or "",
            "changed_load_ids": ";".join(str(v) for v in changed),
            "previous_alpha_values": "" if previous_alpha is None else serialize_values(previous_alpha, changed),
            "candidate_alpha_values": serialize_values(alpha, changed) if changed else _uniform_alpha_label(alpha),
            "step_size": "" if step_size is None else float(step_size),
            "candidate_type": candidate_type,
            "L_shed": _maybe(evaluation.l_shed_total),
            "L_shed_control": _maybe(evaluation.l_shed_control),
            "L_shed_island": _maybe(evaluation.l_shed_island),
            "R_norm": _maybe(evaluation.r_norm),
            "J_trade": _maybe(evaluation.j_trade),
            "PAC_operational": _maybe(evaluation.pac_operational),
            "PAC_AC": _maybe(evaluation.pac_ac),
            "PAC_model": _maybe(evaluation.pac_model),
            "PAC_total": _maybe(evaluation.pac_total),
            "J_total": _maybe(evaluation.j_total),
            "search_objective": _maybe(evaluation.search_objective),
            "improvement_abs": _maybe(improvement_abs),
            "improvement_rel": _maybe(improvement_rel),
            "evaluation_status": str(evaluation.evaluation_status),
            "runtime_seconds": _maybe(evaluation.runtime_seconds),
            "accepted": False,
            "cache_hit": cache_hit,
            "message": evaluation.message,
        }
        for key, value in sorted(evaluation.extra.items()):
            row[f"extra_{key}"] = _traceable_extra(value)
        trace_rows.append(row)
        if actual_count in set(config.checkpoint_counts):
            checkpoint_rows.append(
                {
                    "search_run_id": search_run_id,
                    "backend": backend,
                    "topology_id": topology_id,
                    "actual_evaluation_count": actual_count,
                    "best_alpha_hash": alpha_hash(best_alpha) if best_alpha is not None else "",
                    "best_search_objective": _maybe(best_eval.search_objective if best_eval else None),
                    "best_l_shed_total": _maybe(best_eval.l_shed_total if best_eval else None),
                    "best_r_norm": _maybe(best_eval.r_norm if best_eval else None),
                    "best_j_trade": _maybe(best_eval.j_trade if best_eval else None),
                    "best_j_total": _maybe(best_eval.j_total if best_eval else None),
                }
            )
        return len(trace_rows) - 1, evaluation, cache_hit

    termination_reason = "completed"
    try:
        for seed in config.seed_values:
            if timed_out():
                termination_reason = "time_limit_before_seed_completion"
                break
            alpha = {load_id: float(seed) for load_id in ids}
            row_idx, evaluation, _ = evaluate_candidate(
                alpha,
                parent_hash=None,
                changed_load_ids=(),
                previous_alpha=None,
                step_size=None,
                candidate_type=f"seed_all_{seed:g}",
                current_best=best_eval,
            )
            if evaluation.eligible and (best_eval is None or float(evaluation.search_objective) < float(best_eval.search_objective)):
                best_alpha = dict(alpha)
                best_eval = evaluation
                best_row_index = row_idx
        if best_row_index is not None:
            trace_rows[best_row_index]["accepted"] = True

        if best_alpha is None or best_eval is None:
            termination_reason = "no_eligible_seed"
        else:
            controllable = tuple(load_id for load_id in ids if load_id not in fixed_zero)
            delta_index = 0
            accepted_at_current_delta = 0
            while delta_index < len(config.delta_schedule) and termination_reason == "completed":
                if timed_out():
                    termination_reason = "time_limit"
                    break
                delta = float(config.delta_schedule[delta_index])
                sweep_id += 1
                sweep_best_row: int | None = None
                sweep_best_eval: AlphaEvaluation | None = None
                sweep_best_alpha: dict[int, float] | None = None
                sweep_completed = True

                for load_id in controllable:
                    if timed_out():
                        termination_reason = "time_limit"
                        sweep_completed = False
                        break
                    base_value = float(best_alpha[load_id])
                    directions: list[tuple[str, float]] = []
                    down = max(0.0, base_value - delta)
                    if down < base_value - 1e-12:
                        directions.append(("coordinate_down", down))
                    up = min(1.0, base_value + delta)
                    if base_value < 1.0 - 1e-12 and up > base_value + 1e-12:
                        directions.append(("coordinate_up", up))

                    for candidate_type, new_value in directions:
                        candidate = dict(best_alpha)
                        previous = dict(best_alpha)
                        candidate[load_id] = float(new_value)
                        try:
                            row_idx, evaluation, _ = evaluate_candidate(
                                candidate,
                                parent_hash=alpha_hash(best_alpha),
                                changed_load_ids=(load_id,),
                                previous_alpha=previous,
                                step_size=delta,
                                candidate_type=candidate_type,
                                current_best=best_eval,
                            )
                        except RuntimeError as exc:
                            if str(exc) == "B_ALPHA_EXHAUSTED":
                                termination_reason = "budget_exhausted"
                                sweep_completed = False
                                break
                            raise
                        if evaluation.eligible and (
                            sweep_best_eval is None
                            or float(evaluation.search_objective) < float(sweep_best_eval.search_objective)
                        ):
                            sweep_best_row = row_idx
                            sweep_best_eval = evaluation
                            sweep_best_alpha = candidate
                    if termination_reason != "completed":
                        break

                if not sweep_completed:
                    break
                completed_sweeps += 1
                if (
                    sweep_best_eval is not None
                    and sweep_best_alpha is not None
                    and meaningful_improvement(float(sweep_best_eval.search_objective), float(best_eval.search_objective))
                ):
                    if best_row_index is not None:
                        trace_rows[best_row_index]["accepted"] = False
                    best_alpha = sweep_best_alpha
                    best_eval = sweep_best_eval
                    best_row_index = sweep_best_row
                    if best_row_index is not None:
                        trace_rows[best_row_index]["accepted"] = True
                    accepted_moves += 1
                    accepted_at_current_delta += 1
                    if accepted_at_current_delta >= config.max_accepted_moves_per_delta:
                        delta_index += 1
                        accepted_at_current_delta = 0
                    continue

                delta_index += 1
                accepted_at_current_delta = 0
    except RuntimeError as exc:
        if str(exc) == "B_ALPHA_EXHAUSTED":
            termination_reason = "budget_exhausted"
        else:
            raise

    if best_alpha is None:
        best_alpha = {load_id: 1.0 for load_id in ids}
    return AlphaSearchResult(
        search_run_id=search_run_id,
        backend=backend,
        topology_id=topology_id,
        topology_hash=topo_hash,
        cache_namespace_hash=cache_namespace_hash,
        best_alpha=best_alpha,
        best_alpha_hash=alpha_hash(best_alpha),
        best_evaluation=best_eval,
        trace_rows=tuple(trace_rows),
        checkpoint_rows=tuple(checkpoint_rows),
        termination_reason=termination_reason,
        actual_evaluation_count=actual_count,
        unique_alpha_count=len(cache),
        cache_hit_count=cache_hit_count,
        completed_sweeps=completed_sweeps,
        accepted_moves=accepted_moves,
        interaction_search_performed=bool(config.run_interaction_search and False),
    )


def optimize_screened_scipy_alpha(
    *,
    load_ids: Iterable[int],
    fixed_zero_load_ids: Iterable[int],
    offline_branch_ids: Iterable[int],
    backend: str,
    topology_id: str,
    search_run_id: str,
    evaluator: Evaluator,
    config: ScreenedScipyAlphaConfig | None = None,
    cache_context: Mapping[str, object] | None = None,
) -> ScreenedScipyAlphaResult:
    """Run J7.5 v2: all-load coordinate screen, then SciPy on top-q loads.

    The method still honors the locked full per-load alpha formulation. The
    screen only selects which coordinates receive expensive local refinement;
    all unselected load alphas remain explicit and fixed at the current
    incumbent value.
    """

    config = config or ScreenedScipyAlphaConfig()
    config.validate()
    try:
        from scipy.optimize import minimize
    except ImportError as exc:  # pragma: no cover - environment-specific guard.
        raise RuntimeError("SciPy is required for optimize_screened_scipy_alpha") from exc

    ids = tuple(int(load_id) for load_id in load_ids)
    if not ids:
        raise ValueError("load_ids must be nonempty")
    if len(set(ids)) != len(ids):
        raise ValueError("load_ids must be unique")
    fixed_zero = {int(load_id) for load_id in fixed_zero_load_ids}
    unknown_fixed = sorted(fixed_zero.difference(ids))
    if unknown_fixed:
        raise KeyError(f"fixed_zero_load_ids not in load_ids: {unknown_fixed}")

    topo_hash = topology_hash(offline_branch_ids)
    cache_namespace_hash = _cache_namespace_hash(
        {
            "backend": backend,
            "topology_id": topology_id,
            "topology_hash": topo_hash,
            "optimizer": "screened_scipy_v2",
            **dict(cache_context or {}),
        }
    )
    trace_rows: list[dict[str, object]] = []
    checkpoint_rows: list[dict[str, object]] = []
    screen_rows: list[dict[str, object]] = []
    q_summary_rows: list[dict[str, object]] = []
    cache: dict[tuple[str, str], AlphaEvaluation] = {}
    cache_hit_count = 0
    actual_count = 0
    iteration = 0
    sweep_id = 0
    start = time.monotonic()
    best_alpha: dict[int, float] | None = None
    best_eval: AlphaEvaluation | None = None
    best_row_index: int | None = None
    screening_best_improved = False

    def timed_out() -> bool:
        return config.time_limit_seconds is not None and (time.monotonic() - start) >= config.time_limit_seconds

    def meaningful_improvement(candidate: float, incumbent: float) -> bool:
        improvement = incumbent - candidate
        rel = improvement / max(abs(incumbent), 1e-12)
        return improvement > config.epsilon_abs and rel > config.epsilon_rel

    def normalize_value(value: float) -> float:
        clipped = min(1.0, max(0.0, float(value)))
        return round(clipped, int(config.alpha_round_decimals))

    def serialize_values(alpha: Mapping[int, float], changed: Iterable[int]) -> str:
        return ";".join(f"{int(load_id)}:{float(alpha[int(load_id)]):.6f}" for load_id in changed)

    def evaluate_candidate(
        alpha: Mapping[int, float],
        *,
        parent_hash: str | None,
        changed_load_ids: Iterable[int],
        previous_alpha: Mapping[int, float] | None,
        step_size: float | None,
        candidate_type: str,
        current_best: AlphaEvaluation | None,
        q_value: int | None = None,
    ) -> tuple[int, AlphaEvaluation, bool]:
        nonlocal actual_count, cache_hit_count, iteration

        iteration += 1
        changed = tuple(int(load_id) for load_id in changed_load_ids)
        normalized = {int(load_id): normalize_value(float(alpha[int(load_id)])) for load_id in ids}
        ahash = alpha_hash(normalized)
        cache_key = (cache_namespace_hash, ahash)
        cache_hit = cache_key in cache
        if cache_hit:
            cache_hit_count += 1
            evaluation = cache[cache_key]
        else:
            if actual_count >= config.b_alpha:
                raise RuntimeError("B_ALPHA_EXHAUSTED")
            before = time.monotonic()
            evaluation = evaluator(dict(normalized))
            runtime = time.monotonic() - before
            if evaluation.runtime_seconds is None:
                evaluation = AlphaEvaluation(
                    evaluation_status=evaluation.evaluation_status,
                    search_objective=evaluation.search_objective,
                    l_shed_total=evaluation.l_shed_total,
                    l_shed_control=evaluation.l_shed_control,
                    l_shed_island=evaluation.l_shed_island,
                    r_norm=evaluation.r_norm,
                    j_trade=evaluation.j_trade,
                    pac_operational=evaluation.pac_operational,
                    pac_ac=evaluation.pac_ac,
                    pac_model=evaluation.pac_model,
                    pac_total=evaluation.pac_total,
                    j_total=evaluation.j_total,
                    d_input=evaluation.d_input,
                    runtime_seconds=runtime,
                    message=evaluation.message,
                    extra=evaluation.extra,
                )
            cache[cache_key] = evaluation
            actual_count += 1

        incumbent = float(current_best.search_objective) if current_best and current_best.eligible else None
        objective = float(evaluation.search_objective) if evaluation.eligible else None
        improvement_abs = None if incumbent is None or objective is None else incumbent - objective
        improvement_rel = None if incumbent is None or objective is None else improvement_abs / max(abs(incumbent), 1e-12)
        row = {
            "search_run_id": search_run_id,
            "search_iteration": iteration,
            "sweep_id": sweep_id,
            "topology_id": topology_id,
            "topology_hash": topo_hash,
            "cache_namespace_hash": cache_namespace_hash,
            "backend": backend,
            "alpha_hash": ahash,
            "parent_alpha_hash": parent_hash or "",
            "changed_load_ids": ";".join(str(v) for v in changed),
            "previous_alpha_values": "" if previous_alpha is None else serialize_values(previous_alpha, changed),
            "candidate_alpha_values": serialize_values(normalized, changed) if changed else _uniform_alpha_label(normalized),
            "step_size": "" if step_size is None else float(step_size),
            "candidate_type": candidate_type,
            "screen_or_q": "" if q_value is None else int(q_value),
            "L_shed": _maybe(evaluation.l_shed_total),
            "L_shed_control": _maybe(evaluation.l_shed_control),
            "L_shed_island": _maybe(evaluation.l_shed_island),
            "R_norm": _maybe(evaluation.r_norm),
            "J_trade": _maybe(evaluation.j_trade),
            "PAC_operational": _maybe(evaluation.pac_operational),
            "PAC_AC": _maybe(evaluation.pac_ac),
            "PAC_model": _maybe(evaluation.pac_model),
            "PAC_total": _maybe(evaluation.pac_total),
            "J_total": _maybe(evaluation.j_total),
            "search_objective": _maybe(evaluation.search_objective),
            "improvement_abs": _maybe(improvement_abs),
            "improvement_rel": _maybe(improvement_rel),
            "evaluation_status": str(evaluation.evaluation_status),
            "runtime_seconds": _maybe(evaluation.runtime_seconds),
            "accepted": False,
            "cache_hit": cache_hit,
            "message": evaluation.message,
        }
        for key, value in sorted(evaluation.extra.items()):
            row[f"extra_{key}"] = _traceable_extra(value)
        trace_rows.append(row)
        if actual_count in set(config.checkpoint_counts):
            checkpoint_rows.append(
                {
                    "search_run_id": search_run_id,
                    "backend": backend,
                    "topology_id": topology_id,
                    "actual_evaluation_count": actual_count,
                    "best_alpha_hash": alpha_hash(best_alpha) if best_alpha is not None else "",
                    "best_search_objective": _maybe(best_eval.search_objective if best_eval else None),
                    "best_l_shed_total": _maybe(best_eval.l_shed_total if best_eval else None),
                    "best_r_norm": _maybe(best_eval.r_norm if best_eval else None),
                    "best_j_trade": _maybe(best_eval.j_trade if best_eval else None),
                    "best_j_total": _maybe(best_eval.j_total if best_eval else None),
                }
            )
        return len(trace_rows) - 1, evaluation, cache_hit

    termination_reason = "completed"
    try:
        for seed in config.seed_values:
            if timed_out():
                termination_reason = "time_limit_before_seed_completion"
                break
            alpha = {load_id: float(seed) for load_id in ids}
            row_idx, evaluation, _ = evaluate_candidate(
                alpha,
                parent_hash=None,
                changed_load_ids=(),
                previous_alpha=None,
                step_size=None,
                candidate_type=f"seed_all_{seed:g}",
                current_best=best_eval,
            )
            if evaluation.eligible and (best_eval is None or float(evaluation.search_objective) < float(best_eval.search_objective)):
                best_alpha = {int(load_id): normalize_value(value) for load_id, value in alpha.items()}
                best_eval = evaluation
                best_row_index = row_idx
        if best_row_index is not None:
            trace_rows[best_row_index]["accepted"] = True

        if best_alpha is None or best_eval is None:
            termination_reason = "no_eligible_seed"
        elif termination_reason == "completed":
            sweep_id = 1
            controllable = tuple(load_id for load_id in ids if load_id not in fixed_zero)
            incumbent_before_screen = dict(best_alpha)
            incumbent_eval_before_screen = best_eval
            parent_hash = alpha_hash(best_alpha)
            best_screen_row: int | None = None
            best_screen_eval: AlphaEvaluation | None = None
            best_screen_alpha: dict[int, float] | None = None

            for load_id in controllable:
                if timed_out():
                    termination_reason = "time_limit"
                    break
                candidate = dict(incumbent_before_screen)
                previous = dict(incumbent_before_screen)
                candidate[load_id] = normalize_value(float(candidate[load_id]) - float(config.screen_delta))
                if candidate[load_id] >= previous[load_id] - 1e-12:
                    continue
                try:
                    row_idx, evaluation, _ = evaluate_candidate(
                        candidate,
                        parent_hash=parent_hash,
                        changed_load_ids=(load_id,),
                        previous_alpha=previous,
                        step_size=config.screen_delta,
                        candidate_type="coordinate_screen_down",
                        current_best=incumbent_eval_before_screen,
                    )
                except RuntimeError as exc:
                    if str(exc) == "B_ALPHA_EXHAUSTED":
                        termination_reason = "budget_exhausted"
                        break
                    raise
                objective = float(evaluation.search_objective) if evaluation.eligible else math.inf
                incumbent_objective = float(incumbent_eval_before_screen.search_objective)
                improvement = incumbent_objective - objective if math.isfinite(objective) else -math.inf
                screen_rows.append(
                    {
                        "search_run_id": search_run_id,
                        "load_id": load_id,
                        "candidate_alpha": candidate[load_id],
                        "screen_delta": config.screen_delta,
                        "screen_objective": _maybe(evaluation.search_objective),
                        "screen_improvement_abs": _maybe(improvement),
                        "evaluation_status": evaluation.evaluation_status,
                        "alpha_hash": alpha_hash(candidate),
                    }
                )
                if evaluation.eligible and (
                    best_screen_eval is None
                    or float(evaluation.search_objective) < float(best_screen_eval.search_objective)
                ):
                    best_screen_row = row_idx
                    best_screen_eval = evaluation
                    best_screen_alpha = candidate

            if (
                termination_reason == "completed"
                and best_screen_eval is not None
                and best_screen_alpha is not None
                and meaningful_improvement(
                    float(best_screen_eval.search_objective),
                    float(incumbent_eval_before_screen.search_objective),
                )
            ):
                if best_row_index is not None:
                    trace_rows[best_row_index]["accepted"] = False
                best_alpha = best_screen_alpha
                best_eval = best_screen_eval
                best_row_index = best_screen_row
                if best_row_index is not None:
                    trace_rows[best_row_index]["accepted"] = True
                screening_best_improved = True

            ranked_screen_rows = sorted(
                screen_rows,
                key=lambda row: (
                    -_finite_or_neg_inf(row.get("screen_improvement_abs")),
                    int(row["load_id"]),
                ),
            )
            selected_loads_by_q: dict[int, tuple[int, ...]] = {}
            scipy_base_alpha = dict(best_alpha)
            scipy_base_eval = best_eval
            scipy_base_hash = alpha_hash(scipy_base_alpha)

            for q_value_raw in sorted(set(int(value) for value in config.q_values)):
                if termination_reason != "completed":
                    break
                if timed_out():
                    termination_reason = "time_limit"
                    break
                q_value = min(q_value_raw, len(ranked_screen_rows))
                selected = tuple(int(row["load_id"]) for row in ranked_screen_rows[:q_value])
                selected_loads_by_q[q_value_raw] = selected
                if not selected:
                    q_summary_rows.append(
                        {
                            "search_run_id": search_run_id,
                            "q": q_value_raw,
                            "selected_load_ids": "",
                            "scipy_success": False,
                            "scipy_message": "no controllable loads selected",
                            "scipy_nfev": 0,
                            "best_search_objective": _maybe(scipy_base_eval.search_objective if scipy_base_eval else None),
                            "best_alpha_hash": scipy_base_hash,
                        }
                    )
                    continue

                x0 = [float(scipy_base_alpha[load_id]) for load_id in selected]
                local_best_row: int | None = None
                local_best_eval: AlphaEvaluation | None = scipy_base_eval
                local_best_alpha: dict[int, float] = dict(scipy_base_alpha)

                def local_objective(x):
                    nonlocal local_best_row, local_best_eval, local_best_alpha, best_alpha, best_eval, best_row_index

                    if timed_out():
                        return 1e12
                    candidate = dict(scipy_base_alpha)
                    for idx, load_id in enumerate(selected):
                        candidate[load_id] = normalize_value(float(x[idx]))
                    changed = tuple(
                        load_id
                        for load_id in selected
                        if abs(float(candidate[load_id]) - float(scipy_base_alpha[load_id])) > 10 ** (-config.alpha_round_decimals)
                    )
                    try:
                        row_idx, evaluation, _ = evaluate_candidate(
                            candidate,
                            parent_hash=scipy_base_hash,
                            changed_load_ids=changed or selected,
                            previous_alpha=scipy_base_alpha,
                            step_size=None,
                            candidate_type="scipy_local",
                            current_best=best_eval,
                            q_value=q_value_raw,
                        )
                    except RuntimeError as exc:
                        if str(exc) == "B_ALPHA_EXHAUSTED":
                            return 1e12
                        raise
                    if not evaluation.eligible:
                        return 1e12
                    objective = float(evaluation.search_objective)
                    if local_best_eval is None or objective < float(local_best_eval.search_objective):
                        local_best_eval = evaluation
                        local_best_alpha = {int(k): normalize_value(v) for k, v in candidate.items()}
                        local_best_row = row_idx
                    if best_eval is None or meaningful_improvement(objective, float(best_eval.search_objective)):
                        if best_row_index is not None:
                            trace_rows[best_row_index]["accepted"] = False
                        best_alpha = {int(k): normalize_value(v) for k, v in candidate.items()}
                        best_eval = evaluation
                        best_row_index = row_idx
                        trace_rows[row_idx]["accepted"] = True
                    return objective

                try:
                    scipy_result = minimize(
                        local_objective,
                        x0,
                        method=config.scipy_method,
                        bounds=[(0.0, 1.0)] * len(selected),
                        options={
                            "maxfev": int(config.scipy_maxfev_per_q),
                            "xtol": float(config.scipy_xtol),
                            "ftol": float(config.scipy_ftol),
                            "disp": False,
                        },
                    )
                    scipy_success = bool(scipy_result.success)
                    scipy_message = str(scipy_result.message)
                    scipy_nfev = int(getattr(scipy_result, "nfev", 0) or 0)
                except RuntimeError as exc:
                    if str(exc) == "B_ALPHA_EXHAUSTED":
                        termination_reason = "budget_exhausted"
                        scipy_success = False
                        scipy_message = "B_ALPHA_EXHAUSTED"
                        scipy_nfev = 0
                    else:
                        raise
                q_summary_rows.append(
                    {
                        "search_run_id": search_run_id,
                        "q": q_value_raw,
                        "selected_load_ids": ";".join(str(v) for v in selected),
                        "scipy_success": scipy_success,
                        "scipy_message": scipy_message,
                        "scipy_nfev": scipy_nfev,
                        "best_search_objective": _maybe(local_best_eval.search_objective if local_best_eval else None),
                        "best_l_shed_total": _maybe(local_best_eval.l_shed_total if local_best_eval else None),
                        "best_r_norm": _maybe(local_best_eval.r_norm if local_best_eval else None),
                        "best_j_trade": _maybe(local_best_eval.j_trade if local_best_eval else None),
                        "best_j_total": _maybe(local_best_eval.j_total if local_best_eval else None),
                        "best_alpha_hash": alpha_hash(local_best_alpha),
                        "local_best_trace_row": "" if local_best_row is None else local_best_row,
                    }
                )
                if actual_count >= config.b_alpha:
                    termination_reason = "budget_exhausted"
                    break

            if "selected_loads_by_q" not in locals():
                selected_loads_by_q = {}
    except RuntimeError as exc:
        if str(exc) == "B_ALPHA_EXHAUSTED":
            termination_reason = "budget_exhausted"
            selected_loads_by_q = locals().get("selected_loads_by_q", {})
        else:
            raise

    if best_alpha is None:
        best_alpha = {load_id: 1.0 for load_id in ids}
    selected_loads_by_q = locals().get("selected_loads_by_q", {})
    return ScreenedScipyAlphaResult(
        search_run_id=search_run_id,
        backend=backend,
        topology_id=topology_id,
        topology_hash=topo_hash,
        cache_namespace_hash=cache_namespace_hash,
        best_alpha=best_alpha,
        best_alpha_hash=alpha_hash(best_alpha),
        best_evaluation=best_eval,
        trace_rows=tuple(trace_rows),
        checkpoint_rows=tuple(checkpoint_rows),
        screen_rows=tuple(screen_rows),
        q_summary_rows=tuple(q_summary_rows),
        termination_reason=termination_reason,
        actual_evaluation_count=actual_count,
        unique_alpha_count=len(cache),
        cache_hit_count=cache_hit_count,
        selected_loads_by_q={int(k): tuple(v) for k, v in selected_loads_by_q.items()},
        screening_best_improved=screening_best_improved,
    )


def _uniform_alpha_label(alpha: Mapping[int, float]) -> str:
    values = {round(float(v), 12) for v in alpha.values()}
    if len(values) == 1:
        return f"all:{next(iter(values)):.6f}"
    return json.dumps({str(k): round(float(v), 6) for k, v in sorted(alpha.items())[:10]}, sort_keys=True)


def _cache_namespace_hash(context: Mapping[str, object]) -> str:
    payload = json.dumps(context, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _maybe(value):
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _traceable_extra(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return "" if not math.isfinite(value) else value
    return json.dumps(value, sort_keys=True)


def _finite_or_neg_inf(value) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return numeric if math.isfinite(numeric) else -math.inf
