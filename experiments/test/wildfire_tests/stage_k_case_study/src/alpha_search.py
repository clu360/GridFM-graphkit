"""Bounded external GridSFM alpha search with an exact evaluation budget."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class AlphaSearchResult:
    best_alpha: dict[int, float]
    best_result: Any | None
    trace: tuple[dict[str, Any], ...]


def budgeted_powell_search(
    *,
    load_ids: Iterable[int],
    selected_load_ids: Iterable[int],
    budget: int,
    evaluator: Callable[[Mapping[int, float]], Any],
    objective_getter: Callable[[Any], float | None],
) -> AlphaSearchResult:
    """Run Powell while enforcing an exact upper bound on unique evaluations.

    A five-evaluation smoke is small but technically valid: one full-service
    point is evaluated before Powell spends the remaining unique calls. The
    cache prevents SciPy duplicate requests from consuming the contract budget.
    """

    all_ids = tuple(int(value) for value in load_ids)
    selected = tuple(int(value) for value in selected_load_ids)
    if budget <= 0:
        raise ValueError("alpha budget must be positive")
    cache: dict[tuple[float, ...], Any] = {}
    trace: list[dict[str, Any]] = []
    best_result = None
    best_alpha = {load_id: 1.0 for load_id in all_ids}
    best_objective = float("inf")

    def evaluate(x: Iterable[float]):
        nonlocal best_result, best_alpha, best_objective
        key = tuple(round(float(value), 12) for value in x)
        if key in cache:
            return cache[key]
        if len(cache) >= budget:
            return None
        alpha = {load_id: 1.0 for load_id in all_ids}
        for load_id, value in zip(selected, key, strict=True):
            alpha[load_id] = float(np.clip(value, 0.0, 1.0))
        result = evaluator(alpha)
        cache[key] = result
        objective = objective_getter(result)
        trace.append(
            {
                "evaluation_index": len(cache),
                "selected_alpha": __import__("json").dumps(
                    {str(load_id): alpha[load_id] for load_id in selected}, sort_keys=True
                ),
                "objective": objective,
            }
        )
        if objective is not None and np.isfinite(objective) and objective < best_objective:
            best_objective = float(objective)
            best_result = result
            best_alpha = alpha
        return result

    evaluate(np.ones(len(selected), dtype=float))
    if selected and len(cache) < budget:
        def objective(x):
            result = evaluate(x)
            if result is None:
                return best_objective if np.isfinite(best_objective) else 1e12
            value = objective_getter(result)
            return float(value) if value is not None and np.isfinite(value) else 1e12

        minimize(
            objective,
            np.ones(len(selected), dtype=float),
            method="Powell",
            bounds=[(0.0, 1.0)] * len(selected),
            options={"maxfev": int(budget), "disp": False},
        )
    # If Powell spent fewer unique calls due to repeated points, fill with a
    # deterministic diagonal design so accounting remains explicit.
    for step in range(1, budget + 1):
        if len(cache) >= budget or not selected:
            break
        value = 1.0 - step / budget
        evaluate(np.full(len(selected), value, dtype=float))
    return AlphaSearchResult(best_alpha=best_alpha, best_result=best_result, trace=tuple(trace))
