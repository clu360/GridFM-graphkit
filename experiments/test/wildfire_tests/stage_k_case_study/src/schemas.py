"""Stage K evaluator result contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class ResultStatus(str, Enum):
    SUCCESS = "optimal_success"
    LOCALLY_SOLVED = "locally_solved"
    INFEASIBLE = "infeasible"
    ITERATION_LIMIT = "iteration_limit"
    TIME_LIMIT = "time_limit"
    NUMERICAL_FAILURE = "numerical_failure"
    INPUT_MAPPING_FAILURE = "input_mapping_failure"
    EVALUATOR_EXCEPTION = "evaluator_exception"
    DUPLICATE_CANDIDATE = "duplicate_candidate"


ELIGIBLE_STATUSES = {ResultStatus.SUCCESS.value, ResultStatus.LOCALLY_SOLVED.value}


@dataclass
class CandidateResult:
    evaluator: str
    lambda_r: float
    k: int
    topology_key: str
    offline_branch_ids: str
    status: str
    eligible: bool = False
    search_objective: float | None = None
    r_norm: float | None = None
    l_shed_total: float | None = None
    j_trade: float | None = None
    pac_operational: float | None = None
    pac_ac: float | None = None
    pac_model: float | None = None
    pac_total: float | None = None
    j_total: float | None = None
    max_loading: float | None = None
    loading_gt_1_count: int | None = None
    selected_load_ids: str = ""
    alpha_effective_json: str = ""
    elapsed_seconds: float | None = None
    solver_seconds: float | None = None
    iterations: int | None = None
    peak_memory_mb: float | None = None
    message: str = ""
    state_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["metadata"] = __import__("json").dumps(row["metadata"], sort_keys=True)
        return row


def classify_solver_status(raw_status: str) -> str:
    value = raw_status.strip().lower()
    if any(token in value for token in ("optimal", "solve_succeeded", "success")):
        return ResultStatus.SUCCESS.value
    if any(token in value for token in ("locally_solved", "local")):
        return ResultStatus.LOCALLY_SOLVED.value
    if "infeasible" in value:
        return ResultStatus.INFEASIBLE.value
    if "iteration" in value or "max_iter" in value:
        return ResultStatus.ITERATION_LIMIT.value
    if "time" in value:
        return ResultStatus.TIME_LIMIT.value
    if any(token in value for token in ("numerical", "invalid", "diverg")):
        return ResultStatus.NUMERICAL_FAILURE.value
    return ResultStatus.EVALUATOR_EXCEPTION.value
