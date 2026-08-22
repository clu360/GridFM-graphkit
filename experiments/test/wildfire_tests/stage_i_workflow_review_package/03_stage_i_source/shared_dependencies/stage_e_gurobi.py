from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure


STAGE_E_LAMBDA_CASES = {
    "risk_priority": (0.8, 0.2),
    "balanced": (0.5, 0.5),
    "load_priority": (0.2, 0.8),
}

STAGE_E_LAMBDA_FOLDERS = {
    "risk_priority": "risk",
    "balanced": "bal",
    "load_priority": "load",
}

DEFAULT_PROXY_TYPE = "env_loading_base"
SUPPORTED_PROXY_TYPES = {DEFAULT_PROXY_TYPE, "env_only"}
METHOD_NAME = "gurobi_master_gridfm"


@dataclass(frozen=True)
class StageELambdaCase:
    name: str
    lambda_R: float
    lambda_L: float


def lambda_case(name: str) -> StageELambdaCase:
    if name not in STAGE_E_LAMBDA_CASES:
        raise ValueError(f"Unknown Stage E lambda case: {name}")
    lambda_R, lambda_L = STAGE_E_LAMBDA_CASES[name]
    return StageELambdaCase(name=name, lambda_R=float(lambda_R), lambda_L=float(lambda_L))


def normalize_true_exposure(raw_exposure: float, baseline_exposure: float) -> float:
    baseline = float(baseline_exposure)
    if not np.isfinite(baseline) or baseline <= 1e-12:
        raise ValueError(f"Cannot normalize wildfire exposure with baseline={baseline_exposure}.")
    value = float(raw_exposure) / baseline
    if not np.isfinite(value):
        raise ValueError(f"Normalized wildfire exposure is not finite: {value}")
    return float(value)


def validate_consequence_scores(c_by_line: Dict[int, float]) -> None:
    negative = {int(line_id): float(value) for line_id, value in c_by_line.items() if float(value) < -1e-12}
    if negative:
        raise ValueError(f"Line consequence scores must be nonnegative; got {negative}.")


def candidate_y_from_deenergized(candidate_line_ids: Iterable[int], deenergized_line_ids: Iterable[int]) -> Dict[int, int]:
    deenergized = {int(line_id) for line_id in deenergized_line_ids}
    return {int(line_id): int(int(line_id) in deenergized) for line_id in candidate_line_ids}


def z_from_y(y_by_line: Dict[int, int], num_lines: int) -> Dict[int, int]:
    return {int(line_id): int(1 - int(y_by_line.get(line_id, 0))) for line_id in range(int(num_lines))}


def deenergized_from_y(y_by_line: Dict[int, int]) -> List[int]:
    return sorted(int(line_id) for line_id, value in y_by_line.items() if int(value) == 1)


def vector_string(candidate_line_ids: Iterable[int], values_by_line: Dict[int, int]) -> str:
    return ",".join(str(int(values_by_line.get(int(line_id), 0))) for line_id in candidate_line_ids)


def no_good_hamming_distance(candidate_line_ids: Iterable[int], previous_y: Dict[int, int], proposed_y: Dict[int, int]) -> int:
    return int(
        sum(
            int(int(previous_y.get(int(line_id), 0)) != int(proposed_y.get(int(line_id), 0)))
            for line_id in candidate_line_ids
        )
    )


def proxy_denominator(candidate_line_ids: Iterable[int], p_env_by_line: Dict[int, float], baseline_loading: Sequence[float], proxy_type: str) -> float:
    if proxy_type not in SUPPORTED_PROXY_TYPES:
        raise ValueError(f"Unsupported proxy_type={proxy_type}.")
    loading = np.asarray(baseline_loading, dtype=float)
    total = 0.0
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        if proxy_type == "env_loading_base":
            total += float(p_env_by_line.get(line_id, 0.0)) * float(loading[line_id]) ** 2
        else:
            total += float(p_env_by_line.get(line_id, 0.0))
    if not np.isfinite(total) or total <= 1e-12:
        raise ValueError(f"Cannot compute R_hat with zero/invalid denominator for proxy_type={proxy_type}.")
    return float(total)


def compute_proxy_metrics(
    candidate_line_ids: Iterable[int],
    p_env_by_line: Dict[int, float],
    baseline_loading: Sequence[float],
    c_by_line: Dict[int, float],
    y_by_line: Dict[int, int],
    lambda_R_master: float,
    lambda_L_master: float,
    proxy_type: str = DEFAULT_PROXY_TYPE,
) -> Dict[str, float]:
    validate_consequence_scores(c_by_line)
    loading = np.asarray(baseline_loading, dtype=float)
    denominator = proxy_denominator(candidate_line_ids, p_env_by_line, loading, proxy_type)
    remaining = 0.0
    proxy_L_hat = 0.0
    for line_id in candidate_line_ids:
        line_id = int(line_id)
        y_l = int(y_by_line.get(line_id, 0))
        if proxy_type == "env_loading_base":
            risk_coeff = float(p_env_by_line.get(line_id, 0.0)) * float(loading[line_id]) ** 2
        else:
            risk_coeff = float(p_env_by_line.get(line_id, 0.0))
        remaining += risk_coeff * (1 - y_l)
        proxy_L_hat += float(c_by_line.get(line_id, 0.0)) * y_l
    proxy_R_hat = float(remaining / denominator)
    proxy_objective = float(float(lambda_R_master) * proxy_R_hat + float(lambda_L_master) * proxy_L_hat)
    return {
        "proxy_R_hat": proxy_R_hat,
        "proxy_L_hat": float(proxy_L_hat),
        "proxy_objective": proxy_objective,
        "proxy_R_denominator": denominator,
    }


def compute_true_metrics(
    loading_ratio: Sequence[float],
    p_env_by_line: Dict[int, float],
    z_by_line: Dict[int, int],
    candidate_line_ids: Iterable[int],
    baseline_R_raw_new: float,
    true_L_shed: float,
    true_P_AC: float,
    lambda_R_true: float,
    lambda_L_true: float,
    lambda_P: float = 0.0,
) -> Dict[str, float]:
    true_R_raw, true_by_line = compute_operational_wildfire_exposure(
        np.asarray(loading_ratio, dtype=float),
        p_env_by_line,
        z_by_line,
        candidate_line_ids,
    )
    true_R_norm = normalize_true_exposure(true_R_raw, baseline_R_raw_new)
    load_shed = float(true_L_shed)
    if load_shed < -1e-12 or load_shed > 1.0 + 1e-12:
        raise ValueError(f"L_shed should be in [0, 1], got {load_shed}.")
    true_objective = float(
        float(lambda_R_true) * true_R_norm
        + float(lambda_L_true) * load_shed
        + float(lambda_P) * float(true_P_AC)
    )
    return {
        "true_R_raw": float(true_R_raw),
        "true_R_norm": float(true_R_norm),
        "true_L_shed": load_shed,
        "true_P_AC": float(true_P_AC),
        "true_objective": true_objective,
        "true_exposure_by_line": true_by_line,
    }


def best_candidate_frame(evaluations: pd.DataFrame) -> pd.DataFrame:
    if evaluations.empty:
        return evaluations.copy()
    ok = evaluations[evaluations["status"] == "ok"].copy()
    if ok.empty:
        return ok
    ok = ok.sort_values(
        by=["true_objective", "true_L_shed", "num_deenergized_lines", "deenergized_line_ids"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    return ok.iloc[[0]].copy()

