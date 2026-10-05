"""Configuration loading and frozen Stage K contract validation."""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


REQUIRED_SNAPSHOT = {
    "electrical_scenario": 16,
    "weather_timestamp": "2023-06-23 16:00 CDT",
    "weather_timestamp_utc": "2023-06-23T21:00:00+00:00",
    "p_env_column": "hazard_score_p_cumulative",
    "baseline_loading_column": "baseline_loading",
}
PAC_FREEZE = {"rho_phys": 2.0, "w_op": 1.0, "w_ac": 1.0, "w_model": 0.0}


def load_config(path: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Stage K config must be a YAML mapping")
    validate_config(config)
    resolved = deepcopy(config)
    resolved["config_sha256"] = config_hash(config)
    return resolved


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(payload.encode("utf-8")).hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    snapshot = config.get("snapshot", {})
    for key, expected in REQUIRED_SNAPSHOT.items():
        if snapshot.get(key) != expected:
            raise ValueError(f"frozen snapshot mismatch for {key}: {snapshot.get(key)!r} != {expected!r}")
    objective = config.get("objective", {})
    for key, expected in PAC_FREEZE.items():
        if float(objective.get(key, float("nan"))) != expected:
            raise ValueError(f"frozen PAC weight mismatch for {key}")
    lambdas = config.get("search", {}).get("lambda_r", [])
    if not lambdas or any(not 0.0 <= float(value) <= 1.0 for value in lambdas):
        raise ValueError("search.lambda_r must contain values in [0, 1]")
    search = config["search"]
    positive = (
        "k1_count",
        "parent_count",
        "k2_children_per_parent",
        "k2_max_unique",
        "gridsfm_q",
        "gridsfm_alpha_budget",
    )
    for key in positive:
        if int(search.get(key, 0)) <= 0:
            raise ValueError(f"search.{key} must be positive")
    expected_k2 = int(search["parent_count"]) * int(search["k2_children_per_parent"])
    if int(search["k2_max_unique"]) > expected_k2:
        raise ValueError("k2_max_unique cannot exceed parent_count * k2_children_per_parent")
