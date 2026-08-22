"""Stage J baseline, proxy, and diagnostic scenario construction."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .goc500_adapter import GOC500Identity, source_less_load_ids
from .metrics import compute_r_base


@dataclass(frozen=True)
class DiagnosticScenario:
    """One Stage J wildfire scenario paired with a shared electrical scenario."""

    scenario_id: str
    electrical_scenario_id: str
    wildfire_scenario_id: str
    description: str
    target_branch_ids: tuple[int, ...]
    expected_behavior: str
    r_base: float
    p_env_by_line: Mapping[int, float]


def load_baseline_loading_csv(path: str | Path) -> dict[int, float]:
    """Load exact intact AC baseline loading keyed by canonical branch ID."""

    out: dict[int, float] = {}
    with Path(path).open(newline="") as fh:
        for row in csv.DictReader(fh):
            branch_id = int(row["branch_id"])
            loading = float(row["baseline_loading"])
            if not np.isfinite(loading):
                raise ValueError(f"baseline_loading for branch {branch_id} is not finite")
            out[branch_id] = loading
    return out


def connectivity_service_impact_proxy(identity: GOC500Identity, candidate_branch_ids: Iterable[int]) -> dict[int, float]:
    """Compute shared source-less single-outage service consequence `c_l`.

    This is the conservative connectivity/source-less-only proxy from the Stage J
    design. It is model-independent and is supplied unchanged to Guided-DC and
    Guided-GridSFM.
    """

    total_pd = sum(load.pd_pre for load in identity.loads)
    if total_pd <= 0.0:
        raise ValueError("total pre-intervention demand must be positive to compute c_l")
    pd_by_load = {load.canonical_load_id: load.pd_pre for load in identity.loads}
    out: dict[int, float] = {}
    valid_branch_ids = set(identity.branch_by_id)
    for branch_id in candidate_branch_ids:
        bid = int(branch_id)
        if bid not in valid_branch_ids:
            raise KeyError(f"candidate branch {bid} is not in identity")
        source_less = source_less_load_ids(identity, [bid])
        source_less_pd = sum(pd_by_load[load_id] for load_id in source_less)
        out[bid] = max(0.0, source_less_pd / total_pd)
    return out


def has_alternate_path_after_outage(identity: GOC500Identity, branch_id: int) -> bool:
    """Return whether the branch endpoints remain connected without `branch_id`."""

    branch = identity.branch_by_id[int(branch_id)]
    parent = {bus_id: bus_id for bus_id in identity.bus_ids}

    def find(bus_id: int) -> int:
        while parent[bus_id] != bus_id:
            parent[bus_id] = parent[parent[bus_id]]
            bus_id = parent[bus_id]
        return bus_id

    def union(a: int, b: int) -> None:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[rb] = ra

    for record in identity.branches:
        if record.canonical_branch_id == int(branch_id):
            continue
        union(record.from_bus_id, record.to_bus_id)
    return find(branch.from_bus_id) == find(branch.to_bus_id)


def construct_stage_j_s1_s3_scenarios(
    identity: GOC500Identity,
    *,
    baseline_loading: Mapping[int, float],
    c_by_line: Mapping[int, float],
    electrical_scenario_id: str = "e0",
    high_p_env: float = 1.0,
    low_p_env: float = 0.01,
    epsilon: float = 1e-9,
) -> tuple[list[DiagnosticScenario], list[dict[str, object]]]:
    """Construct the initial S1-S3 diagnostic wildfire scenarios."""

    candidate_ids = sorted(branch.canonical_branch_id for branch in identity.branches if branch.is_risk and branch.is_candidate)
    if not candidate_ids:
        candidate_ids = sorted(branch.canonical_branch_id for branch in identity.branches if branch.is_risk)
    if not candidate_ids:
        raise ValueError("Stage J scenario construction requires at least one risk branch")

    missing_loading = sorted(set(candidate_ids).difference(baseline_loading))
    missing_c = sorted(set(candidate_ids).difference(c_by_line))
    if missing_loading:
        raise KeyError(f"candidate branches missing baseline loading: {missing_loading[:10]}")
    if missing_c:
        raise KeyError(f"candidate branches missing c_l proxy: {missing_c[:10]}")

    score_rows = []
    for branch_id in candidate_ids:
        c_value = float(c_by_line[branch_id])
        loading = float(baseline_loading[branch_id])
        score_rows.append(
            {
                "branch_id": int(branch_id),
                "baseline_loading": loading,
                "c_l": c_value,
                "alternate_path_after_single_outage": has_alternate_path_after_outage(identity, branch_id),
            }
        )

    def unused(rows: list[dict[str, object]], already: set[int]) -> list[dict[str, object]]:
        return [row for row in rows if int(row["branch_id"]) not in already]

    chosen: set[int] = set()

    low_impact_rows = [row for row in score_rows if float(row["c_l"]) <= epsilon]
    if not low_impact_rows:
        low_impact_rows = score_rows[:]
    s1_row = max(low_impact_rows, key=lambda row: (float(row["baseline_loading"]), -float(row["c_l"])))
    chosen.add(int(s1_row["branch_id"]))

    high_impact_rows = [row for row in unused(score_rows, chosen) if float(row["c_l"]) > epsilon]
    if not high_impact_rows:
        high_impact_rows = unused(score_rows, chosen) or score_rows[:]
    s2_row = max(high_impact_rows, key=lambda row: (float(row["c_l"]), float(row["baseline_loading"])))
    chosen.add(int(s2_row["branch_id"]))

    redundant_rows = [
        row
        for row in unused(score_rows, chosen)
        if bool(row["alternate_path_after_single_outage"]) and float(row["c_l"]) <= epsilon
    ]
    if not redundant_rows:
        redundant_rows = [row for row in unused(score_rows, chosen) if bool(row["alternate_path_after_single_outage"])]
    if not redundant_rows:
        redundant_rows = unused(score_rows, chosen) or score_rows[:]
    s3_row = max(redundant_rows, key=lambda row: float(row["baseline_loading"]))

    specs = [
        (
            "J-S1",
            "S1 high-risk / low-impact line.",
            "Should prefer shutting a high-risk line when source-less service impact is low.",
            s1_row,
        ),
        (
            "J-S2",
            "S2 high-risk / high-impact line.",
            "Should reveal wildfire/service conflict when a high-risk line has high service consequence.",
            s2_row,
        ),
        (
            "J-S3",
            "S3 high-risk redundant-path line.",
            "Should identify a high-risk line whose endpoints remain connected after outage.",
            s3_row,
        ),
    ]

    scenarios: list[DiagnosticScenario] = []
    risk_line_ids = sorted(branch.canonical_branch_id for branch in identity.branches if branch.is_risk)
    for scenario_id, description, expected, row in specs:
        target = int(row["branch_id"])
        p_env = {line_id: float(low_p_env) for line_id in risk_line_ids}
        p_env[target] = float(high_p_env)
        r_base = compute_r_base(p_env, baseline_loading, risk_line_ids)
        scenarios.append(
            DiagnosticScenario(
                scenario_id=scenario_id,
                electrical_scenario_id=electrical_scenario_id,
                wildfire_scenario_id=scenario_id,
                description=description,
                target_branch_ids=(target,),
                expected_behavior=expected,
                r_base=r_base,
                p_env_by_line=p_env,
            )
        )

    return scenarios, score_rows


def write_stage_j_scenario_artifacts(
    scenarios: Iterable[DiagnosticScenario],
    score_rows: Iterable[Mapping[str, object]],
    *,
    out_dir: str | Path,
) -> dict[str, str]:
    """Write scenario/proxy tables using standard CSV/JSON only."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario_path = out / "stage_j_scenario_register.csv"
    score_path = out / "stage_j_candidate_line_scores.csv"
    p_env_path = out / "stage_j_p_env_by_scenario.csv"
    summary_path = out / "stage_j_scenario_summary.json"

    scenarios_list = list(scenarios)
    score_list = list(score_rows)

    with scenario_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "scenario_id",
                "electrical_scenario_id",
                "wildfire_scenario_id",
                "description",
                "target_branch_ids",
                "expected_behavior",
                "r_base",
            ],
        )
        writer.writeheader()
        for scenario in scenarios_list:
            writer.writerow(
                {
                    "scenario_id": scenario.scenario_id,
                    "electrical_scenario_id": scenario.electrical_scenario_id,
                    "wildfire_scenario_id": scenario.wildfire_scenario_id,
                    "description": scenario.description,
                    "target_branch_ids": ";".join(str(line_id) for line_id in scenario.target_branch_ids),
                    "expected_behavior": scenario.expected_behavior,
                    "r_base": scenario.r_base,
                }
            )

    with score_path.open("w", newline="") as fh:
        fieldnames = ["branch_id", "baseline_loading", "c_l", "alternate_path_after_single_outage"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in score_list:
            writer.writerow({key: row[key] for key in fieldnames})

    with p_env_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scenario_id", "branch_id", "p_env"])
        writer.writeheader()
        for scenario in scenarios_list:
            for branch_id, value in sorted(scenario.p_env_by_line.items()):
                writer.writerow({"scenario_id": scenario.scenario_id, "branch_id": branch_id, "p_env": value})

    with summary_path.open("w") as fh:
        json.dump(
            {
                "scenario_count": len(scenarios_list),
                "candidate_line_count": len(score_list),
                "proxy_backend": "connectivity_source_less_single_outage",
                "scenario_ids": [scenario.scenario_id for scenario in scenarios_list],
                "target_branch_ids": {
                    scenario.scenario_id: list(scenario.target_branch_ids) for scenario in scenarios_list
                },
            },
            fh,
            indent=2,
            sort_keys=True,
        )

    return {
        "scenario_register": str(scenario_path),
        "candidate_line_scores": str(score_path),
        "p_env_by_scenario": str(p_env_path),
        "summary": str(summary_path),
    }
