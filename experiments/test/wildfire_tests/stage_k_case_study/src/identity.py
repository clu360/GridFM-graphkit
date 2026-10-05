"""Canonical modified-Texas2k identity and authoritative snapshot binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.goc500_adapter import (
    BranchIdentity,
    GOC500Identity,
    LoadIdentity,
)
from experiments.test.wildfire_tests.stage_k_case_study.stage_k_texas2k_helpers import (
    build_branch_geography,
    load_stage_k_inputs,
)


@dataclass(frozen=True)
class StageKIdentity:
    buses: pd.DataFrame
    loads: pd.DataFrame
    generators: pd.DataFrame
    branches: pd.DataFrame
    branch_model_mapping: pd.DataFrame
    stage_j_identity: GOC500Identity

    @property
    def l_trans(self) -> tuple[int, ...]:
        rows = self.branches.loc[self.branches["is_switchable_transmission"]]
        return tuple(rows["canonical_branch_id"].astype(int))

    @property
    def l_fixed(self) -> tuple[int, ...]:
        rows = self.branches.loc[~self.branches["is_switchable_transmission"]]
        return tuple(rows["canonical_branch_id"].astype(int))


def _scenario_loads(case: dict[str, object], scenario: int) -> pd.DataFrame:
    buses = case["bus"].copy()
    selected = case["loads"].loc[case["loads"]["scenario"] == scenario].copy()
    if selected["bus_id"].duplicated().any():
        raise ValueError(f"Scenario {scenario} contains duplicate bus_id load rows")
    row_to_bus = buses.set_index("row_id")["BUS_I"].astype(int)
    missing = sorted(set(selected["bus_id"].astype(int)).difference(row_to_bus.index.astype(int)))
    if missing:
        raise ValueError(f"Scenario {scenario} load bus rows missing from MATPOWER case: {missing[:10]}")
    selected["canonical_load_id"] = np.arange(len(selected), dtype=int)
    selected["bus_row_id"] = selected["bus_id"].astype(int)
    bus_demand = buses.set_index("row_id")[["PD", "QD"]]
    selected["pd_parquet_mw"] = selected["p_mw"].astype(float)
    selected["qd_parquet_mvar"] = selected["q_mvar"].astype(float)
    selected["pd_requested_mw"] = selected["bus_row_id"].map(bus_demand["PD"]).astype(float)
    selected["qd_requested_mvar"] = selected["bus_row_id"].map(bus_demand["QD"]).astype(float)
    selected["case_to_scenario16_p_scale"] = (
        selected["pd_requested_mw"] / selected["pd_parquet_mw"].replace(0.0, np.nan)
    )
    selected["bus_id"] = selected["bus_row_id"].map(row_to_bus).astype(int)
    columns = [
        "canonical_load_id", "bus_row_id", "bus_id", "scenario", "timestamp",
        "pd_parquet_mw", "qd_parquet_mvar", "case_to_scenario16_p_scale",
        "pd_requested_mw", "qd_requested_mvar",
    ]
    return selected[columns].sort_values("canonical_load_id").reset_index(drop=True)


def build_identity(
    *,
    scenario: int = 16,
    environment_path: str | Path | None = None,
) -> StageKIdentity:
    case = load_stage_k_inputs()
    buses = case["bus"].copy()
    buses = buses.rename(columns={"row_id": "canonical_bus_index", "BUS_I": "bus_id"})
    buses["bus_id"] = buses["bus_id"].astype(int)
    buses["canonical_bus_index"] = buses["canonical_bus_index"].astype(int)

    loads = _scenario_loads(case, scenario)
    generators = case["gen"].copy().rename(columns={"gen_id": "canonical_generator_id", "GEN_BUS": "bus_id"})
    generators["canonical_generator_id"] = generators["canonical_generator_id"].astype(int)
    generators["bus_id"] = generators["bus_id"].astype(int)

    branch_geo = build_branch_geography(case)
    environment_path = Path(environment_path) if environment_path else (
        Path(__file__).resolve().parents[2]
        / "texas_2k_results"
        / "stage_k"
        / "environment_snapshot_tau0p50" / "data" / "cum_hazard_risk.parquet"
    )
    environment = pd.read_parquet(environment_path)
    required = {
        "branch_id", "baseline_loading", "hazard_score_p_cumulative",
        "weather_coverage_valid", "weather_timestamp", "weather_timestamp_utc",
        "electrical_scenario",
    }
    missing_columns = required.difference(environment.columns)
    if missing_columns:
        raise ValueError(f"environment snapshot missing columns: {sorted(missing_columns)}")
    if environment["branch_id"].duplicated().any():
        raise ValueError("environment snapshot contains duplicate branch_id rows")
    if set(environment["weather_timestamp"].astype(str)) != {"2023-06-23 16:00 CDT"}:
        raise ValueError("environment snapshot is not the frozen June 23 16:00 CDT snapshot")
    if set(environment["electrical_scenario"].astype(int)) != {scenario}:
        raise ValueError("environment snapshot electrical scenario mismatch")

    branches = branch_geo.merge(environment[list(required)], on="branch_id", how="left", validate="one_to_one")
    branches = branches.rename(
        columns={
            "branch_id": "canonical_branch_id",
            "F_BUS": "from_bus_id",
            "T_BUS": "to_bus_id",
            "RATE_A": "rate_a_mva",
            "BR_STATUS": "status",
            "hazard_score_p_cumulative": "p_env",
        }
    )
    branches["canonical_branch_id"] = branches["canonical_branch_id"].astype(int)
    branches["powermodels_branch_id"] = branches["canonical_branch_id"] + 1
    branches["from_bus_id"] = branches["from_bus_id"].astype(int)
    branches["to_bus_id"] = branches["to_bus_id"].astype(int)
    is_transformer = (branches["TAP"].abs() > 1e-12) | (
        (branches["from_BASE_KV"] - branches["to_BASE_KV"]).abs() > 1e-9
    )
    branches["branch_family"] = np.where(is_transformer, "transformer", "ac_line")
    branches["is_switchable_transmission"] = (
        ~is_transformer
        & (branches["status"] == 1)
        & np.isfinite(branches["rate_a_mva"])
        & (branches["rate_a_mva"] > 0.0)
    )
    branches["exclusion_reason"] = np.where(
        branches["is_switchable_transmission"], "", "transformer_or_other_fixed"
    )
    if branches[["baseline_loading", "p_env", "rate_a_mva"]].isna().any().any():
        raise ValueError("canonical branch/environment join is incomplete")
    if not branches["weather_coverage_valid"].astype(bool).all():
        raise ValueError("frozen environmental snapshot lacks complete branch coverage")

    family_counts = branches.groupby("branch_family").cumcount().astype(int)
    model_mapping = pd.DataFrame(
        {
            "canonical_branch_id": branches["canonical_branch_id"].astype(int),
            "powermodels_branch_id": branches["powermodels_branch_id"].astype(int),
            "model_edge_family": branches["branch_family"],
            "model_family_index": family_counts,
            "from_bus_id": branches["from_bus_id"].astype(int),
            "to_bus_id": branches["to_bus_id"].astype(int),
            "orientation_sign": 1,
        }
    )

    branch_records = tuple(
        BranchIdentity(
            canonical_branch_id=int(row.canonical_branch_id),
            original_case_branch_id=int(row.powermodels_branch_id),
            edge_family=str(row.branch_family),
            family_index=int(model_mapping.loc[idx, "model_family_index"]),
            from_bus_id=int(row.from_bus_id),
            to_bus_id=int(row.to_bus_id),
            from_bus_index=int(row.from_row_id),
            to_bus_index=int(row.to_row_id),
            orientation_sign=1,
            rate_a=float(row.rate_a_mva),
            is_risk=bool(row.is_switchable_transmission),
            is_candidate=bool(row.is_switchable_transmission),
        )
        for idx, row in branches.iterrows()
    )
    bus_index = buses.set_index("bus_id")["canonical_bus_index"].to_dict()
    load_records = tuple(
        LoadIdentity(
            canonical_load_id=int(row.canonical_load_id),
            load_index=int(row.canonical_load_id),
            bus_id=int(row.bus_id),
            bus_index=int(bus_index[int(row.bus_id)]),
            pd_pre=float(row.pd_requested_mw),
            qd_pre=float(row.qd_requested_mvar),
        )
        for row in loads.itertuples(index=False)
    )
    source_buses = tuple(sorted(set(
        generators.loc[generators["GEN_STATUS"] > 0, "bus_id"].astype(int)
    )))
    stage_j_identity = GOC500Identity(
        branches=branch_records,
        loads=load_records,
        bus_ids=tuple(buses["bus_id"].astype(int)),
        generator_bus_ids=source_buses,
    )
    return StageKIdentity(
        buses=buses.reset_index(drop=True),
        loads=loads,
        generators=generators.reset_index(drop=True),
        branches=branches.reset_index(drop=True),
        branch_model_mapping=model_mapping.reset_index(drop=True),
        stage_j_identity=stage_j_identity,
    )


def source_less_load_ids(identity: StageKIdentity, offline_branch_ids: Iterable[int]) -> tuple[int, ...]:
    from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.goc500_adapter import source_less_load_ids as reused

    return reused(identity.stage_j_identity, offline_branch_ids)
