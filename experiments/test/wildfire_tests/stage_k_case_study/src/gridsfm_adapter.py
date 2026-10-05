"""Build and mutate the official raw GridSFM JSON representation of Texas2k."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping

import numpy as np

from .identity import StageKIdentity, source_less_load_ids


def build_raw_gridsfm_case(identity: StageKIdentity, *, base_mva: float = 100.0) -> dict[str, Any]:
    """Convert canonical MATPOWER rows to the released GridSFM raw schema.

    Electrical powers and ratings are converted to per unit. Angle limits and
    phase shifts are converted from MATPOWER degrees to radians.
    """

    buses = identity.buses
    generators = identity.generators.loc[identity.generators["GEN_STATUS"] > 0].copy()
    loads = identity.loads
    branches = identity.branches
    bus_to_index = dict(zip(buses["bus_id"].astype(int), buses["canonical_bus_index"].astype(int), strict=True))

    bus_nodes = [
        [float(row.BASE_KV), int(row.BUS_TYPE), float(row.VMIN), float(row.VMAX)]
        for row in buses.itertuples(index=False)
    ]
    cost_case = __import__(
        "experiments.test.wildfire_tests.stage_k_case_study.stage_k_texas2k_helpers",
        fromlist=["load_stage_k_inputs"],
    ).load_stage_k_inputs()["gencost"]
    cost_case = cost_case.iloc[generators["canonical_generator_id"].astype(int)].reset_index(drop=True)
    if len(cost_case) != len(generators):
        raise ValueError("online generator and selected gencost row counts differ")
    generator_nodes = []
    for row, cost in zip(generators.itertuples(index=False), cost_case.itertuples(index=False), strict=True):
        generator_nodes.append(
            [
                float(row.MBASE) / base_mva,
                float(row.PG) / base_mva,
                float(row.PMIN) / base_mva,
                float(row.PMAX) / base_mva,
                float(row.QG) / base_mva,
                float(row.QMIN) / base_mva,
                float(row.QMAX) / base_mva,
                float(row.VG),
                float(cost.C2) * base_mva**2,
                float(cost.C1) * base_mva,
                float(cost.C0),
            ]
        )
    load_nodes = [
        [float(row.pd_requested_mw) / base_mva, float(row.qd_requested_mvar) / base_mva]
        for row in loads.itertuples(index=False)
    ]
    shunt_rows = buses.loc[(buses["GS"].abs() > 0.0) | (buses["BS"].abs() > 0.0)]
    shunt_nodes = [[float(row.BS) / base_mva, float(row.GS) / base_mva] for row in shunt_rows.itertuples(index=False)]

    edge_blocks: dict[str, dict[str, list[Any]]] = {}
    branch_ids_by_family: dict[str, list[int]] = {}
    for family in ("ac_line", "transformer"):
        frame = branches.loc[branches["branch_family"] == family]
        senders, receivers, features, branch_ids = [], [], [], []
        for row in frame.itertuples(index=False):
            senders.append(bus_to_index[int(row.from_bus_id)])
            receivers.append(bus_to_index[int(row.to_bus_id)])
            branch_ids.append(int(row.canonical_branch_id))
            angmin = np.deg2rad(float(row.ANGMIN))
            angmax = np.deg2rad(float(row.ANGMAX))
            rate_a = float(row.rate_a_mva) / base_mva
            rate_b = float(row.RATE_B) / base_mva
            rate_c = float(row.RATE_C) / base_mva
            b_half = float(row.BR_B) / 2.0
            if family == "ac_line":
                features.append([
                    angmin, angmax, b_half, b_half, float(row.BR_R), float(row.BR_X),
                    rate_a, rate_b, rate_c,
                ])
            else:
                tap = float(row.TAP) if abs(float(row.TAP)) > 1e-12 else 1.0
                features.append([
                    angmin, angmax, float(row.BR_R), float(row.BR_X), rate_a, rate_b,
                    rate_c, tap, np.deg2rad(float(row.SHIFT)), b_half, b_half,
                ])
        edge_blocks[family] = {"senders": senders, "receivers": receivers, "features": features}
        branch_ids_by_family[family] = branch_ids

    gen_senders = list(range(len(generators)))
    gen_receivers = [bus_to_index[int(value)] for value in generators["bus_id"]]
    load_senders = list(range(len(loads)))
    load_receivers = [bus_to_index[int(value)] for value in loads["bus_id"]]
    shunt_senders = list(range(len(shunt_rows)))
    shunt_receivers = [bus_to_index[int(value)] for value in shunt_rows["bus_id"]]
    edge_blocks.update(
        {
            "generator_link": {"senders": gen_senders, "receivers": gen_receivers},
            "load_link": {"senders": load_senders, "receivers": load_receivers},
            "shunt_link": {"senders": shunt_senders, "receivers": shunt_receivers},
        }
    )
    return {
        "grid": {
            "nodes": {"bus": bus_nodes, "generator": generator_nodes, "load": load_nodes, "shunt": shunt_nodes},
            "edges": edge_blocks,
            "context": [base_mva],
        },
        "solution": {"nodes": {}, "edges": {}, "duals": {}},
        "metadata": {
            "bus_id_map": buses["bus_id"].astype(int).tolist(),
            "gen_id_map": generators["canonical_generator_id"].astype(int).tolist(),
            "load_id_map": loads["canonical_load_id"].astype(int).tolist(),
            "gen_bus_map": generators["bus_id"].astype(int).tolist(),
            "load_bus_map": loads["bus_id"].astype(int).tolist(),
            "ac_line_branch_ids": branch_ids_by_family["ac_line"],
            "transformer_branch_ids": branch_ids_by_family["transformer"],
            "scenario_id": 16,
            "source": "modifiedTexas2k_stage_k",
            "feasible": True,
        },
    }


def mutate_candidate(
    raw_case: Mapping[str, Any],
    identity: StageKIdentity,
    *,
    offline_branch_ids: Iterable[int],
    alpha_requested: Mapping[int, float],
) -> tuple[dict[str, Any], dict[int, float], tuple[int, ...]]:
    offline = {int(value) for value in offline_branch_ids}
    unknown = offline.difference(identity.l_trans)
    if unknown:
        raise ValueError(f"attempt to open non-L_trans branches: {sorted(unknown)}")
    islanded = source_less_load_ids(identity, offline)
    islanded_set = set(islanded)
    mutated = deepcopy(raw_case)
    effective: dict[int, float] = {}
    for load_id, row in enumerate(mutated["grid"]["nodes"]["load"]):
        alpha = float(alpha_requested.get(load_id, 1.0))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha[{load_id}] outside [0, 1]")
        effective[load_id] = 0.0 if load_id in islanded_set else alpha
        row[0] *= effective[load_id]
        row[1] *= effective[load_id]
    ac_ids = list(mutated["metadata"]["ac_line_branch_ids"])
    keep = [int(branch_id) not in offline for branch_id in ac_ids]
    block = mutated["grid"]["edges"]["ac_line"]
    for key in ("senders", "receivers", "features"):
        block[key] = [value for value, retain in zip(block[key], keep, strict=True) if retain]
    mutated["metadata"]["ac_line_branch_ids"] = [value for value, retain in zip(ac_ids, keep, strict=True) if retain]
    mutated["metadata"]["stage_k_mutation"] = {
        "offline_branch_ids": sorted(offline),
        "alpha_requested": {str(key): float(value) for key, value in alpha_requested.items()},
        "alpha_effective": {str(key): value for key, value in effective.items()},
        "source_less_load_ids": list(islanded),
    }
    return mutated, effective, islanded
