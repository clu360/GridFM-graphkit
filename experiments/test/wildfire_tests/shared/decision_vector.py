from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.gridfm_support.scenario_data import ScenarioData


@dataclass
class DecisionBounds:
    u_min: np.ndarray
    u_max: np.ndarray
    u_base: np.ndarray


class FirstPassDecisionVector:
    """
    Reduced first-pass decision vector:

        u = [Delta_Pg_selected, alpha_selected]

    Non-selected loads keep alpha=1.0. Qg, y, and z stay fixed.
    """

    def __init__(
        self,
        scenario: ScenarioData,
        selected_generator_buses: Sequence[int],
        selected_load_buses: Sequence[int],
        delta_pg_bound_mw: float = 5.0,
        alpha_min: float = 0.00,
        alpha_max: float = 1.00,
    ):
        self.scenario = scenario
        self.selected_generator_buses = np.asarray(selected_generator_buses, dtype=int)
        self.selected_load_buses = np.asarray(selected_load_buses, dtype=int)
        self.delta_pg_bound_mw = float(delta_pg_bound_mw)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        self._validate_bus_ids()

        self.n_generators = int(len(self.selected_generator_buses))
        self.n_loads = int(len(self.selected_load_buses))
        self.n_total = self.n_generators + self.n_loads

        self.u_base = np.hstack(
            [
                np.zeros(self.n_generators, dtype=float),
                np.ones(self.n_loads, dtype=float),
            ]
        )
        self.u_min = np.hstack(
            [
                -self.delta_pg_bound_mw * np.ones(self.n_generators, dtype=float),
                self.alpha_min * np.ones(self.n_loads, dtype=float),
            ]
        )
        self.u_max = np.hstack(
            [
                self.delta_pg_bound_mw * np.ones(self.n_generators, dtype=float),
                self.alpha_max * np.ones(self.n_loads, dtype=float),
            ]
        )

    def _validate_bus_ids(self) -> None:
        for bus in np.hstack([self.selected_generator_buses, self.selected_load_buses]):
            if bus < 0 or bus >= self.scenario.num_buses:
                raise ValueError(f"Bus index {bus} is invalid for {self.scenario.num_buses} buses.")

    def split_decision_vector(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float)
        return u[: self.n_generators], u[self.n_generators :]

    def combine_decision_vector(self, delta_pg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return np.hstack([np.asarray(delta_pg, dtype=float), np.asarray(alpha, dtype=float)])

    def full_alpha(self, u: np.ndarray) -> np.ndarray:
        _, alpha_selected = self.split_decision_vector(u)
        alpha = np.ones(self.scenario.num_buses, dtype=float)
        if self.n_loads:
            alpha[self.selected_load_buses] = alpha_selected
        return alpha

    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        from gridfm_graphkit.datasets.globals import PD, PG, QD

        delta_pg, alpha_selected = self.split_decision_vector(u)
        node_features = self.scenario.get_baseline_node_features().copy()

        if self.n_generators:
            buses = self.selected_generator_buses
            node_features[buses, PG] = self.scenario.Pg_base[buses] + delta_pg

        if self.n_loads:
            buses = self.selected_load_buses
            node_features[buses, PD] = self.scenario.Pd_base[buses] * alpha_selected
            node_features[buses, QD] = self.scenario.Qd_base[buses] * alpha_selected

        return node_features

    def check_bounds(self, u: np.ndarray) -> Tuple[bool, str]:
        u = np.asarray(u, dtype=float)
        if u.shape != self.u_base.shape:
            return False, f"Decision has shape {u.shape}; expected {self.u_base.shape}."
        lower = int(np.sum(u < self.u_min - 1e-12))
        upper = int(np.sum(u > self.u_max + 1e-12))
        if lower or upper:
            return False, f"Decision bound violations: {lower} lower, {upper} upper"
        return True, "All bounds satisfied"

    def unserved_demand(self, u: np.ndarray, weights: np.ndarray | None = None) -> float:
        alpha = self.full_alpha(u)
        if weights is None:
            weights = np.maximum(self.scenario.Pd_base, 0.0)
        return float(np.sum(weights * (1.0 - alpha)))

    def equal_bus_load_shedding(self, u: np.ndarray) -> float:
        alpha = self.full_alpha(u)
        return float(np.sum(1.0 - alpha))

    def demand_weighted_load_shedding(self, u: np.ndarray) -> float:
        demand = np.maximum(np.asarray(self.scenario.Pd_base, dtype=float), 0.0)
        total_demand = float(np.sum(demand))
        if total_demand <= 1e-12:
            return 0.0
        weights = demand / total_demand
        alpha = self.full_alpha(u)
        return float(np.sum(weights * (1.0 - alpha)))

    def load_shedding(self, u: np.ndarray) -> float:
        return self.demand_weighted_load_shedding(u)

    def normalized_generator_movement(self, u: np.ndarray) -> float:
        delta_pg, _ = self.split_decision_vector(u)
        if len(delta_pg) == 0:
            return 0.0
        return float(np.sum((delta_pg / max(self.delta_pg_bound_mw, 1e-12)) ** 2))

    def metadata_frame(self, u: np.ndarray | None = None) -> pd.DataFrame:
        u = self.u_base if u is None else np.asarray(u, dtype=float)
        delta_pg, alpha = self.split_decision_vector(u)
        rows: list[Dict] = []
        for idx, bus in enumerate(self.selected_generator_buses):
            rows.append(
                {
                    "decision_type": "delta_pg",
                    "decision_index": idx,
                    "bus_idx": int(bus),
                    "value": float(delta_pg[idx]),
                    "lower_bound": float(self.u_min[idx]),
                    "upper_bound": float(self.u_max[idx]),
                }
            )
        offset = self.n_generators
        for idx, bus in enumerate(self.selected_load_buses):
            rows.append(
                {
                    "decision_type": "alpha",
                    "decision_index": offset + idx,
                    "bus_idx": int(bus),
                    "value": float(alpha[idx]),
                    "lower_bound": float(self.u_min[offset + idx]),
                    "upper_bound": float(self.u_max[offset + idx]),
                }
            )
        return pd.DataFrame(rows)


def auto_select_decision_buses(
    scenario: ScenarioData,
    n_generators: int,
    n_loads: int,
) -> tuple[np.ndarray, np.ndarray]:
    pv = scenario.get_pv_buses()
    pq = scenario.get_pq_buses()
    gen_scores = np.abs(scenario.Pg_base[pv]) if len(pv) else np.array([])
    load_scores = np.maximum(scenario.Pd_base[pq], 0.0) if len(pq) else np.array([])
    selected_g = pv[np.argsort(gen_scores)[::-1][:n_generators]] if len(pv) else np.array([], dtype=int)
    selected_l = pq[np.argsort(load_scores)[::-1][:n_loads]] if len(pq) else np.array([], dtype=int)
    return selected_g.astype(int), selected_l.astype(int)


class PgQgAlphaDecisionVector:
    """
    Stage G continuous recourse decision vector:

        u = [Delta_Pg_selected, Delta_Qg_selected, alpha_selected]

    Selected generator buses receive real and reactive generation controls.
    Selected load buses receive one alpha value that scales both Pd and Qd.
    """

    def __init__(
        self,
        scenario: ScenarioData,
        selected_generator_buses: Sequence[int],
        selected_load_buses: Sequence[int],
        delta_pg_bound_mw: float = 5.0,
        delta_qg_bound_mvar: float = 5.0,
        alpha_min: float = 0.0,
        alpha_max: float = 1.0,
    ):
        self.scenario = scenario
        self.selected_generator_buses = np.asarray(selected_generator_buses, dtype=int)
        self.selected_load_buses = np.asarray(selected_load_buses, dtype=int)
        self.delta_pg_bound_mw = float(delta_pg_bound_mw)
        self.delta_qg_bound_mvar = float(delta_qg_bound_mvar)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)

        self._validate_bus_ids()

        self.n_generators = int(len(self.selected_generator_buses))
        self.n_loads = int(len(self.selected_load_buses))
        self.n_delta_pg = self.n_generators
        self.n_delta_qg = self.n_generators
        self.n_total = self.n_delta_pg + self.n_delta_qg + self.n_loads

        self.u_base = np.hstack(
            [
                np.zeros(self.n_delta_pg, dtype=float),
                np.zeros(self.n_delta_qg, dtype=float),
                np.ones(self.n_loads, dtype=float),
            ]
        )
        self.u_min = np.hstack(
            [
                -self.delta_pg_bound_mw * np.ones(self.n_delta_pg, dtype=float),
                -self.delta_qg_bound_mvar * np.ones(self.n_delta_qg, dtype=float),
                self.alpha_min * np.ones(self.n_loads, dtype=float),
            ]
        )
        self.u_max = np.hstack(
            [
                self.delta_pg_bound_mw * np.ones(self.n_delta_pg, dtype=float),
                self.delta_qg_bound_mvar * np.ones(self.n_delta_qg, dtype=float),
                self.alpha_max * np.ones(self.n_loads, dtype=float),
            ]
        )

    def _validate_bus_ids(self) -> None:
        for bus in np.hstack([self.selected_generator_buses, self.selected_load_buses]):
            if bus < 0 or bus >= self.scenario.num_buses:
                raise ValueError(f"Bus index {bus} is invalid for {self.scenario.num_buses} buses.")

    @property
    def alpha_offset(self) -> int:
        return self.n_delta_pg + self.n_delta_qg

    def split_decision_vector(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.asarray(u, dtype=float)
        i1 = self.n_delta_pg
        i2 = i1 + self.n_delta_qg
        return u[:i1], u[i1:i2], u[i2:]

    def combine_decision_vector(self, delta_pg: np.ndarray, delta_qg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        return np.hstack(
            [
                np.asarray(delta_pg, dtype=float),
                np.asarray(delta_qg, dtype=float),
                np.asarray(alpha, dtype=float),
            ]
        )

    def full_alpha(self, u: np.ndarray) -> np.ndarray:
        _delta_pg, _delta_qg, alpha_selected = self.split_decision_vector(u)
        alpha = np.ones(self.scenario.num_buses, dtype=float)
        if self.n_loads:
            alpha[self.selected_load_buses] = alpha_selected
        return alpha

    def controlled_values(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        node_features = self.u_to_node_features(u)
        return {
            "Pd": node_features[:, 0].copy(),
            "Qd": node_features[:, 1].copy(),
            "Pg": node_features[:, 2].copy(),
            "Qg": node_features[:, 3].copy(),
        }

    def controlled_feature_mask(self, u: np.ndarray | None = None) -> np.ndarray:
        mask = np.zeros((self.scenario.num_buses, 6), dtype=bool)
        if self.n_generators:
            mask[self.selected_generator_buses, 2] = True
            mask[self.selected_generator_buses, 3] = True
        if self.n_loads:
            mask[self.selected_load_buses, 0] = True
            mask[self.selected_load_buses, 1] = True
        return mask[:, : int(self.scenario.mask.shape[1])]

    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        delta_pg, delta_qg, alpha_selected = self.split_decision_vector(u)
        node_features = self.scenario.get_baseline_node_features().copy()

        if self.n_generators:
            buses = self.selected_generator_buses
            node_features[buses, 2] = self.scenario.Pg_base[buses] + delta_pg
            node_features[buses, 3] = self.scenario.Qg_base[buses] + delta_qg

        if self.n_loads:
            buses = self.selected_load_buses
            node_features[buses, 0] = self.scenario.Pd_base[buses] * alpha_selected
            node_features[buses, 1] = self.scenario.Qd_base[buses] * alpha_selected

        return node_features

    def check_bounds(self, u: np.ndarray) -> Tuple[bool, str]:
        u = np.asarray(u, dtype=float)
        if u.shape != self.u_base.shape:
            return False, f"Decision has shape {u.shape}; expected {self.u_base.shape}."
        lower = int(np.sum(u < self.u_min - 1e-12))
        upper = int(np.sum(u > self.u_max + 1e-12))
        if lower or upper:
            return False, f"Decision bound violations: {lower} lower, {upper} upper"
        return True, "All bounds satisfied"

    def demand_weighted_load_shedding(self, u: np.ndarray) -> float:
        demand = np.maximum(np.asarray(self.scenario.Pd_base, dtype=float), 0.0)
        total_demand = float(np.sum(demand))
        if total_demand <= 1e-12:
            return 0.0
        alpha = self.full_alpha(u)
        return float(np.sum(demand * (1.0 - alpha)) / total_demand)

    def load_shedding(self, u: np.ndarray) -> float:
        return self.demand_weighted_load_shedding(u)

    def metadata_frame(self, u: np.ndarray | None = None) -> pd.DataFrame:
        u = self.u_base if u is None else np.asarray(u, dtype=float)
        delta_pg, delta_qg, alpha = self.split_decision_vector(u)
        rows: list[Dict] = []
        for idx, bus in enumerate(self.selected_generator_buses):
            rows.append(
                {
                    "decision_type": "delta_pg",
                    "decision_index": idx,
                    "bus_idx": int(bus),
                    "value": float(delta_pg[idx]),
                    "lower_bound": float(self.u_min[idx]),
                    "upper_bound": float(self.u_max[idx]),
                }
            )
        qg_offset = self.n_delta_pg
        for idx, bus in enumerate(self.selected_generator_buses):
            rows.append(
                {
                    "decision_type": "delta_qg",
                    "decision_index": qg_offset + idx,
                    "bus_idx": int(bus),
                    "value": float(delta_qg[idx]),
                    "lower_bound": float(self.u_min[qg_offset + idx]),
                    "upper_bound": float(self.u_max[qg_offset + idx]),
                }
            )
        alpha_offset = self.alpha_offset
        for idx, bus in enumerate(self.selected_load_buses):
            rows.append(
                {
                    "decision_type": "alpha",
                    "decision_index": alpha_offset + idx,
                    "bus_idx": int(bus),
                    "value": float(alpha[idx]),
                    "lower_bound": float(self.u_min[alpha_offset + idx]),
                    "upper_bound": float(self.u_max[alpha_offset + idx]),
                }
            )
        return pd.DataFrame(rows)
