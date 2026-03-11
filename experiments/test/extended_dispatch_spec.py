"""
Extended dispatch specification combining PV generation and load shedding.

The combined decision vector is

    u = [u_Pg ; u_delta]

where `u_Pg` are PV-bus generator redispatch variables and `u_delta` are PQ-bus
load shedding variables in MW. For the restored IEEE-30 single-scenario path,
this is currently 5 generator variables plus 24 shedding variables.
"""

import numpy as np
from typing import Dict, Tuple

from .scenario_data import ScenarioData
from .pv_dispatch import PVDispatchDecisionSpec
from .load_shedding_spec import LoadSheddingSpec


class ExtendedDispatchSpec:
    def __init__(
        self,
        scenario: ScenarioData,
        max_shed_fraction: float = 1.0,
    ):
        self.scenario = scenario
        self.pv_spec = PVDispatchDecisionSpec(scenario)
        self.shed_spec = LoadSheddingSpec(scenario, max_shed_fraction)

        self.n_pv = self.pv_spec.n_pv
        self.n_pq = self.shed_spec.n_pq
        self.n_total = self.n_pv + self.n_pq

        self.u_min = np.hstack([self.pv_spec.u_min, self.shed_spec.delta_min])
        self.u_base = np.hstack([self.pv_spec.u_base, self.shed_spec.delta_base])
        self.u_max = np.hstack([self.pv_spec.u_max, self.shed_spec.delta_max])

    def split_decision_vector(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Pg_pv = u[:self.n_pv]
        delta = u[self.n_pv:]
        return Pg_pv, delta

    def combine_decision_vector(self, Pg_pv: np.ndarray, delta: np.ndarray) -> np.ndarray:
        return np.hstack([Pg_pv, delta])

    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        from gridfm_graphkit.datasets.globals import PD, PG

        Pg_pv, delta = self.split_decision_vector(u)
        node_features = self.scenario.get_baseline_node_features().copy()
        node_features[self.pv_spec.pv_bus_indices, PG] = Pg_pv
        node_features[:, PD] = self.shed_spec.delta_to_Pd(delta)
        return node_features

    def check_bounds(self, u: np.ndarray) -> Tuple[bool, str]:
        violations_min = np.sum(u < self.u_min)
        violations_max = np.sum(u > self.u_max)

        if violations_min > 0 or violations_max > 0:
            msg = f"Decision bound violations: {violations_min} lower, {violations_max} upper"
            return False, msg

        return True, "All bounds satisfied"

    def get_distance_from_baseline(self, u: np.ndarray) -> float:
        return float(np.linalg.norm(u - self.u_base))

    def get_summary(self) -> Dict[str, float]:
        Pg_pv, _ = self.split_decision_vector(self.u_base)
        summary = {
            "n_total": self.n_total,
            "n_pv": self.n_pv,
            "n_pq": self.n_pq,
            "Pg_pv_base_mean": float(np.mean(Pg_pv)),
            "Pg_pv_bounds_range": float(np.mean(self.pv_spec.u_max - self.pv_spec.u_min)),
        }
        shed_summary = self.shed_spec.get_summary()
        summary.update({f"shed_{k}": v for k, v in shed_summary.items()})
        return summary
