"""
Load shedding decision variable specification.

Defines how the optimization decision vector `delta` (load shed amounts at PQ
buses, in MW) maps to and from the full scenario representation.
"""

import numpy as np
from typing import Dict, Tuple

from .scenario_data import ScenarioData


class LoadSheddingSpec:
    """
    Decision variable specification for load shedding at PQ buses.
    """

    def __init__(self, scenario: ScenarioData, max_shed_fraction: float = 1.0):
        self.scenario = scenario
        self.max_shed_fraction = max_shed_fraction

        self.pq_bus_indices = scenario.get_pq_buses()
        self.n_pq = len(self.pq_bus_indices)

        Pd_pq = scenario.Pd_base[self.pq_bus_indices]
        self.delta_base = np.zeros(self.n_pq)
        self.delta_min = np.zeros(self.n_pq)
        self.delta_max = Pd_pq * max_shed_fraction

    def delta_to_Pd(self, delta: np.ndarray) -> np.ndarray:
        """
        Convert a shedding vector into the full post-shedding Pd array.
        """
        Pd = self.scenario.Pd_base.copy()
        Pd[self.pq_bus_indices] = Pd[self.pq_bus_indices] - delta
        return Pd

    def Pd_to_delta(self, Pd: np.ndarray) -> np.ndarray:
        """
        Extract the shedding vector from a full Pd array.
        """
        delta = self.scenario.Pd_base[self.pq_bus_indices] - Pd[self.pq_bus_indices]
        return delta.copy()

    def check_bounds(self, delta: np.ndarray) -> Tuple[bool, str]:
        """
        Check if the shedding vector satisfies its box constraints.
        """
        violations_min = np.sum(delta < self.delta_min)
        violations_max = np.sum(delta > self.delta_max)

        if violations_min > 0 or violations_max > 0:
            msg = f"Shedding bound violations: {violations_min} lower, {violations_max} upper"
            return False, msg

        return True, "Shedding bounds satisfied"

    def get_total_shed(self, delta: np.ndarray) -> float:
        return float(np.sum(delta))

    def get_shed_fraction(self, delta: np.ndarray) -> float:
        total_pq_demand = float(np.sum(self.scenario.Pd_base[self.pq_bus_indices]))
        if total_pq_demand <= 0.0:
            return 0.0
        return float(np.sum(delta)) / total_pq_demand

    def get_summary(self) -> Dict[str, float]:
        Pd_pq = self.scenario.Pd_base[self.pq_bus_indices]
        baseline_pq_demand = float(np.sum(Pd_pq))
        max_shed_total = float(np.sum(self.delta_max))

        return {
            "n_pq": self.n_pq,
            "max_shed_total_MW": max_shed_total,
            "max_shed_fraction": self.max_shed_fraction,
            "baseline_pq_demand_MW": baseline_pq_demand,
            "max_shed_pct_of_pq": 100.0 * max_shed_total / baseline_pq_demand
            if baseline_pq_demand > 0.0
            else 0.0,
        }
