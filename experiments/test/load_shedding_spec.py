"""
Load shedding decision variable specification.

Defines how the optimization decision vector δ (load shed amounts at PQ buses)
maps to and from the full scenario representation.
"""

import numpy as np
from typing import Dict, Tuple
from .scenario_data import ScenarioData


class LoadSheddingSpec:
    """
    Decision variable specification for load shedding at PQ buses.
    
    Maps between the optimization variable δ (load shed at PQ buses)
    and the full bus demand representation.
    
    Attributes
    ----------
    pq_bus_indices : np.ndarray
        Indices of PQ buses in the scenario, shape (n_pq,)
    n_pq : int
        Number of PQ buses (dimension of shedding decision vector)
    delta_base : np.ndarray
        Baseline shedding (all zeros), shape (n_pq,)
    delta_min : np.ndarray
        Lower bounds on shedding (all zeros - can't negative shed), shape (n_pq,)
    delta_max : np.ndarray
        Upper bounds on shedding (up to 100% of Pd), shape (n_pq,)
    """
    
    def __init__(self, scenario: ScenarioData, max_shed_fraction: float = 1.0):
        """
        Initialize load shedding spec from scenario data.
        
        Parameters
        ----------
        scenario : ScenarioData
            Canonical scenario representation
        max_shed_fraction : float, optional
            Maximum fraction of Pd that can be shed (0 to 1, default: 1.0 = 100%)
        """
        self.scenario = scenario
        self.max_shed_fraction = max_shed_fraction
        
        # Identify PQ buses
        self.pq_bus_indices = scenario.get_pq_buses()
        self.n_pq = len(self.pq_bus_indices)
        
        # Get baseline demands at PQ buses
        Pd_pq = scenario.Pd_base[self.pq_bus_indices]
        
        # Shedding bounds
        self.delta_base = np.zeros(self.n_pq)  # No shedding at baseline
        self.delta_min = np.zeros(self.n_pq)   # Can't shed negative amount
        self.delta_max = Pd_pq * max_shed_fraction  # Max shed = fraction of baseline Pd
    
    def delta_to_Pd(self, delta: np.ndarray) -> np.ndarray:
        """
        Convert shedding vector δ to full Pd array after shedding.
        
        Parameters
        ----------
        delta : np.ndarray
            Load shedding amounts at PQ buses, shape (n_pq,)
        
        Returns
        -------
        np.ndarray
            Full Pd array after shedding, shape (num_buses,)
            Pd_new = Pd_base - δ at PQ buses, unchanged elsewhere
        """
        Pd = self.scenario.Pd_base.copy()
        Pd[self.pq_bus_indices] = Pd[self.pq_bus_indices] - delta
        return Pd
    
    def Pd_to_delta(self, Pd: np.ndarray) -> np.ndarray:
        """
        Extract shedding vector δ from full Pd array.
        
        Parameters
        ----------
        Pd : np.ndarray
            Full active demand array after shedding, shape (num_buses,)
        
        Returns
        -------
        np.ndarray
            Shedding amounts at PQ buses, shape (n_pq,)
        """
        delta = self.scenario.Pd_base[self.pq_bus_indices] - Pd[self.pq_bus_indices]
        return delta.copy()
    
    def check_bounds(self, delta: np.ndarray) -> Tuple[bool, str]:
        """
        Check if shedding vector satisfies bounds.
        
        Parameters
        ----------
        delta : np.ndarray
            Load shedding amounts at PQ buses, shape (n_pq,)
        
        Returns
        -------
        tuple
            (is_feasible: bool, message: str)
        """
        violations_min = np.sum(delta < self.delta_min)
        violations_max = np.sum(delta > self.delta_max)
        
        if violations_min > 0 or violations_max > 0:
            msg = f"Shedding bound violations: {violations_min} lower, {violations_max} upper"
            return False, msg
        
        return True, "Shedding bounds satisfied"
    
    def get_total_shed(self, delta: np.ndarray) -> float:
        """
        Compute total load shed across all PQ buses.
        
        Parameters
        ----------
        delta : np.ndarray
            Load shedding amounts at PQ buses, shape (n_pq,)
        
        Returns
        -------
        float
            Total shed in MW = sum(delta)
        """
        return float(np.sum(delta))
    
    def get_shed_fraction(self, delta: np.ndarray) -> float:
        """
        Compute fraction of baseline PQ demand that is shed.
        
        Parameters
        ----------
        delta : np.ndarray
            Load shedding amounts at PQ buses, shape (n_pq,)
        
        Returns
        -------
        float
            Fraction of baseline PQ load shed: sum(delta) / sum(Pd_pq_base)
        """
        total_pq_demand = np.sum(self.scenario.Pd_base[self.pq_bus_indices])
        return float(np.sum(delta)) / total_pq_demand
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of shedding capability.
        
        Returns
        -------
        dict
            Summary statistics: n_pq, max_shed_total, max_shed_fraction, etc.
        """
        Pd_pq = self.scenario.Pd_base[self.pq_bus_indices]
        return {
            "n_pq": self.n_pq,
            "max_shed_total_MW": float(np.sum(self.delta_max)),
            "max_shed_fraction": self.max_shed_fraction,
            "baseline_pq_demand_MW": float(np.sum(Pd_pq)),
            "max_shed_pct_of_pq": 100.0 * float(np.sum(self.delta_max)) / float(np.sum(Pd_pq)),
        }
