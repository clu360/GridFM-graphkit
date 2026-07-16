"""
PV-bus active power (Pg) decision variable specification.

Defines how the optimization decision vector u = [Pg_i for i in PV buses]
maps to and from the full scenario representation.
"""

import numpy as np
from typing import Dict, Tuple
from .scenario_data import ScenarioData


class PVDispatchDecisionSpec:
    """
    Decision variable specification for PV-bus active power dispatch.
    
    Maps between the optimization variable u (active generation at PV buses)
    and the full bus state representation.
    
    Attributes
    ----------
    pv_bus_indices : np.ndarray
        Indices of PV buses in the scenario, shape (n_pv,)
    n_pv : int
        Number of PV buses (dimension of decision vector u)
    u_base : np.ndarray
        Baseline dispatch at PV buses, shape (n_pv,)
    u_min : np.ndarray
        Lower bounds on PV dispatch, shape (n_pv,)
    u_max : np.ndarray
        Upper bounds on PV dispatch, shape (n_pv,)
    """
    
    def __init__(self, scenario: ScenarioData):
        """
        Initialize decision spec from scenario data.
        
        Parameters
        ----------
        scenario : ScenarioData
            Canonical scenario representation
        """
        self.scenario = scenario
        
        # Identify PV buses
        self.pv_bus_indices = scenario.get_pv_buses()
        self.n_pv = len(self.pv_bus_indices)
        
        # Extract baseline and bounds at PV buses
        self.u_base = scenario.Pg_base[self.pv_bus_indices].copy()
        self.u_min = scenario.Pg_min[self.pv_bus_indices].copy()
        self.u_max = scenario.Pg_max[self.pv_bus_indices].copy()
    
    def u_to_Pg(self, u: np.ndarray) -> np.ndarray:
        """
        Convert decision vector u to full Pg array.
        
        Parameters
        ----------
        u : np.ndarray
            Active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        np.ndarray
            Full Pg array, shape (num_buses,), with u injected at PV buses
        """
        Pg = self.scenario.Pg_base.copy()
        Pg[self.pv_bus_indices] = u
        return Pg
    
    def Pg_to_u(self, Pg: np.ndarray) -> np.ndarray:
        """
        Extract decision vector u from full Pg array.
        
        Parameters
        ----------
        Pg : np.ndarray
            Full active generation array, shape (num_buses,)
        
        Returns
        -------
        np.ndarray
            Decision vector, shape (n_pv,)
        """
        return Pg[self.pv_bus_indices].copy()
    
    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        """
        Convert decision vector u to full node feature array.
        
        Injects u into the Pg column of a baseline node feature array.
        Demand (Pd, Qd) remains at baseline.
        
        Parameters
        ----------
        u : np.ndarray
            Active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        np.ndarray
            Node features [Pd, Qd, Pg, Qg, Vm, Va], shape (num_buses, 6)
        """
        from gridfm_graphkit.datasets.globals import PG
        
        node_features = self.scenario.get_baseline_node_features().copy()
        node_features[self.pv_bus_indices, PG] = u
        return node_features
    
    def check_bounds(self, u: np.ndarray) -> Tuple[bool, str]:
        """
        Check if decision vector satisfies bounds.
        
        Parameters
        ----------
        u : np.ndarray
            Active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        tuple
            (is_feasible: bool, message: str)
        """
        violations_min = np.sum(u < self.u_min)
        violations_max = np.sum(u > self.u_max)
        
        if violations_min > 0 or violations_max > 0:
            msg = f"Bound violations: {violations_min} lower, {violations_max} upper"
            return False, msg
        
        return True, "Bounds satisfied"
    
    def get_distance_from_baseline(self, u: np.ndarray) -> float:
        """
        Compute L2 distance from baseline dispatch.
        
        Parameters
        ----------
        u : np.ndarray
            Active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        float
            ||u - u_base||_2
        """
        return float(np.linalg.norm(u - self.u_base))
    
    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of decision variable bounds.
        
        Returns
        -------
        dict
            Summary statistics: n_pv, u_base_mean, u_base_min, u_base_max,
            u_min_mean, u_max_mean, etc.
        """
        return {
            "n_pv": self.n_pv,
            "u_base_mean": float(np.mean(self.u_base)),
            "u_base_min": float(np.min(self.u_base)),
            "u_base_max": float(np.max(self.u_base)),
            "u_min_mean": float(np.mean(self.u_min)),
            "u_max_mean": float(np.mean(self.u_max)),
            "bound_range_mean": float(np.mean(self.u_max - self.u_min)),
        }
