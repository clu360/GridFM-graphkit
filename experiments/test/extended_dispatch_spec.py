"""
Extended dispatch specification combining PV generation and load shedding.

Combines PVDispatchDecisionSpec and LoadSheddingSpec into a unified
decision vector for the optimization problem.
"""

import numpy as np
from typing import Dict, Tuple
from .scenario_data import ScenarioData
from .pv_dispatch import PVDispatchDecisionSpec
from .load_shedding_spec import LoadSheddingSpec


class ExtendedDispatchSpec:
    """
    Extended decision variable specification for PV generation + load shedding.
    
    Maps between the combined optimization variable u_extended and the full
    scenario representation:
    
    u_extended = [Pg_at_PV_buses, Delta_at_PQ_buses]
                  (10 variables)    (48 variables)  = 58 total for IEEE-30
    
    Attributes
    ----------
    pv_spec : PVDispatchDecisionSpec
        PV generation decision specification
    shed_spec : LoadSheddingSpec
        Load shedding decision specification
    n_pv : int
        Number of PV buses (dimension of Pg part)
    n_pq : int
        Number of PQ buses (dimension of shedding part)
    n_total : int
        Total decision dimension = n_pv + n_pq
    """
    
    def __init__(
        self,
        scenario: ScenarioData,
        max_shed_fraction: float = 1.0,
    ):
        """
        Initialize extended dispatch spec.
        
        Parameters
        ----------
        scenario : ScenarioData
            Canonical scenario representation
        max_shed_fraction : float, optional
            Max fraction of Pd that can be shed at each PQ bus (default: 1.0)
        """
        self.scenario = scenario
        
        # Create sub-specs
        self.pv_spec = PVDispatchDecisionSpec(scenario)
        self.shed_spec = LoadSheddingSpec(scenario, max_shed_fraction)
        
        # Combined dimensions
        self.n_pv = self.pv_spec.n_pv
        self.n_pq = self.shed_spec.n_pq
        self.n_total = self.n_pv + self.n_pq
        
        # Combined bounds
        self.u_min = np.hstack([self.pv_spec.u_min, self.shed_spec.delta_min])
        self.u_base = np.hstack([self.pv_spec.u_base, self.shed_spec.delta_base])
        self.u_max = np.hstack([self.pv_spec.u_max, self.shed_spec.delta_max])
    
    def split_decision_vector(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split extended decision vector into Pg and delta components.
        
        Parameters
        ----------
        u : np.ndarray
            Extended decision vector, shape (n_total,)
        
        Returns
        -------
        tuple
            (Pg_pv, delta): Pg at PV buses and shedding amounts
        """
        Pg_pv = u[:self.n_pv]
        delta = u[self.n_pv:]
        return Pg_pv, delta
    
    def combine_decision_vector(self, Pg_pv: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Combine Pg and delta components into extended decision vector.
        
        Parameters
        ----------
        Pg_pv : np.ndarray
            Generation at PV buses, shape (n_pv,)
        delta : np.ndarray
            Load shedding amounts, shape (n_pq,)
        
        Returns
        -------
        np.ndarray
            Extended decision vector, shape (n_total,)
        """
        return np.hstack([Pg_pv, delta])
    
    def u_to_node_features(self, u: np.ndarray) -> np.ndarray:
        """
        Convert extended decision vector u to full node feature array.
        
        Injects u into node features: Pg at PV buses and Pd at PQ buses.
        
        Parameters
        ----------
        u : np.ndarray
            Extended decision vector, shape (n_total,)
        
        Returns
        -------
        np.ndarray
            Node features [Pd, Qd, Pg, Qg, Vm, Va], shape (num_buses, 6)
        """
        from gridfm_graphkit.datasets.globals import PD, PG
        
        # Split decision vector
        Pg_pv, delta = self.split_decision_vector(u)
        
        # Get baseline node features
        node_features = self.scenario.get_baseline_node_features().copy()
        
        # Inject Pg at PV buses
        node_features[self.pv_spec.pv_bus_indices, PG] = Pg_pv
        
        # Inject Pd at PQ buses (reduced by shedding)
        Pd_full = self.shed_spec.delta_to_Pd(delta)
        node_features[:, PD] = Pd_full
        
        return node_features
    
    def check_bounds(self, u: np.ndarray) -> Tuple[bool, str]:
        """
        Check if extended decision vector satisfies bounds.
        
        Parameters
        ----------
        u : np.ndarray
            Extended decision vector, shape (n_total,)
        
        Returns
        -------
        tuple
            (is_feasible: bool, message: str)
        """
        violations_min = np.sum(u < self.u_min)
        violations_max = np.sum(u > self.u_max)
        
        if violations_min > 0 or violations_max > 0:
            msg = f"Decision bound violations: {violations_min} lower, {violations_max} upper"
            return False, msg
        
        return True, "All bounds satisfied"
    
    def get_distance_from_baseline(self, u: np.ndarray) -> float:
        """
        Compute L2 distance from baseline decision.
        
        Parameters
        ----------
        u : np.ndarray
            Extended decision vector, shape (n_total,)
        
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
            Summary statistics for both Pg and shedding
        """
        Pg_pv, delta_base = self.split_decision_vector(self.u_base)
        
        summary = {
            "n_total": self.n_total,
            "n_pv": self.n_pv,
            "n_pq": self.n_pq,
            "Pg_pv_base_mean": float(np.mean(Pg_pv)),
            "Pg_pv_bounds_range": float(np.mean(self.pv_spec.u_max - self.pv_spec.u_min)),
        }
        
        # Add shedding summary
        shed_summary = self.shed_spec.get_summary()
        summary.update({f"shed_{k}": v for k, v in shed_summary.items()})
        
        return summary
