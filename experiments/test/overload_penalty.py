"""
Overload penalty evaluation module.

Computes branch loading, overload amounts, and total overload penalty
from predicted bus state.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.sparse import csr_matrix
from .scenario_data import ScenarioData


class OverloadPenaltyEvaluator:
    """
    Evaluates line overload penalties from predicted bus state.
    
    Maps predicted voltages to branch loading using network admittance,
    computes per-line overload, and aggregates to total penalty.
    
    Attributes
    ----------
    scenario : ScenarioData
        Canonical scenario data with topology and network parameters
    sn_mva : float
        System base power in MVA (default: 100)
    """
    
    def __init__(self, scenario: ScenarioData, sn_mva: float = 100.0):
        """
        Initialize overload evaluator.
        
        Parameters
        ----------
        scenario : ScenarioData
            Canonical scenario data
        sn_mva : float, optional
            Base power in MVA (default: 100)
        """
        self.scenario = scenario
        self.sn_mva = sn_mva
        
        # Build admittance matrices if not already present
        if self.scenario.Yf is None or self.scenario.Yt is None:
            self._build_admittance_matrices()
    
    def _build_admittance_matrices(self):
        """
        Build from-end and to-end admittance matrices from edge data.
        
        Uses the circuit model: for each line (f, t),
        Yf[b, :] = y_ff * e_f + y_ft * e_t
        Yt[b, :] = y_tf * e_f + y_tt * e_t
        """
        edge_index = self.scenario.edge_index.numpy()
        f = edge_index[0, :].astype(np.int32)
        t = edge_index[1, :].astype(np.int32)
        
        # Extract admittance components from G and B
        # Assuming simple line model: y = G + j*B
        Y = self.scenario.G + 1j * self.scenario.B
        
        # For a simplified model, Yff ~ Y, Yft ~ -Y, Ytf ~ -Y, Ytt ~ Y
        # (exact values depend on line model; this is a basic approximation)
        Yff = Y
        Yft = -Y
        Ytf = -Y
        Ytt = Y
        
        nl = len(f)
        nb = self.scenario.num_buses
        
        # Build Yf matrix: Yf[b, :] = y_ff * e_f + y_ft * e_t
        i = np.hstack([np.arange(nl), np.arange(nl)])
        j = np.hstack([f, t])
        data = np.hstack([Yff, Yft])
        self.scenario.Yf = csr_matrix((data, (i, j)), shape=(nl, nb))
        
        # Build Yt matrix: Yt[b, :] = y_tf * e_f + y_tt * e_t
        data = np.hstack([Ytf, Ytt])
        self.scenario.Yt = csr_matrix((data, (i, j)), shape=(nl, nb))
        
        # Store base voltages and ratings
        if self.scenario.base_kv is None:
            # Default: 1.0 pu for all buses
            self.scenario.base_kv = np.ones(nb)
        
        if self.scenario.rate_a is None:
            # Default: unity loading limit (no thermal limit)
            self.scenario.rate_a = np.ones(len(f))
    
    def compute_branch_currents_pu(
        self,
        Vm_pred: np.ndarray,
        Va_pred: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute branch currents in per-unit.
        
        Parameters
        ----------
        Vm_pred : np.ndarray
            Predicted bus voltage magnitudes, shape (num_buses,)
        Va_pred : np.ndarray
            Predicted bus voltage angles (radians), shape (num_buses,)
        
        Returns
        -------
        tuple
            (If_pu, It_pu): from-end and to-end currents in per-unit
        """
        # Construct complex voltage vector
        V = Vm_pred * np.exp(1j * Va_pred)
        
        # Compute from-end and to-end currents
        If_pu = np.abs(self.scenario.Yf @ V)
        It_pu = np.abs(self.scenario.Yt @ V)
        
        return If_pu, It_pu
    
    def compute_loading(
        self,
        Vm_pred: np.ndarray,
        Va_pred: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-branch loading (normalized by rating).
        
        Parameters
        ----------
        Vm_pred : np.ndarray
            Predicted bus voltage magnitudes, shape (num_buses,)
        Va_pred : np.ndarray
            Predicted bus voltage angles (radians), shape (num_buses,)
        
        Returns
        -------
        np.ndarray
            Per-branch loading: current / rating, shape (num_edges,)
        """
        If_pu, It_pu = self.compute_branch_currents_pu(Vm_pred, Va_pred)
        
        edge_index = self.scenario.edge_index.numpy()
        f = edge_index[0, :].astype(np.int32)
        t = edge_index[1, :].astype(np.int32)
        
        # Get base voltages and ratings
        Vf_base_kV = self.scenario.base_kv[f]
        Vt_base_kV = self.scenario.base_kv[t]
        rate_a = self.scenario.rate_a  # in MVA
        
        # Current limits in per-unit: I_limit = S_limit / (sqrt(3) * V_base)
        limitf = rate_a / (Vf_base_kV * np.sqrt(3))
        limitt = rate_a / (Vt_base_kV * np.sqrt(3))
        
        # Loading: I / I_limit
        loadingf = If_pu / limitf
        loadingt = It_pu / limitt
        
        # Max of from and to side
        loading = np.maximum(loadingf, loadingt)
        
        return loading
    
    def compute_overload_penalty(
        self,
        Vm_pred: np.ndarray,
        Va_pred: np.ndarray,
        penalty_type: str = "quadratic",
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """
        Compute total overload penalty.
        
        Parameters
        ----------
        Vm_pred : np.ndarray
            Predicted bus voltage magnitudes, shape (num_buses,)
        Va_pred : np.ndarray
            Predicted bus voltage angles (radians), shape (num_buses,)
        penalty_type : str, optional
            Type of overload penalty: "quadratic" or "linear" (default: "quadratic")
        
        Returns
        -------
        tuple
            (total_penalty, details) where:
            - total_penalty: scalar overload penalty
            - details: dict with per-line penalties, max overload, etc.
        """
        loading = self.compute_loading(Vm_pred, Va_pred)
        
        # Compute overload: max(0, loading - 1)
        overloads = np.maximum(0, loading - 1.0)
        
        # Penalty
        if penalty_type == "quadratic":
            per_line_penalty = overloads ** 2
        elif penalty_type == "linear":
            per_line_penalty = overloads
        else:
            raise ValueError(f"Unknown penalty type: {penalty_type}")
        
        total_penalty = float(np.sum(per_line_penalty))
        
        details = {
            "loading": loading,
            "overloads": overloads,
            "per_line_penalty": per_line_penalty,
            "n_overloaded_lines": int(np.sum(loading > 1.0)),
            "max_loading": float(np.max(loading)),
            "max_overload": float(np.max(overloads)),
            "mean_loading": float(np.mean(loading)),
        }
        
        return total_penalty, details
    
    def evaluate(
        self,
        Vm_pred: np.ndarray,
        Va_pred: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate overload metrics from predicted state.
        
        Parameters
        ----------
        Vm_pred : np.ndarray
            Predicted bus voltage magnitudes, shape (num_buses,)
        Va_pred : np.ndarray
            Predicted bus voltage angles (radians), shape (num_buses,)
        
        Returns
        -------
        dict
            Overload evaluation results: penalty, n_overloaded, max_loading, etc.
        """
        penalty, details = self.compute_overload_penalty(Vm_pred, Va_pred)
        
        results = {
            "total_penalty": penalty,
            "n_overloaded_lines": details["n_overloaded_lines"],
            "max_loading": details["max_loading"],
            "max_overload": details["max_overload"],
            "mean_loading": details["mean_loading"],
        }
        
        return results
    
    def evaluate_batch(
        self,
        Vm_pred_batch: np.ndarray,
        Va_pred_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate overload for batch of predictions.
        
        Parameters
        ----------
        Vm_pred_batch : np.ndarray
            Batch of voltage magnitudes, shape (n_samples, num_buses)
        Va_pred_batch : np.ndarray
            Batch of voltage angles, shape (n_samples, num_buses)
        
        Returns
        -------
        dict
            Batch results: penalties, n_overloaded, etc., each shape (n_samples,)
        """
        n_samples = len(Vm_pred_batch)
        
        penalties = []
        n_overloaded = []
        max_loading = []
        
        for i in range(n_samples):
            penalty, details = self.compute_overload_penalty(
                Vm_pred_batch[i],
                Va_pred_batch[i],
            )
            penalties.append(penalty)
            n_overloaded.append(details["n_overloaded_lines"])
            max_loading.append(details["max_loading"])
        
        return {
            "penalties": np.array(penalties),
            "n_overloaded": np.array(n_overloaded),
            "max_loading": np.array(max_loading),
        }
    
    def evaluate_baseline(self) -> Dict[str, float]:
        """
        Evaluate overload for baseline dispatch.
        
        Returns
        -------
        dict
            Baseline overload metrics
        """
        baseline = self.scenario.get_baseline_state()
        return self.evaluate(baseline["Vm"], baseline["Va"])
