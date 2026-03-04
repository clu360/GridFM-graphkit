"""
Canonical scenario data representation for predict-then-optimize pipeline.

This module defines a single in-memory representation for one power grid scenario,
containing all fixed topology, bus data, edge data, positional encoding, and
masking information needed to run model inference and optimization.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ScenarioData:
    """
    Canonical representation of a single power grid scenario.
    
    Contains fixed scenario data (topology, bus types, baseline state) and 
    decision variables (PV-bus Pg to be optimized).
    
    Attributes
    ----------
    scenario_id : str
        Identifier for the scenario
    
    # Fixed bus data
    num_buses : int
        Number of buses in the network
    bus_indices : np.ndarray
        Array of bus indices (shape: num_buses)
    
    # Baseline state (fixed)
    Pd_base : np.ndarray
        Active power demand, shape (num_buses,)
    Qd_base : np.ndarray
        Reactive power demand, shape (num_buses,)
    Pg_base : np.ndarray
        Baseline active generation, shape (num_buses,)
    Qg_base : np.ndarray
        Baseline reactive generation, shape (num_buses,)
    Vm_base : np.ndarray
        Baseline voltage magnitude, shape (num_buses,)
    Va_base : np.ndarray
        Baseline voltage angle, shape (num_buses,)
    
    # Bus type indicators (one-hot, fixed)
    PQ_mask : np.ndarray
        One-hot mask for PQ buses, shape (num_buses,)
    PV_mask : np.ndarray
        One-hot mask for PV buses, shape (num_buses,)
    REF_mask : np.ndarray
        One-hot mask for reference bus, shape (num_buses,)
    
    # Topology (fixed)
    edge_index : torch.Tensor
        Edge list, shape (2, num_edges)
    G : np.ndarray
        Conductance per edge, shape (num_edges,)
    B : np.ndarray
        Susceptance per edge, shape (num_edges,)
    
    # Positional encoding (fixed)
    pe : torch.Tensor
        Positional encoding, shape (num_buses, pe_dim)
    
    # Node feature mask (fixed)
    mask : torch.Tensor
        Mask tensor indicating which features are masked, shape (num_buses, mask_dim)
    
    # Normalizers and metadata (fixed)
    node_normalizer : object
        Normalizer for node features
    edge_normalizer : object
        Normalizer for edge features
    
    # Generator bounds (for optimization constraints)
    Pg_min : np.ndarray
        Min active generation per bus, shape (num_buses,)
    Pg_max : np.ndarray
        Max active generation per bus, shape (num_buses,)
    """
    
    scenario_id: str
    num_buses: int
    bus_indices: np.ndarray
    
    # Baseline state
    Pd_base: np.ndarray
    Qd_base: np.ndarray
    Pg_base: np.ndarray
    Qg_base: np.ndarray
    Vm_base: np.ndarray
    Va_base: np.ndarray
    
    # Bus types
    PQ_mask: np.ndarray
    PV_mask: np.ndarray
    REF_mask: np.ndarray
    
    # Topology
    edge_index: torch.Tensor
    G: np.ndarray
    B: np.ndarray
    
    # Positional encoding and mask
    pe: torch.Tensor
    mask: torch.Tensor
    
    # Metadata
    node_normalizer: object
    edge_normalizer: object
    
    # Generator bounds
    Pg_min: np.ndarray
    Pg_max: np.ndarray
    
    # Optional: edge admittance matrices for postprocessing
    Yf: Optional[object] = None
    Yt: Optional[object] = None
    base_kv: Optional[np.ndarray] = None
    rate_a: Optional[np.ndarray] = None
    sn_mva: float = 100.0
    
    def get_pv_buses(self) -> np.ndarray:
        """
        Get indices of PV buses.
        
        Returns
        -------
        np.ndarray
            Indices of PV buses
        """
        return np.where(self.PV_mask)[0]
    
    def get_pq_buses(self) -> np.ndarray:
        """
        Get indices of PQ buses.
        
        Returns
        -------
        np.ndarray
            Indices of PQ buses
        """
        return np.where(self.PQ_mask)[0]
    
    def get_ref_bus(self) -> Optional[int]:
        """
        Get index of reference bus.
        
        Returns
        -------
        int or None
            Index of reference bus, or None if not found
        """
        ref_indices = np.where(self.REF_mask)[0]
        return ref_indices[0] if len(ref_indices) > 0 else None
    
    def get_baseline_state(self) -> Dict[str, np.ndarray]:
        """
        Get full baseline bus state as a dictionary.
        
        Returns
        -------
        dict
            Dictionary with keys: Pd, Qd, Pg, Qg, Vm, Va
        """
        return {
            "Pd": self.Pd_base.copy(),
            "Qd": self.Qd_base.copy(),
            "Pg": self.Pg_base.copy(),
            "Qg": self.Qg_base.copy(),
            "Vm": self.Vm_base.copy(),
            "Va": self.Va_base.copy(),
        }
    
    def get_baseline_node_features(self) -> np.ndarray:
        """
        Get baseline node features as a stacked array.
        
        Returns
        -------
        np.ndarray
            Shape (num_buses, 6), columns [Pd, Qd, Pg, Qg, Vm, Va]
        """
        return np.column_stack([
            self.Pd_base,
            self.Qd_base,
            self.Pg_base,
            self.Qg_base,
            self.Vm_base,
            self.Va_base,
        ])
    
    def get_bus_types_onehot(self) -> np.ndarray:
        """
        Get bus type indicators as a stacked array.
        
        Returns
        -------
        np.ndarray
            Shape (num_buses, 3), columns [PQ, PV, REF]
        """
        return np.column_stack([self.PQ_mask, self.PV_mask, self.REF_mask])
    
    def get_edge_features(self) -> np.ndarray:
        """
        Get edge features as a stacked array.
        
        Returns
        -------
        np.ndarray
            Shape (num_edges, 2), columns [G, B]
        """
        return np.column_stack([self.G, self.B])
    
    def to_device(self, device: str) -> "ScenarioData":
        """
        Move tensor fields to device.
        
        Parameters
        ----------
        device : str
            Target device (e.g., 'cpu', 'cuda')
        
        Returns
        -------
        ScenarioData
            Self (modified in place)
        """
        self.edge_index = self.edge_index.to(device)
        self.pe = self.pe.to(device)
        self.mask = self.mask.to(device)
        return self


def extract_scenario_from_batch(
    batch,
    node_normalizer,
    edge_normalizer,
    scenario_idx: int = 0,
    scenario_id: str = "default",
) -> ScenarioData:
    """
    Extract a single scenario from a PyTorch Geometric batch.
    
    This function converts a batch object (from the datamodule) into a 
    canonical ScenarioData representation.
    
    Parameters
    ----------
    batch : torch_geometric.data.Batch
        Batch containing node/edge data, mask, pe, etc.
    node_normalizer : Normalizer
        Normalizer for node features
    edge_normalizer : Normalizer
        Normalizer for edge features
    scenario_idx : int, optional
        Index of scenario within batch (default: 0)
    scenario_id : str, optional
        Human-readable scenario identifier (default: "default")
    
    Returns
    -------
    ScenarioData
        Canonical scenario representation
    """
    from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, PQ, PV, REF
    
    # Denormalize node features to get baseline state
    x_denorm = node_normalizer.inverse_transform(batch.x)
    
    Pd_base = x_denorm[:, PD].cpu().numpy()
    Qd_base = x_denorm[:, QD].cpu().numpy()
    Pg_base = x_denorm[:, PG].cpu().numpy()
    Qg_base = x_denorm[:, QG].cpu().numpy()
    Vm_base = x_denorm[:, VM].cpu().numpy()
    Va_base = x_denorm[:, VA].cpu().numpy()
    
    # Extract bus types
    PQ_mask = x_denorm[:, PQ].cpu().numpy() > 0.5
    PV_mask = x_denorm[:, PV].cpu().numpy() > 0.5
    REF_mask = x_denorm[:, REF].cpu().numpy() > 0.5
    
    num_buses = x_denorm.shape[0]
    bus_indices = np.arange(num_buses)
    
    # Denormalize edge features
    edge_attr_denorm = edge_normalizer.inverse_transform(batch.edge_attr)
    G = edge_attr_denorm[:, 0].cpu().numpy()
    B = edge_attr_denorm[:, 1].cpu().numpy()
    
    # Move to CPU for storage
    edge_index_cpu = batch.edge_index.cpu()
    pe_cpu = batch.pe.cpu()
    mask_cpu = batch.mask.cpu()
    
    # Generator bounds: initialize as baseline +/- 20% if not available
    Pg_min = np.maximum(Pg_base * 0.8, 0)
    Pg_max = Pg_base * 1.2
    
    return ScenarioData(
        scenario_id=scenario_id,
        num_buses=num_buses,
        bus_indices=bus_indices,
        Pd_base=Pd_base,
        Qd_base=Qd_base,
        Pg_base=Pg_base,
        Qg_base=Qg_base,
        Vm_base=Vm_base,
        Va_base=Va_base,
        PQ_mask=PQ_mask,
        PV_mask=PV_mask,
        REF_mask=REF_mask,
        edge_index=edge_index_cpu,
        G=G,
        B=B,
        pe=pe_cpu,
        mask=mask_cpu,
        node_normalizer=node_normalizer,
        edge_normalizer=edge_normalizer,
        Pg_min=Pg_min,
        Pg_max=Pg_max,
    )
