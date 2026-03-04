"""
Neural solver wrapper for GridFM pretrained models.

Provides a unified interface to both GNN_v0.1 and GPS_v0.2 models,
handling input preparation (tensors, masking), model inference, and output denormalization.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Literal, Tuple, Optional
from .scenario_data import ScenarioData
from .pv_dispatch import PVDispatchDecisionSpec


class NeuralSolverWrapper:
    """
    Unified wrapper for GridFM pretrained neural solvers.
    
    Manages model loading, input preparation, inference, and denormalization
    for both GNN_TransformerConv (v0.1) and GPSTransformer (v0.2).
    
    Attributes
    ----------
    model : torch.nn.Module
        Pretrained neural network model
    model_type : str
        Type of model: "gnn" or "gps"
    device : str
        Computation device: "cpu" or "cuda"
    scenario : ScenarioData
        Reference scenario for inference
    decision_spec : PVDispatchDecisionSpec
        Decision variable specification
    """
    
    def __init__(
        self,
        model: nn.Module,
        model_type: Literal["gnn", "gps"],
        scenario: ScenarioData,
        decision_spec: PVDispatchDecisionSpec,
        device: str = "cpu",
    ):
        """
        Initialize neural solver wrapper.
        
        Parameters
        ----------
        model : torch.nn.Module
            Pretrained model (GNN_TransformerConv or GPSTransformer)
        model_type : str
            Either "gnn" or "gps"
        scenario : ScenarioData
            Canonical scenario data
        decision_spec : PVDispatchDecisionSpec
            Decision variable specification
        device : str, optional
            Device for computation (default: "cpu")
        """
        self.model = model.to(device)
        self.model.eval()
        self.model_type = model_type.lower()
        self.device = device
        self.scenario = scenario.to_device(device)
        self.decision_spec = decision_spec
        
        if self.model_type not in ["gnn", "gps"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'gnn' or 'gps'.")
    
    def prepare_input_tensors(
        self,
        u: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare input tensors for model inference given decision vector u.
        
        Applies PF masking as specified in the scenario.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector: active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        tuple
            (x, pe, edge_index, edge_attr, batch) tensors ready for model input
        """
        from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA, PQ, PV, REF
        
        # 1. Convert decision vector to full node features
        node_features = self.decision_spec.u_to_node_features(u)
        
        # 2. Stack with bus type indicators
        bus_types = self.scenario.get_bus_types_onehot()
        node_features_extended = np.hstack([node_features, bus_types])  # (num_buses, 9)
        
        # 3. Normalize node features
        node_features_tensor = torch.as_tensor(node_features_extended, dtype=torch.float32)
        x = self.scenario.node_normalizer.transform(node_features_tensor)
        
        # 4. Apply PF mask
        x = x.clone()  # avoid in-place modification issues
        x = x.to(self.device)
        mask_expanded = self.scenario.mask.to(self.device)
        
        # Apply mask: replace masked elements with model's mask_value
        # Note: mask is only for node features (first 6 columns), bus types (last 3 cols) are never masked
        mask_value_expanded = self.model.mask_value.expand(x.shape[0], -1)
        # Apply mask only to node feature columns (columns 0-5), not to bus type columns (6-8)
        x[:, :mask_expanded.shape[1]][mask_expanded] = mask_value_expanded[mask_expanded]
        
        # 5. Prepare other inputs
        edge_index = self.scenario.edge_index.to(self.device)
        edge_attr = self.scenario.get_edge_features()
        edge_attr_tensor = torch.as_tensor(edge_attr, dtype=torch.float32).to(self.device)
        edge_attr = self.scenario.edge_normalizer.transform(edge_attr_tensor)
        
        pe = self.scenario.pe.to(self.device)
        
        # 6. Prepare batch tensor (single graph, so all nodes have batch index 0)
        num_nodes = self.scenario.num_buses
        batch = torch.zeros(num_nodes, dtype=torch.long).to(self.device)
        
        return x, pe, edge_index, edge_attr, batch
    
    def predict_normalized(
        self,
        u: np.ndarray,
        return_all_features: bool = True,
    ) -> torch.Tensor:
        """
        Run model inference on decision vector u, return normalized predictions.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector: active generation at PV buses, shape (n_pv,)
        return_all_features : bool, optional
            If False, return only requested features. (default: True, return all)
        
        Returns
        -------
        torch.Tensor
            Predicted node features (normalized), shape (num_buses, output_dim)
        """
        with torch.no_grad():
            x, pe, edge_index, edge_attr, batch = self.prepare_input_tensors(u)
            
            # Run model
            if self.model_type == "gnn":
                # GNN_TransformerConv doesn't use pe, but still needs it as input
                y_pred = self.model(x, pe, edge_index, edge_attr, batch)
            elif self.model_type == "gps":
                # GPSTransformer uses pe
                y_pred = self.model(x, pe, edge_index, edge_attr, batch)
            
            return y_pred
    
    def predict_state(
        self,
        u: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run model inference and return denormalized predicted state.
        
        This is the primary interface method.
        
        Parameters
        ----------
        u : np.ndarray
            Decision vector: active generation at PV buses, shape (n_pv,)
        
        Returns
        -------
        dict
            Predicted state: {Pd, Qd, Pg, Qg, Vm, Va}, each shape (num_buses,)
        """
        from gridfm_graphkit.datasets.globals import PD, QD, PG, QG, VM, VA
        
        # Get normalized prediction
        y_pred_norm = self.predict_normalized(u)
        
        # Denormalize
        y_pred_denorm = self.scenario.node_normalizer.inverse_transform(y_pred_norm)
        y_pred_denorm = y_pred_denorm.cpu().numpy()
        
        # Extract predicted features
        Pd_pred = y_pred_denorm[:, PD]
        Qd_pred = y_pred_denorm[:, QD]
        Pg_pred = y_pred_denorm[:, PG]
        Qg_pred = y_pred_denorm[:, QG]
        Vm_pred = y_pred_denorm[:, VM]
        Va_pred = y_pred_denorm[:, VA]
        
        return {
            "Pd": Pd_pred,
            "Qd": Qd_pred,
            "Pg": Pg_pred,
            "Qg": Qg_pred,
            "Vm": Vm_pred,
            "Va": Va_pred,
        }
    
    def validate_baseline(self) -> Dict[str, float]:
        """
        Validate that solver reproduces baseline prediction correctly.
        
        Runs inference on baseline dispatch and compares with baseline state.
        
        Returns
        -------
        dict
            Validation metrics: errors per feature
        """
        # Run inference on baseline
        u_base = self.decision_spec.u_base
        pred = self.predict_state(u_base)
        
        # Compute errors
        errors = {}
        baseline = self.scenario.get_baseline_state()
        
        for key in ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]:
            pred_vals = pred[key]
            true_vals = baseline[key]
            rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
            mae = float(np.mean(np.abs(pred_vals - true_vals)))
            errors[f"{key}_rmse"] = rmse
            errors[f"{key}_mae"] = mae
        
        return errors
    
    def predict_batch(
        self,
        u_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Run predictions on multiple decision vectors.
        
        Parameters
        ----------
        u_batch : np.ndarray
            Batch of decision vectors, shape (n_samples, n_pv)
        
        Returns
        -------
        dict
            Batch predictions: {Pd, Qd, Pg, Qg, Vm, Va}, each shape (n_samples, num_buses)
        """
        predictions = {
            "Pd": [],
            "Qd": [],
            "Pg": [],
            "Qg": [],
            "Vm": [],
            "Va": [],
        }
        
        for i in range(len(u_batch)):
            pred = self.predict_state(u_batch[i])
            for key in predictions.keys():
                predictions[key].append(pred[key])
        
        # Stack into arrays
        for key in predictions.keys():
            predictions[key] = np.array(predictions[key])
        
        return predictions
    
    def to_device(self, device: str) -> "NeuralSolverWrapper":
        """
        Move model and scenario to device.
        
        Parameters
        ----------
        device : str
            Target device
        
        Returns
        -------
        NeuralSolverWrapper
            Self
        """
        self.model = self.model.to(device)
        self.scenario = self.scenario.to_device(device)
        self.device = device
        return self
