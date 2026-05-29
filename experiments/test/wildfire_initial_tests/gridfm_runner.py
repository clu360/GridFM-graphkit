from __future__ import annotations

from experiments.test.neural_solver import NeuralSolverWrapper
from experiments.test.pipeline_utils import load_gnn_model, load_gps_model

from .config import FirstPassConfig


def load_gridfm_model(config: FirstPassConfig, context):
    model_type = config.model.model_type.lower()
    if model_type == "gps":
        return load_gps_model(
            context.config_dict,
            repo_root=context.repo_root,
            device=config.model.device,
        )[0]
    if model_type == "gnn":
        return load_gnn_model(
            context.args,
            repo_root=context.repo_root,
            device=config.model.device,
        )
    raise ValueError(f"Unsupported model_type: {config.model.model_type}")


class GridFMRunner:
    def __init__(self, model, model_type: str, scenario, decision_vector, device: str = "cpu"):
        self.model_type = model_type.lower()
        self.solver = NeuralSolverWrapper(
            model,
            self.model_type,
            scenario,
            decision_vector,
            device=device,
        )

    def predict(self, u):
        return self.solver.predict_state(u)

    def predict_with_line_outage(self, u, line_id: int):
        """Run GridFM with one scenario edge removed for counterfactual impact."""
        scenario = self.solver.scenario
        line_id = int(line_id)
        num_edges = int(scenario.edge_index.shape[1])
        if line_id < 0 or line_id >= num_edges:
            raise ValueError(f"line_id={line_id} is invalid for {num_edges} edges.")

        original_edge_index = scenario.edge_index
        original_g = scenario.G
        original_b = scenario.B
        original_rate_a = scenario.rate_a

        keep_mask = [idx for idx in range(num_edges) if idx != line_id]
        try:
            scenario.edge_index = original_edge_index[:, keep_mask]
            scenario.G = original_g[keep_mask]
            scenario.B = original_b[keep_mask]
            if original_rate_a is not None:
                scenario.rate_a = original_rate_a[keep_mask]
            return self.solver.predict_state(u)
        finally:
            scenario.edge_index = original_edge_index
            scenario.G = original_g
            scenario.B = original_b
            scenario.rate_a = original_rate_a
