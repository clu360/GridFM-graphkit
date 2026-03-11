"""
Smoke test for the predict-then-optimize pipeline using a synthetic single-graph batch.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from experiments.test import (
    DispatchOptimizationProblem,
    NeuralSolverWrapper,
    PVDispatchDecisionSpec,
    PipelineValidationHarness,
    WildfirePenaltyEvaluator,
    extract_scenario_from_batch,
)
from gridfm_graphkit.io.param_handler import NestedNamespace, load_model


class MockBatch:
    """Minimal single-graph object compatible with extract_scenario_from_batch."""

    def __init__(self, num_buses: int = 30, num_edges: int = 40):
        nodes = torch.randn(num_buses, 6)
        bus_types = torch.zeros(num_buses, 3)

        bus_types[:, 0] = 1
        bus_types[1:6, 0] = 0
        bus_types[1:6, 1] = 1
        bus_types[0, 0] = 0
        bus_types[0, 2] = 1

        x_normalized = (nodes - nodes.min(dim=0)[0]) / (
            nodes.max(dim=0)[0] - nodes.min(dim=0)[0] + 1e-6
        )

        self.x = torch.cat([x_normalized, bus_types], dim=1)
        self.edge_attr = torch.randn(num_edges, 2)
        self.edge_index = torch.randint(0, num_buses, (2, num_edges)).long()
        self.pe = torch.randn(num_buses, 20)
        self.mask = torch.ones(num_buses, 6, dtype=torch.bool)
        self.mask[bus_types[:, 2] > 0.5, 4:] = False
        self.mask[bus_types[:, 1] > 0.5, 3] = False
        self.batch = torch.zeros(num_buses, dtype=torch.long)
        self.ptr = torch.tensor([0, num_buses], dtype=torch.long)
        self.y = self.x.clone()
        self.num_nodes = num_buses
        self.num_edges = num_edges


class IdentityNormalizer:
    """Minimal normalizer for synthetic smoke tests."""

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def main() -> None:
    print("=" * 80)
    print("PIPELINE MODULE TEST SUITE")
    print("=" * 80)

    print("\n[1/6] Testing module imports...")
    print("[OK] Imports succeeded")

    print("\n[2/6] Creating synthetic single-scenario batch...")
    batch = MockBatch()
    config_path = repo_root / "tests" / "config" / "gridFMv0.1_dummy.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    args = NestedNamespace(**config_dict)

    node_normalizer = IdentityNormalizer()
    edge_normalizer = IdentityNormalizer()
    print(f"[OK] Mock batch created: {batch.num_nodes} nodes, {batch.num_edges} edges")

    print("\n[3/6] Testing scenario extraction...")
    scenario = extract_scenario_from_batch(
        batch,
        node_normalizer,
        edge_normalizer,
        scenario_idx=0,
        scenario_id="synthetic-test",
    )
    print(
        f"[OK] Scenario extracted: {scenario.num_buses} buses, "
        f"{len(scenario.get_pv_buses())} PV, REF={scenario.get_ref_bus()}"
    )

    print("\n[4/6] Testing PV dispatch specification...")
    decision_spec = PVDispatchDecisionSpec(scenario)
    u = decision_spec.u_base.copy()
    assert np.allclose(u, decision_spec.Pg_to_u(decision_spec.u_to_Pg(u)))
    print(f"[OK] Decision spec works: dimension={decision_spec.n_pv}")

    print("\n[5/6] Testing solver wrapper...")
    model = load_model(args)
    solver = NeuralSolverWrapper(model, "gnn", scenario, decision_spec, device="cpu")
    pred = solver.predict_state(u)
    assert set(pred.keys()) == {"Pd", "Qd", "Pg", "Qg", "Vm", "Va"}
    print(
        f"[OK] Solver works: Vm mean={np.mean(pred['Vm']):.4f}, "
        f"Va mean={np.mean(pred['Va']):.4f}"
    )

    print("\n[6/6] Testing wildfire evaluator, objective, and validation harness...")
    wildfire_eval = WildfirePenaltyEvaluator(scenario)
    problem = DispatchOptimizationProblem(
        scenario=scenario,
        decision_spec=decision_spec,
        solver=solver,
        wildfire_eval=wildfire_eval,
        lambda_gen=1.0,
        lambda_shed=50.0,
        lambda_wf=10.0,
    )
    obj, details = problem.objective(u, return_details=True)
    report = PipelineValidationHarness.full_validation(
        scenario,
        decision_spec,
        solver,
        solver_gps=None,
        wildfire_eval=wildfire_eval,
    )
    print(
        f"[OK] Objective works: total={obj:.4f}, "
        f"gen_dev={details['generator_deviation_cost']:.4f}, "
        f"wildfire={details['wildfire_cost']:.4f}"
    )
    print(f"[OK] Validation summary all_passed={report['all_passed']}")

    print("\n" + "=" * 80)
    print("[OK] SYNTHETIC PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
