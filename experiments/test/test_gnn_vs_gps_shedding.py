"""
Compare GNN and GPS behavior on the extended dispatch space with load shedding.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from experiments.test import (
    DispatchOptimizationProblem,
    ExtendedDispatchSpec,
    get_gnn_checkpoint_path,
    get_gps_checkpoint_path,
    NeuralSolverWrapper,
    WildfirePenaltyEvaluator,
    load_gnn_model,
    load_gps_model,
    load_single_test_scenario,
)


def main() -> None:
    print("=" * 80)
    print("LOAD SHEDDING EVALUATION - GNN VS GPS")
    print("=" * 80)

    context = load_single_test_scenario(scenario_idx=0, scenario_id="IEEE-30-test")
    repo_root = context.repo_root
    args = context.args
    config_dict = context.config_dict
    scenario = context.scenario
    decision_spec = ExtendedDispatchSpec(scenario, max_shed_fraction=1.0)
    wildfire_eval = WildfirePenaltyEvaluator(scenario)

    print(
        f"[OK] Scenario: buses={scenario.num_buses}, PV={decision_spec.n_pv}, "
        f"PQ={decision_spec.n_pq}, dim={decision_spec.n_total}"
    )
    print(
        f"[OK] Physical baseline wildfire cost={wildfire_eval.evaluate_baseline()['wildfire_cost']:.6f}"
    )

    gnn_path = get_gnn_checkpoint_path(repo_root)
    model_gnn = load_gnn_model(args, repo_root=repo_root, device="cpu")
    if gnn_path.exists():
        print(f"[OK] Loaded GNN checkpoint: {gnn_path.name}")
    else:
        print(f"[WARN] GNN checkpoint not found: {gnn_path}")
    solver_gnn = NeuralSolverWrapper(model_gnn, "gnn", scenario, decision_spec, device="cpu")
    problem_gnn = DispatchOptimizationProblem(
        scenario,
        decision_spec,
        solver_gnn,
        wildfire_eval,
        lambda_gen=1.0,
        lambda_shed=50.0,
        lambda_wf=10.0,
    )
    obj_gnn, det_gnn = problem_gnn.objective(decision_spec.u_base, return_details=True)

    print("\nGNN baseline under extended dispatch:")
    print(f"  objective={obj_gnn:.6f}")
    print(f"  wildfire_cost={det_gnn['wildfire_cost']:.6f}")
    print(f"  active_risk_branches={det_gnn['n_active_risk_branches']}")
    print(f"  max_loading={det_gnn['max_loading']:.4f}")

    gps_path = get_gps_checkpoint_path(repo_root)
    if not gps_path.exists():
        print("\n[WARN] GPS checkpoint not found; skipping GPS comparison")
        return

    model_gps, _ = load_gps_model(config_dict, repo_root=repo_root, device="cpu")
    print(f"[OK] Loaded GPS checkpoint: {gps_path.name}")
    solver_gps = NeuralSolverWrapper(model_gps, "gps", scenario, decision_spec, device="cpu")
    problem_gps = DispatchOptimizationProblem(
        scenario,
        decision_spec,
        solver_gps,
        wildfire_eval,
        lambda_gen=1.0,
        lambda_shed=50.0,
        lambda_wf=10.0,
    )
    obj_gps, det_gps = problem_gps.objective(decision_spec.u_base, return_details=True)

    print("\nGPS baseline under extended dispatch:")
    print(f"  objective={obj_gps:.6f}")
    print(f"  wildfire_cost={det_gps['wildfire_cost']:.6f}")
    print(f"  active_risk_branches={det_gps['n_active_risk_branches']}")
    print(f"  max_loading={det_gps['max_loading']:.4f}")

    print("\nComparison:")
    print(f"  GNN objective={obj_gnn:.6f}")
    print(f"  GPS objective={obj_gps:.6f}")
    print(f"  objective_gap={obj_gps - obj_gnn:.6f}")


if __name__ == "__main__":
    main()
