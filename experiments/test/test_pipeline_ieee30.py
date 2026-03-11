"""
Pipeline smoke test using the real IEEE-30 test data path.
"""

import sys
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

from experiments.test import (
    DispatchOptimizationProblem,
    NeuralSolverWrapper,
    PVDispatchDecisionSpec,
    PipelineValidationHarness,
    WildfirePenaltyEvaluator,
    load_gnn_model,
    load_single_test_scenario,
)


def main() -> None:
    print("=" * 80)
    print("PIPELINE TEST SUITE - IEEE-30 REAL DATA")
    print("=" * 80)

    print("\n[1/6] Loading test data...")
    context = load_single_test_scenario(scenario_idx=0, scenario_id="IEEE-30-test")
    repo_root = context.repo_root
    args = context.args
    batch = context.batch
    scenario = context.scenario
    print(f"[OK] Loaded batch: nodes={batch.num_nodes}, edges={batch.num_edges}")

    print("\n[2/6] Extracting a single IEEE-30 graph from the batch...")
    print(
        f"[OK] Scenario extracted: buses={scenario.num_buses}, "
        f"PQ={int(scenario.PQ_mask.sum())}, PV={int(scenario.PV_mask.sum())}, "
        f"REF={int(scenario.REF_mask.sum())}"
    )

    print("\n[3/6] Building decision specification...")
    decision_spec = PVDispatchDecisionSpec(scenario)
    print(f"[OK] PV decision dimension={decision_spec.n_pv}")

    print("\n[4/6] Creating solver and baseline prediction...")
    model = load_gnn_model(args, repo_root=repo_root, device="cpu")
    solver = NeuralSolverWrapper(model, "gnn", scenario, decision_spec, device="cpu")
    pred = solver.predict_state(decision_spec.u_base)
    print(
        f"[OK] Prediction generated: Vm mean={np.mean(pred['Vm']):.4f}, "
        f"Va mean={np.mean(pred['Va']):.4f}"
    )

    print("\n[5/6] Evaluating wildfire risk and objective...")
    wildfire_eval = WildfirePenaltyEvaluator(scenario)
    baseline_physical = wildfire_eval.evaluate_baseline()
    problem = DispatchOptimizationProblem(
        scenario=scenario,
        decision_spec=decision_spec,
        solver=solver,
        wildfire_eval=wildfire_eval,
        lambda_gen=1.0,
        lambda_shed=50.0,
        lambda_wf=10.0,
    )
    obj, details = problem.objective(decision_spec.u_base, return_details=True)
    print(
        f"[OK] Physical baseline wildfire cost={baseline_physical['wildfire_cost']:.6f}, "
        f"surrogate wildfire cost={details['wildfire_cost']:.6f}"
    )

    print("\n[6/6] Running validation harness...")
    report = PipelineValidationHarness.full_validation(
        scenario,
        decision_spec,
        solver,
        solver_gps=None,
        overload_eval=wildfire_eval,
    )
    print(f"[OK] Validation all_passed={report['all_passed']}")

    print("\n" + "=" * 80)
    print("[OK] IEEE-30 PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
