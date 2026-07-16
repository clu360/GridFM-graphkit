import numpy as np

from experiments.test.wildfire_tests.shared.config import FirstPassConfig
from experiments.test.wildfire_tests.shared.decision_vector import FirstPassDecisionVector
from experiments.test.wildfire_tests.shared.optimization_problem import FirstPassOptimizationProblem
from experiments.test.wildfire_tests.stage_b_multigroup.run_multistart_optimization import build_grid_seed_candidates
from experiments.test.wildfire_tests.shared.wildfire_scenario import WildfireLineGroup, WildfireScenario


class DummyScenario:
    scenario_id = "dummy"
    num_buses = 3
    Pd_base = np.array([0.0, 10.0, 20.0])
    Qd_base = np.array([0.0, 1.0, 2.0])
    Pg_base = np.array([30.0, 0.0, 0.0])

    def get_baseline_node_features(self):
        return np.column_stack(
            [
                self.Pd_base,
                self.Qd_base,
                self.Pg_base,
                np.zeros(3),
                np.ones(3),
                np.zeros(3),
            ]
        )


class DummyRunner:
    def predict(self, u):
        alpha = float(u[1])
        return {
            "Vm": np.ones(3),
            "Va": np.array([0.0, 1.0 - alpha, 0.0]),
            "Pd": np.array([0.0, 10.0 * alpha, 20.0]),
            "Qd": np.array([0.0, alpha, 2.0]),
            "Pg": np.array([30.0 + float(u[0]), 0.0, 0.0]),
            "Qg": np.zeros(3),
        }

    def predict_with_line_outage(self, u, line_id):
        return self.predict(u)


def test_grid_seed_candidates_include_best_one_variable_point(monkeypatch):
    from experiments.test.wildfire_tests import optimization_problem as op_mod

    def fake_extract_state_quantities(_scenario, _prediction, standard_rate_a_mva=100.0):
        return {
            "Vm": np.ones(3),
            "Va": np.zeros(3),
            "loading_ratio": np.array([1.0, 0.9]),
            "prediction_has_nan": False,
            "prediction_has_inf": False,
            "num_nan": 0,
            "num_inf": 0,
            "max_voltage": 1.0,
            "min_voltage": 1.0,
            "max_loading_ratio": 1.0,
            "state_extraction_passed": True,
        }

    def fake_line_impacts(u, _scenario, _runner, _wildfire, _prediction):
        alpha = float(u[1])
        return np.array([0.5 + 0.5 * alpha, 0.0])

    monkeypatch.setattr(op_mod, "extract_state_quantities", fake_extract_state_quantities)
    monkeypatch.setattr(op_mod, "compute_counterfactual_line_impacts", fake_line_impacts)

    config = FirstPassConfig()
    config.objective.lambda_R = 0.999001
    config.objective.lambda_L = 0.000999
    config.objective.risk_normalizer = 5.0
    config.objective.load_shedding_normalizer = 1.0

    decision_vector = FirstPassDecisionVector(DummyScenario(), [0], [1])
    wildfire = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("g", [0])],
        hazard_by_line={0: 5.0},
        impact_by_line={0: 1.0},
    )
    problem = FirstPassOptimizationProblem(
        DummyScenario(),
        decision_vector,
        DummyRunner(),
        wildfire,
        config,
    )

    starts, seed_frame = build_grid_seed_candidates(problem, num_points=3, max_seeds=2)

    assert len(starts) == 2
    assert seed_frame.iloc[0]["decision_label"] == "alpha_1_bus1"
    assert np.isclose(starts[0][1], 0.0)
    assert seed_frame["objective_total"].iloc[0] < seed_frame["objective_total"].iloc[-1]
