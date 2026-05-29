import numpy as np

from experiments.test.wildfire_initial_tests.config import FirstPassConfig
from experiments.test.wildfire_initial_tests.decision_vector import FirstPassDecisionVector
from experiments.test.wildfire_initial_tests.optimization_problem import FirstPassOptimizationProblem
from experiments.test.wildfire_initial_tests.wildfire_scenario import WildfireLineGroup, WildfireScenario


class DummyScenario:
    scenario_id = "dummy"
    num_buses = 3
    Pd_base = np.array([0.0, 10.0, 20.0])
    Qd_base = np.array([0.0, 1.0, 2.0])
    Pg_base = np.array([30.0, 0.0, 0.0])
    edge_index = np.array([[0, 1], [1, 2]])

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
        delta = float(u[0])
        alpha = float(u[1])
        return {
            "Vm": np.array([1.0 + 0.001 * delta, 1.0, 1.0]),
            "Va": np.array([0.0, 0.01 * (1.0 - alpha), 0.0]),
            "Pd": np.array([0.0, 10.0 * alpha, 20.0]),
            "Qd": np.array([0.0, alpha, 2.0]),
            "Pg": np.array([30.0 + delta, 0.0, 0.0]),
            "Qg": np.zeros(3),
        }

    def predict_with_line_outage(self, u, line_id):
        prediction = self.predict(u)
        if int(line_id) == 0:
            prediction["Pd"] = prediction["Pd"].copy()
            prediction["Pd"][1] = 0.0
        return prediction


def test_basic_problem_baseline_objective_is_finite(monkeypatch):
    from experiments.test.wildfire_initial_tests import optimization_problem as op_mod

    def fake_extract_state_quantities(_scenario, prediction, standard_rate_a_mva=100.0):
        return {
            "Vm": prediction["Vm"],
            "Va": prediction["Va"],
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

    monkeypatch.setattr(op_mod, "extract_state_quantities", fake_extract_state_quantities)

    spec = FirstPassDecisionVector(DummyScenario(), [0], [1])
    wildfire = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("g", [0])],
        hazard_by_line={0: 5.0},
        impact_by_line={0: 1.0},
    )
    problem = FirstPassOptimizationProblem(
        DummyScenario(),
        spec,
        DummyRunner(),
        wildfire,
        FirstPassConfig(),
    )
    components = problem.evaluate(spec.u_base)
    assert np.isfinite(components["objective_total"])
