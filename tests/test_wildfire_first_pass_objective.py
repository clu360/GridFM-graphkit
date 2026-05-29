import numpy as np

from experiments.test.wildfire_initial_tests.decision_vector import FirstPassDecisionVector
from experiments.test.wildfire_initial_tests.objective import compute_first_pass_objective_components
from experiments.test.wildfire_initial_tests.wildfire_scenario import WildfireLineGroup, WildfireScenario


class DummyScenario:
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


def test_objective_returns_finite_scalar_for_baseline():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1])
    wildfire = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("g", [0])],
        hazard_by_line={0: 5.0},
        impact_by_line={0: 1.0},
    )
    state = {
        "loading_ratio": np.array([1.0, 0.5]),
        "max_loading_ratio": 1.0,
        "max_voltage": 1.1,
        "min_voltage": 0.9,
        "num_nan": 0,
        "num_inf": 0,
    }
    objective, components = compute_first_pass_objective_components(
        spec.u_base,
        spec,
        state,
        wildfire,
        lambda_R=1.0,
        lambda_L=1.0,
    )
    assert np.isfinite(objective)
    assert components["load_shedding"] == 0.0
    assert components["generator_movement_objective_weight"] == 0.0


def test_objective_uses_demand_weighted_load_shedding_fraction():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1])
    wildfire = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("g", [0])],
        hazard_by_line={0: 5.0},
        impact_by_line={0: 1.0},
    )
    state = {
        "loading_ratio": np.array([1.0, 0.5]),
        "max_loading_ratio": 1.0,
        "max_voltage": 1.1,
        "min_voltage": 0.9,
        "num_nan": 0,
        "num_inf": 0,
    }
    u = spec.combine_decision_vector(np.array([0.0]), np.array([0.90]))
    objective, components = compute_first_pass_objective_components(
        u,
        spec,
        state,
        wildfire,
        lambda_R=0.0,
        lambda_L=1.0,
        normalize_terms=True,
        risk_normalizer=1.0,
        load_shedding_normalizer=1.0,
    )
    expected = (10.0 / 30.0) * 0.10
    assert np.isclose(components["load_shedding"], expected)
    assert np.isclose(components["equal_bus_load_shedding"], 0.10)
    assert np.isclose(components["unserved_demand_mw"], 1.0)
    assert np.isclose(components["normalized_load_shedding"], expected)
    assert np.isclose(objective, expected)
