import numpy as np

from experiments.test.wildfire_initial_tests.wildfire_risk import (
    compute_counterfactual_line_impacts,
    compute_grouped_wildfire_risk,
    compute_line_risk,
)
from experiments.test.wildfire_initial_tests.wildfire_scenario import WildfireLineGroup, WildfireScenario


def test_zero_hazard_gives_zero_risk():
    scenario = WildfireScenario(
        name="zero",
        line_groups=[WildfireLineGroup("g", [0, 1])],
        hazard_by_line={},
        impact_by_line={},
        default_hazard=0.0,
        default_impact=1.0,
    )
    risk = compute_line_risk(np.array([1.0, 2.0]), scenario)
    assert np.allclose(risk, 0.0)


def test_risk_monotonicity_with_loading():
    scenario = WildfireScenario(
        name="mono",
        line_groups=[WildfireLineGroup("g", [0, 1])],
        hazard_by_line={},
        impact_by_line={},
        default_hazard=1.0,
        default_impact=1.0,
    )
    low, *_ = compute_grouped_wildfire_risk(np.array([0.5, 0.5]), scenario)
    high, *_ = compute_grouped_wildfire_risk(np.array([1.0, 1.0]), scenario)
    assert high > low


class DummyCounterfactualScenario:
    num_buses = 3
    Pd_base = np.array([10.0, 20.0, 0.0])
    edge_index = np.array([[0, 1], [1, 2]])


class DummyCounterfactualRunner:
    def predict_with_line_outage(self, u, line_id):
        if int(line_id) == 0:
            return {"Pd": np.array([5.0, 20.0, 0.0])}
        return {"Pd": np.array([10.0, 20.0, 0.0])}


def test_counterfactual_impact_is_relative_equal_weight_service_loss():
    scenario = WildfireScenario(
        name="impact",
        line_groups=[WildfireLineGroup("g", [0, 1])],
        hazard_by_line={0: 1.0, 1: 1.0},
        impact_by_line={},
        default_impact=1.0,
    )
    impacts = compute_counterfactual_line_impacts(
        np.array([]),
        DummyCounterfactualScenario(),
        DummyCounterfactualRunner(),
        scenario,
        {"Pd": np.array([10.0, 20.0, 0.0])},
    )
    assert np.isclose(impacts[0], 0.5 / 3.0)
    assert np.isclose(impacts[1], 0.0)
