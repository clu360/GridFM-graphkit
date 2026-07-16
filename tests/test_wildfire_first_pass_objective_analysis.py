import numpy as np

from experiments.test.wildfire_tests.shared.decision_vector import FirstPassDecisionVector
from experiments.test.wildfire_tests.analysis.objective_analysis import build_line_impact_objective_sweep
from experiments.test.wildfire_tests.shared.wildfire_scenario import WildfireLineGroup, WildfireScenario


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


def test_line_impact_sweep_changes_only_frozen_consequence_term():
    decision_vector = FirstPassDecisionVector(DummyScenario(), [0], [1])
    wildfire = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("g", [0])],
        hazard_by_line={0: 1.0},
        impact_by_line={0: 0.5},
    )
    state = {
        "loading_ratio": np.array([2.0]),
        "max_loading_ratio": 2.0,
        "max_voltage": 1.1,
        "min_voltage": 0.9,
        "num_nan": 0,
        "num_inf": 0,
    }
    frame = build_line_impact_objective_sweep(
        decision_vector.u_base,
        decision_vector,
        state,
        wildfire,
        base_line_impact=np.array([0.5]),
        line_id=0,
        impact_values=[0.0, 0.5, 1.0],
        lambda_R=1.0,
        lambda_L=1.0,
        risk_normalizer=4.0,
        load_shedding_normalizer=3.0,
    )

    assert np.allclose(frame["load_shedding"], 0.0)
    assert np.allclose(frame["wildfire_group_risk"], [0.0, 2.0, 4.0])
    assert np.allclose(frame["objective_total"], [0.0, 0.5, 1.0])


def test_decision_metadata_labels_are_available_for_sweeps():
    decision_vector = FirstPassDecisionVector(DummyScenario(), [0], [1])
    frame = decision_vector.metadata_frame(decision_vector.u_base)
    assert frame["decision_type"].tolist() == ["delta_pg", "alpha"]
    assert frame["decision_index"].tolist() == [0, 1]
    assert np.allclose(frame["lower_bound"], [-5.0, 0.0])
    assert np.allclose(frame["upper_bound"], [5.0, 1.0])
