import numpy as np

from experiments.test.wildfire_tests.shared.decision_vector import FirstPassDecisionVector


class DummyScenario:
    num_buses = 4
    Pd_base = np.array([0.0, 10.0, 20.0, 30.0])
    Qd_base = np.array([0.0, 1.0, 2.0, 3.0])
    Pg_base = np.array([50.0, 0.0, 0.0, 0.0])

    def get_baseline_node_features(self):
        return np.column_stack(
            [
                self.Pd_base,
                self.Qd_base,
                self.Pg_base,
                np.zeros(4),
                np.ones(4),
                np.zeros(4),
            ]
        )


def test_decision_vector_pack_unpack_reversible():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1, 2])
    u = spec.combine_decision_vector(np.array([1.5]), np.array([0.95, 0.99]))
    delta, alpha = spec.split_decision_vector(u)
    assert np.allclose(delta, [1.5])
    assert np.allclose(alpha, [0.95, 0.99])


def test_alpha_one_gives_zero_unserved_demand():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1, 2])
    assert spec.unserved_demand(spec.u_base) == 0.0
    assert spec.load_shedding(spec.u_base) == 0.0


def test_load_shedding_is_demand_weighted_fraction():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1, 2])
    u = spec.combine_decision_vector(np.array([0.0]), np.array([0.95, 0.90]))
    assert np.isclose(spec.equal_bus_load_shedding(u), 0.15)
    assert np.isclose(spec.load_shedding(u), (10.0 / 60.0) * 0.05 + (20.0 / 60.0) * 0.10)
    assert np.isclose(spec.unserved_demand(u), 2.5)


def test_bounds_are_constructed():
    spec = FirstPassDecisionVector(DummyScenario(), [0], [1], delta_pg_bound_mw=5.0)
    assert np.allclose(spec.u_min, [-5.0, 0.0])
    assert np.allclose(spec.u_max, [5.0, 1.0])
    assert spec.check_bounds(np.array([0.0, 0.95]))[0]
    assert not spec.check_bounds(np.array([6.0, 0.95]))[0]
