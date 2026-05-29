import numpy as np
import pytest

from experiments.test.wildfire_initial_tests.wildfire_scenario import (
    WildfireLineGroup,
    WildfireScenario,
    build_synthetic_wildfire_scenario,
    validate_connected_line_group,
)


def test_wildfire_scenario_validates_line_ids():
    scenario = WildfireScenario(
        name="test",
        line_groups=[WildfireLineGroup("bad", [0, 4])],
        hazard_by_line={0: 5.0},
        impact_by_line={0: 1.0},
    )
    with pytest.raises(ValueError):
        scenario.validate(num_lines=4)


def test_build_synthetic_scenario_selects_top_loaded_lines():
    loading = np.array([0.1, 0.9, 0.2, 0.8])
    scenario = build_synthetic_wildfire_scenario(loading, num_high_risk_lines=2)
    assert scenario.line_groups[0].line_ids == [1, 3]
    assert scenario.hazard_by_line[1] == 5.0
    assert scenario.hazard_by_line[3] == 5.0


def test_connected_line_group_validation_rejects_disconnected_edges():
    edge_index = np.array(
        [
            [0, 1, 4],
            [1, 2, 5],
        ]
    )
    validate_connected_line_group(edge_index, [0, 1])
    with pytest.raises(ValueError):
        validate_connected_line_group(edge_index, [0, 2])
