import numpy as np
import pytest

from experiments.test.wildfire_tests.shared.wildfire_setup import automatic_visualization_summary_fields
from experiments.test.wildfire_tests.shared.wildfire_scenario import (
    WildfireLineGroup,
    WildfireScenario,
    build_automatic_risk_component_scenario,
    build_synthetic_wildfire_scenario,
    connected_components_from_line_ids,
    select_top_fraction_line_ids,
    top_fraction_line_count,
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


def test_manual_connected_scenario_preserves_configured_corridor():
    loading = np.array([0.1, 0.9, 0.2, 0.8])
    scenario = build_synthetic_wildfire_scenario(
        loading,
        selected_line_ids=[2, 0],
        selection_method="manual_connected",
        high_hazard=1.0,
        default_hazard=0.1,
    )

    assert scenario.line_groups[0].line_ids == [2, 0]
    assert scenario.line_groups[0].name == "high_risk_corridor"
    assert scenario.hazard_by_line == {2: 1.0, 0: 1.0}
    assert scenario.default_hazard == 0.1


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


def test_top_fraction_selection_uses_ceil_and_reports_expected_lines():
    scores = np.array([1.0, 10.0, 3.0, 7.0, 2.0, 6.0])

    assert top_fraction_line_count(len(scores), 0.10) == 1
    assert top_fraction_line_count(len(scores), 0.20) == 2
    assert select_top_fraction_line_ids(scores, 0.20) == [1, 3]


def test_connected_component_grouping_allows_single_or_multiple_groups():
    edge_index = np.array(
        [
            [0, 1, 4],
            [1, 2, 5],
        ]
    )

    one_group = connected_components_from_line_ids(edge_index, [0, 1])
    many_groups = connected_components_from_line_ids(edge_index, [0, 2])

    assert len(one_group) == 1
    assert one_group[0]["line_ids"] == [0, 1]
    assert len(many_groups) == 2
    assert [group["line_ids"] for group in many_groups] == [[0], [2]]


def test_automatic_risk_components_builds_audit_metadata():
    edge_index = np.array(
        [
            [0, 1, 4, 5],
            [1, 2, 5, 6],
        ]
    )
    loading = np.array([1.0, 0.1, 0.9, 0.8])
    impact = np.ones(4)

    scenario, line_scores, group_summary, metadata = build_automatic_risk_component_scenario(
        edge_index,
        loading,
        impact,
        top_fraction=0.50,
        candidate_hazard=1.0,
        default_hazard=1.0,
    )

    assert scenario.line_groups[0].name == "G_1"
    assert scenario.line_groups[1].name == "G_2"
    assert metadata["requested_top_fraction"] == 0.50
    assert metadata["num_selected_lines"] == 2
    assert metadata["realized_selected_fraction"] == 0.5
    assert metadata["num_groups"] == 2
    assert set(line_scores.columns) >= {
        "line_id",
        "from_bus",
        "to_bus",
        "p_env",
        "loading_base",
        "impact_base",
        "score",
        "rank",
        "selected_high_risk",
        "group_id",
    }
    assert set(group_summary["group_id"]) == {"G_1", "G_2"}


def test_automatic_visualization_summary_fields_include_group_coverage():
    fields = automatic_visualization_summary_fields(
        {
            "selection_method": "automatic_risk_components",
            "selected_high_risk_line_ids": [1, 2],
            "group_ids": ["G_1"],
            "group_line_ids": {"G_1": [1, 2]},
            "group_bus_ids": {"G_1": [3, 4]},
            "collapsed_to_single_group": True,
        }
    )

    assert fields["selection_method"] == "automatic_risk_components"
    assert fields["selected_high_risk_line_ids"] == [1, 2]
    assert fields["group_ids"] == ["G_1"]
    assert fields["collapsed_to_single_group"] is True
