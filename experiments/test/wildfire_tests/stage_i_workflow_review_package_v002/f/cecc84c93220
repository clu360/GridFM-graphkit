import math

import numpy as np
import pandas as pd

from experiments.test.wildfire_tests.stage_h_heuristic_comparison.run_stage_h_heuristic_baseline_comparison import (
    _combined_best_for_summary,
    area_heuristic_groups,
    select_threshold_lines,
    select_top_fraction_lines,
    select_top_k_lines,
)


def test_transmission_heuristic_selects_scores_at_or_above_percentile():
    scores = {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}

    selected, threshold = select_threshold_lines(scores, 75)

    assert math.isclose(threshold, float(np.percentile(list(scores.values()), 75)))
    assert selected == [4]


def test_transmission_heuristic_keeps_ties_at_percentile_cutoff():
    scores = {1: 0.1, 2: 0.4, 3: 0.4, 4: 0.4}

    selected, threshold = select_threshold_lines(scores, 60)

    assert math.isclose(threshold, 0.4)
    assert selected == [2, 3, 4]


def test_area_heuristic_top_fraction_uses_ceiling_count():
    scores = {line_id: float(line_id) for line_id in range(10)}

    selected = select_top_fraction_lines(scores, 0.30)

    assert selected == [9, 8, 7]


def test_transmission_heuristic_top_k_uses_ranked_scores_with_line_id_tiebreak():
    scores = {10: 0.5, 2: 0.5, 7: 0.9, 4: 0.1}

    selected = select_top_k_lines(scores, 3)

    assert selected == [7, 2, 10]


def test_transmission_heuristic_top_k_clamps_to_available_candidates():
    scores = {1: 0.1, 2: 0.3}

    selected = select_top_k_lines(scores, 5)

    assert selected == [2, 1]


def test_area_heuristic_groups_connected_components_and_selects_highest_average():
    edge_index = np.asarray(
        [
            [0, 1, 2, 10, 11, 20],
            [1, 2, 3, 11, 12, 21],
        ]
    )
    scores = {
        0: 0.60,
        1: 0.59,
        2: 0.58,
        3: 0.95,
        4: 0.94,
        5: 0.10,
    }

    groups = area_heuristic_groups(edge_index, scores, 5 / 6)
    selected = groups[groups["ah_selected_group"].astype(bool)].iloc[0]

    assert set(groups["ah_group_line_ids"]) == {"0,1,2", "3,4"}
    assert selected["ah_group_line_ids"] == "3,4"
    assert math.isclose(float(selected["ah_group_average_score"]), 0.945)


def test_stage_h_summary_preserves_stage_d_stage_e_th_and_ah_methods():
    best = pd.DataFrame(
        [
            {"stage_label": "TH top 2", "heuristic_method": "TH", "R_norm": 0.2, "L_shed": 0.3, "PAC_total": 1.0},
            {"stage_label": "AH connected", "heuristic_method": "AH", "R_norm": 0.3, "L_shed": 0.4, "PAC_total": 1.2},
        ]
    )
    reference = pd.DataFrame(
        [
            {"stage_label": "Stage D exhaustive", "R_norm": 0.1, "L_shed": 0.2, "PAC_total": 0.8},
            {"stage_label": "Stage E k2", "R_norm": 0.15, "L_shed": 0.25, "PAC_total": 0.9},
            {"stage_label": "Stage E unconstrained", "R_norm": 0.12, "L_shed": 0.18, "PAC_total": 0.7},
        ]
    )

    combined = _combined_best_for_summary(best, reference)

    assert {
        "Stage D exhaustive",
        "Stage E k2",
        "Stage E unconstrained",
        "TH top 2",
        "AH connected",
    }.issubset(set(combined["display_method"].astype(str)))
