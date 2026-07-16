import numpy as np
import pytest

from experiments.test.wildfire_tests.shared.wildfire_risk import compute_operational_wildfire_exposure
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.gurobi_master import (
    GurobiUnavailableError,
    solve_gurobi_master_next_candidate,
    solve_tiny_gurobi_smoke,
)
from experiments.test.wildfire_tests.stage_e_gurobi_implementation.stage_e_gurobi import (
    compute_proxy_metrics,
    compute_true_metrics,
    no_good_hamming_distance,
    normalize_true_exposure,
    validate_consequence_scores,
)


def test_true_exposure_excludes_impact_and_consequence_scores():
    total, by_line = compute_operational_wildfire_exposure(
        loading_ratio=np.array([2.0, 3.0]),
        p_env_by_line={0: 1.0, 1: 2.0},
        z_by_line={0: 1, 1: 1},
        candidate_line_ids=[0, 1],
    )

    assert np.isclose(by_line[0], 4.0)
    assert np.isclose(by_line[1], 18.0)
    assert np.isclose(total, 22.0)


def test_z_zero_removes_line_from_true_exposure():
    total, by_line = compute_operational_wildfire_exposure(
        loading_ratio=np.array([2.0, 3.0]),
        p_env_by_line={0: 1.0, 1: 2.0},
        z_by_line={0: 0, 1: 1},
        candidate_line_ids=[0, 1],
    )

    assert by_line[0] == 0.0
    assert np.isclose(total, by_line[1])


def test_baseline_normalization_equals_one_and_allows_greater_than_one():
    assert normalize_true_exposure(10.0, 10.0) == 1.0
    assert normalize_true_exposure(12.0, 10.0) == 1.2


def test_true_metrics_preserve_proxy_independent_fields_and_load_bounds():
    metrics = compute_true_metrics(
        loading_ratio=np.array([2.0]),
        p_env_by_line={0: 1.0},
        z_by_line={0: 1},
        candidate_line_ids=[0],
        baseline_R_raw_new=2.0,
        true_L_shed=0.25,
        true_P_AC=0.0,
        lambda_R_true=0.8,
        lambda_L_true=0.2,
    )

    assert set(["true_R_raw", "true_R_norm", "true_L_shed", "true_P_AC", "true_objective"]).issubset(metrics)
    assert np.isclose(metrics["true_R_norm"], 2.0)
    assert np.isclose(metrics["true_objective"], 0.8 * 2.0 + 0.2 * 0.25)


def test_true_metrics_reject_load_shed_outside_unit_interval():
    with pytest.raises(ValueError, match="L_shed"):
        compute_true_metrics(
            loading_ratio=np.array([1.0]),
            p_env_by_line={0: 1.0},
            z_by_line={0: 1},
            candidate_line_ids=[0],
            baseline_R_raw_new=1.0,
            true_L_shed=1.25,
            true_P_AC=0.0,
            lambda_R_true=0.5,
            lambda_L_true=0.5,
        )


def test_consequence_scores_must_be_nonnegative():
    validate_consequence_scores({0: 0.0, 1: 0.2})
    with pytest.raises(ValueError, match="nonnegative"):
        validate_consequence_scores({0: -0.1})


def test_zero_denominators_raise_clear_errors():
    with pytest.raises(ValueError, match="baseline"):
        normalize_true_exposure(1.0, 0.0)
    with pytest.raises(ValueError, match="denominator"):
        compute_proxy_metrics(
            candidate_line_ids=[0],
            p_env_by_line={0: 0.0},
            baseline_loading=np.array([1.0]),
            c_by_line={0: 0.0},
            y_by_line={0: 0},
            lambda_R_master=0.5,
            lambda_L_master=0.5,
        )


def test_proxy_and_true_objectives_are_separate_and_lambdas_explicit():
    proxy = compute_proxy_metrics(
        candidate_line_ids=[0, 1],
        p_env_by_line={0: 1.0, 1: 1.0},
        baseline_loading=np.array([1.0, 2.0]),
        c_by_line={0: 0.1, 1: 0.2},
        y_by_line={0: 1, 1: 0},
        lambda_R_master=0.8,
        lambda_L_master=0.2,
    )
    true = compute_true_metrics(
        loading_ratio=np.array([1.0, 2.0]),
        p_env_by_line={0: 1.0, 1: 1.0},
        z_by_line={0: 0, 1: 1},
        candidate_line_ids=[0, 1],
        baseline_R_raw_new=5.0,
        true_L_shed=0.3,
        true_P_AC=0.0,
        lambda_R_true=0.8,
        lambda_L_true=0.2,
    )

    row = {
        "lambda_R_master": 0.8,
        "lambda_L_master": 0.2,
        "lambda_R_true": 0.8,
        "lambda_L_true": 0.2,
        **proxy,
        **{key: true[key] for key in ["true_R_raw", "true_R_norm", "true_L_shed", "true_P_AC", "true_objective"]},
    }
    assert "proxy_objective" in row
    assert "true_objective" in row
    assert row["lambda_R_master"] == row["lambda_R_true"]


def test_no_good_hamming_distance_detects_duplicate_vectors():
    previous = {1: 0, 2: 1, 3: 0}
    duplicate = {1: 0, 2: 1, 3: 0}
    different = {1: 1, 2: 1, 3: 0}

    assert no_good_hamming_distance([1, 2, 3], previous, duplicate) == 0
    assert no_good_hamming_distance([1, 2, 3], previous, different) == 1


def test_gurobi_master_respects_k_and_no_good_cuts_when_available():
    try:
        solve_tiny_gurobi_smoke()
    except Exception as exc:
        pytest.skip(f"gurobi smoke unavailable: {exc}")

    first = solve_gurobi_master_next_candidate(
        candidate_line_ids=[0, 1, 2],
        p_env_by_line={0: 1.0, 1: 1.0, 2: 1.0},
        baseline_loading=np.array([3.0, 2.0, 1.0]),
        c_by_line={0: 0.01, 1: 0.01, 2: 0.01},
        lambda_R_master=0.8,
        lambda_L_master=0.2,
        max_deenergized_lines=1,
    )
    second = solve_gurobi_master_next_candidate(
        candidate_line_ids=[0, 1, 2],
        p_env_by_line={0: 1.0, 1: 1.0, 2: 1.0},
        baseline_loading=np.array([3.0, 2.0, 1.0]),
        c_by_line={0: 0.01, 1: 0.01, 2: 0.01},
        lambda_R_master=0.8,
        lambda_L_master=0.2,
        max_deenergized_lines=1,
        evaluated_y_vectors=[first["y_by_line"]],
    )

    assert sum(first["y_by_line"].values()) <= 1
    assert sum(second["y_by_line"].values()) <= 1
    assert no_good_hamming_distance([0, 1, 2], first["y_by_line"], second["y_by_line"]) >= 1

