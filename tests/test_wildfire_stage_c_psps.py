import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.test.wildfire_initial_tests.stage_c_psps import (
    apply_environmental_case,
    compute_fixed_line_consequence_scores,
    compute_line_risk_with_z,
    demand_weighted_load_shed_from_prediction,
    identify_largest_group,
    psps_line_count,
    select_psps_lines,
)
from experiments.test.wildfire_initial_tests.wildfire_scenario import WildfireLineGroup, WildfireScenario


class DummyScenario:
    num_buses = 3
    Pd_base = np.array([10.0, 20.0, 0.0])
    edge_index = np.array([[0, 1, 0], [1, 2, 2]])


class DummyRunner:
    def predict_with_line_outage(self, _u, line_id):
        if int(line_id) == 0:
            return {"Pd": np.array([5.0, 20.0, 0.0])}
        if int(line_id) == 1:
            return {"Pd": np.array([10.0, 10.0, 0.0])}
        return {"Pd": np.array([10.0, 20.0, 0.0])}


def test_fixed_demand_weighted_consequence_scores():
    impacts, frame = compute_fixed_line_consequence_scores(
        DummyScenario(),
        DummyRunner(),
        np.array([]),
    )

    assert np.isclose(impacts[0], 5.0 / 30.0)
    assert np.isclose(impacts[1], 10.0 / 30.0)
    assert np.isclose(impacts[2], 0.0)
    assert set(frame.columns) >= {"S_D_base", "S_D_outage", "I_l", "service_loss_MW"}


def test_psps_top_fraction_ceil_and_candidate_restriction():
    candidates = [4, 2, 9]
    risk = {4: 1.0, 2: 3.0, 9: 3.0, 0: 99.0}

    assert psps_line_count(3, 0.10) == 1
    assert select_psps_lines(candidates, risk, 0.50) == [2, 9]


def test_z_zero_removes_line_risk_contribution():
    total, by_line = compute_line_risk_with_z(
        np.array([2.0, 3.0]),
        {0: 1.0, 1: 2.0},
        np.array([0.5, 0.25]),
        {0: 0, 1: 1},
        [0, 1],
    )

    assert by_line[0] == 0.0
    assert np.isclose(by_line[1], 2.0 * 9.0 * 0.25)
    assert np.isclose(total, by_line[1])


def test_largest_group_tiebreak_and_seeded_environment_reproducibility():
    group_summary = pd.DataFrame(
        [
            {"group_id": "G_2", "line_ids": "2,3", "num_lines": 2, "baseline_group_risk": 5.0},
            {"group_id": "G_1", "line_ids": "0,1", "num_lines": 2, "baseline_group_risk": 5.0},
            {"group_id": "G_3", "line_ids": "4", "num_lines": 1, "baseline_group_risk": 100.0},
        ]
    )
    wildfire = WildfireScenario(
        name="auto",
        line_groups=[
            WildfireLineGroup("G_1", [0, 1]),
            WildfireLineGroup("G_2", [2, 3]),
            WildfireLineGroup("G_3", [4]),
        ],
        hazard_by_line={i: 1.0 for i in range(5)},
        impact_by_line={},
        default_hazard=1.0,
    )

    largest_id, largest_lines = identify_largest_group(group_summary)
    env1 = apply_environmental_case(wildfire, group_summary, "largest_group_high")
    env2 = apply_environmental_case(wildfire, group_summary, "largest_group_high")

    assert largest_id == "G_1"
    assert largest_lines == [0, 1]
    assert env1.p_env_by_line[0] == 1.0
    assert env1.p_env_by_line[1] == 1.0
    assert env1.group_p_env == env2.group_p_env
    assert all(0.6 <= value <= 0.9 for group, value in env1.group_p_env.items() if group != "G_1")


def test_environment_case_applies_before_psps_ranking():
    wildfire = WildfireScenario(
        name="auto",
        line_groups=[
            WildfireLineGroup("G_1", [0]),
            WildfireLineGroup("G_2", [1]),
        ],
        hazard_by_line={0: 1.0, 1: 1.0},
        impact_by_line={},
        default_hazard=1.0,
    )
    group_summary = pd.DataFrame(
        [
            {"group_id": "G_1", "line_ids": "0", "num_lines": 1, "baseline_group_risk": 0.1},
            {"group_id": "G_2", "line_ids": "1", "num_lines": 1, "baseline_group_risk": 1.0},
        ]
    )
    env = apply_environmental_case(wildfire, group_summary, "largest_group_high")
    loading = np.array([1.0, 1.0])
    impact = np.array([1.0, 1.0])
    risk = {line_id: env.p_env_by_line[line_id] * loading[line_id] ** 2 * impact[line_id] for line_id in [0, 1]}

    assert select_psps_lines([0, 1], risk, 0.5) == [1]


def test_post_psps_demand_weighted_load_shed():
    shed = demand_weighted_load_shed_from_prediction(
        {"Pd": np.array([5.0, 20.0, 0.0])},
        DummyScenario(),
    )

    assert np.isclose(shed, 5.0 / 30.0)


def test_stage_c_runner_summary_rows(monkeypatch, tmp_path):
    from experiments.test.wildfire_initial_tests import run_stage_c_psps_baseline as runner

    def fake_root():
        return tmp_path / "stage_c_psps"

    def fake_context(model_type, grouping_top_fraction):
        return {"model_type": model_type, "grouping_top_fraction": grouping_top_fraction}

    def fake_case(model_context, case_name, output_root, grouping_top_fraction, psps_top_fraction):
        run_dir = Path(output_root) / "fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "model_type": model_context["model_type"],
            "environmental_risk_case": case_name,
            "grouping_top_fraction": grouping_top_fraction,
            "psps_top_fraction": psps_top_fraction,
            "realized_psps_fraction": 0.125,
            "evaluation_mode": "psps_only",
            "status": "ok",
            "error": "",
            "run_dir": str(run_dir),
        }

    monkeypatch.setattr(runner, "_stage_c_root", fake_root)
    monkeypatch.setattr(runner, "_build_model_context", fake_context)
    monkeypatch.setattr(runner, "run_stage_c_case", fake_case)

    summary_csv = runner.run_stage_c_psps_baseline(
        grouping_top_fraction=0.30,
        psps_top_fraction=0.10,
        models=["gps"],
        cases=["auto_env", "largest_group_high"],
    )
    frame = pd.read_csv(summary_csv)
    with open(tmp_path / "stage_c_psps" / "stage_c_psps_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert len(frame) == 2
    assert set(frame["environmental_risk_case"]) == {"auto_env", "largest_group_high"}
    assert set(frame["evaluation_mode"]) == {"psps_only"}
    assert summary["num_successful_runs"] == 2
