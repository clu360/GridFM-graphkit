import json
from pathlib import Path

import pandas as pd

from experiments.test.wildfire_initial_tests.run_multi_group_threshold_sensitivity import (
    run_threshold_sensitivity,
    threshold_label,
)


def test_threshold_label_uses_p_format():
    assert threshold_label(0.10) == "threshold_0p10"
    assert threshold_label(0.125) == "threshold_0p125"


def test_threshold_sensitivity_writes_one_row_per_combination(monkeypatch, tmp_path):
    def fake_run_multistart(config_path, num_seed_points, max_seeds, output_root):
        run_dir = Path(output_root) / "fake_run"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_selected_lines": 2,
                    "realized_selected_fraction": 0.2,
                    "num_groups": 1,
                    "largest_group_num_lines": 2,
                    "largest_group_fraction_of_selected_lines": 1.0,
                    "collapsed_to_single_group": True,
                    "baseline_objective": 0.999001,
                    "best_seed_objective": 0.9,
                    "best_start_index": 0,
                    "best_objective": 0.8,
                    "best_grouped_risk": 1.2,
                    "best_load_shedding": 0.03,
                    "optimizer_success": True,
                    "optimizer_message": "ok",
                    "risk_normalizer": 1.5,
                },
                f,
            )
        return run_dir

    monkeypatch.setattr(
        "experiments.test.wildfire_initial_tests.run_multi_group_threshold_sensitivity.run_multistart_optimization",
        fake_run_multistart,
    )

    summary_csv = run_threshold_sensitivity(
        top_fractions=[0.10, 0.125],
        models=["gps", "gnn"],
        tradeoff_cases=["risk", "balanced", "shed"],
        output_root=tmp_path / "multi_group",
        generated_config_root=tmp_path / "generated_configs",
    )

    summary = pd.read_csv(summary_csv)
    assert len(summary) == 12
    assert set(summary["requested_top_fraction"]) == {0.10, 0.125}
    assert set(summary["model_type"]) == {"gps", "gnn"}
    assert set(summary["tradeoff_case"]) == {"risk", "balanced", "shed"}
    assert (summary["status"] == "ok").all()
    assert "realized_selected_fraction" in summary.columns
    assert "best_seed_objective" in summary.columns
    assert "best_start_index" in summary.columns
