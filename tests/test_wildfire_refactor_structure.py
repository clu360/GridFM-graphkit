from pathlib import Path

from experiments.test.wildfire_tests.shared.lambda_cases import CANONICAL_LAMBDA_CASES
from experiments.test.wildfire_tests.shared.paths import CONFIGS_ROOT, RESULTS_ROOT, WILDFIRE_TESTS_ROOT


def test_canonical_lambda_cases_are_normalized():
    assert CANONICAL_LAMBDA_CASES == {
        "risk_leaning": (0.9, 0.1),
        "balanced": (0.5, 0.5),
        "service_leaning": (0.1, 0.9),
    }
    assert all(abs(lambda_R + lambda_L - 1.0) < 1e-12 for lambda_R, lambda_L in CANONICAL_LAMBDA_CASES.values())


def test_refactor_paths_are_under_wildfire_tests():
    assert WILDFIRE_TESTS_ROOT.name == "wildfire_tests"
    assert CONFIGS_ROOT == WILDFIRE_TESTS_ROOT / "stage_a_first_pass" / "configs"
    assert RESULTS_ROOT == WILDFIRE_TESTS_ROOT / "results"


def test_refactored_import_path_exposes_config():
    from experiments.test.wildfire_tests.shared.config import FirstPassConfig

    assert FirstPassConfig.__name__ == "FirstPassConfig"


def test_parity_compare_schema(tmp_path):
    from experiments.test.wildfire_tests.analysis.parity_compare import compare_run_directories
    from experiments.test.wildfire_tests.shared.reporting import write_json

    old_run = tmp_path / "old"
    new_run = tmp_path / "new"
    old_run.mkdir()
    new_run.mkdir()
    write_json(old_run / "optimization_summary.json", {"objective": 1.0})
    write_json(new_run / "optimization_summary.json", {"objective": 1.0})
    (new_run / "figures").mkdir()
    for relative in [
        "objective_trace.csv",
        "risk_by_group_before_after.csv",
        "risk_by_line_before_after.csv",
        "wildfire_scenario.json",
        "visualization_summary.json",
        "figures/optimization_behavior.png",
        "figures/ieee30_network_changes.png",
    ]:
        path = new_run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")

    result = compare_run_directories(old_run, new_run)

    assert result["all_summary_keys_match"] is True
    assert "artifact_comparisons" in result
