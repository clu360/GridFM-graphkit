from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WILDFIRE_TESTS_ROOT = REPO_ROOT / "experiments" / "test" / "wildfire_tests"
CONFIGS_ROOT = WILDFIRE_TESTS_ROOT / "stage_a_first_pass" / "configs"
RESULTS_ROOT = WILDFIRE_TESTS_ROOT / "results"
METHODOLOGY_TESTING_RESULTS_ROOT = WILDFIRE_TESTS_ROOT / "methodology_testing_results"
ARCHIVED_RESULTS_ROOT = METHODOLOGY_TESTING_RESULTS_ROOT


def result_root(*parts: str) -> Path:
    return RESULTS_ROOT.joinpath(*parts)


def legacy_results_root() -> Path:
    return METHODOLOGY_TESTING_RESULTS_ROOT / "old"
