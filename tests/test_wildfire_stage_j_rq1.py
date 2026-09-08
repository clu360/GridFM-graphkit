from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import rq1_common
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune import summarize_rq1_frozen_evaluator as summary


def _components(*, qd_delta: float = 0.0):
    return rq1_common.canonical_decision_components(
        branch_ids=[3, 1, 2],
        offline_branch_ids=[2],
        load_ids=[2, 1],
        alpha_effective={1: 1.0, 2: 0.5},
        pd_by_load={1: 3.0, 2: 2.0},
        qd_by_load={1: 0.4, 2: 0.2 + qd_delta},
    )


def test_rq1_canonical_identity_is_order_invariant_and_covers_all_components() -> None:
    first = _components()
    second = rq1_common.canonical_decision_components(
        branch_ids=[2, 3, 1],
        offline_branch_ids=[2],
        load_ids=[1, 2],
        alpha_effective={2: 0.5, 1: 1.0},
        pd_by_load={2: 2.0, 1: 3.0},
        qd_by_load={2: 0.2, 1: 0.4},
    )
    assert first == second
    hashes = rq1_common.decision_hashes(first)
    assert set(hashes) == {
        "z_sha256", "alpha_effective_sha256", "pd_sha256", "qd_sha256",
        "decision_sha256",
    }
    assert hashes == rq1_common.decision_hashes(second)


def test_rq1_canonical_identity_changes_when_reactive_demand_changes() -> None:
    baseline = rq1_common.decision_hashes(_components())
    changed = rq1_common.decision_hashes(_components(qd_delta=1e-9))
    assert baseline["z_sha256"] == changed["z_sha256"]
    assert baseline["alpha_effective_sha256"] == changed["alpha_effective_sha256"]
    assert baseline["pd_sha256"] == changed["pd_sha256"]
    assert baseline["qd_sha256"] != changed["qd_sha256"]
    assert baseline["decision_sha256"] != changed["decision_sha256"]


def test_rq1_geometric_mean_and_bootstrap_are_paired_and_deterministic() -> None:
    values = np.array([2.0, 4.0, 8.0])
    assert np.isclose(summary._gmean(values), 4.0)
    first = summary._bootstrap(values, np.median, 500, 42)
    second = summary._bootstrap(values, np.median, 500, 42)
    assert first == second
    assert first[0] <= np.median(values) <= first[1]


def test_rq1_publication_is_sibling_of_sealed_ft7_package() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/configs"
        / "rq1_frozen_m0_evaluator.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    publication = Path(config["publication_root"])
    assert publication.name == "rq1_frozen_m0_evaluator_study"
    assert "refined_finetune_study" not in publication.parts


def test_rq1_family_contract_uses_only_m0_as_evaluator() -> None:
    assert rq1_common.M0_SHA256 == (
        "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
    )
    assert set(rq1_common.FAMILY_SPECS) == {"dc", "m0", "m1", "m2", "m3"}
