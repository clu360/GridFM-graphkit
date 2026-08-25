"""Verify completed FT6 contracts without loading model weights or OPFData."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ft6_common import METRIC_KEYS, MODEL_ORDER, finite_metrics, validate_training_config
from run_ft6_evaluation import _validate_config as validate_evaluation_config


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "ft6"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    configs = ROOT / "configs"
    for model in ("m2", "m3"):
        validate_training_config(_read(configs / f"ft6_{model}_{'fulltop' if model == 'm2' else 'n1'}_500.json"))
    validate_evaluation_config(_read(configs / "ft6_evaluation_375_375.json"))

    preflight = _read(RESULTS / "ft6_preflight_manifest.json")
    assert preflight["status"] == "FT6_P1_DATA_CONTRACT_PASS"
    assert all(preflight["checks"].values())

    expected_training_status = {
        "m2": "FT6_P2A_M2_TRAINING_PASS",
        "m3": "FT6_P2B_M3_TRAINING_PASS",
    }
    for model, status in expected_training_status.items():
        manifest = _read(RESULTS / model / f"{model}_manifest.json")
        log = _read(RESULTS / model / f"{model}_training_log.json")
        assert manifest["status"] == status
        assert all(manifest["checks"].values())
        assert len(log) == 10
        assert [row["epoch"] for row in log] == list(range(10))
        assert all(row["n_train_iters"] == 63 and row["n_train_skipped"] == 0 for row in log)
        for row in log:
            for variant in ("fulltop", "n1"):
                metrics = {key: row[f"val_{variant}_{key}"] for key in METRIC_KEYS}
                assert finite_metrics(metrics)
                assert metrics["n_graphs"] == 375

    evaluation = _read(RESULTS / "evaluation" / "ft6_evaluation_manifest.json")
    assert evaluation["status"] == "FT6_P3_FOUR_MODEL_EVALUATION_PASS"
    assert all(evaluation["checks"].values())
    assert tuple(evaluation["results"]) == MODEL_ORDER
    for model in MODEL_ORDER:
        for stratum, count in (("fulltop", 375), ("n1", 375), ("combined", 750)):
            metrics = evaluation["results"][model][stratum]["metrics"]
            assert finite_metrics(metrics)
            assert metrics["n_graphs"] == count

    with (RESULTS / "review" / "ft6_test_metrics.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 12
    with (RESULTS / "review" / "ft6_comparison_to_m1.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 3 * 3 * 13
    with (RESULTS / "review" / "ft6_training_trajectories.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 20

    terminal = _read(RESULTS / "ft6_status.json")
    assert terminal["status"] == "FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL"
    assert terminal["m2_checkpoint_sha256"] == evaluation["checkpoint_sha256"]["m2"]
    assert terminal["m3_checkpoint_sha256"] == evaluation["checkpoint_sha256"]["m3"]
    print("FT6 driver and completed-artifact verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
