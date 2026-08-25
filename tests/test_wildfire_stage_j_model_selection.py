from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.model_selection import (
    FT_CHECKPOINT_VARIANTS,
    resolve_model_selection,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune.run_ft3_ft4 import (
    _exclusive_run_lock,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune.run_ft5_crossed_warm_starts import (
    FAMILY_ORDER,
    FROZEN_SHA256,
    START_ORDER,
    _validate as validate_crossed_warm_starts,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.finetune.publish_ft3_ft4 import (
    publish,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.run_stage_j_complete import (
    _build_finalists,
    _j8_complete,
    _prepare_topology_pool,
    _reference_complete,
    _validate_ft_result_counts,
)
from experiments.test.wildfire_tests.stage_j_gridsfm_goc500.summarize_stage_j_complete import (
    _validate_pareto_frontiers,
    build_evaluated_pareto_frontiers,
    build_model_comparison,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_model_selection_dc_and_ft_provenance(tmp_path: Path):
    dc = resolve_model_selection("dc", gridsfm_root=tmp_path)
    assert dc.model_variant == "guided_dc"
    assert dc.checkpoint_path is None

    checkpoint = tmp_path / "ft.pt"
    checkpoint.write_bytes(b"fine-tuned-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest().upper()
    ft = resolve_model_selection(
        "ft", gridsfm_root=tmp_path, checkpoint=checkpoint, expected_sha256=digest
    )
    assert ft.model_selection == "ft"
    assert ft.model_variant == "fulltop_ft_n1000"
    assert ft.checkpoint_sha256 == digest

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        resolve_model_selection(
            "ft", gridsfm_root=tmp_path, checkpoint=checkpoint,
            expected_sha256="0" * 64,
        )


def test_known_ft_checkpoint_hash_selects_exact_variant(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "m2.pt"
    checkpoint.write_bytes(b"placeholder")
    digest, variant = next(
        (key, value) for key, value in FT_CHECKPOINT_VARIANTS.items()
        if value == "fulltop_ft_n1000_then_fulltop_n500"
    )
    monkeypatch.setattr(
        "experiments.test.wildfire_tests.stage_j_gridsfm_goc500.model_selection.sha256_file",
        lambda _: digest,
    )
    selected = resolve_model_selection(
        "ft", gridsfm_root=tmp_path, checkpoint=checkpoint, expected_sha256=digest
    )
    assert selected.model_variant == variant


def test_completion_checks_reject_wrong_model_or_checkpoint(tmp_path: Path):
    main = tmp_path / "main"
    summary = main / "guided-gridsfm" / "j8_guided_gridsfm_summary.json"
    _write_json(summary, {
        "status": "PASS", "best_topology": {"topology_id": "1"},
        "model_selection": "ft", "checkpoint_sha256": "ABC",
    })
    assert _j8_complete(
        main, "guided-gridsfm",
        expected_model_selection="ft", expected_checkpoint_sha256="ABC",
    )
    assert not _j8_complete(
        main, "guided-gridsfm",
        expected_model_selection="frozen", expected_checkpoint_sha256="ABC",
    )

    refs = tmp_path / "refs"
    _write_json(refs / "j9_j10_reference_smoke_summary.json", {
        "reference_a_rows": 3, "warm_start_rows": 15, "reference_b_rows": 6,
        "model_selection": "ft", "checkpoint_sha256": "ABC",
    })
    assert _reference_complete(
        refs, 3, expected_model_selection="ft", expected_checkpoint_sha256="ABC"
    )
    assert not _reference_complete(
        refs, 3, expected_model_selection="ft", expected_checkpoint_sha256="DEF"
    )


def test_topology_pool_reuse_is_byte_identical(tmp_path: Path):
    source_root = tmp_path / "frozen"
    source = source_root / "settings" / "s1_l0p0" / "pools" / "guided.csv"
    _write_csv(source, [{"rank": 1, "topology_id": "1"}, {"rank": 2, "topology_id": "2"}])
    destination = tmp_path / "ft" / "s1_l0p0" / "pools" / "guided.csv"
    record = _prepare_topology_pool(
        source_root=source_root, setting_code="s1_l0p0",
        pool_name="guided.csv", destination=destination,
    )
    assert record["ordered_byte_identical"] is True
    assert record["row_count"] == 2
    assert destination.read_bytes() == source.read_bytes()

    destination.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from frozen source"):
        _prepare_topology_pool(
            source_root=source_root, setting_code="s1_l0p0",
            pool_name="guided.csv", destination=destination,
        )


def test_ft_finalists_exclude_preexisting_dc_results(tmp_path: Path):
    main = tmp_path / "main"
    _write_json(main / "guided-gridsfm" / "j8_guided_gridsfm_summary.json", {
        "lambda_r": 0.8,
        "best_topology": {"topology_id": "10;20", "best_found": True},
    })
    _write_csv(main / "th-gridsfm" / "j8_th_gridsfm_topology_summary.csv", [
        {"lambda_r": 0.8, "num_shutoffs": 1, "topology_id": "10", "best_found": True},
        {"lambda_r": 0.8, "num_shutoffs": 2, "topology_id": "10;20", "best_found": True},
    ])
    path = _build_finalists(
        main, tmp_path, model_selection="ft", model_variant="fulltop_ft_n1000"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert [row["method"] for row in payload["finalists"]] == [
        "Guided-GridSFM", "TH-GridSFM-top1", "TH-GridSFM-top2"
    ]
    assert all(row["model_selection"] == "ft" for row in payload["finalists"])


def test_guided_only_finalist_and_count_contract(tmp_path: Path):
    main = tmp_path / "main"
    _write_json(main / "guided-gridsfm" / "j8_guided_gridsfm_summary.json", {
        "lambda_r": 0.5,
        "best_topology": {"topology_id": "10", "best_found": True},
    })
    path = _build_finalists(
        main, tmp_path, model_selection="ft",
        model_variant="fulltop_ft_n1000_then_fulltop_n500", include_th=False,
    )
    assert [row["method"] for row in json.loads(path.read_text())["finalists"]] == [
        "Guided-GridSFM"
    ]

    expected_counts = {
        "topology_objectives_all.csv": 2,
        "candidate_evaluations_all.csv": 6,
        "method_finalists_all.csv": 1,
        "reference_a_all.csv": 1,
        "reference_b_all.csv": 2,
        "warm_start_all.csv": 5,
        "state_fidelity_all.csv": 8,
    }
    for filename, count in expected_counts.items():
        _write_csv(
            tmp_path / "core_results" / filename,
            [{"row": index} for index in range(count)],
        )
    validation = _validate_ft_result_counts(
        tmp_path, n_settings=1, continuous_eval_budget=3,
        topology_budget=2, include_th=False,
    )
    assert validation["status"] == "PASS"
    assert validation["observed"] == expected_counts


def test_model_comparison_uses_frozen_th_and_ft_guided_only():
    frozen = pd.DataFrame({
        "method": [
            "Guided-DC", "Guided-GridSFM", "TH-GridSFM-top1", "TH-GridSFM-top2"
        ],
        "value": [1, 2, 3, 4],
    })
    fine_tuned = pd.DataFrame({
        "method": ["Guided-GridSFM", "TH-GridSFM-top1", "TH-GridSFM-top2"],
        "value": [5, 6, 7],
    })

    comparison = build_model_comparison(fine_tuned, frozen)

    assert comparison["method"].tolist() == [
        "Guided-DC",
        "Guided-GridSFM (frozen)",
        "TH-GridSFM-top1",
        "TH-GridSFM-top2",
        "Guided-GridSFM (fine-tuned)",
    ]
    assert comparison["comparison_variant"].tolist() == [
        "dc", "frozen", "th_frozen", "th_frozen", "ft"
    ]
    assert comparison.loc[comparison["comparison_variant"] == "ft", "value"].tolist() == [5]


def test_evaluated_pareto_frontier_pools_lambda_and_removes_dominated_points():
    rows = []
    for index, (load, risk, lambda_r, status) in enumerate([
        (0.0, 5.0, 0.0, "ok"),
        (1.0, 4.0, 0.2, "ok"),
        (2.0, 4.0, 0.5, "ok"),
        (3.0, 3.0, 0.8, "ok"),
        (1.0, 4.0, 1.0, "ok"),
        (4.0, 2.0, 1.0, "methodology_failure"),
    ], start=1):
        rows.append({
            "scenario_id": "J-S1", "method": "Guided-DC",
            "l_shed_total": load, "r_norm": risk, "lambda_r": lambda_r,
            "topology_rank": index, "candidate_index": index,
            "evaluation_status": status,
        })
    candidates = pd.DataFrame(rows)

    frontier = build_evaluated_pareto_frontiers(candidates)

    assert frontier[["l_shed_total", "r_norm"]].to_records(index=False).tolist() == [
        (0.0, 5.0), (1.0, 4.0), (3.0, 3.0),
    ]
    assert frontier["evaluated_candidate_count"].unique().tolist() == [5]
    assert frontier["pareto_front_size"].unique().tolist() == [3]
    _validate_pareto_frontiers(candidates, frontier)


def test_ft3_launcher_rejects_a_second_writer(tmp_path: Path):
    with _exclusive_run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another FT3/FT4 process"):
            with _exclusive_run_lock(tmp_path):
                pass


def test_ft3_publication_converts_csv_to_parquet(tmp_path: Path):
    source = tmp_path / "source"
    publication = tmp_path / "publication"
    _write_json(source / "FT4_VALIDATION.json", {"status": "PASS"})
    _write_json(source / "FT3_FT4_STATUS.json", {"status": "FT3_FT4_COMPLETE"})
    _write_csv(source / "core_results" / "trace.csv", [{"value": 1}, {"value": 2}])
    config = tmp_path / "config.json"
    _write_json(config, {
        "artifact_id": "test", "checkpoint_sha256": "ABC",
        "expected_checkpoint_sha256": "ABC", "final_root": str(source),
        "model_selection": "ft", "model_variant": "fulltop_ft_n1000",
        "publication_root": str(publication),
    })

    manifest = publish(config)

    assert manifest["status"] == "PASS"
    assert manifest["csv_files_converted_to_parquet"] == 1
    record = next(row for row in manifest["artifacts"] if row["format"] == "parquet")
    frame = __import__("pandas").read_parquet(publication / record["published_relative_path"])
    assert frame["value"].tolist() == [1, 2]


def test_ft5_crossed_warm_start_validation_requires_balanced_five_by_five():
    rows = []
    for setting_index in range(15):
        for family in FAMILY_ORDER:
            instance = f"{setting_index}:{family}"
            for warm_type in START_ORDER:
                row = {
                    "setting_code": f"s{setting_index}",
                    "finalist_family": family,
                    "warm_start_type": warm_type,
                    "reference_a_instance_id": instance,
                    "objective": "100.0",
                    "status": "LOCALLY_SOLVED",
                    "same_reference_a_instance": "True",
                    "solver_start_source": "csv_start_values_over_explicit_generic",
                    "start_payload": "Pg,Qg,V,theta",
                    "warm_start_checkpoint_sha256": "",
                }
                if warm_type == "cold_start":
                    row.update({
                        "solver_start_source": "explicit_generic_V1_theta0_Pg_midpoint_Qg0",
                        "start_vm_min": "1.0",
                        "start_vm_max": "1.0",
                        "start_va_abs_max": "0.0",
                        "start_pg_midpoint_max_abs_error": "0.0",
                        "start_qg_abs_max": "0.0",
                    })
                elif warm_type == "gridsfm_frozen_full_warm":
                    row["warm_start_checkpoint_sha256"] = FROZEN_SHA256
                elif warm_type == "gridsfm_ft_full_warm":
                    row["warm_start_checkpoint_sha256"] = "FT_SHA"
                rows.append(row)

    validation = validate_crossed_warm_starts(
        rows,
        SimpleNamespace(objective_spread_tolerance=1e-3, ft_sha256="FT_SHA"),
    )
    assert validation["status"] == "PASS"
    assert validation["rows"] == 375

    rows.pop()
    failed = validate_crossed_warm_starts(
        rows,
        SimpleNamespace(objective_spread_tolerance=1e-3, ft_sha256="FT_SHA"),
    )
    assert failed["status"] == "FAIL"
    assert failed["checks"]["all_25_combinations_have_15_settings"] is False
