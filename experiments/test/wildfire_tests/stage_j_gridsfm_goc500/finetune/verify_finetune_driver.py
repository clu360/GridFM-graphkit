"""Focused FT wrapper checks that do not require loading OPFData."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    driver_path = Path(__file__).with_name("run_finetune.py")
    driver = _load_module("stage_j_ft_driver", driver_path)

    config_path = Path(__file__).parent / "configs" / "ft0_smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    driver._validate_config(config)
    ft1_config = json.loads(
        (Path(__file__).parent / "configs" / "ft1_fulltop_1000.json").read_text(encoding="utf-8")
    )
    driver._validate_config(ft1_config)

    frozen = {0: 1.0, 1: 2.0}
    same_after_json = {"0": 1.0, "1": 2.0}
    changed_after_json = {"0": 1.0, "1": 2.01}
    assert not driver._mapping_values_differ(frozen, same_after_json)
    assert driver._mapping_values_differ(frozen, changed_after_json)

    ft2 = _load_module("stage_j_ft2_driver", Path(__file__).with_name("run_ft2_evaluation.py"))
    ft2_config = json.loads(
        (Path(__file__).parent / "configs" / "ft2_fulltop_n1_test.json").read_text(
            encoding="utf-8"
        )
    )
    ft2._validate_config(ft2_config)
    frozen_metrics = {key: 2.0 for key in ft2.METRIC_KEYS}
    fine_tuned_metrics = {key: 1.0 for key in ft2.METRIC_KEYS}
    comparison = ft2._comparison(frozen_metrics, fine_tuned_metrics)
    assert comparison["loss"]["delta_fine_tuned_minus_frozen"] == -1.0
    assert comparison["loss"]["ratio_fine_tuned_over_frozen"] == 0.5
    print("FT driver verification: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
