"""Shared contracts for the Stage J FT6 sequential fine-tuning ablation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


M0_SHA256 = "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
M1_SHA256 = "A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD"
GRIDSFM_COMMIT = "1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6"
PRIVATE_SOURCE_SHA256 = {
    "checkpoint.py": "C05AB84B1A221E5B9FD033CDAAA1194E9D2856865BA12A53FDAF79B709779039",
    "finetune_opfdata.py": "7C01551BA89A540E526F170C19BA01830406225AF603B61947F30BC3DD0A4CE2",
    "loss.py": "FE1B13BDC232B08629BDC7F5090308D6341C64BF88D439587BB3EF72A36C268B",
    "opfdata_train.py": "DA8104E40A80FE6E223615658BB4BECE905A7C073CF9B922A636CDE05467B62E",
}
METRIC_KEYS = (
    "loss", "cost_mape", "pg_mae", "qg_mae", "V_mae", "theta_mae",
    "brP_mae", "brQ_mae", "kcl_P_resid", "kcl_Q_resid",
    "thermal_max_loading", "thermal_frac_overload", "feas_acc", "n_graphs",
)
MODEL_ORDER = ("m0", "m1", "m2", "m3")
VARIANT_ORDER = ("fulltop", "n1")


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def configured_indices(spec: Mapping[str, Any]) -> list[int]:
    start = int(spec["index_start"])
    count = int(spec["count"])
    return list(range(start, start + count))


def validate_split_spec(
    spec: Mapping[str, Any], *, variant: str, split: str, start: int, count: int
) -> None:
    observed = (str(spec.get("variant")), str(spec.get("split")), configured_indices(spec))
    expected = (variant, split, list(range(start, start + count)))
    if observed != expected:
        raise ValueError(f"split contract mismatch: observed={observed}, expected={expected}")


def validate_training_config(config: Mapping[str, Any]) -> None:
    required = {
        "artifact_id", "batch_size", "case_name", "checkpoint_dir",
        "checkpoint_name", "device", "epochs", "expected_gridsfm_commit",
        "expected_parent_sha256", "expected_source_sha256", "gridsfm_root",
        "infeas_prob", "learning_rate", "model_id", "model_variant",
        "num_groups", "num_workers", "opfdata_root", "output_dir",
        "parent_checkpoint", "preflight_manifest", "seed", "train",
        "validation", "weight_decay",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"missing FT6 training config keys: {missing}")
    if config["case_name"] != "pglib_opf_case500_goc":
        raise ValueError("FT6 is frozen to pglib_opf_case500_goc")
    if config["expected_parent_sha256"] != M1_SHA256:
        raise ValueError("both FT6 continuation branches must use the frozen M1 SHA")
    if config["expected_gridsfm_commit"] != GRIDSFM_COMMIT:
        raise ValueError("FT6 GridSFM commit differs from the frozen protocol")
    if config["expected_source_sha256"] != PRIVATE_SOURCE_SHA256:
        raise ValueError("FT6 private-source hashes differ from the frozen protocol")
    if int(config["batch_size"]) != 8 or int(config["epochs"]) != 10:
        raise ValueError("FT6 requires batch_size=8 and epochs=10")
    if not math.isclose(float(config["learning_rate"]), 1e-4):
        raise ValueError("FT6 learning rate is frozen at 1e-4")
    if not math.isclose(float(config["weight_decay"]), 1e-4):
        raise ValueError("FT6 weight decay is frozen at 1e-4")
    if not math.isclose(float(config["infeas_prob"]), 0.3):
        raise ValueError("FT6 infeas_prob is frozen at 0.3")
    if int(config["seed"]) != 42 or int(config["num_groups"]) != 1:
        raise ValueError("FT6 requires seed=42 and num_groups=1")
    model_id = str(config["model_id"])
    if model_id == "m2":
        validate_split_spec(config["train"], variant="fulltop", split="train", start=1000, count=500)
        if config["model_variant"] != "fulltop_ft_n1000_then_fulltop_n500":
            raise ValueError("M2 model_variant is not canonical")
    elif model_id == "m3":
        validate_split_spec(config["train"], variant="n1", split="train", start=0, count=500)
        if config["model_variant"] != "fulltop_ft_n1000_then_n1_n500":
            raise ValueError("M3 model_variant is not canonical")
    else:
        raise ValueError("FT6 training model_id must be m2 or m3")
    validation = config["validation"]
    if [item["variant"] for item in validation] != list(VARIANT_ORDER):
        raise ValueError("FT6 validation strata must be ordered FullTop then N-1")
    for item, variant in zip(validation, VARIANT_ORDER, strict=True):
        validate_split_spec(item, variant=variant, split="val", start=0, count=375)


def finite_metrics(metrics: Mapping[str, Any]) -> bool:
    return set(metrics) == set(METRIC_KEYS) and all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in metrics.values()
    )


def metric_comparison(baseline: Mapping[str, Any], updated: Mapping[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for key in METRIC_KEYS:
        old = float(baseline[key])
        new = float(updated[key])
        comparison[key] = {
            "baseline": baseline[key],
            "updated": updated[key],
            "delta": new - old,
            "relative_change": (new - old) / old if old != 0.0 else None,
        }
    return comparison


def _hash_value(digest: Any, value: Any) -> None:
    import torch

    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        digest.update(b"mapping")
        for key in sorted(value, key=lambda item: repr(item)):
            digest.update(repr(key).encode("utf-8"))
            _hash_value(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _hash_value(digest, item)
    else:
        digest.update(repr(value).encode("utf-8"))


def graph_sha256(graph: Any) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, graph.to_dict())
    return digest.hexdigest().upper()


def fingerprint_dataset(dataset: Any, indices: Iterable[int]) -> list[dict[str, Any]]:
    records = []
    for order, index in enumerate(indices):
        records.append({"order": order, "index": int(index), "sha256": graph_sha256(dataset[int(index)])})
    return records
