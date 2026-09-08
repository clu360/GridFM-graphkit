"""Shared contracts for the Stage J RQ1 frozen-M0 evaluator benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[5]
M0_SHA256 = "F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831"
SUCCESS = {"LOCALLY_SOLVED", "OPTIMAL"}

FAMILY_SPECS = {
    "dc": {"label": "Guided-DC", "package": "frozen", "method": "Guided-DC", "stub": "gdc"},
    "m0": {
        "label": "Guided-GridSFM (released v1.1)",
        "package": "frozen",
        "method": "Guided-GridSFM",
        "stub": "gsfm",
    },
    "m1": {
        "label": "Guided-GridSFM (FullTop-1000)",
        "package": "m1",
        "method": "Guided-GridSFM",
        "stub": "gsfm",
    },
    "m2": {
        "label": "Guided-GridSFM (FullTop-1500)",
        "package": "m2",
        "method": "Guided-GridSFM",
        "stub": "gsfm",
    },
    "m3": {
        "label": "Guided-GridSFM (FullTop+N-1)",
        "package": "m3",
        "method": "Guided-GridSFM",
        "stub": "gsfm",
    },
}


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def canonical_float(value: float) -> str:
    """Return an exact, platform-stable representation of a binary float."""

    return float(value).hex()


def canonical_decision_components(
    *,
    branch_ids: Iterable[int],
    offline_branch_ids: Iterable[int],
    load_ids: Iterable[int],
    alpha_effective: Mapping[int, float],
    pd_by_load: Mapping[int, float],
    qd_by_load: Mapping[int, float],
) -> dict[str, object]:
    """Build the canonical z, alpha-effective, Pd, and Qd identity payload."""

    offline = {int(value) for value in offline_branch_ids}
    branches = sorted(int(value) for value in branch_ids)
    loads = sorted(int(value) for value in load_ids)
    return {
        "z": [[branch_id, 0 if branch_id in offline else 1] for branch_id in branches],
        "alpha_effective": [
            [load_id, canonical_float(alpha_effective[load_id])] for load_id in loads
        ],
        "pd": [[load_id, canonical_float(pd_by_load[load_id])] for load_id in loads],
        "qd": [[load_id, canonical_float(qd_by_load[load_id])] for load_id in loads],
    }


def decision_hashes(components: Mapping[str, object]) -> dict[str, str]:
    hashes = {f"{name}_sha256": sha256_value(components[name]) for name in ("z", "alpha_effective", "pd", "qd")}
    hashes["decision_sha256"] = sha256_value(
        {name: hashes[f"{name}_sha256"] for name in ("z", "alpha_effective", "pd", "qd")}
    )
    return hashes


def line_ids(value: object) -> list[int]:
    return sorted(
        int(part)
        for part in str(value or "").replace(",", ";").split(";")
        if part.strip()
    )


def canonical_selected_alpha(value: object) -> str:
    parsed = value if isinstance(value, Mapping) else json.loads(str(value or "{}"))
    normalized = {str(int(key)): float(item) for key, item in parsed.items()}
    return canonical_json(normalized)


def settings(root: str | Path) -> list[Path]:
    paths = sorted(
        path for path in Path(root).glob("s*_l*") if (path / "finalists.json").is_file()
    )
    if len(paths) != 15:
        raise RuntimeError(f"expected 15 settings under {root}, observed {len(paths)}")
    return paths


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    config = read_json(resolve_repo_path(path))
    config["_config_path"] = str(resolve_repo_path(path))
    return config


def package_roots(config: Mapping[str, Any]) -> dict[str, Path]:
    ft7 = read_json(resolve_repo_path(config["ft7_config"]))
    return {
        "frozen": Path(ft7["frozen_cache_root"]).resolve(),
        "m1": Path(ft7["m1_cache_root"]).resolve(),
        "m2": Path(ft7["models"]["m2"]["cache_root"]).resolve(),
        "m3": Path(ft7["models"]["m3"]["cache_root"]).resolve(),
    }


def finalist_entry(setting: Path, family_id: str) -> dict[str, Any]:
    spec = FAMILY_SPECS[family_id]
    rows = read_json(setting / "finalists.json")["finalists"]
    matches = [row for row in rows if row.get("method") == spec["method"]]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {spec['method']} finalist in {setting}, observed {len(matches)}"
        )
    return matches[0]


def alpha_effective_path(setting: Path, family_id: str) -> Path:
    return setting / "refs" / FAMILY_SPECS[family_id]["stub"] / "alpha_effective_full.csv"


def read_alpha_effective(path: str | Path) -> dict[int, float]:
    rows = read_csv(path)
    return {int(row["load_id"]): float(row["alpha"]) for row in rows}

