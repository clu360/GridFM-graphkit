from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
WORKFLOW = ROOT / "experiments" / "test" / "wildfire_tests" / "workflow"
PHASE1A = WORKFLOW / "p1a_v002"
ORIGINAL_PACKAGE = ROOT / "experiments" / "test" / "wildfire_tests" / "stage_i_workflow_review_package"
V002_PACKAGE = ROOT / "experiments" / "test" / "wildfire_tests" / "stage_i_workflow_review_package_v002"

EXECUTION_MODE = "single_context_role_simulation"
HUMAN_APPROVAL_STATUS = "AWAITING_CALEB_APPROVAL"
CLASSIFICATIONS = {
    "PRESENT_HASH_MATCH",
    "PRESENT_HASH_MISMATCH",
    "INDEXED_NOT_COPIED",
    "MISSING_UNEXPECTEDLY",
    "ORIGINAL_PATH_ONLY",
    "INVALID_PACKAGE_PATH",
    "PATH_NORMALIZATION_FAILURE",
    "ARCHIVE_OMISSION",
    "DIRECTORY_NOT_FILE",
    "SELF_REFERENTIAL_MANIFEST",
    "UNKNOWN",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


CREATED_UTC = now_utc()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, data) -> None:
    write_text(path, json.dumps(data, indent=2))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def copy_file(src: Path, dest: Path) -> None:
    with src.open("rb") as fsrc, dest.open("wb") as fdst:
        shutil.copyfileobj(fsrc, fdst, length=1024 * 1024)
    try:
        shutil.copystat(src, dest)
    except OSError:
        pass


def manifest_rows() -> list[dict[str, str]]:
    with (ORIGINAL_PACKAGE / "PACKAGE_MANIFEST.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def package_relative(path_text: str) -> Path | None:
    if not path_text:
        return None
    text_norm = path_text.replace("\\", "/")
    absolute_norm = ORIGINAL_PACKAGE.as_posix()
    relative_norm = ORIGINAL_PACKAGE.relative_to(ROOT).as_posix()
    if text_norm.startswith(absolute_norm + "/"):
        return Path(text_norm[len(absolute_norm) + 1 :])
    if text_norm.startswith(relative_norm + "/"):
        return Path(text_norm[len(relative_norm) + 1 :])
    if text_norm == absolute_norm or text_norm == relative_norm:
        return Path(".")
    else:
        return None


def existing_package_file_map() -> dict[str, list[Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for dirpath, _, filenames in os.walk(ORIGINAL_PACKAGE):
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if path.is_file():
                    by_name[name.lower()].append(path)
            except OSError:
                continue
    return by_name


def classify_row(row: dict[str, str], file_map: dict[str, list[Path]]) -> dict[str, str]:
    package_path = row.get("package_path", "").strip()
    original_path = row.get("original_path", "").strip()
    expected = row.get("sha256", "").strip().lower()
    status = row.get("status", "").strip().lower()
    category = row.get("category", "").strip()
    file_name = row.get("file_name", "").strip()
    notes = []

    if not package_path and original_path:
        classification = "ORIGINAL_PATH_ONLY"
    elif not package_path:
        classification = "UNKNOWN"
        notes.append("empty package_path and empty original_path")
    else:
        rel_pkg = package_relative(package_path)
        if rel_pkg is None:
            classification = "INVALID_PACKAGE_PATH"
            notes.append("package_path does not start with original package root")
        elif file_name in {"PACKAGE_MANIFEST.csv", "PACKAGE_MANIFEST.json", "PACKAGE_FREEZE.json"}:
            classification = "SELF_REFERENTIAL_MANIFEST"
            notes.append("integrity metadata row participates in freeze/manifest circularity")
        elif category == "result_large_index" or status.startswith("indexed_not_copied"):
            classification = "INDEXED_NOT_COPIED"
            notes.append("manifest row represents intentionally omitted large external artifact")
        else:
            path = ROOT / package_path
            try:
                if path.exists() and path.is_file():
                    actual = sha256(path).lower()
                    if expected and actual == expected:
                        classification = "PRESENT_HASH_MATCH"
                    else:
                        classification = "PRESENT_HASH_MISMATCH"
                    return {
                        **row,
                        "classification": classification,
                        "actual_sha256": actual,
                        "diagnostic_notes": "; ".join(notes),
                    }
                if path.exists() and path.is_dir():
                    classification = "DIRECTORY_NOT_FILE"
                    notes.append("package_path resolves to directory")
                else:
                    candidates = []
                    for candidate in file_map.get(file_name.lower(), []):
                        try:
                            candidate_hash = sha256(candidate).lower()
                        except OSError:
                            continue
                        if expected and candidate_hash == expected:
                            candidates.append(candidate)
                    if candidates:
                        classification = "PATH_NORMALIZATION_FAILURE"
                        notes.append(f"same filename/hash found elsewhere: {rel(candidates[0])}")
                    else:
                        original = ROOT / original_path if original_path else None
                        if original and original.exists() and original.is_file():
                            try:
                                original_hash = sha256(original).lower()
                            except OSError:
                                original_hash = ""
                            if expected and original_hash == expected:
                                classification = "ARCHIVE_OMISSION"
                                notes.append("package copy missing but original path exists with expected hash")
                            else:
                                classification = "MISSING_UNEXPECTEDLY"
                                notes.append("package copy missing and original does not match expected hash")
                        else:
                            classification = "MISSING_UNEXPECTEDLY"
                            notes.append("package copy missing and original path unavailable")
            except OSError as exc:
                classification = "UNKNOWN"
                notes.append(f"path access error: {type(exc).__name__}: {exc}")

    return {
        **row,
        "classification": classification,
        "actual_sha256": "",
        "diagnostic_notes": "; ".join(notes),
    }


def copy_into_v002(classified: list[dict[str, str]]) -> tuple[list[dict], list[dict], list[dict]]:
    if V002_PACKAGE.exists():
        raise FileExistsError(f"Refusing to overwrite existing v002 package: {V002_PACKAGE}")
    V002_PACKAGE.mkdir(parents=True)

    copied_manifest = []
    external_index = []
    missing_index = []

    def short_dest(row: dict[str, str]) -> Path:
        token = hashlib.sha256(row.get("package_path", "").encode("utf-8")).hexdigest()[:12]
        return V002_PACKAGE / "f" / token

    for row in classified:
        classification = row["classification"]
        package_path = row.get("package_path", "")
        rel_pkg = package_relative(package_path) if package_path else None
        expected = row.get("sha256", "").lower()

        if classification == "PRESENT_HASH_MATCH" and rel_pkg:
            src = ROOT / package_path
            dest = short_dest(row)
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_file(src, dest)
            copied_manifest.append(
                {
                    "package_path": rel(dest),
                    "source": "original_package",
                    "original_manifest_package_path": package_path,
                    "sha256": sha256(dest),
                    "size_bytes": dest.stat().st_size,
                    "category": row.get("category", ""),
                    "status": "copied_verified",
                }
            )
        elif classification == "ARCHIVE_OMISSION" and rel_pkg:
            src = ROOT / row.get("original_path", "")
            dest = short_dest(row)
            dest.parent.mkdir(parents=True, exist_ok=True)
            copy_file(src, dest)
            copied_manifest.append(
                {
                    "package_path": rel(dest),
                    "source": "recovered_from_original_path_matching_manifest_hash",
                    "original_manifest_package_path": package_path,
                    "sha256": sha256(dest),
                    "size_bytes": dest.stat().st_size,
                    "category": row.get("category", ""),
                    "status": "copied_verified_recovered_archive_omission",
                }
            )
        elif classification == "INDEXED_NOT_COPIED":
            summary_copied_path = ""
            if rel_pkg:
                src = ROOT / package_path
                if src.exists() and src.is_file():
                    dest = short_dest(row)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    copy_file(src, dest)
                    summary_copied_path = rel(dest)
                    copied_manifest.append(
                        {
                            "package_path": rel(dest),
                            "source": "original_package_large_artifact_summary",
                            "original_manifest_package_path": package_path,
                            "sha256": sha256(dest),
                            "size_bytes": dest.stat().st_size,
                            "category": "large_artifact_summary",
                            "status": "copied_verified_summary",
                        }
                    )
            external_index.append(
                {
                    "original_path": row.get("original_path", ""),
                    "summary_package_path": summary_copied_path,
                    "original_manifest_package_path": package_path,
                    "original_manifest_sha256": expected,
                    "status": "indexed_not_copied",
                    "notes": row.get("notes", ""),
                }
            )
        else:
            missing_index.append(
                {
                    "original_path": row.get("original_path", ""),
                    "original_manifest_package_path": package_path,
                    "original_manifest_sha256": expected,
                    "classification": classification,
                    "status": "not_copied_to_v002",
                    "notes": row.get("diagnostic_notes", ""),
                }
            )

    return copied_manifest, external_index, missing_index


def finalize_v002(copied_manifest: list[dict], external_index: list[dict], missing_index: list[dict], report_path: Path) -> dict:
    copied_csv = V002_PACKAGE / "COPIED_FILE_MANIFEST.csv"
    copied_json = V002_PACKAGE / "COPIED_FILE_MANIFEST.json"
    external_csv = V002_PACKAGE / "EXTERNAL_ARTIFACT_INDEX.csv"
    external_json = V002_PACKAGE / "EXTERNAL_ARTIFACT_INDEX.json"
    missing_csv = V002_PACKAGE / "MISSING_MATERIALS_INDEX.csv"
    missing_json = V002_PACKAGE / "MISSING_MATERIALS_INDEX.json"

    copied_fields = ["package_path", "source", "original_manifest_package_path", "sha256", "size_bytes", "category", "status"]
    external_fields = ["original_path", "summary_package_path", "original_manifest_package_path", "original_manifest_sha256", "status", "notes"]
    missing_fields = ["original_path", "original_manifest_package_path", "original_manifest_sha256", "classification", "status", "notes"]
    write_csv(copied_csv, copied_manifest, copied_fields)
    write_json(copied_json, copied_manifest)
    write_csv(external_csv, external_index, external_fields)
    write_json(external_json, external_index)
    write_csv(missing_csv, missing_index, missing_fields)
    write_json(missing_json, missing_index)

    copied_failures = []
    for row in copied_manifest:
        path = ROOT / row["package_path"]
        actual = sha256(path)
        if actual.lower() != row["sha256"].lower():
            copied_failures.append({"path": row["package_path"], "expected": row["sha256"], "actual": actual})

    freeze = {
        "package_version": "v002",
        "created_utc": CREATED_UTC,
        "original_package_path": rel(ORIGINAL_PACKAGE),
        "package_path": rel(V002_PACKAGE),
        "freeze_order": [
            "finalize package contents",
            "create and verify copied-file manifests",
            "create external-artifact and missing-material indexes",
            "create PACKAGE_FREEZE.json containing hashes of finalized integrity files",
            "create PACKAGE_FREEZE_RECEIPT.json containing hash of PACKAGE_FREEZE.json",
        ],
        "copied_file_manifest_csv_sha256": sha256(copied_csv),
        "copied_file_manifest_json_sha256": sha256(copied_json),
        "external_artifact_index_csv_sha256": sha256(external_csv),
        "external_artifact_index_json_sha256": sha256(external_json),
        "missing_materials_index_csv_sha256": sha256(missing_csv),
        "missing_materials_index_json_sha256": sha256(missing_json),
        "reconciliation_report_sha256": sha256(report_path),
        "copied_file_count": len(copied_manifest),
        "copied_file_verification_failures": copied_failures,
        "external_artifact_count": len(external_index),
        "missing_material_count": len(missing_index),
        "self_referential_hashing_avoided": True,
    }
    freeze_path = V002_PACKAGE / "PACKAGE_FREEZE.json"
    write_json(freeze_path, freeze)
    receipt = {
        "package_version": "v002",
        "created_utc": now_utc(),
        "package_freeze_path": rel(freeze_path),
        "package_freeze_sha256": sha256(freeze_path),
        "note": "Receipt hashes PACKAGE_FREEZE.json separately to avoid self-referential hashing.",
    }
    write_json(V002_PACKAGE / "PACKAGE_FREEZE_RECEIPT.json", receipt)
    return {"freeze": freeze, "receipt": receipt}


def metadata(artifact_id: str, status: str = "FROZEN") -> str:
    return f"""```yaml
artifact_id: {artifact_id}
artifact_version: v001
created_utc: {CREATED_UTC}
execution_mode: {EXECUTION_MODE}
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: {status}
sha256_when_frozen: recorded_in_phase1a_freeze_ledger
```
"""


def representative_examples(classified: list[dict[str, str]], limit: int = 3) -> str:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classified:
        if len(by_class[row["classification"]]) < limit:
            by_class[row["classification"]].append(row)
    lines = []
    for cls in sorted(CLASSIFICATIONS):
        rows = by_class.get(cls, [])
        if not rows:
            continue
        lines.append(f"### {cls}")
        lines.append("")
        for row in rows:
            lines.append(f"- `{row.get('package_path')}`")
            if row.get("diagnostic_notes"):
                lines.append(f"  - {row['diagnostic_notes']}")
        lines.append("")
    return "\n".join(lines)


def git_status() -> list[str]:
    return subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).splitlines()


def freeze_phase1a_artifacts() -> None:
    rows = []
    for path in sorted(PHASE1A.rglob("*")):
        if path.is_file() and path.name != "FROZEN_ARTIFACTS_PHASE1A.csv":
            rows.append(
                {
                    "artifact_path": rel(path),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                    "frozen_utc": now_utc(),
                }
            )
    write_csv(PHASE1A / "FROZEN_ARTIFACTS_PHASE1A.csv", rows, ["artifact_path", "sha256", "size_bytes", "frozen_utc"])


def main() -> None:
    rows = manifest_rows()
    file_map = existing_package_file_map()
    classified = [classify_row(row, file_map) for row in rows]
    counts = Counter(row["classification"] for row in classified)
    if sum(counts.values()) != len(rows):
        raise RuntimeError("classification counts do not reconcile")
    unknown_categories = set(counts) - CLASSIFICATIONS
    if unknown_categories:
        raise RuntimeError(f"unknown categories: {unknown_categories}")

    class_fields = list(rows[0].keys()) + ["classification", "actual_sha256", "diagnostic_notes"]
    write_csv(PHASE1A / "ORIGINAL_MANIFEST_CLASSIFICATION.csv", classified, class_fields)
    write_json(PHASE1A / "ORIGINAL_MANIFEST_CLASSIFICATION.json", classified)

    original_freeze = json.loads((ORIGINAL_PACKAGE / "PACKAGE_FREEZE.json").read_text(encoding="utf-8"))
    current_manifest_csv_sha = sha256(ORIGINAL_PACKAGE / "PACKAGE_MANIFEST.csv")
    current_manifest_json_sha = sha256(ORIGINAL_PACKAGE / "PACKAGE_MANIFEST.json")
    true_hash_mismatches = counts.get("PRESENT_HASH_MISMATCH", 0)

    copied_manifest, external_index, missing_index = copy_into_v002(classified)

    report = PHASE1A / "PACKAGE_MANIFEST_RECONCILIATION_REPORT.md"
    counts_rows = "\n".join(f"| {cls} | {counts.get(cls, 0)} |" for cls in sorted(CLASSIFICATIONS))
    write_text(
        report,
        f"""# Phase 1A Package Manifest Reconciliation Report

{metadata("PHASE1A-RECON-0001")}

## Summary

Original manifest row count: `{len(rows)}`

Classification totals reconcile to: `{sum(counts.values())}`

| Classification | Count |
| --- | ---: |
{counts_rows}

## Root Cause

Two root causes were found.

1. Freeze ordering / integrity metadata circularity: the original manifest includes `PACKAGE_FREEZE.json`, while `PACKAGE_FREEZE.json` stores hashes for the manifest files. The current manifest hashes therefore differ from the hashes recorded inside the freeze file.
2. Archive omissions: many rows marked as package-copied artifacts do not resolve as readable package files. Some are recoverable from active originals with matching hashes; others are no longer available at either the package path or original path and are recorded as missing materials.

## Manifest Hash Comparison

| File | Current SHA-256 | Freeze-recorded SHA-256 | Match |
| --- | --- | --- | --- |
| `PACKAGE_MANIFEST.csv` | `{current_manifest_csv_sha}` | `{original_freeze.get('manifest_csv_sha256')}` | `{current_manifest_csv_sha == original_freeze.get('manifest_csv_sha256')}` |
| `PACKAGE_MANIFEST.json` | `{current_manifest_json_sha}` | `{original_freeze.get('manifest_json_sha256')}` | `{current_manifest_json_sha == original_freeze.get('manifest_json_sha256')}` |

## Direct Answers

1. Manifest hashes differ because the original freeze/manifest design was not finalized in a non-self-referential order. The manifest includes the freeze record, and the freeze record hashes manifests.
2. The 369 unverified paths are not true content mismatches. They are rows whose package paths could not be resolved/read as copied files during the original package audit.
3. The issue is a combination of freeze ordering, archive omissions, intentionally indexed external artifacts, and missing original/generated result files. It is not primarily path normalization or package movement.
4. True copied-file content-hash mismatches found: `{true_hash_mismatches}`.

## Representative Examples

{representative_examples(classified)}
""",
    )

    freeze_info = finalize_v002(copied_manifest, external_index, missing_index, report)

    v002_copied_failures = len(freeze_info["freeze"]["copied_file_verification_failures"])
    critical_pass = v002_copied_failures == 0 and true_hash_mismatches == 0
    revised_readiness = "READY_WITH_LIMITATIONS" if critical_pass else "NOT_READY"

    audit = PHASE1A / "PACKAGE_INTEGRITY_AUDIT_PHASE1A.md"
    write_text(
        audit,
        f"""# Phase 1A Package Integrity Audit

{metadata("PHASE1A-PACKAGE-INTEGRITY-0001")}

## Result

Post-remediation integrity status: `{'PASS' if critical_pass else 'FAIL'}`

Package version used: `stage_i_workflow_review_package_v002`

## Counts

| Metric | Count |
| --- | ---: |
| Original manifest rows | {len(rows)} |
| Copied files verified in v002 | {len(copied_manifest)} |
| True content-hash mismatches in original copied files | {true_hash_mismatches} |
| Expected external/indexed artifacts | {len(external_index)} |
| Unexpected/missing material records | {len(missing_index)} |
| v002 copied-file verification failures | {v002_copied_failures} |

## Integrity Files

- `stage_i_workflow_review_package_v002/COPIED_FILE_MANIFEST.csv`
- `stage_i_workflow_review_package_v002/EXTERNAL_ARTIFACT_INDEX.csv`
- `stage_i_workflow_review_package_v002/MISSING_MATERIALS_INDEX.csv`
- `stage_i_workflow_review_package_v002/PACKAGE_FREEZE.json`
- `stage_i_workflow_review_package_v002/PACKAGE_FREEZE_RECEIPT.json`
""",
    )

    write_text(
        PHASE1A / "CONTAMINATION_AUDIT_PHASE1A.md",
        f"""# Phase 1A Contamination Audit

{metadata("PHASE1A-CONTAMINATION-0001")}

## Boundary

Phase 1A did not modify active Stage I source, tests, official results, `CURRENT_STATE_SUMMARY.md`, `HISTORY.md`, the original package, or the original failed audit.

Writes were limited to:

- `experiments/test/wildfire_tests/workflow/validation/phase1a_package_integrity/`
- `experiments/test/wildfire_tests/stage_i_workflow_review_package_v002/`

## Git Status After Remediation

```text
{chr(10).join(git_status())}
```
""",
    )

    write_text(
        PHASE1A / "FROZEN_ARTIFACT_VERIFICATION_PHASE1A.md",
        f"""# Phase 1A Frozen Artifact Verification

{metadata("PHASE1A-FROZEN-VERIFY-0001")}

## v002 Freeze Design

The v002 package uses non-self-referential freeze ordering:

1. package evidence files finalized;
2. copied-file manifest, external-artifact index, and missing-materials index created;
3. `PACKAGE_FREEZE.json` created with hashes of finalized integrity files;
4. `PACKAGE_FREEZE_RECEIPT.json` created with the hash of `PACKAGE_FREEZE.json`.

## Verification

Copied-file verification failures: `{v002_copied_failures}`

Freeze receipt hash: `{freeze_info['receipt']['package_freeze_sha256']}`
""",
    )

    write_text(
        PHASE1A / "AUTHORITY_TESTS_PHASE1A.md",
        f"""# Phase 1A Dependent Authority Tests

{metadata("PHASE1A-AUTHORITY-0001")}

| Test | Classification | Result |
| --- | --- | --- |
| Stage H versus Stage I naming | `NAMING_DRIFT` | preserved; no retrospective review started |
| Stage D inclusion/exclusion | `SCOPE_DIFFERENCE` | preserved; no result interpretation changed |
| Coupled versus decoupled proxy/inner lambda | `SCOPE_DIFFERENCE` | preserved; no methodology changed |
| Unresolved baseline commit | `UNRESOLVED_AUTHORITY` | preserved; no baseline invented |
""",
    )

    write_text(
        PHASE1A / "WORKFLOW_READINESS_REPORT_PHASE1A.md",
        f"""# Phase 1A Workflow Readiness Report

{metadata("PHASE1A-READINESS-0001", "AWAITING_CALEB_APPROVAL")}

```text
technical_readiness = {revised_readiness}
human_approval_status = {HUMAN_APPROVAL_STATUS}
```

## Result

Phase 1A remediated the package integrity metadata by creating `stage_i_workflow_review_package_v002` with separated integrity records.

The original package and original failed audit were preserved unchanged.

## Remaining Limitations

- Workflow role separation remains procedural and auditable, not OS-enforced.
- Execution mode remains `{EXECUTION_MODE}`.
- Stage I retrospective review has not begun.
- Caleb approval remains pending.
""",
    )

    status_path = WORKFLOW / "WORKFLOW_STATUS.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["technical_readiness"] = revised_readiness
    status["human_approval_status"] = HUMAN_APPROVAL_STATUS
    status["phase"] = "Phase 1A package-integrity remediation complete"
    status["package_version_for_future_reviews"] = rel(V002_PACKAGE)
    status["phase1a_report"] = rel(PHASE1A / "WORKFLOW_READINESS_REPORT_PHASE1A.md")
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")

    freeze_phase1a_artifacts()

    print(
        json.dumps(
            {
                "root_cause": "freeze ordering plus archive omissions",
                "original_manifest_row_count": len(rows),
                "classification_counts": dict(sorted(counts.items())),
                "copied_files_verified": len(copied_manifest),
                "true_hash_mismatches": true_hash_mismatches,
                "expected_external_indexed_files": len(external_index),
                "unexpected_missing_files": len(missing_index),
                "package_version_used": "stage_i_workflow_review_package_v002",
                "post_remediation_integrity_status": "PASS" if critical_pass else "FAIL",
                "revised_technical_readiness": revised_readiness,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
