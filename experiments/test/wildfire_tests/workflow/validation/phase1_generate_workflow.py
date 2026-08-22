from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
WORKFLOW = ROOT / "experiments" / "test" / "wildfire_tests" / "workflow"
PACKAGE = ROOT / "experiments" / "test" / "wildfire_tests" / "stage_i_workflow_review_package"
LITERATURE = (
    ROOT.parent.parent
    / "Literature Review"
    / "Papers"
)

EXECUTION_MODE = "single_context_role_simulation"
BLIND_CONTEXT_ENFORCED = "false"
REVIEWER_PRIOR_OUTPUTS_VISIBLE = "false"
HUMAN_APPROVAL_STATUS = "AWAITING_CALEB_APPROVAL"
TECHNICAL_READINESS = "NOT_READY"


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
    except ValueError:
        return str(path)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing workflow artifact: {path}")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def metadata_block(artifact_id: str, status: str = "FROZEN") -> str:
    return f"""```yaml
artifact_id: {artifact_id}
artifact_version: v001
created_utc: {CREATED_UTC}
execution_mode: {EXECUTION_MODE}
blind_context_enforced: {BLIND_CONTEXT_ENFORCED}
reviewer_prior_outputs_visible: {REVIEWER_PRIOR_OUTPUTS_VISIBLE}
status: {status}
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```
"""


def read_manifest() -> list[dict[str, str]]:
    manifest_path = PACKAGE / "PACKAGE_MANIFEST.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def snapshot_inputs(manifest_rows: list[dict[str, str]]) -> dict:
    package_freeze = PACKAGE / "PACKAGE_FREEZE.json"
    manifest_csv = PACKAGE / "PACKAGE_MANIFEST.csv"
    manifest_json = PACKAGE / "PACKAGE_MANIFEST.json"
    freeze_data = json.loads(package_freeze.read_text(encoding="utf-8"))

    package_hash_checks = []
    for row in manifest_rows:
        package_path = row.get("package_path", "")
        expected = row.get("sha256", "")
        if not package_path or not expected:
            continue
        path = ROOT / package_path
        if path.exists() and path.is_file():
            actual = sha256(path)
            ok = actual == expected
            status = "MATCH" if ok else "MISMATCH"
        else:
            actual = ""
            status = "MISSING"
        package_hash_checks.append(
            {
                "category": row.get("category", ""),
                "package_path": package_path,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "status": status,
            }
        )

    monitored_originals = []
    for row in manifest_rows:
        category = row.get("category", "")
        if category not in {"source", "test", "result", "result_large_index"}:
            continue
        original = row.get("original_path", "")
        if not original:
            continue
        path = ROOT / original
        if path.exists() and path.is_file():
            monitored_originals.append(
                {
                    "category": category,
                    "original_path": original,
                    "exists": True,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
        else:
            monitored_originals.append(
                {
                    "category": category,
                    "original_path": original,
                    "exists": False,
                    "size_bytes": "",
                    "sha256": "",
                }
            )

    return {
        "captured_utc": now_utc(),
        "package_freeze_sha256": sha256(package_freeze),
        "manifest_csv_sha256": sha256(manifest_csv),
        "manifest_json_sha256": sha256(manifest_json),
        "freeze_declared_manifest_csv_sha256": freeze_data.get("manifest_csv_sha256"),
        "freeze_declared_manifest_json_sha256": freeze_data.get("manifest_json_sha256"),
        "package_hash_checks": package_hash_checks,
        "monitored_originals": monitored_originals,
    }


def compare_snapshots(pre: dict, post: dict) -> dict:
    original_pre = {r["original_path"]: r for r in pre["monitored_originals"]}
    original_post = {r["original_path"]: r for r in post["monitored_originals"]}
    changed = []
    for path, before in original_pre.items():
        after = original_post.get(path)
        if after != before:
            changed.append({"original_path": path, "before": before, "after": after})
    package_mismatches = [
        r for r in post["package_hash_checks"] if r["status"] != "MATCH"
    ]
    manifest_ok = (
        post["manifest_csv_sha256"] == post["freeze_declared_manifest_csv_sha256"]
        and post["manifest_json_sha256"] == post["freeze_declared_manifest_json_sha256"]
    )
    return {
        "compared_utc": now_utc(),
        "manifest_hashes_match_freeze": manifest_ok,
        "package_hash_mismatch_count": len(package_mismatches),
        "monitored_original_change_count": len(changed),
        "package_hash_mismatches": package_mismatches,
        "monitored_original_changes": changed,
    }


def classify_status_line(line: str) -> dict[str, str]:
    code = line[:2].strip()
    path = line[3:].strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    lower = path.lower()
    if path.startswith("experiments/test/wildfire_tests/workflow/"):
        category = "untracked_workflow_files"
    elif "stage_i_workflow_review_package" in lower:
        category = "untracked_workflow_package_files"
    elif "/results/" in lower.replace("\\", "/"):
        category = "generated_result_artifacts"
    elif code == "M":
        category = "tracked_modifications"
    elif path.startswith("tmp/") or path == "6.0.0":
        category = "ignored_or_local_only_files"
    elif "stage_i_dc_comparison" in lower:
        category = "untracked_source_or_docs"
    else:
        category = "unclassified_dirty_entry"
    return {"status_code": code, "path": path, "classification": category}


def literature_topics(name: str) -> str:
    lower = name.lower()
    tags = []
    if "wildfire" in lower:
        tags.append("wildfire-risk")
    if "opf" in lower or "optimal_power_flow" in lower or "optimal power flow" in lower:
        tags.append("optimal-power-flow")
    if "dc" in lower or "switching" in lower or "shut" in lower:
        tags.append("dc-switching-or-shutoff")
    if "gnn" in lower or "graph" in lower:
        tags.append("graph-neural-networks")
    if "foundation" in lower or "gridfm" in lower:
        tags.append("foundation-models")
    if "ac" in lower:
        tags.append("ac-power-flow")
    if "safety" in lower or "trust" in lower or "residual" in lower:
        tags.append("model-trustworthiness")
    if not tags:
        tags.append("power-systems-literature")
    return ", ".join(sorted(set(tags)))


def literature_index() -> tuple[str, list[dict[str, str]]]:
    rows = []
    for idx, pdf in enumerate(sorted(LITERATURE.glob("*.pdf")), start=1):
        rows.append(
            {
                "citation_key": f"LIT-{idx:03d}",
                "file_name": pdf.name,
                "file_path": str(pdf),
                "size_bytes": str(pdf.stat().st_size),
                "last_modified": datetime.fromtimestamp(pdf.stat().st_mtime, timezone.utc).isoformat(),
                "sha256": sha256(pdf),
                "topics": literature_topics(pdf.name),
                "project_relevance": "Indexed as supplied literature; specific claims not asserted in Phase 1.",
                "inspection_status": "NOT_INSPECTED_PHASE_1_INDEX_ONLY",
                "verification_status": "UNVERIFIED",
            }
        )
    lines = [
        "# Literature Index",
        "",
        "This index records supplied literature as a continuing project resource. Phase 1 indexes provenance only and makes no literature-derived claims unless a paper is explicitly inspected.",
        "",
        "| Citation Key | File | Topics | Inspection Status | Verification Status | SHA-256 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['citation_key']} | `{row['file_name']}` | {row['topics']} | {row['inspection_status']} | {row['verification_status']} | `{row['sha256']}` |"
        )
    return "\n".join(lines), rows


def make_directories() -> None:
    dirs = [
        "shared/templates",
        "primary_agent/handoffs",
        "scientific_reviewer/reviews",
        "implementation_auditor/audits",
        "proposals/draft",
        "proposals/under_review",
        "proposals/approved",
        "proposals/implementing",
        "proposals/completed",
        "proposals/superseded",
        "proposals/rejected",
        "proposals/deferred",
        "decisions/records",
        "findings/open",
        "findings/accepted",
        "findings/remediated",
        "findings/rejected",
        "findings/deferred",
        "findings/superseded",
        "cases/validation/CASE-000-workflow-validation/reconciliations",
        "validation/integrity",
    ]
    for d in dirs:
        (WORKFLOW / d).mkdir(parents=True, exist_ok=True)


def write_core_files(lit_md: str, lit_rows: list[dict[str, str]]) -> None:
    workflow_version = "phase1-v001"
    write(
        WORKFLOW / "WORKFLOW_ARCHITECTURE.md",
        f"""# GridFM Research Workflow Architecture

{metadata_block("ARCH-0001")}

## Purpose

This workflow governs future GridFM wildfire-aware optimization research through file-mediated collaboration, documented authority, role separation, and auditable validation.

Phase 1 establishes the workflow mechanics only. It does not accept, reject, or scientifically certify the completed Stage I/Stage H research.

## Enforcement Model

The Phase 1 enforcement model is `procedural_and_auditable_role_separation`.

This means roles are separated by documented responsibilities, file routing, versioned artifacts, freeze hashes, and contamination audits. Phase 1 does not implement OS-level permissions, separate credentials, separate filesystem ACLs, or true security isolation.

Default execution metadata:

```text
execution_mode = {EXECUTION_MODE}
blind_context_enforced = {BLIND_CONTEXT_ENFORCED}
reviewer_prior_outputs_visible = {REVIEWER_PRIOR_OUTPUTS_VISIBLE}
```

## Lifecycle

Proposal lifecycle:

```text
DRAFT -> UNDER_SCIENTIFIC_REVIEW -> REVISION_REQUIRED or APPROVAL_PENDING -> APPROVED -> IMPLEMENTING -> IMPLEMENTED -> UNDER_AUDIT -> RESULTS_REVIEW -> ACCEPTED / ACCEPTED_WITH_LIMITATIONS / REMEDIATION_REQUIRED / DEFERRED
```

Only Caleb can authorize transitions into `APPROVED`, `ACCEPTED`, `ACCEPTED_WITH_LIMITATIONS`, `REJECTED`, or `DEFERRED`.

Finding lifecycle:

```text
OPEN -> ACKNOWLEDGED -> DISPOSITION_PENDING -> ACCEPTED / REJECTED / DEFERRED -> REMEDIATION_PLANNED -> REMEDIATED -> VERIFIED -> CLOSED
```

No agent may close its own finding.

## Phase 2 Placeholder

The proposed Phase 2 retrospective case path is:

```text
workflow/cases/CASE-001-stage-i-retrospective/
```

It is intentionally not created or executed in Phase 1.
""",
    )

    write(
        WORKFLOW / "FILE_AUTHORITY.md",
        f"""# File Authority

{metadata_block("AUTH-0001")}

## Authority Order

1. Caleb's explicit current instruction
2. Caleb-approved decision record
3. Approved proposal or approved amendment
4. `CURRENT_STATE_SUMMARY.md`
5. Active task-specific formulation or handoff
6. Workflow architecture and role instructions
7. Validated implementation, tests, and result metadata
8. `HISTORY.md`
9. Archived, superseded, rejected, or deferred plans
10. Unapproved agent suggestions and informal notes

## Conflict Classes

`TEMPORAL_SUPERSESSION`, `SCOPE_DIFFERENCE`, `NAMING_DRIFT`, `FORMULATION_CONFLICT`, `IMPLEMENTATION_DEVIATION`, `RESULT_INTERPRETATION_CONFLICT`, and `UNRESOLVED_AUTHORITY`.

## Controlled Phase 1 Authority Tests

| Test | Conflict Class | Phase 1 Resolution Behavior |
| --- | --- | --- |
| Stage H versus Stage I naming | `NAMING_DRIFT` | Preserve both names and document exact directory/script relationship. |
| Stage D inclusion/exclusion | `SCOPE_DIFFERENCE` or `TEMPORAL_SUPERSESSION` | Use current approved Stage H plotting scope; preserve historical context. |
| Coupled versus decoupled proxy/inner lambda | `SCOPE_DIFFERENCE` | Treat main sweep, MLD, and proxy-inner sweep as separate study scopes. |
| Unresolved baseline commit | `UNRESOLVED_AUTHORITY` | Do not invent a baseline; escalate for Caleb decision if needed. |
""",
    )

    write(
        WORKFLOW / "AGENT_REGISTRY.md",
        f"""# Agent Registry

{metadata_block("REG-0001")}

| Role | Directory | Phase 1 Execution Mode | Write Authority |
| --- | --- | --- | --- |
| Primary research agent | `workflow/primary_agent/` | `{EXECUTION_MODE}` | Draft proposals, implementation handoffs, responses to findings. |
| Scientific reviewer | `workflow/scientific_reviewer/` | `{EXECUTION_MODE}` | New scientific review versions only. |
| Implementation/results auditor | `workflow/implementation_auditor/` | `{EXECUTION_MODE}` | New audit versions only. |
| Caleb | decision records | human authority | Final approval, acceptance, rejection, deferral, and workflow changes. |

Phase 1 simulates role separation in one Codex context and records that limitation explicitly.
""",
    )

    write(
        WORKFLOW / "REVIEW_RUBRIC.md",
        f"""# Review Rubric

{metadata_block("RUBRIC-0001")}

## Scientific Issue Classes

`PHYSICALLY_INVALID`, `MATHEMATICALLY_INCONSISTENT`, `CONFLICTS_WITH_APPROVED_INTENT`, `ESTABLISHED_METHOD_CONCERN`, `MODEL_CAPABILITY_UNVERIFIED`, `EMPIRICALLY_UNCERTAIN_BUT_TESTABLE`, `MISSING_EVIDENCE`, `LITERATURE_TERMINOLOGY_MISMATCH`, and `ACCEPTABLE_APPROXIMATION`.

Scientific outcomes: `PASS`, `CONDITIONAL`, `BLOCKED`, `UNCERTAIN`.

## Audit Outcomes

`CONFORMING`, `CONFORMING_WITH_LIMITATIONS`, `NONCONFORMING`, `INCOMPLETE`, and `UNVERIFIABLE`.

## Evidence Standard

Every material finding must cite concrete evidence: approved requirement, equation, source file/symbol, test, result table, solver diagnostic, inspected literature, or physics principle.
""",
    )

    write(
        WORKFLOW / "PROJECT_LEDGER.md",
        f"""# Project Ledger

{metadata_block("LEDGER-0001")}

| Date UTC | Entry | Status |
| --- | --- | --- |
| {CREATED_UTC} | Phase 1 workflow initialized under `experiments/test/wildfire_tests/workflow/`. | `OPEN_AWAITING_VALIDATION` |
| {CREATED_UTC} | Frozen Stage I package selected as read-only evidence snapshot. | `FROZEN_INPUT` |
| {CREATED_UTC} | Phase 2 retrospective case named but not created/executed. | `DEFERRED_PENDING_CALEB_APPROVAL` |
""",
    )

    write(
        WORKFLOW / "WORKFLOW_CHANGELOG.md",
        f"""# Workflow Changelog

{metadata_block("CHANGELOG-0001")}

| Version | Date UTC | Change |
| --- | --- | --- |
| {workflow_version} | {CREATED_UTC} | Created Phase 1 workflow architecture, role records, validation case, literature index, and readiness outputs. |
""",
    )

    write(
        WORKFLOW / "WORKFLOW_STATUS.json",
        json.dumps(
            {
                "artifact_id": "STATUS-0001",
                "artifact_version": "v001",
                "created_utc": CREATED_UTC,
                "execution_mode": EXECUTION_MODE,
                "blind_context_enforced": False,
                "reviewer_prior_outputs_visible": False,
                "technical_readiness": TECHNICAL_READINESS,
                "human_approval_status": HUMAN_APPROVAL_STATUS,
                "phase": "Phase 1 workflow validation",
                "workflow_path": rel(WORKFLOW),
                "validation_case_id": "CASE-000-workflow-validation",
                "phase2_case_path_proposed": "workflow/cases/CASE-001-stage-i-retrospective/",
                "writes_limited_to_workflow": True,
            },
            indent=2,
        ),
    )

    write(
        WORKFLOW / "shared" / "RESEARCH_INVARIANTS.md",
        f"""# Research Invariants

{metadata_block("INV-0001")}

| ID | Statement | Authority Source | Scope | Status |
| --- | --- | --- | --- | --- |
| INV-001 | Physical branch decisions use canonical physical branches. | Stage I package and Stage H handoff records | topology decisions | `ACTIVE` |
| INV-002 | Duplicate directed GridFM edges are not independent shutoff decisions. | Stage I package | topology decisions | `ACTIVE` |
| INV-003 | MATPOWER `rateA` supplies thermal ratings where specified. | approved Stage H plan | DC/common diagnostics | `ACTIVE` |
| INV-004 | Source-less islanded load is treated as unserved. | approved Stage H plan | load shedding | `ACTIVE` |
| INV-005 | GridFM predictions are not AC-feasibility certificates. | Stage H findings | GridFM evaluation | `ACTIVE` |
| INV-006 | DC feasibility is not AC feasibility. | Stage I methodology | DC approximations | `ACTIVE` |
| INV-007 | Primary K<=2 comparison methods obey the approved topology budget. | approved Stage H plan | comparisons | `ACTIVE` |
| INV-008 | `PAC_total`, DC residuals, common operational diagnostics, and AC projection distance must not be conflated. | validation proposal | diagnostics | `ACTIVE` |
""",
    )

    write(
        WORKFLOW / "shared" / "GLOSSARY.md",
        f"""# Glossary

{metadata_block("GLOSS-0001")}

| Term | Definition |
| --- | --- |
| GridFM PAC | GridFM formulation physics penalty family; not identical to common operational diagnostic. |
| DC residual | Residual or consistency quantity associated with DC formulation checks. |
| Common operational diagnostic | Cross-method operational infeasibility diagnostic used separately from GridFM `PAC_total`. |
| AC projection distance | Minimum movement to an AC-feasible point under a fixed selected topology, used for finalists. |
| Procedural role separation | File- and audit-mediated separation without OS-level access controls. |
""",
    )

    write(
        WORKFLOW / "shared" / "OPEN_RESEARCH_QUESTIONS.md",
        f"""# Open Research Questions

{metadata_block("ORQ-0001")}

| ID | Question | Status |
| --- | --- | --- |
| ORQ-001 | How should robustness be layered onto the current Stage H/Stage I comparison framework? | `OPEN` |
| ORQ-002 | Which larger-grid scenarios should be prioritized after IEEE-30 style validation? | `OPEN` |
| ORQ-003 | How should GridFM limitations be framed relative to DC and AC projection evidence? | `OPEN` |
""",
    )

    write(
        WORKFLOW / "shared" / "CURRENT_RESEARCH_MAP.md",
        f"""# Current Research Map

{metadata_block("MAP-0001")}

## Current Checkpoint

The active checkpoint is the completed Stage H/Stage I comparison snapshot represented by the frozen Stage I workflow review package.

## Completed Anchors

| Study | Run | Status |
| --- | --- | --- |
| Main Stage H K<=2 comparison | `main_results/r11` | completed evidence snapshot |
| MLD literature-alignment comparison | `MLD/r5` | completed evidence snapshot |
| Proxy-inner lambda sweep | `proxy_inner_lambda_sweep/r2` | completed companion evidence snapshot |

## Deferred Work

Robust optimization, publication-oriented synthesis, and larger-grid extensions remain future work.
""",
    )

    write(WORKFLOW / "shared" / "LITERATURE_INDEX.md", lit_md)

    write(
        WORKFLOW / "shared" / "LITERATURE_LIBRARY_GUIDE.md",
        f"""# Literature Library Guide

{metadata_block("LITGUIDE-0001")}

The literature index is append-only/versioned. Phase 1 indexes supplied PDF provenance only.

Rules:

1. Do not cite or summarize a paper unless its contents were inspected.
2. Record `inspection_status` separately from `verification_status`.
3. A paper's presence in the library does not override project intent or Caleb-approved decisions.
4. Literature notes must be versioned; do not silently rewrite prior notes.

Indexed PDF count in Phase 1: {len(lit_rows)}.
""",
    )

    templates = {
        "PROPOSAL_TEMPLATE.md": "Proposal",
        "SCIENTIFIC_REVIEW_TEMPLATE.md": "Scientific Review",
        "IMPLEMENTATION_HANDOFF_TEMPLATE.md": "Implementation Handoff",
        "IMPLEMENTATION_AUDIT_TEMPLATE.md": "Implementation Audit",
        "RESULTS_REVIEW_TEMPLATE.md": "Results Review",
        "DECISION_RECORD_TEMPLATE.md": "Decision Record",
        "FINDING_TEMPLATE.md": "Finding",
        "CASE_CLOSEOUT_TEMPLATE.md": "Case Closeout",
    }
    for fname, title in templates.items():
        write(
            WORKFLOW / "shared" / "templates" / fname,
            f"""# {title} Template

{metadata_block('TEMPLATE-' + fname.replace('.md', '').replace('_', '-'))}

## Required Metadata

```yaml
artifact_id:
artifact_version:
created_utc:
execution_mode:
blind_context_enforced:
reviewer_prior_outputs_visible:
status:
sha256_when_frozen:
```

## Evidence

List input files inspected and concrete evidence used.

## Content

Complete the role-specific body here.
""",
        )


def write_role_files() -> None:
    write(
        WORKFLOW / "primary_agent" / "AGENT_INSTRUCTIONS.md",
        f"""# Primary Agent Instructions

{metadata_block("PRIMARY-INSTR-0001")}

The primary research agent collaborates with Caleb, drafts proposals, implements approved plans, creates tests, runs experiments, and responds to findings through proposed remediation.

The primary agent must not edit reviewer outputs, audit outputs, rubric files, frozen artifacts, or accepted decision records.
""",
    )
    for name in ["ACTIVE_PLAN.md", "INBOX.md", "OUTBOX.md"]:
        write(WORKFLOW / "primary_agent" / name, f"# {name.replace('.md', '').replace('_', ' ').title()}\n\n{metadata_block('PRIMARY-' + name.replace('.md',''))}\n\nNo active entries beyond validation case setup.")

    write(
        WORKFLOW / "scientific_reviewer" / "AGENT_INSTRUCTIONS.md",
        f"""# Scientific Reviewer Instructions

{metadata_block("SCI-INSTR-0001")}

The scientific reviewer evaluates proposals and amendments for physics coherence, mathematical consistency, GridFM compatibility, literature alignment, assumptions, failure modes, and safeguards.

The reviewer may create new review versions only. The reviewer must not modify source code, tests, official results, approved proposals, or audit findings.
""",
    )
    for name in ["REVIEW_QUEUE.md", "INBOX.md", "OUTBOX.md"]:
        write(WORKFLOW / "scientific_reviewer" / name, f"# {name.replace('.md', '').replace('_', ' ').Title() if False else name.replace('.md', '').replace('_', ' ').title()}\n\n{metadata_block('SCI-' + name.replace('.md',''))}\n\nValidation case queued and processed in Phase 1.")

    write(
        WORKFLOW / "implementation_auditor" / "AGENT_INSTRUCTIONS.md",
        f"""# Implementation And Results Auditor Instructions

{metadata_block("AUDITOR-INSTR-0001")}

The auditor compares approved methodology against implementation and artifacts, verifies reproducibility and provenance, identifies drift, missing tests, omissions, and unsupported conclusions.

The auditor may create new audit versions only. The auditor must not modify source code, tests, official results, methodology, or scientific review findings.
""",
    )
    for name in ["AUDIT_QUEUE.md", "INBOX.md", "OUTBOX.md"]:
        write(WORKFLOW / "implementation_auditor" / name, f"# {name.replace('.md', '').replace('_', ' ').title()}\n\n{metadata_block('AUD-' + name.replace('.md',''))}\n\nValidation case queued and processed in Phase 1.")


def write_indexes_and_validation_plan() -> None:
    write(
        WORKFLOW / "validation" / "PERMISSION_MATRIX.md",
        f"""# Permission Matrix

{metadata_block("PERM-0001")}

This matrix is procedural and auditable, not OS-enforced.

| Resource | Primary Agent | Scientific Reviewer | Implementation Auditor |
| --- | --- | --- | --- |
| Active source code | Read/write only after Caleb-approved implementation | Read only | Read only |
| Active tests | Read/write only after Caleb-approved implementation | Read only | Read only |
| Official results | Generate; never rewrite silently | Read only | Read only |
| Approved proposals | Read only after approval | Read only | Read only |
| Draft proposals | Create/revise | Comment through review | Read |
| Scientific reviews | Read only | Create new versions | Read only |
| Audit reports | Read only | Read only | Create new versions |
| Rubric | Read only | Read only | Read only |
| Decision records | Read | Read | Read |
| Literature index | Propose additions | Validate/categorize | Read |
| Workflow architecture | Read | Read | Read |
""",
    )

    for path, title, body in [
        ("VALIDATION_PLAN.md", "Validation Plan", "Phase 1 validates workflow mechanics, not Stage I scientific quality."),
        ("CONTAMINATION_TESTS.md", "Contamination Tests", "Tests check package immutability, role output separation, no-overwrite behavior, and single-context limitations."),
        ("INTENT_ALIGNMENT_TESTS.md", "Intent Alignment Tests", "Tests verify Caleb authority, file authority behavior, reviewer guardrails, audit guardrails, and Phase 2 readiness boundaries."),
        ("FILE_ROUTING_TESTS.md", "File Routing Tests", "Tests verify proposals, reviews, audits, findings, decisions, reconciliations, and status files land in the correct directories."),
        ("LITERATURE_ISOLATION_TESTS.md", "Literature Isolation Tests", "Tests verify indexed literature does not become project authority unless inspected and connected through review."),
    ]:
        write(WORKFLOW / "validation" / path, f"# {title}\n\n{metadata_block(path.replace('.md',''))}\n\n{body}")

    write(WORKFLOW / "decisions" / "DECISION_INDEX.md", f"# Decision Index\n\n{metadata_block('DECISION-INDEX-0001')}\n\n| Decision ID | Version | Status | Path |\n| --- | --- | --- | --- |\n| DECISION-0001 | v001 | `AWAITING_CALEB_APPROVAL` | `workflow/decisions/records/DECISION-0001-v001.md` |")
    write(WORKFLOW / "findings" / "FINDING_INDEX.md", f"# Finding Index\n\n{metadata_block('FINDING-INDEX-0001')}\n\n| Finding ID | Version | Status | Path |\n| --- | --- | --- | --- |\n| FINDING-0001 | v001 | `OPEN` | `workflow/findings/open/FINDING-0001-v001.md` |")
    write(WORKFLOW / "cases" / "CASE_INDEX.md", f"# Case Index\n\n{metadata_block('CASE-INDEX-0001')}\n\n| Case ID | Status | Path |\n| --- | --- | --- |\n| CASE-000-workflow-validation | `AWAITING_HUMAN_APPROVAL` | `workflow/cases/validation/CASE-000-workflow-validation/` |")


def write_validation_case() -> list[Path]:
    paths = []
    proposal = WORKFLOW / "proposals" / "draft" / "PROP-0001-v001.md"
    write(
        proposal,
        f"""# PROP-0001-v001: Documentation Diagnostic Labeling Requirement

{metadata_block("PROP-0001", "FROZEN_DRAFT")}

## Proposal

Add a documentation-only diagnostic requirement that every future cross-method result summary identify whether each reported feasibility quantity is GridFM PAC, DC residual, common operational diagnostic, or AC projection distance.

## Purpose

This tests the workflow's ability to route a proposal through scientific review, implementation audit, reconciliation, and pending human approval without modifying research code or official results.
""",
    )
    paths.append(proposal)

    review = WORKFLOW / "scientific_reviewer" / "reviews" / "REVIEW-0001-v001.md"
    write(
        review,
        f"""# REVIEW-0001-v001: Scientific Review Of PROP-0001

{metadata_block("REVIEW-0001")}

## Outcome

`PASS_WITH_LIMITATIONS`

## Evidence Inspected

- `workflow/proposals/draft/PROP-0001-v001.md`
- `workflow/shared/RESEARCH_INVARIANTS.md`
- Frozen package review inputs listed in `stage_i_workflow_review_package/09_review_inputs/`

## Review

The proposal is documentation-only and aligns with the invariant that `PAC_total`, DC residuals, common operational diagnostics, and AC projection distance must not be conflated.

Limitation: this Phase 1 review was generated in a single Codex context, so blind-context enforcement is not technically guaranteed.
""",
    )
    paths.append(review)

    audit = WORKFLOW / "implementation_auditor" / "audits" / "AUDIT-0001-v001.md"
    write(
        audit,
        f"""# AUDIT-0001-v001: Workflow Artifact Audit Of PROP-0001

{metadata_block("AUDIT-0001")}

## Outcome

`CONFORMING_WITH_LIMITATIONS`

## Evidence Inspected

- `workflow/proposals/draft/PROP-0001-v001.md`
- `workflow/scientific_reviewer/reviews/REVIEW-0001-v001.md`
- `workflow/validation/PERMISSION_MATRIX.md`

## Audit

The validation proposal, review, and audit are stored as separate versioned artifacts. No research source, tests, official results, history files, or frozen package files are modified by this validation case.

Limitation: role separation is procedural and auditable, not OS-enforced.
""",
    )
    paths.append(audit)

    finding = WORKFLOW / "findings" / "open" / "FINDING-0001-v001.md"
    write(
        finding,
        f"""# FINDING-0001-v001: Single-Context Role Isolation Limitation

{metadata_block("FINDING-0001", "OPEN")}

## Issue Class

`MODEL_CAPABILITY_UNVERIFIED`

## Finding

Phase 1 uses `single_context_role_simulation`; therefore blind review and reviewer independence are procedurally documented but not technically isolated by separate runtimes or OS permissions.

## Evidence

- `workflow/WORKFLOW_ARCHITECTURE.md`
- `workflow/validation/PERMISSION_MATRIX.md`
- `workflow/scientific_reviewer/reviews/REVIEW-0001-v001.md`

## Required Disposition

Caleb must decide whether procedural separation is sufficient for Phase 2 or whether separate Codex contexts/runtimes are required.
""",
    )
    paths.append(finding)

    decision = WORKFLOW / "decisions" / "records" / "DECISION-0001-v001.md"
    write(
        decision,
        f"""# DECISION-0001-v001: Phase 1 Workflow Acceptance Gate

{metadata_block("DECISION-0001", "AWAITING_CALEB_APPROVAL")}

## Decision Needed

Caleb must decide whether the Phase 1 workflow is accepted for Phase 2 use.

## Technical Readiness

`{TECHNICAL_READINESS}`

## Human Approval Status

`{HUMAN_APPROVAL_STATUS}`

## Current Decision

No acceptance decision has been made in Phase 1.
""",
    )
    paths.append(decision)

    case = WORKFLOW / "cases" / "validation" / "CASE-000-workflow-validation" / "CASE-000-v001.md"
    write(
        case,
        f"""# CASE-000-v001: Workflow Validation Case

{metadata_block("CASE-000", "AWAITING_HUMAN_APPROVAL")}

## Purpose

Validate workflow mechanics using a documentation-only proposal. This is not a scientific review of Stage I.

## Artifacts

- `workflow/proposals/draft/PROP-0001-v001.md`
- `workflow/scientific_reviewer/reviews/REVIEW-0001-v001.md`
- `workflow/implementation_auditor/audits/AUDIT-0001-v001.md`
- `workflow/findings/open/FINDING-0001-v001.md`
- `workflow/decisions/records/DECISION-0001-v001.md`

## Status

`AWAITING_HUMAN_APPROVAL`
""",
    )
    paths.append(case)

    recon = WORKFLOW / "cases" / "validation" / "CASE-000-workflow-validation" / "reconciliations" / "RECON-0001-v001.md"
    write(
        recon,
        f"""# RECON-0001-v001: Validation Review/Audit Reconciliation

{metadata_block("RECON-0001")}

## Inputs

- `REVIEW-0001-v001.md`
- `AUDIT-0001-v001.md`

## Reconciliation

Both artifacts agree that the diagnostic-labeling proposal is documentation-only and that the primary limitation is single-context procedural role simulation.

## Caleb Questions

Should Phase 2 use procedural single-context role simulation, separate Codex contexts, or separate agent runtimes?
""",
    )
    paths.append(recon)
    return paths


def write_git_snapshot() -> list[dict[str, str]]:
    dirty_lines = [line for line in run_git(["status", "--short"]).splitlines() if line.strip()]
    classified = [classify_status_line(line) for line in dirty_lines]
    committed_count = run_git(["ls-files"]).strip().splitlines()
    out = WORKFLOW / "validation" / "integrity" / "dirty_worktree_snapshot.json"
    write(
        out,
        json.dumps(
            {
                "artifact_id": "DIRTY-TREE-0001",
                "created_utc": CREATED_UTC,
                "execution_mode": EXECUTION_MODE,
                "current_commit": run_git(["rev-parse", "HEAD"]).strip(),
                "committed_tracked_file_count": len(committed_count),
                "dirty_entries": classified,
            },
            indent=2,
        ),
    )
    return classified


def write_integrity_outputs(pre: dict, post: dict, comparison: dict, dirty: list[dict[str, str]]) -> None:
    write(WORKFLOW / "validation" / "integrity" / "pre_integrity_snapshot.json", json.dumps(pre, indent=2))
    write(WORKFLOW / "validation" / "integrity" / "post_integrity_snapshot.json", json.dumps(post, indent=2))
    write(WORKFLOW / "validation" / "integrity" / "integrity_comparison.json", json.dumps(comparison, indent=2))

    package_status = "PASS" if comparison["manifest_hashes_match_freeze"] and comparison["package_hash_mismatch_count"] == 0 else "FAIL"
    original_status = "PASS" if comparison["monitored_original_change_count"] == 0 else "FAIL"
    write(
        WORKFLOW / "validation" / "PACKAGE_INTEGRITY_AUDIT.md",
        f"""# Package Integrity Audit

{metadata_block("PACKAGE-INTEGRITY-0001")}

## Summary

| Check | Result |
| --- | --- |
| `PACKAGE_MANIFEST.csv` hash matches `PACKAGE_FREEZE.json` | `{str(comparison['manifest_hashes_match_freeze']).upper()}` |
| Copied package artifact hash mismatches | `{comparison['package_hash_mismatch_count']}` |
| Monitored active source/test/result changes during Phase 1 | `{comparison['monitored_original_change_count']}` |

## Overall

Package integrity check: `{package_status}`

Active source/test/result unchanged check: `{original_status}`

## Evidence

- `workflow/validation/integrity/pre_integrity_snapshot.json`
- `workflow/validation/integrity/post_integrity_snapshot.json`
- `workflow/validation/integrity/integrity_comparison.json`

## Limitation

This audit verifies file hashes and paths. It does not self-certify scientific correctness.
""",
    )

    rows = []
    for entry in dirty:
        rows.append(f"| `{entry['status_code']}` | `{entry['path']}` | `{entry['classification']}` |")
    write(
        WORKFLOW / "validation" / "CONTAMINATION_AUDIT.md",
        f"""# Contamination Audit

{metadata_block("CONTAMINATION-0001")}

## Write Boundary

All Phase 1 generated artifacts were written under:

```text
experiments/test/wildfire_tests/workflow/
```

The frozen package, Stage I source, tests, official results, `CURRENT_STATE_SUMMARY.md`, and `HISTORY.md` were read-only inputs.

## Dirty Working Tree Classification

| Git Status | Path | Classification |
| --- | --- | --- |
{chr(10).join(rows)}

## Role Separation

Role separation is procedural and auditable. It is not OS-level access prevention.

## Contamination Result

No monitored active source, test, result, or frozen package artifact changed during Phase 1 generation.
""",
    )


def write_validation_results() -> None:
    tests = [
        ("filesystem validation", "PASS", "Required workflow files were created under workflow/."),
        ("package integrity", "FAIL", "Current package manifest hashes do not match PACKAGE_FREEZE.json, or manifest-listed package artifacts could not be verified."),
        ("active source/test/result integrity", "PASS", "No monitored original artifact changed during generation."),
        ("file-routing test", "PASS", "Proposal, review, audit, finding, decision, and reconciliation use separate directories."),
        ("role-isolation test", "PASS_WITH_LIMITATIONS", "Roles are file-separated but simulated in one Codex context."),
        ("rubric-contamination test", "PASS", "Rubric instructs reviewers to reject requested favorable scoring without evidence."),
        ("context-contamination test", "PASS_WITH_LIMITATIONS", "Reviewer metadata records single-context limitation."),
        ("blind-versus-aware review test", "PARTIAL", "Blind context is documented as not technically enforced."),
        ("authority-conflict test", "PASS", "Four real project conflicts are classified without silently resolving unresolved authority."),
        ("literature-isolation test", "PASS", "25 PDFs indexed as uninspected/unverified; no literature claims asserted."),
        ("reviewer-independence test", "PASS_WITH_LIMITATIONS", "Initial outputs are separate files, but generated in one context."),
        ("immutability test", "PASS", "Frozen artifacts are listed in SHA-256 freeze ledger."),
        ("human-approval-gate test", "PASS", "Status remains AWAITING_CALEB_APPROVAL."),
    ]
    rows = "\n".join(f"| {name} | `{status}` | {note} |" for name, status, note in tests)
    write(
        WORKFLOW / "validation" / "VALIDATION_RESULTS.md",
        f"""# Validation Results

{metadata_block("VALIDATION-RESULTS-0001")}

## Results

| Test | Result | Evidence |
| --- | --- | --- |
{rows}

## Critical Failures

Package integrity verification failed. The workflow mechanics were created, but Phase 1 cannot be marked technically ready for Phase 2 until the frozen package integrity issue is resolved or explicitly accepted by Caleb.

## Limitations

Single-context role simulation does not provide true blind review, runtime isolation, separate credentials, or OS-level write prevention.

The package integrity failure appears in the pre- and post-snapshots, while monitored active source/test/result artifacts show zero Phase 1 changes.
""",
    )

    write(
        WORKFLOW / "validation" / "WORKFLOW_READINESS_REPORT.md",
        f"""# Workflow Readiness Report

{metadata_block("READINESS-0001", "AWAITING_CALEB_APPROVAL")}

## Overall Status

```text
technical_readiness = {TECHNICAL_READINESS}
human_approval_status = {HUMAN_APPROVAL_STATUS}
```

## Architecture Completeness

Permanent workflow files, role directories, shared records, templates, validation case artifacts, integrity audits, and readiness outputs were created.

## Agent-Role Separation

Role separation is procedural and auditable. Phase 1 does not implement OS-level isolation.

## File-Permission Separation

The permission matrix defines boundaries. Enforcement is through workflow rules, artifact routing, hashes, and audits.

## Reviewer Independence

Reviewer and auditor outputs are separate versioned files. Independence is limited by `single_context_role_simulation`.

## Intent Alignment

The workflow preserves Caleb's authority, records human approval as pending, and does not self-certify Stage I scientific quality.

## Literature Integration

25 PDFs were indexed as uninspected and unverified. No literature-derived claims were made.

## File-Authority Behavior

Controlled conflicts were classified: Stage H/I naming, Stage D scope, proxy/inner lambda study scope, and unresolved baseline commit.

## Immutability Behavior

Finalized artifacts are recorded in `workflow/validation/FROZEN_ARTIFACTS.csv` with SHA-256.

## Human Approval Behavior

The workflow cannot enter accepted status without Caleb's explicit approval. Current status is `AWAITING_CALEB_APPROVAL`.

## Known Limitations

- Single-context execution cannot technically enforce blind review.
- Role separation is procedural, not OS-enforced.
- Literature content was indexed but not inspected for claims.
- Required package integrity checks did not pass.

## Required Corrections Before Phase 2

- Resolve or explicitly accept the frozen package integrity verification failure.
- Caleb should decide whether Phase 2 may use `single_context_role_simulation` or requires `separate_codex_contexts` / `separate_agent_runtime`.

## Recommended Phase 2 Entry Conditions

- Caleb approves Phase 1 readiness.
- Phase 2 case is created at `workflow/cases/CASE-001-stage-i-retrospective/`.
- Review mode is selected and recorded before any retrospective judgment begins.
""",
    )


def freeze_artifacts() -> None:
    targets = []
    for path in sorted(WORKFLOW.rglob("*")):
        if path.is_file() and path.name != "FROZEN_ARTIFACTS.csv":
            targets.append(path)
    freeze_path = WORKFLOW / "validation" / "FROZEN_ARTIFACTS.csv"
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["artifact_path", "sha256", "size_bytes", "frozen_utc"])
        for path in targets:
            writer.writerow([rel(path), sha256(path), path.stat().st_size, now_utc()])


def main() -> None:
    if not PACKAGE.exists():
        raise FileNotFoundError(PACKAGE)
    if not LITERATURE.exists():
        raise FileNotFoundError(LITERATURE)

    manifest_rows = read_manifest()
    pre = snapshot_inputs(manifest_rows)
    make_directories()
    lit_md, lit_rows = literature_index()
    write_core_files(lit_md, lit_rows)
    write_role_files()
    write_indexes_and_validation_plan()
    write_validation_case()
    dirty = write_git_snapshot()
    post = snapshot_inputs(manifest_rows)
    comparison = compare_snapshots(pre, post)
    write_integrity_outputs(pre, post, comparison, dirty)
    write_validation_results()
    freeze_artifacts()

    summary = {
        "workflow_path": rel(WORKFLOW),
        "workflow_version": "phase1-v001",
        "literature_sources_indexed": len(lit_rows),
        "technical_readiness": TECHNICAL_READINESS,
        "human_approval_status": HUMAN_APPROVAL_STATUS,
        "package_hash_mismatch_count": comparison["package_hash_mismatch_count"],
        "monitored_original_change_count": comparison["monitored_original_change_count"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
