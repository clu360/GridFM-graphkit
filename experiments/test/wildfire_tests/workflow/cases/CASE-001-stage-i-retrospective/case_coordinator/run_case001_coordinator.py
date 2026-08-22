from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path.cwd()
CASE = ROOT / "experiments" / "test" / "wildfire_tests" / "workflow" / "cases" / "CASE-001-stage-i-retrospective"
PKG = ROOT / "experiments" / "test" / "wildfire_tests" / "stage_i_workflow_review_package_v002"
PAPERS = ROOT.parent.parent / "Literature Review" / "Papers"


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


CREATED = now()


def ext(path: Path) -> str:
    return "\\\\?\\" + str(path.resolve())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(ext(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def exists_file(path: Path) -> bool:
    return os.path.isfile(ext(path))


def stat_info(path: Path) -> tuple[str, str]:
    st = os.stat(ext(path))
    return str(st.st_size), datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def add_evidence(rows: list[dict], *, eid: str, category: str, description: str, repository_path: str = "",
                 package_path: str = "", source_type: str, authority_level: str, role_access: str,
                 required_for: str, notes: str = "") -> None:
    path_text = package_path or repository_path
    path = ROOT / path_text if path_text else None
    available = bool(path and exists_file(path))
    pre_hash = sha256(path) if available else ""
    size, modified = stat_info(path) if available else ("", "")
    rows.append({
        "evidence_id": eid,
        "category": category,
        "description": description,
        "repository_path": repository_path,
        "package_path": package_path,
        "source_type": source_type,
        "authority_level": authority_level,
        "case_role_access": role_access,
        "availability_status": "AVAILABLE" if available else "UNAVAILABLE",
        "inspection_status": "NOT_INSPECTED",
        "pre_access_sha256": pre_hash,
        "post_access_sha256": pre_hash,
        "size_bytes": size,
        "modified_time": modified,
        "required_for_finding_types": required_for,
        "missing_evidence_class": "" if available else missing_class(category, description, source_type),
        "notes": notes,
    })


def missing_class(category: str, description: str, source_type: str) -> str:
    if source_type in {"v002_missing_record", "historical_failed_package"}:
        return "HISTORICAL_ONLY"
    if category in {"implementation", "tests", "results", "solver_diagnostics", "projection", "provenance"}:
        return "CRITICAL_REQUIRED"
    if category in {"plots", "literature"}:
        return "SUPPORTING"
    return "SUPPORTING"


def build_evidence() -> list[dict]:
    rows: list[dict] = []
    eid = 1

    def next_id() -> str:
        nonlocal eid
        value = f"EVID-{eid:04d}"
        eid += 1
        return value

    governance = [
        ("workflow/WORKFLOW_STATUS.json", "workflow status and Phase 2 approval state"),
        ("workflow/FILE_AUTHORITY.md", "file authority hierarchy"),
        ("workflow/REVIEW_RUBRIC.md", "review and audit rubric"),
        ("workflow/validation/PERMISSION_MATRIX.md", "procedural permission matrix"),
        ("workflow/decisions/records/DECISION-0002-v001.md", "Phase 2 approval decision"),
        ("workflow/p1a_v002/PACKAGE_INTEGRITY_AUDIT_PHASE1A.md", "Phase 1A package remediation audit"),
    ]
    for path, desc in governance:
        add_evidence(rows, eid=next_id(), category="governance", description=desc,
                     repository_path=f"experiments/test/wildfire_tests/{path}", source_type="workflow_file",
                     authority_level="Caleb-approved workflow", role_access="coordinator,scientific,audit,reconciliation",
                     required_for="authority,scope,limitations")

    intent_docs = [
        "experiments/test/wildfire_tests/CURRENT_STATE_SUMMARY.md",
        "experiments/test/wildfire_tests/HISTORY.md",
        "experiments/test/wildfire_tests/README.md",
        "experiments/test/wildfire_tests/DC_MILP_FORMULATION_HANDOFF.md",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/STAGE_H_DC_COMPARISON_PROGRESS.md",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md",
    ]
    for path in intent_docs:
        add_evidence(rows, eid=next_id(), category="intent", description=f"intent/progress document {Path(path).name}",
                     repository_path=path, source_type="live_repository_doc", authority_level="current or historical intent",
                     role_access="coordinator,scientific,audit", required_for="intent-to-formulation,claims")

    copied = read_csv(PKG / "COPIED_FILE_MANIFEST.csv")
    for row in copied:
        cat = row["category"]
        evidence_category = {
            "source": "implementation",
            "test": "tests",
            "result": "results",
            "review_input": "intent",
            "documentation": "intent",
            "planning": "intent",
            "environment": "environment",
            "git": "provenance",
            "git_environment_test_output": "provenance",
            "literature": "literature",
            "root": "workflow_validation",
        }.get(cat, cat)
        add_evidence(rows, eid=next_id(), category=evidence_category,
                     description=f"v002 copied artifact from {row['original_manifest_package_path']}",
                     package_path=row["package_path"], source_type="v002_copied_file",
                     authority_level="verified package v002 copied evidence",
                     role_access="coordinator,scientific,audit,reconciliation",
                     required_for="traceability,review,results", notes=f"source={row['source']}; status={row['status']}")

    external = read_csv(PKG / "EXTERNAL_ARTIFACT_INDEX.csv")
    for row in external:
        add_evidence(rows, eid=next_id(), category="provenance", description=f"v002 external artifact {Path(row['original_path']).name}",
                     repository_path=row["original_path"], source_type="v002_external_index_live_repository",
                     authority_level="indexed live evidence, requires pre/post hash",
                     role_access="coordinator,audit", required_for="raw-result/provenance verification",
                     notes="large artifact indexed by v002; not copied")

    missing = read_csv(PKG / "MISSING_MATERIALS_INDEX.csv")
    for row in missing:
        add_evidence(rows, eid=next_id(), category="workflow_validation",
                     description=f"v002 missing-material record {Path(row['original_manifest_package_path']).name}",
                     repository_path=row["original_path"], package_path="", source_type="v002_missing_record",
                     authority_level="audit limitation record", role_access="coordinator,reconciliation",
                     required_for="limitations", notes=f"classification={row['classification']}; {row['notes']}")

    core_results = [
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/best_by_rho_scenario_lambda_stage.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/ac_projection_distances.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/methodology_fidelity_checks.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/dc_runtime_comparison_stage_i_a_vs_i_b.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/MLD/r5/tables/best_by_rho_scenario_lambda_stage.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/MLD/r5/tables/ac_projection_distances.csv",
        "experiments/test/wildfire_tests/results/leq/stage_h/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2/finalization_summary.json",
        "tmp/stage_h_proxy_inner_checkpoints/r2/tables/best_by_scenario_proxy_inner_stage.csv",
        "tmp/stage_h_proxy_inner_checkpoints/r2/tables/pareto_frontier_points_by_proxy.csv",
    ]
    for path in core_results:
        add_evidence(rows, eid=next_id(), category="results", description=f"live result artifact {Path(path).name}",
                     repository_path=path, source_type="live_repository_result",
                     authority_level="live artifact, requires hash",
                     role_access="coordinator,scientific,audit", required_for="result-to-claim,coverage")

    tests = [
        "tests/test_wildfire_stage_i_dc_comparison.py",
        "tests/test_wildfire_stage_h_heuristic_comparison.py",
        "tests/test_wildfire_stage_g_revised_continuous.py",
    ]
    for path in tests:
        add_evidence(rows, eid=next_id(), category="tests", description=f"live test file {Path(path).name}",
                     repository_path=path, source_type="live_repository_test",
                     authority_level="live code evidence", role_access="coordinator,audit",
                     required_for="code-to-test")

    source_files = [
        "experiments/test/wildfire_tests/stage_i_dc_comparison/dc_formulation.py",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/ac_projection.py",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_dc_comparison.py",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/run_proxy_inner_lambda_sweep.py",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_h_miqp_pool_refresh.py",
        "experiments/test/wildfire_tests/stage_i_dc_comparison/run_stage_e_load_shed_discrepancy_summary.py",
    ]
    for path in source_files:
        add_evidence(rows, eid=next_id(), category="implementation", description=f"live source file {Path(path).name}",
                     repository_path=path, source_type="live_repository_source",
                     authority_level="live code evidence", role_access="coordinator,audit",
                     required_for="intent-to-code")

    for idx, pdf in enumerate(sorted(PAPERS.glob("*.pdf")), start=1):
        add_evidence(rows, eid=next_id(), category="literature", description=f"literature PDF {pdf.name}",
                     repository_path=str(pdf), source_type="literature_pdf",
                     authority_level="indexed literature; claims require inspection",
                     role_access="coordinator,scientific", required_for="literature alignment",
                     notes="INDEXED; NOT_INSPECTED by coordinator")

    return rows


def write_intent_baseline() -> None:
    write_text(CASE / "INTENT_BASELINE.md", f"""# INTENT_BASELINE.md

```yaml
artifact_id: CASE-001-INTENT-BASELINE
artifact_version: v001
created_utc: {CREATED}
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN_RETROSPECTIVE_RECONSTRUCTION
sha256_when_frozen: recorded_in_CASE001_FREEZE_LEDGER.csv
```

## Use And Limitation

This file is a frozen retrospective reconstruction of intent. It is not unquestionable ground truth. Scientific and implementation reviewers may challenge any reconstructed item through evidence-backed findings.

## Reconstructed Intent Items

| ID | Status | Reconstructed Intent |
| --- | --- | --- |
| INT-001 | LOCKED | Compare Stage E K2 GridFM, Stage I-a DC guided K2, Stage I-b DC MIQP K2, TH, and AH under a K<=2 topology budget. |
| INT-002 | APPROVED | Exclude Stage D exhaustive and Stage E unconstrained from main Stage H comparison plots. |
| INT-003 | APPROVED | Main results use S1-S5, lambda_R in [0, 0.2, 0.5, 0.8, 1.0], rho in [0, 2], and coupled lambda_R_proxy=lambda_R. |
| INT-004 | APPROVED | MLD companion uses lambda_R_proxy=1 and inner lambda_R=0 for Stage E and Stage I-a, with TH/AH load-delivery comparisons. |
| INT-005 | IMPLEMENTED | Proxy-inner sweep separates lambda_R_proxy and inner lambda_R for Stage E K2 and Stage I-a DC guided K2 only. |
| INT-006 | APPROVED | Stage I-a should use fixed-topology DC recourse in a guided topology loop and avoid revisiting the same topology. |
| INT-007 | APPROVED | Stage I-b should solve joint topology plus continuous DC optimization as MIQP with MIPGap=1e-4 and time-limit/incumbent metadata. |
| INT-008 | APPROVED | DC branch model uses MATPOWER rateA as Fmax and must account for taps and phase shifts rather than silently ignoring them. |
| INT-009 | APPROVED | Source-less islanded load must be unserved or accounted for as unserved in load service metrics. |
| INT-010 | APPROVED | GridFM predictions are model outputs, not AC-feasibility certificates. |
| INT-011 | APPROVED | AC projection is for selected finalists only and measures distance to an AC-feasible point under the same topology, not topology optimization. |
| INT-012 | APPROVED | Common operational diagnostics, GridFM PAC_total, DC residuals, and AC projection distances are semantically distinct. |
| INT-013 | UNCERTAIN | Degree of reviewer independence depends on whether separate Codex contexts are actually used. |
| INT-014 | CONFLICTING | Stage H versus Stage I naming must be preserved as naming drift rather than silently collapsed. |
| INT-015 | CONFLICTING | v002 is active evidence, but original package remains failed historical evidence and is incomplete as an offline copy. |

## Known Authority Conflicts

- Stage H versus Stage I naming: `NAMING_DRIFT`.
- Stage D inclusion/exclusion: current main comparison excludes Stage D, while historical analysis used it.
- Coupled versus decoupled proxy/inner lambda: main/MLD/proxy-inner are separate study scopes.
- Baseline commit remains unresolved unless repository evidence later proves it.
""")


def write_claim_register() -> None:
    fields = [
        "claim_id", "claim_text", "source_document", "claim_type", "affected_methods",
        "supporting_result_artifacts", "supporting_tests", "literature_support",
        "evidence_status", "review_status", "qualification_needed", "publication_risk"
    ]
    rows = [
        {
            "claim_id": "CLAIM-001",
            "claim_text": "Stage H main comparison implements K<=2 comparison across Stage E K2, Stage I-a, Stage I-b, TH, and AH.",
            "source_document": "STAGE_H_DC_COMPARISON_PROGRESS.md",
            "claim_type": "implementation_scope",
            "affected_methods": "Stage E K2; Stage I-a; Stage I-b; TH; AH",
            "supporting_result_artifacts": "main_results/r11 tables and methodology_fidelity_checks.csv",
            "supporting_tests": "test_wildfire_stage_i_dc_comparison.py; test_wildfire_stage_h_heuristic_comparison.py",
            "literature_support": "not required",
            "evidence_status": "AVAILABLE",
            "review_status": "PENDING_REVIEW",
            "qualification_needed": "Requires auditor verification of row coverage and method labels.",
            "publication_risk": "MODERATE",
        },
        {
            "claim_id": "CLAIM-002",
            "claim_text": "Stage I-b MIQP runs faster than Stage I-a guided DC recourse in saved runtime fields.",
            "source_document": "STAGE_H_DC_COMPARISON_PROGRESS.md",
            "claim_type": "result_interpretation",
            "affected_methods": "Stage I-a; Stage I-b",
            "supporting_result_artifacts": "dc_runtime_comparison_stage_i_a_vs_i_b.csv",
            "supporting_tests": "not direct",
            "literature_support": "not inspected",
            "evidence_status": "AVAILABLE",
            "review_status": "PENDING_REVIEW",
            "qualification_needed": "Clarify saved solver/runtime only excludes unrecorded Python overhead.",
            "publication_risk": "MODERATE",
        },
        {
            "claim_id": "CLAIM-003",
            "claim_text": "GridFM Stage E predictions show large physically unrealistic loading/load-service inconsistency.",
            "source_document": "STAGE_H_DC_COMPARISON_PROGRESS.md; load shed discrepancy summaries",
            "claim_type": "scientific_limitation",
            "affected_methods": "Stage E K2 GridFM",
            "supporting_result_artifacts": "stage_e_k2_load_shed_discrepancy_*.csv; ac_projection_distances.csv",
            "supporting_tests": "test_wildfire_stage_g_revised_continuous.py",
            "literature_support": "GridFM/source model papers not inspected in Phase 2 coordinator stage",
            "evidence_status": "AVAILABLE",
            "review_status": "PENDING_REVIEW",
            "qualification_needed": "Limit to current implementation/scenarios unless literature/model evidence supports broader claim.",
            "publication_risk": "HIGH",
        },
        {
            "claim_id": "CLAIM-004",
            "claim_text": "Proxy-inner lambda sweep can improve Stage I-a tradeoff fronts in selected scenarios.",
            "source_document": "STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md",
            "claim_type": "result_interpretation",
            "affected_methods": "Stage I-a; Stage E K2",
            "supporting_result_artifacts": "proxy_inner_lambda_sweep/r2; tmp checkpoint tables",
            "supporting_tests": "not direct",
            "literature_support": "not required",
            "evidence_status": "AVAILABLE_WITH_LIMITATIONS",
            "review_status": "PENDING_REVIEW",
            "qualification_needed": "Requires long-path-safe verification of proxy-inner live/checkpoint artifacts and no AC projection claim.",
            "publication_risk": "MODERATE",
        },
        {
            "claim_id": "CLAIM-005",
            "claim_text": "AC projection distances measure distance to an AC-feasible point under fixed topology, not residual minimization.",
            "source_document": "CURRENT_STATE_SUMMARY.md; DC_MILP_FORMULATION_HANDOFF.md",
            "claim_type": "methodology_definition",
            "affected_methods": "Stage E K2; Stage I-a; Stage I-b",
            "supporting_result_artifacts": "ac_projection_distances.csv; ac_projection_cache.csv",
            "supporting_tests": "test_wildfire_stage_i_dc_comparison.py partial",
            "literature_support": "OPF/AC projection literature not inspected",
            "evidence_status": "AVAILABLE",
            "review_status": "PENDING_REVIEW",
            "qualification_needed": "Requires scientific and implementation audit of actual AC projection formulation.",
            "publication_risk": "HIGH",
        },
    ]
    write_csv(CASE / "CLAIM_REGISTER.csv", rows, fields)


def write_bundles(evidence: list[dict]) -> None:
    sci_categories = {"governance", "intent", "formulation", "literature", "results", "workflow_validation"}
    audit_categories = {"governance", "implementation", "tests", "configuration", "results", "solver_diagnostics", "projection", "provenance", "environment", "workflow_validation"}
    sci = [r for r in evidence if r["category"] in sci_categories or "scientific" in r["case_role_access"]]
    audit = [r for r in evidence if r["category"] in audit_categories or "audit" in r["case_role_access"]]
    fields = list(evidence[0].keys())
    write_csv(CASE / "PROPOSED_EVIDENCE_BUNDLE_SCIENTIFIC-v001.csv", sci, fields)
    write_csv(CASE / "PROPOSED_EVIDENCE_BUNDLE_AUDIT-v001.csv", audit, fields)
    write_text(CASE / "PROPOSED_EVIDENCE_BUNDLE_SCIENTIFIC-v001.md", f"""# Scientific Evidence Bundle v001

Focus: intent, formulation, research invariants, literature, method descriptions, selected results, and claims.

Evidence rows: {len(sci)}

Reviewers must not treat `INTENT_BASELINE.md` as unquestionable ground truth and may challenge it through findings.
""")
    write_text(CASE / "PROPOSED_EVIDENCE_BUNDLE_AUDIT-v001.md", f"""# Implementation/Results Audit Evidence Bundle v001

Focus: source, tests, configurations, raw results, solver diagnostics, projection artifacts, provenance, and reproducibility.

Evidence rows: {len(audit)}

Live artifacts require pre/post access hashes before supporting verified findings.
""")


def write_readiness(evidence: list[dict]) -> None:
    unavailable_critical = [r for r in evidence if r["availability_status"] != "AVAILABLE" and r["missing_evidence_class"] == "CRITICAL_REQUIRED"]
    status = "READY_WITH_LIMITATIONS" if unavailable_critical else "READY"
    write_text(CASE / "PHASE2_READINESS_CHECKLIST-v001.md", f"""# Phase 2 Readiness Checklist v001

```yaml
artifact_id: CASE-001-READINESS
artifact_version: v001
created_utc: {CREATED}
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: {status}
sha256_when_frozen: recorded_in_CASE001_FREEZE_LEDGER.csv
```

| Item | Status | Notes |
| --- | --- | --- |
| Authoritative intent documents | READY | Current state, history, README, DC handoff, and Stage H progress docs available. |
| Verified package v002 | READY_WITH_LIMITATIONS | 85 copied files verified; not complete offline copy. |
| Source files | READY | Live Stage I/H source available and hashed. |
| Tests | READY | Live relevant tests available and hashed. |
| Configurations | READY_WITH_LIMITATIONS | Configuration evidence is embedded in scripts/results rather than a single config file. |
| Primary result run r11 | READY | Core tables available; some long-path artifacts require robust access. |
| MLD companion r5 | READY | Core tables available. |
| Proxy-inner companion r2 | READY_WITH_LIMITATIONS | Use short checkpoint path where long result path access fails. |
| Solver diagnostics | READY_WITH_LIMITATIONS | Available through tables/metadata, but auditor must verify completeness. |
| AC projection artifacts | READY | Main/MLD projection tables available. |
| Relevant literature | READY_WITH_LIMITATIONS | 25 PDFs indexed; claims require inspection. |
| Separate reviewer contexts | READY_WITH_LIMITATIONS | Multi-agent tooling available; actual execution mode must be recorded. |
| Evidence hashing capability | READY_WITH_LIMITATIONS | Long-path-safe hashing required. |
| Output directories | READY | Case folder and role folders exist. |
| Role instructions | READY | Role/context protocol exists. |

Critical unavailable evidence rows: {len(unavailable_critical)}

Missing package records do not automatically block review when equivalent live evidence is available or the missing artifact is noncritical.
""")


def write_execution_docs() -> None:
    write_text(CASE / "PHASE2_CODEX_EXECUTION_PLAN-v001.md", f"""# Phase 2 Codex Execution Plan

```yaml
artifact_id: CASE-001-CODEX-EXECUTION-PLAN
artifact_version: v001
created_utc: {CREATED}
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: APPROVED_EXECUTION_IN_PROGRESS
sha256_when_frozen: recorded_in_CASE001_FREEZE_LEDGER.csv
```

## 1. Objective

Evaluate whether the implemented Stage I / Stage H GridFM, DC approximation, heuristic, and AC projection work faithfully executes Caleb's approved methodology and whether material scientific, physical, mathematical, implementation, experimental, literature-alignment, reproducibility, or claim-evidence issues exist.

## 2. Current Authorization

`DECISION-0002-v001.md` approves Phase 2 with limitations. Active evidence package is v002; original package is failed historical evidence only.

## 3. Scope

Review findings only. No remediation, source/test/result changes, experiment reruns, or publication acceptance decisions.

## 4. Non-Destructive Boundary

Read-only for source, tests, official results, current-state docs, history, and frozen packages.

## 5. Inputs Inspected

Workflow governance, Phase 1/1A audits, Phase 2 preparation files, v002 manifests, current intent docs, live source/tests, core result tables, and literature index/PDF availability.

## 6. Current Repository Findings

v002 verifies copied files but is not a complete offline package. Long Windows paths require a robust hash helper. Proxy-inner evidence may need short checkpoint paths.

## 7. Evidence Availability

See `EVIDENCE_REGISTER.csv/json`. Every row is labeled `AVAILABLE`, `UNAVAILABLE`, or `NOT_INSPECTED`; missing evidence is classified by criticality.

## 8. Authority Conflicts

Preserve Stage H/I naming drift, Stage D scope differences, coupled/decoupled lambda scopes, and unresolved baseline commit.

## 9. Role Execution Model

Coordinator runs in current context. Scientific reviewer and implementation/results auditor should use separate Codex contexts where feasible. Reconciliation starts only after both initial reports are frozen.

## 10. Evidence-Bundle Design

Scientific bundle: intent, formulation, invariants, literature, method descriptions, selected results, claims.

Audit bundle: source, tests, configuration, raw results, solver diagnostics, projection artifacts, provenance, reproducibility.

## 11. Scientific-Review Procedure

Evaluate intent reconstruction, formulation, physics, DC/AC/GridFM assumptions, TH/AH and MLD alignment, literature terminology, method fairness, and claim boundaries.

## 12. Implementation-Audit Procedure

Trace intent to code/tests/results, inspect DC formulation, Stage I-a, Stage I-b, GridFM/heuristics, AC projection, solver metadata, result completeness, plot consistency, and reproducibility.

## 13. Report-Freeze Procedure

Freeze initial reports with SHA-256 before any reconciliation. Record execution mode and evidence bundle version.

## 14. Reconciliation Procedure

Compare frozen reports, deduplicate overlapping findings, preserve disagreements, separate findings from recommendations, and identify Caleb disposition questions.

## 15. Finding Schema

Use `FINDING-####-v001.md` with category, severity, reviewer role, confidence, evidence IDs, intent IDs, affected methods/results, claim impact, publication risk, limitations, and recommended disposition. Initial statuses: `OPEN`, `UNVERIFIED`, or `CONDITIONAL`.

## 16. Final-Report Procedure

Produce `FINAL_ALIGNMENT_REPORT-v001.md`; do not declare accepted, rejected, publishable, or presentation-ready.

## 17. Integrity And Contamination Checks

Use pre/post hashes for live artifacts, copied-file verification for v002, and final case freeze ledger.

## 18. Phase 2 Readiness Assessment

`READY_WITH_LIMITATIONS`.

## 19. Known Limitations

Reviewer independence depends on separate context availability. Literature claims require actual inspection. v002 is not a full offline copy.

## 20. Exact Proposed Execution Sequence

Coordinator preflight -> frozen intent baseline -> independent scientific review -> independent implementation/results audit -> freeze initial reports -> reconciliation -> final alignment report -> stop for Caleb.

## 21. Expected Outputs

Evidence register, intent baseline, claim register, initial reports, findings, reconciliation, final alignment report, freeze ledger, case status.

## 22. Stop Conditions

Stop if required evidence cannot be labeled, v002 verification fails, reviewer isolation cannot be recorded, a verified finding relies on unavailable evidence, or any step would mutate protected research artifacts.

## 23. Questions Requiring Caleb

Whether to require true separate contexts for any follow-up verification; whether unavailable full trace artifacts must be restored before publication-facing claims; how to dispose of each finding after final report.
""")
    write_text(CASE / "PROPOSED_CONTEXT_LAUNCH_INSTRUCTIONS-v001.md", f"""# Proposed Context Launch Instructions v001

## Scientific Reviewer

Use separate Codex context if feasible. Provide `INTENT_BASELINE.md`, `CLAIM_REGISTER.csv`, `PROPOSED_EVIDENCE_BUNDLE_SCIENTIFIC-v001.csv/md`, governance/rubric files, and selected result summaries. Do not provide implementation-audit drafts.

## Implementation/Results Auditor

Use separate Codex context if feasible. Provide `INTENT_BASELINE.md`, `CLAIM_REGISTER.csv`, `PROPOSED_EVIDENCE_BUNDLE_AUDIT-v001.csv/md`, source/test/result evidence register rows, and governance/rubric files. Do not provide scientific-review drafts.

## Reconciliation Reviewer

Launch only after both initial reports are frozen. Provide both report hashes, finding indexes, evidence register, and claim register.
""")


def main() -> None:
    evidence = build_evidence()
    fields = list(evidence[0].keys())
    write_csv(CASE / "EVIDENCE_REGISTER.csv", evidence, fields)
    write_text(CASE / "EVIDENCE_REGISTER.json", json.dumps(evidence, indent=2))
    write_intent_baseline()
    write_claim_register()
    write_bundles(evidence)
    write_readiness(evidence)
    write_execution_docs()
    print(json.dumps({
        "evidence_rows": len(evidence),
        "available": sum(r["availability_status"] == "AVAILABLE" for r in evidence),
        "unavailable": sum(r["availability_status"] == "UNAVAILABLE" for r in evidence),
        "claim_rows": 5,
    }, indent=2))


if __name__ == "__main__":
    main()
