# Phase 2 Codex Execution Plan

```yaml
artifact_id: CASE-001-CODEX-EXECUTION-PLAN
artifact_version: v001
created_utc: 2026-07-31T00:02:53+00:00
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
