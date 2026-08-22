# Validation Results

```yaml
artifact_id: VALIDATION-RESULTS-0001
artifact_version: v001
created_utc: 2026-07-30T03:32:04+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_workflow/validation/FROZEN_ARTIFACTS.csv
```


## Results

| Test | Result | Evidence |
| --- | --- | --- |
| filesystem validation | `PASS` | Required workflow files were created under workflow/. |
| package integrity | `FAIL` | Current package manifest hashes do not match `PACKAGE_FREEZE.json`, and 369 manifest-listed package artifact paths could not be verified. |
| active source/test/result integrity | `PASS` | No monitored original artifact changed during generation. |
| file-routing test | `PASS` | Proposal, review, audit, finding, decision, and reconciliation use separate directories. |
| role-isolation test | `PASS_WITH_LIMITATIONS` | Roles are file-separated but simulated in one Codex context. |
| rubric-contamination test | `PASS` | Rubric instructs reviewers to reject requested favorable scoring without evidence. |
| context-contamination test | `PASS_WITH_LIMITATIONS` | Reviewer metadata records single-context limitation. |
| blind-versus-aware review test | `PARTIAL` | Blind context is documented as not technically enforced. |
| authority-conflict test | `PASS` | Four real project conflicts are classified without silently resolving unresolved authority. |
| literature-isolation test | `PASS` | 25 PDFs indexed as uninspected/unverified; no literature claims asserted. |
| reviewer-independence test | `PASS_WITH_LIMITATIONS` | Initial outputs are separate files, but generated in one context. |
| immutability test | `PASS` | Frozen artifacts are listed in SHA-256 freeze ledger. |
| human-approval-gate test | `PASS` | Status remains AWAITING_CALEB_APPROVAL. |

## Critical Failures

Package integrity verification failed. The workflow directory was created and role mechanics were validated procedurally, but Phase 1 cannot be marked technically ready for Phase 2 until the frozen package integrity issue is resolved or Caleb explicitly accepts the limitation.

## Limitations

Single-context role simulation does not provide true blind review, runtime isolation, separate credentials, or OS-level write prevention.

The package integrity failure appears in the pre- and post-snapshots, while monitored active source/test/result artifacts show zero Phase 1 changes.
