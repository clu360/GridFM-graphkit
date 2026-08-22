# Phase 1A Package Integrity Audit

```yaml
artifact_id: PHASE1A-PACKAGE-INTEGRITY-0001
artifact_version: v001
created_utc: 2026-07-30T23:36:58+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: false
status: FROZEN
sha256_when_frozen: recorded_in_phase1a_freeze_ledger
```


## Result

Post-remediation integrity status: `PASS`

Package version used: `stage_i_workflow_review_package_v002`

## Counts

| Metric | Count |
| --- | ---: |
| Original manifest rows | 442 |
| Copied files verified in v002 | 85 |
| True content-hash mismatches in original copied files | 0 |
| Expected external/indexed artifacts | 11 |
| Unexpected/missing material records | 346 |
| v002 copied-file verification failures | 0 |

## Integrity Files

- `stage_i_workflow_review_package_v002/COPIED_FILE_MANIFEST.csv`
- `stage_i_workflow_review_package_v002/EXTERNAL_ARTIFACT_INDEX.csv`
- `stage_i_workflow_review_package_v002/MISSING_MATERIALS_INDEX.csv`
- `stage_i_workflow_review_package_v002/PACKAGE_FREEZE.json`
- `stage_i_workflow_review_package_v002/PACKAGE_FREEZE_RECEIPT.json`
