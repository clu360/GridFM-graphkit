# RECONCILIATION-CASE001-v001

```yaml
artifact_id: RECONCILIATION-CASE001
artifact_version: v001
created_utc: 2026-07-31T00:27:00+00:00
execution_mode: separate_codex_contexts
blind_context_enforced: true
reviewer_prior_outputs_visible: true
status: FROZEN_RECONCILIATION
sha256_when_frozen: recorded_in_CASE001_FINAL_FREEZE.csv
```

## Frozen Input Reports

- `scientific_review/SCIENTIFIC-REVIEW-CASE001-v001.md`
- `implementation_audit/IMPLEMENTATION-AUDIT-CASE001-v001.md`

Both initial reports were frozen before reconciliation in `integrity/CASE001_INITIAL_REPORTS_FREEZE.csv`.

## Overlapping Findings

| Consolidated Theme | Scientific Finding | Audit Finding | Reconciliation |
| --- | --- | --- | --- |
| GridFM/AC physical-realism limits | `SCI-FINDING-0001`, `SCI-FINDING-0002` | `AUD-FINDING-0003` | Both reports warn against treating model/projection outputs as full AC-operational feasibility. Audit adds the specific Qg-bound limitation. |
| Proxy-inner claims are DC-only | `SCI-FINDING-0004` | `AUD-FINDING-0005` | Both agree proxy-inner results should not be framed as AC-realizability improvements without projection evidence. |
| Archive/evidence limitations | `SCI-FINDING-0005` partly | `AUD-FINDING-0001` | Both require evidence-status transparency. Audit gives stronger package/archive completeness detail. |
| Numeric/prose qualifications | `SCI-FINDING-0002` notes MLD projection count mismatch | `AUD-FINDING-0002` | Both agree live tables should be numeric authority over progress prose. |

## Science-Only Findings

- `SCI-FINDING-0003`: DC methods are acceptable approximations within DC scope.
- `SCI-FINDING-0006`: S1-S5 scenario reuse limits generalization.
- Literature support is not inspected and cannot yet support literature-backed claims.

## Implementation-Only Findings

- `AUD-FINDING-0004`: test coverage is too thin for several core implementation promises.
- `AUD-FINDING-0003`: AC projection uses relaxed Qg bounds, which must qualify feasibility language.

## Disagreements

No direct contradiction was found. The scientific reviewer emphasizes scientific claim boundaries, while the implementation auditor adds implementation-specific qualifications.

## Duplicate Consolidation

Do not double-count `SCI-FINDING-0002` and `AUD-FINDING-0005` as separate proxy-inner AC-evidence failures in summary counts; they are one consolidated limitation with both scientific and implementation evidence.

Do not double-count `SCI-FINDING-0002` and `AUD-FINDING-0003` as identical: the scientific issue is incomplete projection coverage, while the audit issue is the relaxed-Qg formulation of successful projections.

## Unresolved Evidence

- Literature PDFs remain indexed but not inspected.
- Proxy-inner has no AC projection table.
- Some v002 archive records remain missing/historical, although live core evidence exists.
- Dirty worktree state limits commit-only reproducibility.

## Questions Requiring Caleb

1. Should publication-facing language require AC projection reruns for failed finalists and proxy-inner finalists?
2. Should Qg limits be added before describing projection outputs as AC feasible?
3. Should a complete long-path-safe archive be created before sharing evidence externally?
4. Should literature review be performed before using MLD/TH/AH/prior-work framing in slides or writing?
5. Should scenario-generalization experiments be required before meeting/presentation claims?

## Proposed Final Finding Index

| Final ID | Source Findings | Severity | Theme |
| --- | --- | --- | --- |
| FINAL-FINDING-001 | `SCI-FINDING-0001` | HIGH | GridFM surrogate physical-realism limitation. |
| FINAL-FINDING-002 | `SCI-FINDING-0002`, `AUD-FINDING-0005` | HIGH | AC projection/result coverage incomplete; proxy-inner is DC-only. |
| FINAL-FINDING-003 | `AUD-FINDING-0003` | MODERATE | AC projection success uses relaxed Qg bounds. |
| FINAL-FINDING-004 | `AUD-FINDING-0004` | MODERATE | Test coverage gaps for core promises. |
| FINAL-FINDING-005 | `SCI-FINDING-0005`, `AUD-FINDING-0001` | MODERATE | Literature/package evidence limitations. |
| FINAL-FINDING-006 | `SCI-FINDING-0006` | MODERATE | Scenario-generalization limitation. |
| FINAL-FINDING-007 | `AUD-FINDING-0002` | MINOR | Progress prose has numeric drift from live tables. |
| FINAL-FINDING-008 | `SCI-FINDING-0003` | LOW | DC methods are credible DC approximations, not AC validation. |

## Recommendations

Recommendations are possible responses, not implemented remediation:

- Qualify GridFM and projection claims.
- Use live tables as numeric authority.
- Add focused tests for DC/MIQP/projection promises.
- Add Qg limits or explicitly call projections relaxed-Qg.
- Add AC projection for proxy-inner finalists if AC claims are desired.
- Inspect literature before citation-backed prior-work claims.
- Consider scenario-generalization experiments before broad claims.

