# INTENT_BASELINE.md

```yaml
artifact_id: CASE-001-INTENT-BASELINE
artifact_version: v001
created_utc: 2026-07-31T00:02:53+00:00
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
