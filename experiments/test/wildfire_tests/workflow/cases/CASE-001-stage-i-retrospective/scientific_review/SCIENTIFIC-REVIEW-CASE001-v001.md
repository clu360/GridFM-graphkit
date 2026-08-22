# SCIENTIFIC-REVIEW-CASE001-v001

```yaml
artifact_id: SCIENTIFIC-REVIEW-CASE001
artifact_version: v001
created_utc: 2026-07-31T00:20:00+00:00
execution_mode: separate_codex_contexts
blind_context_enforced: true
reviewer_prior_outputs_visible: false
evidence_bundle_version: v001
status: FROZEN_INITIAL_REPORT
sha256_when_frozen: recorded_in_CASE001_INITIAL_REPORTS_FREEZE.csv
```

## Scope

This is a non-destructive scientific retrospective review for `CASE-001-stage-i-retrospective`. The reviewer did not modify files, implement remediation, or inspect implementation-auditor outputs.

## Evidence Inspected

Primary evidence inspected:

- `PROPOSED_EVIDENCE_BUNDLE_SCIENTIFIC-v001.md`
- `PROPOSED_EVIDENCE_BUNDLE_SCIENTIFIC-v001.csv`
- `INTENT_BASELINE.md`
- `CLAIM_REGISTER.csv`
- `REVIEW_RUBRIC.md`
- `RESEARCH_INVARIANTS.md`
- `LITERATURE_INDEX.md`
- `CURRENT_STATE_SUMMARY.md`
- `DC_MILP_FORMULATION_HANDOFF.md`
- `STAGE_H_DC_COMPARISON_PROGRESS.md`
- `STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md`

Referenced result evidence inspected:

- `main_results/r11/tables/methodology_fidelity_checks.csv`
- `main_results/r11/tables/best_by_rho_scenario_lambda_stage.csv`
- `main_results/r11/tables/ac_projection_distances.csv`
- `main_results/r11/tables/dc_branch_model_audit.json`
- `main_results/r11/tables/dc_runtime_comparison_stage_i_a_vs_i_b.csv`
- `main_results/r11/tables/stage_e_k2_load_shed_discrepancy_summary_overall.csv`
- `MLD/r5/tables/methodology_fidelity_checks.csv`
- `MLD/r5/tables/ac_projection_distances.csv`
- `MLD/r5/tables/dc_runtime_comparison_stage_i_a_vs_i_b.csv`
- `proxy_inner_lambda_sweep/r2/finalization_summary.json`
- `proxy_inner_lambda_sweep/r2/tables/best_by_scenario_proxy_inner_stage.csv`
- `proxy_inner_lambda_sweep/r2/tables/pareto_frontier_points_by_proxy.csv`
- `proxy_inner_lambda_sweep/r2/tables/stage_i_a_best_proxy_setting_for_summary_figures.csv`

## Evidence Unavailable / Not Inspected

- Literature PDFs were not inspected. Literature support is therefore `NOT_INSPECTED`.
- Implementation source was not reviewed as implementation evidence.
- Implementation-auditor outputs were not inspected.
- Bundle rows `EVID-0462` and `EVID-0463` point to unavailable `tmp/...` proxy-inner artifacts, but equivalent final live result tables were found and inspected under `proxy_inner_lambda_sweep/r2/tables/`.
- MLD projection count differs slightly between narrative and table evidence: the progress note says `finite projection distances 32 / 60`, while `methodology_fidelity_checks.csv` reports `finite_distances=31/60`.

## Execution Mode

Separate scientific-review context. Read-only inspection only.

## Blindness Limitations

The current review was conducted as blind to implementation-auditor outputs. However, several upstream governance/intent artifacts record `single_context_role_simulation` or `blind_context_enforced: false`. Therefore, blindness is enforced for this review context, not guaranteed for all artifacts used as retrospective evidence.

## Intent Alignment

The main Stage H/I comparison is broadly aligned with reconstructed intent items `INT-001` through `INT-012`.

- Main comparison includes Stage E K2, Stage I-a DC K2, Stage I-b DC MIQP K2, TH top-1/top-2, and AH K2.
- Result rows satisfy `K <= 2`; observed max shutoff count is `2`.
- Stage D exhaustive and Stage E unconstrained are absent from main result checks.
- Main run uses S1-S5, lambda values `[0.0, 0.2, 0.5, 0.8, 1.0]`, and rho values `[0.0, 2.0]`.
- Corrected MLD run uses `lambda_R_proxy=1.0` and `lambda_R=0.0`.
- DC branch audit records `rateA`-based branch model review and trivial taps/shifts for this MATPOWER case.

Intent caveat: `INTENT_BASELINE.md` is a retrospective reconstruction, and Stage H vs Stage I naming drift remains real. The reviewer does not treat the baseline as unquestionable ground truth.

## Mathematical Assessment

The DC formulation evidence is mathematically coherent as a DC approximation.

- Main methodology checks report DC residuals within tolerance `5e-4`.
- Main max balance residual: `3.52244e-4`.
- Main max angle-flow residual: `1.09275e-4`.
- MLD residuals are much smaller: max balance `1.13687e-13`, max angle-flow `1.78395e-09`.
- Stage I-b selected main rows all report `optimality_certified=True`; max observed MIP gap among selected Stage I-b rows is approximately `8.58e-05`, below `1e-4`.

Mathematical limitation: DC feasibility is not AC feasibility. The evidence respects this distinction in intent and invariants, but publication language must preserve it.

## Physics Assessment

The strongest physics concern is GridFM physical realism under topology/control intervention. Stage E K2 evidence shows:

- Mean `R_norm`: `21.2784`
- Mean `PAC_total`: `51.9530`
- Mean `L_shed_hybrid - L_shed_cmd`: `0.1471`
- Max absolute `L_shed_hybrid - L_shed_cmd`: `0.3820`
- Main AC projection for Stage E K2 solved only `10/50`; `40/50` returned infeasible or residual status.

This supports a limitation claim about the current surrogate/evaluator behavior, not a general indictment of GridFM as a modeling family.

## GridFM Assessment

GridFM Stage E K2 is valid as a surrogate-based workflow, not as an AC-feasibility certificate. The evidence supports the claim that GridFM-predicted loading/load-service behavior can be physically problematic in the current experiment. Claims about Stage E comparative weakness or strength should be framed as behavior of this implementation and scenario set.

## DC Assessment

Stage I-a and Stage I-b are scientifically credible DC baselines within their approximation scope.

- Stage I-a uses many fixed-topology DC recourse solves.
- Stage I-b solves direct DC MIQP with certified selected incumbents.
- Main saved solver/runtime evidence: Stage I-a `10.9501 s` across `2066` recourse solves; Stage I-b `1.7108 s` across `25` MIQP solves; ratio `6.4005x`.
- MLD saved solver/runtime evidence: Stage I-a `1.5883 s`; Stage I-b `0.0602 s`; ratio `26.3650x`.

Runtime claim boundary: these are saved solver/runtime fields, not total workflow wall-clock including all Python/topology/projection overhead.

## AC Projection Assessment

AC projection evidence is useful but incomplete:

- Main run: `160/300` finite projection distances.
- Main Stage E K2: `10/50` solved.
- Main Stage I-a: `40/50` solved.
- Main Stage I-b: `50/50` solved.
- TH/AH rows: only `20/50` solved for each heuristic family.
- MLD run: table evidence reports `31/60` finite distances.
- Proxy-inner sweep has no AC projection table.

Therefore, AC projection supports conditional finalist-distance analysis only. It does not support broad AC-realizability claims for all methods, especially Stage E K2, heuristics, or proxy-inner improvements.

## Heuristic / Literature Assessment

TH/AH are valid sparse K<=2 heuristic comparators in this experiment. However, literature alignment remains unverified because the indexed PDFs were not inspected. The MLD framing may be described as intended to align with maximum-load-delivery style comparisons, but should not be claimed as literature-supported until paper content is inspected.

## Experimental-Design Assessment

The experiment is internally coherent for S1-S5, but generalization is limited.

- The handoff explicitly says fairer `p_env` construction is deferred.
- Current scenarios were retained for continuity.
- The documentation warns that designed target-margin scenarios may overstate heuristic and formulation performance.

Publication claims should therefore be limited to this IEEE-30/MATPOWER-style test setting and the existing five scenarios.

## Claim-Boundary Assessment

- `CLAIM-001`: Supported with limitations. Row coverage, method coverage, K<=2, and exclusion checks are supported.
- `CLAIM-002`: Supported only as saved-solver-runtime comparison; not total runtime.
- `CLAIM-003`: Supported for current Stage E K2 implementation/scenarios; not a broad GridFM literature claim.
- `CLAIM-004`: Supported for DC tradeoff/search design only; no AC projection or universal dominance claim.
- `CLAIM-005`: Conceptually supported as a definition, but empirical projection success is incomplete.

## Findings

### SCI-FINDING-0001

- `category`: `MODEL_CAPABILITY_UNVERIFIED`, `PHYSICALLY_INVALID`
- `severity`: `HIGH`
- `status`: `SUPPORTED`
- `confidence`: `HIGH`
- `evidence_ids or file paths`: `EVID-0011`, `EVID-0456`, `main_results/r11/tables/stage_e_k2_load_shed_discrepancy_summary_overall.csv`, `main_results/r11/tables/ac_projection_distances.csv`
- `intent_requirement_ids`: `INT-010`, `INT-012`, `INV-005`, `INV-008`
- `description`: Stage E K2 GridFM selected rows show substantial load-service discrepancy and high normalized risk/PAC behavior under the current surrogate evaluation.
- `why_it_matters`: It can confound comparisons by making Stage E behavior reflect surrogate physical-realism limits rather than only optimization methodology.
- `affected_methods`: Stage E K2 GridFM
- `affected_results`: Stage E K2 main comparison rows, Stage E K2 load-shed discrepancy tables, AC projection summaries
- `claim_impact`: Qualifies `CLAIM-003`; supports limitation language only.
- `publication_impact`: High; must avoid presenting GridFM predictions as physically certified outputs.
- `recommended_disposition`: Keep claim with strong limitation.
- `required_follow_up`: Validate against AC projection and/or inspected physical diagnostics before stronger claims.
- `limitations`: Scientific review did not inspect implementation code or GridFM model literature.

### SCI-FINDING-0002

- `category`: `EMPIRICALLY_UNCERTAIN_BUT_TESTABLE`, `MISSING_EVIDENCE`
- `severity`: `HIGH`
- `status`: `SUPPORTED`
- `confidence`: `HIGH`
- `evidence_ids or file paths`: `EVID-0456`, `EVID-0460`, `main_results/r11/tables/ac_projection_distances.csv`, `MLD/r5/tables/ac_projection_distances.csv`
- `intent_requirement_ids`: `INT-011`, `INV-006`
- `description`: AC projection is incomplete: main finite distances are `160/300`; MLD finite distances are `31/60`; proxy-inner has no projection evidence.
- `why_it_matters`: AC projection is the primary bridge from DC/GridFM finalists to AC-feasible interpretation.
- `affected_methods`: Stage E K2, Stage I-a, Stage I-b, TH, AH, proxy-inner Stage I-a
- `affected_results`: Projection distance plots/tables and any AC-realizability interpretation
- `claim_impact`: Qualifies `CLAIM-004` and `CLAIM-005`.
- `publication_impact`: High; AC claims must be conditional and row-status-aware.
- `recommended_disposition`: Do not claim general AC feasibility.
- `required_follow_up`: Run or improve AC projection for failed finalists; add proxy-inner finalist projection.
- `limitations`: Projection failures may reflect projection algorithm limits, true infeasibility, or both.

### SCI-FINDING-0003

- `category`: `ACCEPTABLE_APPROXIMATION`
- `severity`: `LOW`
- `status`: `SUPPORTED_WITH_LIMITATIONS`
- `confidence`: `HIGH`
- `evidence_ids or file paths`: `main_results/r11/tables/methodology_fidelity_checks.csv`, `main_results/r11/tables/dc_branch_model_audit.json`, `main_results/r11/tables/best_by_rho_scenario_lambda_stage.csv`
- `intent_requirement_ids`: `INT-006`, `INT-007`, `INT-008`, `INV-003`, `INV-006`, `INV-007`
- `description`: DC methods satisfy the documented K<=2, residual, branch-audit, and solver-certification expectations in the inspected main result evidence.
- `why_it_matters`: This supports treating Stage I-a/I-b as credible DC baselines.
- `affected_methods`: Stage I-a DC K2, Stage I-b DC MIQP K2
- `affected_results`: Main DC comparison rows
- `claim_impact`: Supports `CLAIM-001` and bounded support for `CLAIM-002`.
- `publication_impact`: Positive but conditional on DC approximation scope.
- `recommended_disposition`: Accept DC comparison as a DC baseline, not AC validation.
- `required_follow_up`: Preserve DC-vs-AC distinction in writeup.
- `limitations`: Code-level implementation correctness was not audited here.

### SCI-FINDING-0004

- `category`: `EMPIRICALLY_UNCERTAIN_BUT_TESTABLE`
- `severity`: `MODERATE`
- `status`: `SUPPORTED`
- `confidence`: `HIGH`
- `evidence_ids or file paths`: `EVID-0012`, `proxy_inner_lambda_sweep/r2/finalization_summary.json`, `proxy_inner_lambda_sweep/r2/tables/stage_i_a_best_proxy_setting_for_summary_figures.csv`
- `intent_requirement_ids`: `INT-005`, `INT-006`
- `description`: Proxy-inner lambda decoupling improves or expands Stage I-a DC search evidence in scenario-dependent ways, but does not prove universal dominance or AC-realizability.
- `why_it_matters`: It affects how the method should be presented: as expanded search design, not a guaranteed improvement.
- `affected_methods`: Stage I-a DC K2; Stage E K2 in proxy-inner comparison
- `affected_results`: Proxy-inner sweep `r2`
- `claim_impact`: Qualifies `CLAIM-004`.
- `publication_impact`: Moderate; useful methodological result if bounded correctly.
- `recommended_disposition`: Keep as conditional DC search-design finding.
- `required_follow_up`: Add AC projection for selected proxy-inner finalists.
- `limitations`: No proxy-inner AC projection artifacts were found.

### SCI-FINDING-0005

- `category`: `LITERATURE_TERMINOLOGY_MISMATCH`, `MISSING_EVIDENCE`
- `severity`: `MODERATE`
- `status`: `SUPPORTED`
- `confidence`: `HIGH`
- `evidence_ids or file paths`: `LITERATURE_INDEX.md`, `EVID-0473` through literature rows
- `intent_requirement_ids`: none direct; relevant to MLD/literature framing
- `description`: Literature PDFs are indexed but not inspected, so literature support cannot be claimed.
- `why_it_matters`: MLD and wildfire shutoff framing could otherwise be overclaimed.
- `affected_methods`: MLD companion study, TH/AH heuristic framing, DC shutoff framing
- `affected_results`: Literature-alignment language
- `claim_impact`: Limits literature support fields in `CLAIM_REGISTER.csv`.
- `publication_impact`: Moderate; cite only after inspecting papers.
- `recommended_disposition`: Mark literature support `NOT_INSPECTED`.
- `required_follow_up`: Inspect relevant PDFs before literature-backed claims.
- `limitations`: This review did not open PDF content.

### SCI-FINDING-0006

- `category`: `EMPIRICALLY_UNCERTAIN_BUT_TESTABLE`
- `severity`: `MODERATE`
- `status`: `SUPPORTED`
- `confidence`: `MEDIUM`
- `evidence_ids or file paths`: `CURRENT_STATE_SUMMARY.md`, `DC_MILP_FORMULATION_HANDOFF.md`
- `intent_requirement_ids`: `INT-003`
- `description`: Existing S1-S5 scenario reuse is intentional, but scenario construction/generalization remains limited.
- `why_it_matters`: Results may not generalize beyond the designed IEEE-30 scenario set.
- `affected_methods`: All compared methods
- `affected_results`: Main and MLD comparisons
- `claim_impact`: Limits external validity of all performance comparisons.
- `publication_impact`: Moderate to high depending on claim breadth.
- `recommended_disposition`: Report as small controlled study, not general benchmark.
- `required_follow_up`: Run fairer or less target-obvious `p_env` scenarios before broad claims.
- `limitations`: Raw scenario-generation code was not inspected by the scientific reviewer.

## Recommendations

Keep the main result as `CONDITIONAL`: scientifically useful and mostly intent-aligned, but not publication-ready for broad AC or literature claims.

Before publication-strength claims:

1. Add AC projection for proxy-inner finalists.
2. Resolve or explain projection failures, especially Stage E K2 and heuristic rows.
3. Inspect and cite the relevant literature PDFs directly.
4. State runtime claims as saved solver/runtime comparisons only.
5. Keep GridFM, DC residuals, PAC metrics, and AC projection distances semantically separate.
6. Add scenario-generalization experiments with less constructed environmental-risk patterns.

## Uncertainties

- Whether AC projection failures indicate true AC infeasibility, projection formulation limitations, or solver convergence issues.
- Whether implementation source exactly matches the documented formulation; this review did not inspect implementation code.
- Whether literature alignment is valid; PDFs were not inspected.
- Minor inconsistency in MLD projection counts: `32/60` in progress prose versus `31/60` in inspected table evidence.

## Review Outcome

`CONDITIONAL`

The workflow is scientifically coherent as a controlled K<=2 DC/GridFM/heuristic comparison with strong claim-boundary requirements. It should not yet be treated as broad AC-feasibility evidence, broad GridFM model evidence, or literature-supported wildfire shutoff methodology without the follow-up checks above.

