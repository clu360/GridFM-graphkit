# Intent Reconstruction

This reconstruction uses repository evidence only. It does not certify that the
implementation is correct.

## Sources inspected

- `CURRENT_STATE_SUMMARY.md` [HISTORICAL]
- `HISTORY.md` [HISTORICAL]
- `DC_MILP_FORMULATION_HANDOFF.md` [LOCKED/HISTORICAL mixed]
- `STAGE_H_DC_COMPARISON_PROGRESS.md` [IMPLEMENTED/current]
- `STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md` [IMPLEMENTED/current]

## Intended research question

[LOCKED] Compare the current GridFM-guided wildfire-aware methodology against DC
approximation methods and baseline heuristics under a common sparse K<=2 setting.

## Intended method families

[LOCKED] Stage E K2 GridFM, Stage I-a DC guided K2, Stage I-b DC MIQP K2,
TH top-1/top-2, and AH K2 budgeted.

## Stage E K2 GridFM role

[IMPLEMENTED] Guided proxy/no-good topology search with GridFM-linked continuous
recourse/evaluation. GridFM remains a learned surrogate evaluator, not a hard
AC solver.

## Stage I-a intended role

[LOCKED] Separate Stage-E-style guided topology proposal loop, followed by
fixed-topology DC continuous recourse.

## Stage I-b intended role

[LOCKED] Direct joint topology and DC continuous MIQP with `sum y_l <= 2`,
`MIPGap=1e-4`, and saved solver metadata.

## TH/AH heuristic role

[LOCKED] Budget-compatible heuristic comparison points. They are not intended to
be optimization-equivalent to Stage E or Stage I.

## AC projection role

[LOCKED] Finalist-only projection to nearest AC-feasible point under the same
topology, with nonfatal failures logged. [IMPLEMENTED] Main and MLD runs include
projection tables. [UNCERTAIN] Proxy-inner sweep does not yet include projection.

## Common evaluator role

[LOCKED] Provide cross-method comparable diagnostics, avoiding direct semantic
mixing of GridFM PAC and DC feasibility.

## Locked scenarios

[LOCKED] Existing S1-S5 decision-quality scenarios.

## Locked topology budget

[LOCKED] K<=2. Stage D exhaustive and unconstrained variants excluded from main
Stage H/Stage I comparison.

## Lambda and rho settings

[LOCKED] Main run uses lambda_R in `[0, 0.2, 0.5, 0.8, 1]` and rho in `[0, 2]`.
[IMPLEMENTED] Main uses coupled `lambda_R_proxy=lambda_R`.
[IMPLEMENTED] Corrected MLD uses `lambda_R_proxy=1`, `lambda_R=0`.
[IMPLEMENTED] Proxy-inner sweep separates proxy and inner lambdas for Stage E K2
and Stage I-a only, with rho=0.

## Shared normalization rules

[LOCKED] Use stored scenario baseline loading denominator and MATPOWER `rateA`
for line ratings.

## Required solver diagnostics

[LOCKED] Stage I-b must save solver status, objective, best bound, MIP gap,
runtime, node count, solution count, time-limit status, and certification flag.

## Required physics and topology semantics

[LOCKED] Canonical physical branch IDs, tap/shift audit, offline flow behavior,
source-less island handling, load service bounds, generator bounds, nodal
balance, and DC branch equations.

## Required outputs

[IMPLEMENTED] Main, corrected MLD, proxy-inner summary outputs, plots, and
methodology checks are present in the result anchors indexed by this package.

## Explicitly excluded or deferred items

[LOCKED] Stage D exhaustive excluded from main comparison. Unconstrained/higher-K
variants deferred. [DEFERRED] Proxy-inner AC projection distances are not yet
generated.

## Known plan changes over time

[IMPLEMENTED] MLD was corrected from coupled lambda to `lambda_R_proxy=1`,
`lambda_R=0`. Proxy-inner sweep was added to test decoupled topology and inner
objective weights.

## Conflicts or ambiguities between planning documents

[UNCERTAIN] Older documents use Stage I naming, while implemented code/result
paths use Stage H and `stage_i_dc_comparison`. See naming relationship note in
`REPOSITORY_SNAPSHOT.md`.

## Most likely current authoritative intent

[INFERRED] Treat `STAGE_H_DC_COMPARISON_PROGRESS.md` as the latest state of the
implemented experiment, while using `DC_MILP_FORMULATION_HANDOFF.md` as the
locked methodology source where it does not conflict with later implemented
updates.

## Confidence and unresolved questions

Confidence is high for run anchors and implemented output locations. Confidence
is lower for baseline commit selection and full literature alignment because
those materials are not yet fully provided.
