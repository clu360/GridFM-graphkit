# Final Stage A-I Research Alignment Report

Generated UTC: 2026-07-31T00:59:40+00:00

## One-Minute Overview

The project began as a GridFM wildfire-risk/load-shedding experiment and evolved into a staged comparative workflow for wildfire-aware topology control. The strongest current evidence is the Stage H/I comparison: GridFM-guided decisions, Rhodes-inspired TH/AH heuristics, Stage I-a DC guided recourse, Stage I-b DC MIQP, MLD alignment, proxy-inner lambda exploration, and AC projection diagnostics. The main limitation is that the current work is not a full security-constrained power shutoff formulation; it lacks explicit N-1 contingencies and GridFM predictions show physical-realism issues. The advisor decision is whether to frame the next step as comparative feasibility/decision-quality, GridFM feasibility improvement, or a move toward SC-OPS/robust optimization.

## Ten-Minute Development Narrative

Stage A built the first GridFM wildfire harness. Stage B broadened scenario construction through automatic multigroup risk selection. Stage C introduced a deterministic PSPS baseline. Stage D enumerated K<=2 shutoffs as an interpretable small-budget reference. Stage E replaced pure enumeration with Gurobi-proposed GridFM topology search and explored continuous recourse. Stage F introduced five decision-quality scenarios. Stage G corrected load-service, islanding, raw/evaluated state, and PAC accounting. Stage H compared the revised GridFM formulation to Rhodes-inspired TH/AH heuristics. Stage I added DC guided recourse, a compact DC MIQP, MLD alignment, proxy-inner lambda decoupling, common diagnostics, and AC projection.

The central research question changed from “can GridFM score wildfire-aware operations?” to “how should GridFM-guided wildfire topology decisions be evaluated against established heuristic and optimization baselines under physical feasibility diagnostics?”

## Detailed Technical Synthesis

The project objective family is a wildfire-risk/load tradeoff:

```text
J = lambda_R R_norm + (1 - lambda_R) L_shed
```

GridFM stages later use:

```text
J_true = lambda_R R_norm + (1 - lambda_R) L_shed_hybrid + rho_phys PAC_total
PAC_total = PAC_operational + PAC_AC + PAC_model_consistency
```

Stage I DC methods use DC nodal balance, generator limits, load-service bounds, branch physics, thermal limits, and `sum y_l <= 2`. Stage I-b solves topology and continuous variables jointly as MIQP; Stage I-a uses guided topology proposals with fixed-topology DC recourse. AC projection is a fixed-topology distance-to-feasible-point diagnostic, not a topology optimizer or residual-only metric.

## Stage-By-Stage Alignment

The full stage-level details are recorded in `STAGE_A_I_DEVELOPMENT_MAP.md`. In brief:

- Stage A: foundational, historical, not publication evidence by itself.
- Stage B: scenario/multigroup expansion, partly superseded.
- Stage C: PSPS baseline, Rhodes-adjacent but not optimized/security-constrained.
- Stage D: exhaustive K<=2 reference, useful baseline/ablation.
- Stage E: GridFM-guided topology search, important method precursor and source of later limitations.
- Stage F: five scenario decision-quality surface.
- Stage G: active GridFM evaluator correction and limitation evidence.
- Stage H: heuristic comparison and naming-drift bridge to Stage I outputs.
- Stage I: current active DC/MIQP/projection comparison.

## Literature Alignment

The two Rhodes papers are Tier 1. Rhodes 2021 directly anchors OPS, the risk/load Pareto framing, AH/TH heuristics, and MLD-style heuristic evaluation. Rhodes 2023 anchors SC-OPS and the need for post-contingency security constraints. Our work adapts the OPS lineage but departs by adding GridFM learned-surrogate evaluation, common diagnostics, DC benchmark variants, and AC projection. It should not be described as security-constrained unless explicit contingency modeling is added.

Tier 2 papers support OPF/DC switching and learned-surrogate context. Tier 3 papers are background. Out-of-scope papers are indexed but not forced into the narrative.

## What Has Been Established

- Formulation: wildfire-risk/load tradeoff and K<=2 topology comparison are well documented.
- Implementation: Stage I main/MLD/proxy-inner result families exist with audits.
- Empirical behavior: GridFM physical-realism and load-service discrepancies are material in the current workflow.
- Solver-backed results: Stage I-b MIQP has saved solver metadata and certified rows in the inspected evidence.
- Feasibility conclusions: DC rows are physically consistent under the DC approximation; AC projection is incomplete and relaxed-Qg qualified.
- Comparative conclusions: TH/AH are useful sparse baselines; DC methods give a cleaner feasibility benchmark than GridFM alone.

## What Remains Uncertain

GridFM command-faithfulness, AC projection failure interpretation, Qg-bound treatment, broader scenario generalization, full literature support, complete reproducible packaging, and true security-constrained/contingency modeling remain open.

## Current Contribution Candidates

- Comparative wildfire topology-control evaluation workflow: `PROMISING`.
- GridFM limitation analysis under wildfire topology interventions: `PROMISING`.
- DC approximation and MIQP benchmark against GridFM: `STRONG_WITHIN_SCOPE`.
- Full security-constrained shutoff method: `NOT_YET_SUPPORTED`.
- GridFM-centered operational optimizer: `PRELIMINARY`.

## Framing Options

See `FRAMING_OPTIONS.md`. The safest current framing is comparative feasibility and decision quality. A GridFM-centered publication needs more remediation. A SC-OPS/robust extension is a longer-term route.

## Recommended Professor Discussion Points

1. Is the strongest near-term paper an evaluation/comparison paper?
2. Should full SC-OPS contingency constraints be added before publication claims?
3. Should GridFM be presented as a candidate method or as a motivating limitation?
4. Which experiments are necessary before presentation?
5. Which venue/community is most appropriate after external verification?

## Limitations Of This Synthesis

This synthesis is repository-grounded but not a new experiment. PDF extraction used text layers rather than full visual equation/figure OCR. Tier 2 literature was inspected only at targeted depth. Result claims remain stage-specific and tied to available evidence.

## Registers

- Evidence items registered: 36
- Literature PDFs registered: 25

## Final Status

`CASE002_SYNTHESIS_COMPLETE_AWAITING_CALEB_INTERPRETATION`
