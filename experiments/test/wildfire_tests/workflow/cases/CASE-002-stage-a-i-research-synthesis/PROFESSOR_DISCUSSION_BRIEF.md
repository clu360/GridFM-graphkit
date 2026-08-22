# GridFM Wildfire Research - Advisor Discussion Brief

## Current Objective

Understand whether GridFM-guided wildfire-aware topology decisions can be compared fairly with heuristic and DC optimization methods, and decide the strongest research framing.

## Work Completed

Stages A-I built a progression from first-pass GridFM wildfire scoring, through PSPS/heuristic baselines and topology enumeration, into Stage E GridFM-guided search, Stage G physics/load corrections, Stage H heuristic comparison, and Stage I DC/AC projection benchmarking.

## Strongest Results

The strongest current result is the comparative Stage I evidence: DC methods provide a grounded benchmark, Stage I-b MIQP is certified in saved rows, and GridFM limitations are made visible through load/PAC/projection diagnostics.

## Main Limitations

The current work is not fully security-constrained. It lacks explicit N-1 contingency modeling and post-contingency constraints like Rhodes SC-OPS. AC projection is incomplete and uses relaxed Qg bounds. GridFM predictions show large physical-realism concerns in the current implementation.

## Relationship To Rhodes Papers

Rhodes 2021 anchors OPS, AH/TH heuristics, MLD, and risk/load Pareto framing. Rhodes 2023 anchors SC-OPS and shows why contingency/security constraints matter. Our work follows/adapts the risk-load shutoff lineage but departs by introducing GridFM surrogate evaluation, DC comparison, and AC projection diagnostics.

## Advisor Decisions

1. Should the near-term story be a comparative feasibility/decision-quality study rather than a GridFM-centered optimization claim?
2. Should the next technical step complete AC projection and scenario generalization, or move toward SC-OPS/robust optimization?
3. How strongly should GridFM limitations be foregrounded?
4. Which publication/presentation route is best after targeted cleanup?

## Events And Conferences

Any future deadline or venue recommendation requires `REQUIRES_CURRENT_EXTERNAL_VERIFICATION`.
