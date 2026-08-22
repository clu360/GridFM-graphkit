# Stage A-I Development Map

Generated UTC: 2026-07-31T00:59:40+00:00

## Stage A - First-pass connected-corridor GridFM wildfire experiments

- historical_stage_name: `stage_a_first_pass`
- current_reconstructed_name: `First-pass connected-corridor GridFM wildfire experiments`
- approximate_development_period: `May 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: overlaps early demand-weighted objective refactors
- development_outcome: `MODIFIED`

### Research purpose
Establish the first GridFM wildfire-aware optimization harness and see whether a risk/load objective could be evaluated at all.

### Motivation
The project needed a working end-to-end surface before deeper wildfire shutoff claims could be made.

### Inputs
IEEE-30 processed GridFM tensors, initial wildfire corridor definitions, GridFM surrogate predictions.

### Decision variables or evaluated quantities
`u` controls, grouped wildfire risk, demand-weighted load shedding.

### Methodology
Run fixed-topology GridFM experiments and score wildfire/load objectives.

### Objective and constraints
`J = lambda_R R_norm + (1-lambda_R) L_shed`, with later stages adding `rho_phys PAC_total` or DC-specific exact recourse.

### Outputs
Connected-corridor summaries, figures, and first-pass result artifacts.

### Main findings
The harness worked but was too preliminary for publication-level conclusions.

### Limitations
Not a topology-control or security-constrained study.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Conceptual only. This stage is upstream harness/scenario construction rather than a direct Rhodes OPS/SC-OPS analogue.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `FOUNDATIONAL`; publication value: `HISTORICAL_DEVELOPMENT`.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage B - Automatic multigroup wildfire-risk selection and multistart sensitivity

- historical_stage_name: `stage_b_multigroup`
- current_reconstructed_name: `Automatic multigroup wildfire-risk selection and multistart sensitivity`
- approximate_development_period: `Late May-June 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: extends Stage A selection/scenario construction
- development_outcome: `PARTIALLY_SUPERSEDED`

### Research purpose
Replace manual single-corridor thinking with automatic multigroup risk selection and threshold sensitivity.

### Motivation
Manual risk corridors were too brittle for general claims.

### Inputs
Stage A harness, automatic line-risk scores, threshold fractions, GNN/GPS model variants.

### Decision variables or evaluated quantities
selected risk components, group membership, multistart seeds, objective values.

### Methodology
Rank lines by automatic risk scores, form components, and run multistart sensitivity.

### Objective and constraints
`J = lambda_R R_norm + (1-lambda_R) L_shed`, with later stages adding `rho_phys PAC_total` or DC-specific exact recourse. Constraints were mostly implicit through selected groups and optimizer bounds.

### Outputs
Threshold sensitivity summaries, automatic group CSV/JSON artifacts.

### Main findings
Automatic risk components avoided manual-only grouping but did not yet answer topology-control quality.

### Limitations
Threshold choices and synthetic risk remain exploratory.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Conceptual only. This stage is upstream harness/scenario construction rather than a direct Rhodes OPS/SC-OPS analogue.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `PARTIALLY_SUPERSEDED`; publication value: `SUPPORTING_METHOD`.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage C - Deterministic PSPS threshold baseline

- historical_stage_name: `stage_c_psps_baseline`
- current_reconstructed_name: `Deterministic PSPS threshold baseline`
- approximate_development_period: `Late May-June 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: shares risk ranking with Stage D/H heuristic baselines
- development_outcome: `RETAINED`

### Research purpose
Create a deterministic PSPS-style threshold baseline for comparison against optimization methods.

### Motivation
A PSPS-like baseline was needed because de-energization is a standard operational wildfire mitigation action.

### Inputs
Automatic candidate sets, fixed demand-weighted consequence score, PSPS thresholds.

### Decision variables or evaluated quantities
PSPS selected lines and post-PSPS GridFM service fraction.

### Methodology
Deactivate top-risk PSPS lines, run GridFM post-topology evaluation, and summarize risk/load.

### Objective and constraints
Threshold PSPS has no optimization after line selection except diagnostic load-service evaluation.

### Outputs
PSPS summaries, selected PSPS lines, risk/load diagnostics.

### Main findings
PSPS thresholds can sharply reduce risk but may cause large service impacts.

### Limitations
Not optimized over load/risk jointly and not N-1 secure.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `BASELINE`; publication value: `BASELINE`.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage D - Limited enumerated K<=2 de-energization reference

- historical_stage_name: `stage_d_deenergization`
- current_reconstructed_name: `Limited enumerated K<=2 de-energization reference`
- approximate_development_period: `Late May-June 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: serves as exhaustive K<=2 reference for early Stage E/G
- development_outcome: `PARTIALLY_SUPERSEDED`

### Research purpose
Enumerate small-budget line de-energization choices to create an interpretable K<=2 reference.

### Motivation
A single PSPS threshold could not reveal whether better small-budget shutoff sets existed.

### Inputs
Stage C candidate construction, K<=2 subset enumeration, GridFM post-topology predictions.

### Decision variables or evaluated quantities
de-energized line subset `S_off` with `|S_off| <= 2`, `R_norm`, `L_shed`, `J`.

### Methodology
Evaluate all candidate subsets of size 0, 1, or 2 and select best by lambda-weighted objective.

### Objective and constraints
Enumerated objective `J = lambda_R R_norm + lambda_L L_shed` over `|S_off| <= 2`.

### Outputs
Candidate evaluation tables, optimized de-energization decisions, Stage D summary.

### Main findings
Small-budget enumeration provided a strong reference inside the candidate space.

### Limitations
Enumeration is small-budget and fixed-control; not scalable.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `BASELINE`; publication value: `ABLATION`.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage E - Gurobi-proposed GridFM topology search and continuous-control variants

- historical_stage_name: `stage_e_gurobi_implementation`
- current_reconstructed_name: `Gurobi-proposed GridFM topology search and continuous-control variants`
- approximate_development_period: `June 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: straddles topology search and GridFM recourse experiments
- development_outcome: `MODIFIED`

### Research purpose
Use a Gurobi proxy master to propose topology candidates that are evaluated by GridFM, including constrained and unconstrained variants.

### Motivation
Enumeration was not a scalable research path, so topology search needed a guided proposal mechanism.

### Inputs
Stage D/E candidate lines, Gurobi proxy model, GridFM true evaluator, lambda settings.

### Decision variables or evaluated quantities
binary line states proposed by Gurobi proxy, optional continuous controls `Delta_Pg` and `alpha`, GridFM-evaluated objective.

### Methodology
Use Gurobi as proxy topology proposer, then evaluate candidates through GridFM true objective; later add continuous recourse call-budget studies.

### Objective and constraints
`J_true = lambda_R R_norm + lambda_L L_shed + rho_phys PAC_total`; Gurobi proxy uses no-good cuts to propose topology candidates.

### Outputs
Gurobi/GridFM run folders, proxy-vs-true summaries, continuous objective traces, alpha diagnostics.

### Main findings
Guided topology search could match or miss exhaustive K<=2 depending on rho/lambda; continuous recourse was costly and sometimes barely moved controls.

### Limitations
GridFM true evaluation is only as reliable as GridFM predictions; no hard AC/security constraints.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `PARTIALLY_SUPERSEDED`; publication value: `MOTIVATING_FAILURE` and method precursor.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage F - Five scenario decision-quality suite

- historical_stage_name: `stage_f_decision_quality`
- current_reconstructed_name: `Five scenario decision-quality suite`
- approximate_development_period: `June 2026`
- confidence_in_stage_boundary: `MODERATE`
- overlap_with_other_stages: scenario definitions feed Stage G/H/I
- development_outcome: `RETAINED`

### Research purpose
Construct five decision-quality scenarios to stress whether topology choices behave consistently across different target structures.

### Motivation
Early cases risked being overfit to one risk construction.

### Inputs
Existing Stage C/D/E methods and five scenario definitions S1-S5.

### Decision variables or evaluated quantities
scenario target margins, expected target lines, Stage D/E objective gaps.

### Methodology
Evaluate Stage C/D/E methods across five controlled decision-quality scenarios.

### Objective and constraints
Same Stage C/D/E objective family, applied to five scenario surfaces.

### Outputs
Decision-quality scenario definitions and Stage D/E comparison outputs.

### Main findings
Scenario structure matters; method performance should not be inferred from one case.

### Limitations
Five scenarios are controlled, not broad generalization.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Conceptual only. This stage is upstream harness/scenario construction rather than a direct Rhodes OPS/SC-OPS analogue.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `FOUNDATIONAL`; publication value: `SUPPORTING_METHOD`.

### Confidence assessment
- reconstructed purpose: `MODERATE`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage G - Physics/load/PAC corrections and revised continuous GridFM evaluator

- historical_stage_name: `stage_g_implementation_revision`
- current_reconstructed_name: `Physics/load/PAC corrections and revised continuous GridFM evaluator`
- approximate_development_period: `Late June-July 2026`
- confidence_in_stage_boundary: `HIGH`
- overlap_with_other_stages: corrects Stage E/F evaluation semantics
- development_outcome: `RETAINED`

### Research purpose
Correct and deepen GridFM evaluation semantics: source-less island service, commanded versus raw predictions, hybrid load, and PAC decomposition.

### Motivation
Observed load-service and physics inconsistencies made prior objective values too optimistic or under-specified.

### Inputs
Stage F scenarios, GridFM raw outputs, MATPOWER metadata, revised source-less island and PAC logic.

### Decision variables or evaluated quantities
`x_raw`, `x_eval`, `L_shed_cmd`, `L_shed_gridfm_raw`, `L_shed_gridfm_effective`, `L_shed_hybrid`, PAC groups.

### Methodology
Separate commanded/evaluated/raw states, correct islanded load, add hybrid load metric and PAC operational/AC/model consistency groups.

### Objective and constraints
`J_true = lambda_R R_norm + lambda_L L_shed_hybrid + rho_phys(PAC_operational + PAC_AC + PAC_model_consistency)`.

### Outputs
Revised continuous tables, cross-rho/per-rho/summary plots, methodology checks.

### Main findings
Prior metrics understated load loss and physics/model inconsistency; hybrid load and PAC decomposition became necessary.

### Limitations
Diagnostics reveal issues but do not solve surrogate feasibility.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Conceptual only. This stage is upstream harness/scenario construction rather than a direct Rhodes OPS/SC-OPS analogue.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `ACTIVE_METHOD`; publication value: `LIMITATION_ANALYSIS`.

### Confidence assessment
- reconstructed purpose: `HIGH`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage H - Heuristic baseline and later DC-comparison result framing

- historical_stage_name: `stage_h_heuristic_comparison and Stage H result folders`
- current_reconstructed_name: `Heuristic baseline and later DC-comparison result framing`
- approximate_development_period: `July 2026`
- confidence_in_stage_boundary: `MODERATE`
- overlap_with_other_stages: name overlaps with Stage I DC comparison outputs
- development_outcome: `MODIFIED`

### Research purpose
Compare formulation-based GridFM choices against Rhodes-inspired TH/AH heuristics, then house the later DC-comparison result family.

### Motivation
The work needed paper-inspired heuristic baselines and clearer decision-quality visualizations.

### Inputs
Stage G revised continuous evaluator, target line sets, baseline loading scores, TH/AH heuristic definitions.

### Decision variables or evaluated quantities
TH top-k lines, AH connected group, expected-vs-selected hits, recall, precision, GridFM recourse metrics.

### Methodology
Run TH top-k and AH connected heuristics through the Stage G evaluator and compare against Stage G/Stage E references.

### Objective and constraints
TH/AH use the Stage G recourse evaluator; recall/precision are diagnostic, not optimized.

### Outputs
Heuristic comparison top-k and revised-load/PAC result folders, expected-vs-selected plots, TH/AH audits.

### Main findings
TH can be competitive in structured scenarios; AH is often weaker; revised load/PAC exposed larger GridFM-implied degradation.

### Limitations
Heuristic adaptations are Rhodes-inspired but not exact geographic RTS reproductions; some references have accounting caveats.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `ACTIVE_METHOD`; publication value: `BASELINE` and comparative evidence.

### Confidence assessment
- reconstructed purpose: `MODERATE`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`

## Stage I - DC approximation, MIQP, AC projection, MLD, proxy-inner comparison

- historical_stage_name: `stage_i_dc_comparison`
- current_reconstructed_name: `DC approximation, MIQP, AC projection, MLD, proxy-inner comparison`
- approximate_development_period: `July 2026`
- confidence_in_stage_boundary: `MODERATE`
- overlap_with_other_stages: implemented in stage_i_dc_comparison but results live under Stage H path
- development_outcome: `RETAINED`

### Research purpose
Add DC approximation baselines, direct MIQP, MLD alignment, proxy-inner lambda exploration, and AC projection diagnostics.

### Motivation
GridFM physical-realism concerns motivated transparent DC optimization and AC feasibility/projection diagnostics.

### Inputs
Stage G/H GridFM evaluator, MATPOWER `rateA`, DC branch data, Stage E-style topology loops, Gurobi MIQP, AC projection backend.

### Decision variables or evaluated quantities
Stage I-a fixed-topology DC variables `theta,f,Pg,s`; Stage I-b binary `z,y` plus continuous DC variables; projection distances.

### Methodology
Run Stage E K2 GridFM, Stage I-a guided DC recourse, Stage I-b DC MIQP, TH/AH, MLD, proxy-inner sweep, and AC projection for selected finalists.

### Objective and constraints
Stage I-a/I-b DC objective minimizes normalized squared line-risk flow plus load shedding under DC balance, generator/load bounds, thermal limits, topology budget `sum y <= 2`; AC projection minimizes distance to fixed-topology AC-feasible point where possible.

### Outputs
Main r11, MLD r5, proxy-inner r2 tables/plots; solver diagnostics; projection distances; CASE-001 audit findings.

### Main findings
DC methods make the comparison more physically grounded; Stage I-b is compact and certified in saved rows; GridFM limitations remain central; projection evidence is incomplete.

### Limitations
DC approximation is not AC truth; AC projection finite only for subset and uses relaxed Qg bounds; no full SC-OPS contingency model.

### Relationship to prior and later stages
See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison.

### Relationship to Rhodes wildfire shutoff literature
Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS.

### Broader literature alignment
Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only.

### Current status and publication value
Status: `ACTIVE_METHOD`; publication value: `CORE_CONTRIBUTION` candidate with limitations.

### Confidence assessment
- reconstructed purpose: `MODERATE`
- implementation description: `HIGH`
- result interpretation: `MODERATE`
- literature alignment: `MODERATE`
- current role in project: `MODERATE`
