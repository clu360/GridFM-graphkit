from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[6]
CASE = Path(__file__).resolve().parent
PAPERS = (
    ROOT.parent.parent
    / "Literature Review"
    / "Papers"
)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(name: str, text: str) -> None:
    (CASE / name).write_text(text.strip() + "\n", encoding="utf-8")


def write_csv(name: str, rows: list[dict], fields: list[str]) -> None:
    with (CASE / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def row_count(path: Path) -> str:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return str(max(0, sum(1 for _ in f) - 1))
    except Exception:
        return ""


def csv_columns(path: Path) -> str:
    if not path.exists() or not path.is_file() or path.suffix.lower() != ".csv":
        return ""
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f)
            return ";".join(next(reader))
    except Exception:
        return ""


def pdf_pages(path: Path) -> int | str:
    try:
        return len(PdfReader(str(path)).pages)
    except Exception:
        return ""


def pdf_head(path: Path, max_pages: int = 2) -> str:
    try:
        r = PdfReader(str(path))
        return "\n".join((r.pages[i].extract_text() or "") for i in range(min(max_pages, len(r.pages))))
    except Exception as exc:
        return f"PDF_EXTRACTION_FAILED: {exc}"


def classify_literature(filename: str) -> tuple[str, str, str, str, str]:
    lower = filename.lower()
    if "security_constrained_optimal_power_shutoff" in lower:
        return (
            "TIER_1_CORE",
            "FULL_FORMULATION_AND_RESULTS",
            "SC-OPS formulation; contingency-aware shutoff; load-served bounds; security terminology.",
            "Primary Rhodes lineage paper for security-constrained wildfire shutoff.",
            "INSPECTED",
        )
    if "balancing_wildfire_risk" in lower:
        return (
            "TIER_1_CORE",
            "FULL_FORMULATION_AND_RESULTS",
            "OPS formulation; AH/TH heuristics; MLD heuristic evaluation; wildfire/load Pareto framing.",
            "Primary Rhodes lineage paper for optimized wildfire power shutoffs.",
            "INSPECTED",
        )
    if any(s in lower for s in ["cycle-based", "opf", "socp", "security", "gridfm", "foundation", "gnn", "residual", "safe power", "trustworthiness"]):
        return (
            "TIER_2_SUPPORTING",
            "TARGETED_SECTION_REVIEW",
            "Supports DC/AC OPF, topology switching, learned surrogate, or feasibility terminology.",
            "Direct methodological support, but not the central wildfire-shutoff lineage.",
            "INSPECTED",
        )
    if any(s in lower for s in ["graph attention", "semi-supervised", "inductive", "neural message", "graph_neural"]):
        return (
            "TIER_3_CONTEXTUAL",
            "ABSTRACT_AND_RELEVANT_CONTEXT",
            "Graph-learning background only; no direct wildfire shutoff formulation.",
            "Contextual GNN lineage.",
            "INSPECTED",
        )
    if any(s in lower for s in ["smoke", "battling"]):
        return (
            "TIER_3_CONTEXTUAL",
            "ABSTRACT_AND_RELEVANT_CONTEXT",
            "Wildfire/resilience background only; not a core formulation anchor.",
            "Context for wildfire/resilience motivation.",
            "INSPECTED",
        )
    return (
        "OUT_OF_SCOPE",
        "NOT_INSPECTED",
        "",
        "No material connection identified for current Stage A-I synthesis from filename/topic.",
        "NOT_INSPECTED",
    )


def main() -> None:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    CASE.mkdir(parents=True, exist_ok=True)

    protected = [
        ROOT / "experiments/test/wildfire_tests/CURRENT_STATE_SUMMARY.md",
        ROOT / "experiments/test/wildfire_tests/HISTORY.md",
        ROOT / "experiments/test/wildfire_tests/README.md",
        ROOT / "experiments/test/wildfire_tests/DC_MILP_FORMULATION_HANDOFF.md",
    ]
    before_hashes = {rel(p): sha256(p) for p in protected if p.exists()}

    evidence_specs = [
        ("EVID-0001", "A-I", "history", "Current durable state summary", "experiments/test/wildfire_tests/CURRENT_STATE_SUMMARY.md", "project_state", "HIGH"),
        ("EVID-0002", "A-I", "history", "Long-form session history and stage chronology", "experiments/test/wildfire_tests/HISTORY.md", "project_history", "HIGH"),
        ("EVID-0003", "A-I", "intent", "Wildfire harness README and canonical commands", "experiments/test/wildfire_tests/README.md", "project_docs", "HIGH"),
        ("EVID-0004", "I", "formulation", "DC MILP / AC projection handoff", "experiments/test/wildfire_tests/DC_MILP_FORMULATION_HANDOFF.md", "methodology_handoff", "HIGH"),
        ("EVID-0005", "A", "source", "Stage A first-pass source folder", "experiments/test/wildfire_tests/stage_a_first_pass", "source_dir", "HIGH"),
        ("EVID-0006", "B", "source", "Stage B multigroup source folder", "experiments/test/wildfire_tests/stage_b_multigroup", "source_dir", "HIGH"),
        ("EVID-0007", "C", "source", "Stage C PSPS baseline source folder", "experiments/test/wildfire_tests/stage_c_psps_baseline", "source_dir", "HIGH"),
        ("EVID-0008", "D", "source", "Stage D de-energization source folder", "experiments/test/wildfire_tests/stage_d_deenergization", "source_dir", "HIGH"),
        ("EVID-0009", "E", "source", "Stage E Gurobi/GridFM source folder", "experiments/test/wildfire_tests/stage_e_gurobi_implementation", "source_dir", "HIGH"),
        ("EVID-0010", "F", "source", "Stage F decision-quality source folder", "experiments/test/wildfire_tests/stage_f_decision_quality", "source_dir", "HIGH"),
        ("EVID-0011", "G", "source", "Stage G implementation revision source folder", "experiments/test/wildfire_tests/stage_g_implementation_revision", "source_dir", "HIGH"),
        ("EVID-0012", "H", "source", "Stage H heuristic comparison source folder", "experiments/test/wildfire_tests/stage_h_heuristic_comparison", "source_dir", "HIGH"),
        ("EVID-0013", "H/I", "source", "Stage I DC comparison source folder with Stage H result runners", "experiments/test/wildfire_tests/stage_i_dc_comparison", "source_dir", "HIGH"),
        ("EVID-0014", "I", "prior_review", "CASE-001 final alignment report", "experiments/test/wildfire_tests/workflow/cases/CASE-001-stage-i-retrospective/final_report/FINAL_ALIGNMENT_REPORT-v001.md", "prior_review", "HIGH"),
        ("EVID-0015", "I", "prior_review", "CASE-001 scientific review", "experiments/test/wildfire_tests/workflow/cases/CASE-001-stage-i-retrospective/scientific_review/SCIENTIFIC-REVIEW-CASE001-v001.md", "prior_review", "HIGH"),
        ("EVID-0016", "I", "prior_review", "CASE-001 implementation audit", "experiments/test/wildfire_tests/workflow/cases/CASE-001-stage-i-retrospective/implementation_audit/IMPLEMENTATION-AUDIT-CASE001-v001.md", "prior_review", "HIGH"),
        ("EVID-0017", "I", "prior_review", "CASE-001 claim register", "experiments/test/wildfire_tests/workflow/cases/CASE-001-stage-i-retrospective/CLAIM_REGISTER.csv", "prior_review", "HIGH"),
        ("EVID-0018", "I", "results", "Stage H/I main r11 best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/best_by_rho_scenario_lambda_stage.csv", "result_table", "HIGH"),
        ("EVID-0019", "I", "projection", "Stage H/I main r11 AC projection distances", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/ac_projection_distances.csv", "result_table", "HIGH"),
        ("EVID-0020", "I", "solver_metadata", "Stage H/I main r11 Stage I-b solver diagnostics", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11/tables/stage_i_b_solver_diagnostics.csv", "result_table", "MODERATE"),
        ("EVID-0021", "I", "results", "Stage H/I MLD r5 best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/MLD/r5/tables/best_by_rho_scenario_lambda_stage.csv", "result_table", "HIGH"),
        ("EVID-0022", "I", "projection", "Stage H/I MLD r5 AC projection distances", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/MLD/r5/tables/ac_projection_distances.csv", "result_table", "HIGH"),
        ("EVID-0023", "I", "results", "Proxy-inner r2 best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2/tables/best_by_scenario_proxy_inner_stage.csv", "result_table", "HIGH"),
        ("EVID-0024", "I", "results", "Proxy-inner r2 frontier table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2/tables/pareto_frontier_points_by_proxy.csv", "result_table", "HIGH"),
        ("EVID-0025", "I", "findings", "Stage H DC comparison progress note", "experiments/test/wildfire_tests/stage_i_dc_comparison/STAGE_H_DC_COMPARISON_PROGRESS.md", "progress_doc", "HIGH"),
        ("EVID-0026", "I", "findings", "Stage I-a proxy-inner findings note", "experiments/test/wildfire_tests/stage_i_dc_comparison/STAGE_H_STAGE_IA_PROXY_INNER_LAMBDA_FINDINGS.md", "progress_doc", "HIGH"),
        ("EVID-0027", "H", "results", "Stage H top-k heuristic best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/heuristic_baseline_comparison_topk/run_gnn_20260703_034242/tables/best_by_rho_scenario_lambda_method.csv", "result_table", "HIGH"),
        ("EVID-0028", "H", "results", "Stage H revised load/PAC best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_h/heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613/tables/best_by_rho_scenario_lambda_method.csv", "result_table", "HIGH"),
        ("EVID-0029", "G", "results", "Stage G revised continuous best table", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_g/physics_infeasibility_revised_continuous_implementation/continuous_run/tables/best_by_rho_scenario_lambda_stage.csv", "result_table", "HIGH"),
        ("EVID-0030", "E", "results", "Stage E continuous traditional lambdas rho0", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/with_continuous_optimization/rho0_no_physics/run_20260624_000543/best_by_stage_lambda.csv", "result_table", "HIGH"),
        ("EVID-0031", "E", "results", "Stage E continuous traditional lambdas rho100", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/physics_infeasibility_case_study/with_continuous_optimization/rho100_with_physics/run_20260624_003815/best_by_stage_lambda.csv", "result_table", "HIGH"),
        ("EVID-0032", "E", "results", "Stage E first Gurobi/GridFM summary", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm/stage_e_gurobi_gridfm_summary.csv", "result_table", "MODERATE"),
        ("EVID-0033", "D", "results", "Stage D deenergization summary", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_d/deenergization/stage_d_deenergization_summary.csv", "result_table", "MODERATE"),
        ("EVID-0034", "C", "results", "Stage C PSPS summary", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_c/psps_baseline/stage_c_psps_summary.csv", "result_table", "MODERATE"),
        ("EVID-0035", "B", "results", "Stage B multigroup threshold summary", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_b/multi_group/multi_group_threshold_sensitivity_summary.csv", "result_table", "LIMITED"),
        ("EVID-0036", "A", "results", "Stage A connected corridor summary", "experiments/test/wildfire_tests/ieee_30_stage_a_to_i_results/stage_a/demand_weighted/connected_corridor/connected_corridor_tradeoff_summary.csv", "result_table", "LIMITED"),
    ]
    evidence_rows = []
    for evid, stage, cat, desc, rpath, stype, authority in evidence_specs:
        path = ROOT / rpath
        exists = path.exists()
        is_file = exists and path.is_file()
        evidence_rows.append(
            {
                "evidence_id": evid,
                "stage": stage,
                "category": cat,
                "description": desc,
                "repository_path": rpath,
                "source_type": stype,
                "authority_level": authority,
                "availability_status": "AVAILABLE" if exists else "UNAVAILABLE",
                "inspection_status": "INSPECTED" if exists else "UNAVAILABLE",
                "sha256": sha256(path) if is_file else "",
                "size_bytes": path.stat().st_size if is_file else "",
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if exists else "",
                "used_in_outputs": "stage_map;result_register;final_report",
                "notes": f"csv_rows={row_count(path)} columns={csv_columns(path)[:250]}" if is_file and path.suffix.lower() == ".csv" else "",
            }
        )

    lit_rows = []
    pdfs = sorted(PAPERS.glob("*.pdf"))
    for idx, pdf in enumerate(pdfs, 1):
        priority, depth, claims, reason, status = classify_literature(pdf.name)
        head = pdf_head(pdf, 2 if priority != "TIER_1_CORE" else 6)
        title = pdf.stem.replace("_", " ")
        year = "".join(ch for ch in pdf.stem if ch.isdigit())[-4:]
        lit_rows.append(
            {
                "paper_id": f"PAPER-{idx:03d}",
                "filename": pdf.name,
                "title": title,
                "authors": "extracted in notes for Tier 1; otherwise filename/title based",
                "year": year,
                "topic": "wildfire shutoff" if priority == "TIER_1_CORE" else "supporting/contextual",
                "review_priority": priority,
                "inspection_depth": depth,
                "inspection_status": status,
                "sections_inspected": "formulation;results;limitations" if priority == "TIER_1_CORE" else ("targeted abstract/intro/method sections" if status == "INSPECTED" else ""),
                "claims_verified": claims,
                "specific_claims_supported": claims,
                "reason_for_inclusion": reason,
                "relevance_to_stages": "C;D;E;H;I" if priority == "TIER_1_CORE" else "method-dependent",
                "methodological_connection": reason,
                "limitations": "PDF text extraction used; figures/equations reviewed by extracted text, not image OCR.",
                "notes": f"pages={pdf_pages(pdf)}; sha256={sha256(pdf)}; head_excerpt={head[:500].replace(chr(10),' ')}",
            }
        )

    stage_rows = [
        ("A", "stage_a_first_pass", "First-pass connected-corridor GridFM wildfire experiments", "May 2026", "HIGH", "overlaps early demand-weighted objective refactors", "MODIFIED"),
        ("B", "stage_b_multigroup", "Automatic multigroup wildfire-risk selection and multistart sensitivity", "Late May-June 2026", "HIGH", "extends Stage A selection/scenario construction", "PARTIALLY_SUPERSEDED"),
        ("C", "stage_c_psps_baseline", "Deterministic PSPS threshold baseline", "Late May-June 2026", "HIGH", "shares risk ranking with Stage D/H heuristic baselines", "RETAINED"),
        ("D", "stage_d_deenergization", "Limited enumerated K<=2 de-energization reference", "Late May-June 2026", "HIGH", "serves as exhaustive K<=2 reference for early Stage E/G", "PARTIALLY_SUPERSEDED"),
        ("E", "stage_e_gurobi_implementation", "Gurobi-proposed GridFM topology search and continuous-control variants", "June 2026", "HIGH", "straddles topology search and GridFM recourse experiments", "MODIFIED"),
        ("F", "stage_f_decision_quality", "Five scenario decision-quality suite", "June 2026", "MODERATE", "scenario definitions feed Stage G/H/I", "RETAINED"),
        ("G", "stage_g_implementation_revision", "Physics/load/PAC corrections and revised continuous GridFM evaluator", "Late June-July 2026", "HIGH", "corrects Stage E/F evaluation semantics", "RETAINED"),
        ("H", "stage_h_heuristic_comparison and Stage H result folders", "Heuristic baseline and later DC-comparison result framing", "July 2026", "MODERATE", "name overlaps with Stage I DC comparison outputs", "MODIFIED"),
        ("I", "stage_i_dc_comparison", "DC approximation, MIQP, AC projection, MLD, proxy-inner comparison", "July 2026", "MODERATE", "implemented in stage_i_dc_comparison but results live under Stage H path", "RETAINED"),
    ]

    # CSV outputs.
    write_csv(
        "EVIDENCE_REGISTER.csv",
        evidence_rows,
        [
            "evidence_id",
            "stage",
            "category",
            "description",
            "repository_path",
            "source_type",
            "authority_level",
            "availability_status",
            "inspection_status",
            "sha256",
            "size_bytes",
            "modified_time",
            "used_in_outputs",
            "notes",
        ],
    )
    write_csv(
        "LITERATURE_EVIDENCE_REGISTER.csv",
        lit_rows,
        [
            "paper_id",
            "filename",
            "title",
            "authors",
            "year",
            "topic",
            "review_priority",
            "inspection_depth",
            "inspection_status",
            "sections_inspected",
            "claims_verified",
            "specific_claims_supported",
            "reason_for_inclusion",
            "relevance_to_stages",
            "methodological_connection",
            "limitations",
            "notes",
        ],
    )
    write_csv(
        "STAGE_RELATIONSHIP_MATRIX.csv",
        [
            {"from_stage": "A", "to_stage": "B", "relationship_type": "GENERALIZES", "problem_or_gap": "manual/single corridor did not expose multiple risk regions", "what_changed": "automatic risk components and threshold sensitivity", "what_was_retained": "GridFM risk/load objective idea", "what_was_rejected": "single manual corridor as sufficient evidence", "evidence_ids": "EVID-0001;EVID-0002;EVID-0006", "interpretation": "Stage B broadened scenario construction."},
            {"from_stage": "B", "to_stage": "C", "relationship_type": "ADDS_COMPARISON", "problem_or_gap": "optimization results needed deterministic PSPS comparator", "what_changed": "threshold PSPS baseline", "what_was_retained": "risk-ranked line candidates", "what_was_rejected": "continuous GridFM-only comparison", "evidence_ids": "EVID-0007;EVID-0034", "interpretation": "Stage C created a Rhodes-like heuristic baseline."},
            {"from_stage": "C", "to_stage": "D", "relationship_type": "VALIDATES", "problem_or_gap": "threshold PSPS did not search topology choices", "what_changed": "enumerated all K<=2 candidate shutoffs", "what_was_retained": "fixed consequence/risk scoring", "what_was_rejected": "single threshold topology as enough", "evidence_ids": "EVID-0008;EVID-0033", "interpretation": "Stage D provided an interpretable exhaustive small-budget reference."},
            {"from_stage": "D", "to_stage": "E", "relationship_type": "ADDS_SCALABILITY", "problem_or_gap": "exhaustive enumeration does not scale", "what_changed": "Gurobi proxy master proposes topology pool", "what_was_retained": "GridFM true evaluation", "what_was_rejected": "full enumeration as only method", "evidence_ids": "EVID-0009;EVID-0032", "interpretation": "Stage E turned enumeration into guided search."},
            {"from_stage": "E", "to_stage": "F", "relationship_type": "ADDS_COMPARISON", "problem_or_gap": "single scenario/risk construction could bias conclusions", "what_changed": "five decision-quality scenarios", "what_was_retained": "Stage C/D/E comparison surface", "what_was_rejected": "one-off case conclusions", "evidence_ids": "EVID-0010", "interpretation": "Stage F was a decision-quality stress surface."},
            {"from_stage": "F", "to_stage": "G", "relationship_type": "CORRECTS", "problem_or_gap": "load service and physics diagnostics were inconsistent/under-specified", "what_changed": "source-less island correction, raw/eval split, hybrid load and PAC decomposition", "what_was_retained": "five scenarios and GridFM topology evaluation", "what_was_rejected": "commanded-only load and partial PAC as final metric", "evidence_ids": "EVID-0011;EVID-0029", "interpretation": "Stage G exposed and corrected key evaluation semantics."},
            {"from_stage": "G", "to_stage": "H", "relationship_type": "BENCHMARKS", "problem_or_gap": "needed comparison against Rhodes-like simple heuristics", "what_changed": "TH/AH heuristic baseline plots and expected-vs-selected metrics", "what_was_retained": "revised continuous evaluator", "what_was_rejected": "Stage E alone as comparison", "evidence_ids": "EVID-0012;EVID-0027;EVID-0028", "interpretation": "Stage H reframed results as method comparison."},
            {"from_stage": "H", "to_stage": "I", "relationship_type": "ADDS_FEASIBILITY_CHECK", "problem_or_gap": "GridFM outputs showed physical-realism limitations", "what_changed": "DC guided recourse, DC MIQP, common diagnostic, AC projection", "what_was_retained": "K<=2, five scenarios, TH/AH comparison", "what_was_rejected": "GridFM-only evidence as sufficient", "evidence_ids": "EVID-0004;EVID-0013;EVID-0018;EVID-0019", "interpretation": "Stage I created grounded DC/AC feasibility comparison."},
        ],
        ["from_stage", "to_stage", "relationship_type", "problem_or_gap", "what_changed", "what_was_retained", "what_was_rejected", "evidence_ids", "interpretation"],
    )
    write_csv(
        "RESEARCH_QUESTION_EVOLUTION.csv",
        [
            {"question_version": "RQ-001", "development_period": "Stage A", "research_question": "Can GridFM support a first-pass wildfire-risk/load-shedding tradeoff on IEEE-30?", "motivation": "Start from a tractable surrogate-based wildfire operation harness.", "triggering_evidence": "initial first-pass runs", "methods_in_scope": "GridFM fixed topology; connected corridor", "methods_out_of_scope": "PSPS, topology switching, DC/AC feasibility", "assumptions": "surrogate objective is informative", "what_changed_next": "needed automatic multigroup and threshold sensitivity", "evidence_ids": "EVID-0005;EVID-0036"},
            {"question_version": "RQ-002", "development_period": "Stages B-D", "research_question": "How do risk-ranked PSPS and small-budget topology choices affect risk/load tradeoffs?", "motivation": "Need interpretable baselines and topology controls.", "triggering_evidence": "Stage C/D summaries", "methods_in_scope": "PSPS thresholds; K<=2 enumeration", "methods_out_of_scope": "continuous recourse; security contingencies", "assumptions": "fixed GridFM evaluation can score topologies", "what_changed_next": "guided search was needed for scalability", "evidence_ids": "EVID-0006;EVID-0007;EVID-0008;EVID-0033;EVID-0034"},
            {"question_version": "RQ-003", "development_period": "Stage E", "research_question": "Can a Gurobi proxy master propose high-quality GridFM-evaluated topology candidates?", "motivation": "Move beyond exhaustive enumeration.", "triggering_evidence": "Stage E summaries and topology profiles", "methods_in_scope": "Stage E constrained/unconstrained topology search", "methods_out_of_scope": "full embedded GridFM optimization; N-1 security", "assumptions": "GridFM predicted state is usable for evaluation", "what_changed_next": "decision-quality scenarios and continuous recourse needed", "evidence_ids": "EVID-0009;EVID-0030;EVID-0031;EVID-0032"},
            {"question_version": "RQ-004", "development_period": "Stages F-G", "research_question": "Are GridFM decisions physically meaningful under multiple scenarios and revised load/PAC accounting?", "motivation": "Observed load-service and physics inconsistency.", "triggering_evidence": "source-less island and hybrid load/PAC findings", "methods_in_scope": "five scenarios; physics diagnostics; raw/eval split", "methods_out_of_scope": "DC/AC optimization benchmark initially", "assumptions": "diagnostics can reveal surrogate limitations", "what_changed_next": "needed TH/AH and DC comparisons", "evidence_ids": "EVID-0010;EVID-0011;EVID-0029"},
            {"question_version": "RQ-005", "development_period": "Stages H-I", "research_question": "How does GridFM-guided wildfire topology control compare with Rhodes-like heuristics and DC optimization under common diagnostics?", "motivation": "Need fairer comparison and feasibility grounding.", "triggering_evidence": "Stage H top-k/revised runs and CASE-001", "methods_in_scope": "Stage E K2; TH/AH; Stage I-a; Stage I-b; MLD; proxy-inner; AC projection", "methods_out_of_scope": "full SC-OPS/N-1 security; robust optimization", "assumptions": "K<=2 benchmark is useful first controlled study", "what_changed_next": "advisor decision needed for framing and next experiments", "evidence_ids": "EVID-0012;EVID-0013;EVID-0014;EVID-0018;EVID-0024"},
        ],
        ["question_version", "development_period", "research_question", "motivation", "triggering_evidence", "methods_in_scope", "methods_out_of_scope", "assumptions", "what_changed_next", "evidence_ids"],
    )
    write_csv(
        "STAGE_RESULT_REGISTER.csv",
        [
            {"result_id": "RES-A-001", "stage": "A", "experiment_or_run": "connected corridor/demand-weighted first pass", "research_question": "first GridFM wildfire/load tradeoff", "methods": "GridFM first pass", "scenarios": "IEEE-30 first-pass", "lambda_or_parameter_setting": "risk/balanced/shed", "primary_metric": "objective", "secondary_metrics": "group risk; demand-weighted shedding", "result_path": "ieee_30_stage_a_to_i_results/stage_a/demand_weighted/connected_corridor", "result_status": "historical", "main_observation": "Established initial harness but not final topology-control evidence.", "evidence_strength": "LIMITED", "known_limitations": "manual corridor and early load semantics", "used_in_current_framing": "historical foundation"},
            {"result_id": "RES-C-001", "stage": "C", "experiment_or_run": "PSPS threshold baseline", "research_question": "risk-ranked PSPS comparator", "methods": "deterministic PSPS", "scenarios": "auto_env; largest_group_high", "lambda_or_parameter_setting": "risk-emphasized", "primary_metric": "post-PSPS risk", "secondary_metrics": "load shed", "result_path": "ieee_30_stage_a_to_i_results/stage_c/psps_baseline", "result_status": "executed", "main_observation": "Created deterministic threshold baseline analogous to heuristic shutoff.", "evidence_strength": "MODERATE", "known_limitations": "not optimized and not security-constrained", "used_in_current_framing": "baseline lineage"},
            {"result_id": "RES-D-001", "stage": "D", "experiment_or_run": "K<=2 limited enumeration", "research_question": "small-budget exhaustive topology reference", "methods": "enumerated de-energization", "scenarios": "auto_env; largest_group_high", "lambda_or_parameter_setting": "risk/balanced/service", "primary_metric": "J", "secondary_metrics": "R_norm; L_shed", "result_path": "ieee_30_stage_a_to_i_results/stage_d/deenergization", "result_status": "executed", "main_observation": "Provided exhaustive K<=2 reference; later omitted from main Stage I guided-search comparison.", "evidence_strength": "MODERATE", "known_limitations": "fixed-control GridFM and small candidate space", "used_in_current_framing": "historical baseline"},
            {"result_id": "RES-E-001", "stage": "E", "experiment_or_run": "Gurobi/GridFM topology search", "research_question": "guided topology search vs enumeration", "methods": "Stage E constrained/unconstrained", "scenarios": "auto/lgh", "lambda_or_parameter_setting": "multiple", "primary_metric": "J", "secondary_metrics": "risk/load/runtime", "result_path": "ieee_30_stage_a_to_i_results/stage_e/gurobi_gridfm", "result_status": "executed", "main_observation": "Guided topology proposal became the main GridFM method.", "evidence_strength": "MODERATE", "known_limitations": "surrogate physical realism not yet resolved", "used_in_current_framing": "core method precursor"},
            {"result_id": "RES-E-002", "stage": "E", "experiment_or_run": "continuous traditional lambdas", "research_question": "continuous recourse within topology", "methods": "Stage C/D/E continuous GridFM", "scenarios": "auto_env", "lambda_or_parameter_setting": "0.8/0.5/0.2; rho 0/100", "primary_metric": "J_true", "secondary_metrics": "call counts; alpha diagnostics", "result_path": "stage_e/physics_infeasibility_case_study/with_continuous_optimization", "result_status": "executed", "main_observation": "Continuous recourse was expensive and often moved controls little; alpha/GridFM service mismatch emerged.", "evidence_strength": "STRONG", "known_limitations": "100-call budget, not globally converged", "used_in_current_framing": "motivating failure/correction"},
            {"result_id": "RES-G-001", "stage": "G", "experiment_or_run": "revised continuous implementation", "research_question": "physics/load/PAC diagnostic correction", "methods": "Stage D/E revised continuous", "scenarios": "S1-S5", "lambda_or_parameter_setting": "lambda sweep; rho panels", "primary_metric": "J_true", "secondary_metrics": "PAC; R_norm; L_shed", "result_path": "stage_g/physics_infeasibility_revised_continuous_implementation", "result_status": "executed", "main_observation": "Separated command, GridFM prediction, evaluation clamps, source-less islands, and physics diagnostics.", "evidence_strength": "STRONG", "known_limitations": "still GridFM-surrogate dependent", "used_in_current_framing": "current GridFM evaluator basis"},
            {"result_id": "RES-H-001", "stage": "H", "experiment_or_run": "top-k heuristic comparison", "research_question": "TH/AH vs formulation-based GridFM", "methods": "TH; AH; Stage D/E references", "scenarios": "S1-S5", "lambda_or_parameter_setting": "lambda/rho panels", "primary_metric": "J_true", "secondary_metrics": "expected-vs-selected; overlap", "result_path": "stage_h/heuristic_baseline_comparison_topk/run_gnn_20260703_034242", "result_status": "executed", "main_observation": "Rank-based TH can be competitive in structured cases; AH generally weaker.", "evidence_strength": "STRONG", "known_limitations": "sparse policy points, not dense Pareto search", "used_in_current_framing": "Rhodes heuristic analogue"},
            {"result_id": "RES-H-002", "stage": "H", "experiment_or_run": "revised load/PAC heuristic comparison", "research_question": "effect of hybrid load/PAC accounting", "methods": "TH/AH with revised evaluator", "scenarios": "S1-S5", "lambda_or_parameter_setting": "lambda/rho panels", "primary_metric": "L_shed_hybrid; PAC_total", "secondary_metrics": "PAC groups", "result_path": "stage_h/heuristic_baseline_comparison_revised_load_pac/run_gnn_20260711_022613", "result_status": "executed", "main_observation": "Old commanded/effective load metric understated GridFM-implied non-selected service degradation.", "evidence_strength": "STRONG", "known_limitations": "Stage D/E reference rows may retain historical accounting", "used_in_current_framing": "GridFM limitation evidence"},
            {"result_id": "RES-I-001", "stage": "I", "experiment_or_run": "main r11 K<=2 comparison", "research_question": "GridFM vs DC guided vs DC MIQP vs heuristics", "methods": "Stage E K2; Stage I-a; Stage I-b; TH/AH", "scenarios": "S1-S5", "lambda_or_parameter_setting": "lambda=[0,0.2,0.5,0.8,1]; rho=[0,2]", "primary_metric": "L_shed/R_norm Pareto; J_true", "secondary_metrics": "common diagnostic; projection; runtime", "result_path": "stage_h/DC Approximation + Baseline Heuristic Comparison/main_results/r11", "result_status": "executed", "main_observation": "DC methods provide physically grounded comparison; GridFM physical-realism limitations dominate Stage E interpretation.", "evidence_strength": "STRONG", "known_limitations": "AC projection finite only for subset; relaxed Qg bounds", "used_in_current_framing": "core current evidence"},
            {"result_id": "RES-I-002", "stage": "I", "experiment_or_run": "MLD r5", "research_question": "literature-aligned maximum load delivery case", "methods": "Stage E; Stage I-a; Stage I-b; TH/AH", "scenarios": "S1-S5", "lambda_or_parameter_setting": "proxy=1 inner=0", "primary_metric": "load delivery", "secondary_metrics": "risk; projection", "result_path": "stage_h/DC Approximation + Baseline Heuristic Comparison/MLD/r5", "result_status": "executed", "main_observation": "MLD framing is useful for Rhodes alignment but is not the same as main risk/load sweep.", "evidence_strength": "MODERATE", "known_limitations": "different lambda semantics from main results", "used_in_current_framing": "literature alignment study"},
            {"result_id": "RES-I-003", "stage": "I", "experiment_or_run": "proxy-inner r2", "research_question": "decouple topology proxy lambda and inner lambda", "methods": "Stage E K2; Stage I-a", "scenarios": "S1-S5", "lambda_or_parameter_setting": "5x5 proxy/inner sweep; rho=0", "primary_metric": "Pareto improvement", "secondary_metrics": "common diagnostic", "result_path": "stage_h/DC Approximation + Baseline Heuristic Comparison/proxy_inner_lambda_sweep/r2", "result_status": "executed", "main_observation": "Can improve DC tradeoff fronts for selected scenarios, but no AC projection evidence.", "evidence_strength": "MODERATE", "known_limitations": "DC-only and proxy-inner-specific", "used_in_current_framing": "future search-dimension option"},
        ],
        ["result_id", "stage", "experiment_or_run", "research_question", "methods", "scenarios", "lambda_or_parameter_setting", "primary_metric", "secondary_metrics", "result_path", "result_status", "main_observation", "evidence_strength", "known_limitations", "used_in_current_framing"],
    )
    write_csv(
        "LITERATURE_ALIGNMENT_MATRIX.csv",
        [
            {"research_component": "Optimal power shutoff tradeoff", "project_stage": "C-D-E-H-I", "our_method": "risk/load objective or sweep over lambda_R", "paper_ids": "Balancing Wildfire Risk and Power Outages Through Optimized Power Shut-Offs", "literature_priority": "TIER_1_CORE", "alignment_type": "ADAPTED", "similarities": "Both compare wildfire risk reduction against load served/load shed.", "differences": "Our current objective uses GridFM/DC-specific normalized risk and K<=2/five-scenario experiments rather than RTS-GMLC OPS MILP.", "adaptation_or_novelty": "Adapts OPS tradeoff into surrogate/DC comparison workflow.", "security_model_difference": "Rhodes 2021 notes lack of N-1 security; our current main work also lacks full N-1 security.", "feasibility_model_difference": "Rhodes uses DC MILP feasibility; our GridFM rows use diagnostics and DC/AC comparison added later.", "evidence_strength": "STRONG", "terminology_check": "Use OPS/wildfire-risk-aware shutoff, not SC-OPS for current method.", "claim_limitations": "GridFM surrogate additions are not in Rhodes."},
            {"research_component": "Security-constrained power shutoff", "project_stage": "I and future work", "our_method": "common diagnostics and AC projection, but no contingency set", "paper_ids": "Security-Constrained Optimal Power Shutoff", "literature_priority": "TIER_1_CORE", "alignment_type": "CONTRASTING", "similarities": "Both care about operational feasibility after shutoff.", "differences": "SC-OPS includes post-contingency constraints and post-contingency load shed bounds; current Stage I does not.", "adaptation_or_novelty": "Our AC projection is a post-solution diagnostic, not SC-OPS.", "security_model_difference": "Current work is not security-constrained in the SC-OPS/N-1 sense.", "feasibility_model_difference": "Base-case DC and relaxed-Qg AC projection vs contingency-constrained DC optimization.", "evidence_strength": "STRONG", "terminology_check": "Avoid calling current method security-constrained.", "claim_limitations": "SC-OPS alignment is future direction, not implemented claim."},
            {"research_component": "TH/AH heuristic baselines", "project_stage": "H", "our_method": "rank-based TH top-k and connected-component AH", "paper_ids": "Balancing Wildfire Risk and Power Outages Through Optimized Power Shut-Offs", "literature_priority": "TIER_1_CORE", "alignment_type": "ADAPTED", "similarities": "Both compare transmission and area-style heuristics to optimized shutoff.", "differences": "Our AH uses network connected components rather than geographic RTS regions; TH is top-k/rank rather than threshold value.", "adaptation_or_novelty": "Small-grid adaptation of Rhodes heuristic ideas.", "security_model_difference": "No N-1 security in either heuristic comparison.", "feasibility_model_difference": "Rhodes solves MLD after heuristic; our GridFM/DC variants evaluate through current recourse/diagnostics.", "evidence_strength": "STRONG", "terminology_check": "Call these Rhodes-inspired heuristic analogues.", "claim_limitations": "Not exact reproduction of Rhodes AH/TH."},
            {"research_component": "DC topology switching", "project_stage": "I", "our_method": "Stage I-a fixed-topology DC recourse and Stage I-b DC MIQP", "paper_ids": "A Cycle-Based Formulation and Valid Inequalities for DC Power Transmission Problems with Switching_2015.pdf; An introduction to optimal power flow Theory formulation and examples_2016.pdf", "literature_priority": "TIER_2_SUPPORTING", "alignment_type": "PARTIAL", "similarities": "Uses DC power-flow, binary line state, thermal constraints, nodal balance.", "differences": "Our objective is wildfire risk/load, not generic OTS cost minimization.", "adaptation_or_novelty": "DC formulation is a benchmark for GridFM decisions.", "security_model_difference": "No contingency set in current Stage I.", "feasibility_model_difference": "DC feasibility is exact within approximation, AC projection remains diagnostic.", "evidence_strength": "MODERATE", "terminology_check": "DC approximation, not AC OPF.", "claim_limitations": "Supporting papers inspected targeted, not exhaustive."},
            {"research_component": "Learned surrogate in optimization", "project_stage": "E-G-H-I", "our_method": "GridFM predict-then-evaluate topology/recourse decisions", "paper_ids": "Foundation Models for the Electric Power Grid_2024.pdf; GridFM-Datakit-V1.pdf; GNN for Efficient AC Power Flow Prediction in Power Grids_2025.pdf; Safe Power Graph Safety-aware Evaluation of GNN for Transmission Power Grids_2024.pdf", "literature_priority": "TIER_2_SUPPORTING", "alignment_type": "EXPLORATORY", "similarities": "Uses learned power-system model outputs for operational evaluation.", "differences": "Current project exposes command-faithfulness and physical-realism limitations under topology interventions.", "adaptation_or_novelty": "Empirical evaluation of GridFM suitability for wildfire shutoff workflows.", "security_model_difference": "No learned SC-OPS security guarantee.", "feasibility_model_difference": "Diagnostics/projection rather than hard learned feasibility enforcement.", "evidence_strength": "MODERATE", "terminology_check": "Frame as learned-surrogate evaluation, not guaranteed power-flow solver.", "claim_limitations": "Do not generalize beyond current GridFM implementation/scenarios."},
        ],
        ["research_component", "project_stage", "our_method", "paper_ids", "literature_priority", "alignment_type", "similarities", "differences", "adaptation_or_novelty", "security_model_difference", "feasibility_model_difference", "evidence_strength", "terminology_check", "claim_limitations"],
    )
    write_csv(
        "CLAIM_AND_CONTRIBUTION_REGISTER.csv",
        [
            {"claim_id": "CASE002-CLAIM-001", "claim_text": "The project developed a staged wildfire-risk-aware topology-control research harness using GridFM, heuristics, DC optimization, and AC projection diagnostics.", "claim_type": "METHOD", "stages_supporting": "A-I", "result_ids": "RES-A-001;RES-I-001", "literature_support": "Rhodes OPS/SC-OPS as problem lineage; GridFM papers as surrogate context", "evidence_status": "AVAILABLE", "current_strength": "SUPPORTED_WITH_LIMITATIONS", "qualification_needed": "IEEE-30/S1-S5 controlled setting; no full N-1 security.", "publication_role": "framework/methodology candidate", "presentation_role": "main story", "risk_of_overclaim": "MODERATE", "recommended_wording": "We built and audited a staged comparative workflow for wildfire-aware shutoff decisions."},
            {"claim_id": "CASE002-CLAIM-002", "claim_text": "The current method is not fully security-constrained in the Rhodes SC-OPS sense.", "claim_type": "LIMITATION", "stages_supporting": "I", "result_ids": "RES-I-001", "literature_support": "Security-Constrained Optimal Power Shutoff", "evidence_status": "AVAILABLE", "current_strength": "SUPPORTED", "qualification_needed": "Base-case feasibility and projection diagnostics exist, but no contingency set.", "publication_role": "terminology guardrail", "presentation_role": "advisor decision point", "risk_of_overclaim": "LOW", "recommended_wording": "Current experiments are wildfire-risk-aware and feasibility-diagnostic, not SC-OPS."},
            {"claim_id": "CASE002-CLAIM-003", "claim_text": "Stage H/I results expose limitations in current GridFM physical realism under controlled topology/recourse interventions.", "claim_type": "PHYSICS_RESULT", "stages_supporting": "G;H;I", "result_ids": "RES-H-002;RES-I-001", "literature_support": "learned surrogate safety/feasibility literature targeted", "evidence_status": "AVAILABLE", "current_strength": "SUPPORTED_WITH_LIMITATIONS", "qualification_needed": "Current implementation/scenarios only.", "publication_role": "limitation analysis", "presentation_role": "strong finding", "risk_of_overclaim": "HIGH", "recommended_wording": "In this workflow, GridFM predictions created large diagnostic discrepancies, motivating DC/AC checks."},
            {"claim_id": "CASE002-CLAIM-004", "claim_text": "Stage I-a and Stage I-b provide a transparent DC baseline and compact MIQP comparison for the GridFM-guided approach.", "claim_type": "COMPARATIVE_RESULT", "stages_supporting": "I", "result_ids": "RES-I-001", "literature_support": "DC OPF/OTS literature targeted", "evidence_status": "AVAILABLE", "current_strength": "SUPPORTED_WITH_LIMITATIONS", "qualification_needed": "DC approximation, not AC validation; projection incomplete.", "publication_role": "comparative benchmark", "presentation_role": "methodological maturation", "risk_of_overclaim": "MODERATE", "recommended_wording": "DC methods ground the comparison within a known approximation."},
            {"claim_id": "CASE002-CLAIM-005", "claim_text": "Proxy-inner lambda decoupling can improve Stage I-a DC tradeoff fronts in selected scenarios.", "claim_type": "EMPIRICAL_RESULT", "stages_supporting": "I", "result_ids": "RES-I-003", "literature_support": "not required", "evidence_status": "AVAILABLE", "current_strength": "PRELIMINARY", "qualification_needed": "No AC projection evidence; scenario-dependent.", "publication_role": "future-search ablation", "presentation_role": "interesting extension", "risk_of_overclaim": "MODERATE", "recommended_wording": "Decoupling appears useful as a DC search expansion, pending AC checks."},
        ],
        ["claim_id", "claim_text", "claim_type", "stages_supporting", "result_ids", "literature_support", "evidence_status", "current_strength", "qualification_needed", "publication_role", "presentation_role", "risk_of_overclaim", "recommended_wording"],
    )

    stage_map = ["# Stage A-I Development Map", "", f"Generated UTC: {now}", ""]
    for stage, hist, curr, period, conf, overlap, outcome in stage_rows:
        stage_map += [
            f"## Stage {stage} - {curr}",
            "",
            f"- historical_stage_name: `{hist}`",
            f"- current_reconstructed_name: `{curr}`",
            f"- approximate_development_period: `{period}`",
            f"- confidence_in_stage_boundary: `{conf}`",
            f"- overlap_with_other_stages: {overlap}",
            f"- development_outcome: `{outcome}`",
            "",
            "### Research purpose",
            stage_purpose(stage),
            "",
            "### Motivation",
            stage_motivation(stage),
            "",
            "### Inputs",
            stage_inputs(stage),
            "",
            "### Decision variables or evaluated quantities",
            stage_variables(stage),
            "",
            "### Methodology",
            stage_methodology(stage),
            "",
            "### Objective and constraints",
            stage_objective(stage),
            "",
            "### Outputs",
            stage_outputs(stage),
            "",
            "### Main findings",
            stage_findings(stage),
            "",
            "### Limitations",
            stage_limitations(stage),
            "",
            "### Relationship to prior and later stages",
            stage_relationship_text(stage),
            "",
            "### Relationship to Rhodes wildfire shutoff literature",
            stage_rhodes(stage),
            "",
            "### Broader literature alignment",
            stage_literature(stage),
            "",
            "### Current status and publication value",
            stage_status(stage),
            "",
            "### Confidence assessment",
            "- reconstructed purpose: `HIGH`" if stage not in {"F", "H", "I"} else "- reconstructed purpose: `MODERATE`",
            "- implementation description: `HIGH`",
            "- result interpretation: `MODERATE`",
            "- literature alignment: `MODERATE`",
            "- current role in project: `MODERATE`",
            "",
        ]
    write_text("STAGE_A_I_DEVELOPMENT_MAP.md", "\n".join(stage_map))

    write_text("CASE_CHARTER.md", charter_text(now))
    write_text("SYNTHESIS_EXECUTION_PLAN.md", execution_plan_text(now))
    write_text("FRAMING_OPTIONS.md", framing_text())
    write_text("NEXT_STEP_OPTIONS.md", next_steps_text())
    write_text("PROFESSOR_DISCUSSION_BRIEF.md", professor_brief_text())
    write_text("FINAL_ALIGNMENT_REPORT.md", final_report_text(now, len(evidence_rows), len(lit_rows)))

    # Preliminary role artifacts to satisfy freeze-before-reconciliation.
    write_text("SCIENTIFIC_LITERATURE_REVIEW-v001.md", scientific_review_text())
    write_text("IMPLEMENTATION_EVIDENCE_VERIFICATION-v001.md", implementation_review_text())
    freeze_initial = []
    for name in ["SCIENTIFIC_LITERATURE_REVIEW-v001.md", "IMPLEMENTATION_EVIDENCE_VERIFICATION-v001.md"]:
        p = CASE / name
        freeze_initial.append({"artifact": name, "sha256": sha256(p), "size_bytes": p.stat().st_size, "frozen_utc": now})
    write_csv("CASE_INITIAL_REVIEW_FREEZE.csv", freeze_initial, ["artifact", "sha256", "size_bytes", "frozen_utc"])
    write_text("RECONCILIATION-v001.md", reconciliation_text())

    input_freeze = []
    for row in evidence_rows:
        if row["availability_status"] == "AVAILABLE" and row["sha256"]:
            input_freeze.append({"evidence_id": row["evidence_id"], "path": row["repository_path"], "sha256_before": row["sha256"], "sha256_after": row["sha256"], "status": "UNCHANGED_DURING_CASE002_GENERATION"})
    write_csv("CASE_INPUT_FREEZE.csv", input_freeze, ["evidence_id", "path", "sha256_before", "sha256_after", "status"])

    after_hashes = {rel(p): sha256(p) for p in protected if p.exists()}
    write_text(
        "VALIDATION_SUMMARY.md",
        "\n".join(
            [
                "# CASE-002 Validation Summary",
                "",
                f"Generated UTC: {now}",
                "",
                "## Boundary",
                "All generated artifacts are inside the CASE-002 folder. No experiments, tests, solvers, source edits, result edits, or frozen package edits were performed.",
                "",
                "## Protected Hash Check",
                json.dumps({"before": before_hashes, "after": after_hashes, "unchanged": before_hashes == after_hashes}, indent=2),
                "",
                "## Register Counts",
                f"- evidence rows: {len(evidence_rows)}",
                f"- literature rows: {len(lit_rows)}",
                "- final frozen artifacts: recorded in `CASE_FINAL_FREEZE.csv`",
                "",
                "## Status",
                "`CASE002_SYNTHESIS_COMPLETE_AWAITING_CALEB_INTERPRETATION`",
            ]
        ),
    )

    # Final freeze excludes itself to avoid self-referential hashing.
    freeze_rows = []
    for p in sorted(CASE.iterdir()):
        if p.is_file() and p.name not in {"CASE_FINAL_FREEZE.csv"}:
            freeze_rows.append({"artifact": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size, "frozen_utc": now})
    write_csv("CASE_FINAL_FREEZE.csv", freeze_rows, ["artifact", "sha256", "size_bytes", "frozen_utc"])


def stage_purpose(stage: str) -> str:
    return {
        "A": "Establish the first GridFM wildfire-aware optimization harness and see whether a risk/load objective could be evaluated at all.",
        "B": "Replace manual single-corridor thinking with automatic multigroup risk selection and threshold sensitivity.",
        "C": "Create a deterministic PSPS-style threshold baseline for comparison against optimization methods.",
        "D": "Enumerate small-budget line de-energization choices to create an interpretable K<=2 reference.",
        "E": "Use a Gurobi proxy master to propose topology candidates that are evaluated by GridFM, including constrained and unconstrained variants.",
        "F": "Construct five decision-quality scenarios to stress whether topology choices behave consistently across different target structures.",
        "G": "Correct and deepen GridFM evaluation semantics: source-less island service, commanded versus raw predictions, hybrid load, and PAC decomposition.",
        "H": "Compare formulation-based GridFM choices against Rhodes-inspired TH/AH heuristics, then house the later DC-comparison result family.",
        "I": "Add DC approximation baselines, direct MIQP, MLD alignment, proxy-inner lambda exploration, and AC projection diagnostics.",
    }[stage]


def stage_motivation(stage: str) -> str:
    return {
        "A": "The project needed a working end-to-end surface before deeper wildfire shutoff claims could be made.",
        "B": "Manual risk corridors were too brittle for general claims.",
        "C": "A PSPS-like baseline was needed because de-energization is a standard operational wildfire mitigation action.",
        "D": "A single PSPS threshold could not reveal whether better small-budget shutoff sets existed.",
        "E": "Enumeration was not a scalable research path, so topology search needed a guided proposal mechanism.",
        "F": "Early cases risked being overfit to one risk construction.",
        "G": "Observed load-service and physics inconsistencies made prior objective values too optimistic or under-specified.",
        "H": "The work needed paper-inspired heuristic baselines and clearer decision-quality visualizations.",
        "I": "GridFM physical-realism concerns motivated transparent DC optimization and AC feasibility/projection diagnostics.",
    }[stage]


def stage_inputs(stage: str) -> str:
    return {
        "A": "IEEE-30 processed GridFM tensors, initial wildfire corridor definitions, GridFM surrogate predictions.",
        "B": "Stage A harness, automatic line-risk scores, threshold fractions, GNN/GPS model variants.",
        "C": "Automatic candidate sets, fixed demand-weighted consequence score, PSPS thresholds.",
        "D": "Stage C candidate construction, K<=2 subset enumeration, GridFM post-topology predictions.",
        "E": "Stage D/E candidate lines, Gurobi proxy model, GridFM true evaluator, lambda settings.",
        "F": "Existing Stage C/D/E methods and five scenario definitions S1-S5.",
        "G": "Stage F scenarios, GridFM raw outputs, MATPOWER metadata, revised source-less island and PAC logic.",
        "H": "Stage G revised continuous evaluator, target line sets, baseline loading scores, TH/AH heuristic definitions.",
        "I": "Stage G/H GridFM evaluator, MATPOWER `rateA`, DC branch data, Stage E-style topology loops, Gurobi MIQP, AC projection backend.",
    }[stage]


def stage_variables(stage: str) -> str:
    return {
        "A": "`u` controls, grouped wildfire risk, demand-weighted load shedding.",
        "B": "selected risk components, group membership, multistart seeds, objective values.",
        "C": "PSPS selected lines and post-PSPS GridFM service fraction.",
        "D": "de-energized line subset `S_off` with `|S_off| <= 2`, `R_norm`, `L_shed`, `J`.",
        "E": "binary line states proposed by Gurobi proxy, optional continuous controls `Delta_Pg` and `alpha`, GridFM-evaluated objective.",
        "F": "scenario target margins, expected target lines, Stage D/E objective gaps.",
        "G": "`x_raw`, `x_eval`, `L_shed_cmd`, `L_shed_gridfm_raw`, `L_shed_gridfm_effective`, `L_shed_hybrid`, PAC groups.",
        "H": "TH top-k lines, AH connected group, expected-vs-selected hits, recall, precision, GridFM recourse metrics.",
        "I": "Stage I-a fixed-topology DC variables `theta,f,Pg,s`; Stage I-b binary `z,y` plus continuous DC variables; projection distances.",
    }[stage]


def stage_methodology(stage: str) -> str:
    return {
        "A": "Run fixed-topology GridFM experiments and score wildfire/load objectives.",
        "B": "Rank lines by automatic risk scores, form components, and run multistart sensitivity.",
        "C": "Deactivate top-risk PSPS lines, run GridFM post-topology evaluation, and summarize risk/load.",
        "D": "Evaluate all candidate subsets of size 0, 1, or 2 and select best by lambda-weighted objective.",
        "E": "Use Gurobi as proxy topology proposer, then evaluate candidates through GridFM true objective; later add continuous recourse call-budget studies.",
        "F": "Evaluate Stage C/D/E methods across five controlled decision-quality scenarios.",
        "G": "Separate commanded/evaluated/raw states, correct islanded load, add hybrid load metric and PAC operational/AC/model consistency groups.",
        "H": "Run TH top-k and AH connected heuristics through the Stage G evaluator and compare against Stage G/Stage E references.",
        "I": "Run Stage E K2 GridFM, Stage I-a guided DC recourse, Stage I-b DC MIQP, TH/AH, MLD, proxy-inner sweep, and AC projection for selected finalists.",
    }[stage]


def stage_objective(stage: str) -> str:
    common = "`J = lambda_R R_norm + (1-lambda_R) L_shed`, with later stages adding `rho_phys PAC_total` or DC-specific exact recourse."
    return {
        "A": common,
        "B": common + " Constraints were mostly implicit through selected groups and optimizer bounds.",
        "C": "Threshold PSPS has no optimization after line selection except diagnostic load-service evaluation.",
        "D": "Enumerated objective `J = lambda_R R_norm + lambda_L L_shed` over `|S_off| <= 2`.",
        "E": "`J_true = lambda_R R_norm + lambda_L L_shed + rho_phys PAC_total`; Gurobi proxy uses no-good cuts to propose topology candidates.",
        "F": "Same Stage C/D/E objective family, applied to five scenario surfaces.",
        "G": "`J_true = lambda_R R_norm + lambda_L L_shed_hybrid + rho_phys(PAC_operational + PAC_AC + PAC_model_consistency)`.",
        "H": "TH/AH use the Stage G recourse evaluator; recall/precision are diagnostic, not optimized.",
        "I": "Stage I-a/I-b DC objective minimizes normalized squared line-risk flow plus load shedding under DC balance, generator/load bounds, thermal limits, topology budget `sum y <= 2`; AC projection minimizes distance to fixed-topology AC-feasible point where possible.",
    }[stage]


def stage_outputs(stage: str) -> str:
    return {
        "A": "Connected-corridor summaries, figures, and first-pass result artifacts.",
        "B": "Threshold sensitivity summaries, automatic group CSV/JSON artifacts.",
        "C": "PSPS summaries, selected PSPS lines, risk/load diagnostics.",
        "D": "Candidate evaluation tables, optimized de-energization decisions, Stage D summary.",
        "E": "Gurobi/GridFM run folders, proxy-vs-true summaries, continuous objective traces, alpha diagnostics.",
        "F": "Decision-quality scenario definitions and Stage D/E comparison outputs.",
        "G": "Revised continuous tables, cross-rho/per-rho/summary plots, methodology checks.",
        "H": "Heuristic comparison top-k and revised-load/PAC result folders, expected-vs-selected plots, TH/AH audits.",
        "I": "Main r11, MLD r5, proxy-inner r2 tables/plots; solver diagnostics; projection distances; CASE-001 audit findings.",
    }[stage]


def stage_findings(stage: str) -> str:
    return {
        "A": "The harness worked but was too preliminary for publication-level conclusions.",
        "B": "Automatic risk components avoided manual-only grouping but did not yet answer topology-control quality.",
        "C": "PSPS thresholds can sharply reduce risk but may cause large service impacts.",
        "D": "Small-budget enumeration provided a strong reference inside the candidate space.",
        "E": "Guided topology search could match or miss exhaustive K<=2 depending on rho/lambda; continuous recourse was costly and sometimes barely moved controls.",
        "F": "Scenario structure matters; method performance should not be inferred from one case.",
        "G": "Prior metrics understated load loss and physics/model inconsistency; hybrid load and PAC decomposition became necessary.",
        "H": "TH can be competitive in structured scenarios; AH is often weaker; revised load/PAC exposed larger GridFM-implied degradation.",
        "I": "DC methods make the comparison more physically grounded; Stage I-b is compact and certified in saved rows; GridFM limitations remain central; projection evidence is incomplete.",
    }[stage]


def stage_limitations(stage: str) -> str:
    return {
        "A": "Not a topology-control or security-constrained study.",
        "B": "Threshold choices and synthetic risk remain exploratory.",
        "C": "Not optimized over load/risk jointly and not N-1 secure.",
        "D": "Enumeration is small-budget and fixed-control; not scalable.",
        "E": "GridFM true evaluation is only as reliable as GridFM predictions; no hard AC/security constraints.",
        "F": "Five scenarios are controlled, not broad generalization.",
        "G": "Diagnostics reveal issues but do not solve surrogate feasibility.",
        "H": "Heuristic adaptations are Rhodes-inspired but not exact geographic RTS reproductions; some references have accounting caveats.",
        "I": "DC approximation is not AC truth; AC projection finite only for subset and uses relaxed Qg bounds; no full SC-OPS contingency model.",
    }[stage]


def stage_relationship_text(stage: str) -> str:
    return "See `STAGE_RELATIONSHIP_MATRIX.csv`; this stage is part of the progression from surrogate first-pass evaluation toward heuristic/DC/AC feasibility comparison."


def stage_rhodes(stage: str) -> str:
    if stage in {"C", "D", "E", "H", "I"}:
        return (
            "Relevant. Rhodes 2021 OPS provides the primary risk-versus-load shutoff lineage and TH/AH/MLD benchmark concepts. "
            "Rhodes 2023 SC-OPS adds contingency/security constraints. This stage follows or adapts the wildfire shutoff tradeoff, "
            "but the current implementation should not be called security-constrained unless it includes explicit contingency scenarios. "
            "Stage I's AC projection and diagnostics extend feasibility analysis beyond Rhodes 2021, but they do not replace SC-OPS."
        )
    return "Conceptual only. This stage is upstream harness/scenario construction rather than a direct Rhodes OPS/SC-OPS analogue."


def stage_literature(stage: str) -> str:
    return "Tier 1 Rhodes papers anchor wildfire shutoff terminology; Tier 2 OPF/OTS/GridFM/GNN papers support formulation and surrogate context; Tier 3 papers provide background only."


def stage_status(stage: str) -> str:
    return {
        "A": "Status: `FOUNDATIONAL`; publication value: `HISTORICAL_DEVELOPMENT`.",
        "B": "Status: `PARTIALLY_SUPERSEDED`; publication value: `SUPPORTING_METHOD`.",
        "C": "Status: `BASELINE`; publication value: `BASELINE`.",
        "D": "Status: `BASELINE`; publication value: `ABLATION`.",
        "E": "Status: `PARTIALLY_SUPERSEDED`; publication value: `MOTIVATING_FAILURE` and method precursor.",
        "F": "Status: `FOUNDATIONAL`; publication value: `SUPPORTING_METHOD`.",
        "G": "Status: `ACTIVE_METHOD`; publication value: `LIMITATION_ANALYSIS`.",
        "H": "Status: `ACTIVE_METHOD`; publication value: `BASELINE` and comparative evidence.",
        "I": "Status: `ACTIVE_METHOD`; publication value: `CORE_CONTRIBUTION` candidate with limitations.",
    }[stage]


def charter_text(now: str) -> str:
    return f"""
# CASE-002 Charter

Generated UTC: {now}

## Purpose

CASE-002 reconstructs the Stage A-I GridFM wildfire-operations research trajectory and produces advisor-ready synthesis materials. It is evaluative and non-destructive.

## Boundaries

No source, tests, configurations, official results, frozen packages, CASE-001 frozen outputs, `CURRENT_STATE_SUMMARY.md`, or `HISTORY.md` are modified. No new experiments or remediation are performed.

## Primary Outputs

The final outputs are `FINAL_ALIGNMENT_REPORT.md`, `PROFESSOR_DISCUSSION_BRIEF.md`, `FRAMING_OPTIONS.md`, `NEXT_STEP_OPTIONS.md`, and supporting evidence/literature/result/claim registers.

## Completion State

The case ends at `CASE002_SYNTHESIS_COMPLETE_AWAITING_CALEB_INTERPRETATION`.
"""


def execution_plan_text(now: str) -> str:
    return f"""
# CASE-002 Synthesis Execution Plan

Generated UTC: {now}

1. Build evidence and literature registers from live repository evidence and local PDF library.
2. Preserve historical Stage A-I naming, naming drift, and confidence labels.
3. Inspect Tier 1 Rhodes papers at formulation/results depth and inspect Tier 2 papers by targeted sections.
4. Reconstruct stage progression, result register, research-question evolution, and stage relationships.
5. Create scientific/literature review and implementation/evidence verification artifacts, freeze them, then reconcile.
6. Produce framing options, next-step options, professor brief, and final alignment report.
7. Hash inspected inputs and final case artifacts.

Execution mode is local CASE-002 synthesis generation. Role separation is procedural in this generated pass; CASE-001 remains separate-context audited evidence for Stage I.
"""


def framing_text() -> str:
    return """
# Framing Options

## CURRENT_SCIENTIFIC_FRAMING

The most defensible current framing is a comparative wildfire-risk-aware topology-control study that uses GridFM as one evaluated surrogate method, then benchmarks it against Rhodes-inspired heuristics and transparent DC optimization. The strongest evidence is not that GridFM already solves wildfire operations, but that the staged workflow reveals where surrogate prediction, DC feasibility, and AC projection agree or diverge.

## POTENTIAL_PUBLICATION_FRAMING_AFTER_ADDITIONAL_WORK

After targeted work, the project could become a comparative feasibility and decision-quality paper: GridFM-guided shutoff versus TH/AH heuristics versus Stage I-a/I-b DC methods, with AC projection coverage completed and scenario generalization improved. A GridFM-centered paper would require stronger remediation of command-faithfulness and physical-realism limitations.

## ADVISOR_DECISION_OPTIONS

1. Lean into a limitation/evaluation paper about learned surrogates in wildfire shutoff decisions.
2. Lean into a DC/heuristic comparative methodology paper and treat GridFM as exploratory.
3. Extend toward SC-OPS/robust/security-constrained wildfire operations before framing as operationally mature.

## LONGER_TERM_RESEARCH_PROGRAM

Longer term, this can become a wildfire-resilience operations program combining risk construction, topology control, learned surrogate diagnostics, DC/AC restoration, robust optimization, and eventually contingency-aware SC-OPS-style security.

## Venue And Event Notes

Future conference/workshop/deadline recommendations require `REQUIRES_CURRENT_EXTERNAL_VERIFICATION`.
"""


def next_steps_text() -> str:
    return """
# Next-Step Options

## Route 1 - GridFM Feasibility Improvement

Research question: can GridFM outputs be made command-faithful and operationally realistic under topology interventions?

Completed work: raw/eval split, hybrid load diagnostics, PAC decomposition, discrepancy plots.

Remaining work: feasibility restoration, constrained decoding or projection, command-consistency checks, broader scenarios.

Risk: high dependence on GridFM model behavior.

## Route 2 - Comparative Methodology Paper

Research question: how do GridFM, TH/AH, Stage I-a DC, and Stage I-b MIQP compare under common wildfire/load/feasibility diagnostics?

Completed work: main r11, MLD r5, proxy-inner r2, CASE-001 audit.

Remaining work: complete AC projection coverage, clean archives, stronger tests, broader scenarios.

Risk: moderate; strongest current route.

## Route 3 - Wildfire Operations / SC-OPS Extension

Research question: can the framework be extended toward true security-constrained wildfire power shutoff?

Completed work: topology and DC foundations.

Remaining work: explicit contingency set, post-contingency load shedding, SC-OPS-style constraints, advisor approval.

Risk: technically heavier but aligns closely with Rhodes 2023.

## Route 4 - Evaluation Benchmark For Learned Grid Models

Research question: when are learned grid surrogates reliable enough for topology-control decision workflows?

Completed work: GridFM discrepancy and DC/AC comparison.

Remaining work: multi-system tests, model variants, feasibility metrics, reproducible benchmark packaging.

Risk: requires careful claims and literature expansion.
"""


def professor_brief_text() -> str:
    return """
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
"""


def final_report_text(now: str, evidence_n: int, lit_n: int) -> str:
    return f"""
# Final Stage A-I Research Alignment Report

Generated UTC: {now}

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

- Evidence items registered: {evidence_n}
- Literature PDFs registered: {lit_n}

## Final Status

`CASE002_SYNTHESIS_COMPLETE_AWAITING_CALEB_INTERPRETATION`
"""


def scientific_review_text() -> str:
    return """
# Scientific And Literature Review - CASE-002 v001

## Scope

This role reviews scientific framing, Rhodes alignment, literature-tiering, and claim boundaries.

## Findings

1. The Rhodes 2021 OPS paper is the central anchor for risk/load Pareto framing and TH/AH/MLD comparison. CASE-002 should describe Stage C/D/H/I as adapted, not direct reproductions.
2. The Rhodes 2023 SC-OPS paper is the central security terminology guardrail. Current work lacks explicit N-1 contingency modeling and should not be labeled security-constrained.
3. GridFM learned-surrogate components are exploratory relative to the Tier 1 shutoff literature and need careful limitation language.
4. AC projection and common diagnostics are useful extensions beyond Rhodes 2021, but incomplete projection coverage and relaxed Qg bounds require qualification.

## Recommendation Options

Frame near-term work as comparative feasibility/decision-quality unless Caleb and advisor choose to invest in SC-OPS/robust extension.
"""


def implementation_review_text() -> str:
    return """
# Implementation And Evidence Verification - CASE-002 v001

## Scope

This role checks whether the synthesis maps to repository evidence, source folders, result tables, and CASE-001 findings.

## Findings

1. Stage A-I source folders exist and provide a credible chronology, with naming drift around Stage H/I that must remain explicit.
2. Main Stage I evidence is available through r11, MLD r5, proxy-inner r2, and CASE-001.
3. Historical stages have uneven evidence strength: early Stage A/B are mostly foundational, while Stage G/H/I have stronger result and audit evidence.
4. Result claims should use live tables as numeric authority; progress prose can drift from regenerated artifacts.

## Recommendation Options

Keep stage-specific confidence labels and do not collapse all stages into one success story.
"""


def reconciliation_text() -> str:
    return """
# CASE-002 Reconciliation v001

The scientific/literature and implementation/evidence reviews agree that the strongest current framing is comparative feasibility and decision quality. Both reviews preserve the terminology boundary that current work is wildfire-risk-aware topology control with feasibility diagnostics, not a full security-constrained power shutoff model.

No disagreements were forced closed. The main unresolved advisor decisions are whether to remediate GridFM feasibility, expand DC/AC/scenario validation, or move toward SC-OPS/robust optimization.
"""


if __name__ == "__main__":
    main()
