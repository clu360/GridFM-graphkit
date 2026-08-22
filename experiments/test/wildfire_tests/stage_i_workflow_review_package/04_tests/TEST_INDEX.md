# Test Index

## test_wildfire_stage_i_dc_comparison.py

- Original path: `tests/test_wildfire_stage_i_dc_comparison.py`
- Copied path: `experiments/test/wildfire_tests/stage_i_workflow_review_package/04_tests/stage_i_tests/test_wildfire_stage_i_dc_comparison.py`
- Test names: test_dc_network_uses_matpower_rate_a_and_tap_shift_audit; test_common_operational_diagnostic_zero_for_empty_feasible_values; test_methodology_checks_reject_more_than_two_shutoffs
- Behavior tested: Stage I/DC comparison or related Stage G/H regression behavior
- Deterministic: appears deterministic unit/regression test
- Requires Gurobi: not directly in copied unit tests unless imported code requires it
- Requires GridFM: not directly in copied unit tests unless imported code requires it
- Requires external data: repository test fixtures/imports

## test_wildfire_stage_h_heuristic_comparison.py

- Original path: `tests/test_wildfire_stage_h_heuristic_comparison.py`
- Copied path: `experiments/test/wildfire_tests/stage_i_workflow_review_package/04_tests/related_regression_tests/test_wildfire_stage_h_heuristic_comparison.py`
- Test names: test_transmission_heuristic_selects_scores_at_or_above_percentile; test_transmission_heuristic_keeps_ties_at_percentile_cutoff; test_area_heuristic_top_fraction_uses_ceiling_count; test_transmission_heuristic_top_k_uses_ranked_scores_with_line_id_tiebreak; test_transmission_heuristic_top_k_clamps_to_available_candidates; test_area_heuristic_groups_connected_components_and_selects_highest_average; test_stage_h_summary_preserves_stage_d_stage_e_th_and_ah_methods
- Behavior tested: Stage I/DC comparison or related Stage G/H regression behavior
- Deterministic: appears deterministic unit/regression test
- Requires Gurobi: not directly in copied unit tests unless imported code requires it
- Requires GridFM: not directly in copied unit tests unless imported code requires it
- Requires external data: repository test fixtures/imports

## test_wildfire_stage_g_revised_continuous.py

- Original path: `tests/test_wildfire_stage_g_revised_continuous.py`
- Copied path: `experiments/test/wildfire_tests/stage_i_workflow_review_package/04_tests/related_regression_tests/test_wildfire_stage_g_revised_continuous.py`
- Test names: test_pg_qg_alpha_decision_vector_updates_controlled_features; test_controlled_feature_mask_marks_pg_qg_pd_qd_only; test_clamp_prediction_overwrites_controlled_values_only; test_controlled_state_audit_separates_raw_gridfm_from_objective_values; test_load_shedding_provenance_sums_to_effective_alpha_objective; test_load_shedding_provenance_counts_source_less_buses_as_fully_shed; test_physics_components_decomposes_pac_and_flags_branch_consistency_unavailable; test_wildfire_risk_provenance_excludes_impact_and_sums_to_raw_risk; test_normalize_stages_supports_stage_e_unconstrained_zoom; test_run_signature_records_lambda_zoom_stage_filter; test_expected_vs_observed_keeps_lambda_zoom_values
- Behavior tested: Stage I/DC comparison or related Stage G/H regression behavior
- Deterministic: appears deterministic unit/regression test
- Requires Gurobi: not directly in copied unit tests unless imported code requires it
- Requires GridFM: not directly in copied unit tests unless imported code requires it
- Requires external data: repository test fixtures/imports
