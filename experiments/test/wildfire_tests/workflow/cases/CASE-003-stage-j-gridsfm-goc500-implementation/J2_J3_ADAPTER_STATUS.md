# J2/J3 GOC-500 Adapter And Preprocessing Status v001

```yaml
artifact_id: CASE-003-J2-J3-ADAPTER-STATUS
artifact_version: v001
created_local: 2026-08-12T23:24:00-04:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: PASS_WITH_NEXT_GATES
sha256_when_frozen: null
```

## Summary

J2/J3 is implemented for the official GridSFM `case500_goc.pyg.json` sample.

Implemented:

```text
canonical branch identity
canonical load identity
AC-line / transformer family-index mapping
finite positive rateA guard for risk branches
source-less load detection from generator-containing components
raw .pyg.json mutation before official GridSFM preprocessing
full per-load alpha_requested and alpha_effective metadata
L_shed_total / L_shed_control / L_shed_island decomposition
hard D_input / input-integrity failure checks
```

## Official GOC-500 Adapter Smoke

Input:

```text
C:/Users/Caleb Lu/.gridfm_stage_j/repos/GridSFM/model/samples/case500_goc.pyg.json
```

Mutation:

```text
offline_branch_id = 1
uniform alpha_i = 1.0 for all 281 loads
```

Observed:

```text
branches before = 728
AC lines after = 535
transformers after = 192
GridSFM predicted flow rows after mutation = 727
source-less loads = 0
L_shed_total = 0.0
L_shed_control = 0.0
L_shed_island = 0.0
D_input = 0.0
GridSFM feasibility probability = 0.9987537860870361
```

This confirms the candidate mutation is applied to raw JSON before official GridSFM loading/preprocessing/inference.

## Identity Alignment With PGLib

The official GridSFM sample metadata contains 728 active branch IDs. The official PGLib `pglib_opf_case500_goc.m` has 733 branch rows, with inactive row IDs:

```text
49, 58, 210, 504, 550
```

These are exactly the five IDs absent from the GridSFM active branch metadata. This supports using PGLib row IDs as canonical physical branch IDs for Stage J.

## Remaining Gate

The adapter currently validates schema, identity, topology mutation, and load mutation. It does not yet compute wildfire scenarios, PAC calibration, exact finalist audits, or full outer search.
