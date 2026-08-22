# Reconciliation Stage J v001

```yaml
artifact_id: CASE-003-RECONCILIATION-STAGE-J
artifact_version: v001
created_local: 2026-08-12T22:58:28.1336018-04:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: RECONCILED_AFTER_BOOTSTRAP
sha256_when_frozen: null
```

## Reconciled Status

The prior v002 contradictions were corrected in `STAGE_J_PRIMARY_DESIGN.md` v003:

```text
per-load alpha is locked
GridSFM selection includes rho_phys PAC_total
Guided-DC is fixed-(z,alpha) economic DC-OPF
Reference A is fixed-(z,alpha) economic AC-OPF
D_input, PAC_model, and D_state_to_AC are separate
AC/GridSFM loading uses both branch ends
unit compatibility and official preprocessing are hard gates
```

J0.5 bootstrap is now validated for the official external dependencies and examples. The implementation auditor findings remain visible and unresolved where they point to future gates rather than completed work.

## Open Blockers

`BLOCKER`: GOC-500 branch identity adapter and raw mutation pipeline.

`BLOCKER`: fixed-`(z,alpha)` economic DC-OPF backend.

`BLOCKER`: exact GOC-500 economic AC baseline and finalist AC audits.

`NEEDS_PATCH`: richer automatic unit-provenance checking.

These blockers do not invalidate J0.5, but they block the S1-S3 wildfire experiment.
