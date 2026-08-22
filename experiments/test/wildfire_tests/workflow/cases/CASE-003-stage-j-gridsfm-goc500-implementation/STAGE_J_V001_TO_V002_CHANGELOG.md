# Stage J v001 To v002 Changelog

```yaml
artifact_id: CASE-003-STAGE-J-V001-TO-V002-CHANGELOG
artifact_version: v001
created_utc: 2026-08-12T00:00:00+00:00
execution_mode: single_context_role_simulation
blind_context_enforced: false
reviewer_prior_outputs_visible: true
status: DRAFT_FOR_CALEB_REVIEW
sha256_when_frozen: null
```

## Purpose

This changelog records the substantive changes made to `STAGE_J_PRIMARY_DESIGN.md` when revising the draft from v001 to v002.

## Major Changes

1. Reframed Stage J as the primary controlled comparison between Guided-DC and Guided-GridSFM, with heuristics, DC MIQP, exact AC audits, and warm-starts as secondary/reference comparisons.

2. Locked the Stage J control hierarchy:

```text
wildfire-side controls = z, alpha
GridSFM recourse outputs = Pg, Qg, V, theta, P/Q branch flows
```

3. Replaced `Pd_base` / `Qd_base` conceptually with scenario pre-intervention requested demand:

```text
Pd_pre[i,s]
Qd_pre[i,s]
```

4. Removed the old GridFM hybrid-load accounting from Stage J. Native Stage J load shedding is now model-independent and computed from `alpha_eff` and `Pd_pre`.

5. Strengthened source-less island handling as a hard preprocessing invariant and independent post-evaluation assertion.

6. Locked exact intact AC baseline loading as the shared risk normalization source.

7. Preserved the existing outer wildfire proxy structure, including:

```text
R_proxy(y)
L_proxy(y) = sum_l c_l y_l
```

with `c_l` used only as topology-search guidance.

8. Locked the primary heuristic as simple baseline wildfire-risk ranking:

```text
score_TH[l] = p_env[l] baseline_loading[l]^2
```

without multiplying by `c_l`.

9. Reduced the first diagnostic scenario suite from S1-S5 to S1-S3.

10. Reframed `C` and `K <= 2` as experimental search restrictions rather than intrinsic OPS assumptions.

11. Split exact AC references into:

```text
Reference A: fixed-z-alpha exact AC state/feasibility reference
Reference B: fixed-z Rhodes-style AC redispatch / MLD audit
Reference C: warm-start computational reference
```

12. Added generator-flexibility audit requirements for disconnected exact AC redispatch cases.

13. Preserved the three diagnostic families from earlier work:

```text
operational diagnostics
AC-physics consistency
model / interface consistency
```

14. Added recourse-change metrics:

```text
Delta_L_shed
Delta_alpha
Delta_Pg
Delta_Qg
Delta_V
Delta_flow
```

15. Added topology-specific warm-start headroom:

```text
eta_m = (T_cold - T_m) / (T_cold - T_GT)
```

16. Strengthened the branch identity implementation gate with orientation, family, Pij/Pji, Qij/Qji, and risk-attachment checks.

17. Updated environment naming to distinguish:

```text
GRIDFM_GRAPHKIT_ROOT = existing research repository
GRIDSFM_ROOT = Microsoft GridSFM checkout
```

18. Expanded data schemas with scenario, hash, baseline, impact-proxy, diagnostic, recourse-change, and exact-AC problem-type fields.

19. Revised plotting terminology, including renaming target-line plots to `Diagnostic Target Agreement`.

20. Replaced unresolved clarification questions with locked Stage J v1 decisions plus implementation-audit items.
