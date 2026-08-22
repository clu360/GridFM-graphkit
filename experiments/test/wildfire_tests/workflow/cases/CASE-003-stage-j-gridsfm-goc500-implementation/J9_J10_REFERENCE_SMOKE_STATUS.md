# J9/J10 Reference and Warm-Start Smoke Status

Created: 2026-08-17

## Status

`PASS_WITH_LIMITATIONS`

The exact-reference and warm-start plumbing was exercised successfully on the
two method-selected J8 finalists only:

```text
scenario: J-S1
lambda_R: 0.8
Guided methods: Guided-DC, Guided-GridSFM
J8 finalist topology for both methods: {276, 473}
Reference B requested: yes
```

This is a post-hoc audit. No Reference A, Reference B, or warm-start result was
fed back into the Stage J topology or alpha search.

## Artifacts

The valid v005 result root is:

```text
experiments/test/wildfire_tests/goc_500_results/stage_j/
  j9_j10_smoke/s1_l08_v005/
```

It contains 99 files, including:

```text
reference_a_summary_table.csv
reference_a_state_fidelity_metrics.csv
reference_a_warm_start_summary.csv
reference_b_summary_table.csv
generator_identity_ranges.csv
j9_j10_reference_smoke_summary.json
```

The summary reports:

```text
Reference A rows:       2 / 2 LOCALLY_SOLVED
State-fidelity rows:   16
Warm-start rows:        8 / 8 LOCALLY_SOLVED
Reference B rows:       4 / 4 LOCALLY_SOLVED
```

## Exact Reference Semantics Verified

### Reference A

For each method-selected finalist, the wrapper:

```text
fixes topology z
fixes effective alpha
uses original Pg and Qg bounds
solves economic AC-OPF with PowerModels/Ipopt
exports Pg, Qg, V, theta, and both-end branch P/Q flows
```

Both Reference A solves were locally solved in approximately 6.8 solver seconds.
The fixed alpha rule held exactly at reporting precision:

```text
Guided-DC      L_shed_native = L_shed_Reference_A = 0.0092953921
Guided-GridSFM L_shed_native = L_shed_Reference_A = 0.0092208011
```

The realized wildfire tradeoff was:

```text
method           native R_norm  AC R_norm      native J_trade  AC J_true
Guided-DC        0.34788280     0.39369845     0.28016532      0.31681784
Guided-GridSFM   0.45375378     0.39379447     0.36484718      0.31687974
```

Thus, on this single finalist:

```text
Guided-DC underestimates AC R_norm by 0.04581565.
Guided-GridSFM overestimates AC R_norm by 0.05995930.
```

These are instance-level fidelity observations, not a general method ranking.

### Reference B

Reference B uses the same fixed topology but releases load service, enforces
the source-less rule, uses fixed shunts, and permits `0 <= Pg <= Pgmax` only in
this restoration solve. B1 maximizes served active load; B2 preserves B1
service within a declared numerical tolerance and minimizes generation cost.

Both method finalists have the same selected topology. Consequently, their
Reference B capability results are identical, as expected:

```text
B1: L_shed_AC_MLD = 0.0, R_norm_AC_MLD = 0.36950873
B2: L_shed_AC_MLD = 0.0000000663, R_norm_AC_MLD = 0.40922206
```

The B2 service lock is explicitly saved:

```text
configured B1-to-B2 service tolerance: 1.0e-5 Pd units
raw B2 constraint residual:             1.7773e-6 Pd units
solver-tolerance-qualified verification: true
B2 economic objective:                  445867.60032755
```

The raw constraint indicator remains false because IPOPT leaves the recorded
positive feasibility residual. That residual is not hidden; it is separately
compared against the independently saved `1.0e-5` solver verification tolerance.

The B1 service-recovery value differs only because the two methods selected
different fixed alpha vectors before Reference B releases alpha.

## State Fidelity and Warm Start

The state-fidelity table reports raw and normalized family-level errors. DC is
correctly marked `N/A` for Qg, V, and reactive branch flows because it does not
represent those quantities. GridSFM has populated Pg, Qg, V, aligned theta,
and both-end P/Q branch-flow metrics.

The warm-start comparison solved the identical fixed-z/fixed-alpha Reference A
instance in all cases. It saves solver-only runtime, Julia subprocess wall time,
measured regenerated-start construction time, and end-to-end time. Solver-only
runtimes were:

```text
method           cold       DC start   GridSFM start  GT start
Guided-DC        6.822 s    6.484 s    6.583 s        6.342 s
Guided-GridSFM   6.817 s    6.527 s    6.510 s        6.348 s
```

All four starts for each method reached the same AC objective to numerical
precision. Partial starts produce small, instance-dependent changes while the
GT start is a practical initialization ceiling. This is not yet a statistical
performance conclusion.

## Implementation Fix and Audit

The initial J9/J10 invocation produced one Guided-DC Reference A and then
failed exporting Guided-GridSFM artifacts. The failure was a Windows legacy
path-length boundary:

```text
Guided-DC reference-A branch path:      257 characters
Guided-GridSFM reference-A branch path: 262 characters
```

The runner now uses short internal artifact stubs (`gdc`, `gsfm`, `ra`, `rb1`,
`rb2`, `ws`) while retaining full method names in all result tables. The
original limited `s1_l08` directory and intermediate v002-v004 runs are
retained as diagnostic evidence; v005 is the valid smoke output.

## Remaining Limitations Before a Full J9/J10 Sweep

1. PowerModels/Ipopt iteration counts are not yet exposed as structured fields.
2. GridSFM start construction is measured end-to-end but not yet split into
   preprocessing and inference subcomponents.
3. The detailed family metrics are primary. A single composite
   `D_state_to_AC` is intentionally not emitted because its weighting has not
   been locked; the design permits it only as an optional, documented summary.
4. Risk-branch-specific fidelity summaries and relative objective-fidelity
   columns should be added before presenting a full multi-scenario study.
5. This smoke covers only the two guided finalists. TH-GridSFM is intentionally
   excluded from J9/J10 under the current scope decision.

## Implementation Outcome

The fixed-z/fixed-alpha economic AC reference, fixed-z AC restoration, and
cold/DC/GridSFM/GT warm-start interfaces now work on real J8 outputs. An
independent implementation audit found no critical formulation violation. The
next methodological step is to add the remaining report columns/plots, then
extend the same post-hoc audit to the final selected results of the approved
Stage J comparison without changing the frozen search semantics.
