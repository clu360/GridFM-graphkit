# FT6/FT7 Sequential Fine-Tuning Ablation Plan

## Status

```text
plan_status: FT7_COMPLETE_AWAITING_PAPER_REVIEW
FT6: COMPLETE_AND_VALIDATED
FT7: COMPLETE_AND_VALIDATED
```

Completed FT6 checkpoints:

```text
M2 SHA-256: 08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6
M3 SHA-256: 4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D
review report: FT6_SEQUENTIAL_FINETUNE_STATUS.md
```

This document defines the concluding Stage J fine-tuning ablation. It extends
the completed FT0 through FT5 evidence without changing or replacing those
artifacts. The study is divided into two separately gated parts:

1. FT6 trains two sibling continuation checkpoints and evaluates four GridSFM
   checkpoints on one sealed 750-case diagnostic test set.
2. FT7 applies only the two new checkpoints to the frozen Stage J OPS
   methodology, joins existing evidence, and updates the exact and warm-start
   comparisons.

FT7 must not begin automatically. FT6 closes in
`AWAITING_CALEB_FT7_APPROVAL`, even if every internal validation passes.

Caleb explicitly approved FT7 on 2026-08-24. The decision is recorded in
`FT7_APPROVAL.json`. The fail-closed preflight passed before M2 execution and
is recorded under `goc_500_results/stage_j/refined_finetune_study`.

## Scientific Objective

The extension asks whether the behavior of GridSFM in the Stage J OPS setting
depends on additional same-topology exposure or on explicit N-1 topology
exposure. It is a model-adaptation ablation, not a new OPS formulation and not
an architecture comparison.

The four-model OPFData study is:

| ID | Model variant | Parent | Additional training |
| --- | --- | --- | --- |
| M0 | `released_v1_1` | released GridSFM v1.1 | none |
| M1 | `fulltop_ft_n1000` | released GridSFM v1.1 | existing 1,000 FullTop cases |
| M2 | `fulltop_ft_n1000_then_fulltop_n500` | M1 | 500 new FullTop cases |
| M3 | `fulltop_ft_n1000_then_n1_n500` | M1 | 500 N-1 cases |

M2 and M3 are sibling continuations from the byte-identical M1 checkpoint:

```text
M1 checkpoint SHA-256:
A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD
```

M2 must never initialize from M3, and M3 must never initialize from M2.
Because the checkpoint retains model weights but not the original optimizer
state, both continuations use separately initialized but identically configured
optimizers. The models are described as sequential continuations, not as fresh
models jointly trained on shuffled 1,500-case datasets.

The planned contrasts are:

- M0 versus M1: initial GOC-500 FullTop fine-tuning;
- M1 versus M2: 500 additional same-distribution cases and equal continuation;
- M2 versus M3: same parent, additional case count, epochs, batches, optimizer,
  and evaluation data, with only the added topology distribution changed;
- M1 versus M3: incremental effect of N-1-aware continuation.

## Frozen Dependency And Method Boundary

The existing official dependency boundary remains in force. Training,
preprocessing, checkpoint loading, and canonical aggregate evaluation continue
through the pinned Microsoft/GridSFM implementation at commit:

```text
1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6
```

The wrapper may orchestrate datasets, checkpoint provenance, per-epoch metric
capture, result joining, and validation. It must not duplicate or alter the
GridSFM architecture, loss, OPFData adapter, topology transforms, optimizer
loop, or official metric definitions. The existing private-helper source hashes
and pinned environment must pass before either training branch starts.

The released checkpoint and M1 checkpoint path and SHA-256 must be recorded
directly in every FT6 manifest. Each child checkpoint path and SHA-256 must be
added immediately after it is written and verified by fresh-process reload.

## Part I: FT6 Training And Four-Model Evaluation

### FT6.0 Protocol And Manifest Freeze

Before model or data loading, write an immutable FT6 protocol manifest that
records:

- all four model IDs and display labels;
- released and M1 checkpoint paths and SHA-256 values;
- planned M2/M3 checkpoint paths;
- GridSFM commit, environment freeze, and private-helper source hashes;
- training, validation, and test split identities and ordered indices;
- seed, batch size, epochs, learning rate, weight decay, and infeasibility mix;
- explicit optimizer-reset behavior at M2 and M3 initialization;
- canonical metric list and checkpoint-selection rule;
- output, cache, result, and quarantine roots;
- code commit and dirty-worktree disclosure;
- no-overwrite and resume policy.

Internal checkpoint:

```text
FT6_P0_PROTOCOL_FROZEN
```

Any later change creates a new protocol version. It must not silently mutate the
frozen manifest.

### FT6.1 Disjoint Data Contract

The continuation training sets are:

```text
M2: variant=fulltop, split=train, ordered indices 1000..1499
M3: variant=n1,      split=train, ordered indices    0..499
```

M2 FullTop indices do not overlap the M1 FullTop training indices `0..999`.
M3 changes the topology variant and uses only the OPFData training split.

Both training branches receive the same held-out per-epoch validation suite:

```text
FullTop validation: variant=fulltop, split=val, indices 0..374
N-1 validation:     variant=n1,      split=val, indices 0..374
validation total: 750 graphs per epoch
```

The validation sets monitor same-topology retention and N-1 transfer. They are
not training data. The default checkpoint-selection rule is the fixed final
epoch, matching FT1 and preventing retrospective selection after test results.
Per-epoch metrics for both validation strata remain mandatory.

The sealed diagnostic test set is separate from training and validation:

```text
FullTop test: variant=fulltop, split=test, indices 0..374
N-1 test:     variant=n1,      split=test, indices 0..374
test total: 750 graphs
```

These test cases are a frozen subset of the existing FT2 sealed test splits.
They were not used by FT0/FT1 training or validation and may not be used for
FT6 training, epoch selection, hyperparameter changes, or reruns conditioned on
observed model performance.

The data manifest must store ordered graph identity hashes, variant, split,
group, index, count, and cache source. It must explicitly validate:

- 500 unique training graphs per continuation branch;
- 375 unique graphs in each validation stratum;
- 375 unique graphs in each test stratum;
- no train/validation/test identity overlap;
- identical validation and test ordering for M0 through M3;
- finite and schema-compatible graph tensors;
- no download requirement when the existing processed caches pass hashes.

Internal checkpoint:

```text
FT6_P1_DATA_CONTRACT_PASS
```

### FT6.2 Sibling Continuation Training

Unless a versioned amendment is approved before FT6 execution, both M2 and M3
inherit the FT1 training controls:

```text
batch size: 8
epochs: 10
learning rate: 1e-4
weight decay: 1e-4
SyntheticMixedDataset infeas_prob: 0.3
seed: 42
device: CPU/auto resolved and recorded
optimizer loop: official gridsfm.finetune_opfdata
```

Run order must not alter semantics. Each branch reloads M1 in a fresh process,
verifies the expected parent SHA, creates a fresh optimizer, trains only its
specified 500-case continuation set, and evaluates both 375-case validation
strata after every epoch.

Each branch writes:

- a resolved config and environment record;
- a parent checkpoint identity record;
- ten epoch rows with training and both validation-stratum metrics;
- elapsed and epoch runtimes, iteration counts, and skipped-batch counts;
- parameter-change and finite-value audits;
- child checkpoint path, size, and SHA-256;
- fresh-process strict reload and output-schema checks;
- a resumable status that cannot confuse M2 and M3 artifacts.

Internal checkpoints:

```text
FT6_P2A_M2_TRAINING_PASS
FT6_P2B_M3_TRAINING_PASS
```

A worse metric, feasibility-head change, or evidence of specialization is a
scientific result and does not fail training. Hash drift, split leakage,
nonfinite weights, missing epoch metrics, skipped batches, or reload/schema
failure stops the workflow.

### FT6.3 Four-Model Sealed Evaluation

After both child checkpoints pass, evaluate M0, M1, M2, and M3 on the exact same
ordered 750-case test manifest. All four models must be run in the same pinned
environment through official GridSFM evaluation code.

Results must be reported at three levels:

1. FullTop test stratum, 375 graphs;
2. N-1 test stratum, 375 graphs;
3. equal-weight combined diagnostic set, 750 graphs.

The canonical table retains official loss, cost MAPE, Pg/Qg/V/theta MAE,
branch-P/branch-Q MAE, KCL residuals, thermal loading metrics, feasibility-head
accuracy, graph count, and runtime. Combined metrics must be recomputed from
equal stratum counts or documented sufficient statistics, not by averaging
percent changes.

Required comparisons include absolute values and changes for M0->M1, M1->M2,
M1->M3, and M2->M3. Feasibility-head results remain separate from continuous
regression and physics metrics because all retained OPFData cases are labeled
feasible. Any per-case paired statistics are supplemental; official aggregate
metrics remain canonical.

The evaluator writes one comparison manifest containing all four checkpoint
paths and hashes, the shared test-manifest hash, ordered execution records,
output-schema checks, and a fresh recomputation of every displayed table.

Internal checkpoint:

```text
FT6_P3_FOUR_MODEL_EVALUATION_PASS
```

### Primary External Checkpoint A

FT6 concludes by producing a concise scientific review packet with:

- training trajectories for M2 and M3 on both validation strata;
- four-model FullTop, N-1, and combined test tables;
- checkpoint and data provenance;
- same-topology retention, N-1 transfer, and feasibility-head interpretation;
- immediate divergences and failed claims;
- measured runtime and an updated FT7 projection.

The terminal status is always:

```text
FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL
```

No FT7 OPS candidate evaluation, exact reference, warm-start solve, joined
figure, or publication may begin until Caleb explicitly reviews FT6 and
approves FT7. Internal FT6 success is not approval.

## Part II: FT7 Stage J OPS Application

### FT7.0 Approval And Preflight

FT7 begins only after explicit approval is recorded in a versioned decision
artifact that identifies the accepted M2 and M3 checkpoint hashes. Preflight
then revalidates:

- the FT6 completion and approval artifacts;
- all five comparison identities;
- checkpoint, environment, PAC, input, and code provenance;
- byte-identical Stage J opportunity pools and ordering;
- unchanged scenarios, lambdas, budgets, alpha policy, and exact formulations;
- separate cache and result roots for M2 and M3;
- available disk space, short-path publication mapping, and single-writer locks.

Internal checkpoint:

```text
FT7_P0_PREFLIGHT_PASS
```

### FT7.1 Two New OPS Runs

The Stage J mathematical contract remains frozen:

```text
scenarios: J-S1, J-S2, J-S3
lambda_R: 0, 0.2, 0.5, 0.8, 1
lambda_R_proxy: coupled to lambda_R
K: <= 2
guided topology budget: 100
continuous evaluation budget: 20 per topology
selected-load correction: q=5
PAC weights and rho_phys: unchanged
Reference A and Reference B: unchanged
candidate ordering and opportunity sets: unchanged
```

Run only M2 and M3 through the existing Guided-GridSFM pipeline. Guided-DC,
released GridSFM, and M1 are existing hash-bound evidence and are not rerun.
TH is not executed for either new checkpoint.

For each new model, the required new core counts are:

```text
settings:                  15
guided topology rows:   1,500
candidate evaluations: 30,000
guided finalists:           15
Reference A rows:            15
Reference B rows:            30
```

State-fidelity and warm-start records are generated according to the same
definitions as the completed FT3-FT5 study. Execution is resumable and reports
progress at preflight, every completed setting, 5/15, 10/15, 15/15, exact
reference completion, and validation. Runtime projections are refreshed from
observed setting averages.

Internal checkpoints:

```text
FT7_P1A_M2_OPS_PASS
FT7_P1B_M3_OPS_PASS
FT7_P2_EXACT_AUDIT_PASS
```

### FT7.2 Five-Method Evidence And Figures

Every new follow-up table and figure uses exactly this comparison set:

1. Guided-DC;
2. Guided-GridSFM, released v1.1;
3. Guided-GridSFM, `fulltop_ft_n1000`;
4. Guided-GridSFM, sequential FullTop-1500 (M2);
5. Guided-GridSFM, sequential FullTop-1000 + N-1-500 (M3).

Existing DC, released, and M1 evidence is joined by scenario, lambda,
opportunity-set identity, method family, and provenance keys. Only M2 and M3
rows are newly generated. TH rows remain preserved in the completed historical
packages but are excluded from all FT7 comparison tables, summaries, legends,
and figures.

The existing FT4 methodology is retained for candidate-level evaluated Pareto
frontiers, topology outcomes, objective convergence, Reference A/B outcomes,
native-to-exact discrepancies, physics/state fidelity, and runtime. Plot scales,
eligibility rules, nondominance logic, and aggregation definitions may not be
retuned after seeing M2/M3 results. Each figure must expose all five labels on
shared axes where the metric is comparable.

Internal checkpoint:

```text
FT7_P3_FIVE_METHOD_FIGURES_PASS
```

### FT7.3 TH-Free Crossed Warm-Start Study

The revised warm-start study removes TH finalist outcomes and crosses five
retained finalist families:

1. Guided-DC finalist;
2. released GridSFM guided finalist;
3. M1 guided finalist;
4. M2 guided finalist;
5. M3 guided finalist.

Each fixed topology/alpha finalist is solved with seven initialization policies:

1. audited generic cold start;
2. DC partial start;
3. released GridSFM full AC-state start;
4. M1 full AC-state start;
5. M2 full AC-state start;
6. M3 full AC-state start;
7. exact Reference A primal start.

The complete crossed contract is:

```text
5 finalist families x 15 settings x 7 starts = 525 rows
```

Verified existing rows may be reused only when the setting, finalist topology,
full alpha vector, start policy, checkpoint hash, solver configuration, and
timing definition match exactly. The expected reusable core is 225 rows from
the three non-TH existing families and five existing start policies; the
expected newly solved balance is 300 rows. Validation recomputes these counts
from keys rather than trusting the estimate. The 300 rows comprise 120 base
audits generated with the M2/M3 OPS packages and 180 missing checkpoint-crossed
solves needed to complete all four GridSFM starts for every finalist family.

The primary runtime remains native IPOPT solve time after model construction
and initialization loading. GridSFM inference and start construction remain
outside this primary timer and are reported separately. Every fixed-instance
start must converge to objective spread at most `1e-3` or retain an explicit
failure row. The summary reports mean, median, interquartile spread, paired
wins/losses, and seconds saved versus cold on one shared runtime axis.

Internal checkpoint:

```text
FT7_P4_WARM_START_525_PASS
```

### FT7.4 Validation And Publication

The working evidence remains external local CSV during execution. After all
semantic validations pass, the Git publication converts CSV tables to Parquet,
preserves readable relative artifact names, records source and published hashes
and row counts, and performs Parquet read-back comparison. Writes use the
existing retry/staging safeguards for Windows and OneDrive path behavior.

Validation must confirm:

- exact five-method coverage in every FT7 comparison artifact;
- no TH rows in FT7 figures or warm-start summaries;
- direct checkpoint path/SHA provenance on all model-backed rows;
- complete 15-setting coverage and variant-aware cache keys;
- candidate, finalist, reference, state, and warm-start key uniqueness;
- figure non-emptiness and recomputed source-table agreement;
- no mutation of the completed FT0-FT5 packages.

Internal checkpoint:

```text
FT7_P5_PUBLICATION_VALIDATION_PASS
```

### Primary External Checkpoint B

After FT7 validation, provide the complete five-method results, revised
warm-start interpretation, immediate divergences, and paper-facing claim
boundaries for Caleb's review. Final paper language and designation of this
extension as the Stage J closeout require explicit approval.

## Runtime Planning

Using measured FT1 and FT3/FT4 runtimes, the current CPU estimate is:

| Work | Expected wall time |
| --- | ---: |
| FT6 protocol/cache preflight | 10-20 minutes |
| two 500-case, ten-epoch continuation runs | 3.0-3.6 hours |
| four-model 375+375 test evaluation | 20-30 minutes |
| FT6 review packet | 10-20 minutes |
| mandatory external review | user-controlled pause |
| two new Stage J guided OPS runs and exact audits | 7-8 hours |
| 525-row warm-start completion, figures, validation, publication | 0.5-1.0 hour |
| active compute total | approximately 11-13 hours |
| conservative active compute total | approximately 13-15 hours |

The FullTop and N-1 processed train/val/test caches already exist locally, so
FT6 currently requires no network download. If cache integrity fails, OPFData
may fetch a complete group rather than only the selected 500/375 graphs; any
such download is a reported stop-and-review event when offline.

## Global Stop Conditions

Stop and retain evidence for:

- checkpoint, source, environment, data-manifest, or opportunity-pool drift;
- train/validation/test leakage or graph-identity duplication;
- parent/child checkpoint ambiguity or cross-branch initialization;
- nonfinite weights, missing per-epoch metrics, reload failure, or schema drift;
- changed Stage J PAC, objective, search, reference, or timing methodology;
- result reuse without complete key and hash identity;
- missing five-method coverage or unexpected TH inclusion;
- any attempt to begin FT7 without explicit external approval.

Scientifically worse model metrics, different OPS decisions, exact-AC failures,
weaker warm starts, and tradeoffs between FullTop and N-1 behavior are retained
as results and are not implementation failures.
