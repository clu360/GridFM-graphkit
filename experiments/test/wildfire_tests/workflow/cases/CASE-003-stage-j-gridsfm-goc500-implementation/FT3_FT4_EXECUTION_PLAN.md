# FT3 And FT4 Linked Execution Plan

## Objective

FT3 tests whether the model-level improvements observed in FT2 change the
decisions and exact-audit outcomes of the completed Stage J experiment. FT4 is
the linked evidence-generation phase for that run. FT3 produces the complete
raw experiment; FT4 must consume that exact completed run and reproduce the
existing table and figure suite before any new FT5 analysis is designed.

The sole scientific intervention is:

```text
released_v1_1 -> fulltop_ft_n1000
```

No Stage J hyperparameter, candidate definition, search budget, PAC weight,
reference formulation, or plotting definition may be retuned in response to
the fine-tuned model.

## Frozen Stage J Contract

```text
scenarios: J-S1, J-S2, J-S3
lambda_R: 0, 0.2, 0.5, 0.8, 1
lambda_R_proxy: coupled to lambda_R
K: <= 2
guided topology budget: 100
continuous evaluation budget: 20 per topology
selected-load correction: q=5
candidate branch set and c_l: unchanged
PAC definition, weights, and rho_phys: unchanged
Reference A: unchanged fixed-z/fixed-alpha economic AC-OPF
Reference B: unchanged fixed-z AC maximum-load-delivery plus tie-break
warm starts: unchanged cold/DC/GridSFM/exact definitions
seed and candidate ordering: unchanged
```

## Baseline And Opportunity-Set Audit

The completed frozen run is a valid released-v1.1 baseline. Its retained J8
commands contain no `--checkpoint` override, and the runner default is
`gridsfm_open_v1.1.pt`. Before FT3, an audit manifest will bind those command
logs and the existing result package to the released checkpoint path and
SHA-256 `F8A439...831`.

The topology opportunity sets are checkpoint-independent:

- Guided-DC creates `guided.csv` from the wildfire-side proxy/no-good-cut
  process; Guided-GridSFM reads the same ordered file.
- TH creates `th.csv` from environmental probability and baseline loading,
  without loading GridSFM.

FT3 will therefore reuse byte-identical copies of all 15 frozen `guided.csv`
and `th.csv` files. A pool manifest will record source/destination hashes, row
counts, ordering, scenario, and lambda. Any mismatch stops execution.

## Required Implementation Changes

1. Add explicit `--gridsfm-checkpoint`, `--expected-checkpoint-sha256`, and
   `--model-variant` arguments to the complete-run driver.
2. Propagate the checkpoint to both J8 GridSFM-backed methods and to J9/J10.
   J9/J10 currently hardcodes released v1.1 for native-state and GridSFM warm
   starts, so it must receive the same FT3 checkpoint explicitly.
3. Record model variant, checkpoint path, checkpoint SHA-256, GridSFM commit,
   environment freeze, and Stage J code commit in run, method, reference, and
   aggregate manifests.
4. Make cache-completion checks variant- and hash-aware. A successful frozen
   artifact must never satisfy an FT3 completion check.
5. Add a pool-reuse preflight that verifies every ordered opportunity set
   against the frozen package before model inference starts.
6. Keep the external cache and final result roots separate from the frozen run:

```text
external cache:
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\ft3_fulltop_ft_n1000_v001

repository results:
experiments/test/wildfire_tests/goc_500_results/stage_j/
  fulltop_ft_n1000_complete_run/
```

## FT3 Execution

Preflight will verify the FT1 manifest status, direct checkpoint path/hash,
GridSFM commit, private-helper hashes, PAC freeze, input manifests, frozen run
configuration, pool hashes, and available disk space.

The linked runner will then execute all 15 scenario/lambda settings. It will:

1. evaluate `fulltop_ft_n1000` Guided-GridSFM over the identical 100-topology
   guided pools and 20-evaluation alpha budget;
2. evaluate `fulltop_ft_n1000` TH-GridSFM over the identical TH pools;
3. retain Guided-DC and released-v1.1 as existing, hash-bound comparison
   packages without rerunning or copying their result rows into the FT package;
4. build FT finalists using the unchanged selection rule;
5. run Reference A, Reference B, state-fidelity, and all four warm-start
   definitions only for the three newly selected FT finalist families;
6. write resumable status after every method and reference bundle.

The primary paired comparison is released-v1.1 versus `fulltop_ft_n1000` on
identical opportunity sets. Guided-DC remains the established approximation
comparator. Exact references assess the decisions selected by each model; they
are not reused across different finalists.

FT3 completion requires the following newly generated FT package counts:

```text
topology outcomes:       1,530
candidate evaluations: 30,600
finalists:                  45
Reference A rows:           45
Reference B rows:           90
warm-start rows:           180
state-fidelity rows:       360
```

These comprise 1,500 guided FT topology outcomes, 30,000 guided FT evaluations,
30 TH outcomes, and 600 TH evaluations. Exact audits are generated only for
the 45 new FT finalists. Frozen and DC rows remain in the existing complete-run
package and are joined later by model-selection and provenance keys.

## FT4 Evidence Generation

FT4 starts automatically only after the FT3 completion manifest passes. It
reuses the existing summary and exact-audit plotting code and reproduces:

1. evaluated risk/load scatter;
2. `J_trade` by topology rank;
3. best-observed objective convergence;
4. Reference A economic AC-OPF cost;
5. native versus Reference-A risk discrepancy;
6. native versus Reference-A load discrepancy;
7. native-state distance to Reference A;
8. Reference B maximum-load-delivery results;
9. component-level state-distance heatmap;
10. warm-start study.

Corresponding raw tables, method summaries, full alpha vectors, run status,
runtime summaries, and provenance manifests will be retained. Existing plot
definitions, labels, scales, and aggregation rules remain unchanged. Every
generated row will carry explicit `model_selection` and `model_variant`
metadata. The shared selector supports:

```text
model_selection=dc      model_variant=guided_dc
model_selection=frozen  model_variant=released_v1_1
model_selection=ft      model_variant=fulltop_ft_n1000
```

The active FT3 run is pinned to `model_selection=ft`; `dc` and `frozen` are
comparison identities for already completed evidence, not execution tasks.

FT4 validation will compare schemas and artifact counts against the frozen
package, verify finite plotted fields, confirm all figures are nonempty, and
recompute reported aggregates from raw tables. New cross-model scientific plots
belong to FT5 and will not be introduced during FT4.

## Runtime And Progress Reporting

The previous run spent about 9,740 seconds on 30,000 Guided-GridSFM evaluations
and 191 seconds on 600 TH evaluations. Because FT1 has the same architecture,
FT3 model inference should be similar. Including exact references, warm starts,
aggregation, and validation, the planned wall-clock range is approximately four
to six hours on the current CPU environment. FT4 table and figure generation
should take minutes after FT3 completes.

Progress will be reported at preflight, after each of the 15 setting bundles,
at `5/15`, `10/15`, and `15/15` aggregate checkpoints, through exact-reference
completion, and at FT4 artifact validation. Runtime projections will be updated
from observed completed-setting averages.

## Stop Conditions

Execution stops for checkpoint/hash or environment drift, pool mismatch,
baseline-version ambiguity, altered frozen settings, output-schema divergence,
missing exact-reference rows, nonfinite required metrics, or cache provenance
that cannot distinguish frozen from fine-tuned output.

Model behavior differences, including worse PAC, different finalists, exact AC
failures, or weaker warm starts, are scientific results and are retained rather
than treated as implementation failures.

## Execution Result

FT3 and FT4 completed on August 22, 2026. All planned counts, metadata checks,
30 pool identity checks, 13 figure checks, and four derived-table checks passed.
The authoritative working root is external local storage because newly copied
files under the OneDrive-managed repository results tree were not reliably
readable. The Git package is published as Parquet under
`finetune/results/ft3_ft4_v002`; see `FT3_FT4_EXECUTION_STATUS.md` for exact
counts, provenance, storage details, and immediate divergences.
