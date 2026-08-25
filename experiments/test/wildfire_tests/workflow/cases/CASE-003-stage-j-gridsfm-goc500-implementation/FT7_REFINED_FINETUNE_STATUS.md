# FT7 Refined Fine-Tuning OPS Status

## Terminal State

```text
FT7_COMPLETE_AWAITING_PAPER_REVIEW
```

Caleb explicitly approved FT7 on 2026-08-24. Preflight bound the approved M2
and M3 checkpoint hashes, GridSFM commit, existing DC/M0/M1 packages, all 15
guided topology pools, corrected native-IPOPT FT5 rows, and the no-TH output
contract before execution.

## Execution

Only M2 and M3 were newly run through the frozen Stage J Guided-GridSFM OPS
path. Each passed 15 settings, 1,500 topology rows, 30,000 candidate rows, 15
finalists, 15 Reference A rows, 30 Reference B rows, and 120 state-fidelity
rows. Both package validators confirmed the expected checkpoint hash/variant
on all candidates and `th_executed=false`.

The combined package contains DC, M0, M1, M2, and M3. Core validation passed
with 7,500 topology rows, 150,000 candidates, 75 finalists, 75 Reference A
rows, 150 Reference B rows, and 600 state-fidelity rows. All exact solves and
B2 service locks passed; source provenance and primary-key uniqueness passed.

## Pareto And Warm Starts

The updated Pareto study pooled all eligible finite candidate evaluations
across lambda within each scenario/model group. It retained 149,601 eligible
rows and recomputed 603 nondominated points over all 15 required fronts.

The warm-start study contains 525 unique rows: five finalist families by 15
settings by seven starts. Its final evidence is a controlled rerun of all 525
solves with the seven starts rotated across execution positions. Every start
appeared 10 or 11 times in each position. All solves succeeded, full GridSFM
starts supplied `Pg,Qg,V,theta`, every barrier-iteration count was positive,
and maximum within-instance objective spread was `8.406234e-06` against
`1e-3`.

Mean/median native IPOPT times were 5.221/5.113 seconds for cold, 5.151/5.075
for DC, 5.160/5.104 for M0, 5.108/5.039 for M1, 5.104/5.055 for M2,
5.159/5.067 for M3, and 5.216/5.089 for exact primal. Median iteration counts
were 33 for cold, 30 for DC, and 31 for every GridSFM start and exact. The
earlier apparent M2/M3 slowdown is superseded as a mixed-batch artifact. M1
and M2 are effectively tied by solve time; inference and construction remain
outside the primary metric.

## Scientific Interpretation

M2 gave the strongest FullTop OPS result among GridSFM variants. Mean native
`J_trade` improved from 0.33363 (M0) and 0.30843 (M1) to 0.30351 (M2); M3 was
0.30979. M2 also had the smallest native-to-Reference-A discrepancies and the
lowest seven-family GridSFM state NRMSE. M3 still improved substantially over
M0 and generally M1, while FT6 showed its stronger N-1 held-out composite. This
supports distribution-specific adaptation with tradeoffs, not unconditional
model dominance.

## Resolved Wrapper Issues

Fail-closed checks caught and resolved four wrapper issues without changing
model, OPS, or solver semantics:

- explicit method filters prevent historical frozen manifests from executing
  TH finalists in FT7;
- `m0` is an additive artifact identifier while model selection stays frozen;
- FT5 `timing_rerun_objective` is canonicalized during assembly;
- short-path staging and extended-length copies publish readable files under
  the deep OneDrive path.

## Publication

Results are in `goc_500_results/stage_j/refined_finetune_study`. The package
has 48 files, 20 read-back-verified Parquet tables, 15 PNG figures, zero CSVs,
and zero checkpoints. `PUBLICATION_MANIFEST.json` records paths, rows, sizes,
and hashes. See `REFINED_FINETUNE_STUDY_SUMMARY.md` for full interpretation.
