# FT5 Crossed Warm-Start Study Handoff

## Resume Status

```text
status: RESUMED_AND_COMPLETED
paused: 2026-08-22
completed: 2026-08-23
active experiment processes: none
```

This document preserves the resume point used on August 23. The frozen
augmentation, crossed evaluation, canonical assembly, validation, and shared-axis
figure are now complete. Final results are reported in
`FT5_CROSSED_WARM_START_STATUS.md`.

## Locked Scientific Design

Retain the final evaluated topology and alpha decisions from five finalist
families:

1. Guided-DC.
2. Guided-GridSFM using released GridSFM v1.1.
3. Guided-GridSFM using `fulltop_ft_n1000`.
4. TH-GridSFM-top1 from the released-checkpoint run.
5. TH-GridSFM-top2 from the released-checkpoint run.

For every retained fixed `(z*, alpha*)` decision, solve exact AC Reference A
with these five initialization policies:

1. Cold: `V=1`, `theta=0`, `Pg=(Pmin+Pmax)/2`, `Qg=0`.
2. DC: existing DC partial warm start.
3. GridSFM frozen: full released-v1.1 AC state (`Pg`, `Qg`, `V`, `theta`).
4. GridSFM fine-tuned: full `fulltop_ft_n1000` AC state.
5. Exact: Reference A ground-truth warm start.

This is a crossed design: both GridSFM checkpoints must initialize every
retained finalist family. The frozen and fine-tuned starts must remain separate
categories in tables and figures. Existing GridSFM partial-state rows may be
retained as an ablation but must not replace either full-state category in the
primary graph.

Expected primary table size:

```text
5 finalist families x 15 scenario/lambda settings x 5 starts = 375 rows
```

## Completed Work

- `stage_j_ac_reference.jl` explicitly applies the locked cold-start policy
  before any supplied start values and records start-state audit fields.
- `run_j9_j10_reference_smoke.py` supports `--warm-starts-only`, writes full
  GridSFM `Pg/Qg/V/theta` starts, and preserves prior Reference A/B evidence.
- The fine-tuned package augmentation completed for all 15 settings:

```text
cache:
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\ft3_fulltop_ft_n1000_v002

working results:
C:\Users\Caleb Lu\.gridfm_stage_j\results\stage_j\fulltop_ft_n1000_complete_run

rows: 225 = 3 finalists x 5 starts x 15 settings
status: PASS
max objective spread across starts: 8.40437132865191E-06
```

- Cold-start audits confirmed 500 bus starts, 224 PowerModels generator
  records, unit voltage, zero angle, midpoint active generation, and zero
  reactive generation.
- The fine-tuned full-state smoke solved all 15 runs for one setting. Runtime
  changes versus the partial-state GridSFM start were mixed, so improvement
  must be determined from the complete crossed evidence rather than assumed.
- Focused tests passed before the long augmentation (`30 passed, 2 skipped`),
  followed by a focused `7 passed` run.

## Interrupted Frozen Augmentation

The frozen augmentation was stopped because its projected runtime exceeded ten
minutes. Its Python and Julia process tree was terminated; no experiment process
was active when this handoff was written.

Frozen cache state at interruption:

```text
s1_l0p0: 20 rows (complete)
s1_l0p2: 20 rows (complete)
s1_l0p5: 20 rows (complete)
s1_l0p8: 20 rows (complete)
remaining 11 settings: 16 rows each (prior four-start evidence)
```

The augmentation is idempotent. Rerunning without `--force` skips settings that
already pass its five-start completeness audit.

Resume command from repository root:

```powershell
python experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/augment_full_warm_starts.py `
  --python "C:\Users\Caleb Lu\.gridfm_stage_j\envs\gridsfm\Scripts\python.exe" `
  --cache-root "C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\complete_run_v001" `
  --final-root "experiments/test/wildfire_tests/goc_500_results/stage_j/complete_run" `
  --gridsfm-root "C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM" `
  --input-dir "C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\inputs\case500_goc_e0" `
  --model-selection frozen `
  --expected-checkpoint-sha256 F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831 `
  --workers 4
```

Expected remaining runtime is approximately 15 to 20 minutes with four
workers. Confirm that the resulting `FULL_WARM_START_AUGMENTATION.json` reports
`PASS`, 15 settings, 300 rows, and objective spread no greater than `1e-3`.

## Exact Next Steps

1. Resume and validate the interrupted frozen augmentation with the command
   above. Do not use `--force`.
2. Extend the warm-start runner with a focused append mode that evaluates one
   named full GridSFM checkpoint without recomputing cold, DC, exact, or the
   other checkpoint start.
3. Give full starts unambiguous identifiers:
   `gridsfm_frozen_full_warm` and `gridsfm_ft_full_warm`. Normalize existing
   generic `gridsfm_full_warm` rows according to their checkpoint provenance.
4. Apply the fine-tuned full state to all four retained frozen-run finalist
   families: Guided-DC, frozen Guided-GridSFM, TH top1, and TH top2.
5. Apply the frozen full state to the fine-tuned Guided-GridSFM finalists only.
   Fine-tuned-run TH finalists are not part of the locked comparison.
6. Assemble one canonical 375-row crossed table keyed by scenario, lambda,
   finalist family, start type, topology identity, and checkpoint provenance.
7. Validate exactly 15 observations for every one of the 25 finalist/start
   combinations; require successful exact solves and objective spread no
   greater than `1e-3` within each fixed finalist instance.
8. Generate one primary shared-y-axis grouped figure. Put the five finalist
   families on the x-axis, solver runtime in seconds on one y-axis, and show
   separate Cold, DC, frozen GridSFM, fine-tuned GridSFM, and Exact categories
   in the legend. Do not facet frozen and fine-tuned GridSFM onto independent
   y-scales.
9. Regenerate comparison summaries, FT4/FT5 validation, the external working
   package, the duplicated `goc_500_results/stage_j` package, and the compact
   Parquet Git publication.
10. Run the full focused test suite and inspect the final figure before making
    any performance claim. Report medians, spread, paired differences, failures,
    and whether either checkpoint gives consistent runtime improvement.

## Completion Gate

The study is complete only when the primary comparison contains all 375 crossed
rows, both GridSFM checkpoints are visibly separate on one shared-axis graph,
all provenance hashes are present, validation passes, and the documentation
states that warm starts change solver trajectory/runtime but not the fixed
Reference A optimization problem.
