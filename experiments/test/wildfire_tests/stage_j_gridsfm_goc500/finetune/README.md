# Stage J GridSFM fine-tuning

This directory is a thin experiment wrapper around the official
`microsoft/GridSFM` v1.1 OPFData fine-tuning implementation. It does not copy
the model, losses, dataset adapter, topology transforms, evaluation metrics, or
optimizer loop.

FT0 validates this path only:

`released v1.1 -> GOC-500 FullTop OPFData -> official fine-tune -> external checkpoint -> fresh-process reload -> held-out evaluation -> unchanged Stage J candidate evaluation`

Run from the repository root with the dedicated GridSFM environment:

```powershell
& 'C:\Users\Caleb Lu\.gridfm_stage_j\envs\gridsfm\Scripts\python.exe' `
  experiments\test\wildfire_tests\stage_j_gridsfm_goc500\finetune\run_finetune.py `
  --config experiments\test\wildfire_tests\stage_j_gridsfm_goc500\finetune\configs\ft0_smoke.json
```

The first run may download and process a complete 15,000-graph OPFData shard;
the upstream adapter applies `n_graphs` only after that cache exists. OPFData
and checkpoints stay under `C:\Users\Caleb Lu\.gridfm_stage_j` and outside Git.

Do not run FT1 until the FT0 status and workflow report have been reviewed.

FT2 is an inference-only, paired comparison of the released frozen checkpoint
and the completed FT1 checkpoint. It evaluates both checkpoints on the same
ordered 750-graph FullTop test split and the same ordered 750-graph N-1 test
split from one OPFData group. Neither test split may be used for fine-tuning.

```powershell
& 'C:\Users\Caleb Lu\.gridfm_stage_j\envs\gridsfm\Scripts\python.exe' `
  experiments\test\wildfire_tests\stage_j_gridsfm_goc500\finetune\run_ft2_evaluation.py `
  --config experiments\test\wildfire_tests\stage_j_gridsfm_goc500\finetune\configs\ft2_fulltop_n1_test.json
```

Run the lightweight wrapper checks without loading OPFData:

```powershell
& 'C:\Users\Caleb Lu\.gridfm_stage_j\envs\gridsfm\Scripts\python.exe' `
  experiments\test\wildfire_tests\stage_j_gridsfm_goc500\finetune\verify_finetune_driver.py
```

The planned sequential fine-tuning ablation is governed by:

```text
workflow/cases/CASE-003-stage-j-gridsfm-goc500-implementation/
  FT6_FT7_PLAN.md
```

FT6 trained two sibling continuations from `fulltop_ft_n1000` and compared
four GridSFM checkpoints on a sealed 375-FullTop plus 375-N-1 test set. P1,
both P2 training checkpoints, and P3 evaluation passed. The result packet is
`FT6_SEQUENTIAL_FINETUNE_STATUS.md` in the workflow case, with compact tables
and figures under `results/ft6/review`.

FT6 closed as `FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL`. Caleb explicitly
approved FT7 on 2026-08-24, and `FT7_APPROVAL.json` records the decision and
accepted M2/M3 hashes. FT7 preflight passed before execution began.

FT7 runs only the two new M2/M3 Guided-GridSFM variants and reuses the existing
DC, released-M0, and FullTop-1000-M1 evidence. TH is excluded. The final study
contains five model families, Pareto fronts derived from all eligible evaluated
candidates pooled across lambda per scenario/model, and a five-finalist by
seven-initialization warm-start matrix using native IPOPT solve time. Completing
that matrix requires 180 new crossed solves because each M2/M3 base package
contains its own selected checkpoint start but not every other checkpoint
start. Results publish under
`goc_500_results/stage_j/refined_finetune_study`; checkpoints remain external.
