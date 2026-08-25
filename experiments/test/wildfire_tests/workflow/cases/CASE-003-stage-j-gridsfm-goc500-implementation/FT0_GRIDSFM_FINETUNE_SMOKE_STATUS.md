# FT0 GridSFM GOC-500 Fine-Tuning Smoke Status

## Outcome

FT0 completed the approved end-to-end smoke on August 21, 2026. It validates
the released-checkpoint, official OPFData fine-tuning, external checkpoint,
fresh-process reload, held-out evaluation, and unchanged Stage J candidate
evaluation path. FT0 is pipeline evidence only and is not a model-quality or
paper result.

## A. Files Created Or Modified

The implementation is contained in:

```text
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/
  README.md
  run_finetune.py
  configs/ft0_smoke.json
  results/ft0/
```

The controlled-extension design is recorded in
`STAGE_J_FINETUNE_EXTENSION_DESIGN.md`. This status report, `HISTORY.md`, and
`CURRENT_STATE_SUMMARY.md` provide the workflow handoff. No Stage J evaluator,
objective, PAC, search, mapping, or exact-reference implementation was changed.

## B-D. Base Checkpoint And Environment

The completed Stage J implementation used the official default checkpoint:

```text
file: gridsfm_open_v1.1.pt
SHA-256: F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831
release: microsoft/GridSFM_Open
GridSFM version: 1.1.0
GridSFM commit: 1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6
Python: 3.12.11
torch: 2.8.0+cpu
torch_geometric: 2.8.0.post1
platform/device: Windows / CPU
```

This is v1.1, so `BASELINE_VERSION_MISMATCH` does not apply. The completed
run's `RUN_CONFIG.json` did not itself retain checkpoint provenance; the exact
checkpoint is established by the Stage J runner defaults and the frozen J0.5
bootstrap manifest. Future Stage J model variants should record checkpoint
path and hash directly in each run manifest.

## E-G. Data, Preprocessing, And Training Controls

FT0 used `pglib_opf_case500_goc`, `fulltop`, train indices 0 through 9. The
held-out set was the distinct `test` split, indices 0 through 9. No N-1, N-2,
Stage J candidate, Reference A/B, or alpha-modified state was used for
training. The official PyG graphs expose objective and feasibility metadata but
no stable sample-ID field, so `(case, variant, split, index)` is the recorded
reproducible sample identity.

The training set was wrapped by the official
`SyntheticMixedDataset(infeas_prob=0.3, seed=42)`. The required transform ran
through the official `CycleBasisCache` / `prepare_for_grid_transformer_` and
`LaplacianFactorizationCache` / `attach_pe_features_` functions. Training used
the official `finetune_opfdata` AdamW loop with batch size 2, two epochs,
learning rate `1e-4`, weight decay `1e-4`, and the package-default loss and
gradient clipping.

PyG downloaded one complete OPFData group before applying the ten-graph cap,
as documented by the official adapter. The first run materialized 15,001 raw
files totaling approximately 13.05 GB plus approximately 1.35 GB of processed
split artifacts under the external cache. No OPFData content is in Git.

## H-J. Training Evidence

```text
epoch 0: train_loss 0.0992720324, 5 batches, 0 skipped
epoch 1: train_loss 0.0970960006, 5 batches, 0 skipped

training runtime: 23.69 s
first end-to-end run including download/process: 557.3 s
cached corrected verification run: 46.1 s
device: CPU
```

All 1,221 floating parameter tensors changed. Total parameter L2 change was
`1.0202201`; maximum absolute change was approximately `9.76e-4`; every
updated parameter remained finite.

## K-L. Checkpoint And Fresh Reload

The disposable FT0 checkpoint is external to Git:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\gridsfm_goc500_fulltop_ft0_n10.pt
size: 61,003,599 bytes
SHA-256: CF72E0F6036D1E37FB5A298EB3DB5E69A6524F48BC7B0B8B5CD34E758DDFE4A2
```

A fresh Python process loaded this file through official `load_model`, passed
the checkpoint's embedded state-dict integrity hash, and produced finite bus,
generator, feasibility, and P/Q flow outputs. Frozen and FT0 output keys and
shapes were identical, and predictions differed.

## M-N. Held-Out FullTop Metrics

The same ten test graphs were evaluated with official `eval_pass`:

| Metric | Frozen v1.1 | FT0 |
| --- | ---: | ---: |
| loss | 0.091718 | 0.106599 |
| cost MAPE | 0.008387 | 0.008194 |
| Pg MAE | 0.008701 | 0.008554 |
| Qg MAE | 0.048191 | 0.054371 |
| V MAE | 0.002325 | 0.002401 |
| theta MAE | 0.019871 | 0.020469 |
| branch P MAE | 0.074641 | 0.072342 |
| branch Q MAE | 0.052274 | 0.056970 |
| KCL P residual | 0.000660 | 0.000690 |
| KCL Q residual | 0.001329 | 0.001863 |
| feasibility accuracy | 1.0 | 1.0 |

The mixed direction is expected for ten train graphs and two epochs. These
numbers demonstrate execution and changed predictions; they do not establish
improvement or degradation. A tiny N-1 inference smoke was deferred because
the upstream adapter would require another full external dataset group and is
not an FT0 pass criterion.

## P-Q. Identical Stage J Candidate Smoke

The paired smoke loaded the existing Guided-GridSFM finalist for `J-S1`,
`lambda_R=0.2`, topology rank 6, offline branches `{331, 473}`, and its exact
saved alpha command. Both checkpoints used the unchanged Stage J evaluator.

| Field | Frozen v1.1 | FT0 |
| --- | ---: | ---: |
| evaluation status | model_output_penalized | model_output_penalized |
| D_input | 0.0 | 0.0 |
| feasibility head | 0.984616 | 0.988595 |
| R_norm | 0.466487 | 0.466002 |
| PAC_total | 0.000405 | 0.000307 |
| J_total | 0.094108 | 0.093814 |

Full Pg, Qg, V, theta, two-ended branch flows, loadings, PAC components, and
runtimes are retained in `results/ft0/ft0_stage_j_smoke.json`. The existing
raw-case identity, branch/rateA/topology, load-command, units/baseMVA, and
source-less-load guards executed unchanged. Both runs passed `D_input=0`, so
`CHECKPOINT_INTERFACE_DEVIATION` does not apply. The small FT0 difference is
pipeline evidence only.

## R. Methodology Audit And Immediate Divergences

The implementation follows the pinned Microsoft/GridSFM v1.1 README, example
notebook, package tests, and fine-tuning source. It also respects OPFData's
paper-level role as solved AC-OPF data with explicit topology variants: FT0
uses only FullTop adaptation data and reserves topology-shift evaluation for
later stages. Sources: [official GridSFM fine-tuning documentation](https://github.com/microsoft/GridSFM/blob/main/model/README.md)
and [OPFData paper](https://arxiv.org/abs/2406.07234).

Immediate implementation notes:

- GridSFM has no public fine-tuned checkpoint exporter. The wrapper uses the
  documented minimal `{state_dict, metadata}` payload and GridSFM's own private
  `_hash_state_dict`; official `load_model` then strictly verifies the result.
- GridSFM's OPFData evaluation does not publicly return reconstructed branch
  flow tensors. The finite/shape gate uses private `_predicted_flows`, the same
  helper used by official `eval_pass`.
- The official `test_ft.py` displayed its pass/skip completion pattern but the
  pytest process did not exit before a 120-second Windows CPU timeout. The
  direct FT0 execution subsequently validated the relevant real-data paths.
- A strengthened rerun exposed and fixed one local JSON integer/string key
  normalization bug in the prediction-difference assertion. The final cached
  run completed cleanly after the fix.

These are wrapper/API/platform notes, not changes to the GridSFM architecture,
loss, preprocessing, optimizer, OPFData split, or Stage J methodology. FT1
must start fresh from released v1.1 and remains unexecuted pending review.

FT0_VALIDATED_READY_FOR_FT1
