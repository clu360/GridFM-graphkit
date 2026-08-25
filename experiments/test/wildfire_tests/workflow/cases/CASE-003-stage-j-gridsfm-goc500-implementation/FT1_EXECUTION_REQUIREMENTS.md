# FT1 Execution Requirements

## Approval And Scope

FT1 was approved after review of the successful FT0 smoke. FT1 trains the
primary GOC-500 FullTop model only. FT2 model evaluation and FT3 Stage J rerun
remain separate stages. FT1 and FT2 are now complete; FT3 remains unexecuted.

FT1 concluded with status `FT1_TRAINED_READY_FOR_FT2`. All requirements below
passed, and the resulting checkpoint SHA-256 is
`A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD`.

## Frozen FT1 Recipe

```text
base: released gridsfm_open_v1.1.pt
case: pglib_opf_case500_goc
variant/split: fulltop/train
train indices: 0..999
validation variant/split: fulltop/val
validation indices: 0..749
batch size: 8
epochs: 10
learning rate: 1e-4
weight decay: 1e-4
SyntheticMixedDataset infeas_prob: 0.3
seed: 42
```

The final OPFData `test` split is not used during FT1 and remains reserved for
FT2.

## Environment Freeze

FT1 must run at GridSFM commit
`1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6` in the dedicated environment.
Because the wrapper depends on private `_hash_state_dict` and
`_predicted_flows`, preflight must match the frozen SHA-256 values for
`checkpoint.py`, `loss.py`, `finetune_opfdata.py`, and `opfdata_train.py`.
The FT1 result directory and manifest must retain the full `pip freeze`, exact
commit, package versions, source hashes, Python executable, and device.

Any mismatch stops with `FT1_IMPLEMENTATION_DEVIATION_REQUIRES_REVIEW`.

## Checkpoint Provenance

Every FT1 and future FT3 manifest must contain top-level `checkpoint_path` and
`checkpoint_sha256` fields. Nested checkpoint metadata may supplement but not
replace those fields. FT3 must also record both its fine-tuned checkpoint and
the paired frozen-v1.1 checkpoint path and SHA before execution begins.

## Per-Epoch Validation

The official `finetune_opfdata` call must receive the deterministic FullTop
validation loader. Every one of the ten training-log rows must retain all
official `val_*` metrics, including validation loss, cost MAPE, Pg/Qg/V/theta
MAE, branch P/Q MAE, KCL residuals, thermal metrics, feasibility accuracy, and
the validation graph count. The FT1 manifest must embed or directly reference
these per-epoch records; final-checkpoint metrics alone are insufficient.
