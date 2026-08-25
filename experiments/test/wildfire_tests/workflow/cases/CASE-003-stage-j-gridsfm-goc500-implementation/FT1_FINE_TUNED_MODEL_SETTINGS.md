# FT1 Fine-Tuned Model Settings

## Model Identity

```text
model_variant: fulltop_ft_n1000
artifact_id: STAGE-J-GRIDSFM-FT1-FULLTOP-1000
status: FT1_TRAINED_READY_FOR_FT2
architecture: released GridSFM v1.1 architecture, unchanged
GridSFM commit: 1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6
```

Released initialization:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\repos\GridSFM\model\checkpoints\
  gridsfm_open_v1.1.pt
SHA-256: F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831
```

Fine-tuned checkpoint:

```text
C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune\
  gridsfm_goc500_fulltop_ft_n1000.pt
SHA-256: A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD
size: 61,007,770 bytes
```

## Fine-Tuning Recipe

```text
case: pglib_opf_case500_goc
variant: fulltop
training split/indices: train, 0..999
validation split/indices: val, 0..749
test data used during tuning: none
batch size: 8
epochs: 10
learning rate: 1e-4
weight decay: 1e-4
SyntheticMixedDataset infeas_prob: 0.3
seed: 42
device: CPU
optimizer loop: official gridsfm.finetune_opfdata
preprocessing: official cycle-basis and Hodge PE transforms
```

All 1,221 floating parameter tensors changed and remained finite. No training
batch was skipped. Official held-out validation metrics were recorded on all
750 validation graphs after each epoch. Training took 9,082.9 seconds, and the
saved checkpoint passed strict fresh-process reload and output-schema checks.

## Sealed FT2 Evaluation

The model was evaluated against released v1.1 on two separate inference-only
test sets:

```text
FullTop test: one OPFData group, split=test, indices 0..749
N-1 test:     one OPFData group, split=test, indices 0..749
```

Relative changes from released v1.1 to `fulltop_ft_n1000`:

| Metric | FullTop | N-1 |
| --- | ---: | ---: |
| official loss | -55.525% | -37.994% |
| cost MAPE | -16.093% | -24.726% |
| Pg MAE | -40.135% | -33.662% |
| Qg MAE | -61.270% | -52.820% |
| V MAE | -34.391% | -26.882% |
| theta MAE | -53.373% | -51.014% |
| branch P MAE | -45.580% | -43.025% |
| branch Q MAE | -56.494% | -50.182% |

KCL residuals and thermal metrics improved on both test variants. FullTop
feasibility accuracy remained 1.0. N-1 feasibility accuracy changed from
`0.998667` to `0.993333`, or one false rejection of a feasible graph versus
five. Because OPFData test graphs are all labeled feasible, this metric does not
measure balanced feasible/infeasible discrimination.

## Interpretation Boundary

The checkpoint is approved for a controlled FT3 Stage J application. FT2 shows
better held-out continuous AC-OPF prediction and physics diagnostics, including
transfer to unseen N-1 topologies. It does not prove better Stage J candidate
ranking, exact AC feasibility, or unconditional feasibility-head superiority.

The two FT2 test splits are sealed. A future FullTop+N-1 model must exclude them
from training and validation and evaluate on the identical cases.

Canonical machine-readable records are retained in:

```text
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/results/ft1/
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/results/ft2/
```
