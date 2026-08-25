# FT6 Sequential Fine-Tuning Status

## Terminal Status

```text
FT6_P1_DATA_CONTRACT_PASS
FT6_P2A_M2_TRAINING_PASS
FT6_P2B_M3_TRAINING_PASS
FT6_P3_FOUR_MODEL_EVALUATION_PASS
FT6_COMPLETE_AWAITING_CALEB_FT7_APPROVAL
```

FT6 Part I is complete. FT7 has not started and remains blocked pending Caleb's
explicit review and approval.

## Frozen Method Contract

FT6 used Microsoft/GridSFM commit
`1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`, the released v1.1 architecture,
the official `OPFDataAdapterDataset`, `SyntheticMixedDataset`, GridSFM
transform, `finetune_opfdata`, checkpoint loader, loss, and `eval_pass`. No new
model architecture, electrical-state generator, topology representation, loss,
or OPS-specific training objective was introduced.

The official adapter supports `variant="fulltop"` and `variant="n1"`; N-1
selects OPFData's topology-perturbation path. Training M3 on N-1 is therefore a
direct supported API extension. It is not presented as a reproduction of the
white paper's documented FullTop-only GOC-500 fine-tuning experiment.

## Data Contract

| Use | Variant | Split | Frozen indices | Count |
| --- | --- | --- | ---: | ---: |
| M2 continuation train | FullTop | train | 1000-1499 | 500 |
| M3 continuation train | N-1 | train | 0-499 | 500 |
| per-epoch validation | FullTop | val | 0-374 | 375 |
| per-epoch validation | N-1 | val | 0-374 | 375 |
| sealed test | FullTop | test | 0-374 | 375 |
| sealed test | N-1 | test | 0-374 | 375 |

P1 fingerprinted all 2,500 selected graph records. Every stratum had its
expected count, finite tensors, and unique graph hashes. No graph hash crossed
the training, validation, or test partitions. All data came from local OPFData
caches; no download or state generation occurred during FT6.

## Checkpoints And Runtime

| ID | Variant | SHA-256 |
| --- | --- | --- |
| M0 | released v1.1 | `F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831` |
| M1 | FullTop-1000 | `A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD` |
| M2 | sequential FullTop-1500 | `08EDA70270F787DB42C48B751BECC9DA2062B0482550A5C2A761B4EBA0C94FF6` |
| M3 | FullTop-1000 + N-1-500 | `4EC89D36DE80081BE2FC26C14A1BC5A5369D7423B557D71B9DD4A254462F302D` |

M2 and M3 both loaded the byte-identical M1 parent and each received a fresh
AdamW optimizer. Each branch completed ten epochs with 63 successful batches
per epoch and zero skipped batches. M2's ten epoch cycles took `4,571.1 s`
(`76.2 min`); M3 took `4,524.7 s` (`75.4 min`). Both checkpoints changed all
1,221 floating parameter tensors, retained finite parameters and the parent
output schema, and passed fresh-process reload plus dual-stratum validation.

The sealed four-model evaluation took `1,140.2 s` (`19.0 min`).

## Sealed Test Results

Selected official metrics are shown below. Lower is better except feasibility
accuracy.

### FullTop Test, 375 Cases

| Model | Loss | Cost MAPE | Branch-P MAE | P-KCL residual | Feas. acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 released | 0.093298 | 0.008437 | 0.074514 | 6.656e-4 | 1.000000 |
| M1 FullTop-1000 | 0.040794 | 0.007164 | 0.040482 | 2.631e-4 | 1.000000 |
| M2 FullTop-1500 | **0.038112** | 0.005520 | **0.035024** | **2.209e-4** | 1.000000 |
| M3 FullTop+N-1 | 0.040381 | **0.005228** | 0.036046 | 2.398e-4 | 1.000000 |

### N-1 Test, 375 Cases

| Model | Loss | Cost MAPE | Branch-P MAE | P-KCL residual | Feas. acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| M0 released | 0.118472 | 0.011977 | 0.076878 | 6.742e-4 | 0.997333 |
| M1 FullTop-1000 | 0.070146 | 0.008770 | 0.043597 | 2.924e-4 | 0.992000 |
| M2 FullTop-1500 | 0.068235 | 0.007789 | 0.038692 | 2.578e-4 | 0.992000 |
| M3 FullTop+N-1 | **0.056797** | **0.006347** | **0.038042** | **2.550e-4** | **1.000000** |

Relative to M1, M2 reduced every reported error/physics/thermal metric on both
test strata while preserving feasibility accuracy. Its FullTop reductions
included loss `6.58%`, cost MAPE `22.95%`, branch-P MAE `13.48%`, and P-KCL
residual `16.02%`. Its N-1 reductions were `2.72%`, `11.19%`, `11.25%`, and
`11.84%`, respectively.

Relative to M1 on N-1, M3 reduced loss `19.03%`, cost MAPE `27.63%`, branch-P
MAE `12.74%`, and P-KCL residual `12.78%`; feasibility improved by `0.8`
percentage points from three misses to zero. This is the clearest evidence that
explicit N-1 exposure improves the intended held-out topology family.

M3 is not uniformly superior to M2. On FullTop, M3 had lower cost MAPE and
theta MAE, but higher loss, Pg/Qg/V errors, branch-P/Q errors, both KCL
residuals, and thermal metrics. On N-1, M3 improved loss, cost, Pg/Qg/V/theta,
branch-P, P-KCL, and feasibility relative to M2, while branch-Q, Q-KCL, and
thermal-overload metrics were worse. The paper-facing conclusion must remain a
distribution-specific adaptation tradeoff, not unconditional model dominance.

## Review Artifacts

The authoritative evidence is under
`stage_j_gridsfm_goc500/finetune/results/ft6`:

- `ft6_preflight_manifest.json`;
- `m2/m2_manifest.json` and complete per-epoch logs;
- `m3/m3_manifest.json` and complete per-epoch logs;
- `evaluation/ft6_evaluation_manifest.json`;
- `review/ft6_test_metrics.csv`;
- `review/ft6_comparison_to_m1.csv`;
- `review/ft6_training_trajectories.csv`;
- `review/ft6_training_trajectories.png`;
- `review/ft6_test_comparison_to_m1.png`.

## External Gate

FT6 establishes that the planned process is compatible with the existing
GridSFM/OPFData implementation and gives distinct, interpretable model
behavior. It does not establish Stage J OPS outcomes for M2 or M3. Those runs,
the five-method figures, and the revised TH-free warm-start study belong to
FT7 and require explicit approval.

