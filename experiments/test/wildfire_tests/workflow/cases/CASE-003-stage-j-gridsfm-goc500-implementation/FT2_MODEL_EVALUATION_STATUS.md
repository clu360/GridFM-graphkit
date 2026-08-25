# FT2 Frozen Versus Fine-Tuned Model Evaluation

## Status

```text
FT2_EVALUATED_READY_FOR_REVIEW
```

FT2 compared the released GridSFM v1.1 checkpoint and the completed FT1
checkpoint on two inference-only GOC-500 OPFData test sets. Each model saw the
same ordered FullTop test indices 0 through 749 and the same ordered N-1 test
indices 0 through 749 from one 15,000-graph group. Neither test split was used
by FT0 or FT1 training or validation.

## Provenance

```text
GridSFM commit:
1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6

released checkpoint SHA-256:
F8A4396122E603E8303AFDEBE3B093819C0F64DAC0878394AED0BD63205FD831

FT1 checkpoint SHA-256:
A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD
```

The manifest records direct checkpoint paths and hashes, the pinned Python
environment, private-helper source hashes, split identities, graph counts,
ordered evaluation, and official `gridsfm.eval_pass` metrics.

## Results

FullTop test:

| Metric | Frozen | FT1 | Relative change |
| --- | ---: | ---: | ---: |
| loss | 0.093351 | 0.041518 | -55.525% |
| cost MAPE | 0.008835 | 0.007413 | -16.093% |
| Pg MAE | 0.009184 | 0.005498 | -40.135% |
| Qg MAE | 0.048988 | 0.018973 | -61.270% |
| V MAE | 0.002302 | 0.001510 | -34.391% |
| theta MAE | 0.020620 | 0.009614 | -53.373% |
| branch P MAE | 0.074686 | 0.040645 | -45.580% |
| branch Q MAE | 0.052153 | 0.022690 | -56.494% |
| feasibility accuracy | 1.000000 | 1.000000 | unchanged |

N-1 test:

| Metric | Frozen | FT1 | Relative change |
| --- | ---: | ---: | ---: |
| loss | 0.137594 | 0.085317 | -37.994% |
| cost MAPE | 0.011588 | 0.008723 | -24.726% |
| Pg MAE | 0.011443 | 0.007591 | -33.662% |
| Qg MAE | 0.054581 | 0.025751 | -52.820% |
| V MAE | 0.002990 | 0.002186 | -26.882% |
| theta MAE | 0.021137 | 0.010354 | -51.014% |
| branch P MAE | 0.077109 | 0.043933 | -43.025% |
| branch Q MAE | 0.058596 | 0.029191 | -50.182% |
| feasibility accuracy | 0.998667 | 0.993333 | -0.005333 absolute |

FT1 also reduced both KCL residuals, mean maximum thermal loading, and thermal
overload fraction on both test sets. All metrics were finite, all four passes
counted 750 feasible OPF graphs, and frozen/fine-tuned output schemas matched.

## Interpretation And Gate

FullTop-only FT1 materially improves held-out FullTop prediction and physics
metrics. It also generalizes favorably to N-1 regression and physics metrics
without N-1 fine-tuning. The immediate divergence is feasibility-head behavior:
the released model missed one N-1 feasibility classification, while FT1 missed
five. OPFData regression metrics still count all 750 graphs because the ground
truth cases are feasible; the classifier result must remain a separate caveat.

FT2 reports aggregate official metrics and does not yet provide per-case paired
confidence intervals. It is sufficient to approve a separate FT3 Stage J
application, but not to claim that FT1 dominates the released checkpoint on
every model behavior.

Any later FullTop+N-1 fine-tuning experiment must use N-1 training/validation
data that exclude these test cases. The same two FT2 test splits remain sealed
for the frozen, FullTop-only FT1, and future FullTop+N-1 model comparison.

## Artifacts

```text
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/results/ft1/
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/results/ft2/
```

The large OPFData caches and model checkpoints remain under
`C:\Users\Caleb Lu\.gridfm_stage_j` and outside Git.
