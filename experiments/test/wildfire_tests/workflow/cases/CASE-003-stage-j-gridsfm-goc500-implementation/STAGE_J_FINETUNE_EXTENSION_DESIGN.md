# Stage J GridSFM Fine-Tuning Extension Design

## Scope

This extension is a controlled model-weight intervention on the completed
Stage J experiment. The GridSFM architecture and the complete Stage J search,
objective, PAC, scenario, topology, alpha, and exact-reference methodology stay
fixed. The eventual paired comparison is released GridSFM v1.1 versus the same
architecture fine-tuned on GOC-500 FullTop OPFData, with Guided-DC retained as
the established approximation comparator.

FT0 is a pipeline smoke and is not a scientific result. FT0 review approved
FT1 execution with the additional controls in `FT1_EXECUTION_REQUIREMENTS.md`.
FT1 and FT2 are complete. FT2 is ready for review before a separately gated
FT3 application to the completed Stage J methodology. FT4 and FT5 remain
unexecuted.

## Official dependency boundary

The local wrapper imports the following directly from Microsoft/GridSFM:

- `load_model`, `OPFDataAdapterDataset`, and `SyntheticMixedDataset`;
- `CycleBasisCache` and `prepare_for_grid_transformer_`;
- `LaplacianFactorizationCache` and `attach_pe_features_`;
- `eval_pass` and `finetune_opfdata`.

No architecture, training loss, OPFData schema adapter, topology feature
preparation, or optimizer loop is duplicated in this repository.

## FT0 frozen controls

- base checkpoint: `gridsfm_open_v1.1.pt`;
- case: `pglib_opf_case500_goc`;
- train data: FullTop train indices 0 through 9;
- held-out data: FullTop test indices 0 through 9;
- synthetic infeasibility probability: 0.3;
- seed: 42;
- batch size: 2;
- epochs: 2;
- learning rate: `1e-4`;
- model checkpoint and OPFData cache: external to Git.

## Stage J integration gate

The smoke reloads one existing two-outage Guided-GridSFM finalist from the
completed evidence package and evaluates the identical `(scenario, z, alpha)`
with both checkpoints. The existing `evaluate_gridsfm_candidate` function is
used without changing its mathematics. Both evaluations must retain
`D_input = 0` and the existing canonical mapping and unit gates.

## Known upstream API gaps

GridSFM v1.1 exposes no public fine-tuned checkpoint exporter. The wrapper uses
the documented minimal release payload (`state_dict`, `metadata`) and the
package's own `_hash_state_dict` integrity routine so `load_model` performs its
normal strict validation. GridSFM also does not expose OPFData branch flows as
a public evaluation payload; FT0's fresh-reload shape/finite gate uses the same
internal `_predicted_flows` helper called by official `eval_pass`.

These are wrapper-level API gaps, not changes to model or experiment
methodology. They must be reconsidered if the pinned GridSFM commit changes.

## FT1 and FT3 provenance additions

The current GridSFM commit, package environment, and private-helper source
hashes are frozen before FT1. FT1 records official held-out validation metrics
for every epoch. All FT1 and future FT3 manifests must include checkpoint path
and SHA-256 directly; FT3 must record both frozen and fine-tuned checkpoint
provenance before any Stage J execution.

## FT1 and FT2 checkpoint

FT1 completed ten epochs over FullTop train indices 0 through 999 with official
metrics recorded on all 750 FullTop validation graphs after every epoch. Its
fresh-process checkpoint reload passed at SHA-256
`A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD`.

FT2 then evaluated the released and FT1 checkpoints on the separate FullTop
and N-1 test splits, 750 graphs each. FT1 reduced official regression and
physics errors on both variants. N-1 feasibility accuracy declined from
`0.998667` to `0.993333`, so this behavior remains an explicit review item.
See `FT2_MODEL_EVALUATION_STATUS.md` for the complete checkpoint.

The frozen fine-tuning recipe and interpretation are recorded in
`FT1_FINE_TUNED_MODEL_SETTINGS.md`. The linked checkpoint-only Stage J rerun
and existing-evidence reproduction contract is recorded in
`FT3_FT4_EXECUTION_PLAN.md`.

## Sequential fine-tuning ablation extension

FT0 through FT5 are complete historical evidence and are not modified by the
next study. The concluding extension is specified in
`FT6_FT7_PLAN.md`.

FT6 branches two sibling continuations from the byte-identical
`fulltop_ft_n1000` checkpoint. One receives 500 new FullTop training cases and
the other receives 500 N-1 training cases. Both are evaluated with released
v1.1 and `fulltop_ft_n1000` on one sealed 750-case diagnostic set containing
375 FullTop test graphs and 375 N-1 test graphs. Training, validation, and test
splits remain explicitly disjoint.

FT7 is externally gated. It may begin only after FT6 results are reviewed and
Caleb explicitly approves application to Stage J. The follow-up OPS figures use
Guided-DC plus four GridSFM checkpoints; TH remains preserved in prior packages
but is excluded from the new figures and crossed warm-start summary.
