# FT3/FT4 Execution Status

## Status

```text
FT3: COMPLETE
FT4: PASS
model_selection: ft
model_variant: fulltop_ft_n1000
```

FT3 evaluated only the fine-tuned GridSFM checkpoint. Guided-DC and released
GridSFM v1.1 were not rerun; their completed Stage J packages remain the
comparison evidence for later cross-model analysis.

## Provenance

```text
GridSFM commit:
1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6

FT checkpoint SHA-256:
A1378FDAF38AF6B317F172C27DF7C0F14A23138143D65DFE5E1C46C613E119AD

working cache:
C:\Users\Caleb Lu\.gridfm_stage_j\cache\stage_j\
  ft3_fulltop_ft_n1000_v002

authoritative working results:
C:\Users\Caleb Lu\.gridfm_stage_j\results\stage_j\
  fulltop_ft_n1000_complete_run

Git publication:
experiments/test/wildfire_tests/stage_j_gridsfm_goc500/finetune/
  results/ft3_ft4_v002
```

All 30 ordered Guided/TH topology pool files were copied from the frozen Stage J
package and verified byte-identical. The production cache was protected by an
exclusive single-writer lock. The validated v002 computation took approximately
3 hours 51 minutes across one controlled resume.

## Validated Counts

```text
scenario/lambda settings:       15
topology outcomes:           1,530
candidate evaluations:      30,600
FT finalists:                   45
Reference A rows:               45
Reference B rows:               90
warm-start rows:               180
state-fidelity rows:           360
figures:                        13
derived visual tables:           4
```

Every core row identifies `model_selection=ft` and
`model_variant=fulltop_ft_n1000`. Checkpoint, model variant, pool identity,
artifact count, figure non-emptiness, and FT3 completion checks all passed.

## Cross-Model Figure Assembly

The final 13 figures join existing evidence rather than rerunning DC or frozen
GridSFM. Their method set is Guided-DC, frozen Guided-GridSFM, fine-tuned
Guided-GridSFM, TH-GridSFM-top1, and TH-GridSFM-top2. TH uses the established
frozen-run baseline; the FT package contributes only its guided rows.

The three native risk/load figures were subsequently strengthened to use all
eligible finite rows in `compare_candidates` instead of only one retained
alpha per topology. Each scenario pools candidates discovered under all five
lambda settings, deduplicates identical risk/load coordinates, and computes a
separate nondominated frontier for each of the five methods. The 90,201 source
evaluations yield 343 frontier rows. This is an evaluated native frontier and
does not claim exhaustive or exact-AC Pareto optimality.

```text
comparison topology rows:          4,530
comparison candidate rows:        90,600
comparison Reference A rows:          75
comparison Reference B rows:         150
comparison state-fidelity rows:      600
comparison warm-start rows:          300
comparison variants: dc, frozen, ft, th_frozen
```

The comparison tables are written as `core_results/compare_*.csv`. Updated FT4
validation requires all five plotted labels and all four provenance variants in
each table. This closes the earlier visualization gap where the FT-only package
could not display DC and released-checkpoint GridSFM.

## Publication

The external working package retains CSV because the existing Stage J analysis
scripts consume it directly. The Git publication converts every CSV artifact to
Parquet after FT4 validation and retains JSON, Markdown, and figure files.

```text
published artifacts:              386
Parquet tables:                    305
rows read back from Parquet:   199,640
published CSV files:                 0
hash failures:                       0
working source bytes:      186,793,281
published artifact bytes:   22,540,652
size reduction:                  87.9%
```

Windows path limits require short deterministic path-hash filenames in the flat
Git publication. `PUBLICATION_MANIFEST.json` maps each identifier to the full
original relative path and records source/published hashes, sizes, format, and
row count.

## Immediate Divergences

- The pinned fine-tuning environment required `gurobipy==12.0.0` because the
  unchanged J9 DC partial warm-start construction imports it.
- Newly copied result trees under the repository's OneDrive-managed
  `goc_500_results` path became unreadable placeholders. The authoritative
  working package was therefore moved to local external storage before final
  aggregation. This changes storage location only, not experiment semantics.
- FT4 plotting initially summarized only methods physically present in the
  FT-only package. A model-aware comparison layer now joins frozen/DC and FT
  evidence, labels checkpoint provenance explicitly, and retains frozen TH as
  the heuristic baseline.
- Failed publication trials were moved, not deleted, to
  `C:\Users\Caleb Lu\.gridfm_stage_j\quarantine\
  ft3_ft4_failed_publications_20260822`.

## Next Gate

FT5 may join this FT package to the existing released-v1.1 and Guided-DC
packages by scenario, lambda, opportunity-set identity, method family, and
provenance keys. No cross-model claim is made solely from FT3/FT4 completion.
