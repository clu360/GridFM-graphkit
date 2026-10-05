# Full Production Run

This directory contains the retrieved `stage_k_production_v001` result package
and the local post-run analysis built from its frozen primary tables.

- `STAGE_K_PRODUCTION_ANALYSIS.md`: methodology, results, interpretation, and evidence index.
- `stage_k_production_v001/`: retrieved PACE artifacts and validation records.
- `analysis/tables/`: retained data behind every post-run figure.
- `analysis/figures/`: PNG and PDF figure suite.
- `deployed_full.yaml`: exact PACE configuration used by the production run;
  retain this file for hash-matched local validation.
- `stage_k_production_v001.tar.gz`: retained first transfer archive. This archive
  is truncated and is not an analysis source; all used primary tables in the
  extracted package were independently readable.
