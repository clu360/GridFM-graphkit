# Stage J Fine-Tuned GridSFM Checkpoints

These are the actual GOC-500 weights from the earlier Stage J experiments,
provided in this repository for reuse. They are fine-tunes of Microsoft's
released Open GridSFM v1.1 OPF model, not GENCO or Texas2k-trained models.

| File | Model | Training exposure | Bytes |
|---|---|---|---:|
| `gridsfm_goc500_fulltop_ft0_n10.pt` | FT0 smoke | 10 FullTop samples | 61003599 |
| `gridsfm_goc500_fulltop_ft_n1000.pt` | M1 | 1,000 FullTop samples | 61007770 |
| `gridsfm_goc500_ft1000_plus_fulltop500.pt` | M2 | M1 continued on 500 additional FullTop samples | 61012764 |
| `gridsfm_goc500_ft1000_plus_n1_500.pt` | M3 | M1 continued on 500 N-1 samples | 61007856 |

M2 and M3 are sibling continuations from M1, not sequential continuations from
one another. The three main models each use ten official fine-tuning epochs
per recorded training stage. The FT0 file is a smoke artifact rather than a
production study model. See `CHECKPOINT_MANIFEST.json` for SHA-256 hashes and
lineage, and the existing `../results/ft0`, `../results/ft1`, and
`../results/ft6` manifests, logs, environment freezes, and reload checks for
the original evidence. Their historical absolute paths are preserved.

## Loading And Reproduction

Use the pinned official Microsoft GridSFM implementation and the existing
Stage J evaluator/loader rather than the unrelated GridFM or GENCO classes.
The recorded implementation is GridSFM 1.1.0 at commit
`1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`. Point the checkpoint argument at
the appropriate file in this directory. Changing the path does not change
the checkpoint format or scientific preprocessing.

The parent released checkpoint was `microsoft/GridSFM_Open`, filename
`gridsfm_open_v1.1.pt`, SHA-256
`f8a4396122e603e8303afdebe3b093819c0f64dac0878394aed0bd63205fd831`.
It is not duplicated here; these four files are the trained descendants.
Retain applicable upstream license and model-use terms when redistributing
or using the weights; this package does not assign them a new license.

## Local Cache Retirement

The weights were copied byte-for-byte from
`C:\Users\Caleb Lu\.gridfm_stage_j\checkpoints\stage_j_finetune` and checked
against the recorded hashes before publication. At Caleb's request the entire
legacy `C:\Users\Caleb Lu\.gridfm_stage_j` tree is retired after successful
publication: datasets, raw/processed caches, temporary experiment states,
environments, source checkouts, Julia tools, and duplicate checkpoint copies.
The inventoried legacy tree was 66,826,794,709 bytes (62.237 GiB).

The repository's existing reports and training evidence remain, but the full
62-GiB cache is not archived in this package. Reproducing training or rerunning
old commands requires rebuilding the environment, retrieving the original
datasets, and overriding historical external paths. The four published weights
total 244,031,989 bytes (approximately 232.73 MiB).

The files are explicitly tracked even though the repository generally ignores
`*.pt`; large training data and environment directories remain excluded.
