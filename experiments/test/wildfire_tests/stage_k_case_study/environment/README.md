# Frozen GridSFM A100 Environment

`GRIDSFM_ENVIRONMENT_CONTRACT.json` is the machine-readable Stage K contract.
It pins the official Microsoft source revision, released v1.1 checkpoint,
Python packages, and required one-A100 CUDA context.

`setup_gridsfm_a100.sh` creates the environment, checks out the official
Microsoft package at the frozen commit, installs that package under exact
constraints, and downloads the checkpoint once with the real
`huggingface_hub` package. The checkpoint is hash-checked and made read-only.
PACE jobs set `HF_HUB_OFFLINE=1` and load this local file; they do not contact
Hugging Face and do not use the temporary Gate 2 import stub.

The setup requires these absolute variables:

```bash
export STAGE_K_GRIDSFM_ENV_PREFIX='<PROJECT_DIR>/envs/stage-k-gridsfm-a100'
export STAGE_K_GRIDSFM_ROOT='<PROJECT_DIR>/external/GridSFM'
export STAGE_K_GRIDSFM_CHECKPOINT_DIR='<PROJECT_DIR>/checkpoints'
bash experiments/test/wildfire_tests/stage_k_case_study/environment/setup_gridsfm_a100.sh
```

The A100 preflight writes `gridsfm_environment_manifest.json`. That observed
manifest must report `PASS`, and the GridSFM batch job validates it and the live
job environment against the same contract before loading the model.
