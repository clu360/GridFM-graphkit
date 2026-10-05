# Stage K PACE/Phoenix Smoke-Test Runbook

This guide begins on a Windows computer and ends at `smoke_summary.md`. It is
written for a first Phoenix run. Gate 2 does not authorize any command marked
**Submit**. Run through the dry run, inspect it, and obtain Gate 3 approval
before using `--submit`.

## 1. Connect From Windows

Connect to the Georgia Tech VPN using the normal client, then open Windows
Terminal with a PowerShell tab.

Run:

```powershell
ssh <GT_USERNAME>@login-phoenix.pace.gatech.edu
```

This opens a Phoenix login shell. Success shows a Phoenix prompt. If DNS,
timeout, or authentication fails, confirm VPN connectivity, the current PACE
login hostname, Duo authentication, and that the account has Phoenix access.

## 2. Inspect Identity And Accounts

Run on Phoenix:

```bash
id
groups
sacctmgr show assoc user="$USER" format=User,Account,Partition,QOS%30
```

These commands display your Unix identity and Slurm associations. Success is a
row containing your user and at least one authorized account. If no association
appears, stop and contact the project owner or PACE support. Do not guess an
account string.

Set the approved account only in the live shell:

```bash
export PACE_ACCOUNT='<PACE_ACCOUNT>'
```

This value is used by the submission helper but is never committed. Success is
an empty command response. Check it with `printf '%s\n' "$PACE_ACCOUNT"`.

## 3. Discover GNR And A100 Resource Syntax

Run:

```bash
sinfo -o '%P %a %l %D %G %f' | less
scontrol show partition
scontrol show nodes | grep -Ei 'gnr|granite|a100' | head -80
```

These commands show the live partition, feature, and GPU names. Success is a
GNR-capable CPU resource and an A100-capable GPU resource authorized for your
account. If the terms are absent, inspect all features and PACE's current
Phoenix resource documentation or ask PACE support.

Phoenix assigns partitions automatically from the account, QoS, and most
significant requested resource. The September 21, 2026 Gate 3 probes verified
the following resource-driven arguments:

```bash
export STAGE_K_CPU_RESOURCE_ARGS='--constraint=graniterapids --time=02:00:00'
export STAGE_K_GPU_RESOURCE_ARGS='--gres=gpu:v100:1 --time=02:00:00'
```

Do not add an explicit partition unless PACE changes its documented submission
policy. A live CPU probe using the first argument string reached `cpu-gnr` and
reported two Intel Xeon 6972P sockets, 192 cores, and about 1.5 TiB RAM. A live
The smoke-only September 21 deployment amendment uses an idle V100 pool to
avoid the multi-hour A100 queue. Production remains contracted to A100. Step
11 must verify a physical V100 and the separate smoke-only contract before GPU
preflight can pass.

Success is obtained later when `sbatch --test-only` accepts these flags and the
live preflights receive the required hardware. A syntactically accepted dry
run alone is not hardware evidence because Phoenix may rewrite placement.

## 4. Select Storage

Inspect available locations:

```bash
pwd
df -h "$HOME"
env | grep -Ei 'project|scratch'
```

Choose durable project storage for the repository and scratch storage for run
artifacts. Then set explicit paths:

```bash
export STAGE_K_PROJECT_DIR='<PROJECT_DIR>/GridFM-graphkit'
export STAGE_K_SCRATCH_DIR='<SCRATCH_DIR>/stage_k'
export STAGE_K_RUN_ID='stage_k_smoke_v001'
export STAGE_K_RUN_DIR="$STAGE_K_SCRATCH_DIR/$STAGE_K_RUN_ID"
mkdir -p "$STAGE_K_PROJECT_DIR" "$STAGE_K_RUN_DIR/prepared" "$STAGE_K_RUN_DIR/preflight" "$STAGE_K_RUN_DIR/logs"
```

This creates narrowly scoped directories. Success is no error and
`ls -ld "$STAGE_K_PROJECT_DIR" "$STAGE_K_RUN_DIR"` shows both. If creation
fails, check quota, permissions, and that the placeholders were replaced.

## 5. Transfer The Repository

From a second Windows PowerShell tab, run:

```powershell
scp -r "C:\Users\Caleb Lu\OneDrive - Georgia Institute of Technology\GT\Extracurriculars\Research\Grid FM\Experiments\GridFM-graphkit\*" <GT_USERNAME>@login-phoenix.pace.gatech.edu:<PROJECT_DIR>/GridFM-graphkit/
```

This transfers the repository data. Success reaches 100 percent without
errors. For later updates, use the site's approved Git or `rsync` workflow
rather than retransferring outputs.

Back on Phoenix, verify:

```bash
cd "$STAGE_K_PROJECT_DIR"
test -f experiments/test/wildfire_tests/stage_k_case_study/data/raw/modifiedTexas2k.m
test -f experiments/test/wildfire_tests/texas_2k_results/stage_k/environment_snapshot_tau0p50/data/cum_hazard_risk.parquet
```

Success produces no `test` output. If a file is
missing, stop and repair the transfer rather than changing configured inputs.

## 6. Create The Frozen GridSFM Environment And Cache

Inspect available environment tools:

```bash
module avail anaconda 2>&1 | head -60
module avail miniconda 2>&1 | head -60
module avail cuda 2>&1 | head -60
```

Load the site-recommended Conda module. The exact module name comes from the
previous output. The setup manifest itself pins Python; do not substitute a
different site Python module.

```bash
module load <CONDA_MODULE>
export STAGE_K_GRIDSFM_ENV_PREFIX='<PROJECT_DIR>/envs/stage-k-gridsfm-a100'
export STAGE_K_GRIDSFM_ROOT='<PROJECT_DIR>/external/GridSFM'
export STAGE_K_GRIDSFM_CHECKPOINT_DIR='<PROJECT_DIR>/checkpoints'
mkdir -p "$(dirname "$STAGE_K_GRIDSFM_ROOT")" "$STAGE_K_GRIDSFM_CHECKPOINT_DIR"
bash experiments/test/wildfire_tests/stage_k_case_study/environment/setup_gridsfm_a100.sh
```

The script checks out the official `https://github.com/microsoft/GridSFM.git`
repository at commit `1ca775fd436d7ce013a1c0ab946e61ac7ef59ad6`, installs its
official `model` package as GridSFM 1.1.0 under the frozen constraints, and
downloads `microsoft/GridSFM_Open/gridsfm_open_v1.1.pt` once through the real
`huggingface_hub` package at revision
`1b41299b80252adf1869d5c3b479a4a402c52591`. It verifies SHA-256
`f8a4396122e603e8303afdebe3b093819c0f64dac0878394aed0bd63205fd831`
and makes the local file read-only.

If Hugging Face access is unavailable on Phoenix login nodes, transfer that
exact file once into `$STAGE_K_GRIDSFM_CHECKPOINT_DIR` before rerunning setup.
The downloader detects a matching local hash and performs no network request.

Activate and expose the frozen paths:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STAGE_K_GRIDSFM_ENV_PREFIX"
export STAGE_K_PYTHON="$STAGE_K_GRIDSFM_ENV_PREFIX/bin/python"
export STAGE_K_GRIDSFM_CHECKPOINT="$STAGE_K_GRIDSFM_CHECKPOINT_DIR/gridsfm_open_v1.1.pt"
```

Verify the official package imports and exact versions without an import stub:

```bash
"$STAGE_K_PYTHON" -c 'import gridsfm, huggingface_hub, numpy, scipy, torch, torch_geometric; print(gridsfm.__version__, huggingface_hub.__version__, numpy.__version__, scipy.__version__, torch.__version__, torch_geometric.__version__)'
"$STAGE_K_PYTHON" -m pip check
sha256sum "$STAGE_K_GRIDSFM_CHECKPOINT"
```

Success reports GridSFM 1.1.0, Python 3.11.9, torch 2.7.1+cu126, torch-geometric
2.6.1, NumPy 1.26.4, SciPy 1.15.3, huggingface-hub 0.34.4, Lightning 2.6.6, no dependency
errors, and the expected checkpoint hash. Any mismatch blocks Gate 3; do not
repair it by installing individual unrecorded packages.

## 7. Create The Julia Environment

Run:

```bash
module avail julia 2>&1 | head -60
module load <JULIA_MODULE>
julia --project=experiments/test/wildfire_tests/stage_k_case_study/src/native_opf -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

This installs the versions constrained in `Project.toml`. Success completes
precompilation without package errors. If downloads are blocked, use PACE's
Julia package guidance or a project depot; do not edit the scientific model.

Verify:

```bash
julia --project=experiments/test/wildfire_tests/stage_k_case_study/src/native_opf -e 'using PowerModels, Ipopt, JuMP; println("Julia OPF stack OK")'
```

Success prints `Julia OPF stack OK`. If it fails, inspect package status with
`Pkg.status()` and preserve the error for Gate 3 review.

## 8. Fix Threading And Record It

Run:

```bash
export JULIA_NUM_THREADS=8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
env | grep -E 'JULIA_NUM_THREADS|OMP_NUM_THREADS|MKL_NUM_THREADS|OPENBLAS_NUM_THREADS'
```

This prevents hidden BLAS/OpenMP oversubscription. Success prints exactly the
four values. Do not raise them merely because a GNR node has 192 cores.

## 9. Prepare And Validate Immutable Inputs

Run on a login node because this step is data preparation, not OPF compute:

```bash
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope inputs \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" \
  --output-dir "$STAGE_K_RUN_DIR/preflight/inputs"
```

This binds hashes, canonical tables, the stored baseline, `c_l`, and shared K1
pool. Success prints `"status": "PASS"` and writes `PREFLIGHT_PASS.json`. If it
fails, inspect `PREFLIGHT_FAIL.json`; never substitute a different date, scenario, loading,
or branch mapping to make it pass.

Confirm key values:

```bash
"$STAGE_K_PYTHON" -c "import json; p=json.load(open('$STAGE_K_RUN_DIR/prepared/input_manifest.json')); print(p['counts']); print(p['r_base'])"
```

Expected counts are 2,751 buses, 1,125 loads, 1,099 generators, 5,344 physical
branches, 3,993 `L_trans`, and 1,351 fixed transformer/other branches. Expected
`R_base` is approximately `32.9754256993`.

## 10. Run CPU Preflight On GNR

Request an interactive CPU allocation using the verified resource flags:

```bash
srun $STAGE_K_CPU_RESOURCE_ARGS --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --pty bash
```

Success opens a shell on a compute node. If Slurm rejects a flag, return to
Step 3 and correct the complete resource argument string.

Before running preflight, verify that placement was not rewritten:

```bash
test "$SLURM_JOB_PARTITION" = cpu-gnr
lscpu | grep 'Model name'
```

Success reports `Intel(R) Xeon(R) 6972P`. Any other partition or processor
blocks the GNR preflight evidence.

Inside the allocation, reactivate environments and run:

```bash
cd "$STAGE_K_PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STAGE_K_GRIDSFM_ENV_PREFIX"
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope cpu \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" \
  --output-dir "$STAGE_K_RUN_DIR/preflight/cpu"
exit
```

This checks Julia/PowerModels/Ipopt and runs the intact AC deployment audit.
Success requires an accepted solver status and exactly 5,344 unique, finite
physical branch-loading outputs. The comparison against stored Scenario-16
loading remains in the report as a diagnostic and is not a PASS criterion,
because the deployment solve is a newly redispatched economic AC-OPF. The
stored loading remains authoritative and cannot be replaced by this solve.

## 11. Run A100 Preflight

For normal operation, submit the durable preflight batch job and let Slurm
hold it until an A100 is available:

```bash
gpu_preflight_id=$(sbatch --parsable ${PACE_ACCOUNT:+--account="$PACE_ACCOUNT"} \
  $STAGE_K_GPU_RESOURCE_ARGS \
  experiments/test/wildfire_tests/stage_k_case_study/pace/stage_k_gpu_preflight_a100.sbatch)
printf 'GPU preflight job: %s\n' "$gpu_preflight_id"
```

Monitor it with `squeue` and `sacct`. Continue to Step 12 only after it reports
`COMPLETED` and the GPU `PREFLIGHT_PASS.json` exists. The batch script performs
the same hardware, environment, checkpoint, and intact-inference checks shown
below. An interactive allocation may instead be used for troubleshooting:

```bash
srun $STAGE_K_GPU_RESOURCE_ARGS --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --pty bash
```

Then run:

```bash
cd "$STAGE_K_PROJECT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$STAGE_K_GRIDSFM_ENV_PREFIX"
nvidia-smi --query-gpu=index,name,uuid,driver_version --format=csv
export HF_HUB_OFFLINE=1
"$STAGE_K_PYTHON" experiments/test/wildfire_tests/stage_k_case_study/scripts/preflight.py \
  --scope gpu \
  --config experiments/test/wildfire_tests/stage_k_case_study/config/smoke.yaml \
  --prepared-dir "$STAGE_K_RUN_DIR/prepared" \
  --output-dir "$STAGE_K_RUN_DIR/preflight/gpu" \
  --gridsfm-root "$STAGE_K_GRIDSFM_ROOT" \
  --checkpoint "$STAGE_K_GRIDSFM_CHECKPOINT" \
  --device cuda:0
exit
```

Confirm that `SLURM_JOB_PARTITION` is `gpu-a100` and that `nvidia-smi` reports
exactly one A100 visible to the job. Queue delay is acceptable; a different GPU
model or partition blocks the A100 preflight evidence.

This Gate 3 check verifies imports, exact package and source versions, CUDA
12.6, exactly one visible A100 with compute capability 8.0, the immutable local v1.1 checkpoint, and one
intact 2,751-bus Texas2k inference. It writes the observed versions, checkpoint
identity, and device details to
`$STAGE_K_RUN_DIR/preflight/gpu/gridsfm_environment_manifest.json`. GridSFM
preflight is `PASS` only if every contract and inference check succeeds. If it
fails, retain `PREFLIGHT_FAIL.json`, the environment manifest, and logs, then
stop; do not use a CPU fallback, import stub, alternate checkpoint, or silently
reshaped Texas2k input.

## 12. Review Preflight Before Submission

Run:

```bash
find "$STAGE_K_RUN_DIR/preflight" -name 'PREFLIGHT_PASS.json' -print -exec grep -H '"status"' {} \;
export STAGE_K_GRIDSFM_ENV_MANIFEST="$STAGE_K_RUN_DIR/preflight/gpu/gridsfm_environment_manifest.json"
test "$("$STAGE_K_PYTHON" -c "import json; print(json.load(open('$STAGE_K_GRIDSFM_ENV_MANIFEST'))['status'])")" = PASS
```

Success shows PASS for inputs, CPU, and GPU. Any missing/failed file blocks job
submission. Confirm that the environment manifest records the frozen Git
commit, GridSFM/package versions, checkpoint repository/name/hash, Python and
dependency versions, CUDA runtime, and one A100. This is the Gate 3 review
point.

## 13. Dry-Run The Slurm Submission

Run:

```bash
cd "$STAGE_K_PROJECT_DIR"
bash experiments/test/wildfire_tests/stage_k_case_study/pace/submit_smoke.sh --dry-run
```

This prints five intended submissions without calling `sbatch`. Confirm one
GPU job, four GNR jobs, no private value inside repository files, an `afterok`
reference dependency, and a separate `afterok` aggregation dependency. The
aggregation job intentionally requests only 2 CPUs and 8 GB. If any resolved
flag is wrong, edit only live resource variables or reviewed resource config.

Optionally ask Slurm to validate each printed command with `sbatch --test-only`
before submission. Success is a message that the job would be accepted. Queue
delay is not a methodological failure.

## 14. Submit Only After Gate 3 Approval

**Submit:** after explicit review approval, run:

```bash
bash experiments/test/wildfire_tests/stage_k_case_study/pace/submit_smoke.sh --submit | tee "$STAGE_K_RUN_DIR/submitted_job_ids.txt"
```

Success prints five numeric job IDs. If submission partially fails, preserve
the printed IDs, inspect `squeue`, and do not resubmit blindly because that can
duplicate evaluator outputs.

## 15. Monitor And Inspect Logs

Run:

```bash
squeue -u "$USER" -o '%.18i %.22j %.2t %.10M %.6D %R'
sacct -j <JOB_ID_LIST> --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,NodeList
find "$STAGE_K_PROJECT_DIR/logs" -type f -maxdepth 1 -print
```

These commands show queue state, resource use, and logs. Success progresses
from pending/running to completed. For failures, inspect the matching `.err`
and `.out`; preserve candidate failure rows, while treating job-level crashes
as implementation/deployment failures.

The Reference job should remain pending on dependency until GridSFM, DC, and AC
complete. If it shows `DependencyNeverSatisfied`, inspect the failed upstream
job rather than removing the dependency.

## 16. Inspect Final Results

Run:

```bash
test -f "$STAGE_K_RUN_DIR/report/smoke_summary.md"
sed -n '1,160p' "$STAGE_K_RUN_DIR/report/smoke_summary.md"
find "$STAGE_K_RUN_DIR/report" -maxdepth 2 -type f | sort
```

Success displays the diagnostic smoke summary and derived tables/figures. The
result is not yet a scientific validation or production authorization.

Collect final utilization:

```bash
job_ids=$(tr '\n' ',' < "$STAGE_K_RUN_DIR/submitted_job_ids.txt" | grep -oE '[0-9]+' | paste -sd, -)
sacct -j "$job_ids" --format=JobID,JobName,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,NodeList > "$STAGE_K_RUN_DIR/resource_usage.txt"
```

Success writes the evidence needed to judge 8 versus 16 CPUs and estimate the
full run. If `sacct` fields are blank, wait for accounting to settle or use the
PACE-recommended accounting command.

## 17. Stop At Gate 4

Review convergence, failures, deltas, empirical nondominance, parent
divergence, timings, memory, and utilization. Do not switch to `full.yaml` or
submit production jobs until the smoke package receives separate approval.
