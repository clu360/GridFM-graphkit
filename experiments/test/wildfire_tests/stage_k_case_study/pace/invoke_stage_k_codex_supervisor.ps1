param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('ISSUE', 'COMPLETE')]
    [string]$EventType,

    [Parameter(Mandatory = $true)]
    [string]$EventDetails
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..\..\..')).Path
$SupervisorDir = Join-Path $RepoRoot 'experiments\test\wildfire_tests\texas_2k_results\full_run\supervisor'
$Codex = Join-Path $env:APPDATA 'npm\codex.cmd'
$Timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Output = Join-Path $SupervisorDir ("codex-{0}-{1}.md" -f $EventType.ToLowerInvariant(), $Timestamp)

New-Item -ItemType Directory -Force -Path $SupervisorDir | Out-Null
if (-not (Test-Path -LiteralPath $Codex)) {
    throw "Codex CLI not found at $Codex"
}

$Common = @"
This is an automated Stage K Texas2k PACE supervisor event.

Repository: $RepoRoot
Event: $EventType
Event details:
$EventDetails

Begin by reading experiments/test/wildfire_tests/HISTORY.md, the Stage K workflow,
PACE_PRODUCTION_RUNBOOK.md, the active submitted_job_ids.csv on PACE, and all
relevant active-job logs. Preserve the frozen five-lambda Stage K methodology,
candidate budgets, objective, solver settings, evaluator definitions, Reference
A/B definitions, B2 failure policy, hashes, and provenance.

Never make a scientific or methodological change automatically. Do not silently
filter failures, change tolerances, tune solvers, alter budgets, redefine risk,
or replace an evaluator. Authentication failures, account/allocation problems,
ambiguous scientific failures, and required methodological changes are hard
stops: record a clear report for the user and take no speculative action.

Infrastructure recovery is authorized when it preserves scientific identity.
Use exact job IDs, preserve failed logs and ledgers, avoid duplicate completed
work, update submitted_job_ids.csv after a replacement graph, and verify the
new graph rather than assuming submission succeeded.
"@

if ($EventType -eq 'ISSUE') {
    $Task = @"
$Common

Diagnose the active production failure and perform the smallest auditable
infrastructure/code-packaging recovery that preserves the frozen methodology.
Leave healthy evaluator jobs and valid checkpoints untouched. Verify repaired
jobs pass the prior failure point. Write a concise incident and recovery report
under texas_2k_results/full_run/supervisor. If recovery reaches a hard stop,
document exactly what user action is required.
"@
} else {
    $Task = @"
$Common

First verify that RUN_COMPLETE.json is valid and that all evaluator, finalist,
Reference A/B, validation, provenance, and Slurm-accounting requirements pass.
Then retrieve the immutable production package with hash verification into
experiments/test/wildfire_tests/texas_2k_results/full_run/stage_k_production_v001.
Follow texas_2k_results/POST_RUN_ANALYSIS_PLAN.md to construct the local tables,
figures, supporting evidence, and methodological results report. Do not claim
completion if validation, retrieval, or required evidence is incomplete.
"@
}

Write-Host "Launching Codex supervisor for $EventType event" -ForegroundColor Yellow
Write-Host "Final response: $Output"

# PACE recovery needs SSH/network access and the user's private key. The prompt
# above limits autonomous action to the frozen Stage K contract and hard stops.
& $Codex exec --sandbox danger-full-access --cd $RepoRoot --output-last-message $Output $Task
if ($LASTEXITCODE -ne 0) {
    throw "Codex supervisor exited with code $LASTEXITCODE"
}
