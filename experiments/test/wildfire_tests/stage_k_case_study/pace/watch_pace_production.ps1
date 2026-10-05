param(
    [switch]$EnableCodexAutomation
)

$ErrorActionPreference = "Continue"

$Identity = Join-Path $HOME ".ssh\id_ed25519_pace"
$Remote = "clu360@login-phoenix.pace.gatech.edu"
$RunDir = "/storage/scratch1/9/clu360/stage_k/stage_k_production_v001"
$Project = "/storage/scratch1/9/clu360/stage_k/workspace_v3/GridFM-graphkit"
$Supervisor = Join-Path $PSScriptRoot 'invoke_stage_k_codex_supervisor.ps1'
$EventDir = Join-Path $PSScriptRoot '..\..\texas_2k_results\stage_k\full_run\supervisor\events'

function Get-EventHash([string]$Value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    $hash = [Security.Cryptography.SHA256]::Create().ComputeHash($bytes)
    return ([BitConverter]::ToString($hash)).Replace('-', '').ToLowerInvariant()
}

if ($EnableCodexAutomation) {
    New-Item -ItemType Directory -Force -Path $EventDir | Out-Null
}

while ($true) {
    Clear-Host
    Write-Host "Stage K Phoenix production monitor" -ForegroundColor Cyan
    Write-Host (Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz")
    Write-Host "Read-only refresh every 30 seconds. Press Ctrl+C to close.`n"
    Write-Host ("Codex event automation: " + $(if ($EnableCodexAutomation) { 'ENABLED' } else { 'disabled' }))
    Write-Host

    $remoteCommand = @'
RUN_DIR='__RUN_DIR__'
PROJECT='__PROJECT__'
LEDGER="$RUN_DIR/submitted_job_ids.csv"

echo '=== ACTIVE PRODUCTION JOB IDS ==='
cat "$LEDGER" 2>/dev/null || echo 'not submitted yet'

job_ids=$(tail -n +2 "$LEDGER" 2>/dev/null | cut -d, -f2 | paste -sd, -)
echo
echo '=== ACTIVE PRODUCTION QUEUE ==='
if [ -n "$job_ids" ]; then
    squeue -j "$job_ids" -r -o '%i|%j|%P|%T|%M|%l|%R'
else
    echo 'no active production job IDs'
fi

echo
echo '=== ACTIVE PRODUCTION ACCOUNTING ==='
if [ -n "$job_ids" ]; then
    sacct -j "$job_ids" --format=JobID,JobName,State,Elapsed,ExitCode,Reason -n -P
fi

echo
echo '=== GENERATED PRODUCTION STATUS ==='
cat "$RUN_DIR/production_status.json" 2>/dev/null || echo 'not created yet'

echo
echo '=== NONEMPTY STDERR FOR ACTIVE JOB IDS ==='
found_error=0
while IFS=, read -r label job_id; do
    [ "$label" = 'job' ] && continue
    for path in "$PROJECT"/logs/*-"$job_id"*.err; do
        [ -s "$path" ] || continue
        found_error=1
        echo "--- $label: $path"
        tail -n 12 "$path"
    done
done < "$LEDGER"
[ "$found_error" -eq 1 ] || echo 'none'
'@
    $remoteCommand = $remoteCommand.Replace('__RUN_DIR__', $RunDir).Replace('__PROJECT__', $Project)
    & ssh -i $Identity $Remote $remoteCommand

    if ($EnableCodexAutomation) {
        $connectionState = (& ssh -o BatchMode=yes -o ConnectTimeout=10 -i $Identity $Remote 'echo CONNECTED' 2>$null | Select-Object -Last 1)
        if ($LASTEXITCODE -ne 0 -or $connectionState.Trim() -ne 'CONNECTED') {
            $event = 'OFFLINE|PACE SSH unavailable; jobs remain remote and monitoring will retry'
        } else {
            $completeState = (& ssh -i $Identity $Remote "if test -f '$RunDir/RUN_COMPLETE.json'; then echo COMPLETE; else echo INCOMPLETE; fi" | Select-Object -Last 1).Trim()
            $ledgerText = & ssh -i $Identity $Remote "cat '$RunDir/submitted_job_ids.csv' 2>/dev/null"
            $jobs = @($ledgerText | ConvertFrom-Csv)

            if ($completeState -eq 'COMPLETE') {
                $event = 'COMPLETE|RUN_COMPLETE.json present'
            } elseif ($jobs.Count -eq 0 -or -not $jobs[0].job_id) {
                $event = 'ISSUE|active production ledger missing or empty'
            } else {
                $jobIds = ($jobs.job_id -join ',')
                $accounting = & ssh -i $Identity $Remote "sacct -X -j '$jobIds' --format=JobIDRaw,State,ExitCode -n -P"
                $badStates = '^(FAILED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED|CANCELLED)'
                $badRows = @(
                    foreach ($line in $accounting) {
                        $fields = $line.Trim() -split '\|'
                        if ($fields.Count -ge 3 -and $fields[1] -match $badStates) {
                            "{0}:{1}:{2}" -f $fields[0], $fields[1], $fields[2]
                        }
                    }
                ) | Sort-Object -Unique

                if ($badRows.Count -gt 0) {
                    $event = 'ISSUE|terminal Slurm state(s): ' + ($badRows -join ';')
                } else {
                    $aggregateId = ($jobs | Where-Object job -eq 'aggregate').job_id
                    $aggregateLine = $accounting | Where-Object { $_ -like "$aggregateId|*" } | Select-Object -First 1
                    $aggregateState = if ($aggregateLine) { ($aggregateLine -split '\|')[1] } else { '' }
                    if ($aggregateState -eq 'COMPLETED') {
                        $event = 'ISSUE|aggregate completed without RUN_COMPLETE.json'
                    } else {
                        $event = 'HEALTHY|active jobs are running, queued, or dependency-held'
                    }
                }
            }
        }
        Write-Host "`n=== SUPERVISOR EVENT STATE ==="
        Write-Host $event

        if ($event -match '^(ISSUE|COMPLETE)\|(.*)$') {
            $eventType = $Matches[1]
            $eventDetails = $Matches[2]
            $fingerprint = Get-EventHash $event
            $marker = Join-Path $EventDir "$fingerprint.triggered"
            if (-not (Test-Path -LiteralPath $marker)) {
                Set-Content -LiteralPath $marker -Value @(
                    "event=$event"
                    "detected_at=$((Get-Date).ToString('o'))"
                )
                try {
                    & $Supervisor -EventType $eventType -EventDetails $eventDetails
                    if (-not $?) {
                        throw 'Codex supervisor did not complete successfully'
                    }
                } catch {
                    Remove-Item -LiteralPath $marker -Force -ErrorAction SilentlyContinue
                    Write-Warning "Supervisor failed; event fingerprint released for retry: $_"
                }
            } else {
                Write-Host "Event already triggered; no additional Codex run." -ForegroundColor DarkYellow
            }
        }
    }
    Start-Sleep -Seconds 30
}
