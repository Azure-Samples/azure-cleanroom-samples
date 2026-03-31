<#
.SYNOPSIS
    Checks the status of a specific query run by job ID.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves the current status of a query run using the job ID returned
    by 11-run-query.ps1. Can optionally poll until the job reaches a
    terminal state (COMPLETED, FAILED, SUBMISSION_FAILED).

    Prerequisites:
    - 11-run-query.ps1 must have been run (to obtain a job ID).

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER jobId
    The job ID returned by the run query API (e.g., "cl-spark-...").

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER poll
    If set, poll until the job reaches a terminal state instead of checking once.

.PARAMETER pollIntervalSeconds
    Seconds between status polls (default: 15). Uses adaptive backoff when polling.

.PARAMETER timeoutMinutes
    Maximum minutes to wait when polling (default: 30).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$jobId,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [switch]$poll,

    [int]$pollIntervalSeconds = 15,

    [int]$timeoutMinutes = 30,

    [string]$persona,

    [string]$TokenFile,

    [ValidateSet("rest", "cli")]
    [string]$ApiMode = "rest"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend helpers (supports REST and CLI modes)
. "$PSScriptRoot/common/frontend-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint -ApiMode $ApiMode

function Show-RunStatus {
    param([object]$Result)
    if (-not $Result) {
        Write-Host "No result returned (job may not exist yet)." -ForegroundColor Yellow
        return $null
    }
    $state = $Result.status.applicationState.state
    Write-Host "Job ID:  $jobId" -ForegroundColor White
    Write-Host "State:   $state" -ForegroundColor $(if ($state -eq "COMPLETED") { "Green" } elseif ($state -in @("FAILED","SUBMISSION_FAILED")) { "Red" } else { "Yellow" })
    if ($Result.status.applicationState.detail) {
        Write-Host "Detail:  $($Result.status.applicationState.detail)" -ForegroundColor DarkGray
    }
    if ($Result.status.errorDetails) {
        Write-Host "Error:   $($Result.status.errorDetails | ConvertTo-Json -Depth 5 -Compress)" -ForegroundColor Red
    }
    return $state
}

if (-not $poll) {
    # Single check
    Write-Host "=== Checking run status ===" -ForegroundColor Cyan
    $result = Get-FrontendQueryRunResult -Context $feCtx `
        -CollaborationId $collaborationId `
        -JobId $jobId `
        -TokenFile $TokenFile

    $state = Show-RunStatus -Result $result
    Write-Host "`nFull response:" -ForegroundColor Cyan
    $result | ConvertTo-Json -Depth 10
} else {
    # Poll until terminal state
    Write-Host "=== Polling job $jobId (max $timeoutMinutes min, interval ${pollIntervalSeconds}s) ===" -ForegroundColor Cyan
    $startTime = [datetime]::UtcNow
    $timeout = $startTime.AddMinutes($timeoutMinutes)
    $currentInterval = $pollIntervalSeconds
    $pollCount = 0

    while ([datetime]::UtcNow -lt $timeout) {
        $pollCount++
        $result = Get-FrontendQueryRunResult -Context $feCtx `
            -CollaborationId $collaborationId `
            -JobId $jobId `
            -TokenFile $TokenFile

        $elapsed = ([datetime]::UtcNow - $startTime).TotalMinutes
        $elapsedStr = "{0:F1}" -f $elapsed

        if (-not $result) {
            Write-Host "Poll #$pollCount - No result yet (elapsed: ${elapsedStr}min)" -ForegroundColor Yellow
        } else {
            $state = $result.status.applicationState.state
            Write-Host "Poll #$pollCount - State: $state (elapsed: ${elapsedStr}min, interval: ${currentInterval}s)" -ForegroundColor Yellow

            if ($state -eq "COMPLETED") {
                Write-Host "`nQuery execution completed successfully!" -ForegroundColor Green
                Write-Host "Total time: ${elapsedStr} minutes ($pollCount polls)" -ForegroundColor Green
                $result | ConvertTo-Json -Depth 10
                exit 0
            }
            elseif ($state -in @("FAILED", "SUBMISSION_FAILED")) {
                Write-Host "`nQuery execution failed!" -ForegroundColor Red
                Write-Host "Final state: $state" -ForegroundColor Red
                if ($result.status.errorDetails) {
                    Write-Host "`nError Details:" -ForegroundColor Red
                    $result.status.errorDetails | ConvertTo-Json -Depth 5
                }
                $result | ConvertTo-Json -Depth 10
                exit 1
            }
        }

        # Adaptive backoff: 15s -> 30s -> 60s -> 120s (max)
        if ($pollCount -ge 20 -and $currentInterval -lt 120) {
            $currentInterval = 120
            Write-Host "  Switching to 120-second polling interval..." -ForegroundColor Cyan
        }
        elseif ($pollCount -ge 10 -and $currentInterval -lt 60) {
            $currentInterval = 60
            Write-Host "  Switching to 60-second polling interval..." -ForegroundColor Cyan
        }
        elseif ($pollCount -ge 5 -and $currentInterval -lt 30) {
            $currentInterval = 30
            Write-Host "  Switching to 30-second polling interval..." -ForegroundColor Cyan
        }

        Start-Sleep -Seconds $currentInterval
    }

    Write-Host "`nTimed out after $timeoutMinutes minutes ($pollCount polls)." -ForegroundColor Red
    exit 1
}
