<#
.SYNOPSIS
    Executes an approved query in the managed cleanroom.

.DESCRIPTION
    Run by: Woodgrove (consumer).
    Submits a query run request via direct REST calls to the frontend service
    (replaces broken az managedcleanroom frontend CLI), then polls for completion
    until the query succeeds, fails, or times out.

    Prerequisites:
    - 10-vote-query.ps1 must have been run by all collaborators.
    - 07-grant-access.ps1 must have been run by all collaborators.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the approved query to execute (used as document-id).

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER pollIntervalSeconds
    Initial seconds between status polls (default: 15). Will use adaptive backoff.

.PARAMETER timeoutMinutes
    Maximum minutes to wait for completion (default: 60).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [int]$pollIntervalSeconds = 15,

    [int]$timeoutMinutes = 60,

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

# Submit the query run.
Write-Host "=== Submitting query run ===" -ForegroundColor Cyan
$runResponse = Invoke-FrontendQueryRun -Context $feCtx `
    -CollaborationId $collaborationId `
    -DocumentId $queryName `
    -TokenFile $TokenFile

Write-Host "Query run submitted." -ForegroundColor Green
$runResponse | ConvertTo-Json -Depth 5

# Extract the job ID from the response (field may be "jobId" or "id" depending on API version).
$jobId = $runResponse.jobId
if (-not $jobId) {
    $jobId = $runResponse.id
}
if (-not $jobId) {
    Write-Host "ERROR: Could not extract job ID from run response." -ForegroundColor Red
    Write-Host "Response was:" -ForegroundColor Red
    $runResponse | ConvertTo-Json -Depth 5
    exit 1
}

Write-Host "Job ID: $jobId" -ForegroundColor Yellow

# Poll for completion with adaptive backoff.
Write-Host "`n=== Polling for completion (adaptive backoff, max $timeoutMinutes minutes) ===" -ForegroundColor Cyan
$startTime = [datetime]::UtcNow
$timeout = $startTime.AddMinutes($timeoutMinutes)
$currentInterval = $pollIntervalSeconds
$pollCount = 0
$state = "UNKNOWN"

while ([datetime]::UtcNow -lt $timeout) {
    Start-Sleep -Seconds $currentInterval
    $pollCount++

    $result = Get-FrontendQueryRunResult -Context $feCtx `
        -CollaborationId $collaborationId `
        -JobId $jobId `
        -TokenFile $TokenFile

    if (-not $result) {
        Write-Host "Polling... (attempt $pollCount, waiting for result)" -ForegroundColor Yellow
        continue
    }

    # Extract state (confirmed path: status.applicationState.state).
    $state = $result.status.applicationState.state
    $elapsed = ([datetime]::UtcNow - $startTime).TotalMinutes
    $elapsedStr = "{0:F1}" -f $elapsed

    Write-Host "Poll #$pollCount - Status: $state (elapsed: ${elapsedStr}min, interval: ${currentInterval}s)" -ForegroundColor Yellow

    if ($state -eq "COMPLETED") {
        Write-Host "`nQuery execution completed successfully!" -ForegroundColor Green
        Write-Host "Total time: ${elapsedStr} minutes ($pollCount polls)" -ForegroundColor Green
        $result | ConvertTo-Json -Depth 10
        exit 0
    }
    elseif ($state -in @("FAILED", "SUBMISSION_FAILED")) {
        Write-Host "`nQuery execution failed!" -ForegroundColor Red
        Write-Host "Final state: $state" -ForegroundColor Red
        
        # Try to extract error details
        if ($result.status.errorDetails) {
            Write-Host "`nError Details:" -ForegroundColor Red
            $result.status.errorDetails | ConvertTo-Json -Depth 5
        }
        
        $result | ConvertTo-Json -Depth 10
        exit 1
    }
    
    # Adaptive backoff: 15s -> 30s -> 60s -> 120s (max)
    if ($pollCount -ge 20 -and $currentInterval -lt 120) {
        $currentInterval = 120
        Write-Host "  Switching to 120-second polling interval (query running for a while)..." -ForegroundColor Cyan
    }
    elseif ($pollCount -ge 10 -and $currentInterval -lt 60) {
        $currentInterval = 60
        Write-Host "  Switching to 60-second polling interval..." -ForegroundColor Cyan
    }
    elseif ($pollCount -ge 5 -and $currentInterval -lt 30) {
        $currentInterval = 30
        Write-Host "  Switching to 30-second polling interval..." -ForegroundColor Cyan
    }
}

Write-Host "`nQuery execution timed out after $timeoutMinutes minutes ($pollCount polls)." -ForegroundColor Red
Write-Host "Last known state: $state" -ForegroundColor Yellow
exit 1
