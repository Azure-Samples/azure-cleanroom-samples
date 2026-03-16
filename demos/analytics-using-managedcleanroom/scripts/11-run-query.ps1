<#
.SYNOPSIS
    Executes an approved query in the managed cleanroom.

.DESCRIPTION
    Run by: Woodgrove (consumer).
    Submits a query run request via the managed cleanroom frontend CLI, then polls
    for completion until the query succeeds, fails, or times out.

    Prerequisites:
    - 10-vote-query.ps1 must have been run by all collaborators.
    - 07-grant-access.ps1 must have been run by all collaborators.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the approved query to execute (used as document-id).

.PARAMETER pollIntervalSeconds
    Seconds between status polls (default: 15).

.PARAMETER timeoutMinutes
    Maximum minutes to wait for completion (default: 30).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName,

    [int]$pollIntervalSeconds = 15,

    [int]$timeoutMinutes = 30
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Submit the query run.
Write-Host "=== Submitting query run ===" -ForegroundColor Cyan
$runResponse = az managedcleanroom frontend analytics query run `
    --collaboration-id $collaborationId `
    --document-id $queryName | ConvertFrom-Json

Write-Host "Query run submitted." -ForegroundColor Green
$runResponse | ConvertTo-Json -Depth 5

# Extract the job ID from the response (confirmed field: jobId).
$jobId = $runResponse.jobId
if (-not $jobId) {
    Write-Host "ERROR: Could not extract job ID from run response." -ForegroundColor Red
    exit 1
}

Write-Host "Job ID: $jobId" -ForegroundColor Yellow

# Poll for completion.
Write-Host "`n=== Polling for completion ===" -ForegroundColor Cyan
$timeout = [datetime]::UtcNow.AddMinutes($timeoutMinutes)

while ([datetime]::UtcNow -lt $timeout) {
    Start-Sleep -Seconds $pollIntervalSeconds

    try {
        $result = az managedcleanroom frontend analytics query runresult show `
            --collaboration-id $collaborationId `
            --job-id $jobId | ConvertFrom-Json
    }
    catch {
        Write-Host "Polling... (waiting for result)" -ForegroundColor Yellow
        continue
    }

    # Extract state (confirmed path: status.applicationState.state).
    $state = $result.status.applicationState.state

    Write-Host "Status: $state" -ForegroundColor Yellow

    if ($state -eq "COMPLETED") {
        Write-Host "`nQuery execution completed successfully!" -ForegroundColor Green
        $result | ConvertTo-Json -Depth 10
        return
    }
    elseif ($state -in @("FAILED", "SUBMISSION_FAILED")) {
        Write-Host "`nQuery execution failed!" -ForegroundColor Red
        $result | ConvertTo-Json -Depth 10
        exit 1
    }
}

Write-Host "Query execution timed out after $timeoutMinutes minutes." -ForegroundColor Red
exit 1
