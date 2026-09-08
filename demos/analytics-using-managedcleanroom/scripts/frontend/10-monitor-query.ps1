<#
.SYNOPSIS
    Step 10 [ANY] - Monitor a query run until it reaches a terminal state.

.DESCRIPTION
    Polls the run status until COMPLETED, FAILED, or SUBMISSION_FAILED.
    On a non-Ok collaboration, also prints health issues to aid triage
    (e.g. CACI capacity shortages, executor pods stuck in init).

    Mirrors README-API.md Step 10 "Monitor Query".

.PARAMETER Persona
    Any collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER JobId
    Run/job id returned by 09-run-query.ps1.

.PARAMETER IntervalSeconds
    Poll interval (default: 30).

.EXAMPLE
    ./10-monitor-query.ps1 -Persona woodgrove -CollaborationId $collabId -JobId $jobId
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$JobId,
    [int]$IntervalSeconds = 30,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

if ($DryRun) {
    Write-Host "[DRY-RUN] Would poll run status for job $JobId until COMPLETED/FAILED/SUBMISSION_FAILED."
    return
}

do {
    $result = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/runs/$JobId"
    $state = $result.status.applicationState.state
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] State: $state"
    if ($state -notin @("COMPLETED", "FAILED", "SUBMISSION_FAILED")) { Start-Sleep -Seconds $IntervalSeconds }
} while ($state -notin @("COMPLETED", "FAILED", "SUBMISSION_FAILED"))

Write-Host ""
$result | ConvertTo-Json -Depth 10

if ($state -ne "COMPLETED") {
    Write-Warning "Run ended in state '$state'. Checking collaboration health for pod-level/capacity issues..."
    $collabs = (Invoke-Frontend -Context $fe -Path "" -Method GET).collaborations
    $health = ($collabs | Where-Object { $_.collaborationId -eq $CollaborationId }).health
    if ($health) { $health | ConvertTo-Json -Depth 5 }
}
