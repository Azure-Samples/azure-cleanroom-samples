<#
.SYNOPSIS
    Step 10 [ANY] - Monitor a query run until it reaches a terminal state.

.DESCRIPTION
    Polls until COMPLETED, FAILED, SUBMISSION_FAILED, cancellation, or timeout.
    Prints new/updated events while waiting and returns the detailed result on success.
    On failure, attempts to print collaboration health issues to aid triage
    (e.g. CACI capacity shortages, executor pods stuck in init), then throws.
    A health lookup error does not replace the original terminal run failure.

    Mirrors README-API.md Step 10 "Monitor Query".

.PARAMETER Persona
    Any collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER JobId
    Run/job id returned by 09-run-query.ps1.

.PARAMETER IntervalSeconds
    Poll interval (default: 30).

.PARAMETER Once
    Returns one detailed run-result object, including events, without waiting
    or throwing for a failed run. Use this to diagnose a stuck query.

.PARAMETER TimeoutMinutes
    Maximum monitoring time (default: 60 minutes). Timing out does not cancel
    or resubmit the query.

.EXAMPLE
    ./10-monitor-query.ps1 -Persona woodgrove -CollaborationId $collabId -JobId $jobId
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$JobId,
    [ValidateRange(1, 3600)][int]$IntervalSeconds = 30,
    [ValidateRange(1, 1440)][int]$TimeoutMinutes = 60,
    [switch]$Once,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

if ($DryRun) {
    Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/runs/$JobId" | Out-Null
    Write-Host "[DRY-RUN] Read once: $Once; monitoring timeout: $TimeoutMinutes minutes."
    return
}

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
$seenEvents = [System.Collections.Generic.HashSet[string]]::new()
$terminalStates = @("COMPLETED", "FAILED", "SUBMISSION_FAILED", "CANCELLED", "CANCELED")
$timedOut = $false
do {
    $result = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/runs/$JobId"
    if ($Once) { return $result }
    $state = $result.status.applicationState.state
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] State: $state"
    foreach ($event in $result.events) {
        $key = "$($event.reason)|$($event.message)|$($event.count)|$($event.lastTimestamp)"
        if ($seenEvents.Add($key)) {
            Write-Host "  $($event.type) $($event.reason): $($event.message) (count=$($event.count))"
        }
    }
    if ($state -in $terminalStates) { break }
    if ((Get-Date) -ge $deadline) { $timedOut = $true; break }
    $remainingSeconds = [Math]::Max(0, ($deadline - (Get-Date)).TotalSeconds)
    Start-Sleep -Seconds ([Math]::Min($IntervalSeconds, $remainingSeconds))
} while ($true)

Write-Host ""
$result | ConvertTo-Json -Depth 20 | Out-Host

if ($timedOut -or $state -ne "COMPLETED") {
    Write-Warning "Monitoring stopped in state '$state'. Checking collaboration health..."
    try {
        $collabs = (Invoke-Frontend -Context $fe -Path "" -Method GET).collaborations
        $health = ($collabs | Where-Object { $_.collaborationId -eq $CollaborationId }).health
        if ($health) {
            $health | ConvertTo-Json -Depth 5 | Out-Host
        } else {
            Write-Warning "Frontend returned no health details. Use manage-collaboration.ps1 -Action Show and inspect properties.health."
        }
    } catch {
        Write-Warning "Could not retrieve collaboration health: $($_.Exception.Message)"
    }
    if ($timedOut) {
        throw "Timed out monitoring '$JobId' (state='$state'). The run was NOT cancelled. Inspect with -Once or cancel explicitly."
    }
    throw "Query run '$JobId' ended in state '$state'."
}
return $result
