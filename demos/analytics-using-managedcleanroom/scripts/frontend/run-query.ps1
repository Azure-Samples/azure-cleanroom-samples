<#
.SYNOPSIS
    Query orchestrator [OWNER / WOODGROVE] - run a query end to end:
    (optional) publish -> run -> monitor -> results.

.DESCRIPTION
    A single entry point that makes a query "easily runnable by scripts" once the
    collaboration is set up. Chains the per-step scripts (07, 10, 11) and runs the
    query inline to capture the job id.

    Steps performed (each skippable):
      1. Resolve the collaboration UUID.
      2. (optional) Publish the query (07) when -BodyFile is supplied.
         NOTE: after publishing, every affected collaborator must approve it
         (run-collaborator.ps1 / 08-approve-query.ps1) before it can run.
      3. Run the query (capturing the job id).
      4. (optional) Monitor to a terminal state (10).
      5. (optional) Print run history + audit events (11).

    Output download (README Step 11.3) is left to ../11-download-output.ps1
    because it needs the per-persona resource group.

.PARAMETER Persona
    Owner persona (selects the token file), e.g. woodgrove.

.PARAMETER QueryName
    Query name to run, e.g. "query1-v1".

.PARAMETER CollaborationId
    Frontend collaboration UUID. If omitted, resolved via -CollaborationName or
    the first visible collaboration.

.PARAMETER CollaborationName
    Resolve the collaboration by display name when -CollaborationId is not given.

.PARAMETER BodyFile
    Optional query body JSON to publish before running (invokes 07).

.PARAMETER StartDate
    Optional dataset date-range lower bound (e.g. "2025-09-01").

.PARAMETER EndDate
    Optional dataset date-range upper bound (e.g. "2025-09-02").

.PARAMETER SkipMonitor
    Skip polling for completion (returns immediately after submit).

.PARAMETER SkipResults
    Skip printing run history + audit events.

.PARAMETER Frontend
    Optional frontend base URL override.

.PARAMETER TokenFile
    Optional per-persona token file override.

.PARAMETER DryRun
    Print planned requests without calling the service.

.EXAMPLE
    # Run an approved query and wait for results
    ./run-query.ps1 -Persona woodgrove -QueryName query1-v1

.EXAMPLE
    # Publish then run (remember collaborators must approve between the two)
    ./run-query.ps1 -Persona woodgrove -QueryName query1-v1 `
        -BodyFile generated/publish/query1-v1.json
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$QueryName,
    [string]$CollaborationId,
    [string]$CollaborationName,
    [string]$BodyFile,
    [string]$StartDate,
    [string]$EndDate,
    [switch]$SkipMonitor,
    [switch]$SkipResults,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

# Shared args forwarded to child step scripts.
$childArgs = @{ Persona = $Persona }
if ($Frontend) { $childArgs.Frontend = $Frontend }
if ($TokenFile) { $childArgs.TokenFile = $TokenFile }
if ($DryRun) { $childArgs.DryRun = $true }

# 1. Resolve the collaboration UUID.
$CollaborationId = Resolve-CollaborationId -Context $fe -CollaborationId $CollaborationId -CollaborationName $CollaborationName
Write-Host "==> Collaboration: $CollaborationId"

# 2. Publish the query (optional).
if ($BodyFile) {
    Write-Host "==> Publishing query '$QueryName'..."
    & "$PSScriptRoot/07-publish-query.ps1" @childArgs -CollaborationId $CollaborationId `
        -QueryName $QueryName -BodyFile $BodyFile
    Write-Host "    (collaborators must now approve via run-collaborator.ps1 / 08-approve-query.ps1)"
}

# 3. Run the query (capture the job id).
Write-Host "==> Running query '$QueryName'..."
$runBody = @{ runId = [guid]::NewGuid().ToString() }
if ($StartDate) { $runBody.startDate = $StartDate }
if ($EndDate) { $runBody.endDate = $EndDate }
$runResult = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName/run" -Method POST -Body $runBody
$jobId = if ($DryRun) { "<job-id>" } else { $runResult.id }
Write-Host "    Job ID: $jobId"

# 4. Monitor (optional).
if (-not $SkipMonitor) {
    Write-Host "==> Monitoring run $jobId..."
    & "$PSScriptRoot/10-monitor-query.ps1" @childArgs -CollaborationId $CollaborationId -JobId $jobId
}

# 5. Results + audit (optional).
if (-not $SkipResults) {
    Write-Host "==> Fetching run history + audit events..."
    & "$PSScriptRoot/11-results-audit.ps1" @childArgs -CollaborationId $CollaborationId -QueryName $QueryName
}

Write-Host ""
Write-Host "Query workflow complete for '$QueryName' (job $jobId) on collaboration $CollaborationId."
Write-Host "Download output with: ../11-download-output.ps1 -resourceGroup <rg> -JobId $jobId"
