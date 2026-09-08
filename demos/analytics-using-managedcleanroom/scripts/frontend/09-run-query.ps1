<#
.SYNOPSIS
    Step 09 [WOODGROVE] - Execute (run) an approved query.

.DESCRIPTION
    Triggers a run of an approved query and prints the job id. A successful
    response means the run was accepted for scheduling, not that it completed
    (use 10-monitor-query.ps1 to track completion).

    Mirrors README-API.md Step 09 "Execute Query".

.PARAMETER Persona
    Persona triggering the run (woodgrove). Selects the token file.

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER QueryName
    Approved query name, e.g. "query1-v1".

.PARAMETER StartDate
    Optional dataset date-range lower bound (e.g. "2025-09-01").

.PARAMETER EndDate
    Optional dataset date-range upper bound (e.g. "2025-09-02").

.EXAMPLE
    ./09-run-query.ps1 -Persona woodgrove -CollaborationId $collabId -QueryName "query1-v1"
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$QueryName,
    [string]$StartDate,
    [string]$EndDate,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

$runBody = @{ runId = [guid]::NewGuid().ToString() }
if ($StartDate) { $runBody.startDate = $StartDate }
if ($EndDate) { $runBody.endDate = $EndDate }

$runResult = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName/run" -Method POST -Body $runBody
$jobId = $runResult.id
Write-Host "Run submitted. Job ID: $jobId"
Write-Host "Track with: ./10-monitor-query.ps1 -Persona $Persona -CollaborationId $CollaborationId -JobId $jobId"
