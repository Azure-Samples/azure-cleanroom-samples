<#
.SYNOPSIS
    Step 11.1 / 11.2 [WOODGROVE] - Retrieve run history and audit events.

.DESCRIPTION
    Prints the run history for a query (rows read/written, duration) and the
    collaboration audit events. Output download is handled separately by
    scripts/11-download-output.ps1.

    Mirrors README-API.md Step 11.1 "Run History" and 11.2 "Audit Events".

.PARAMETER Persona
    Persona (woodgrove). Selects the token file.

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER QueryName
    Query whose run history to fetch, e.g. "query1-v1".

.PARAMETER SkipAudit
    If set, skips the audit-events call.

.EXAMPLE
    ./11-results-audit.ps1 -Persona woodgrove -CollaborationId $collabId -QueryName "query1-v1"
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$QueryName,
    [switch]$SkipAudit,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

Write-Host "=== Run history for '$QueryName' ==="
$history = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName/runs"
$history | ConvertTo-Json -Depth 10

if (-not $SkipAudit) {
    Write-Host ""
    Write-Host "=== Audit events ==="
    $audit = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/auditevents"
    $audit | ConvertTo-Json -Depth 10
}
