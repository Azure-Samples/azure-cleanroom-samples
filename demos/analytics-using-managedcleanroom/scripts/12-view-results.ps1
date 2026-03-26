<#
.SYNOPSIS
    Views query run history and audit events for the analytics collaboration.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves and displays query run history and collaboration audit events
    via direct REST calls to the frontend service (replaces broken CLI).

    Prerequisites:
    - 11-run-query.ps1 must have been run at least once.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the query to view history for (used as document-id).

.PARAMETER frontendEndpoint
    Frontend service URL.

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

    [string]$persona
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend REST helpers
. "$PSScriptRoot/common/frontend-rest-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint

# Step 1: Get run history.
Write-Host "=== Query Run History ===" -ForegroundColor Cyan
$runHistory = Get-FrontendQueryRunHistory -Context $feCtx `
    -CollaborationId $collaborationId `
    -DocumentId $queryName
if ($runHistory) {
    $runHistory | ConvertTo-Json -Depth 10
} else {
    Write-Host "No run history found." -ForegroundColor Yellow
}

# Step 2: Get audit events.
Write-Host "`n=== Audit Events ===" -ForegroundColor Cyan
$auditEvents = Get-FrontendAuditEvents -Context $feCtx `
    -CollaborationId $collaborationId
if ($auditEvents) {
    $auditEvents | ConvertTo-Json -Depth 10
} else {
    Write-Host "No audit events found." -ForegroundColor Yellow
}

Write-Host "`nResults viewing complete." -ForegroundColor Green
