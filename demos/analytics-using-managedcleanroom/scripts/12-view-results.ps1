<#
.SYNOPSIS
    Views query run history and audit events for the analytics collaboration.

.DESCRIPTION
    Run by: Any collaborator (Northwind or Woodgrove).
    Retrieves and displays query run history and collaboration audit events
    via the managed cleanroom frontend CLI.

    Prerequisites:
    - 11-run-query.ps1 must have been run at least once.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the query to view history for (used as document-id).
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$queryName
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Step 1: Get run history.
Write-Host "=== Query Run History ===" -ForegroundColor Cyan
az managedcleanroom frontend analytics query runhistory list `
    --collaboration-id $collaborationId `
    --document-id $queryName

# Step 2: Get audit events.
Write-Host "`n=== Audit Events ===" -ForegroundColor Cyan
az managedcleanroom frontend analytics auditevent list `
    --collaboration-id $collaborationId

Write-Host "`nResults viewing complete." -ForegroundColor Green
