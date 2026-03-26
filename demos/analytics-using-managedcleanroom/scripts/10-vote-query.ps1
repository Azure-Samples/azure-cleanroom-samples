<#
.SYNOPSIS
    Votes to accept a proposed query in the analytics collaboration.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Votes to accept the query via direct REST calls to the frontend service
    (replaces broken az managedcleanroom frontend CLI).

    Prerequisites:
    - 09-publish-query.ps1 must have been run.

.PARAMETER collaborationId
    The collaboration ARM resource ID.

.PARAMETER queryName
    Name of the query to vote on (used as the document-id).

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

# Step 1: Vote to accept the query.
Write-Host "=== Voting to accept query '$queryName' ===" -ForegroundColor Cyan

try {
    Invoke-FrontendQueryVoteAccept -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $queryName
    Write-Host "Vote submitted." -ForegroundColor Green
}
catch {
    # Check for "already voted/accepted/approved" errors - treat as idempotent success
    $errMsg = $_.Exception.Message
    if ($errMsg -match "already voted|already accepted|already approved|Conflict|duplicate") {
        Write-Host "Vote already submitted (skipped). Server said: $errMsg" -ForegroundColor Yellow
    } else {
        throw
    }
}

# Step 2: Verify query state.
Write-Host "`n=== Verifying query state ===" -ForegroundColor Cyan
$queryInfo = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName
if ($queryInfo) {
    $queryInfo | ConvertTo-Json -Depth 10
}

Write-Host "`nVote complete for query '$queryName'." -ForegroundColor Green
