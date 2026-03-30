<#
.SYNOPSIS
    Votes to accept a proposed query in the analytics collaboration.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Votes to accept the query via the frontend service.
    Supports both REST API and az managedcleanroom CLI modes via -ApiMode parameter.

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

# Step 0: Fetch the query to get its proposalId (required by the vote API).
Write-Host "=== Fetching query '$queryName' to get proposalId ===" -ForegroundColor Cyan
$queryData = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -TokenFile $TokenFile
if (-not $queryData) {
    Write-Host "ERROR: Query '$queryName' not found." -ForegroundColor Red
    exit 1
}
$proposalId = $queryData.proposalId
if (-not $proposalId) {
    Write-Host "ERROR: Query '$queryName' has no proposalId." -ForegroundColor Red
    exit 1
}
Write-Host "ProposalId: $proposalId" -ForegroundColor Yellow

# Step 1: Vote to accept the query.
Write-Host "=== Voting to accept query '$queryName' ===" -ForegroundColor Cyan

try {
    Invoke-FrontendQueryVoteAccept -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $queryName `
        -ProposalId $proposalId `
        -TokenFile $TokenFile
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
$queryInfo = Get-FrontendQuery -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -TokenFile $TokenFile
if ($queryInfo) {
    $queryInfo | ConvertTo-Json -Depth 10
}

Write-Host "`nVote complete for query '$queryName'." -ForegroundColor Green

# Step 3: Enable execution consent on the query.
# Consent requires the query to be in Accepted state (which the vote above achieves).
Write-Host "`n=== Enabling execution consent on query ===" -ForegroundColor Cyan
Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $queryName -Action "enable" -TokenFile $TokenFile
Write-Host "Execution consent enabled for query '$queryName'." -ForegroundColor Green
