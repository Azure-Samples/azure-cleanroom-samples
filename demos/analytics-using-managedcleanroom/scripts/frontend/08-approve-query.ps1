<#
.SYNOPSIS
    Step 08 [EACH COLLABORATOR] - Approve (vote on) a published query.

.DESCRIPTION
    Fetches the query to obtain its proposalId, then casts an "accept" vote.
    In single-collaborator mode Woodgrove's single vote moves the query to
    "Accepted". In multi-collaborator mode every affected collaborator must vote.

    Mirrors README-API.md Step 08 "Approve Query".

.PARAMETER Persona
    Voting collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER QueryName
    Query name to vote on, e.g. "query1-v1".

.PARAMETER VoteAction
    Vote value: accept (default) or reject.

.EXAMPLE
    ./08-approve-query.ps1 -Persona northwind -CollaborationId $collabId -QueryName "query2-v1"
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$QueryName,
    [ValidateSet("accept", "reject")][string]$VoteAction = "accept",
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

# Resolve the proposal id from the query.
$queryInfo = Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName"
$proposalId = $queryInfo.proposalId
if (-not $proposalId -and -not $DryRun) {
    # A missing proposal usually means the query is already decided, not unpublished.
    Write-Host "Query '$QueryName' has no open proposal (state='$($queryInfo.state)'); nothing to vote on."
    return
}
Write-Host "Proposal ID: $proposalId"

# Vote. (Re-voting is idempotent; a Conflict/'already voted' response is safe.)
Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName/vote" -Method POST `
    -Body @{ voteAction = $VoteAction; proposalId = $proposalId }

$state = (Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName").state
Write-Host "Vote '$VoteAction' cast. Query state: $state"
