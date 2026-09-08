<#
.SYNOPSIS
    Collaborator orchestrator [EACH COLLABORATOR] - runs the collaborator-side
    frontend workflow end to end: resolve collaboration -> accept invite ->
    (optional) publish dataset -> vote on an already-published query.

.DESCRIPTION
    A single entry point that chains the per-step scripts (03, 06, 08) for one
    persona. Intended for the common case where the query has ALREADY been
    published by the owner (Woodgrove) and each collaborator just needs to join
    and approve it.

    Steps performed (each skippable):
      1. Resolve the collaboration UUID (by -CollaborationId, -CollaborationName,
         or first visible collaboration).
      2. Accept the pending invitation (03). Tolerates "already accepted".
      3. (optional) Publish a dataset (06) when -DatasetDocumentId + -DatasetBodyFile
         are supplied.
      4. Vote on the published query (08).

    Resource provisioning and OIDC setup (README Steps 04-05: prepare-resources,
    setup-oidc-storage, grant-access, plus JWKS/issuer via 05-fetch-jwks.ps1 /
    05-set-issuer-url.ps1) are prerequisites and are intentionally NOT run here,
    because they require per-persona resource groups / storage. Run those first
    if the collaborator has not been onboarded yet.

.PARAMETER Persona
    Collaborator persona (selects the token file), e.g. northwind.

.PARAMETER QueryName
    The already-published query to vote on, e.g. "query1-v1".

.PARAMETER CollaborationId
    Frontend collaboration UUID. If omitted, resolved via -CollaborationName or
    the first visible collaboration.

.PARAMETER CollaborationName
    Resolve the collaboration by display name when -CollaborationId is not given.

.PARAMETER VoteAction
    Vote value: accept (default) or reject.

.PARAMETER DatasetDocumentId
    Optional dataset document id to publish (e.g. "northwind-input-csv-v1").

.PARAMETER DatasetBodyFile
    Optional dataset body JSON path (required if -DatasetDocumentId is set).

.PARAMETER SkipAccept
    Skip the accept-invitation step (use if already accepted).

.PARAMETER Frontend
    Optional frontend base URL override.

.PARAMETER TokenFile
    Optional per-persona token file override.

.PARAMETER DryRun
    Print planned requests without calling the service.

.EXAMPLE
    # Northwind joins and approves an already-published query
    ./run-collaborator.ps1 -Persona northwind -QueryName query1-v1

.EXAMPLE
    # Also publish a dataset before voting
    ./run-collaborator.ps1 -Persona northwind -QueryName query2-v1 `
        -DatasetDocumentId northwind-input-csv-v1 `
        -DatasetBodyFile generated/publish/northwind-input-dataset.json
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$QueryName,
    [string]$CollaborationId,
    [string]$CollaborationName,
    [ValidateSet("accept", "reject")][string]$VoteAction = "accept",
    [string]$DatasetDocumentId,
    [string]$DatasetBodyFile,
    [switch]$SkipAccept,
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

# ---------------------------------------------------------------------------
# 1. Resolve the collaboration UUID.
# ---------------------------------------------------------------------------
$CollaborationId = Resolve-CollaborationId -Context $fe -CollaborationId $CollaborationId -CollaborationName $CollaborationName
Write-Host "==> Collaboration: $CollaborationId"

# ---------------------------------------------------------------------------
# 2. Accept the invitation (tolerate already-accepted).
# ---------------------------------------------------------------------------
if (-not $SkipAccept) {
    Write-Host "==> Accepting invitation..."
    try {
        & "$PSScriptRoot/03-accept-invitation.ps1" @childArgs -CollaborationId $CollaborationId
    }
    catch {
        Write-Warning "Accept step skipped/failed (may already be accepted): $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------
# 3. Publish a dataset (optional).
# ---------------------------------------------------------------------------
if ($DatasetDocumentId) {
    if (-not $DatasetBodyFile) { throw "-DatasetBodyFile is required when -DatasetDocumentId is set." }
    Write-Host "==> Publishing dataset '$DatasetDocumentId'..."
    & "$PSScriptRoot/06-publish-dataset.ps1" @childArgs -CollaborationId $CollaborationId `
        -DocumentId $DatasetDocumentId -BodyFile $DatasetBodyFile
}

# ---------------------------------------------------------------------------
# 4. Vote on the already-published query.
# ---------------------------------------------------------------------------
Write-Host "==> Voting '$VoteAction' on query '$QueryName'..."
& "$PSScriptRoot/08-approve-query.ps1" @childArgs -CollaborationId $CollaborationId `
    -QueryName $QueryName -VoteAction $VoteAction

Write-Host ""
Write-Host "Collaborator workflow complete for persona '$Persona' on collaboration $CollaborationId."
