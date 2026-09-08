<#
.SYNOPSIS
    Step 06.2 / 06.3 [EACH COLLABORATOR] - Publish a dataset to the collaboration.

.DESCRIPTION
    Publishes an input dataset (all collaborators) or an output dataset
    (Woodgrove only) using a pre-built dataset body JSON produced by
    scripts/08-build-dataset-body.ps1.

    Mirrors README-API.md Step 6.2 (input) and 6.3 (output).
    Execution consent is enabled by default at publish time (see -DisableConsent).

.PARAMETER Persona
    Collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER DocumentId
    Dataset document id, e.g. "woodgrove-input-csv-v1".

.PARAMETER BodyFile
    Path to the dataset body JSON (e.g. generated/publish/woodgrove-input-dataset.json).

.PARAMETER DisableConsent
    If set, disables execution consent for this document after publishing.

.EXAMPLE
    ./06-publish-dataset.ps1 -Persona woodgrove -CollaborationId $collabId `
        -DocumentId "woodgrove-input-csv-v1" -BodyFile generated/publish/woodgrove-input-dataset.json
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$DocumentId,
    [Parameter(Mandatory)][string]$BodyFile,
    [switch]$DisableConsent,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

if (-not (Test-Path $BodyFile) -and -not $DryRun) { throw "Dataset body file not found: $BodyFile" }

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

$body = if (Test-Path $BodyFile) { Get-Content $BodyFile -Raw } else { '{"_dryRun":true}' }
Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/datasets/$DocumentId/publish" -Method POST -Body $body
Write-Host "Published dataset '$DocumentId'."

if ($DisableConsent) {
    Invoke-Frontend -Context $fe -Path "$CollaborationId/consent/$DocumentId" -Method PUT -Body @{ consentAction = "disable" }
    Write-Host "Execution consent disabled for '$DocumentId'."
}
