<#
.SYNOPSIS
    Step 07.2 [WOODGROVE] - Publish (propose) a query to the collaboration.

.DESCRIPTION
    Publishes a query using a pre-built query body JSON produced by
    scripts/09-build-query-body.ps1. Publishing proposes the query for approval
    (Step 08).

    Mirrors README-API.md Step 7.2 "Publish Query".

.PARAMETER Persona
    Persona of the query owner (woodgrove). Selects the token file.

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER QueryName
    Query name, e.g. "query1-v1".

.PARAMETER BodyFile
    Path to the query body JSON (e.g. generated/publish/query1-v1.json).

.EXAMPLE
    ./07-publish-query.ps1 -Persona woodgrove -CollaborationId $collabId `
        -QueryName "query1-v1" -BodyFile generated/publish/query1-v1.json
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$QueryName,
    [Parameter(Mandatory)][string]$BodyFile,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

if (-not (Test-Path $BodyFile) -and -not $DryRun) { throw "Query body file not found: $BodyFile" }

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

$body = if (Test-Path $BodyFile) { Get-Content $BodyFile -Raw } else { '{"_dryRun":true}' }
Invoke-Frontend -Context $fe -Path "$CollaborationId/analytics/queries/$QueryName/publish" -Method POST -Body $body
Write-Host "Published query '$QueryName'. Awaiting approval (Step 08)."
