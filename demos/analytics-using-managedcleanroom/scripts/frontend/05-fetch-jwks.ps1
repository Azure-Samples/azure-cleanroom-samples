<#
.SYNOPSIS
    Step 05.1 [EACH COLLABORATOR] - Fetch the collaboration OIDC JWKS.

.DESCRIPTION
    Downloads the collaboration's JWKS (public signing keys) from the frontend
    and writes it to <outDir>/jwks.json for the OIDC storage setup
    (scripts/06-setup-oidc-storage.ps1).

    Mirrors README-API.md Step 5.1 "Fetch JWKS from Frontend".

.PARAMETER Persona
    Collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER outDir
    Output directory for jwks.json (default: ./generated/<personaRg>). If a bare
    directory is given it is created if missing.

.EXAMPLE
    ./05-fetch-jwks.ps1 -Persona woodgrove -CollaborationId $collabId -outDir generated/cr-e2e-woodgrove-rg
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [Parameter(Mandatory)][string]$outDir,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

New-Item -ItemType Directory -Path $outDir -Force | Out-Null
$jwks = Invoke-Frontend -Context $fe -Path "$CollaborationId/oidc/keys" -Method GET
if ($DryRun) { Write-Host "[DRY-RUN] Would write JWKS to $(Join-Path $outDir 'jwks.json')"; return }
$jwksPath = Join-Path $outDir "jwks.json"
$jwks | ConvertTo-Json -Depth 10 | Out-File $jwksPath -Encoding utf8

Write-Host "JWKS written to $jwksPath"
