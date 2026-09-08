<#
.SYNOPSIS
    Step 05.3 [EACH COLLABORATOR] - Register the OIDC issuer URL with the frontend.

.DESCRIPTION
    After the OIDC issuer documents are uploaded (scripts/06-setup-oidc-storage.ps1
    produces issuer-url.txt), this registers the issuer URL so the clean room can
    exchange its attested JWT for an Azure AD token at runtime.

    Mirrors README-API.md Step 5.3 "Register Issuer URL with Frontend".

.PARAMETER Persona
    Collaborator persona (selects the token file).

.PARAMETER CollaborationId
    Frontend collaboration UUID (from Step 03).

.PARAMETER IssuerUrl
    The OIDC issuer URL. If omitted, read from <outDir>/issuer-url.txt.

.PARAMETER outDir
    Directory containing issuer-url.txt (used when -IssuerUrl is not supplied).

.EXAMPLE
    ./05-set-issuer-url.ps1 -Persona woodgrove -CollaborationId $collabId -outDir generated/cr-e2e-woodgrove-rg
#>
param(
    [Parameter(Mandatory)][string]$Persona,
    [Parameter(Mandatory)][string]$CollaborationId,
    [string]$IssuerUrl,
    [string]$outDir,
    [string]$Frontend,
    [string]$TokenFile,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/Invoke-Frontend.ps1"

$fe = Get-FrontendContext -Persona $Persona -Frontend $Frontend -TokenFile $TokenFile -DryRun:$DryRun

if (-not $IssuerUrl -and $DryRun) { $IssuerUrl = "https://example.blob.core.windows.net/oidc" }
if (-not $IssuerUrl) {
    if (-not $outDir) { throw "Provide -IssuerUrl or -outDir (containing issuer-url.txt)." }
    $issuerFile = Join-Path $outDir "issuer-url.txt"
    if (-not (Test-Path $issuerFile)) { throw "Issuer URL file not found: $issuerFile" }
    $IssuerUrl = (Get-Content $issuerFile -Raw).Trim()
}

Invoke-Frontend -Context $fe -Path "$CollaborationId/oidc/setIssuerUrl" -Method POST -Body @{ url = $IssuerUrl }
Write-Host "Registered issuer URL: $IssuerUrl"
