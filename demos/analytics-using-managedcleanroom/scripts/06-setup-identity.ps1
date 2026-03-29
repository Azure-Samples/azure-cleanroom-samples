<#
.SYNOPSIS
    Sets up identity and OIDC issuer WITHOUT az cleanroom CLI.

.DESCRIPTION
    Replaces: az cleanroom collaboration context set + az cleanroom collaboration identity add az-federated
    Uses: az identity show (standard Azure CLI)

    The OIDC issuer setup (setup-oidc-issuer.ps1) already uses only standard CLI.
    The only `az cleanroom` commands in the original 06-setup-identity.ps1 were:
      1. `az cleanroom collaboration context set` - sets a local context file (we skip this)
      2. `az cleanroom collaboration identity add az-federated` - registers identity to local config

    In this replacement, we save the identity metadata to a JSON file that
    the dataset publish script reads when building the DatasetSpecification.
    The identity registration with the collaboration happens implicitly when
    the DatasetSpecification (which embeds the identity) is published to the
    frontend API.

.PARAMETER resourceGroup
    Azure resource group containing the managed identity.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER collaborationName
    Name of the collaboration.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$outDir = "./generated",

    [string]$TokenFile
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load generated resource names.
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

# Step 1: Set up OIDC issuer (this script already uses standard Azure CLI only).
Write-Host "=== Step 1: Setting up OIDC issuer ===" -ForegroundColor Cyan
& "$PSScriptRoot/common/setup-oidc-issuer.ps1" `
    -resourceGroup $resourceGroup `
    -collaborationId $collaborationId `
    -frontendEndpoint $frontendEndpoint `
    -outDir $outDir `
    -TokenFile $TokenFile

# Step 2: Read issuer URL.
Write-Host "`n=== Step 2: Reading issuer URL ===" -ForegroundColor Cyan
$issuerUrlFile = Join-Path $outDir $resourceGroup "issuer-url.txt"
if (-not (Test-Path $issuerUrlFile)) {
    Write-Host "ERROR: '$issuerUrlFile' not found. OIDC issuer setup may have failed." -ForegroundColor Red
    exit 1
}
$issuerUrl = (Get-Content $issuerUrlFile -Raw).Trim()
Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow

# Step 3: Get managed identity properties.
Write-Host "`n=== Step 3: Retrieving managed identity properties ===" -ForegroundColor Cyan
$identityJson = az identity show `
    --name $MANAGED_IDENTITY_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$clientId = $identityJson.clientId
$tenantId = $identityJson.tenantId

Write-Host "Client ID: $clientId"
Write-Host "Tenant ID: $tenantId"

# Step 4: Save identity metadata for downstream scripts.
# This replaces `az cleanroom collaboration context set` + `identity add az-federated`.
# The identity info is embedded into the DatasetSpecification JSON when publishing.
Write-Host "`n=== Step 4: Saving identity metadata ===" -ForegroundColor Cyan
$identityMetadata = @{
    identityName    = "$persona-identity"
    clientId        = $clientId
    tenantId        = $tenantId
    tokenIssuerUrl  = $issuerUrl
    backingIdentity = "cleanroom_cgs_oidc"
}

$outputDir = Join-Path $outDir $resourceGroup
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}
$identityFile = Join-Path $outputDir "identity-metadata.json"
$identityMetadata | ConvertTo-Json -Depth 5 | Out-File -FilePath $identityFile -Encoding utf8
Write-Host "Identity metadata saved to: $identityFile" -ForegroundColor Yellow

Write-Host "`nIdentity setup complete for '$persona' (no az cleanroom CLI used)." -ForegroundColor Green
