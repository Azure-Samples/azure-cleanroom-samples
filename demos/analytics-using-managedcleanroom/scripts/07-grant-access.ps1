<#
.SYNOPSIS
    Grants the cleanroom workload access to the collaborator's resources.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Configures RBAC roles and federated credentials so the cleanroom workload
    can access the collaborator's storage account. For SSE, Key Vault access
    is not required (no client-side encryption keys).

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - 06-setup-identity.ps1 must have been run (OIDC issuer configured).

.PARAMETER resourceGroup
    Azure resource group containing the resources.

.PARAMETER collaborationName
    Name of the collaboration resource.

.PARAMETER contractId
    Contract identifier used to compute the federation subject (default: analytics).

.PARAMETER userId
    User identifier used to compute the federation subject.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [string]$contractId = "analytics",

    [Parameter(Mandatory)]
    [string]$userId,

    [string]$outDir = "./generated",

    [string]$persona
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Read issuer URL.
$issuerUrlFile = Join-Path $outDir $resourceGroup "issuer-url.txt"
if (-not (Test-Path $issuerUrlFile)) {
    Write-Host "ERROR: '$issuerUrlFile' not found. Run 06-setup-identity.ps1 first." -ForegroundColor Red
    exit 1
}
$issuerUrl = (Get-Content $issuerUrlFile -Raw).Trim()

# Compute federation subject.
$subject = "$contractId-$userId"
Write-Host "Granting access for subject: $subject" -ForegroundColor Cyan
Write-Host "Issuer URL: $issuerUrl" -ForegroundColor Yellow

# Call common setup-access.ps1 without Key Vault setup (SSE doesn't need KV for the cleanroom).
Write-Host "`n=== Setting up access ===" -ForegroundColor Cyan
& "$PSScriptRoot/common/setup-access.ps1" `
    -resourceGroup $resourceGroup `
    -collaborationId $collaborationId `
    -subject $subject `
    -issuerUrl $issuerUrl `
    -outDir $outDir `
    -setupKeyVault:$false

Write-Host "`nAccess granted for subject '$subject'." -ForegroundColor Green
