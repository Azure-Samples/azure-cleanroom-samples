<#
.SYNOPSIS
    Grants the cleanroom workload access to the collaborator's resources.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Configures RBAC roles and federated credentials so the cleanroom workload
    can access the collaborator's storage account. For CPK, also grants
    Key Vault access (Crypto Officer + Secrets User).

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - 06-setup-identity.ps1 must have been run (OIDC issuer configured).

.PARAMETER resourceGroup
    Azure resource group containing the resources.

.PARAMETER variant
    Encryption variant: "sse" or "cpk".

.PARAMETER collaborationId
    Collaboration identifier.

.PARAMETER contractId
    Contract identifier used to compute the federation subject (default: analytics).

.PARAMETER userId
    User identifier used to compute the federation subject.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("sse", "cpk")]
    [string]$variant,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [string]$contractId = "analytics",

    [Parameter(Mandatory)]
    [string]$userId,

    [string]$outDir = "./generated"
)

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

# Call common setup-access.ps1. CPK needs Key Vault RBAC; SSE does not.
$setupKv = ($variant -eq "cpk")
Write-Host "`n=== Setting up access ($variant) ===" -ForegroundColor Cyan
& "$PSScriptRoot/common/setup-access.ps1" `
    -resourceGroup $resourceGroup `
    -collaborationId $collaborationId `
    -subject $subject `
    -issuerUrl $issuerUrl `
    -outDir $outDir `
    -setupKeyVault:$setupKv

Write-Host "`nAccess granted for subject '$subject'." -ForegroundColor Green
