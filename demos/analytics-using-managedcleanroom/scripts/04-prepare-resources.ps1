<#
.SYNOPSIS
    Provisions Azure resources for the SSE analytics scenario.

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Calls the common prepare-resources.ps1 script with storageType "blob"
    to create a storage account, Key Vault, and managed identity.

    For SSE (Server-Side Encryption), Azure Storage handles encryption at rest.
    No client-side encryption keys are needed.

    Prerequisites: Azure CLI logged in with appropriate permissions.

.PARAMETER resourceGroup
    Azure resource group for provisioning resources.

.PARAMETER location
    Azure region (default: westus).

.PARAMETER outDir
    Output directory for generated resource metadata (default: ./generated).

.PARAMETER persona
    Persona (northwind or woodgrove) for naming/logging.
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [string]$location = "westus",

    [string]$outDir = "./generated",

    [string]$persona
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

Write-Host "Preparing resources for SSE analytics scenario..." -ForegroundColor Cyan
Write-Host "Storage type: blob (SSE - server-side encryption)" -ForegroundColor Yellow

& "$PSScriptRoot/common/prepare-resources.ps1" `
    -resourceGroup $resourceGroup `
    -location $location `
    -outDir $outDir `
    -storageType blob

Write-Host "`nResource preparation complete." -ForegroundColor Green
