<#
.SYNOPSIS
    Provisions Azure resources for an analytics scenario (SSE or CPK).

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).
    Calls the common prepare-resources.ps1 script to create a resource group,
    storage account, managed identity and RBAC assignments.

    - SSE: Skips Key Vault (Azure handles encryption at rest).
    - CPK: Creates a Key Vault (Premium SKU) for encryption keys.

.PARAMETER resourceGroup
    Azure resource group for provisioning resources.

.PARAMETER variant
    Encryption variant: "sse" (server-side) or "cpk" (customer-provided key).

.PARAMETER location
    Azure region (default: westus).

.PARAMETER outDir
    Output directory for generated resource metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("sse", "cpk")]
    [string]$variant,

    [string]$location = "westus",

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$label = if ($variant -eq "sse") { "SSE — server-side encryption" } else { "CPK — customer-provided key" }
Write-Host "Preparing resources for $label scenario..." -ForegroundColor Cyan

$skipKv = ($variant -eq "sse")

& "$PSScriptRoot/common/prepare-resources.ps1" `
    -resourceGroup $resourceGroup `
    -location $location `
    -outDir $outDir `
    -storageType blob `
    -skipKeyVault:$skipKv

Write-Host "`nResource preparation complete." -ForegroundColor Green
