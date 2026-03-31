<#
.SYNOPSIS
    Retrieves app credentials from Azure Key Vault.

.DESCRIPTION
    Helper script to retrieve service principal credentials stored in Key Vault.
    Supports certificate-based authentication.
    Returns appId, certPath, tenantId, and spObjectId for the specified persona.

.PARAMETER keyVaultName
    Name of the Key Vault containing app credentials.

.PARAMETER persona
    Persona (northwind or woodgrove).

.PARAMETER downloadDir
    Directory to download certificate to (default: ./generated/certs).

.OUTPUTS
    PSCustomObject with appId, certPath, tenantId, spObjectId properties.
#>
param(
    [Parameter(Mandatory)]
    [string]$keyVaultName,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [string]$downloadDir = "./generated/certs"
)

$ErrorActionPreference = 'Stop'

try {
    # Retrieve app ID
    $appId = az keyvault secret show `
        --vault-name $keyVaultName `
        --name "$persona-app-id" `
        --query value -o tsv

    # Retrieve certificate name
    $certName = az keyvault secret show `
        --vault-name $keyVaultName `
        --name "$persona-cert-name" `
        --query value -o tsv

    # Retrieve tenant ID
    $tenantId = az keyvault secret show `
        --vault-name $keyVaultName `
        --name "$persona-tenant-id" `
        --query value -o tsv

    # Retrieve service principal object ID
    $spObjectId = az keyvault secret show `
        --vault-name $keyVaultName `
        --name "$persona-sp-object-id" `
        --query value -o tsv

    if (-not $appId -or -not $certName -or -not $tenantId -or -not $spObjectId) {
        throw "One or more credentials not found in Key Vault '$keyVaultName' for persona '$persona'"
    }

    # Create download directory if it doesn't exist
    if (-not (Test-Path $downloadDir)) {
        New-Item -ItemType Directory -Path $downloadDir -Force | Out-Null
    }

    # Download certificate from Key Vault (PFX format with private key)
    $certPath = Join-Path $downloadDir "$persona-auth.pfx"
    
    Write-Host "Downloading certificate '$certName' from Key Vault..." -ForegroundColor Cyan
    az keyvault secret download `
        --vault-name $keyVaultName `
        --name $certName `
        --file $certPath `
        --encoding base64 | Out-Null

    if (-not (Test-Path $certPath)) {
        throw "Failed to download certificate to '$certPath'"
    }

    return [PSCustomObject]@{
        appId       = $appId
        certPath    = $certPath
        certName    = $certName
        tenantId    = $tenantId
        spObjectId  = $spObjectId
    }
}
catch {
    Write-Host "ERROR: Failed to retrieve app credentials from Key Vault." -ForegroundColor Red
    Write-Host "  Key Vault: $keyVaultName" -ForegroundColor Red
    Write-Host "  Persona: $persona" -ForegroundColor Red
    Write-Host "  Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
