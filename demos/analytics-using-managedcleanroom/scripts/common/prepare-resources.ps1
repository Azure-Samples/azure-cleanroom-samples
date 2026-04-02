###############################################################################################
# prepare-resources.ps1
#
# Provisions Azure resources needed for a managed cleanroom big-data analytics sample.
# Creates a resource group, storage account, Key Vault (premium), and managed identity,
# then assigns the required RBAC roles to the logged-in user. Outputs resource metadata
# to generated files for use by downstream scripts.
###############################################################################################

param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [string]$location = "westus",

    [ValidateSet("akvpremium")]
    [string]$kvType = "akvpremium",

    [ValidateSet("blob", "adlsgen2")]
    [string]$storageType = "blob",

    [switch]$skipKeyVault,

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

. "$PSScriptRoot/utils.ps1"

# Derive deterministic, globally-unique resource names from resource group + subscription.
$subscriptionId = az account show --query id -o tsv
$hash = Get-ResourceNameHash -seed "$subscriptionId-$resourceGroup"
$storageAccountName = ("sa" + $hash).Substring(0, 24).ToLower() -replace '[^a-z0-9]', ''
$keyVaultName = ("kv-" + $hash).Substring(0, 24)
$managedIdentityName = ("id-" + $hash).Substring(0, 24)

$outputDir = Join-Path $outDir $resourceGroup
if (-not (Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
}

# -- Resource Group --------------------------------------------------------------
Write-Host "Checking resource group '$resourceGroup'..." -ForegroundColor Cyan
$rgExists = az group exists --name $resourceGroup 2>$null
if ($rgExists -eq "false") {
    Write-Host "Creating resource group '$resourceGroup' in '$location'..." -ForegroundColor Yellow
    az group create --name $resourceGroup --location $location --output none
}
Write-Host "Resource group '$resourceGroup' is ready." -ForegroundColor Green

# -- Storage Account -------------------------------------------------------------
Write-Host "Creating storage account '$storageAccountName'..." -ForegroundColor Cyan
$storageArgs = @(
    "storage", "account", "create",
    "--name", $storageAccountName,
    "--resource-group", $resourceGroup,
    "--location", $location,
    "--sku", "Standard_LRS",
    "--output", "json"
)
if ($storageType -eq "adlsgen2") {
    $storageArgs += "--hns"
    $storageArgs += "true"
}
$storageJson = az @storageArgs | ConvertFrom-Json
$storageId = $storageJson.id
Write-Host "Storage account '$storageAccountName' created." -ForegroundColor Green

# -- Key Vault -------------------------------------------------------------------
if ($skipKeyVault) {
    Write-Host "Skipping Key Vault creation (not required for SSE)." -ForegroundColor Yellow
    $kvId = $null
    $keyVaultName = $null
} else {
Write-Host "Creating Key Vault '$keyVaultName' (premium, RBAC)..." -ForegroundColor Cyan
$existingKv = Invoke-AzSafe @("keyvault", "show", "--name", $keyVaultName, "--resource-group", $resourceGroup, "--output", "json")
if ($existingKv) {
    $kvJson = $existingKv | ConvertFrom-Json
    Write-Host "Key Vault '$keyVaultName' already exists." -ForegroundColor Green
} else {
    $deletedKv = Invoke-AzSafe @("keyvault", "show-deleted", "--name", $keyVaultName)
    if ($deletedKv) {
        Write-Host "Recovering soft-deleted Key Vault '$keyVaultName'..." -ForegroundColor Yellow
        $kvJson = az keyvault recover --name $keyVaultName --output json | ConvertFrom-Json
    } else {
        $kvJson = az keyvault create `
            --name $keyVaultName `
            --resource-group $resourceGroup `
            --location $location `
            --sku premium `
            --enable-rbac-authorization true `
            --output json | ConvertFrom-Json
    }
    Write-Host "Key Vault '$keyVaultName' ready." -ForegroundColor Green
}
$kvId = $kvJson.id
}

# -- Managed Identity ------------------------------------------------------------
Write-Host "Creating managed identity '$managedIdentityName'..." -ForegroundColor Cyan
$existingId = Invoke-AzSafe @("identity", "show", "--name", $managedIdentityName, "--resource-group", $resourceGroup, "--output", "json")
if ($existingId) {
    $idJson = $existingId | ConvertFrom-Json
    Write-Host "Managed identity '$managedIdentityName' already exists." -ForegroundColor Green
} else {
    $idJson = az identity create `
        --name $managedIdentityName `
        --resource-group $resourceGroup `
        --location $location `
        --output json | ConvertFrom-Json
    Write-Host "Managed identity '$managedIdentityName' created." -ForegroundColor Green
}

# -- RBAC Role Assignments ------------------------------------------------------
Write-Host "Assigning RBAC roles to the logged-in user..." -ForegroundColor Cyan
$callerObjectId = az ad signed-in-user show --query id -o tsv

# Key Vault roles (skip if already assigned or if Key Vault was not created)
if (-not $skipKeyVault) {
    Invoke-AzSafe @("role", "assignment", "create", "--role", "Key Vault Crypto Officer", "--assignee-object-id", $callerObjectId, "--assignee-principal-type", "User", "--scope", $kvId, "--output", "none")
    Invoke-AzSafe @("role", "assignment", "create", "--role", "Key Vault Secrets Officer", "--assignee-object-id", $callerObjectId, "--assignee-principal-type", "User", "--scope", $kvId, "--output", "none")
}

# Storage role (skip if already assigned)
Invoke-AzSafe @("role", "assignment", "create", "--role", "Storage Blob Data Contributor", "--assignee-object-id", $callerObjectId, "--assignee-principal-type", "User", "--scope", $storageId, "--output", "none")

Write-Host "RBAC role assignments completed." -ForegroundColor Green

# -- Output Files ----------------------------------------------------------------
$resourcesJson = @{
    storageAccount  = @{
        id   = $storageId
        name = $storageAccountName
    }
    keyVault        = @{
        id   = $kvId
        name = $keyVaultName
    }
    managedIdentity = @{
        id          = $idJson.id
        name        = $managedIdentityName
        clientId    = $idJson.clientId
        tenantId    = $idJson.tenantId
        principalId = $idJson.principalId
    }
} | ConvertTo-Json -Depth 4

$resourcesJsonPath = Join-Path $outputDir "resources.generated.json"
$resourcesJson | Out-File -FilePath $resourcesJsonPath -Encoding utf8
Write-Host "Wrote resource metadata to '$resourcesJsonPath'." -ForegroundColor Green

$namesPs1Content = @"
`$STORAGE_ACCOUNT_NAME = "$storageAccountName"
`$KEYVAULT_NAME = "$keyVaultName"
`$MANAGED_IDENTITY_NAME = "$managedIdentityName"
"@
$namesPs1Path = Join-Path $outputDir "names.generated.ps1"
$namesPs1Content | Out-File -FilePath $namesPs1Path -Encoding utf8
Write-Host "Wrote variable assignments to '$namesPs1Path'." -ForegroundColor Green

Write-Host "Resource provisioning complete." -ForegroundColor Green
