###############################################################################################
# setup-access.ps1
#
# Configures access and workload identity federation for a managed cleanroom collaboration.
# Assigns the managed identity the required RBAC roles on the storage account (and optionally
# the Key Vault), then creates a federated credential so the cleanroom workload can
# authenticate via OIDC token exchange.
###############################################################################################

param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$subject,  # Format: {contractId}-{userId}

    [Parameter(Mandatory)]
    [string]$issuerUrl,

    [string]$outDir = "./generated",

    [switch]$setupKeyVault = $false
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Runs an az command, returning $null instead of throwing if it fails.
function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# -- Load generated resource names -----------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

# -- Resolve resource IDs and principal ID ---------------------------------------
Write-Host "Resolving managed identity '$MANAGED_IDENTITY_NAME'..." -ForegroundColor Cyan
$identityJson = az identity show `
    --name $MANAGED_IDENTITY_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$principalId = $identityJson.principalId

Write-Host "Resolving storage account '$STORAGE_ACCOUNT_NAME'..." -ForegroundColor Cyan
$storageId = az storage account show `
    --name $STORAGE_ACCOUNT_NAME `
    --resource-group $resourceGroup `
    --query id -o tsv

# -- Storage RBAC ----------------------------------------------------------------
Write-Host "Assigning 'Storage Blob Data Owner' to managed identity on storage account..." -ForegroundColor Cyan
Invoke-AzSafe @("role", "assignment", "create", "--role", "Storage Blob Data Owner", "--assignee-object-id", $principalId, "--assignee-principal-type", "ServicePrincipal", "--scope", $storageId, "--output", "none")
Write-Host "Storage role assignment complete." -ForegroundColor Green

# -- Key Vault RBAC (optional) --------------------------------------------------
if ($setupKeyVault) {
    Write-Host "Resolving Key Vault '$KEYVAULT_NAME'..." -ForegroundColor Cyan
    $kvId = az keyvault show `
        --name $KEYVAULT_NAME `
        --resource-group $resourceGroup `
        --query id -o tsv

    Write-Host "Assigning 'Key Vault Crypto Officer' to managed identity..." -ForegroundColor Cyan
    Invoke-AzSafe @("role", "assignment", "create", "--role", "Key Vault Crypto Officer", "--assignee-object-id", $principalId, "--assignee-principal-type", "ServicePrincipal", "--scope", $kvId, "--output", "none")

    Write-Host "Assigning 'Key Vault Secrets User' to managed identity..." -ForegroundColor Cyan
    Invoke-AzSafe @("role", "assignment", "create", "--role", "Key Vault Secrets User", "--assignee-object-id", $principalId, "--assignee-principal-type", "ServicePrincipal", "--scope", $kvId, "--output", "none")

    Write-Host "Key Vault role assignments complete." -ForegroundColor Green
}

# -- Federated Credential -------------------------------------------------------
$federationName = "$subject-federation"
Write-Host "Creating federated credential '$federationName'..." -ForegroundColor Cyan
$existingFed = Invoke-AzSafe @("identity", "federated-credential", "show", "--name", $federationName, "--identity-name", $MANAGED_IDENTITY_NAME, "--resource-group", $resourceGroup, "--output", "json")
if ($existingFed) {
    Write-Host "Federated credential '$federationName' already exists." -ForegroundColor Green
} else {
    az identity federated-credential create `
        --name $federationName `
        --identity-name $MANAGED_IDENTITY_NAME `
        --resource-group $resourceGroup `
        --issuer $issuerUrl `
        --subject $subject `
        --audiences "api://AzureADTokenExchange" `
        --output none
    Write-Host "Federated credential '$federationName' created." -ForegroundColor Green
}

Write-Host "Access setup complete for subject '$subject'." -ForegroundColor Green
