<#
.SYNOPSIS
    Uploads data to Azure Blob Storage WITHOUT az cleanroom CLI (SSE mode).

.DESCRIPTION
    Replaces: az cleanroom datastore add + az cleanroom datastore upload
    Uses: az storage blob upload-batch (standard Azure CLI)

    For SSE (Server-Side Encryption), no client-side encryption is needed.
    Data is uploaded as plain files; Azure Storage encrypts at rest.

    Also generates a local datastore metadata JSON file that downstream
    scripts use to build the DatasetSpecification payload.

.PARAMETER resourceGroup
    Azure resource group containing the storage account.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER dataDir
    Path to the directory containing input data files.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$dataDir,

    [string]$datasetSuffix = "",

    [string]$outDir = "./generated"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

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

# Load generated resource names from prepare-resources step.
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

if (-not (Test-Path $dataDir)) {
    Write-Host "ERROR: Data directory '$dataDir' not found." -ForegroundColor Red
    exit 1
}

# Resolve storage account details.
Write-Host "Resolving storage account '$STORAGE_ACCOUNT_NAME'..." -ForegroundColor Cyan
$storageJson = az storage account show `
    --name $STORAGE_ACCOUNT_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$storageAccountId = $storageJson.id
$storageBlobEndpoint = $storageJson.primaryEndpoints.blob

# Container name convention: use persona name as container.
$inputContainer = "$persona-input"

# Step 1: Create blob container for input data.
Write-Host "`n=== Step 1: Creating container '$inputContainer' ===" -ForegroundColor Cyan
Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $inputContainer, "--auth-mode", "login", "--output", "none")
Write-Host "Container '$inputContainer' ready." -ForegroundColor Green

# Step 2: Upload data files.
Write-Host "`n=== Step 2: Uploading data from '$dataDir' ===" -ForegroundColor Cyan
az storage blob upload-batch `
    --account-name $STORAGE_ACCOUNT_NAME `
    --destination $inputContainer `
    --source $dataDir `
    --auth-mode login `
    --overwrite `
    --output none
Write-Host "Data uploaded successfully." -ForegroundColor Green

# Step 3: Create output container (Woodgrove only).
$outputContainer = "$persona-output"
if ($persona -eq "woodgrove") {
    Write-Host "`n=== Step 3: Creating output container '$outputContainer' ===" -ForegroundColor Cyan
    Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $outputContainer, "--auth-mode", "login", "--output", "none")
    Write-Host "Output container '$outputContainer' ready." -ForegroundColor Green
}

# Step 4: Save datastore metadata for downstream scripts.
# This replaces what `az cleanroom datastore add` would store in its config.
Write-Host "`n=== Step 4: Saving datastore metadata ===" -ForegroundColor Cyan
$datastoreDir = Join-Path $outDir "datastores"
if (-not (Test-Path $datastoreDir)) {
    New-Item -ItemType Directory -Path $datastoreDir -Force | Out-Null
}

$inputSchema = @{
    format = "csv"
    fields = @(
        @{ fieldName = "date"; fieldType = "date" }
        @{ fieldName = "time"; fieldType = "string" }
        @{ fieldName = "author"; fieldType = "string" }
        @{ fieldName = "mentions"; fieldType = "string" }
    )
}

$datastoreMetadata = @{
    input = @{
        name         = "$persona-input-csv$datasetSuffix"
        storeType    = "Azure_BlobStorage"
        storeId      = $storageAccountId
        storeUrl     = $storageBlobEndpoint
        containerName = $inputContainer
        encryptionMode = "SSE"
        schema       = $inputSchema
    }
}

if ($persona -eq "woodgrove") {
    $outputSchema = @{
        format = "csv"
        fields = @(
            @{ fieldName = "author"; fieldType = "string" }
            @{ fieldName = "Number_Of_Mentions"; fieldType = "long" }
            @{ fieldName = "Restricted_Sum"; fieldType = "number" }
        )
    }
    $datastoreMetadata.output = @{
        name         = "woodgrove-output-csv$datasetSuffix"
        storeType    = "Azure_BlobStorage"
        storeId      = $storageAccountId
        storeUrl     = $storageBlobEndpoint
        containerName = $outputContainer
        encryptionMode = "SSE"
        schema       = $outputSchema
    }
}

$metadataFile = Join-Path $datastoreDir "$persona-datastore-metadata.json"
$datastoreMetadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataFile -Encoding utf8
Write-Host "Datastore metadata saved to: $metadataFile" -ForegroundColor Yellow

Write-Host "`nData preparation complete for '$persona' (no az cleanroom CLI used)." -ForegroundColor Green
