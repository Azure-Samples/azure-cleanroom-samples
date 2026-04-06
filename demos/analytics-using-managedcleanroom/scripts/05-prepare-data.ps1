<#
.SYNOPSIS
    Uploads data to Azure Blob Storage (SSE or CPK mode).

.DESCRIPTION
    For SSE (Server-Side Encryption):
      - Upload files directly; Azure handles encryption at rest.

    For CPK (Client-Provided Keys):
      1. Generate a random AES-256 DEK (Data Encryption Key)
      2. Upload data with azcopy --cpk-by-value (Azure Storage encrypts server-side)
      3. Save DEK binary file for Phase B (KEK creation in 08-publish-dataset.ps1)

    Both modes generate datastore metadata for the dataset publish step.

.PARAMETER resourceGroup
    Azure resource group containing the storage account (and Key Vault for CPK).

.PARAMETER variant
    Encryption variant: "sse" or "cpk".

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER dataDir
    Path to the directory containing input data files.

.PARAMETER datasetSuffix
    Optional suffix for dataset names (e.g., "-cpk-v1").

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
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$dataDir,

    [string]$datasetSuffix = "",

    [string]$outDir = "./generated",

    [string]$appId,

    [string]$appTenantId,

    [string]$appCertPemPath
)

# Auth: app-based (SPN) or user-based
. "$PSScriptRoot/common/setup-local-auth.ps1"
if ($appId -and $appCertPemPath -and $appTenantId) {
    Initialize-AppAuth -appId $appId -tenantId $appTenantId -certPemPath $appCertPemPath
}

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Resolve paths to absolute (avoids .NET vs PowerShell CWD mismatch).
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)
$dataDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($dataDir)

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

# CPK prerequisites
if ($variant -eq "cpk") {
    $azcopyPath = Get-Command azcopy -ErrorAction SilentlyContinue
    if (-not $azcopyPath) {
        Write-Host "ERROR: azcopy not found in PATH. Install from https://aka.ms/azcopy" -ForegroundColor Red
        exit 1
    }
}

# -- Resolve storage account ----------------------------------------------------
Write-Host "Resolving storage account '$STORAGE_ACCOUNT_NAME'..." -ForegroundColor Cyan
$storageJson = az storage account show `
    --name $STORAGE_ACCOUNT_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$storageAccountId = $storageJson.id
$storageBlobEndpoint = $storageJson.primaryEndpoints.blob

if ($variant -eq "cpk") {
    Write-Host "Resolving Key Vault '$KEYVAULT_NAME'..." -ForegroundColor Cyan
    $kvJson = az keyvault show `
        --name $KEYVAULT_NAME `
        --resource-group $resourceGroup `
        --output json | ConvertFrom-Json
    $kvUrl = $kvJson.properties.vaultUri.TrimEnd('/')
}

# Container names: each iteration gets unique containers to avoid CPK key conflicts.
# For SSE, the suffix is still applied for consistency (idempotent container create).
$inputContainer = if ($datasetSuffix) { "$persona-input$datasetSuffix" } else { "$persona-input" }
$outputContainer = if ($datasetSuffix) { "$persona-output$datasetSuffix" } else { "$persona-output" }

# Dataset naming
$inputDatasetName = "$persona-input-csv$datasetSuffix"
$outputDatasetName = "woodgrove-output-csv$datasetSuffix"

# -- Step 1: Create containers --------------------------------------------------
Write-Host "`n=== Step 1: Creating blob containers ===" -ForegroundColor Cyan
Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $inputContainer, "--auth-mode", "login", "--output", "none")
Write-Host "Container '$inputContainer' ready." -ForegroundColor Green

if ($persona -eq "woodgrove") {
    Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $outputContainer, "--auth-mode", "login", "--output", "none")
    Write-Host "Container '$outputContainer' ready." -ForegroundColor Green
}

# -- CPK: Generate DEK ----------------------------------------------------------
if ($variant -eq "cpk") {
    Write-Host "`n=== Step 2: Generating Data Encryption Key (DEK) ===" -ForegroundColor Cyan
    $dekDir = Join-Path $outDir "datastores" "keys"
    New-Item -ItemType Directory -Path $dekDir -Force | Out-Null

    $inputDekFile = Join-Path $dekDir "$inputDatasetName-dek.bin"
    $dek = [byte[]]::new(32)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($dek)
    [System.IO.File]::WriteAllBytes($inputDekFile, $dek)
    $inputDekBytes = [System.IO.File]::ReadAllBytes($inputDekFile)
    $inputDekBase64 = [Convert]::ToBase64String($inputDekBytes)
    $inputDekSha256 = [Convert]::ToBase64String(
        [System.Security.Cryptography.SHA256]::HashData($inputDekBytes)
    )
    Write-Host "Input DEK generated (32 bytes): $inputDekFile" -ForegroundColor Green

    if ($persona -eq "woodgrove") {
        $outputDekFile = Join-Path $dekDir "$outputDatasetName-dek.bin"
        $dek = [byte[]]::new(32)
        [System.Security.Cryptography.RandomNumberGenerator]::Fill($dek)
        [System.IO.File]::WriteAllBytes($outputDekFile, $dek)
        Write-Host "Output DEK generated (32 bytes): $outputDekFile" -ForegroundColor Green
    }
}

# -- Upload data -----------------------------------------------------------------
if ($variant -eq "sse") {
    Write-Host "`n=== Step 2: Uploading data from '$dataDir' ===" -ForegroundColor Cyan
    az storage blob upload-batch `
        --account-name $STORAGE_ACCOUNT_NAME `
        --destination $inputContainer `
        --source $dataDir `
        --auth-mode login `
        --overwrite `
        --output none
    Write-Host "Data uploaded successfully." -ForegroundColor Green
} else {
    Write-Host "`n=== Step 3: Uploading data with CPK (azcopy --cpk-by-value) ===" -ForegroundColor Cyan

    $tenantId = (az account show --query tenantId -o tsv)
    $containerUrl = "${storageBlobEndpoint}${inputContainer}/"

    Write-Host "  Uploading to: $containerUrl" -ForegroundColor Yellow
    Invoke-AzSafe @("storage", "blob", "delete-batch", "--account-name", $STORAGE_ACCOUNT_NAME, "--source", $inputContainer, "--auth-mode", "login")

    $env:CPK_ENCRYPTION_KEY = $inputDekBase64
    $env:CPK_ENCRYPTION_KEY_SHA256 = $inputDekSha256
    $env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
    $env:AZCOPY_TENANT_ID = $tenantId

    try {
        $PSNativeCommandUseErrorActionPreference = $false
        azcopy copy "$dataDir/*" $containerUrl --recursive --cpk-by-value 2>&1 | ForEach-Object {
            Write-Host "  $_" -ForegroundColor DarkGray
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: azcopy upload failed with exit code $LASTEXITCODE" -ForegroundColor Red
            exit 1
        }
        $PSNativeCommandUseErrorActionPreference = $true
        Write-Host "Data uploaded with CPK encryption." -ForegroundColor Green
    }
    finally {
        Remove-Item env:CPK_ENCRYPTION_KEY -ErrorAction SilentlyContinue
        Remove-Item env:CPK_ENCRYPTION_KEY_SHA256 -ErrorAction SilentlyContinue
        Remove-Item env:AZCOPY_AUTO_LOGIN_TYPE -ErrorAction SilentlyContinue
        Remove-Item env:AZCOPY_TENANT_ID -ErrorAction SilentlyContinue
    }
}

# -- Save datastore metadata ----------------------------------------------------
Write-Host "`n=== Saving datastore metadata ===" -ForegroundColor Cyan
$datastoreDir = Join-Path $outDir "datastores"
if (-not (Test-Path $datastoreDir)) {
    New-Item -ItemType Directory -Path $datastoreDir -Force | Out-Null
}

$inputSchema = @{
    format = "csv"
    fields = if ($persona -eq "northwind") {
        @(
            @{ fieldName = "audience_id"; fieldType = "string" }
            @{ fieldName = "hashed_email"; fieldType = "string" }
            @{ fieldName = "annual_income"; fieldType = "long" }
            @{ fieldName = "region"; fieldType = "string" }
        )
    } else {
        @(
            @{ fieldName = "user_id"; fieldType = "string" }
            @{ fieldName = "hashed_email"; fieldType = "string" }
            @{ fieldName = "purchase_history"; fieldType = "string" }
        )
    }
}

$encryptionMode = if ($variant -eq "cpk") { "CPK" } else { "SSE" }

$inputMeta = @{
    name            = $inputDatasetName
    storeType       = "Azure_BlobStorage"
    storeId         = $storageAccountId
    storeUrl        = $storageBlobEndpoint
    containerName   = $inputContainer
    encryptionMode  = $encryptionMode
    schema          = $inputSchema
}

if ($variant -eq "cpk") {
    $inputKekName = "$inputDatasetName-kek"
    $inputWrappedDekSecretName = "wrapped-$inputDatasetName-dek-$inputKekName"
    $inputMeta.encryption = @{
        dekSecretName = $inputWrappedDekSecretName
        dekStoreUrl   = $kvUrl
        kekName       = $inputKekName
        kekStoreUrl   = $kvUrl
    }
    $inputMeta._local = @{ dekFile = $inputDekFile }
}

$datastoreMetadata = @{ input = $inputMeta }

if ($persona -eq "woodgrove") {
    $outputSchema = @{
        format = "csv"
        fields = @(
            @{ fieldName = "user_id"; fieldType = "string" }
        )
    }

    $outputMeta = @{
        name            = $outputDatasetName
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $outputContainer
        encryptionMode  = $encryptionMode
        schema          = $outputSchema
    }

    if ($variant -eq "cpk") {
        $outputKekName = "$outputDatasetName-kek"
        $outputWrappedDekSecretName = "wrapped-$outputDatasetName-dek-$outputKekName"
        $outputMeta.encryption = @{
            dekSecretName = $outputWrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $outputKekName
            kekStoreUrl   = $kvUrl
        }
        $outputMeta._local = @{ dekFile = $outputDekFile }
    }

    $datastoreMetadata.output = $outputMeta
}

$metadataFile = Join-Path $datastoreDir "$persona-datastore-metadata.json"
$datastoreMetadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataFile -Encoding utf8
Write-Host "Datastore metadata saved to: $metadataFile" -ForegroundColor Yellow

Write-Host "`nData preparation complete for '$persona' ($variant mode)." -ForegroundColor Green
