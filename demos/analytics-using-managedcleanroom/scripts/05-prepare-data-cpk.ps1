<#
.SYNOPSIS
    Uploads data with CPK (Customer-Provided Key) encryption via Azure Storage SSE-CPK.

.DESCRIPTION
    Replaces: az cleanroom datastore add + az cleanroom datastore upload
    Uses: azcopy copy --cpk-by-value (Azure Storage server-side encryption with customer-provided keys)

    IMPORTANT: CPK mode does NOT mean client-side encryption. Data is uploaded as plaintext
    and Azure Storage encrypts it server-side using the customer-provided key (DEK). The
    cleanroom's blobfuse layer releases the KEK via SKR, unwraps the DEK, and passes it
    to Azure Storage via CPK headers to transparently decrypt at read time.

    This script performs Phase A of the CPK flow:
    1. Generate a random 32-byte DEK (Data Encryption Key)
    2. Create blob containers (input, and output for woodgrove)
    3. Upload data files using azcopy --cpk-by-value (Azure Storage encrypts server-side)
    4. Save DEK binary file and datastore metadata

    Phase B (KEK creation + DEK wrapping) happens in 08-publish-dataset-cpk.ps1 AFTER
    datasets are published, because the SKR release policy must be fetched from the
    published dataset.

    Requires: azcopy (v10+), python3 with cryptography package

.PARAMETER resourceGroup
    Azure resource group containing the storage account and Key Vault.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER dataDir
    Path to the directory containing input data files.

.PARAMETER outDir
    Output directory for generated metadata (default: ./generated).

.PARAMETER datasetSuffix
    Optional suffix for dataset names (e.g., "-cpk-v4"). Appended to the base
    dataset name (e.g., "northwind-input-csv-cpk-v4").
#>
param(
    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$dataDir,

    [string]$outDir = "./generated",

    [string]$datasetSuffix = ""
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Resolve outDir and dataDir to absolute paths (avoids .NET vs PowerShell CWD mismatch).
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

# Load generated resource names.
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

# Verify azcopy is available.
$azcopyPath = Get-Command azcopy -ErrorAction SilentlyContinue
if (-not $azcopyPath) {
    Write-Host "ERROR: azcopy not found in PATH. Install from https://aka.ms/azcopy" -ForegroundColor Red
    exit 1
}

# Resolve python executable (python3 on Linux/macOS, python on Windows).
# Windows has a python3.exe stub that opens Microsoft Store — test with --version.
$script:pythonExe = $null
foreach ($candidate in @("python3", "python")) {
    try {
        $PSNativeCommandUseErrorActionPreference = $false
        & $candidate --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $script:pythonExe = $candidate; break }
    } catch {}
}
$PSNativeCommandUseErrorActionPreference = $true
if (-not $script:pythonExe) {
    Write-Host "ERROR: Python not found. Install Python 3 from https://python.org" -ForegroundColor Red
    exit 1
}

# Resolve storage and Key Vault details.
Write-Host "Resolving resources..." -ForegroundColor Cyan
$storageJson = az storage account show `
    --name $STORAGE_ACCOUNT_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$storageAccountId = $storageJson.id
$storageBlobEndpoint = $storageJson.primaryEndpoints.blob

$kvJson = az keyvault show `
    --name $KEYVAULT_NAME `
    --resource-group $resourceGroup `
    --output json | ConvertFrom-Json
$kvUrl = $kvJson.properties.vaultUri.TrimEnd('/')

$inputContainer = "$persona-input"
$outputContainer = "$persona-output"

# Dataset naming: base name + optional suffix (e.g., "northwind-input-csv-cpk-v4")
$inputDatasetName = "$persona-input-csv$datasetSuffix"
$outputDatasetName = "woodgrove-output-csv$datasetSuffix"

# Per-dataset KEK naming (used in metadata, key created later in step 08)
$inputKekName = "$inputDatasetName-kek"
$outputKekName = "$outputDatasetName-kek"

# -- Step 1: Create containers --------------------------------------------------
Write-Host "`n=== Step 1: Creating blob containers ===" -ForegroundColor Cyan
Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $inputContainer, "--auth-mode", "login", "--output", "none")
Write-Host "Container '$inputContainer' ready." -ForegroundColor Green

if ($persona -eq "woodgrove") {
    Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $outputContainer, "--auth-mode", "login", "--output", "none")
    Write-Host "Container '$outputContainer' ready." -ForegroundColor Green
}

# -- Step 2: Generate DEK (32-byte random key) ----------------------------------
Write-Host "`n=== Step 2: Generating Data Encryption Key (DEK) ===" -ForegroundColor Cyan
$dekDir = Join-Path $outDir "datastores" "keys"
New-Item -ItemType Directory -Path $dekDir -Force | Out-Null

# Generate a 32-byte random DEK (matches az cleanroom CLI implementation).
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

# -- Step 3: Upload data with azcopy --cpk-by-value ----------------------------
Write-Host "`n=== Step 3: Uploading data with CPK (azcopy --cpk-by-value) ===" -ForegroundColor Cyan

# Get tenant ID for azcopy auto-login.
$tenantId = (az account show --query tenantId -o tsv)
$containerUrl = "${storageBlobEndpoint}${inputContainer}/"

Write-Host "  Uploading to: $containerUrl" -ForegroundColor Yellow
Write-Host "  Using AZCLI auto-login (tenant: $tenantId)" -ForegroundColor Yellow

# Clear existing blobs in the container first (in case of re-run).
Write-Host "  Clearing existing blobs in '$inputContainer'..." -ForegroundColor Yellow
Invoke-AzSafe @("storage", "blob", "delete-batch", "--account-name", $STORAGE_ACCOUNT_NAME, "--source", $inputContainer, "--auth-mode", "login")

# Set environment variables for azcopy CPK mode.
$env:CPK_ENCRYPTION_KEY = $inputDekBase64
$env:CPK_ENCRYPTION_KEY_SHA256 = $inputDekSha256
$env:AZCOPY_AUTO_LOGIN_TYPE = "AZCLI"
$env:AZCOPY_TENANT_ID = $tenantId

try {
    # Upload data files using azcopy with CPK. Azure Storage encrypts server-side.
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
    # Clean up CPK environment variables.
    Remove-Item env:CPK_ENCRYPTION_KEY -ErrorAction SilentlyContinue
    Remove-Item env:CPK_ENCRYPTION_KEY_SHA256 -ErrorAction SilentlyContinue
    Remove-Item env:AZCOPY_AUTO_LOGIN_TYPE -ErrorAction SilentlyContinue
    Remove-Item env:AZCOPY_TENANT_ID -ErrorAction SilentlyContinue
}

# -- Step 4: Save datastore metadata ------------------------------------------
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

# Wrapped DEK secret names follow the convention: wrapped-{datastore}-dek-{kekName}
$inputWrappedDekSecretName = "wrapped-$inputDatasetName-dek-$inputKekName"

$datastoreMetadata = @{
    input = @{
        name            = $inputDatasetName
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $inputContainer
        encryptionMode  = "CPK"
        schema          = $inputSchema
        encryption      = @{
            dekSecretName = $inputWrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $inputKekName
            kekStoreUrl   = $kvUrl
        }
        # Local references for Phase B (KEK creation + DEK wrapping in step 08)
        _local = @{
            dekFile = $inputDekFile
        }
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

    $outputWrappedDekSecretName = "wrapped-$outputDatasetName-dek-$outputKekName"

    $datastoreMetadata.output = @{
        name            = $outputDatasetName
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $outputContainer
        encryptionMode  = "CPK"
        schema          = $outputSchema
        encryption      = @{
            dekSecretName = $outputWrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $outputKekName
            kekStoreUrl   = $kvUrl
        }
        _local = @{
            dekFile = $outputDekFile
        }
    }
}

$metadataFile = Join-Path $datastoreDir "$persona-datastore-metadata.json"
$datastoreMetadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataFile -Encoding utf8
Write-Host "Datastore metadata saved to: $metadataFile" -ForegroundColor Yellow

Write-Host "`n=== Summary ===" -ForegroundColor Cyan
Write-Host "  Input dataset:  $inputDatasetName" -ForegroundColor White
Write-Host "  Input DEK file: $inputDekFile" -ForegroundColor White
Write-Host "  Input KEK name: $inputKekName (will be created in step 08)" -ForegroundColor White
if ($persona -eq "woodgrove") {
    Write-Host "  Output dataset:  $outputDatasetName" -ForegroundColor White
    Write-Host "  Output DEK file: $outputDekFile" -ForegroundColor White
    Write-Host "  Output KEK name: $outputKekName (will be created in step 08)" -ForegroundColor White
}
Write-Host "`nPhase A complete for '$persona'. Run 08-publish-dataset-cpk.ps1 next to:" -ForegroundColor Green
Write-Host "  - Publish datasets to the collaboration" -ForegroundColor Green
Write-Host "  - Fetch SKR release policy from published datasets" -ForegroundColor Green
Write-Host "  - Create per-dataset KEKs with the SKR policy" -ForegroundColor Green
Write-Host "  - Wrap DEKs with KEKs and store as Key Vault secrets" -ForegroundColor Green
