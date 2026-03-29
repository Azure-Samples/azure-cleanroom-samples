<#
.SYNOPSIS
    Uploads data with client-side encryption WITHOUT az cleanroom CLI (CPK mode).

.DESCRIPTION
    Replaces: az cleanroom datastore add + az cleanroom datastore upload + az cleanroom secretstore add
    Uses: PowerShell crypto + az keyvault + az storage blob upload

    For CPK (Client-Provided Keys), data must be encrypted BEFORE uploading:
    1. Generate a random AES-256 DEK (Data Encryption Key)
    2. Encrypt each file locally with the DEK
    3. Upload encrypted files to Azure Blob Storage
    4. Wrap the DEK with a KEK (Key Encryption Key) in Key Vault
    5. Store the wrapped DEK as a Key Vault secret

    Also generates datastore metadata including encryption key references
    for the dataset publish step.

.PARAMETER resourceGroup
    Azure resource group containing the storage account and Key Vault.

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

    [string]$outDir = "./generated"
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

# -- Step 1: Create containers --------------------------------------------------
Write-Host "`n=== Step 1: Creating blob containers ===" -ForegroundColor Cyan
Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $inputContainer, "--auth-mode", "login", "--output", "none")
Write-Host "Container '$inputContainer' ready." -ForegroundColor Green

if ($persona -eq "woodgrove") {
    Invoke-AzSafe @("storage", "container", "create", "--account-name", $STORAGE_ACCOUNT_NAME, "--name", $outputContainer, "--auth-mode", "login", "--output", "none")
    Write-Host "Container '$outputContainer' ready." -ForegroundColor Green
}

# -- Step 2: Generate DEK (AES-256 key) ----------------------------------------
Write-Host "`n=== Step 2: Generating Data Encryption Key (DEK) ===" -ForegroundColor Cyan
$aes = [System.Security.Cryptography.Aes]::Create()
$aes.KeySize = 256
$aes.GenerateKey()
$dekBytes = $aes.Key
$dekBase64 = [Convert]::ToBase64String($dekBytes)
Write-Host "DEK generated (AES-256, $($dekBytes.Length) bytes)." -ForegroundColor Green

# -- Step 3: Create KEK in Key Vault -------------------------------------------
Write-Host "`n=== Step 3: Creating Key Encryption Key (KEK) in Key Vault ===" -ForegroundColor Cyan
$kekName = "$persona-kek"
az keyvault key create `
    --vault-name $KEYVAULT_NAME `
    --name $kekName `
    --kty RSA `
    --size 2048 `
    --ops encrypt decrypt wrapKey unwrapKey `
    --output none
Write-Host "KEK '$kekName' created in Key Vault." -ForegroundColor Green

# -- Step 4: Wrap DEK with KEK ------------------------------------------------
Write-Host "`n=== Step 4: Wrapping DEK with KEK ===" -ForegroundColor Cyan
# Write DEK to a temp file for wrapping.
$dekTempFile = Join-Path $outDir $resourceGroup "dek-temp.bin"
New-Item -ItemType Directory -Path (Split-Path $dekTempFile) -Force | Out-Null
[System.IO.File]::WriteAllBytes($dekTempFile, $dekBytes)

$wrappedResult = az keyvault key encrypt `
    --vault-name $KEYVAULT_NAME `
    --name $kekName `
    --algorithm RSA-OAEP-256 `
    --value $dekBase64 `
    --output json | ConvertFrom-Json
$wrappedDekBase64 = $wrappedResult.result

# Store wrapped DEK as a Key Vault secret.
$wrappedDekSecretName = "wrapped-$persona-input-csv-dek-$kekName"
az keyvault secret set `
    --vault-name $KEYVAULT_NAME `
    --name $wrappedDekSecretName `
    --value $wrappedDekBase64 `
    --output none
Write-Host "Wrapped DEK stored as secret '$wrappedDekSecretName'." -ForegroundColor Green

# Clean up temp DEK file.
Remove-Item $dekTempFile -ErrorAction SilentlyContinue

# -- Step 5: Encrypt and upload data files -------------------------------------
Write-Host "`n=== Step 5: Encrypting and uploading data files ===" -ForegroundColor Cyan

$encryptedDir = Join-Path $outDir $resourceGroup "encrypted-$persona"
if (Test-Path $encryptedDir) {
    Remove-Item $encryptedDir -Recurse -Force
}
New-Item -ItemType Directory -Path $encryptedDir -Force | Out-Null

$dataFiles = Get-ChildItem -Path $dataDir -File -Recurse
foreach ($file in $dataFiles) {
    Write-Host "  Encrypting $($file.Name)..." -ForegroundColor Yellow

    # Read file content.
    $plainBytes = [System.IO.File]::ReadAllBytes($file.FullName)

    # Encrypt with AES-256-CBC.
    $aesEnc = [System.Security.Cryptography.Aes]::Create()
    $aesEnc.Key = $dekBytes
    $aesEnc.Mode = [System.Security.Cryptography.CipherMode]::CBC
    $aesEnc.Padding = [System.Security.Cryptography.PaddingMode]::PKCS7
    $aesEnc.GenerateIV()
    $iv = $aesEnc.IV

    $encryptor = $aesEnc.CreateEncryptor()
    $encryptedBytes = $encryptor.TransformFinalBlock($plainBytes, 0, $plainBytes.Length)

    # Prepend IV to encrypted data (IV needed for decryption).
    $outputBytes = New-Object byte[] ($iv.Length + $encryptedBytes.Length)
    [Array]::Copy($iv, 0, $outputBytes, 0, $iv.Length)
    [Array]::Copy($encryptedBytes, 0, $outputBytes, $iv.Length, $encryptedBytes.Length)

    # Preserve subdirectory structure (e.g., date-partitioned folders).
    $relativePath = $file.FullName.Substring($dataDir.Length).TrimStart('\', '/')
    $encryptedFilePath = Join-Path $encryptedDir $relativePath
    New-Item -ItemType Directory -Path (Split-Path $encryptedFilePath) -Force | Out-Null
    [System.IO.File]::WriteAllBytes($encryptedFilePath, $outputBytes)
    $encryptor.Dispose()
    $aesEnc.Dispose()
}
Write-Host "All files encrypted." -ForegroundColor Green

# Upload encrypted files.
Write-Host "  Uploading encrypted files..." -ForegroundColor Yellow
az storage blob upload-batch `
    --account-name $STORAGE_ACCOUNT_NAME `
    --destination $inputContainer `
    --source $encryptedDir `
    --auth-mode login `
    --overwrite `
    --output none
Write-Host "Encrypted data uploaded." -ForegroundColor Green

# Clean up encrypted temp files.
Remove-Item $encryptedDir -Recurse -Force

# -- Step 6: Save datastore metadata ------------------------------------------
Write-Host "`n=== Step 6: Saving datastore metadata ===" -ForegroundColor Cyan
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
        name            = "$persona-input-csv"
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $inputContainer
        encryptionMode  = "CPK"
        schema          = $inputSchema
        encryption      = @{
            dekSecretName = $wrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $kekName
            kekStoreUrl   = $kvUrl
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

    # Output dataset needs its own DEK/KEK for write encryption.
    $outputWrappedDekSecretName = "wrapped-woodgrove-output-csv-dek-$kekName"
    az keyvault secret set `
        --vault-name $KEYVAULT_NAME `
        --name $outputWrappedDekSecretName `
        --value $wrappedDekBase64 `
        --output none

    $datastoreMetadata.output = @{
        name            = "woodgrove-output-csv"
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $outputContainer
        encryptionMode  = "CPK"
        schema          = $outputSchema
        encryption      = @{
            dekSecretName = $outputWrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $kekName
            kekStoreUrl   = $kvUrl
        }
    }
}

$metadataFile = Join-Path $datastoreDir "$persona-datastore-metadata.json"
$datastoreMetadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataFile -Encoding utf8
Write-Host "Datastore metadata saved to: $metadataFile" -ForegroundColor Yellow

$aes.Dispose()
Write-Host "`nData preparation complete for '$persona' (no az cleanroom CLI used, CPK mode)." -ForegroundColor Green
