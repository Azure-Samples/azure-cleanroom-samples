<#
.SYNOPSIS
    Uploads data to Azure Blob Storage (SSE or CPK mode).

.DESCRIPTION
    For SSE (Server-Side Encryption):
      - Upload files directly; Azure handles encryption at rest.

    For CPK (Client-Provided Keys):
      1. Generate a random AES-256 DEK (Data Encryption Key)
      2. Generate a local RSA-2048 KEK (Key Encryption Key)
      3. Retrieve the clean room policy hash from the frontend
      4. Import the KEK into Key Vault with an SKR release policy
      5. Wrap the DEK locally with the KEK and store in Key Vault
      6. Upload data with CPK encryption headers

    Both modes generate datastore metadata for the dataset publish step.

.PARAMETER resourceGroup
    Azure resource group containing the storage account (and Key Vault for CPK).

.PARAMETER variant
    Encryption variant: "sse" or "cpk".

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER dataDir
    Path to the directory containing input data files.

.PARAMETER collaborationId
    Collaboration identifier (required for CPK; used to fetch clean room policy).

.PARAMETER maaEndpoint
    Microsoft Azure Attestation endpoint (CPK only; default: sharedeus.eus.attest.azure.net).

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

    [string]$collaborationId,

    [string]$maaEndpoint = "sharedeus.eus.attest.azure.net",

    [string]$outDir = "./generated"
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

if ($variant -eq "cpk" -and -not $collaborationId) {
    Write-Host "ERROR: -collaborationId is required for CPK variant." -ForegroundColor Red
    exit 1
}

# Resolve paths to absolute (avoids .NET vs PowerShell CWD mismatch).
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)
$dataDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($dataDir)

. "$PSScriptRoot/common/utils.ps1"

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

# -- CPK: Generate keys, import with SKR, wrap DEK ------------------------------
if ($variant -eq "cpk") {
    # Step 2: Generate DEK and KEK.
    Write-Host "`n=== Step 2: Generating DEK and KEK ===" -ForegroundColor Cyan

    $dekBytes = [byte[]]::new(32)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($dekBytes)
    $dekBase64 = [Convert]::ToBase64String($dekBytes)
    $dekSha256Base64 = [Convert]::ToBase64String(
        [System.Security.Cryptography.SHA256]::HashData($dekBytes))
    Write-Host "DEK generated (AES-256, $($dekBytes.Length) bytes)." -ForegroundColor Green

    $rsa = [System.Security.Cryptography.RSA]::Create(2048)
    $kekPrivatePem = $rsa.ExportPkcs8PrivateKeyPem()
    $kekPrivateFile = Join-Path $outDir $resourceGroup "$persona-kek-private.pem"
    New-Item -ItemType Directory -Path (Split-Path $kekPrivateFile) -Force | Out-Null
    $kekPrivatePem | Set-Content -Path $kekPrivateFile -NoNewline
    Write-Host "KEK generated (RSA-2048) and saved to '$kekPrivateFile'." -ForegroundColor Green

    # Step 3: Import KEK with Secure Key Release policy.
    Write-Host "`n=== Step 3: Importing KEK with Secure Key Release policy ===" -ForegroundColor Cyan
    $kekName = "$persona-kek"

    Write-Host "  Fetching clean room policy hash..." -ForegroundColor Yellow
    $policyResponse = az managedcleanroom frontend analytics cleanroompolicy `
        -c $collaborationId `
        --output json | ConvertFrom-Json
    $ccePolicyHash = $policyResponse.policy.ccePolicyHash
    Write-Host "  Clean room policy hash: $ccePolicyHash" -ForegroundColor Green

    Write-Host "  Importing KEK '$kekName' with SKR policy..." -ForegroundColor Yellow
    az keyvault key import `
        --vault-name $KEYVAULT_NAME `
        --name $kekName `
        --pem-file $kekPrivateFile `
        --protection hsm `
        --exportable true `
        --ops wrapKey unwrapKey `
        --policy @"
{
    "anyOf": [
        {
            "allOf": [
                {
                    "claim": "x-ms-sevsnpvm-hostdata",
                    "equals": "$ccePolicyHash"
                },
                {
                    "claim": "x-ms-compliance-status",
                    "equals": "azure-compliant-uvm"
                },
                {
                    "claim": "x-ms-attestation-type",
                    "equals": "sevsnpvm"
                }
            ],
            "authority": "https://$maaEndpoint"
        }
    ],
    "version": "1.0.0"
}
"@ `
        --output none
    Write-Host "KEK '$kekName' imported with SKR policy." -ForegroundColor Green

    # Step 4: Wrap DEK with KEK and store in Key Vault.
    Write-Host "`n=== Step 4: Wrapping DEK with KEK ===" -ForegroundColor Cyan

    $wrappedDekBytes = $rsa.Encrypt($dekBytes, [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
    $wrappedDekHex = ($wrappedDekBytes | ForEach-Object { $_.ToString("x2") }) -join ''

    $wrappedDekSecretName = "$persona-dek-wrapped"
    az keyvault secret set `
        --vault-name $KEYVAULT_NAME `
        --name $wrappedDekSecretName `
        --value $wrappedDekHex `
        --output none
    Write-Host "Wrapped DEK stored as secret '$wrappedDekSecretName'." -ForegroundColor Green

    $rsa.Dispose()
}

# -- Upload data files -----------------------------------------------------------
$uploadStep = if ($variant -eq "cpk") { "Step 5" } else { "Step 2" }
Write-Host "`n=== ${uploadStep}: Uploading data files ($variant) ===" -ForegroundColor Cyan

if ($variant -eq "cpk") {
    # CPK upload via REST API with encryption headers.
    $tokenJson = az account get-access-token --resource https://storage.azure.com --output json | ConvertFrom-Json
    $accessToken = $tokenJson.accessToken

    $dataFiles = Get-ChildItem -Path $dataDir -File -Recurse
    foreach ($file in $dataFiles) {
        $relativePath = $file.FullName.Substring($dataDir.Length).TrimStart('\', '/') -replace '\\', '/'
        Write-Host "  Uploading $relativePath (CPK)..." -ForegroundColor Yellow

        $blobUrl = "${storageBlobEndpoint}${inputContainer}/${relativePath}"
        $fileBytes = [System.IO.File]::ReadAllBytes($file.FullName)

        $headers = @{
            "Authorization"                = "Bearer $accessToken"
            "x-ms-blob-type"               = "BlockBlob"
            "x-ms-version"                 = "2024-11-04"
            "x-ms-encryption-key"          = $dekBase64
            "x-ms-encryption-key-sha256"   = $dekSha256Base64
            "x-ms-encryption-algorithm"    = "AES256"
        }

        Invoke-RestMethod -Uri $blobUrl -Method Put -Headers $headers `
            -Body $fileBytes -ContentType "application/octet-stream"
    }
} else {
    # SSE upload via az CLI (Azure handles encryption at rest).
    az storage blob upload-batch `
        --account-name $STORAGE_ACCOUNT_NAME `
        --destination $inputContainer `
        --source $dataDir `
        --auth-mode login `
        --overwrite `
        --output none
}
Write-Host "Data uploaded successfully." -ForegroundColor Green

# -- Save datastore metadata -----------------------------------------------------
$metaStep = if ($variant -eq "cpk") { "Step 6" } else { "Step 3" }
Write-Host "`n=== ${metaStep}: Saving datastore metadata ===" -ForegroundColor Cyan
$datastoreDir = Join-Path $outDir "datastores"
if (-not (Test-Path $datastoreDir)) {
    New-Item -ItemType Directory -Path $datastoreDir -Force | Out-Null
}

$inputSchema = @{
    format = "csv"
    fields = @(
        @{ fieldName = "audience_id"; fieldType = "string" }
        @{ fieldName = "hashed_email"; fieldType = "string" }
        @{ fieldName = "annual_income"; fieldType = "long" }
        @{ fieldName = "region"; fieldType = "string" }
    )
}

if ($persona -eq "woodgrove") {
    $inputSchema = @{
        format = "csv"
        fields = @(
            @{ fieldName = "user_id"; fieldType = "string" }
            @{ fieldName = "hashed_email"; fieldType = "string" }
            @{ fieldName = "purchase_history"; fieldType = "string" }
        )
    }
}

$datastoreMetadata = @{
    input = @{
        name            = "$persona-input-csv"
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $inputContainer
        encryptionMode  = $variant.ToUpper()
        schema          = $inputSchema
    }
}

if ($variant -eq "cpk") {
    $datastoreMetadata.input.encryption = @{
        dekSecretName = $wrappedDekSecretName
        dekStoreUrl   = $kvUrl
        kekName       = $kekName
        kekStoreUrl   = $kvUrl
    }
}

if ($persona -eq "woodgrove") {
    $outputSchema = @{
        format = "csv"
        fields = @(
            @{ fieldName = "user_id"; fieldType = "string" }
        )
    }

    $outputMeta = @{
        name            = "woodgrove-output-csv"
        storeType       = "Azure_BlobStorage"
        storeId         = $storageAccountId
        storeUrl        = $storageBlobEndpoint
        containerName   = $outputContainer
        encryptionMode  = $variant.ToUpper()
        schema          = $outputSchema
    }

    if ($variant -eq "cpk") {
        $outputWrappedDekSecretName = "woodgrove-output-dek-wrapped"
        az keyvault secret set `
            --vault-name $KEYVAULT_NAME `
            --name $outputWrappedDekSecretName `
            --value $wrappedDekHex `
            --output none

        $outputMeta.encryption = @{
            dekSecretName = $outputWrappedDekSecretName
            dekStoreUrl   = $kvUrl
            kekName       = $kekName
            kekStoreUrl   = $kvUrl
        }
    }

    $datastoreMetadata.output = $outputMeta
}

$metadataFile = Join-Path $datastoreDir "$persona-datastore-metadata.json"
$datastoreMetadata | ConvertTo-Json -Depth 10 | Out-File -FilePath $metadataFile -Encoding utf8
Write-Host "Datastore metadata saved to: $metadataFile" -ForegroundColor Yellow

Write-Host "`nData preparation complete for '$persona' ($variant mode)." -ForegroundColor Green
