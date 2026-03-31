<#
.SYNOPSIS
    Publishes dataset metadata and creates encryption keys (CPK variant).

.DESCRIPTION
    Run by: Each collaborator (Northwind and Woodgrove).

    This script implements the publish-first CPK flow:
    Phase A (done in 05-prepare-data-cpk.ps1): DEK generated, data uploaded with azcopy --cpk-by-value
    Phase B (this script):
      1. Publish dataset metadata to the collaboration (with DEK/KEK references)
      2. Fetch the SKR release policy from the just-published dataset
      3. Create a per-dataset KEK locally (RSA-2048) and import to Key Vault with the SKR policy
      4. Wrap the DEK with the KEK (RSA-OAEP-SHA256, client-side) and store as a Key Vault secret
      5. Enable execution consent on published datasets

    This matches the az cleanroom CLI's actual implementation:
    - KEK is generated locally as RSA-2048, then imported via `az keyvault key import`
    - DEK wrapping uses client-side RSA-OAEP-SHA256 (NOT `az keyvault key encrypt`)
    - Per-dataset KEKs (not shared per-persona)

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run (creates Key Vault with Premium SKU).
    - 05-prepare-data-cpk.ps1 must have been run (generates DEK, uploads data, saves metadata).
    - 06-setup-identity.ps1 must have been run (OIDC setup, saves identity metadata).
    - Python 3 with `cryptography` package installed.

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER resourceGroup
    The Azure resource group containing provisioned resources.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER frontendEndpoint
    Frontend service URL.

.PARAMETER maaUrl
    MAA (Microsoft Azure Attestation) URL for the SKR release policy authority.

.PARAMETER outDir
    Output directory for generated configuration files (default: ./generated).

.PARAMETER TokenFile
    Optional path to a pre-generated MSAL IdToken file (for persona switching).

.PARAMETER ApiMode
    API mode: "rest" (default) or "cli".
#>
param(
    [Parameter(Mandatory)]
    [string]$collaborationId,

    [Parameter(Mandatory)]
    [string]$resourceGroup,

    [Parameter(Mandatory)]
    [ValidateSet("northwind", "woodgrove", IgnoreCase = $false)]
    [string]$persona,

    [Parameter(Mandatory)]
    [string]$frontendEndpoint,

    [string]$maaUrl = "https://sharedeus.eus.attest.azure.net",

    [string]$outDir = "./generated",

    [string]$TokenFile,

    [ValidateSet("rest", "cli")]
    [string]$ApiMode = "rest"
)

# Configure Private CleanRoom cloud and verify local user auth
. "$PSScriptRoot/common/setup-local-auth.ps1"

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Load common frontend helpers (supports REST and CLI modes)
. "$PSScriptRoot/common/frontend-helpers.ps1"
$feCtx = New-FrontendContext -frontendEndpoint $frontendEndpoint -ApiMode $ApiMode

# Resolve outDir to absolute path
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)

# Python helper scripts directory
$pythonScriptsDir = "$PSScriptRoot/common"

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

# Runs an az command, returning $null instead of throwing if it fails.
function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# -- Load generated resource names ----------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

# -- Load metadata from previous steps -----------------------------------------
$datastoreMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found. Run 05-prepare-data-cpk.ps1 first." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json

$identityMetadataFile = Join-Path $outDir $resourceGroup "identity-metadata.json"
if (-not (Test-Path $identityMetadataFile)) {
    Write-Host "ERROR: '$identityMetadataFile' not found. Run 06-setup-identity.ps1 first." -ForegroundColor Red
    exit 1
}
$identityMeta = Get-Content $identityMetadataFile -Raw | ConvertFrom-Json

# Load OIDC issuer URL from generated metadata
$issuerUrlFile = Join-Path $outDir $resourceGroup "issuer-url.txt"
if (-not (Test-Path $issuerUrlFile)) {
    Write-Host "ERROR: '$issuerUrlFile' not found. Run 06-setup-identity.ps1 first." -ForegroundColor Red
    exit 1
}
$oidcIssuerUrl = (Get-Content $issuerUrlFile -Raw).Trim()
Write-Host "Using OIDC issuer URL: $oidcIssuerUrl" -ForegroundColor Cyan

$inputMeta = $datastoreMeta.input

# -- Build dataset publish body -------------------------------------------------
function New-DatasetPublishBody {
    param(
        [PSCustomObject]$Meta,
        [PSCustomObject]$Identity,
        [string]$AccessMode,
        [string[]]$AllowedFields,
        [string]$EncryptionMode = "CPK"
    )

    $body = [ordered]@{
        name = $Meta.name
        datasetSchema = [ordered]@{
            format = $Meta.schema.format
            fields = @($Meta.schema.fields | ForEach-Object {
                [ordered]@{ fieldName = $_.fieldName; fieldType = $_.fieldType }
            })
        }
        datasetAccessPolicy = [ordered]@{
            accessMode = $AccessMode
        }
        store = [ordered]@{
            storageAccountUrl  = $Meta.storeUrl
            containerName      = $Meta.containerName
            storageAccountType = $Meta.storeType
            encryptionMode     = $EncryptionMode
        }
        identity = [ordered]@{
            name      = $Identity.identityName
            clientId  = $Identity.clientId
            tenantId  = $Identity.tenantId
            issuerUrl = $script:oidcIssuerUrl
        }
        # CPK-specific: encryption key references
        dek = [ordered]@{
            keyVaultUrl = $Meta.encryption.dekStoreUrl
            secretId    = $Meta.encryption.dekSecretName
        }
        kek = [ordered]@{
            keyVaultUrl = $Meta.encryption.kekStoreUrl
            secretId    = $Meta.encryption.kekName
            maaUrl      = $script:maaUrl
        }
    }

    if ($AllowedFields -and $AllowedFields.Count -gt 0) {
        $body.datasetAccessPolicy.allowedFields = @($AllowedFields)
    }

    return $body
}

# ==============================================================================
# PHASE B-1: Publish datasets to the collaboration
# ==============================================================================

Write-Host "`n=== Phase B-1: Publishing datasets ===" -ForegroundColor Cyan

# -- Publish input dataset ------------------------------------------------------
Write-Host "Publishing input dataset '$($inputMeta.name)' (CPK)..." -ForegroundColor Cyan

$existingInput = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -TokenFile $TokenFile
if ($existingInput) {
    Write-Host "Input dataset '$($inputMeta.name)' already published (skipped)." -ForegroundColor Yellow
} else {
    $inputBody = New-DatasetPublishBody `
        -Meta $inputMeta `
        -Identity $identityMeta `
        -AccessMode "read" `
        -AllowedFields @("date", "author", "mentions")

    Publish-FrontendDataset -Context $feCtx `
        -CollaborationId $collaborationId `
        -DocumentId $inputMeta.name `
        -Body $inputBody `
        -TokenFile $TokenFile
    Write-Host "Input dataset '$($inputMeta.name)' published." -ForegroundColor Green
}

# Show the dataset
$datasetInfo = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -TokenFile $TokenFile
if ($datasetInfo) {
    $datasetInfo | ConvertTo-Json -Depth 10
}

# -- Publish output dataset (Woodgrove only) -----------------------------------
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Write-Host "`nPublishing output dataset '$($datastoreMeta.output.name)' (CPK)..." -ForegroundColor Cyan

    $existingOutput = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -TokenFile $TokenFile
    if ($existingOutput) {
        Write-Host "Output dataset '$($datastoreMeta.output.name)' already published (skipped)." -ForegroundColor Yellow
    } else {
        $outputBody = New-DatasetPublishBody `
            -Meta $datastoreMeta.output `
            -Identity $identityMeta `
            -AccessMode "write" `
            -AllowedFields @("author", "Number_Of_Mentions")

        Publish-FrontendDataset -Context $feCtx `
            -CollaborationId $collaborationId `
            -DocumentId $datastoreMeta.output.name `
            -Body $outputBody `
            -TokenFile $TokenFile
        Write-Host "Output dataset '$($datastoreMeta.output.name)' published." -ForegroundColor Green
    }

    $outputInfo = Get-FrontendDataset -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -TokenFile $TokenFile
    if ($outputInfo) {
        $outputInfo | ConvertTo-Json -Depth 10
    }
}

# ==============================================================================
# PHASE B-2: Create per-dataset KEKs and wrap DEKs
# ==============================================================================

Write-Host "`n=== Phase B-2: Creating KEKs and wrapping DEKs ===" -ForegroundColor Cyan

# Helper function: Create KEK and wrap DEK for a single dataset.
function New-DatasetKekAndWrappedDek {
    param(
        [PSCustomObject]$DatasetMeta,
        [string]$CollaborationId,
        [string]$KeyVaultName,
        [string]$MaaUrl,
        [string]$OutputDir
    )

    $datasetName = $DatasetMeta.name
    $kekName = $DatasetMeta.encryption.kekName
    $dekFile = $DatasetMeta._local.dekFile
    $wrappedDekSecretName = $DatasetMeta.encryption.dekSecretName

    Write-Host "`n  --- Dataset: $datasetName ---" -ForegroundColor White
    Write-Host "  KEK name: $kekName" -ForegroundColor Yellow
    Write-Host "  DEK file: $dekFile" -ForegroundColor Yellow

    if (-not (Test-Path $dekFile)) {
        Write-Host "  ERROR: DEK file '$dekFile' not found. Run 05-prepare-data-cpk.ps1 first." -ForegroundColor Red
        exit 1
    }

    # Step 1: Fetch SKR release policy from the just-published dataset.
    Write-Host "  Fetching SKR release policy from dataset '$datasetName'..." -ForegroundColor Yellow
    $PSNativeCommandUseErrorActionPreference = $false
    $skrPolicy = Invoke-AzCli @(
        "managedcleanroom", "frontend", "analytics", "skr-policy",
        "--collaboration-id", $CollaborationId,
        "--dataset-id", $datasetName
    ) -Description "Fetch SKR policy for $datasetName"
    $PSNativeCommandUseErrorActionPreference = $true

    # Override the authority with the maaUrl.
    $skrPolicy.anyOf[0].authority = $MaaUrl
    $skrPolicyJson = $skrPolicy | ConvertTo-Json -Depth 10 -Compress
    Write-Host "  SKR policy fetched. ccePolicyHash: $($skrPolicy.anyOf[0].allOf[0].equals)" -ForegroundColor Green

    # Step 2: Create KEK locally (RSA-2048) and import to Key Vault with SKR policy.
    Write-Host "  Creating KEK '$kekName' (local RSA-2048 + import to KV)..." -ForegroundColor Yellow
    $kekOutputDir = Join-Path $OutputDir "datastores" "keys"

    # Delete existing key if present (KEK must be recreated with correct SKR policy).
    $existingKey = Invoke-AzSafe @("keyvault", "key", "show", "--vault-name", $KeyVaultName, "--name", $kekName)
    if ($existingKey) {
        Write-Host "  Deleting existing key '$kekName' (will recreate with new SKR policy)..." -ForegroundColor Yellow
        Invoke-AzSafe @("keyvault", "key", "delete", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
        Invoke-AzSafe @("keyvault", "key", "purge", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
    }

    $PSNativeCommandUseErrorActionPreference = $false
    & $script:pythonExe "$pythonScriptsDir/create-kek.py" `
        --kek-name $kekName `
        --output-dir $kekOutputDir `
        --skr-policy-json $skrPolicyJson `
        --key-vault-url "https://${KeyVaultName}.vault.azure.net" 2>&1 | ForEach-Object {
        Write-Host "  $_" -ForegroundColor DarkGray
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to create KEK '$kekName'" -ForegroundColor Red
        exit 1
    }
    $PSNativeCommandUseErrorActionPreference = $true
    Write-Host "  KEK '$kekName' imported to Key Vault." -ForegroundColor Green

    # Step 3: Wrap DEK with KEK public key (RSA-OAEP-SHA256, client-side).
    Write-Host "  Wrapping DEK with KEK..." -ForegroundColor Yellow
    $kekPemFile = Join-Path $kekOutputDir "$kekName.pem"

    $PSNativeCommandUseErrorActionPreference = $false
    $wrappedDekBase64 = & $script:pythonExe "$pythonScriptsDir/generate-wrapped-dek.py" `
        --dek-file $dekFile `
        --kek-public-key-file $kekPemFile
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Failed to wrap DEK" -ForegroundColor Red
        exit 1
    }
    $PSNativeCommandUseErrorActionPreference = $true
    Write-Host "  DEK wrapped ($(($wrappedDekBase64).Length) chars base64)." -ForegroundColor Green

    # Step 4: Store wrapped DEK as a Key Vault secret.
    Write-Host "  Storing wrapped DEK as secret '$wrappedDekSecretName'..." -ForegroundColor Yellow
    az keyvault secret set `
        --vault-name $KeyVaultName `
        --name $wrappedDekSecretName `
        --value $wrappedDekBase64 `
        --output none
    Write-Host "  Wrapped DEK stored as secret '$wrappedDekSecretName'." -ForegroundColor Green
}

# Create KEK + wrap DEK for input dataset.
New-DatasetKekAndWrappedDek `
    -DatasetMeta $inputMeta `
    -CollaborationId $collaborationId `
    -KeyVaultName $KEYVAULT_NAME `
    -MaaUrl $maaUrl `
    -OutputDir $outDir

# Create KEK + wrap DEK for output dataset (Woodgrove only).
if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    New-DatasetKekAndWrappedDek `
        -DatasetMeta $datastoreMeta.output `
        -CollaborationId $collaborationId `
        -KeyVaultName $KEYVAULT_NAME `
        -MaaUrl $maaUrl `
        -OutputDir $outDir
}

# ==============================================================================
# PHASE B-3: Enable execution consent on published datasets
# ==============================================================================

Write-Host "`n=== Phase B-3: Enabling execution consent ===" -ForegroundColor Cyan
Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $inputMeta.name -Action "enable" -TokenFile $TokenFile
Write-Host "Execution consent enabled for input dataset '$($inputMeta.name)'." -ForegroundColor Green

if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    Set-FrontendConsent -Context $feCtx -CollaborationId $collaborationId -DocumentId $datastoreMeta.output.name -Action "enable" -TokenFile $TokenFile
    Write-Host "Execution consent enabled for output dataset '$($datastoreMeta.output.name)'." -ForegroundColor Green
}

Write-Host "`n=== CPK dataset publishing complete for '$persona' ===" -ForegroundColor Green
Write-Host "  Datasets published with per-dataset KEKs and wrapped DEKs." -ForegroundColor Green
Write-Host "  Next: Run 09-publish-query.ps1 to publish the analytics query." -ForegroundColor Green
