<#
.SYNOPSIS
    Prepares CPK encryption keys for datasets (KEK creation, DEK wrapping).

.DESCRIPTION
    CPK-only utility script. For each dataset in the datastore metadata:
    1. Fetches the SKR release policy from the frontend (for the attestation hash).
    2. Generates a local RSA-2048 KEK.
    3. Imports the KEK to Key Vault with the SKR release policy.
    4. Wraps the DEK with the KEK (RSA-OAEP-SHA256, client-side).
    5. Stores the wrapped DEK as a Key Vault secret.

    This script makes frontend calls internally to fetch SKR policies.
    The same script works regardless of whether the README uses AZ CLI or REST API.

    Prerequisites:
    - 04-prepare-resources.ps1 must have been run.
    - 05-prepare-data.ps1 must have been run with -variant cpk (generates DEK files).

.PARAMETER collaborationId
    The collaboration frontend UUID.

.PARAMETER resourceGroup
    Azure resource group.

.PARAMETER persona
    The collaborator persona (northwind or woodgrove).

.PARAMETER frontendEndpoint
    Frontend service URL (needed for SKR policy fetch).

.PARAMETER maaUrl
    MAA URL for the SKR release policy authority.

.PARAMETER outDir
    Output directory.

.PARAMETER TokenFile
    Token file for frontend authentication.
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

    [string]$TokenFile
)

$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

# Resolve outDir to absolute path
$outDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($outDir)

function Invoke-AzSafe {
    param([string[]]$Arguments)
    $PSNativeCommandUseErrorActionPreference = $false
    $result = & az @Arguments 2>$null
    if ($LASTEXITCODE -eq 0) { return $result }
    return $null
}

# -- Load prerequisites -----------------------------------------------------------
$namesFile = Join-Path $outDir $resourceGroup "names.generated.ps1"
if (-not (Test-Path $namesFile)) {
    Write-Host "ERROR: '$namesFile' not found. Run 04-prepare-resources.ps1 first." -ForegroundColor Red
    exit 1
}
. $namesFile

$datastoreMetadataFile = Join-Path $outDir "datastores" "$persona-datastore-metadata.json"
if (-not (Test-Path $datastoreMetadataFile)) {
    Write-Host "ERROR: '$datastoreMetadataFile' not found. Run 05-prepare-data.ps1 with -variant cpk first." -ForegroundColor Red
    exit 1
}
$datastoreMeta = Get-Content $datastoreMetadataFile -Raw | ConvertFrom-Json

if ($datastoreMeta.input.encryptionMode -ne "CPK") {
    Write-Host "Encryption mode is not CPK — nothing to do." -ForegroundColor Yellow
    exit 0
}

# -- Set up frontend auth (used internally for SKR policy fetch) -------------------
$feBase = $frontendEndpoint.TrimEnd('/')
if ($feBase.EndsWith('/collaborations')) {
    $feBase = $feBase.Substring(0, $feBase.Length - '/collaborations'.Length)
}

# Try CLI first, fallback to REST
function Get-SkrPolicy {
    param([string]$DatasetName)

    # Try az CLI
    $env:AZURE_CLI_DISABLE_CONNECTION_VERIFICATION = "1"
    if ($TokenFile) {
        $env:MANAGEDCLEANROOM_ACCESS_TOKEN = (Get-Content $TokenFile -Raw).Trim()
    }
    az managedcleanroom frontend configure --endpoint $feBase 2>&1 | Out-Null

    $PSNativeCommandUseErrorActionPreference = $false
    $raw = az managedcleanroom frontend analytics skr-policy `
        --collaboration-id $collaborationId `
        --dataset-id $DatasetName 2>&1
    $PSNativeCommandUseErrorActionPreference = $true

    if ($LASTEXITCODE -eq 0) {
        $jsonLines = $raw | Where-Object { $_ -is [string] }
        return $jsonLines | ConvertFrom-Json
    }

    # Fallback to REST
    $token = if ($TokenFile) { (Get-Content $TokenFile -Raw).Trim() } else { az account get-access-token --query accessToken -o tsv }
    $headers = @{ Authorization = "Bearer $token"; "Content-Type" = "application/json" }
    $url = "$feBase/collaborations/$collaborationId/analytics/datasets/$DatasetName/skr-policy?api-version=2026-03-01-preview"
    return Invoke-RestMethod -Uri $url -Headers $headers -Method Get -SkipCertificateCheck
}

# -- Process each dataset ----------------------------------------------------------
function New-DatasetKekAndWrappedDek {
    param(
        [PSCustomObject]$DatasetMeta,
        [string]$KeyVaultName,
        [string]$OutputDir
    )

    $datasetName = $DatasetMeta.name
    $kekName = $DatasetMeta.encryption.kekName
    $dekFile = $DatasetMeta._local.dekFile
    $wrappedDekSecretName = $DatasetMeta.encryption.dekSecretName

    Write-Host "`n  --- Dataset: $datasetName ---" -ForegroundColor White
    Write-Host "  KEK name: $kekName" -ForegroundColor Yellow

    if (-not (Test-Path $dekFile)) {
        Write-Host "  ERROR: DEK file '$dekFile' not found." -ForegroundColor Red
        exit 1
    }

    $dekBytes = [System.IO.File]::ReadAllBytes($dekFile)

    # Step 1: Fetch SKR release policy
    Write-Host "  Fetching SKR release policy..." -ForegroundColor Yellow
    $skrPolicy = Get-SkrPolicy -DatasetName $datasetName
    $skrPolicy.anyOf[0].authority = $maaUrl
    $skrPolicyJson = $skrPolicy | ConvertTo-Json -Depth 10 -Compress
    Write-Host "  SKR policy fetched. ccePolicyHash: $($skrPolicy.anyOf[0].allOf[0].equals)" -ForegroundColor Green

    # Step 2: Generate RSA-2048 KEK locally
    Write-Host "  Generating KEK '$kekName' (RSA-2048)..." -ForegroundColor Yellow
    $kekOutputDir = Join-Path $OutputDir "datastores" "keys"
    New-Item -ItemType Directory -Path $kekOutputDir -Force | Out-Null

    $rsa = [System.Security.Cryptography.RSA]::Create(2048)
    $kekPemFile = Join-Path $kekOutputDir "$kekName.pem"
    $rsa.ExportPkcs8PrivateKeyPem() | Set-Content -Path $kekPemFile -NoNewline -Encoding utf8

    $skrPolicyFile = Join-Path $kekOutputDir "$kekName-skr-policy.json"
    [System.IO.File]::WriteAllText($skrPolicyFile, $skrPolicyJson)

    # Step 3: Import KEK to Key Vault with SKR policy
    Write-Host "  Importing KEK to Key Vault..." -ForegroundColor Yellow
    $existingKey = Invoke-AzSafe @("keyvault", "key", "show", "--vault-name", $KeyVaultName, "--name", $kekName)
    if ($existingKey) {
        Invoke-AzSafe @("keyvault", "key", "delete", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
        Invoke-AzSafe @("keyvault", "key", "purge", "--vault-name", $KeyVaultName, "--name", $kekName)
        Start-Sleep -Seconds 5
    }

    az keyvault key import --vault-name $KeyVaultName --name $kekName `
        --pem-file $kekPemFile --protection hsm --exportable true `
        --ops wrapKey unwrapKey --policy "@$skrPolicyFile" --output none
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to import KEK '$kekName'."
    }
    Write-Host "  KEK imported." -ForegroundColor Green

    # Step 4: Wrap DEK with KEK (RSA-OAEP-SHA256, client-side)
    $wrappedDekBytes = $rsa.Encrypt($dekBytes, [System.Security.Cryptography.RSAEncryptionPadding]::OaepSHA256)
    $wrappedDekBase64 = [Convert]::ToBase64String($wrappedDekBytes)
    $rsa.Dispose()

    # Step 5: Store wrapped DEK as KV secret
    Write-Host "  Storing wrapped DEK as secret '$wrappedDekSecretName'..." -ForegroundColor Yellow
    az keyvault secret set --vault-name $KeyVaultName --name $wrappedDekSecretName `
        --value $wrappedDekBase64 --output none
    Write-Host "  Done." -ForegroundColor Green
}

Write-Host "=== Preparing CPK keys ===" -ForegroundColor Cyan

New-DatasetKekAndWrappedDek -DatasetMeta $datastoreMeta.input `
    -KeyVaultName $KEYVAULT_NAME -OutputDir $outDir

if ($persona -eq "woodgrove" -and $datastoreMeta.output) {
    New-DatasetKekAndWrappedDek -DatasetMeta $datastoreMeta.output `
        -KeyVaultName $KEYVAULT_NAME -OutputDir $outDir
}

Write-Host "`nCPK key preparation complete for '$persona'." -ForegroundColor Green
